from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset
from pydicom.pixels import apply_windowing

from src.data.prepare import scale_radiograph_to_uint8
from src.robustness.radiography_shifts import (
    RadiographyShift,
    apply_radiography_shift,
    dicom_linear_window,
    load_acquisition_shift_config,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "acquisition_shifts.yaml"


def _dicom_metadata() -> Dataset:
    dataset = Dataset()
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PhotometricInterpretation = "MONOCHROME2"
    return dataset


def test_acquisition_config_uses_frozen_seed_sample_and_all_shift_families() -> None:
    config = load_acquisition_shift_config(CONFIG_PATH)

    assert config.seed == 17
    assert config.inputs.subset_manifest.name == "test_robustness_seed17_n300.csv"
    assert {shift.family for shift in config.shifts} == {
        "voi_window",
        "dose_noise",
        "detector_blur",
    }
    assert len(config.shifts) == 10


def test_shared_minmax_scaler_preserves_original_conversion_semantics() -> None:
    pixels = np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)

    monochrome2 = scale_radiograph_to_uint8(
        pixels,
        photometric_interpretation="MONOCHROME2",
        invert_monochrome1=True,
    )
    monochrome1 = scale_radiograph_to_uint8(
        pixels,
        photometric_interpretation="MONOCHROME1",
        invert_monochrome1=True,
    )

    assert monochrome2.tolist() == [[0, 85], [170, 255]]
    assert monochrome1.tolist() == [[255, 170], [85, 0]]


def test_dicom_linear_window_matches_identity_and_pydicom_reference() -> None:
    values = np.arange(256, dtype=np.float64)
    identity = dicom_linear_window(
        values,
        center=128.0,
        width=256.0,
        output_min=0.0,
        output_max=255.0,
    )
    np.testing.assert_allclose(identity, values, rtol=0, atol=1e-12)

    dataset = _dicom_metadata()
    dataset.WindowCenter = 96.5
    dataset.WindowWidth = 121.0
    dataset.VOILUTFunction = "LINEAR"
    expected = apply_windowing(values, dataset)
    actual = dicom_linear_window(
        values,
        center=96.5,
        width=121.0,
        output_min=0.0,
        output_max=255.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_dicom_width_one_is_the_standard_threshold_case() -> None:
    values = np.asarray([4.49, 4.5, 4.51, 5.0])
    output = dicom_linear_window(
        values,
        center=5.0,
        width=1.0,
        output_min=0.0,
        output_max=255.0,
    )

    assert output.tolist() == [0.0, 0.0, 255.0, 255.0]


def test_poisson_shift_is_deterministic_and_signal_dependent() -> None:
    dataset = _dicom_metadata()
    pixels = np.empty((256, 256), dtype=np.float64)
    pixels[:, :128] = 64.0
    pixels[:, 128:] = 192.0
    shift = RadiographyShift(
        id="poisson_test",
        family="dose_noise",
        kind="poisson_dose",
        reference_full_scale_counts=4096.0,
        dose_fraction=0.25,
    )

    first = apply_radiography_shift(pixels, dataset, shift, seed=17, image_id="image.png")
    second = apply_radiography_shift(pixels, dataset, shift, seed=17, image_id="image.png")
    different = apply_radiography_shift(pixels, dataset, shift, seed=17, image_id="other.png")

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert float(np.var(first[:, 128:])) > float(np.var(first[:, :128]))


def test_declared_gaussian_kernel_reduces_high_frequency_energy() -> None:
    dataset = _dicom_metadata()
    checkerboard = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.float64)
    shift = RadiographyShift(
        id="blur_test",
        family="detector_blur",
        kind="gaussian_blur",
        gaussian_kernel_size=5,
        gaussian_sigma_pixels=1.0,
    )

    blurred = apply_radiography_shift(
        checkerboard,
        dataset,
        shift,
        seed=17,
        image_id="image.png",
    )

    original_energy = np.abs(np.diff(checkerboard, axis=1)).mean()
    shifted_energy = np.abs(np.diff(blurred, axis=1)).mean()
    assert shifted_energy < original_energy
