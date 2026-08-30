from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset
from pydicom.pixels import apply_windowing

from src.data.prepare import scale_radiograph_to_uint8
from src.robustness.radiography_shifts import (
    RadiographyShift,
    _metadata_audit_row,
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


def _shift_metadata(kind: str) -> dict[str, str]:
    common = {
        "display_name": "test shift",
        "interpretation": "test-only interpretation",
    }
    if kind == "dicom_linear_window":
        return {
            **common,
            "family": "display_transform",
            "scientific_class": "A",
            "retained_use_class": "A",
        }
    if kind == "poisson_like_noise":
        return {
            **common,
            "family": "poisson_like_intensity_noise",
            "scientific_class": "D",
            "retained_use_class": "B",
        }
    return {
        **common,
        "family": "spatial_resolution_blur",
        "scientific_class": "C",
        "retained_use_class": "C",
    }


def test_acquisition_config_uses_frozen_seed_sample_and_all_shift_families() -> None:
    config = load_acquisition_shift_config(CONFIG_PATH)

    assert config.seed == 17
    assert config.inputs.subset_manifest.name == "test_robustness_seed17_n300.csv"
    assert {shift.family for shift in config.shifts} == {
        "display_transform",
        "poisson_like_intensity_noise",
        "spatial_resolution_blur",
    }
    assert {shift.scientific_class for shift in config.shifts} == {"A", "C", "D"}
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


def test_dicom_linear_window_matches_standard_4096_level_numeric_example() -> None:
    values = np.asarray([0.0, 1.0, 2048.0, 4095.0, 4096.0])
    output = dicom_linear_window(
        values,
        center=2048.0,
        width=4096.0,
        output_min=0.0,
        output_max=255.0,
    )
    expected = np.asarray(
        [
            0.0,
            255.0 / 4095.0,
            ((2048.0 - 2047.5) / 4095.0 + 0.5) * 255.0,
            255.0,
            255.0,
        ]
    )
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-12)


def test_dicom_width_one_is_the_standard_threshold_case() -> None:
    values = np.asarray([2047.49, 2047.5, 2047.51, 2048.0])
    output = dicom_linear_window(
        values,
        center=2048.0,
        width=1.0,
        output_min=0.0,
        output_max=255.0,
    )

    assert output.tolist() == [0.0, 0.0, 255.0, 255.0]


def test_dicom_linear_window_matches_standard_zero_center_example() -> None:
    values = np.asarray([-50.0, -49.0, 0.0, 49.0, 50.0])
    output = dicom_linear_window(
        values,
        center=0.0,
        width=100.0,
        output_min=0.0,
        output_max=255.0,
    )
    expected = np.asarray(
        [
            0.0,
            ((-49.0 + 0.5) / 99.0 + 0.5) * 255.0,
            ((0.0 + 0.5) / 99.0 + 0.5) * 255.0,
            255.0,
            255.0,
        ]
    )
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-12)


def test_monochrome1_is_the_display_polarity_inverse_of_monochrome2() -> None:
    pixels = np.asarray([[0.0, 128.0, 255.0], [0.0, 128.0, 255.0]])
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

    np.testing.assert_array_equal(monochrome1, 255 - monochrome2)


def test_per_image_minmax_cancels_an_unclipped_linear_window_rescaling() -> None:
    dataset = _dicom_metadata()
    pixels = np.arange(256, dtype=np.float64).reshape(16, 16)
    shift = RadiographyShift(
        id="wide_window_test",
        kind="dicom_linear_window",
        **_shift_metadata("dicom_linear_window"),
        center_offset_fraction=0.0,
        width_multiplier=1.25,
    )

    shifted = apply_radiography_shift(
        pixels,
        dataset,
        shift,
        seed=17,
        image_id="image.png",
    )
    assert not np.array_equal(shifted, pixels)

    clean_scaled = scale_radiograph_to_uint8(
        pixels,
        photometric_interpretation="MONOCHROME2",
        invert_monochrome1=True,
    )
    shifted_scaled = scale_radiograph_to_uint8(
        shifted,
        photometric_interpretation="MONOCHROME2",
        invert_monochrome1=True,
    )
    np.testing.assert_array_equal(shifted_scaled, clean_scaled)


def test_secondary_capture_missing_signal_relationship_is_not_poisson_proxy_eligible() -> None:
    dataset = _dicom_metadata()
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    dataset.Modality = "CR"
    dataset.ConversionType = "WSD"
    dataset.LossyImageCompression = "01"
    dataset.LossyImageCompressionMethod = "ISO_10918_1"

    row = _metadata_audit_row(
        dataset,
        processed_name="image.png",
        raw_name="image.dcm",
    )

    assert row["sop_class_name"] == "Secondary Capture Image Storage"
    assert row["pixel_intensity_relationship_missing"] is True
    assert row["pixel_intensity_relationship_sign_missing"] is True
    assert row["xray_signal_poisson_proxy_eligible"] is False
    assert row["stored_pixel_relationship_class"] == "unknown_processed_stored_values"


def test_poisson_like_shift_is_deterministic_and_signal_dependent() -> None:
    dataset = _dicom_metadata()
    pixels = np.empty((256, 256), dtype=np.float64)
    pixels[:, :128] = 64.0
    pixels[:, 128:] = 192.0
    shift = RadiographyShift(
        id="poisson_test",
        kind="poisson_like_noise",
        **_shift_metadata("poisson_like_noise"),
        reference_full_scale_counts=4096.0,
        relative_count_fraction=0.25,
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
        kind="gaussian_blur",
        **_shift_metadata("gaussian_blur"),
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
