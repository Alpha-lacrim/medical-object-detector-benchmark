from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from meddet_benchmark.corruptions import (
    CorruptionCondition,
    CorruptionDefinition,
    SeverityLevel,
    apply_corruption,
    corruption_fingerprint,
    expand_conditions,
    load_corruptions,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "corruptions.yaml"


def patterned_image() -> Image.Image:
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(checker[:, :, None], 3, axis=2), mode="RGB")


def condition(name: str, kind: str, value: float) -> CorruptionCondition:
    families = {
        "brightness": "lighting",
        "gaussian_noise": "noise",
        "salt_pepper": "noise",
        "gaussian_blur": "blur",
        "motion_blur": "blur",
        "jpeg": "compression",
    }
    return CorruptionCondition(
        name=name,
        family=families[kind],
        kind=kind,
        unit="test",
        severity=1,
        value=value,
    )


def high_frequency_energy(image: Image.Image) -> float:
    values = np.asarray(image, dtype=np.float32)
    horizontal = np.diff(values, axis=1)
    vertical = np.diff(values, axis=0)
    return float(np.mean(np.abs(horizontal)) + np.mean(np.abs(vertical)))


def test_phase6_matrix_has_all_required_types_and_five_levels() -> None:
    config = load_corruptions(CONFIG_PATH)
    by_name = {item.name: item for item in config.corruptions}

    assert set(by_name) == {
        "darker",
        "brighter",
        "gaussian_noise",
        "salt_and_pepper",
        "gaussian_blur",
        "motion_blur",
        "jpeg",
    }
    assert all(len(item.levels) == 5 for item in by_name.values())
    assert [level.value for level in by_name["jpeg"].levels] == [90, 70, 50, 35, 20]
    assert len(expand_conditions(config)) == 35
    assert len(corruption_fingerprint(config)) == 64


def test_every_condition_is_deterministic_and_preserves_geometry() -> None:
    config = load_corruptions(CONFIG_PATH)
    clean = patterned_image()
    clean_before = np.asarray(clean).copy()

    for item in expand_conditions(config):
        first = apply_corruption(clean, item, base_seed=config.seed, image_id="case-1")
        second = apply_corruption(clean, item, base_seed=config.seed, image_id="case-1")
        assert first.mode == "RGB"
        assert first.size == clean.size
        np.testing.assert_array_equal(first, second, err_msg=item.condition_id)

    np.testing.assert_array_equal(clean, clean_before)


@pytest.mark.parametrize("kind,value", [("gaussian_noise", 0.05), ("motion_blur", 9)])
def test_stochastic_conditions_are_order_independent_per_image(kind: str, value: float) -> None:
    clean = patterned_image()
    item = condition(kind, kind, value)

    first = apply_corruption(clean, item, base_seed=7, image_id="first")
    apply_corruption(clean, item, base_seed=7, image_id="unrelated")
    repeated = apply_corruption(clean, item, base_seed=7, image_id="first")
    other = apply_corruption(clean, item, base_seed=7, image_id="other")

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)


def test_brightness_direction_is_correct() -> None:
    values = np.full((32, 32, 3), 100, dtype=np.uint8)
    clean = Image.fromarray(values, mode="RGB")
    dark = apply_corruption(
        clean,
        condition("dark", "brightness", 0.6),
        base_seed=1,
        image_id="case",
    )
    bright = apply_corruption(
        clean,
        condition("bright", "brightness", 1.4),
        base_seed=1,
        image_id="case",
    )

    assert np.asarray(dark).mean() < np.asarray(clean).mean()
    assert np.asarray(bright).mean() > np.asarray(clean).mean()


@pytest.mark.parametrize("kind,value", [("gaussian_blur", 2), ("motion_blur", 9)])
def test_blur_reduces_high_frequency_content(kind: str, value: float) -> None:
    clean = patterned_image()
    blurred = apply_corruption(
        clean,
        condition(kind, kind, value),
        base_seed=3,
        image_id="case",
    )

    assert high_frequency_energy(blurred) < high_frequency_energy(clean)


@pytest.mark.parametrize(
    "definition",
    [
        {
            "name": "bad_motion",
            "family": "blur",
            "kind": "motion_blur",
            "unit": "pixels",
            "levels": [
                {"severity": 1, "value": 4},
                {"severity": 2, "value": 5},
                {"severity": 3, "value": 7},
            ],
        },
        {
            "name": "bad_jpeg",
            "family": "compression",
            "kind": "jpeg",
            "unit": "quality",
            "levels": [
                {"severity": 1, "value": 80},
                {"severity": 2, "value": 90},
                {"severity": 3, "value": 20},
            ],
        },
    ],
)
def test_invalid_severity_curves_are_rejected(definition: dict) -> None:
    with pytest.raises(ValidationError):
        CorruptionDefinition.model_validate(definition)


def test_severity_must_be_contiguous() -> None:
    with pytest.raises(ValidationError):
        CorruptionDefinition(
            name="noise",
            family="noise",
            kind="gaussian_noise",
            unit="std",
            levels=(
                SeverityLevel(severity=1, value=0.01),
                SeverityLevel(severity=3, value=0.02),
                SeverityLevel(severity=4, value=0.03),
            ),
        )
