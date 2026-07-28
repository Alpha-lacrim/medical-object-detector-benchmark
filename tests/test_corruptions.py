from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from meddet_benchmark.corruptions import (
    CorruptionConfig,
    CorruptionLevel,
    apply_corruption,
    corruption_fingerprint,
    load_corruptions,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "corruptions.yaml"


def patterned_image() -> Image.Image:
    values = np.arange(32 * 24 * 3, dtype=np.uint16).reshape(24, 32, 3) % 256
    return Image.fromarray(values.astype(np.uint8), mode="RGB")


def test_required_corruption_matrix_is_frozen() -> None:
    config = load_corruptions(CONFIG_PATH)
    kinds = {level.kind for level in config.levels}
    jpeg_qualities = {level.value for level in config.levels if level.kind == "jpeg"}

    assert kinds == {
        "brightness",
        "gaussian_noise",
        "salt_pepper",
        "gaussian_blur",
        "motion_blur",
        "jpeg",
    }
    assert jpeg_qualities == {20, 50}
    assert len(corruption_fingerprint(config)) == 64


def test_every_corruption_is_deterministic_and_preserves_geometry() -> None:
    config = load_corruptions(CONFIG_PATH)
    clean = patterned_image()
    clean_before = np.asarray(clean).copy()

    for level in config.levels:
        first = apply_corruption(clean, level, base_seed=config.seed, image_id="case-1")
        second = apply_corruption(clean, level, base_seed=config.seed, image_id="case-1")
        assert first.mode == "RGB"
        assert first.size == clean.size
        np.testing.assert_array_equal(first, second, err_msg=level.name)

    np.testing.assert_array_equal(clean, clean_before)


def test_stochastic_corruption_is_order_independent_per_image() -> None:
    clean = patterned_image()
    level = CorruptionLevel(name="noise", kind="gaussian_noise", value=0.05)

    first = apply_corruption(clean, level, base_seed=7, image_id="first")
    apply_corruption(clean, level, base_seed=7, image_id="unrelated")
    repeated = apply_corruption(clean, level, base_seed=7, image_id="first")
    other = apply_corruption(clean, level, base_seed=7, image_id="other")

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)


def test_brightness_direction_is_correct() -> None:
    clean = patterned_image()
    dark = apply_corruption(
        clean,
        CorruptionLevel(name="dark", kind="brightness", value=0.6),
        base_seed=1,
        image_id="case",
    )
    bright = apply_corruption(
        clean,
        CorruptionLevel(name="bright", kind="brightness", value=1.4),
        base_seed=1,
        image_id="case",
    )

    assert np.asarray(dark).mean() < np.asarray(clean).mean()
    assert np.asarray(bright).mean() > np.asarray(clean).mean()


@pytest.mark.parametrize(
    "level",
    [
        {"name": "bad", "kind": "motion_blur", "value": 4},
        {"name": "bad", "kind": "jpeg", "value": 101},
        {"name": "bad", "kind": "salt_pepper", "value": 1.1},
    ],
)
def test_invalid_severity_is_rejected(level: dict) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["levels"] = [level]

    with pytest.raises(ValidationError):
        CorruptionConfig.model_validate(payload)
