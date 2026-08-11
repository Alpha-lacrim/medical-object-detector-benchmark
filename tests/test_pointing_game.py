import numpy as np
import pytest

from src.explainability.pointing_game import (
    evaluate_box_attention,
    pixel_center_box_mask,
)


def test_energy_and_pointing_use_pixel_center_box_mask() -> None:
    heatmap = np.zeros((4, 4), dtype=np.float64)
    heatmap[1, 1] = 3
    heatmap[3, 3] = 1

    result = evaluate_box_attention(
        heatmap,
        [1, 1, 3, 3],
        zero_energy_epsilon=1e-12,
    )

    assert result.valid
    assert result.energy_in_box == pytest.approx(0.75)
    assert result.pointing_hit is True
    assert result.box_pixel_fraction == pytest.approx(0.25)
    assert result.energy_lift_over_area == pytest.approx(0.5)
    assert (result.peak_x, result.peak_y) == (1, 1)


def test_zero_energy_is_explicitly_invalid() -> None:
    result = evaluate_box_attention(
        np.zeros((3, 3)),
        [0, 0, 1, 1],
        zero_energy_epsilon=1e-12,
    )

    assert not result.valid
    assert result.energy_in_box is None
    assert result.pointing_hit is None


def test_box_mask_rejects_out_of_bounds_boxes() -> None:
    with pytest.raises(ValueError, match="inside"):
        pixel_center_box_mask((4, 4), [0, 0, 5, 1])
