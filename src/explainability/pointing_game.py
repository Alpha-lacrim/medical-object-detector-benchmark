"""Energy-in-box and pointing-game metrics for nonnegative saliency maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class BoxAttentionResult:
    """Localization measurements for one heatmap and one ground-truth box."""

    valid: bool
    total_energy: float
    energy_in_box: float | None
    pointing_hit: bool | None
    peak_x: int | None
    peak_y: int | None
    box_pixel_fraction: float
    energy_lift_over_area: float | None

    def to_dict(self) -> dict[str, float | bool | int | None]:
        """Return a JSON-safe representation."""

        return asdict(self)


def _validated_heatmap(value: ArrayLike) -> NDArray[np.float64]:
    heatmap = np.asarray(value, dtype=np.float64)
    if heatmap.ndim != 2 or 0 in heatmap.shape:
        raise ValueError("heatmap must be a non-empty two-dimensional array")
    if not np.isfinite(heatmap).all() or np.any(heatmap < 0):
        raise ValueError("heatmap must contain finite nonnegative values")
    return heatmap


def _validated_box(value: ArrayLike, *, width: int, height: int) -> NDArray[np.float64]:
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xyxy must contain four finite values")
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError("box_xyxy must have positive area inside the heatmap")
    return box


def pixel_center_box_mask(shape: tuple[int, int], box_xyxy: ArrayLike) -> NDArray[np.bool_]:
    """Rasterize a continuous box by testing each heatmap pixel center."""

    if len(shape) != 2 or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
    ):
        raise ValueError("shape must contain positive integer height and width")
    height, width = shape
    x1, y1, x2, y2 = _validated_box(box_xyxy, width=width, height=height)
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    return ((ys[:, None] >= y1) & (ys[:, None] < y2)) & ((xs[None, :] >= x1) & (xs[None, :] < x2))


def evaluate_box_attention(
    heatmap: ArrayLike,
    box_xyxy: ArrayLike,
    *,
    zero_energy_epsilon: float,
) -> BoxAttentionResult:
    """Measure Grad-CAM energy and peak localization inside one box.

    Zero-energy maps are explicitly invalid. They remain in the per-target table
    but are excluded from aggregate energy and pointing estimates.
    """

    if zero_energy_epsilon <= 0:
        raise ValueError("zero_energy_epsilon must be positive")
    saliency = _validated_heatmap(heatmap)
    mask = pixel_center_box_mask(saliency.shape, box_xyxy)
    area_fraction = float(mask.mean())
    total = float(saliency.sum(dtype=np.float64))
    if total <= zero_energy_epsilon:
        return BoxAttentionResult(
            valid=False,
            total_energy=total,
            energy_in_box=None,
            pointing_hit=None,
            peak_x=None,
            peak_y=None,
            box_pixel_fraction=area_fraction,
            energy_lift_over_area=None,
        )
    flat_peak = int(np.argmax(saliency))
    peak_y, peak_x = np.unravel_index(flat_peak, saliency.shape)
    energy = float(saliency[mask].sum(dtype=np.float64) / total)
    return BoxAttentionResult(
        valid=True,
        total_energy=total,
        energy_in_box=energy,
        pointing_hit=bool(mask[peak_y, peak_x]),
        peak_x=int(peak_x),
        peak_y=int(peak_y),
        box_pixel_fraction=area_fraction,
        energy_lift_over_area=energy - area_fraction,
    )
