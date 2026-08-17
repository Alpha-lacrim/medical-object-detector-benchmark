"""Framework-neutral Grad-CAM primitives and module resolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def resolve_module(root: Any, path: str) -> Any:
    """Resolve a dotted attribute/index path below ``root``.

    Numeric components index containers such as ``ModuleList``. Named
    components use attributes, which keeps model-specific layer choices in the
    YAML experiment contract instead of Python code.
    """

    if not path or any(not component for component in path.split(".")):
        raise ValueError("module path must contain non-empty dotted components")
    current = root
    for component in path.split("."):
        if component.isdecimal():
            try:
                current = current[int(component)]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"cannot index component {component!r} in module path {path!r}"
                ) from error
        else:
            try:
                current = getattr(current, component)
            except AttributeError as error:
                raise ValueError(
                    f"cannot resolve component {component!r} in module path {path!r}"
                ) from error
    if not callable(getattr(current, "register_forward_hook", None)):
        raise ValueError(f"module path does not resolve to a hookable module: {path!r}")
    return current


class ActivationCapture:
    """Capture the tensor output of one layer for a differentiable forward pass."""

    def __init__(self, module: Any) -> None:
        self.activation: Any | None = None
        self._handle = module.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Sequence[Any], output: Any) -> None:
        if not hasattr(output, "ndim") or output.ndim != 4:
            raise ValueError("Grad-CAM target layer must emit one BCHW tensor")
        self.activation = output

    def clear(self) -> None:
        """Forget the previous graph before the next forward pass."""

        self.activation = None

    def remove(self) -> None:
        """Remove the forward hook."""

        self._handle.remove()

    def __enter__(self) -> ActivationCapture:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.remove()


def gradcam_from_tensors(
    activation: Any,
    gradient: Any,
    *,
    output_size: tuple[int, int],
    interpolation_mode: str,
    align_corners: bool,
    epsilon: float,
) -> Any:
    """Return one nonnegative, max-normalized Grad-CAM tensor in image space."""

    import torch
    from torch.nn import functional as functional

    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise ValueError("activation and gradient must have the same BCHW shape")
    if activation.shape[0] != 1:
        raise ValueError("Grad-CAM explanation requires batch size one")
    if len(output_size) != 2 or any(size <= 0 for size in output_size):
        raise ValueError("output_size must contain positive height and width")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    activation_float = activation.float()
    gradient_float = gradient.float()
    weights = gradient_float.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation_float).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam,
        size=output_size,
        mode=interpolation_mode,
        align_corners=align_corners,
    )[0, 0]
    maximum = cam.max()
    cam = cam / maximum if torch.isfinite(maximum) and maximum > epsilon else torch.zeros_like(cam)
    if not torch.isfinite(cam).all():
        raise ValueError("Grad-CAM produced non-finite values")
    return cam


def gradcam_for_score(
    score: Any,
    activation: Any,
    *,
    output_size: tuple[int, int],
    interpolation_mode: str,
    align_corners: bool,
    epsilon: float,
    retain_graph: bool,
) -> Any:
    """Differentiate one scalar detector score and return its Grad-CAM map."""

    import torch

    if score.numel() != 1 or not score.requires_grad:
        raise ValueError("Grad-CAM target score must be one differentiable scalar")
    gradient = torch.autograd.grad(
        score,
        activation,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradcam_from_tensors(
        activation,
        gradient,
        output_size=output_size,
        interpolation_mode=interpolation_mode,
        align_corners=align_corners,
        epsilon=epsilon,
    )


def restore_letterboxed_heatmap(
    heatmap: Any,
    *,
    original_size: tuple[int, int],
    interpolation_mode: str,
    align_corners: bool,
    epsilon: float,
) -> Any:
    """Remove centered YOLO letterbox padding and restore original geometry."""

    import torch
    from torch.nn import functional as functional

    if heatmap.ndim != 2:
        raise ValueError("letterboxed heatmap must be two-dimensional")
    input_height, input_width = (int(size) for size in heatmap.shape)
    original_height, original_width = original_size
    if original_height <= 0 or original_width <= 0:
        raise ValueError("original_size must contain positive height and width")
    gain = min(input_height / original_height, input_width / original_width)
    resized_height = min(input_height, round(original_height * gain))
    resized_width = min(input_width, round(original_width * gain))
    top = round((input_height - resized_height) / 2 - 0.1)
    left = round((input_width - resized_width) / 2 - 0.1)
    cropped = heatmap[top : top + resized_height, left : left + resized_width]
    if cropped.numel() == 0:
        raise ValueError("letterbox crop is empty")
    restored = functional.interpolate(
        cropped[None, None].float(),
        size=original_size,
        mode=interpolation_mode,
        align_corners=align_corners,
    )[0, 0]
    maximum = restored.max()
    restored = (
        restored / maximum
        if torch.isfinite(maximum) and maximum > epsilon
        else torch.zeros_like(restored)
    )
    return restored
