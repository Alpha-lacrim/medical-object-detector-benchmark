import numpy as np
import pytest

from src.explainability.gradcam import (
    gradcam_for_score,
    resolve_module,
    restore_letterboxed_heatmap,
)


def test_gradcam_uses_global_average_gradient_weights() -> None:
    torch = pytest.importorskip("torch")
    activation = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]]],
        requires_grad=True,
    )
    score = activation[:, 0].sum()

    heatmap = gradcam_for_score(
        score,
        activation,
        output_size=(2, 2),
        interpolation_mode="bilinear",
        align_corners=False,
        epsilon=1e-12,
        retain_graph=False,
    )

    np.testing.assert_allclose(heatmap.detach().numpy(), [[0, 1 / 3], [2 / 3, 1]])


def test_module_resolution_supports_attributes_and_indices() -> None:
    torch = pytest.importorskip("torch")
    root = torch.nn.Module()
    root.blocks = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))

    assert resolve_module(root, "blocks.0") is root.blocks[0]
    with pytest.raises(ValueError, match="cannot resolve"):
        resolve_module(root, "missing")


def test_letterbox_restoration_removes_centered_padding() -> None:
    torch = pytest.importorskip("torch")
    letterboxed = torch.zeros((4, 4))
    letterboxed[1:3] = 1

    restored = restore_letterboxed_heatmap(
        letterboxed,
        original_size=(2, 4),
        interpolation_mode="bilinear",
        align_corners=False,
        epsilon=1e-12,
    )

    assert tuple(restored.shape) == (2, 4)
    assert torch.all(restored == 1)
