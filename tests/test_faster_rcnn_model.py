from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.models.faster_rcnn_model import enforce_backbone_trainability


class FakeParameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self


class FakeBody:
    def __init__(self) -> None:
        self.parameters = {
            "conv1.weight": FakeParameter(True),
            "bn1.weight": FakeParameter(True),
            "layer1.0.conv1.weight": FakeParameter(True),
            "layer2.0.conv1.weight": FakeParameter(False),
            "layer3.0.conv1.weight": FakeParameter(False),
            "layer4.0.conv1.weight": FakeParameter(False),
        }

    def named_parameters(self):
        return self.parameters.items()


def test_configured_three_layer_freeze_mask_is_enforced_without_torchvision() -> None:
    body = FakeBody()
    model = SimpleNamespace(backbone=SimpleNamespace(body=body))

    enforce_backbone_trainability(model, 3)

    assert body.parameters["conv1.weight"].requires_grad is False
    assert body.parameters["bn1.weight"].requires_grad is False
    assert body.parameters["layer1.0.conv1.weight"].requires_grad is False
    assert body.parameters["layer2.0.conv1.weight"].requires_grad is True
    assert body.parameters["layer3.0.conv1.weight"].requires_grad is True
    assert body.parameters["layer4.0.conv1.weight"].requires_grad is True


def test_all_backbone_stages_include_stem_batch_norm_at_five_layers() -> None:
    body = FakeBody()
    model = SimpleNamespace(backbone=SimpleNamespace(body=body))

    enforce_backbone_trainability(model, 5)

    assert all(parameter.requires_grad for parameter in body.parameters.values())


@pytest.mark.parametrize("value", [-1, 6])
def test_invalid_trainable_layer_count_is_rejected(value: int) -> None:
    model = SimpleNamespace(backbone=SimpleNamespace(body=FakeBody()))
    with pytest.raises(ValueError, match="between 0 and 5"):
        enforce_backbone_trainability(model, value)
