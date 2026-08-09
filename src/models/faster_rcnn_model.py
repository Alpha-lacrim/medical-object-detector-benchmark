"""Torchvision Faster R-CNN construction and BatchNorm policy helpers."""

from __future__ import annotations

from typing import Any

from src.models.faster_rcnn_config import ModelSettings


def enforce_backbone_trainability(model: Any, trainable_layers: int) -> int:
    """Apply torchvision's ResNet layer-freezing convention deterministically.

    Torchvision otherwise ignores ``trainable_backbone_layers`` when a detector
    is constructed without pretrained weights, which is how smoke and
    finalize-only recovery avoid network downloads.
    """

    if isinstance(trainable_layers, bool) or not isinstance(trainable_layers, int):
        raise TypeError("trainable_layers must be an integer")
    if not 0 <= trainable_layers <= 5:
        raise ValueError("trainable_layers must be between 0 and 5")
    ordered_layers = ["layer4", "layer3", "layer2", "layer1", "conv1"]
    trainable_prefixes = ordered_layers[:trainable_layers]
    if trainable_layers == 5:
        trainable_prefixes.append("bn1")
    changed = 0
    for name, parameter in model.backbone.body.named_parameters():
        requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)
        if bool(parameter.requires_grad) != requires_grad:
            changed += 1
        parameter.requires_grad_(requires_grad)
    return changed


def build_faster_rcnn(
    num_foreground_classes: int,
    settings: ModelSettings,
    *,
    use_pretrained_weights: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Build the configured detector and replace its COCO classification head.

    The foreground count is derived from the canonical COCO categories. Model
    label zero remains reserved for background.
    """

    if isinstance(num_foreground_classes, bool) or num_foreground_classes <= 0:
        raise ValueError("num_foreground_classes must be a positive integer")

    try:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ModuleNotFoundError as error:
        if error.name in {"torch", "torchvision"}:
            raise RuntimeError("PyTorch and torchvision are required for Faster R-CNN") from error
        raise

    weights = (
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if use_pretrained_weights else None
    )
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=(
            settings.trainable_backbone_layers if weights is not None else None
        ),
        min_size=settings.min_size,
        max_size=settings.max_size,
        image_mean=list(settings.image_mean),
        image_std=list(settings.image_std),
        box_detections_per_img=settings.box_detections_per_image,
        box_score_thresh=settings.box_score_threshold,
        box_nms_thresh=settings.box_nms_threshold,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        input_features,
        num_foreground_classes + 1,
    )
    enforce_backbone_trainability(model, settings.trainable_backbone_layers)

    weight_metadata: dict[str, Any] = {}
    if weights is not None:
        weight_metadata = {
            "name": weights.name,
            "torchvision_reference_gflops": weights.meta.get("_ops"),
            "torchvision_reference_metrics": weights.meta.get("_metrics"),
            "categories_in_pretrained_head": len(weights.meta.get("categories", ())),
            "configured_trainable_backbone_layers": settings.trainable_backbone_layers,
        }
    else:
        weight_metadata = {
            "name": None,
            "torchvision_reference_gflops": None,
            "torchvision_reference_metrics": None,
            "categories_in_pretrained_head": 0,
            "configured_trainable_backbone_layers": settings.trainable_backbone_layers,
        }
    return model, weight_metadata


def freeze_batch_norm_statistics(model: Any) -> int:
    """Put every BatchNorm layer in eval mode while leaving affine parameters trainable."""

    try:
        from torch.nn.modules.batchnorm import _BatchNorm
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to configure BatchNorm") from error

    count = 0
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()
            count += 1
    return count


def apply_training_mode(model: Any, settings: ModelSettings) -> int:
    """Enable detector training and apply the configured BatchNorm policy."""

    model.train()
    if settings.batch_norm_policy == "freeze_statistics":
        return freeze_batch_norm_statistics(model)
    if settings.batch_norm_policy == "train":
        return 0
    raise ValueError(f"Unsupported BatchNorm policy: {settings.batch_norm_policy}")


def count_parameters(model: Any) -> dict[str, int]:
    """Return exact total and trainable parameter counts."""

    parameters = tuple(model.parameters())
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }
