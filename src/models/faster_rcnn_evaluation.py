"""Validation inference and unified detection metrics for Faster R-CNN."""

from __future__ import annotations

import time
from typing import Any

from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    evaluate_operating_point,
)
from src.models.faster_rcnn_config import EvaluationSettings


def _scalar_image_id(value: Any) -> str:
    """Convert a one-element image-id tensor to the canonical string key."""

    if value.numel() != 1:
        raise ValueError("target image_id must contain exactly one value")
    return str(int(value.reshape(-1)[0].item()))


def _remap_labels(labels: Any, label_to_category_id: dict[int, int]) -> list[int]:
    """Map contiguous detector labels back to canonical COCO category IDs."""

    result: list[int] = []
    for raw_label in labels.detach().cpu().tolist():
        label = int(raw_label)
        if label not in label_to_category_id:
            raise ValueError(f"model produced unknown foreground label {label}")
        result.append(label_to_category_id[label])
    return result


def _records_for_image(
    image: Any,
    target: dict[str, Any],
    output: dict[str, Any],
    *,
    label_to_category_id: dict[int, int],
) -> tuple[ImagePrediction, ImageTarget]:
    """Convert one torchvision target/output pair to tested evaluator records."""

    height, width = int(image.shape[-2]), int(image.shape[-1])
    image_id = _scalar_image_id(target["image_id"])
    prediction = ImagePrediction(
        image_id=image_id,
        image_size=(height, width),
        boxes_xyxy=output["boxes"].detach().cpu().numpy(),
        labels=_remap_labels(output["labels"], label_to_category_id),
        scores=output["scores"].detach().cpu().numpy(),
    )
    truth = ImageTarget(
        image_id=image_id,
        image_size=(height, width),
        boxes_xyxy=target["boxes"].detach().cpu().numpy(),
        labels=_remap_labels(target["labels"], label_to_category_id),
    )
    return prediction, truth


def evaluate_model(
    model: Any,
    data_loader: Any,
    *,
    device: Any,
    amp_enabled: bool,
    amp_dtype: Any,
    label_to_category_id: dict[int, int],
    category_names: dict[int, str],
    settings: EvaluationSettings,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Run eval-mode inference and compute COCO AP plus fixed-point metrics."""

    import torch

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when supplied")
    if set(label_to_category_id.values()) != set(category_names):
        raise ValueError("label/category mappings do not match category names")

    model.eval()
    predictions: list[ImagePrediction] = []
    targets: list[ImageTarget] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (batch_images, batch_targets) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            device_images = [image.to(device, non_blocking=True) for image in batch_images]
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                batch_outputs = model(device_images)
            for image, target, output in zip(
                batch_images, batch_targets, batch_outputs, strict=True
            ):
                prediction, truth = _records_for_image(
                    image,
                    target,
                    output,
                    label_to_category_id=label_to_category_id,
                )
                predictions.append(prediction)
                targets.append(truth)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    if not targets:
        raise RuntimeError("validation produced no images")

    class_ids = tuple(sorted(category_names))
    coco = evaluate_coco(
        predictions,
        targets,
        class_ids=class_ids,
        class_names=category_names,
        minimum_score=settings.coco_minimum_score,
        max_detections=settings.max_detections,
    )
    operating_point = evaluate_operating_point(
        predictions,
        targets,
        class_ids=class_ids,
        score_threshold=settings.score_threshold,
        iou_threshold=settings.match_iou_threshold,
        max_detections=settings.max_detections,
    )["overall"]
    metrics = {
        "val_precision": operating_point["precision"],
        "val_recall": operating_point["recall"],
        "val_f1": operating_point["f1"],
        "val_map_50": coco["ap50"],
        "val_map_50_95": coco["ap50_95"],
        "val_true_positives": operating_point["tp"],
        "val_false_positives": operating_point["fp"],
        "val_false_negatives": operating_point["fn"],
        "val_image_count": coco["image_count"],
        "val_annotation_count": coco["annotation_count"],
        "val_prediction_count": operating_point["prediction_count"],
        "val_coco_prediction_count": coco["prediction_count"],
    }
    return metrics, elapsed_seconds
