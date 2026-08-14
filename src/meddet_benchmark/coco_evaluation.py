"""Official COCO AP evaluation for canonical detector records."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from numbers import Integral
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .evaluation import ImagePrediction, ImageTarget


def _safe_average(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def _precision_curve(values: np.ndarray) -> list[float | None]:
    """Average a COCO precision slice while preserving its 101 recall positions."""

    if values.ndim < 2:
        raise ValueError("COCO precision curve input must include recall and averaging axes")
    return [
        _safe_average(np.take(values, recall_index, axis=1))
        for recall_index in range(values.shape[1])
    ]


def _record_maps(
    predictions: list[ImagePrediction], targets: list[ImageTarget]
) -> tuple[dict[str, ImagePrediction], dict[str, ImageTarget]]:
    prediction_map = {record.image_id: record for record in predictions}
    target_map = {record.image_id: record for record in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("image_ids must be unique")
    if set(prediction_map) != set(target_map):
        raise ValueError("prediction and target image_id sets must match")
    for image_id, target in target_map.items():
        if prediction_map[image_id].image_size != target.image_size:
            raise ValueError("prediction and target image_size must match")
    return prediction_map, target_map


def _empty_results_dataset(images: list[dict[str, Any]], categories: list[dict[str, Any]]) -> COCO:
    result = COCO()
    result.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": [],
    }
    result.createIndex()
    return result


def evaluate_coco(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    class_ids: tuple[int, ...],
    class_names: dict[int, str] | None = None,
    minimum_score: float = 0.0,
    max_detections: int = 100,
    include_precision_recall: bool = False,
) -> dict[str, Any]:
    """Compute COCO AP and optionally expose its official 101-point PR curves."""

    if (
        not class_ids
        or len(set(class_ids)) != len(class_ids)
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) or item < 0
            for item in class_ids
        )
    ):
        raise ValueError("class_ids must be unique non-negative integers")
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be in [0, 1]")
    if max_detections < 10:
        raise ValueError("COCO evaluation requires max_detections >= 10")
    if class_names is not None and set(class_names) != set(class_ids):
        raise ValueError("class_names keys must match class_ids")

    prediction_map, target_map = _record_maps(predictions, targets)
    allowed = set(class_ids)
    category_map = {class_id: position + 1 for position, class_id in enumerate(class_ids)}
    image_map = {image_id: position + 1 for position, image_id in enumerate(sorted(target_map))}
    categories = [
        {
            "id": category_map[class_id],
            "name": class_names[class_id] if class_names else str(class_id),
        }
        for class_id in class_ids
    ]
    images = [
        {
            "id": image_map[image_id],
            "file_name": image_id,
            "height": target_map[image_id].image_size[0],
            "width": target_map[image_id].image_size[1],
        }
        for image_id in sorted(target_map)
    ]

    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id in sorted(target_map):
        target, prediction = target_map[image_id], prediction_map[image_id]
        if not set(target.labels).issubset(allowed) or not set(prediction.labels).issubset(allowed):
            raise ValueError("record contains a label outside class_ids")
        for box, label in zip(target.boxes_xyxy, target.labels, strict=True):
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_map[image_id],
                    "category_id": category_map[int(label)],
                    "bbox": [float(box[0]), float(box[1]), width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for box, label, score in zip(
            prediction.boxes_xyxy, prediction.labels, prediction.scores, strict=True
        ):
            if score < minimum_score:
                continue
            detections.append(
                {
                    "image_id": image_map[image_id],
                    "category_id": category_map[int(label)],
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    "score": float(score),
                }
            )

    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    with redirect_stdout(StringIO()):
        ground_truth.createIndex()
        results = (
            ground_truth.loadRes(detections)
            if detections
            else _empty_results_dataset(images, categories)
        )
        evaluator = COCOeval(ground_truth, results, iouType="bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.params.catIds = [category["id"] for category in categories]
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    iou_50_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    per_class: dict[str, dict[str, float | None]] = {}
    for position, class_id in enumerate(class_ids):
        all_thresholds = precision[:, :, position, 0, -1]
        at_50 = precision[iou_50_index, :, position, 0, -1]
        per_class[str(class_id)] = {
            "ap50_95": _safe_average(all_thresholds),
            "ap50": _safe_average(at_50),
        }

    stats = evaluator.stats
    result = {
        "ap50_95": float(stats[0]) if stats[0] >= 0 else None,
        "ap50": float(stats[1]) if stats[1] >= 0 else None,
        "per_class": per_class,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "prediction_count": len(detections),
        "max_detections": max_detections,
    }
    if include_precision_recall:
        # precision has shape [IoU, recall, class, area, max detections]. Values
        # below zero are pycocotools' sentinel for an undefined category/recall.
        full_slice = precision[:, :, :, 0, -1]
        iou_50_slice = precision[iou_50_index : iou_50_index + 1, :, :, 0, -1]
        result["precision_recall"] = {
            "recall": [float(value) for value in evaluator.params.recThrs],
            "precision_iou_50": _precision_curve(iou_50_slice),
            "precision_iou_50_95": _precision_curve(full_slice),
            "per_class": {
                str(class_id): {
                    "precision_iou_50": _precision_curve(
                        iou_50_slice[:, :, position : position + 1]
                    ),
                    "precision_iou_50_95": _precision_curve(
                        full_slice[:, :, position : position + 1]
                    ),
                }
                for position, class_id in enumerate(class_ids)
            },
        }
    return result
