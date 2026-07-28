"""Canonical detection records and shared operating-point evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _image_size(value: tuple[int, int]) -> tuple[int, int]:
    if (
        len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, Integral) for item in value)
        or any(item <= 0 for item in value)
    ):
        raise ValueError("image_size must contain positive integer height and width")
    return int(value[0]), int(value[1])


def _boxes(value: ArrayLike, size: tuple[int, int] | None = None) -> FloatArray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes_xyxy must have strictly positive area")
    if size is not None:
        height, width = size
        if (
            np.any(boxes[:, :2] < 0)
            or np.any(boxes[:, 0::2] > width)
            or np.any(boxes[:, 1::2] > height)
        ):
            raise ValueError("boxes_xyxy must be inside image bounds")
    boxes = boxes.copy()
    boxes.setflags(write=False)
    return boxes


def _labels(value: ArrayLike, count: int) -> IntArray:
    labels = np.asarray(value)
    if count == 0 and labels.size == 0:
        labels = np.empty(0, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError("labels must have shape (N,) matching boxes")
    if labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("labels must be non-negative integers")
    labels = labels.astype(np.int64, copy=True)
    labels.setflags(write=False)
    return labels


def _scores(value: ArrayLike, count: int) -> FloatArray:
    scores = np.asarray(value, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != count:
        raise ValueError("scores must have shape (N,) matching boxes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite values in [0, 1]")
    scores = scores.copy()
    scores.setflags(write=False)
    return scores


@dataclass(frozen=True)
class ImageTarget:
    image_id: str
    image_size: tuple[int, int]
    boxes_xyxy: ArrayLike
    labels: ArrayLike

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id must be non-empty")
        size = _image_size(self.image_size)
        boxes = _boxes(self.boxes_xyxy, size)
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "labels", _labels(self.labels, len(boxes)))


@dataclass(frozen=True)
class ImagePrediction:
    image_id: str
    image_size: tuple[int, int]
    boxes_xyxy: ArrayLike
    labels: ArrayLike
    scores: ArrayLike

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id must be non-empty")
        size = _image_size(self.image_size)
        boxes = _boxes(self.boxes_xyxy, size)
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "labels", _labels(self.labels, len(boxes)))
        object.__setattr__(self, "scores", _scores(self.scores, len(boxes)))


@dataclass(frozen=True)
class Match:
    prediction_index: int
    target_index: int
    label: int
    score: float
    iou: float


@dataclass(frozen=True)
class ImageMatchResult:
    image_id: str
    matches: tuple[Match, ...]
    prediction_indices: tuple[int, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]


def _pairwise_iou(boxes_a: FloatArray, boxes_b: FloatArray) -> FloatArray:
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def box_iou(boxes_a: ArrayLike, boxes_b: ArrayLike) -> FloatArray:
    """Pairwise continuous-coordinate IoU without the legacy ``+1`` convention."""

    return _pairwise_iou(_boxes(boxes_a), _boxes(boxes_b))


def match_image(
    prediction: ImagePrediction,
    target: ImageTarget,
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> ImageMatchResult:
    """Greedily match score-ordered predictions to same-class targets."""

    if prediction.image_id != target.image_id or prediction.image_size != target.image_size:
        raise ValueError("prediction and target identity and image_size must match")
    if not 0 <= score_threshold <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("thresholds must be within their probability ranges")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    eligible = np.flatnonzero(prediction.scores >= score_threshold)
    order = eligible[np.argsort(-prediction.scores[eligible], kind="stable")][:max_detections]
    unmatched_targets = set(range(len(target.boxes_xyxy)))
    matches: list[Match] = []

    for prediction_index in order:
        same_class = [
            index
            for index in sorted(unmatched_targets)
            if target.labels[index] == prediction.labels[prediction_index]
        ]
        if not same_class:
            continue
        overlaps = _pairwise_iou(
            prediction.boxes_xyxy[prediction_index : prediction_index + 1],
            target.boxes_xyxy[same_class],
        )[0]
        best_position = int(np.argmax(overlaps))
        if overlaps[best_position] >= iou_threshold:
            target_index = same_class[best_position]
            matches.append(
                Match(
                    prediction_index=int(prediction_index),
                    target_index=target_index,
                    label=int(prediction.labels[prediction_index]),
                    score=float(prediction.scores[prediction_index]),
                    iou=float(overlaps[best_position]),
                )
            )
            unmatched_targets.remove(target_index)

    matched_predictions = {match.prediction_index for match in matches}
    return ImageMatchResult(
        image_id=target.image_id,
        matches=tuple(matches),
        prediction_indices=tuple(int(index) for index in order),
        unmatched_prediction_indices=tuple(
            int(index) for index in order if index not in matched_predictions
        ),
        unmatched_target_indices=tuple(sorted(unmatched_targets)),
    )


def _summary(tp: int, fp: int, fn: int, ious: list[float]) -> dict[str, Any]:
    prediction_count, target_count = tp + fp, tp + fn
    if prediction_count == 0 and target_count == 0:
        precision = recall = f1 = None
    elif prediction_count == 0 or target_count == 0:
        precision = recall = f1 = 0.0
    else:
        precision = tp / prediction_count
        recall = tp / target_count
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "prediction_count": prediction_count,
        "target_count": target_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mean_iou": float(np.mean(ious)) if ious else None,
        "matched_mean_box_dice": (
            float(np.mean([2 * iou / (1 + iou) for iou in ious])) if ious else None
        ),
    }


def _summarize_class(
    prediction: ImagePrediction,
    target: ImageTarget,
    result: ImageMatchResult,
    class_id: int | None,
) -> dict[str, Any]:
    matches = [match for match in result.matches if class_id is None or match.label == class_id]
    fp = int(
        sum(
            class_id is None or prediction.labels[index] == class_id
            for index in result.unmatched_prediction_indices
        )
    )
    fn = int(
        sum(
            class_id is None or target.labels[index] == class_id
            for index in result.unmatched_target_indices
        )
    )
    return _summary(len(matches), fp, fn, [match.iou for match in matches])


def evaluate_operating_point(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    class_ids: tuple[int, ...],
    score_threshold: float,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> dict[str, Any]:
    """Evaluate one frozen score/IoU operating point with JSON-safe output."""

    if (
        not class_ids
        or len(set(class_ids)) != len(class_ids)
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) or item < 0
            for item in class_ids
        )
    ):
        raise ValueError("class_ids must be unique non-negative integers")
    prediction_map = {item.image_id: item for item in predictions}
    target_map = {item.image_id: item for item in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("image_ids must be unique")
    if set(prediction_map) != set(target_map):
        raise ValueError("prediction and target image_id sets must match")

    allowed = set(class_ids)
    per_image: list[dict[str, Any]] = []
    per_class_parts: dict[int, list[dict[str, Any]]] = {item: [] for item in class_ids}
    per_class_ious: dict[int, list[float]] = {item: [] for item in class_ids}
    overall_parts: list[dict[str, Any]] = []
    overall_ious: list[float] = []
    for image_id in sorted(target_map):
        prediction, target = prediction_map[image_id], target_map[image_id]
        if not set(prediction.labels).issubset(allowed) or not set(target.labels).issubset(allowed):
            raise ValueError("record contains a label outside class_ids")
        result = match_image(
            prediction,
            target,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        image_summary = _summarize_class(prediction, target, result, None)
        per_image.append({"image_id": image_id, **image_summary})
        overall_parts.append(image_summary)
        overall_ious.extend(match.iou for match in result.matches)
        for class_id in class_ids:
            per_class_parts[class_id].append(_summarize_class(prediction, target, result, class_id))
            per_class_ious[class_id].extend(
                match.iou for match in result.matches if match.label == class_id
            )

    def combine(parts: list[dict[str, Any]], ious: list[float]) -> dict[str, Any]:
        return _summary(
            sum(part["tp"] for part in parts),
            sum(part["fp"] for part in parts),
            sum(part["fn"] for part in parts),
            ious,
        )

    per_class = {
        str(class_id): combine(parts, per_class_ious[class_id])
        for class_id, parts in per_class_parts.items()
    }
    overall = combine(overall_parts, overall_ious)
    macro = {}
    for metric in ("precision", "recall", "f1", "matched_mean_iou", "matched_mean_box_dice"):
        values = [entry[metric] for entry in per_class.values() if entry[metric] is not None]
        macro[metric] = float(np.mean(values)) if values else None
    return {"overall": overall, "macro": macro, "per_class": per_class, "per_image": per_image}
