"""Deterministic paired qualitative case selection from frozen predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    box_iou,
    match_image,
)

DETECTORS = ("faster_rcnn", "yolo11s")


def operating_evidence(
    prediction: ImagePrediction,
    target: ImageTarget,
    *,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    """Return per-target TP/FN and per-prediction TP/FP frozen evidence."""

    result = match_image(
        prediction,
        target,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    match_by_target = {item.target_index: item for item in result.matches}
    matched_predictions = {item.prediction_index for item in result.matches}
    target_rows: list[dict[str, Any]] = []
    for target_index in range(len(target.boxes_xyxy)):
        match = match_by_target.get(target_index)
        target_rows.append(
            {
                "status": "true_positive" if match is not None else "false_negative",
                "prediction_index": match.prediction_index if match is not None else None,
                "score": match.score if match is not None else None,
                "iou": match.iou if match is not None else None,
            }
        )
    prediction_rows = [
        {
            "status": "true_positive" if index in matched_predictions else "false_positive",
            "prediction_index": int(index),
            "score": float(prediction.scores[index]),
            "box_xyxy": prediction.boxes_xyxy[index].tolist(),
        }
        for index in result.prediction_indices
    ]
    return {"targets": target_rows, "predictions": prediction_rows}


def _best_proxy(
    prediction: ImagePrediction,
    target: ImageTarget,
    target_index: int,
) -> dict[str, Any]:
    label = int(target.labels[target_index])
    eligible = np.flatnonzero(prediction.labels == label)
    if len(eligible) == 0:
        return {"prediction_index": None, "score": 0.0, "iou": 0.0, "box_xyxy": None}
    overlaps = box_iou(
        prediction.boxes_xyxy[eligible],
        target.boxes_xyxy[target_index : target_index + 1],
    )[:, 0]
    best_iou = float(overlaps.max())
    tied = eligible[np.flatnonzero(np.isclose(overlaps, best_iou, rtol=0, atol=1e-12))]
    best = int(tied[np.argmax(prediction.scores[tied])])
    return {
        "prediction_index": best,
        "score": float(prediction.scores[best]),
        "iou": float(
            box_iou(
                prediction.boxes_xyxy[best : best + 1],
                target.boxes_xyxy[target_index : target_index + 1],
            )[0, 0]
        ),
        "box_xyxy": prediction.boxes_xyxy[best].tolist(),
    }


def _unique_top(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for candidate in candidates:
        image_id = str(candidate["image_id"])
        if image_id in used_images:
            continue
        selected.append(candidate)
        used_images.add(image_id)
        if len(selected) == count:
            break
    return selected


def _quantile_unique(
    candidates: list[dict[str, Any]], quantiles: tuple[float, ...]
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for quantile in quantiles:
        center = round(quantile * (len(candidates) - 1))
        index_order = sorted(
            range(len(candidates)),
            key=lambda index: (abs(index - center), index),
        )
        choice = next(
            (
                candidates[index]
                for index in index_order
                if candidates[index]["image_id"] not in used_images
            ),
            None,
        )
        if choice is None:
            break
        choice = {**choice, "selection_quantile": quantile}
        selected.append(choice)
        used_images.add(str(choice["image_id"]))
    return selected


def select_qualitative_cases(
    predictions: Mapping[str, Mapping[str, ImagePrediction]],
    targets: Mapping[str, ImageTarget],
    *,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    cases_per_category: int,
    failure_quantiles: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Select paired good, false-positive, and false-negative cases objectively."""

    if set(predictions) != set(DETECTORS):
        raise ValueError("paired case selection requires Faster R-CNN and YOLO11s")
    if any(set(items) != set(targets) for items in predictions.values()):
        raise ValueError("prediction and target image sets must match")
    evidence = {
        detector: {
            image_id: operating_evidence(
                predictions[detector][image_id],
                target,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
            for image_id, target in targets.items()
        }
        for detector in DETECTORS
    }

    good: list[dict[str, Any]] = []
    failure: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for image_id in sorted(targets):
        target = targets[image_id]
        for target_index in range(len(target.boxes_xyxy)):
            target_evidence = {
                detector: evidence[detector][image_id]["targets"][target_index]
                for detector in DETECTORS
            }
            if all(item["status"] == "true_positive" for item in target_evidence.values()):
                minimum_iou = min(float(item["iou"]) for item in target_evidence.values())
                good.append(
                    {
                        "category": "good_prediction",
                        "image_id": image_id,
                        "target_index": target_index,
                        "selection_score": minimum_iou,
                        "frozen_evidence": target_evidence,
                    }
                )
            elif all(item["status"] == "false_negative" for item in target_evidence.values()):
                proxies = {
                    detector: _best_proxy(predictions[detector][image_id], target, target_index)
                    for detector in DETECTORS
                }
                failure.append(
                    {
                        "category": "failure_case",
                        "image_id": image_id,
                        "target_index": target_index,
                        "selection_score": float(
                            np.mean([item["iou"] for item in proxies.values()])
                        ),
                        "frozen_evidence": proxies,
                    }
                )

        if len(target.boxes_xyxy) == 0:
            top_false_positives: dict[str, dict[str, Any]] = {}
            for detector in DETECTORS:
                false_positives = [
                    item
                    for item in evidence[detector][image_id]["predictions"]
                    if item["status"] == "false_positive"
                ]
                if false_positives:
                    top_false_positives[detector] = max(
                        false_positives, key=lambda item: float(item["score"])
                    )
            if set(top_false_positives) == set(DETECTORS):
                bad.append(
                    {
                        "category": "bad_prediction",
                        "image_id": image_id,
                        "target_index": None,
                        "selection_score": min(
                            float(item["score"]) for item in top_false_positives.values()
                        ),
                        "frozen_evidence": top_false_positives,
                    }
                )

    good.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            str(item["image_id"]),
            int(item["target_index"]),
        )
    )
    bad.sort(key=lambda item: (-float(item["selection_score"]), str(item["image_id"])))
    failure.sort(
        key=lambda item: (
            float(item["selection_score"]),
            str(item["image_id"]),
            int(item["target_index"]),
        )
    )
    chosen = [
        *_unique_top(good, cases_per_category),
        *_unique_top(bad, cases_per_category),
        *_quantile_unique(failure, failure_quantiles),
    ]
    category_counts = {
        category: sum(item["category"] == category for item in chosen)
        for category in ("good_prediction", "bad_prediction", "failure_case")
    }
    if any(value != cases_per_category for value in category_counts.values()):
        raise ValueError(f"insufficient paired qualitative cases: {category_counts}")
    ranks = {category: 0 for category in category_counts}
    for item in chosen:
        category = str(item["category"])
        ranks[category] += 1
        item["rank"] = ranks[category]
    return chosen
