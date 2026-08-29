"""Fast paired resampling for aggregate object-detection metrics.

The implementation preserves the benchmark's metric definitions. Operating-
point ratios are rebuilt from per-image TP/FP/FN and localization sums. COCO
AP is rebuilt from per-image, score-ordered detection matches at the ten COCO
IoU thresholds, so bootstrap samples and detector-label permutations do not
replace mAP with an unrelated per-image surrogate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    box_iou,
    match_image,
)

METRICS = ("precision", "recall", "f1", "iou", "dice", "map_50", "map_50_95")
METRIC_UNITS = {metric: "ratio" for metric in METRICS}
COCO_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10, dtype=np.float64)
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class DetectionEvidence:
    """Per-image sufficient statistics for all seven predictive metrics."""

    image_ids: tuple[str, ...]
    target_count: IntArray
    tp: IntArray
    fp: IntArray
    fn: IntArray
    iou_sum: FloatArray
    dice_sum: FloatArray
    coco_scores: tuple[FloatArray, ...]
    coco_matches: tuple[BoolArray, ...]
    coco_flat_scores: FloatArray
    coco_flat_image_indices: IntArray
    coco_flat_matches: BoolArray

    @property
    def image_count(self) -> int:
        return len(self.image_ids)


@dataclass(frozen=True)
class EvidencePair:
    """Matched detector evidence, optionally repeated over training seeds."""

    detector_a: tuple[DetectionEvidence, ...]
    detector_b: tuple[DetectionEvidence, ...]

    def __post_init__(self) -> None:
        if not self.detector_a or len(self.detector_a) != len(self.detector_b):
            raise ValueError("evidence pairs must contain the same non-zero number of seeds")
        reference = self.detector_a[0].image_ids
        if any(item.image_ids != reference for item in (*self.detector_a, *self.detector_b)):
            raise ValueError("all evidence must use the same ordered image IDs")

    @property
    def image_count(self) -> int:
        return self.detector_a[0].image_count

    @property
    def seed_count(self) -> int:
        return len(self.detector_a)


@dataclass(frozen=True)
class IndependentEvidenceRuns:
    """Detector run evidence without an assumed cross-detector run pairing."""

    detector_a: tuple[DetectionEvidence, ...]
    detector_b: tuple[DetectionEvidence, ...]

    def __post_init__(self) -> None:
        if not self.detector_a or not self.detector_b:
            raise ValueError("each detector must contribute at least one trained run")
        reference = self.detector_a[0].image_ids
        if any(item.image_ids != reference for item in (*self.detector_a, *self.detector_b)):
            raise ValueError("all detector runs must use the same ordered image IDs")

    @property
    def image_count(self) -> int:
        return self.detector_a[0].image_count

    @property
    def detector_a_run_count(self) -> int:
        return len(self.detector_a)

    @property
    def detector_b_run_count(self) -> int:
        return len(self.detector_b)


@dataclass(frozen=True)
class PatientClusters:
    """Patient-group membership aligned to an evidence pair's image order."""

    image_ids: tuple[str, ...]
    patient_group_ids: tuple[str, ...]
    image_to_group: IntArray

    def __post_init__(self) -> None:
        if not self.image_ids or not self.patient_group_ids:
            raise ValueError("patient clusters must contain images and groups")
        if self.image_to_group.shape != (len(self.image_ids),):
            raise ValueError("patient-group indices must match image count")
        if not np.issubdtype(self.image_to_group.dtype, np.integer):
            raise ValueError("patient-group indices must be integers")
        if np.any(self.image_to_group < 0) or np.any(
            self.image_to_group >= len(self.patient_group_ids)
        ):
            raise ValueError("patient-group indices are out of range")
        if set(self.image_to_group.tolist()) != set(range(len(self.patient_group_ids))):
            raise ValueError("every declared patient group must contain at least one image")

    @property
    def patient_group_count(self) -> int:
        return len(self.patient_group_ids)


def build_patient_clusters(
    image_ids: tuple[str, ...], group_by_image: Mapping[str, str]
) -> PatientClusters:
    """Build deterministic patient membership for the requested image IDs."""

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image IDs must be unique")
    missing = sorted(set(image_ids) - set(group_by_image))
    if missing:
        raise ValueError(f"patient mapping lacks image IDs: {missing[:5]}")
    group_ids_by_image = tuple(str(group_by_image[image_id]).strip() for image_id in image_ids)
    if any(not group_id for group_id in group_ids_by_image):
        raise ValueError("patient group IDs must be non-empty")
    patient_group_ids = tuple(sorted(set(group_ids_by_image)))
    group_indices = {group_id: index for index, group_id in enumerate(patient_group_ids)}
    image_to_group = np.asarray(
        [group_indices[group_id] for group_id in group_ids_by_image], dtype=np.int64
    )
    return PatientClusters(
        image_ids=image_ids,
        patient_group_ids=patient_group_ids,
        image_to_group=image_to_group,
    )


def expand_patient_group_multiplicities(
    patient_group_multiplicities: IntArray, patient_clusters: PatientClusters
) -> IntArray:
    """Apply each sampled patient multiplicity to all images in that patient group."""

    if patient_group_multiplicities.shape != (patient_clusters.patient_group_count,):
        raise ValueError("patient-group multiplicities must match patient-group count")
    if not np.issubdtype(patient_group_multiplicities.dtype, np.integer) or np.any(
        patient_group_multiplicities < 0
    ):
        raise ValueError("patient-group multiplicities must be non-negative integers")
    return patient_group_multiplicities[patient_clusters.image_to_group]


def expand_patient_group_choices(
    patient_group_choices: BoolArray, patient_clusters: PatientClusters
) -> BoolArray:
    """Expand one detector-label choice per patient to its complete image cluster."""

    if patient_group_choices.shape != (patient_clusters.patient_group_count,):
        raise ValueError("patient-group choices must match patient-group count")
    if patient_group_choices.dtype != np.bool_:
        raise ValueError("patient-group choices must be boolean")
    return patient_group_choices[patient_clusters.image_to_group]


def draw_hierarchical_bootstrap_multiplicities(
    rng: np.random.Generator,
    patient_clusters: PatientClusters,
    *,
    seed_count: int,
) -> tuple[IntArray, IntArray]:
    """Draw one existing patient-cluster/seed hierarchical bootstrap sample.

    Patient groups are sampled with replacement and expanded so every image
    from the same patient receives the same multiplicity. Training seeds are
    sampled independently with replacement. Keeping this draw in one shared
    helper prevents downstream analyses from subtly changing the established
    Phase 8 resampling unit.
    """

    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    patient_group_count = patient_clusters.patient_group_count
    patient_probabilities = np.full(patient_group_count, 1 / patient_group_count, dtype=np.float64)
    patient_group_counts = rng.multinomial(patient_group_count, patient_probabilities).astype(
        np.int64, copy=False
    )
    image_multiplicities = expand_patient_group_multiplicities(
        patient_group_counts, patient_clusters
    )
    if seed_count == 1:
        seed_multiplicities = np.ones(1, dtype=np.int64)
    else:
        seed_probabilities = np.full(seed_count, 1 / seed_count, dtype=np.float64)
        seed_multiplicities = rng.multinomial(seed_count, seed_probabilities).astype(
            np.int64, copy=False
        )
    return image_multiplicities, seed_multiplicities


def draw_patient_cluster_bootstrap_multiplicities(
    rng: np.random.Generator,
    patient_clusters: PatientClusters,
) -> IntArray:
    """Draw patients with replacement and expand one count to every patient image."""

    patient_group_count = patient_clusters.patient_group_count
    probabilities = np.full(patient_group_count, 1 / patient_group_count, dtype=np.float64)
    patient_counts = rng.multinomial(patient_group_count, probabilities).astype(
        np.int64, copy=False
    )
    return expand_patient_group_multiplicities(patient_counts, patient_clusters)


def draw_independent_run_bootstrap_multiplicities(
    rng: np.random.Generator,
    *,
    detector_a_run_count: int,
    detector_b_run_count: int,
) -> tuple[IntArray, IntArray]:
    """Draw trained runs independently within each detector.

    Same-number seed labels do not couple the draws. Each returned multinomial
    vector has its detector's eligible run count as both its length and total.
    """

    for name, count in (
        ("detector_a_run_count", detector_a_run_count),
        ("detector_b_run_count", detector_b_run_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a positive integer")

    def draw(count: int) -> IntArray:
        probabilities = np.full(count, 1 / count, dtype=np.float64)
        return rng.multinomial(count, probabilities).astype(np.int64, copy=False)

    return draw(detector_a_run_count), draw(detector_b_run_count)


@dataclass(frozen=True)
class HybridEvidence:
    """One detector pair with a globally score-sorted union for fast swaps."""

    detector_a: DetectionEvidence
    detector_b: DetectionEvidence
    flat_image_indices: IntArray
    flat_source_is_b: BoolArray
    flat_matches: BoolArray


def prepare_hybrid(evidence_a: DetectionEvidence, evidence_b: DetectionEvidence) -> HybridEvidence:
    if evidence_a.image_ids != evidence_b.image_ids:
        raise ValueError("hybrid evidence image IDs differ")
    scores = np.concatenate((evidence_a.coco_flat_scores, evidence_b.coco_flat_scores))
    image_indices = np.concatenate(
        (evidence_a.coco_flat_image_indices, evidence_b.coco_flat_image_indices)
    )
    source_is_b = np.concatenate(
        (
            np.zeros(len(evidence_a.coco_flat_scores), dtype=np.bool_),
            np.ones(len(evidence_b.coco_flat_scores), dtype=np.bool_),
        )
    )
    matches = np.concatenate((evidence_a.coco_flat_matches, evidence_b.coco_flat_matches), axis=1)
    order = np.argsort(-scores, kind="stable")
    return HybridEvidence(
        detector_a=evidence_a,
        detector_b=evidence_b,
        flat_image_indices=image_indices[order],
        flat_source_is_b=source_is_b[order],
        flat_matches=matches[:, order],
    )


def _record_maps(
    predictions: list[ImagePrediction], targets: list[ImageTarget]
) -> tuple[dict[str, ImagePrediction], dict[str, ImageTarget]]:
    prediction_map = {item.image_id: item for item in predictions}
    target_map = {item.image_id: item for item in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("prediction and target image IDs must be unique")
    if set(prediction_map) != set(target_map):
        raise ValueError("prediction and target image IDs must match")
    return prediction_map, target_map


def _coco_image_matches(
    prediction: ImagePrediction,
    target: ImageTarget,
    *,
    class_ids: tuple[int, ...],
    minimum_score: float,
    max_detections: int,
) -> tuple[FloatArray, BoolArray]:
    allowed = set(class_ids)
    if not set(prediction.labels).issubset(allowed) or not set(target.labels).issubset(allowed):
        raise ValueError("record contains a label outside class_ids")
    eligible = np.flatnonzero(prediction.scores >= minimum_score)
    order = eligible[np.argsort(-prediction.scores[eligible], kind="stable")][:max_detections]
    scores = np.asarray(prediction.scores[order], dtype=np.float64)
    matches = np.zeros((len(COCO_IOU_THRESHOLDS), len(order)), dtype=np.bool_)
    if not len(order) or not len(target.boxes_xyxy):
        return scores, matches

    overlaps = box_iou(prediction.boxes_xyxy[order], target.boxes_xyxy)
    for threshold_index, threshold in enumerate(COCO_IOU_THRESHOLDS):
        unmatched = set(range(len(target.boxes_xyxy)))
        for detection_position, prediction_index in enumerate(order):
            candidates = [
                index
                for index in sorted(unmatched)
                if target.labels[index] == prediction.labels[prediction_index]
            ]
            if not candidates:
                continue
            candidate_overlaps = overlaps[detection_position, candidates]
            best_position = int(np.argmax(candidate_overlaps))
            if candidate_overlaps[best_position] >= threshold:
                matches[threshold_index, detection_position] = True
                unmatched.remove(candidates[best_position])
    return scores, matches


def build_evidence(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    class_ids: tuple[int, ...],
    score_threshold: float,
    match_iou_threshold: float,
    coco_minimum_score: float,
    max_detections: int,
) -> DetectionEvidence:
    """Convert canonical records to metric-preserving per-image evidence."""

    prediction_map, target_map = _record_maps(predictions, targets)
    image_ids = tuple(sorted(target_map))
    target_count = np.zeros(len(image_ids), dtype=np.int64)
    tp = np.zeros(len(image_ids), dtype=np.int64)
    fp = np.zeros(len(image_ids), dtype=np.int64)
    fn = np.zeros(len(image_ids), dtype=np.int64)
    iou_sum = np.zeros(len(image_ids), dtype=np.float64)
    dice_sum = np.zeros(len(image_ids), dtype=np.float64)
    coco_scores: list[FloatArray] = []
    coco_matches: list[BoolArray] = []

    for index, image_id in enumerate(image_ids):
        prediction, target = prediction_map[image_id], target_map[image_id]
        result = match_image(
            prediction,
            target,
            score_threshold=score_threshold,
            iou_threshold=match_iou_threshold,
            max_detections=max_detections,
        )
        match_ious = np.asarray([item.iou for item in result.matches], dtype=np.float64)
        target_count[index] = len(target.boxes_xyxy)
        tp[index] = len(result.matches)
        fp[index] = len(result.unmatched_prediction_indices)
        fn[index] = len(result.unmatched_target_indices)
        iou_sum[index] = float(np.sum(match_ious))
        dice_sum[index] = float(np.sum(2 * match_ious / (1 + match_ious)))
        scores, matches = _coco_image_matches(
            prediction,
            target,
            class_ids=class_ids,
            minimum_score=coco_minimum_score,
            max_detections=max_detections,
        )
        coco_scores.append(scores)
        coco_matches.append(matches)

    nonempty = [index for index, scores in enumerate(coco_scores) if len(scores)]
    if nonempty:
        flat_scores = np.concatenate([coco_scores[index] for index in nonempty])
        flat_image_indices = np.concatenate(
            [np.full(len(coco_scores[index]), index, dtype=np.int64) for index in nonempty]
        )
        flat_matches = np.concatenate([coco_matches[index] for index in nonempty], axis=1)
        order = np.argsort(-flat_scores, kind="stable")
        flat_scores = flat_scores[order]
        flat_image_indices = flat_image_indices[order]
        flat_matches = flat_matches[:, order]
    else:
        flat_scores = np.empty(0, dtype=np.float64)
        flat_image_indices = np.empty(0, dtype=np.int64)
        flat_matches = np.empty((len(COCO_IOU_THRESHOLDS), 0), dtype=np.bool_)

    return DetectionEvidence(
        image_ids=image_ids,
        target_count=target_count,
        tp=tp,
        fp=fp,
        fn=fn,
        iou_sum=iou_sum,
        dice_sum=dice_sum,
        coco_scores=tuple(coco_scores),
        coco_matches=tuple(coco_matches),
        coco_flat_scores=flat_scores,
        coco_flat_image_indices=flat_image_indices,
        coco_flat_matches=flat_matches,
    )


def _average_precision(scores: FloatArray, matches: BoolArray, target_count: int) -> FloatArray:
    if target_count <= 0:
        return np.full(len(COCO_IOU_THRESHOLDS), np.nan, dtype=np.float64)
    if not len(scores):
        return np.zeros(len(COCO_IOU_THRESHOLDS), dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ordered = matches[:, order]
    true_positives = np.cumsum(ordered, axis=1, dtype=np.int64)
    false_positives = np.cumsum(~ordered, axis=1, dtype=np.int64)
    recalls = true_positives / target_count
    precisions = true_positives / (true_positives + false_positives)
    precisions = np.maximum.accumulate(precisions[:, ::-1], axis=1)[:, ::-1]
    values = np.zeros(len(COCO_IOU_THRESHOLDS), dtype=np.float64)
    for threshold_index in range(len(COCO_IOU_THRESHOLDS)):
        positions = np.searchsorted(recalls[threshold_index], COCO_RECALL_THRESHOLDS, side="left")
        valid = positions < precisions.shape[1]
        sampled = np.zeros(len(COCO_RECALL_THRESHOLDS), dtype=np.float64)
        sampled[valid] = precisions[threshold_index, positions[valid]]
        values[threshold_index] = float(np.mean(sampled))
    return values


def _weighted_average_precision(
    matches: BoolArray, detection_weights: IntArray, target_count: int
) -> FloatArray:
    """COCO AP for already score-sorted detections with integer image weights."""

    if target_count <= 0:
        return np.full(len(COCO_IOU_THRESHOLDS), np.nan, dtype=np.float64)
    keep = detection_weights > 0
    if not np.any(keep):
        return np.zeros(len(COCO_IOU_THRESHOLDS), dtype=np.float64)
    selected_matches = matches[:, keep]
    weights = detection_weights[keep]
    true_positives = np.cumsum(selected_matches * weights, axis=1, dtype=np.int64)
    false_positives = np.cumsum((~selected_matches) * weights, axis=1, dtype=np.int64)
    recalls = true_positives / target_count
    precisions = true_positives / (true_positives + false_positives)
    precisions = np.maximum.accumulate(precisions[:, ::-1], axis=1)[:, ::-1]
    values = np.zeros(len(COCO_IOU_THRESHOLDS), dtype=np.float64)
    for threshold_index in range(len(COCO_IOU_THRESHOLDS)):
        positions = np.searchsorted(recalls[threshold_index], COCO_RECALL_THRESHOLDS, side="left")
        valid = positions < precisions.shape[1]
        sampled = np.zeros(len(COCO_RECALL_THRESHOLDS), dtype=np.float64)
        sampled[valid] = precisions[threshold_index, positions[valid]]
        values[threshold_index] = float(np.mean(sampled))
    return values


def _operating_ratios(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prediction_count, target_count = tp + fp, tp + fn
    if prediction_count == 0 and target_count == 0:
        return np.nan, np.nan, np.nan
    if prediction_count == 0 or target_count == 0:
        return 0.0, 0.0, 0.0
    precision = tp / prediction_count
    recall = tp / target_count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _aggregate_single(evidence: DetectionEvidence, multiplicities: IntArray) -> FloatArray:
    total_tp = int(np.dot(multiplicities, evidence.tp))
    total_fp = int(np.dot(multiplicities, evidence.fp))
    total_fn = int(np.dot(multiplicities, evidence.fn))
    precision, recall, f1 = _operating_ratios(total_tp, total_fp, total_fn)
    mean_iou = float(np.dot(multiplicities, evidence.iou_sum) / total_tp) if total_tp else np.nan
    mean_dice = float(np.dot(multiplicities, evidence.dice_sum) / total_tp) if total_tp else np.nan
    detection_weights = multiplicities[evidence.coco_flat_image_indices]
    aps = _weighted_average_precision(
        evidence.coco_flat_matches,
        detection_weights,
        int(np.dot(multiplicities, evidence.target_count)),
    )
    return np.asarray(
        [precision, recall, f1, mean_iou, mean_dice, aps[0], np.mean(aps)],
        dtype=np.float64,
    )


def _selected(evidence_a: DetectionEvidence, evidence_b: DetectionEvidence, mask: BoolArray):
    for name in ("target_count", "tp", "fp", "fn", "iou_sum", "dice_sum"):
        yield np.where(mask, getattr(evidence_b, name), getattr(evidence_a, name))


def aggregate_hybrid(
    evidence_a: DetectionEvidence,
    evidence_b: DetectionEvidence,
    *,
    multiplicities: IntArray,
    choose_b: BoolArray,
) -> FloatArray:
    """Aggregate evidence after choosing detector B for selected image units."""

    n = evidence_a.image_count
    if evidence_b.image_ids != evidence_a.image_ids:
        raise ValueError("hybrid evidence image IDs differ")
    if multiplicities.shape != (n,) or choose_b.shape != (n,):
        raise ValueError("multiplicities and selector must match image count")
    if np.any(multiplicities < 0) or not np.issubdtype(multiplicities.dtype, np.integer):
        raise ValueError("multiplicities must be non-negative integers")
    if not np.any(choose_b):
        return _aggregate_single(evidence_a, multiplicities)
    if np.all(choose_b):
        return _aggregate_single(evidence_b, multiplicities)
    target_count, tp, fp, fn, iou_sum, dice_sum = _selected(evidence_a, evidence_b, choose_b)
    total_tp = int(np.dot(multiplicities, tp))
    total_fp = int(np.dot(multiplicities, fp))
    total_fn = int(np.dot(multiplicities, fn))
    precision, recall, f1 = _operating_ratios(total_tp, total_fp, total_fn)
    mean_iou = float(np.dot(multiplicities, iou_sum) / total_tp) if total_tp else np.nan
    mean_dice = float(np.dot(multiplicities, dice_sum) / total_tp) if total_tp else np.nan

    score_parts: list[FloatArray] = []
    match_parts: list[BoolArray] = []
    for index, count in enumerate(multiplicities):
        if count == 0:
            continue
        evidence = evidence_b if choose_b[index] else evidence_a
        if len(evidence.coco_scores[index]):
            score_parts.append(np.tile(evidence.coco_scores[index], int(count)))
            match_parts.append(np.tile(evidence.coco_matches[index], (1, int(count))))
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float64)
    matches = (
        np.concatenate(match_parts, axis=1)
        if match_parts
        else np.empty((len(COCO_IOU_THRESHOLDS), 0), dtype=np.bool_)
    )
    aps = _average_precision(scores, matches, int(np.dot(multiplicities, target_count)))
    return np.asarray(
        [precision, recall, f1, mean_iou, mean_dice, aps[0], np.mean(aps)],
        dtype=np.float64,
    )


def aggregate_prepared_hybrid(
    prepared: HybridEvidence,
    *,
    multiplicities: IntArray,
    choose_b: BoolArray,
) -> FloatArray:
    """Aggregate a detector-label hybrid without re-sorting its detections."""

    evidence_a, evidence_b = prepared.detector_a, prepared.detector_b
    if not np.any(choose_b):
        return _aggregate_single(evidence_a, multiplicities)
    if np.all(choose_b):
        return _aggregate_single(evidence_b, multiplicities)
    target_count, tp, fp, fn, iou_sum, dice_sum = _selected(evidence_a, evidence_b, choose_b)
    total_tp = int(np.dot(multiplicities, tp))
    total_fp = int(np.dot(multiplicities, fp))
    total_fn = int(np.dot(multiplicities, fn))
    precision, recall, f1 = _operating_ratios(total_tp, total_fp, total_fn)
    mean_iou = float(np.dot(multiplicities, iou_sum) / total_tp) if total_tp else np.nan
    mean_dice = float(np.dot(multiplicities, dice_sum) / total_tp) if total_tp else np.nan
    selected_sources = choose_b[prepared.flat_image_indices]
    included = prepared.flat_source_is_b == selected_sources
    detection_weights = multiplicities[prepared.flat_image_indices] * included.astype(np.int64)
    aps = _weighted_average_precision(
        prepared.flat_matches,
        detection_weights,
        int(np.dot(multiplicities, target_count)),
    )
    return np.asarray(
        [precision, recall, f1, mean_iou, mean_dice, aps[0], np.mean(aps)],
        dtype=np.float64,
    )


def estimate_pair(
    pair: EvidencePair,
    *,
    multiplicities: IntArray,
    seed_multiplicities: IntArray | None = None,
    swap_mask: BoolArray | None = None,
    prepared: tuple[HybridEvidence, ...] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Estimate both detectors while preserving image and seed pairing."""

    if seed_multiplicities is None:
        seed_multiplicities = np.ones(pair.seed_count, dtype=np.int64)
    if seed_multiplicities.shape != (pair.seed_count,) or not np.any(seed_multiplicities):
        raise ValueError("seed multiplicities must match and select at least one seed")
    if swap_mask is None:
        swap_mask = np.zeros(pair.image_count, dtype=np.bool_)
    if prepared is None:
        prepared = tuple(
            prepare_hybrid(evidence_a, evidence_b)
            for evidence_a, evidence_b in zip(pair.detector_a, pair.detector_b, strict=True)
        )
    if len(prepared) != pair.seed_count:
        raise ValueError("prepared hybrid count must match seed count")
    complement = ~swap_mask
    estimate_a = np.zeros(len(METRICS), dtype=np.float64)
    estimate_b = np.zeros(len(METRICS), dtype=np.float64)
    total_seed_weight = int(np.sum(seed_multiplicities))
    for seed_index, weight in enumerate(seed_multiplicities):
        if weight == 0:
            continue
        hybrid = prepared[seed_index]
        estimate_a += int(weight) * aggregate_prepared_hybrid(
            hybrid,
            multiplicities=multiplicities,
            choose_b=swap_mask,
        )
        estimate_b += int(weight) * aggregate_prepared_hybrid(
            hybrid,
            multiplicities=multiplicities,
            choose_b=complement,
        )
    return estimate_a / total_seed_weight, estimate_b / total_seed_weight


def estimate_independent_runs(
    runs: IndependentEvidenceRuns,
    *,
    multiplicities: IntArray,
    detector_a_run_multiplicities: IntArray | None = None,
    detector_b_run_multiplicities: IntArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Estimate each detector over its independently weighted trained runs."""

    if multiplicities.shape != (runs.image_count,):
        raise ValueError("image multiplicities must match detector evidence")
    if detector_a_run_multiplicities is None:
        detector_a_run_multiplicities = np.ones(runs.detector_a_run_count, dtype=np.int64)
    if detector_b_run_multiplicities is None:
        detector_b_run_multiplicities = np.ones(runs.detector_b_run_count, dtype=np.int64)

    def estimate(
        evidence_runs: tuple[DetectionEvidence, ...], run_multiplicities: IntArray
    ) -> FloatArray:
        if run_multiplicities.shape != (len(evidence_runs),):
            raise ValueError("run multiplicities must match eligible detector runs")
        if not np.issubdtype(run_multiplicities.dtype, np.integer) or np.any(
            run_multiplicities < 0
        ):
            raise ValueError("run multiplicities must be non-negative integers")
        total_weight = int(np.sum(run_multiplicities))
        if total_weight < 1:
            raise ValueError("run multiplicities must select at least one trained run")
        result = np.zeros(len(METRICS), dtype=np.float64)
        for evidence, weight in zip(evidence_runs, run_multiplicities, strict=True):
            if weight:
                result += int(weight) * _aggregate_single(evidence, multiplicities)
        return result / total_weight

    return estimate(runs.detector_a, detector_a_run_multiplicities), estimate(
        runs.detector_b, detector_b_run_multiplicities
    )


def stable_rng_seed(base_seed: int, label: str) -> int:
    """Derive an order-independent uint32 seed from a comparison label."""

    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down family-wise-error-adjusted p-values."""

    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    count = len(p_values)
    order = np.argsort(np.asarray(p_values), kind="stable")
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _percentile_interval(values: FloatArray, confidence_level: float) -> tuple[float, float, int]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return np.nan, np.nan, 0
    tail = (1 - confidence_level) / 2
    low, high = np.quantile(valid, [tail, 1 - tail])
    return float(low), float(high), len(valid)


def _retention(
    current: tuple[FloatArray, FloatArray], reference: tuple[FloatArray, FloatArray]
) -> tuple[FloatArray, FloatArray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        return current[0] / reference[0], current[1] / reference[1]


def analyze_pair(
    pair: EvidencePair,
    *,
    patient_clusters: PatientClusters,
    base_seed: int,
    comparison_label: str,
    bootstrap_resamples: int,
    permutation_resamples: int,
    confidence_level: float,
    reference_pair: EvidencePair | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run patient-cluster bootstrap and paired patient-label permutation."""

    if bootstrap_resamples < 100 or permutation_resamples < 100:
        raise ValueError("at least 100 bootstrap and permutation resamples are required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    if reference_pair is not None and (
        reference_pair.image_count != pair.image_count
        or reference_pair.seed_count != pair.seed_count
    ):
        raise ValueError("reference evidence must match current images and seeds")
    if patient_clusters.image_ids != pair.detector_a[0].image_ids:
        raise ValueError("patient clusters must match the evidence image order")

    n, seed_count = pair.image_count, pair.seed_count
    patient_group_count = patient_clusters.patient_group_count
    prepared = tuple(
        prepare_hybrid(evidence_a, evidence_b)
        for evidence_a, evidence_b in zip(pair.detector_a, pair.detector_b, strict=True)
    )
    reference_prepared = (
        tuple(
            prepare_hybrid(evidence_a, evidence_b)
            for evidence_a, evidence_b in zip(
                reference_pair.detector_a, reference_pair.detector_b, strict=True
            )
        )
        if reference_pair is not None
        else None
    )
    ones = np.ones(n, dtype=np.int64)
    seed_ones = np.ones(seed_count, dtype=np.int64)
    point_raw = estimate_pair(
        pair,
        multiplicities=ones,
        seed_multiplicities=seed_ones,
        prepared=prepared,
    )
    point: dict[str, tuple[FloatArray, FloatArray]] = {"raw": point_raw}
    if reference_pair is not None:
        reference_point = estimate_pair(
            reference_pair,
            multiplicities=ones,
            seed_multiplicities=seed_ones,
            prepared=reference_prepared,
        )
        point["retention"] = _retention(point_raw, reference_point)

    estimands = tuple(point)
    bootstrap = {
        name: np.full((bootstrap_resamples, 2, len(METRICS)), np.nan, dtype=np.float64)
        for name in estimands
    }
    permutation = {
        name: np.full((permutation_resamples, len(METRICS)), np.nan, dtype=np.float64)
        for name in estimands
    }
    rng = np.random.default_rng(stable_rng_seed(base_seed, comparison_label))
    for resample in range(bootstrap_resamples):
        counts, seed_counts = draw_hierarchical_bootstrap_multiplicities(
            rng,
            patient_clusters,
            seed_count=seed_count,
        )
        current = estimate_pair(
            pair,
            multiplicities=counts,
            seed_multiplicities=seed_counts,
            prepared=prepared,
        )
        bootstrap["raw"][resample] = current
        if reference_pair is not None:
            reference = estimate_pair(
                reference_pair,
                multiplicities=counts,
                seed_multiplicities=seed_counts,
                prepared=reference_prepared,
            )
            bootstrap["retention"][resample] = _retention(current, reference)

    for resample in range(permutation_resamples):
        patient_group_swaps = rng.integers(0, 2, size=patient_group_count, dtype=np.int8).astype(
            np.bool_
        )
        swap_mask = expand_patient_group_choices(patient_group_swaps, patient_clusters)
        current = estimate_pair(
            pair,
            multiplicities=ones,
            seed_multiplicities=seed_ones,
            swap_mask=swap_mask,
            prepared=prepared,
        )
        permutation["raw"][resample] = current[0] - current[1]
        if reference_pair is not None:
            reference = estimate_pair(
                reference_pair,
                multiplicities=ones,
                seed_multiplicities=seed_ones,
                swap_mask=swap_mask,
                prepared=reference_prepared,
            )
            retained = _retention(current, reference)
            permutation["retention"][resample] = retained[0] - retained[1]

    output: dict[str, list[dict[str, Any]]] = {}
    for name in estimands:
        rows: list[dict[str, Any]] = []
        estimate_a, estimate_b = point[name]
        observed = estimate_a - estimate_b
        for metric_index, metric in enumerate(METRICS):
            a_low, a_high, a_valid = _percentile_interval(
                bootstrap[name][:, 0, metric_index], confidence_level
            )
            b_low, b_high, b_valid = _percentile_interval(
                bootstrap[name][:, 1, metric_index], confidence_level
            )
            differences = bootstrap[name][:, 0, metric_index] - bootstrap[name][:, 1, metric_index]
            difference_low, difference_high, difference_valid = _percentile_interval(
                differences, confidence_level
            )
            null = permutation[name][:, metric_index]
            valid_null = null[np.isfinite(null)]
            if np.isfinite(observed[metric_index]) and len(valid_null):
                exceedances = np.count_nonzero(
                    np.abs(valid_null) >= abs(observed[metric_index]) - 1e-15
                )
                p_value = (exceedances + 1) / (len(valid_null) + 1)
            else:
                p_value = np.nan
            effect = float(observed[metric_index])
            estimable = all(
                np.isfinite(value)
                for value in (
                    estimate_a[metric_index],
                    estimate_b[metric_index],
                    observed[metric_index],
                    p_value,
                    effect,
                )
            )
            rows.append(
                {
                    "metric": metric,
                    "unit": METRIC_UNITS[metric],
                    "estimand": name,
                    "detector_a_estimate": float(estimate_a[metric_index]),
                    "detector_a_ci_low": a_low,
                    "detector_a_ci_high": a_high,
                    "detector_b_estimate": float(estimate_b[metric_index]),
                    "detector_b_ci_low": b_low,
                    "detector_b_ci_high": b_high,
                    "difference_a_minus_b": float(observed[metric_index]),
                    "difference_ci_low": difference_low,
                    "difference_ci_high": difference_high,
                    "bootstrap_valid_a": a_valid,
                    "bootstrap_valid_b": b_valid,
                    "bootstrap_valid_difference": difference_valid,
                    "p_value_raw": float(p_value),
                    "permutation_valid": len(valid_null),
                    "effect_size_name": "paired_raw_difference",
                    "effect_size": effect,
                    "effect_size_n": patient_group_count,
                    "status": "complete" if estimable else "not_estimable",
                    "reason": (
                        ""
                        if estimable
                        else "The conditional metric is undefined for at least one detector."
                    ),
                }
            )
        output[name] = rows
    return output


def analyze_training_procedure(
    runs: IndependentEvidenceRuns,
    *,
    patient_clusters: PatientClusters,
    base_seed: int,
    comparison_label: str,
    bootstrap_resamples: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Estimate training-procedure uncertainty from patients and independent runs.

    The held-out patient draw is shared across detectors because both pipelines
    were evaluated on the same test cohort. Trained runs are resampled in two
    separate multinomial draws because the detector training programs did not
    implement a common-random-number design.
    """

    if bootstrap_resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    if patient_clusters.image_ids != runs.detector_a[0].image_ids:
        raise ValueError("patient clusters must match the evidence image order")

    ones = np.ones(runs.image_count, dtype=np.int64)
    point_a, point_b = estimate_independent_runs(runs, multiplicities=ones)
    bootstrap = np.full((bootstrap_resamples, 2, len(METRICS)), np.nan, dtype=np.float64)
    rng = np.random.default_rng(stable_rng_seed(base_seed, comparison_label))
    for resample in range(bootstrap_resamples):
        patient_multiplicities = draw_patient_cluster_bootstrap_multiplicities(
            rng, patient_clusters
        )
        run_multiplicities_a, run_multiplicities_b = draw_independent_run_bootstrap_multiplicities(
            rng,
            detector_a_run_count=runs.detector_a_run_count,
            detector_b_run_count=runs.detector_b_run_count,
        )
        bootstrap[resample] = estimate_independent_runs(
            runs,
            multiplicities=patient_multiplicities,
            detector_a_run_multiplicities=run_multiplicities_a,
            detector_b_run_multiplicities=run_multiplicities_b,
        )

    rows: list[dict[str, Any]] = []
    observed = point_a - point_b
    for metric_index, metric in enumerate(METRICS):
        a_low, a_high, a_valid = _percentile_interval(
            bootstrap[:, 0, metric_index], confidence_level
        )
        b_low, b_high, b_valid = _percentile_interval(
            bootstrap[:, 1, metric_index], confidence_level
        )
        differences = bootstrap[:, 0, metric_index] - bootstrap[:, 1, metric_index]
        difference_low, difference_high, difference_valid = _percentile_interval(
            differences, confidence_level
        )
        values = (
            point_a[metric_index],
            point_b[metric_index],
            observed[metric_index],
            difference_low,
            difference_high,
        )
        estimable = all(np.isfinite(value) for value in values)
        rows.append(
            {
                "metric": metric,
                "unit": METRIC_UNITS[metric],
                "estimand": "training_procedure",
                "detector_a_estimate": float(point_a[metric_index]),
                "detector_a_ci_low": a_low,
                "detector_a_ci_high": a_high,
                "detector_b_estimate": float(point_b[metric_index]),
                "detector_b_ci_low": b_low,
                "detector_b_ci_high": b_high,
                "difference_a_minus_b": float(observed[metric_index]),
                "difference_ci_low": difference_low,
                "difference_ci_high": difference_high,
                "bootstrap_valid_a": a_valid,
                "bootstrap_valid_b": b_valid,
                "bootstrap_valid_difference": difference_valid,
                "effect_size_name": "training_procedure_raw_difference",
                "effect_size": float(observed[metric_index]),
                "effect_size_n": patient_clusters.patient_group_count,
                "detector_a_run_count": runs.detector_a_run_count,
                "detector_b_run_count": runs.detector_b_run_count,
                "status": "complete" if estimable else "not_estimable",
                "reason": (
                    ""
                    if estimable
                    else "The conditional metric is undefined for every eligible run."
                ),
            }
        )
    return rows


def json_number(value: Any) -> Any:
    """Replace non-finite numeric values with JSON-safe nulls recursively."""

    if isinstance(value, dict):
        return {key: json_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_number(item) for item in value]
    if isinstance(value, tuple):
        return [json_number(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value
