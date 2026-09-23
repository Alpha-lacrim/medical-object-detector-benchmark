"""Metric-preserving resampling of frozen detection runs, without inference.

This extends Phase 8's observation/run resampling principles to exact FROC.
COCO matches are cached from pycocotools itself. Integer observation weights
represent distinct image copies; tied detections retain image/copy order.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget, match_image
from src.stats.paired import (
    PatientClusters,
    _operating_ratios,
    _percentile_interval,
    draw_independent_run_bootstrap_multiplicities,
    draw_patient_cluster_bootstrap_multiplicities,
    stable_rng_seed,
)

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def validate_weights(weights: IntArray, size: int) -> None:
    """Reject invalid multiplicities rather than silently coercing them."""
    if (
        weights.shape != (size,)
        or not np.issubdtype(weights.dtype, np.integer)
        or np.any(weights < 0)
        or not np.any(weights)
    ):
        raise ValueError("Observation weights must be nonnegative integers with positive total")


def aligned_records(
    predictions: list[ImagePrediction], targets: list[ImageTarget]
) -> tuple[list[ImagePrediction], list[ImageTarget]]:
    """Align all runs to unique, sorted observation identifiers."""
    pm, tm = {p.image_id: p for p in predictions}, {t.image_id: t for t in targets}
    if len(pm) != len(predictions) or len(tm) != len(targets) or set(pm) != set(tm):
        raise ValueError("Prediction/target IDs must be unique and identical")
    keys = sorted(tm)
    if any(pm[k].image_size != tm[k].image_size for k in keys):
        raise ValueError("Prediction/target dimensions differ")
    return [pm[k] for k in keys], [tm[k] for k in keys]


@dataclass
class APCache:
    """Sparse true-positive ranks with exact COCO per-image tie semantics."""

    image_ids: tuple[str, ...]
    target_counts: IntArray
    group_images: IntArray
    group_lengths: IntArray
    # Per IoU: groups with TP, TP counts/group, and ordered offsets within group.
    positives: tuple[tuple[IntArray, IntArray, IntArray], ...]
    recall_grid: FloatArray

    def evaluate(self, weights: IntArray) -> FloatArray:
        """Reconstruct AP as if every sampled image had a distinct COCO key."""
        validate_weights(weights, len(self.image_ids))
        targets = int(weights @ self.target_counts)
        if not targets:
            return np.full(len(self.positives), np.nan)
        group_weights = weights[self.group_images]
        lengths = self.group_lengths * group_weights
        preceding = np.cumsum(lengths) - lengths
        aps = []
        for groups, counts, offsets in self.positives:
            copies = group_weights[groups]
            repeats = copies * counts
            blocks = np.repeat(np.arange(len(groups)), repeats)
            if not len(blocks):
                aps.append(0.0)
                continue
            local = np.arange(len(blocks)) - np.repeat(np.cumsum(repeats) - repeats, repeats)
            offset_index = (np.cumsum(counts) - counts)[blocks] + local % counts[blocks]
            group = groups[blocks]
            ranks = (
                preceding[group]
                + (local // counts[blocks]) * self.group_lengths[group]
                + offsets[offset_index]
                + 1
            )
            tp = np.arange(1, len(ranks) + 1, dtype=np.float64)
            precision = tp / (ranks + np.spacing(1))
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            positions = np.searchsorted(tp / targets, self.recall_grid, side="left")
            sampled = np.zeros(len(self.recall_grid))
            valid = positions < len(envelope)
            sampled[valid] = envelope[positions[valid]]
            aps.append(float(sampled.mean()))
        return np.asarray(aps)


def build_ap_cache(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    class_ids: tuple[int, ...],
    minimum_score: float,
    max_dets: list[int],
    iou_thresholds: list[float],
    recall_points: int,
) -> APCache:
    """Cache official COCO matches for the frozen single-category, all-area AP."""
    predictions, targets = aligned_records(predictions, targets)
    if len(class_ids) != 1:
        raise ValueError("This analysis requires the frozen single foreground category")
    allowed = set(class_ids)
    if any(not set(r.labels).issubset(allowed) for r in [*predictions, *targets]):
        raise ValueError("Label outside configured category")
    images, annotations, detections = [], [], []
    for i, (p, t) in enumerate(zip(predictions, targets, strict=True)):
        images.append({"id": i + 1, "height": t.image_size[0], "width": t.image_size[1]})
        for box in t.boxes_xyxy:
            x, y, right, bottom = map(float, box)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i + 1,
                    "category_id": class_ids[0],
                    "bbox": [x, y, right - x, bottom - y],
                    "area": (right - x) * (bottom - y),
                    "iscrowd": 0,
                }
            )
        for box, score in zip(p.boxes_xyxy, p.scores, strict=True):
            if score >= minimum_score:
                x, y, right, bottom = map(float, box)
                detections.append(
                    {
                        "image_id": i + 1,
                        "category_id": class_ids[0],
                        "bbox": [x, y, right - x, bottom - y],
                        "score": float(score),
                    }
                )
    categories = [{"id": c, "name": str(c)} for c in class_ids]
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with redirect_stdout(StringIO()):
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {"images": images, "annotations": [], "categories": categories}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = [im["id"] for im in images]
        ev.params.catIds = list(class_ids)
        ev.params.maxDets = max_dets
        ev.params.iouThrs = np.asarray(iou_thresholds)
        ev.params.recThrs = np.linspace(0, 1, recall_points)
        ev.params.areaRng = ev.params.areaRng[:1]
        ev.params.areaRngLbl = ev.params.areaRngLbl[:1]
        ev.evaluate()
    scores, owners, matches = [], [], []
    for i, entry in enumerate(ev.evalImgs):
        if entry is None:
            continue
        if np.any(entry["dtIgnore"]) or np.any(entry["gtIgnore"]):
            raise ValueError("Unexpected ignored COCO annotation/detection in all-area endpoint")
        scores.extend(entry["dtScores"])
        owners.extend([i] * len(entry["dtScores"]))
        matches.append(entry["dtMatches"] > 0)
    scores_array, owners_array = np.asarray(scores), np.asarray(owners, dtype=np.int64)
    order = np.argsort(-scores_array, kind="stable")
    scores_array, owners_array = scores_array[order], owners_array[order]
    match_array = (
        np.concatenate(matches, axis=1)
        if matches
        else np.zeros((len(iou_thresholds), 0), dtype=bool)
    )[:, order]
    starts = (
        np.flatnonzero(np.r_[True, (np.diff(scores_array) != 0) | (np.diff(owners_array) != 0)])
        if len(order)
        else np.array([], int)
    )
    lengths = np.diff(np.r_[starts, len(order)])
    positives = []
    for row in match_array:
        positions = np.flatnonzero(row)
        groups = np.searchsorted(starts, positions, side="right") - 1
        unique, counts = np.unique(groups, return_counts=True)
        positives.append((unique, counts, positions - starts[groups]))
    return APCache(
        tuple(t.image_id for t in targets),
        np.asarray([len(t.labels) for t in targets], dtype=np.int64),
        owners_array[starts],
        lengths,
        tuple(positives),
        ev.params.recThrs,
    )


@dataclass
class FrocCache:
    """Greedy IoU matches ordered by exact score; score ties enter together."""

    image_ids: tuple[str, ...]
    target_counts: IntArray
    scores: FloatArray
    images: IntArray
    matched: NDArray[np.bool_]
    score_ends: IntArray

    def curve(self, weights: IntArray) -> tuple[FloatArray, FloatArray]:
        """Return a noninterpolated FP/image and sensitivity frontier plus origin."""
        validate_weights(weights, len(self.image_ids))
        w = weights[self.images]
        tp = np.r_[0, np.cumsum(w * self.matched)[self.score_ends]]
        fp = np.r_[0, np.cumsum(w * ~self.matched)[self.score_ends]]
        target_count = int(weights @ self.target_counts)
        sensitivity = tp / target_count if target_count else np.full(len(tp), np.nan)
        return fp / weights.sum(), sensitivity

    def budgets(
        self, weights: IntArray, budgets: list[float], tolerance: float
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Return maximal observed sensitivities at fixed FP budgets and support flags."""
        fp, sensitivity = self.curve(weights)
        indices = np.searchsorted(fp, np.asarray(budgets) + tolerance, side="right") - 1
        return sensitivity[indices], fp[-1] < np.asarray(budgets) - tolerance


def build_froc_cache(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    class_ids: tuple[int, ...],
    floor: float,
    iou: float,
    cap: int,
) -> FrocCache:
    """Reuse the canonical one-to-one matcher; preserve negatives and image caps."""
    predictions, targets = aligned_records(predictions, targets)
    scores, owners, matched = [], [], []
    for i, (p, t) in enumerate(zip(predictions, targets, strict=True)):
        if not set(p.labels).issubset(class_ids) or not set(t.labels).issubset(class_ids):
            raise ValueError("Label outside configured category")
        if np.any(p.scores < floor):
            raise ValueError("Retained score below collection floor")
        result = match_image(p, t, score_threshold=floor, iou_threshold=iou, max_detections=cap)
        hits = {m.prediction_index for m in result.matches}
        for j in result.prediction_indices:
            scores.append(p.scores[j])
            owners.append(i)
            matched.append(j in hits)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s = s[order]
    ends = np.flatnonzero(np.r_[np.diff(s) != 0, True]) if len(s) else np.array([], int)
    return FrocCache(
        tuple(t.image_id for t in targets),
        np.asarray([len(t.labels) for t in targets], dtype=np.int64),
        s,
        np.asarray(owners, dtype=np.int64)[order],
        np.asarray(matched, dtype=bool)[order],
        ends,
    )


def operating_counts(cache: FrocCache, threshold: float) -> IntArray:
    """Cache TP/FP per observation at an already frozen threshold."""
    keep = cache.scores >= threshold
    tp = np.bincount(cache.images[keep & cache.matched], minlength=len(cache.image_ids))
    fp = np.bincount(cache.images[keep & ~cache.matched], minlength=len(cache.image_ids))
    return np.column_stack([tp, fp]).astype(np.int64)


@dataclass
class RunEvidence:
    """Separate AP, lower-floor FROC and historical-threshold evidence per run."""

    seed: int
    ap: APCache
    froc: FrocCache
    primary_counts: IntArray

    def evaluate(self, weights: IntArray, budgets: list[float], tolerance: float) -> FloatArray:
        """Recompute cohort metrics before any averaging over trained runs."""
        aps = self.ap.evaluate(weights)
        sensitivities, _ = self.froc.budgets(weights, budgets, tolerance)
        tp, fp = weights @ self.primary_counts
        targets = weights @ self.ap.target_counts
        precision, recall, f1 = _operating_ratios(int(tp), int(fp), int(targets - tp))
        return np.r_[aps[0], aps.mean(), sensitivities, precision, recall, f1, fp / weights.sum()]


def draw_observations(
    rng: np.random.Generator, image_count: int, clusters: PatientClusters | None
) -> IntArray:
    """Use known RSNA patients; otherwise use released images without inventing patients."""
    if clusters is not None:
        return draw_patient_cluster_bootstrap_multiplicities(rng, clusters)
    return rng.multinomial(image_count, np.full(image_count, 1 / image_count))


def bootstrap(
    detector_runs: tuple[tuple[RunEvidence, ...], tuple[RunEvidence, ...]],
    *,
    clusters: PatientClusters | None,
    settings: dict[str, Any],
    label: str,
    budgets: list[float],
    tolerance: float,
    conditional_seed: int,
    progress_every: int,
) -> dict[str, FloatArray]:
    """Shared observation draw, independent detector-run draws, and fixed-model sensitivity."""
    a, b = detector_runs
    if not a or not b:
        raise ValueError("Both detectors require retained runs")
    ids = a[0].ap.image_ids
    for run in (*a, *b):
        if run.ap.image_ids != ids or run.froc.image_ids != ids:
            raise ValueError("Observation order differs between runs")
        if not np.array_equal(run.ap.target_counts, a[0].ap.target_counts):
            raise ValueError("Target population differs between runs")
    if clusters is not None and clusters.image_ids != ids:
        raise ValueError("Patient clusters do not match evidence")
    conditional_indices = [[r.seed for r in runs].index(conditional_seed) for runs in (a, b)]
    metric_count = len(budgets) + 6
    shape = (settings["bootstrap_resamples"], 2, metric_count)
    training = np.full(shape, np.nan)
    conditional = np.full(shape, np.nan)
    rng = np.random.default_rng(stable_rng_seed(settings["bootstrap_seed"], label))
    for draw in range(shape[0]):
        observations = draw_observations(rng, len(ids), clusters)
        run_weights = draw_independent_run_bootstrap_multiplicities(
            rng, detector_a_run_count=len(a), detector_b_run_count=len(b)
        )
        for d, runs in enumerate((a, b)):
            metrics = np.asarray([r.evaluate(observations, budgets, tolerance) for r in runs])
            selected = run_weights[d] > 0
            # Undefined selected values propagate; there is no favorable-run deletion.
            training[draw, d] = np.average(
                metrics[selected], axis=0, weights=run_weights[d][selected]
            )
            conditional[draw, d] = metrics[conditional_indices[d]]
        if progress_every and (draw + 1) % progress_every == 0:
            print(f"{label}: bootstrap {draw + 1}/{shape[0]}", flush=True)
    return {"training_procedure": training, "checkpoint_conditional_seed17": conditional}


def interval_rows(
    point: FloatArray,
    samples: FloatArray,
    names: list[str],
    confidence: float,
) -> list[dict[str, Any]]:
    """Marginal percentile intervals and explicit valid/undefined replicate counts."""
    rows = []
    for j, metric in enumerate(names):
        row: dict[str, Any] = {"metric": metric, "bootstrap_resamples": len(samples)}
        for name, values, estimate in (
            ("a", samples[:, 0, j], point[0, j]),
            ("b", samples[:, 1, j], point[1, j]),
            ("difference", samples[:, 0, j] - samples[:, 1, j], point[0, j] - point[1, j]),
        ):
            low, high, valid = _percentile_interval(values, confidence)
            row.update(
                {
                    f"{name}_estimate": float(estimate),
                    f"{name}_ci_low": low,
                    f"{name}_ci_high": high,
                    f"{name}_valid": valid,
                    f"{name}_undefined": len(samples) - valid,
                }
            )
        row["reason"] = (
            ""
            if row["difference_valid"] == len(samples)
            else ("Undefined sampled endpoint; no retry or imputation; interval uses finite draws")
        )
        rows.append(row)
    return rows


def classify_change(before: float, after: float, tolerance: float) -> dict[str, str]:
    """Describe signed A-minus-B gap changes, without hypothesis tests or preference claims."""
    if not np.isfinite(before) or not np.isfinite(after):
        return {"ordering_change": "undefined", "gap_change": "undefined"}
    s0 = 0 if abs(before) <= tolerance else np.sign(before)
    s1 = 0 if abs(after) <= tolerance else np.sign(after)
    ordering = "reversed" if s0 * s1 < 0 else "unchanged" if s0 == s1 else "tie_changed"
    if ordering == "reversed":
        gap = "reversed"
    elif abs(after - before) <= tolerance:
        gap = "unchanged"
    else:
        gap = "strengthened" if abs(after) > abs(before) else "weakened"
    return {"ordering_change": ordering, "gap_change": gap}
