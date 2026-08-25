"""Detection-specific calibration analysis of frozen Phase 5 predictions.

This module independently implements the full five-dimensional Detection
Expected Calibration Error (D-ECE) from Küppers et al., *Multivariate
Confidence Calibration for Object Detection*, CVPR Workshops 2020,
https://doi.org/10.1109/CVPRW50498.2020.00171. It does not copy or import the
authors' ``netcal`` reference implementation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import load_phase5_config, sha256_file
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget, match_image
from src.stats.run_statistics import load_coco_targets

DetectorName = Literal["faster_rcnn", "yolo11s"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

FEATURE_NAMES = (
    "confidence",
    "relative_center_x",
    "relative_center_y",
    "relative_width",
    "relative_height",
)

SUMMARY_FIELDS = (
    "row_type",
    "detector",
    "seed",
    "seed_count",
    "prediction_count",
    "included_prediction_count",
    "included_prediction_fraction",
    "matched_true_positives",
    "false_positives",
    "mean_confidence",
    "fraction_true_positives",
    "global_calibration_gap",
    "maximum_confidence",
    "detection_ece",
    "detection_ece_std",
    "detection_ece_mean_plus_minus_std",
    "detection_ece_rank_within_detector",
    "sibling_median_detection_ece",
    "detection_ece_to_sibling_median_ratio",
    "nonempty_multivariate_bins",
    "included_multivariate_bins",
    "prediction_score_floor",
    "match_iou_threshold",
    "max_detections_per_image",
    "calibration_dimensions",
    "bins_per_dimension",
    "minimum_samples_per_bin",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen Phase 5 evidence used by the offline analysis."""

    phase5_config: Path
    phase5_summary: Path
    test_annotations: Path


class CalibrationSettings(StrictModel):
    """D-ECE matching, binning, and reliability settings."""

    prediction_score_floor: float = Field(gt=0, le=1)
    match_iou_threshold: float = Field(gt=0, le=1)
    max_detections_per_image: int = Field(ge=1)
    bins_per_dimension: tuple[int, int, int, int, int]
    minimum_samples_per_bin: int = Field(ge=1)
    reliability_bins: int = Field(ge=2, le=100)

    @model_validator(mode="after")
    def validate_bins(self) -> CalibrationSettings:
        if any(value < 2 for value in self.bins_per_dimension):
            raise ValueError("bins_per_dimension values must all be at least two")
        return self


class PlotSettings(StrictModel):
    """Deterministic reliability-figure settings."""

    dpi: int = Field(ge=72, le=600)


class OutputSettings(StrictModel):
    """Generated table, figure, and provenance paths."""

    log_dir: Path
    summary_json: Path
    summary_table: Path
    reliability_figure: Path


class CalibrationConfig(StrictModel):
    """Strict Batch 18 offline-calibration contract."""

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    calibration: CalibrationSettings
    plots: PlotSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True)
class DetectionCalibrationSamples:
    """Per-detection multivariate features and binary match outcomes."""

    features: FloatArray
    matched: BoolArray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        matched = np.asarray(self.matched, dtype=np.bool_)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (N, {len(FEATURE_NAMES)})")
        if matched.ndim != 1 or len(matched) != len(features):
            raise ValueError("matched must have shape (N,) aligned with features")
        if not len(features):
            raise ValueError("calibration requires at least one retained prediction")
        if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
            raise ValueError("all calibration features must be finite values in [0, 1]")
        features = features.copy()
        matched = matched.copy()
        features.setflags(write=False)
        matched.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "matched", matched)

    @property
    def confidence(self) -> FloatArray:
        """Return the confidence dimension."""

        return self.features[:, 0]


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    """Load and strictly validate the Batch 18 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return CalibrationConfig.model_validate(payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a gzip JSON object: {path}")
    return payload


def _deserialize_predictions(payload: Mapping[str, Any]) -> list[ImagePrediction]:
    records = payload.get("predictions")
    if not isinstance(records, list):
        raise ValueError("prediction bundle lacks a predictions list")
    predictions = [
        ImagePrediction(
            image_id=str(item["image_id"]),
            image_size=(int(item["image_size"][0]), int(item["image_size"][1])),
            boxes_xyxy=np.asarray(item["boxes_xyxy"], dtype=np.float64).reshape(-1, 4),
            labels=np.asarray(item["labels"], dtype=np.int64),
            scores=np.asarray(item["scores"], dtype=np.float64),
        )
        for item in records
    ]
    predictions.sort(key=lambda item: item.image_id)
    return predictions


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, encoded)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_detection_calibration_samples(
    predictions: Sequence[ImagePrediction],
    targets: Sequence[ImageTarget],
    *,
    score_floor: float,
    iou_threshold: float,
    max_detections: int,
) -> DetectionCalibrationSamples:
    """Match variable-length detections and encode confidence/location/scale.

    A retained prediction is correct exactly when the canonical, score-ordered,
    same-class matcher assigns it to one as-yet unmatched target at the configured
    IoU. Unmatched predictions are false positives. Missed targets have no score
    and therefore are outside black-box precision calibration, as in Küppers et al.
    """

    prediction_map = {item.image_id: item for item in predictions}
    target_map = {item.image_id: item for item in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("prediction and target image IDs must be unique")
    if set(prediction_map) != set(target_map):
        raise ValueError("prediction and target image ID sets must match")

    feature_rows: list[list[float]] = []
    matched_rows: list[bool] = []
    for image_id in sorted(target_map):
        prediction = prediction_map[image_id]
        target = target_map[image_id]
        result = match_image(
            prediction,
            target,
            score_threshold=score_floor,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        matched_indices = {match.prediction_index for match in result.matches}
        image_height, image_width = prediction.image_size
        for index in result.prediction_indices:
            x1, y1, x2, y2 = prediction.boxes_xyxy[index]
            feature_rows.append(
                [
                    float(prediction.scores[index]),
                    float((x1 + x2) / (2 * image_width)),
                    float((y1 + y2) / (2 * image_height)),
                    float((x2 - x1) / image_width),
                    float((y2 - y1) / image_height),
                ]
            )
            matched_rows.append(index in matched_indices)

    return DetectionCalibrationSamples(
        features=np.asarray(feature_rows, dtype=np.float64),
        matched=np.asarray(matched_rows, dtype=np.bool_),
    )


def detection_expected_calibration_error(
    features: FloatArray,
    matched: BoolArray,
    *,
    bins: Sequence[int],
    minimum_samples_per_bin: int,
) -> dict[str, Any]:
    """Compute full multivariate D-ECE using paper-equivalent equal-width bins.

    The first feature must be confidence and remaining features are relative box
    properties. In every populated five-dimensional cell, empirical precision
    is the fraction of matched detections. The absolute precision-confidence gap
    is weighted by the cell's share of *all* detections. Cells below the sample
    threshold contribute zero, matching the robustness rule used in the paper;
    their sample count is reported explicitly rather than silently discarded.
    """

    features = np.asarray(features, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.bool_)
    bin_counts = tuple(int(value) for value in bins)
    if features.ndim != 2 or features.shape[1] != len(bin_counts):
        raise ValueError("bins must provide one count for each feature dimension")
    if matched.ndim != 1 or len(matched) != len(features) or not len(features):
        raise ValueError("matched must be a non-empty vector aligned with features")
    if any(value < 2 for value in bin_counts):
        raise ValueError("each bin count must be at least two")
    if minimum_samples_per_bin < 1:
        raise ValueError("minimum_samples_per_bin must be positive")
    if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
        raise ValueError("features must be finite and lie in [0, 1]")

    indices = np.column_stack(
        [
            np.minimum((features[:, dimension] * count).astype(np.int64), count - 1)
            for dimension, count in enumerate(bin_counts)
        ]
    )
    flat_indices = np.ravel_multi_index(indices.T, bin_counts)
    unique_bins, counts = np.unique(flat_indices, return_counts=True)
    detection_ece = 0.0
    included_samples = 0
    included_bins = 0
    bin_records: list[dict[str, Any]] = []
    for flat_index, count in zip(unique_bins, counts, strict=True):
        mask = flat_indices == flat_index
        mean_confidence = float(np.mean(features[mask, 0]))
        empirical_precision = float(np.mean(matched[mask]))
        gap = abs(empirical_precision - mean_confidence)
        included = int(count) >= minimum_samples_per_bin
        contribution = float(count / len(features) * gap) if included else 0.0
        if included:
            detection_ece += contribution
            included_samples += int(count)
            included_bins += 1
        bin_records.append(
            {
                "indices": [int(index) for index in np.unravel_index(int(flat_index), bin_counts)],
                "sample_count": int(count),
                "mean_confidence": mean_confidence,
                "fraction_true_positives": empirical_precision,
                "absolute_gap": gap,
                "included": included,
                "weighted_contribution": contribution,
            }
        )

    if included_bins == 0:
        raise ValueError("no multivariate bin meets minimum_samples_per_bin")
    return {
        "detection_ece": float(detection_ece),
        "prediction_count": len(features),
        "included_prediction_count": included_samples,
        "included_prediction_fraction": float(included_samples / len(features)),
        "nonempty_multivariate_bins": len(unique_bins),
        "included_multivariate_bins": included_bins,
        "bins": bin_records,
    }


def reliability_bins(
    confidence: FloatArray, matched: BoolArray, *, bins: int
) -> list[dict[str, Any]]:
    """Return equal-width confidence reliability points for emitted detections."""

    confidence = np.asarray(confidence, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.bool_)
    if confidence.ndim != 1 or matched.ndim != 1 or len(confidence) != len(matched):
        raise ValueError("confidence and matched must be aligned one-dimensional arrays")
    if not len(confidence) or bins < 2:
        raise ValueError("reliability binning requires samples and at least two bins")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and lie in [0, 1]")

    indices = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(bins):
        mask = indices == bin_index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_index": bin_index,
                "lower_bound": float(bin_index / bins),
                "upper_bound": float((bin_index + 1) / bins),
                "sample_count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(confidence[mask])),
                "fraction_true_positives": float(np.mean(matched[mask])),
            }
        )
    return rows


def _validate_upstream(
    config: CalibrationConfig,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[ImageTarget]]:
    phase5_path = config.resolve(config.inputs.phase5_config)
    summary_path = config.resolve(config.inputs.phase5_summary)
    annotation_path = config.resolve(config.inputs.test_annotations)
    phase5 = load_phase5_config(phase5_path)
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Phase 5 summary is not complete")
    if summary.get("config_sha256") != sha256_file(phase5_path):
        raise ValueError("Phase 5 config hash differs from its frozen summary")
    annotation_sha256 = sha256_file(annotation_path)
    if summary.get("test_annotation_sha256") != annotation_sha256:
        raise ValueError("test annotations differ from the Phase 5 evaluation source")
    if summary.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("Phase 5 evaluator settings differ from its frozen summary")

    settings = config.calibration
    if not np.isclose(settings.prediction_score_floor, phase5.evaluation.coco_minimum_score):
        raise ValueError("prediction_score_floor must equal the frozen bundle minimum score")
    if not np.isclose(settings.match_iou_threshold, phase5.evaluation.match_iou_threshold):
        raise ValueError("match_iou_threshold must equal the canonical Phase 5 matcher")
    if settings.max_detections_per_image != phase5.evaluation.max_detections:
        raise ValueError("max_detections_per_image must equal the Phase 5 cap")

    targets, _ = load_coco_targets(annotation_path)
    expected = {
        (detector, seed) for detector in ("faster_rcnn", "yolo11s") for seed in phase5.seeds
    }
    bundles: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for run in summary.get("runs", []):
        detector = str(run.get("detector"))
        seed = int(run.get("seed"))
        key = detector, seed
        if key in seen:
            raise ValueError(f"duplicate Phase 5 bundle identity: {key}")
        seen.add(key)
        comparison = run.get("comparison_row")
        if not isinstance(comparison, dict):
            raise ValueError(f"Phase 5 run lacks a comparison row: {key}")
        bundle_path = Path(str(comparison["prediction_bundle"]))
        if not bundle_path.is_absolute():
            bundle_path = config.resolve(bundle_path)
        expected_hash = str(run.get("prediction_bundle_sha256"))
        if sha256_file(bundle_path) != expected_hash:
            raise ValueError(f"Phase 5 bundle hash mismatch: {bundle_path}")
        payload = _read_bundle(bundle_path)
        if (
            payload.get("schema_version") != 1
            or payload.get("detector") != detector
            or int(payload.get("seed")) != seed
            or payload.get("split") != phase5.split
        ):
            raise ValueError(f"Phase 5 bundle identity mismatch: {bundle_path}")
        if payload.get("annotation_sha256") != annotation_sha256:
            raise ValueError(f"Phase 5 bundle annotation mismatch: {bundle_path}")
        if payload.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
            raise ValueError(f"Phase 5 bundle evaluator mismatch: {bundle_path}")
        predictions = _deserialize_predictions(payload)
        if len(predictions) != len(targets):
            raise ValueError(f"Phase 5 bundle image count mismatch: {bundle_path}")
        bundles.append(
            {
                "detector": detector,
                "seed": seed,
                "path": bundle_path,
                "sha256": expected_hash,
                "predictions": predictions,
            }
        )
    if seen != expected:
        raise ValueError("Phase 5 summary does not contain the complete detector/seed grid")
    if tuple(phase5.seeds) != (17, 42, 137, 271, 314):
        raise ValueError("Batch 18 requires the frozen five-seed Phase 5 grid")
    bundles.sort(key=lambda item: (str(item["detector"]), int(item["seed"])))
    return phase5, summary, bundles, targets


def _seed_row(
    *,
    detector: str,
    seed: int,
    samples: DetectionCalibrationSamples,
    result: Mapping[str, Any],
    settings: CalibrationSettings,
) -> dict[str, Any]:
    confidence = samples.confidence
    fraction_true_positives = float(np.mean(samples.matched))
    mean_confidence = float(np.mean(confidence))
    return {
        "row_type": "seed",
        "detector": detector,
        "seed": seed,
        "seed_count": 1,
        "prediction_count": len(samples.matched),
        "included_prediction_count": result["included_prediction_count"],
        "included_prediction_fraction": result["included_prediction_fraction"],
        "matched_true_positives": int(np.sum(samples.matched)),
        "false_positives": int(np.sum(~samples.matched)),
        "mean_confidence": mean_confidence,
        "fraction_true_positives": fraction_true_positives,
        "global_calibration_gap": abs(fraction_true_positives - mean_confidence),
        "maximum_confidence": float(np.max(confidence)),
        "detection_ece": result["detection_ece"],
        "nonempty_multivariate_bins": result["nonempty_multivariate_bins"],
        "included_multivariate_bins": result["included_multivariate_bins"],
        "prediction_score_floor": settings.prediction_score_floor,
        "match_iou_threshold": settings.match_iou_threshold,
        "max_detections_per_image": settings.max_detections_per_image,
        "calibration_dimensions": "|".join(FEATURE_NAMES),
        "bins_per_dimension": "|".join(str(value) for value in settings.bins_per_dimension),
        "minimum_samples_per_bin": settings.minimum_samples_per_bin,
    }


def _add_detector_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    for detector in ("faster_rcnn", "yolo11s"):
        detector_rows = [row for row in rows if row["detector"] == detector]
        values = np.asarray([row["detection_ece"] for row in detector_rows], dtype=np.float64)
        order = np.argsort(-values, kind="stable")
        for rank, index in enumerate(order, start=1):
            row = detector_rows[int(index)]
            sibling_values = np.delete(values, int(index))
            sibling_median = float(np.median(sibling_values))
            row["detection_ece_rank_within_detector"] = rank
            row["sibling_median_detection_ece"] = sibling_median
            row["detection_ece_to_sibling_median_ratio"] = float(
                row["detection_ece"] / sibling_median
            )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        common = detector_rows[0]
        output.append(
            {
                "row_type": "detector_mean",
                "detector": detector,
                "seed_count": len(detector_rows),
                "detection_ece": mean,
                "detection_ece_std": std,
                "detection_ece_mean_plus_minus_std": f"{mean:.6f} +/- {std:.6f}",
                "prediction_score_floor": common["prediction_score_floor"],
                "match_iou_threshold": common["match_iou_threshold"],
                "max_detections_per_image": common["max_detections_per_image"],
                "calibration_dimensions": common["calibration_dimensions"],
                "bins_per_dimension": common["bins_per_dimension"],
                "minimum_samples_per_bin": common["minimum_samples_per_bin"],
            }
        )
    return output


def _atomic_reliability_figure(
    path: Path,
    reliability: Mapping[tuple[str, int], list[dict[str, Any]]],
    pooled: Mapping[str, list[dict[str, Any]]],
    *,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    colors = {"faster_rcnn": "#2f6f9f", "yolo11s": "#d97706"}
    titles = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for axis, detector in zip(axes, ("faster_rcnn", "yolo11s"), strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.2, label="Ideal")
        for (row_detector, seed), points in reliability.items():
            if row_detector != detector:
                continue
            x = [point["mean_confidence"] for point in points]
            y = [point["fraction_true_positives"] for point in points]
            is_pathology = detector == "yolo11s" and seed == 271
            axis.plot(
                x,
                y,
                marker="o",
                markersize=5 if is_pathology else 3,
                linewidth=2.0 if is_pathology else 0.9,
                linestyle="--" if is_pathology else "-",
                color="#b91c1c" if is_pathology else colors[detector],
                alpha=1.0 if is_pathology else 0.30,
                label="Seed 271" if is_pathology else None,
            )
        pooled_points = pooled[detector]
        axis.plot(
            [point["mean_confidence"] for point in pooled_points],
            [point["fraction_true_positives"] for point in pooled_points],
            marker="o",
            markersize=5,
            linewidth=2.4,
            color=colors[detector],
            label="Pooled detections",
        )
        axis.set_title(titles[detector])
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.20)
        axis.legend(loc="lower right")
    axes[0].set_ylabel("Fraction of true positives")

    pathology = reliability[("yolo11s", 271)][0]
    axes[1].annotate(
        f"Seed 271: {pathology['mean_confidence']:.3f} confidence, "
        f"{pathology['fraction_true_positives']:.3f} TP fraction",
        xy=(pathology["mean_confidence"], pathology["fraction_true_positives"]),
        xytext=(0.19, 0.31),
        arrowprops={"arrowstyle": "->", "color": "#b91c1c"},
        color="#7f1d1d",
        fontsize=9,
    )
    figure.suptitle("Detection reliability at IoU >= 0.50 (all predictions with score >= 0.001)")
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", metadata={"Software": "matplotlib"})
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return path


def preflight(config: CalibrationConfig) -> dict[str, Any]:
    """Validate all immutable evidence without generating result artifacts."""

    phase5, summary, bundles, targets = _validate_upstream(config)
    return {
        "status": "preflight_passed",
        "experiment_id": config.experiment_id,
        "detectors": sorted({str(bundle["detector"]) for bundle in bundles}),
        "seeds": list(phase5.seeds),
        "bundle_count": len(bundles),
        "image_count": len(targets),
        "annotation_count": int(summary["annotation_count"]),
        "prediction_score_floor": config.calibration.prediction_score_floor,
        "match_iou_threshold": config.calibration.match_iou_threshold,
    }


def run_calibration(config: CalibrationConfig) -> dict[str, Any]:
    """Compute five-seed D-ECE, reliability diagrams, and provenance."""

    phase5, phase5_summary, bundles, targets = _validate_upstream(config)
    settings = config.calibration
    rows: list[dict[str, Any]] = []
    samples_by_run: dict[tuple[str, int], DetectionCalibrationSamples] = {}
    reliability: dict[tuple[str, int], list[dict[str, Any]]] = {}
    multivariate_bins: dict[str, Any] = {}
    for bundle in bundles:
        detector = str(bundle["detector"])
        seed = int(bundle["seed"])
        samples = build_detection_calibration_samples(
            bundle["predictions"],
            targets,
            score_floor=settings.prediction_score_floor,
            iou_threshold=settings.match_iou_threshold,
            max_detections=settings.max_detections_per_image,
        )
        result = detection_expected_calibration_error(
            samples.features,
            samples.matched,
            bins=settings.bins_per_dimension,
            minimum_samples_per_bin=settings.minimum_samples_per_bin,
        )
        rows.append(
            _seed_row(
                detector=detector,
                seed=seed,
                samples=samples,
                result=result,
                settings=settings,
            )
        )
        key = detector, seed
        samples_by_run[key] = samples
        reliability[key] = reliability_bins(
            samples.confidence, samples.matched, bins=settings.reliability_bins
        )
        multivariate_bins[f"{detector}_seed{seed}"] = result["bins"]

    output_rows = _add_detector_summaries(rows)
    pooled_reliability: dict[str, list[dict[str, Any]]] = {}
    for detector in ("faster_rcnn", "yolo11s"):
        detector_samples = [
            samples for (name, _), samples in samples_by_run.items() if name == detector
        ]
        pooled_reliability[detector] = reliability_bins(
            np.concatenate([samples.confidence for samples in detector_samples]),
            np.concatenate([samples.matched for samples in detector_samples]),
            bins=settings.reliability_bins,
        )

    table_path = _atomic_csv(config.resolve(config.outputs.summary_table), output_rows)
    figure_path = _atomic_reliability_figure(
        config.resolve(config.outputs.reliability_figure),
        reliability,
        pooled_reliability,
        dpi=config.plots.dpi,
    )
    seed271 = next(row for row in rows if row["detector"] == "yolo11s" and row["seed"] == 271)
    detector_means = {
        row["detector"]: {
            "mean_detection_ece": row["detection_ece"],
            "sample_std_detection_ece": row["detection_ece_std"],
        }
        for row in output_rows
        if row["row_type"] == "detector_mean"
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "method": {
            "name": "full_multivariate_detection_expected_calibration_error",
            "paper": (
                "Kuppers et al., Multivariate Confidence Calibration for Object Detection, "
                "CVPR Workshops 2020"
            ),
            "doi": "10.1109/CVPRW50498.2020.00171",
            "dimensions": list(FEATURE_NAMES),
            "bins_per_dimension": list(settings.bins_per_dimension),
            "minimum_samples_per_bin": settings.minimum_samples_per_bin,
            "reliability_bins": settings.reliability_bins,
            "prediction_score_floor": settings.prediction_score_floor,
            "match_iou_threshold": settings.match_iou_threshold,
            "max_detections_per_image": settings.max_detections_per_image,
            "missed_targets_included": False,
            "reason_missed_targets_excluded": (
                "Black-box precision calibration requires an emitted score."
            ),
        },
        "implementation": {
            "reference_framework_license": "Apache-2.0",
            "repository_license": "AGPL-3.0-only",
            "direct_netcal_dependency": False,
            "implementation_note": (
                "Independent NumPy implementation of the paper's D-ECE definition; "
                "no netcal source copied."
            ),
        },
        "source": {
            "config_path": config.source_path.relative_to(config.project_root).as_posix(),
            "config_sha256": sha256_file(config.source_path),
            "analysis_source": Path(__file__).resolve().relative_to(config.project_root).as_posix(),
            "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
            "phase5_config": config.resolve(config.inputs.phase5_config)
            .relative_to(config.project_root)
            .as_posix(),
            "phase5_config_sha256": phase5_summary["config_sha256"],
            "phase5_summary": config.resolve(config.inputs.phase5_summary)
            .relative_to(config.project_root)
            .as_posix(),
            "phase5_summary_sha256": sha256_file(config.resolve(config.inputs.phase5_summary)),
            "test_annotations": config.resolve(config.inputs.test_annotations)
            .relative_to(config.project_root)
            .as_posix(),
            "test_annotation_sha256": phase5_summary["test_annotation_sha256"],
            "prediction_bundles": [
                {
                    "detector": bundle["detector"],
                    "seed": bundle["seed"],
                    "path": bundle["path"].relative_to(config.project_root).as_posix(),
                    "sha256": bundle["sha256"],
                }
                for bundle in bundles
            ],
        },
        "detector_summaries": detector_means,
        "seed_rows": rows,
        "seed_271_pathology": {
            "retained": True,
            "detection_ece": seed271["detection_ece"],
            "rank_within_yolo11s": seed271["detection_ece_rank_within_detector"],
            "mean_confidence": seed271["mean_confidence"],
            "fraction_true_positives": seed271["fraction_true_positives"],
            "global_calibration_gap": seed271["global_calibration_gap"],
            "maximum_confidence": seed271["maximum_confidence"],
            "detection_ece_to_sibling_median_ratio": seed271[
                "detection_ece_to_sibling_median_ratio"
            ],
        },
        "reliability_bins": {
            f"{detector}_seed{seed}": points for (detector, seed), points in reliability.items()
        },
        "pooled_reliability_bins": pooled_reliability,
        "multivariate_bins": multivariate_bins,
        "artifacts": {},
        "seeds": list(phase5.seeds),
    }
    summary_path = config.resolve(config.outputs.summary_json)
    summary["artifacts"] = {
        table_path.relative_to(config.project_root).as_posix(): sha256_file(table_path),
        figure_path.relative_to(config.project_root).as_posix(): sha256_file(figure_path),
    }
    _atomic_json(summary_path, summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run calibration preflight or the complete offline analysis."""

    args = _parse_args(argv)
    config = load_calibration_config(args.config)
    result = preflight(config) if args.mode == "preflight" else run_calibration(config)
    if args.mode == "preflight":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(config.resolve(config.outputs.summary_json))


if __name__ == "__main__":
    main()
