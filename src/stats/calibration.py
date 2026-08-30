"""Detection-level calibration and support diagnostics for frozen predictions.

The primary endpoint is the five-numeric-dimension Detection Expected
Calibration Error (D-ECE) described by Kuppers et al., *Multivariate Confidence
Calibration for Object Detection*, CVPR Workshops 2020. Classes are categorical
strata, as required by the class-conditional definition, rather than an ordered
numeric histogram axis. This module is an independent NumPy implementation and
does not copy or import the authors' reference implementation.
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

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

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
    "run_count",
    "total_detections",
    "matched_true_positives",
    "false_positives",
    "mean_confidence",
    "fraction_true_positives",
    "global_calibration_gap",
    "maximum_confidence",
    "detection_ece",
    "detection_ece_std",
    "detection_ece_mean_plus_minus_std",
    "detection_ece_defined",
    "class_count",
    "total_possible_multidimensional_cells",
    "occupied_cells",
    "supported_cells",
    "detections_in_supported_cells",
    "fraction_detections_in_supported_cells",
    "supported_cell_size_median",
    "supported_cell_size_min",
    "supported_cell_size_max",
    "prediction_score_floor",
    "match_iou_threshold",
    "max_detections_per_image",
    "numeric_calibration_dimensions",
    "class_handling",
    "bins_per_dimension",
    "minimum_samples_per_cell",
)

SUPPORT_FIELDS = (
    "detector",
    "seed",
    "total_detections_entering_calibration",
    "class_count",
    "total_possible_multidimensional_cells",
    "occupied_cells",
    "cells_meeting_minimum_support",
    "detections_in_supported_cells",
    "fraction_detections_contributing_to_supported_cells",
    "supported_cell_size_median",
    "supported_cell_size_min",
    "supported_cell_size_max",
    "detection_ece",
    "detection_ece_defined",
    "prediction_score_floor",
    "bins_per_dimension",
    "minimum_samples_per_cell",
)

SENSITIVITY_FIELDS = (
    "sensitivity_type",
    "detector",
    "seed",
    "equal_bins_per_dimension",
    "bins_per_dimension",
    "minimum_samples_per_cell",
    "confidence_floor",
    "is_original_setting",
    "detection_ece",
    "detection_ece_defined",
    "undefined_reason",
    "total_detections_entering_calibration",
    "baseline_floor_detection_count",
    "detections_removed_from_baseline_floor",
    "fraction_baseline_detections_retained",
    "total_possible_multidimensional_cells",
    "occupied_cells",
    "cells_meeting_minimum_support",
    "detections_in_supported_cells",
    "fraction_detections_contributing_to_supported_cells",
    "supported_cell_size_median",
    "supported_cell_size_min",
    "supported_cell_size_max",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen Phase 5 evidence used by the offline analysis."""

    phase5_config: Path
    phase5_summary: Path
    test_annotations: Path


class SensitivitySettings(StrictModel):
    """Predeclared descriptive robustness grid."""

    equal_bins_per_dimension: tuple[int, ...]
    minimum_samples_per_cell: tuple[int, ...]
    confidence_floors: tuple[float, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> SensitivitySettings:
        if not self.equal_bins_per_dimension or any(
            value < 2 for value in self.equal_bins_per_dimension
        ):
            raise ValueError("sensitivity bin counts must be non-empty and at least two")
        if len(set(self.equal_bins_per_dimension)) != len(self.equal_bins_per_dimension):
            raise ValueError("sensitivity bin counts must be unique")
        if not self.minimum_samples_per_cell or any(
            value < 1 for value in self.minimum_samples_per_cell
        ):
            raise ValueError("sensitivity minimum-cell values must be positive")
        if len(set(self.minimum_samples_per_cell)) != len(self.minimum_samples_per_cell):
            raise ValueError("sensitivity minimum-cell values must be unique")
        if not self.confidence_floors or any(
            value < 0 or value > 1 for value in self.confidence_floors
        ):
            raise ValueError("sensitivity confidence floors must lie in [0, 1]")
        if len(set(self.confidence_floors)) != len(self.confidence_floors):
            raise ValueError("sensitivity confidence floors must be unique")
        return self


class CalibrationSettings(StrictModel):
    """D-ECE matching, binning, reliability, and sensitivity settings."""

    prediction_score_floor: float = Field(ge=0, le=1)
    match_iou_threshold: float = Field(gt=0, le=1)
    max_detections_per_image: int = Field(ge=1)
    bins_per_dimension: tuple[int, int, int, int, int]
    minimum_samples_per_cell: int = Field(ge=1)
    reliability_bins: int = Field(ge=2, le=100)
    sensitivity: SensitivitySettings

    @model_validator(mode="after")
    def validate_primary_and_grid(self) -> CalibrationSettings:
        if any(value < 2 for value in self.bins_per_dimension):
            raise ValueError("bins_per_dimension values must all be at least two")
        if len(set(self.bins_per_dimension)) != 1:
            raise ValueError("the predeclared sensitivity grid requires equal primary bin counts")
        primary_bin_count = self.bins_per_dimension[0]
        if primary_bin_count not in self.sensitivity.equal_bins_per_dimension:
            raise ValueError("the original bin count must appear in the sensitivity grid")
        if self.minimum_samples_per_cell not in self.sensitivity.minimum_samples_per_cell:
            raise ValueError("the original minimum-cell rule must appear in the sensitivity grid")
        if self.prediction_score_floor not in self.sensitivity.confidence_floors:
            raise ValueError("the original confidence floor must appear in floor sensitivity")
        if any(floor < self.prediction_score_floor for floor in self.sensitivity.confidence_floors):
            raise ValueError("floor sensitivity cannot go below the frozen bundle floor")
        return self


class PlotSettings(StrictModel):
    """Deterministic figure settings."""

    dpi: int = Field(ge=72, le=600)


class OutputSettings(StrictModel):
    """Versioned generated table, figure, and provenance paths."""

    log_dir: Path
    summary_json: Path
    summary_table: Path
    support_table: Path
    sensitivity_table: Path
    reliability_figure: Path
    support_figure: Path
    binning_sensitivity_figure: Path
    confidence_floor_sensitivity_figure: Path


class CalibrationConfig(StrictModel):
    """Strict Batch 33 detection-calibration contract."""

    schema_version: Literal[2]
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
    """Per-detection numeric features, class strata, and match outcomes."""

    features: FloatArray
    matched: BoolArray
    class_ids: IntArray

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        matched = np.asarray(self.matched, dtype=np.bool_)
        class_ids = np.asarray(self.class_ids, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (N, {len(FEATURE_NAMES)})")
        if matched.ndim != 1 or len(matched) != len(features):
            raise ValueError("matched must have shape (N,) aligned with features")
        if class_ids.ndim != 1 or len(class_ids) != len(features):
            raise ValueError("class_ids must have shape (N,) aligned with features")
        if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
            raise ValueError("all calibration features must be finite values in [0, 1]")
        features = features.copy()
        matched = matched.copy()
        class_ids = class_ids.copy()
        features.setflags(write=False)
        matched.setflags(write=False)
        class_ids.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "matched", matched)
        object.__setattr__(self, "class_ids", class_ids)

    @property
    def confidence(self) -> FloatArray:
        """Return the confidence dimension."""

        return self.features[:, 0]


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    """Load and strictly validate the Batch 33 YAML configuration."""

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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
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
    """Match detections and encode confidence, class, location, and scale.

    The canonical matcher retains scores greater than or equal to ``score_floor``,
    sorts them stably in descending confidence order, caps each image, and pairs
    a prediction with the highest-IoU unmatched target of the same class when
    IoU is greater than or equal to ``iou_threshold``. A target can be used once.
    Missed targets have no emitted confidence and are outside this population.
    """

    prediction_map = {item.image_id: item for item in predictions}
    target_map = {item.image_id: item for item in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("prediction and target image IDs must be unique")
    if set(prediction_map) != set(target_map):
        raise ValueError("prediction and target image ID sets must match")

    feature_rows: list[list[float]] = []
    matched_rows: list[bool] = []
    class_rows: list[int] = []
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
            class_rows.append(int(prediction.labels[index]))

    return DetectionCalibrationSamples(
        features=np.asarray(feature_rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)),
        matched=np.asarray(matched_rows, dtype=np.bool_),
        class_ids=np.asarray(class_rows, dtype=np.int64),
    )


def _undefined_d_ece_result(
    *, total_possible_cells: int, prediction_count: int, reason: str
) -> dict[str, Any]:
    return {
        "detection_ece": None,
        "detection_ece_defined": False,
        "undefined_reason": reason,
        "prediction_count": prediction_count,
        "detections_in_supported_cells": 0,
        "fraction_detections_in_supported_cells": 0.0,
        "total_possible_multidimensional_cells": total_possible_cells,
        "occupied_cells": 0,
        "supported_cells": 0,
        "supported_cell_size_median": None,
        "supported_cell_size_min": None,
        "supported_cell_size_max": None,
        "bins": [],
    }


def detection_expected_calibration_error(
    features: FloatArray,
    matched: BoolArray,
    *,
    bins: Sequence[int],
    minimum_samples_per_cell: int,
    class_ids: IntArray | None = None,
    possible_class_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute class-conditioned multivariate D-ECE with equal-width cells.

    Each occupied class/numeric cell contributes ``n_cell / N`` multiplied by
    the absolute difference between empirical precision and mean confidence.
    Cells with fewer than ``minimum_samples_per_cell`` detections contribute
    zero; the denominator remains all emitted detections. Numeric bin indices
    are ``floor(value * bin_count)`` clipped so 1.0 enters the final bin.
    """

    features = np.asarray(features, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.bool_)
    bin_counts = tuple(int(value) for value in bins)
    if features.ndim != 2 or features.shape[1] != len(bin_counts):
        raise ValueError("bins must provide one count for each feature dimension")
    if matched.ndim != 1 or len(matched) != len(features):
        raise ValueError("matched must be aligned with features")
    if any(value < 2 for value in bin_counts):
        raise ValueError("each bin count must be at least two")
    if minimum_samples_per_cell < 1:
        raise ValueError("minimum_samples_per_cell must be positive")
    if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
        raise ValueError("features must be finite and lie in [0, 1]")

    if class_ids is None:
        classes = np.zeros(len(features), dtype=np.int64)
    else:
        classes = np.asarray(class_ids, dtype=np.int64)
    if classes.ndim != 1 or len(classes) != len(features):
        raise ValueError("class_ids must be aligned with features")

    if possible_class_ids is None:
        class_universe = tuple(int(value) for value in np.unique(classes)) or (0,)
    else:
        class_universe = tuple(sorted({int(value) for value in possible_class_ids}))
        if not class_universe:
            raise ValueError("possible_class_ids must not be empty")
    unknown_classes = set(int(value) for value in np.unique(classes)) - set(class_universe)
    if unknown_classes:
        raise ValueError("class_ids contain values outside possible_class_ids")

    total_possible_cells = int(np.prod(bin_counts, dtype=np.int64) * len(class_universe))
    if not len(features):
        return _undefined_d_ece_result(
            total_possible_cells=total_possible_cells,
            prediction_count=0,
            reason="no_detections_at_confidence_floor",
        )

    numeric_indices = np.column_stack(
        [
            np.minimum(np.floor(features[:, dimension] * count).astype(np.int64), count - 1)
            for dimension, count in enumerate(bin_counts)
        ]
    )
    class_to_index = {class_id: index for index, class_id in enumerate(class_universe)}
    class_indices = np.asarray([class_to_index[int(value)] for value in classes], dtype=np.int64)
    flat_numeric = np.ravel_multi_index(numeric_indices.T, bin_counts)
    numeric_cell_count = int(np.prod(bin_counts, dtype=np.int64))
    flat_cells = class_indices * numeric_cell_count + flat_numeric
    occupied, counts = np.unique(flat_cells, return_counts=True)

    detection_ece = 0.0
    supported_sizes: list[int] = []
    bin_records: list[dict[str, Any]] = []
    for flat_cell, count_value in zip(occupied, counts, strict=True):
        count = int(count_value)
        mask = flat_cells == flat_cell
        mean_confidence = float(np.mean(features[mask, 0]))
        empirical_precision = float(np.mean(matched[mask]))
        gap = abs(empirical_precision - mean_confidence)
        supported = count >= minimum_samples_per_cell
        contribution = float(count / len(features) * gap) if supported else 0.0
        if supported:
            detection_ece += contribution
            supported_sizes.append(count)
        class_index, numeric_flat = divmod(int(flat_cell), numeric_cell_count)
        bin_records.append(
            {
                "class_id": class_universe[class_index],
                "indices": [int(index) for index in np.unravel_index(numeric_flat, bin_counts)],
                "sample_count": count,
                "mean_confidence": mean_confidence,
                "fraction_true_positives": empirical_precision,
                "absolute_gap": gap,
                "meets_minimum_support": supported,
                "weighted_contribution": contribution,
            }
        )

    supported_detection_count = int(sum(supported_sizes))
    if not supported_sizes:
        result = _undefined_d_ece_result(
            total_possible_cells=total_possible_cells,
            prediction_count=len(features),
            reason="no_occupied_cell_meets_minimum_support",
        )
        result["occupied_cells"] = len(occupied)
        result["bins"] = bin_records
        return result
    return {
        "detection_ece": float(detection_ece),
        "detection_ece_defined": True,
        "undefined_reason": None,
        "prediction_count": len(features),
        "detections_in_supported_cells": supported_detection_count,
        "fraction_detections_in_supported_cells": float(supported_detection_count / len(features)),
        "total_possible_multidimensional_cells": total_possible_cells,
        "occupied_cells": len(occupied),
        "supported_cells": len(supported_sizes),
        "supported_cell_size_median": float(np.median(supported_sizes)),
        "supported_cell_size_min": min(supported_sizes),
        "supported_cell_size_max": max(supported_sizes),
        "bins": bin_records,
    }


def reliability_bins(
    confidence: FloatArray, matched: BoolArray, *, bins: int
) -> list[dict[str, Any]]:
    """Return confidence-only marginal reliability points for detections."""

    confidence = np.asarray(confidence, dtype=np.float64)
    matched = np.asarray(matched, dtype=np.bool_)
    if confidence.ndim != 1 or matched.ndim != 1 or len(confidence) != len(matched):
        raise ValueError("confidence and matched must be aligned one-dimensional arrays")
    if bins < 2:
        raise ValueError("reliability binning requires at least two bins")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and lie in [0, 1]")
    if not len(confidence):
        return []

    indices = np.minimum(np.floor(confidence * bins).astype(np.int64), bins - 1)
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
    expected = {(run.detector, run.seed) for run in phase5.runs}
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
        raise ValueError("Phase 5 summary does not contain its complete configured run grid")
    bundles.sort(key=lambda item: (str(item["detector"]), int(item["seed"])))
    return phase5, summary, bundles, targets


def _result_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detection_ece": result["detection_ece"],
        "detection_ece_defined": result["detection_ece_defined"],
        "total_possible_multidimensional_cells": result["total_possible_multidimensional_cells"],
        "occupied_cells": result["occupied_cells"],
        "supported_cells": result["supported_cells"],
        "detections_in_supported_cells": result["detections_in_supported_cells"],
        "fraction_detections_in_supported_cells": result["fraction_detections_in_supported_cells"],
        "supported_cell_size_median": result["supported_cell_size_median"],
        "supported_cell_size_min": result["supported_cell_size_min"],
        "supported_cell_size_max": result["supported_cell_size_max"],
    }


def _run_summary_row(
    *,
    detector: str,
    seed: int,
    samples: DetectionCalibrationSamples,
    result: Mapping[str, Any],
    settings: CalibrationSettings,
    class_count: int,
) -> dict[str, Any]:
    count = len(samples.matched)
    return {
        "row_type": "run",
        "detector": detector,
        "seed": seed,
        "run_count": 1,
        "total_detections": count,
        "matched_true_positives": int(np.sum(samples.matched)),
        "false_positives": int(np.sum(~samples.matched)),
        "mean_confidence": float(np.mean(samples.confidence)) if count else None,
        "fraction_true_positives": float(np.mean(samples.matched)) if count else None,
        "global_calibration_gap": (
            abs(float(np.mean(samples.matched)) - float(np.mean(samples.confidence)))
            if count
            else None
        ),
        "maximum_confidence": float(np.max(samples.confidence)) if count else None,
        "class_count": class_count,
        **_result_fields(result),
        "prediction_score_floor": settings.prediction_score_floor,
        "match_iou_threshold": settings.match_iou_threshold,
        "max_detections_per_image": settings.max_detections_per_image,
        "numeric_calibration_dimensions": "|".join(FEATURE_NAMES),
        "class_handling": "categorical_class_conditioned_cells",
        "bins_per_dimension": "|".join(str(value) for value in settings.bins_per_dimension),
        "minimum_samples_per_cell": settings.minimum_samples_per_cell,
    }


def _add_detector_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    for detector in sorted({str(row["detector"]) for row in rows}):
        detector_rows = [row for row in rows if row["detector"] == detector]
        values = np.asarray(
            [row["detection_ece"] for row in detector_rows if row["detection_ece_defined"]],
            dtype=np.float64,
        )
        if not len(values):
            mean = None
            std = None
            display = "undefined"
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else None
            display = f"{mean:.6f} +/- {std:.6f}" if std is not None else f"{mean:.6f}"
        common = detector_rows[0]
        output.append(
            {
                "row_type": "detector_descriptive_mean",
                "detector": detector,
                "run_count": len(detector_rows),
                "detection_ece": mean,
                "detection_ece_std": std,
                "detection_ece_mean_plus_minus_std": display,
                "detection_ece_defined": bool(len(values)),
                "prediction_score_floor": common["prediction_score_floor"],
                "match_iou_threshold": common["match_iou_threshold"],
                "max_detections_per_image": common["max_detections_per_image"],
                "numeric_calibration_dimensions": common["numeric_calibration_dimensions"],
                "class_handling": common["class_handling"],
                "bins_per_dimension": common["bins_per_dimension"],
                "minimum_samples_per_cell": common["minimum_samples_per_cell"],
            }
        )
    return output


def _support_row(row: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detector": row["detector"],
        "seed": row["seed"],
        "total_detections_entering_calibration": row["total_detections"],
        "class_count": row["class_count"],
        "total_possible_multidimensional_cells": result["total_possible_multidimensional_cells"],
        "occupied_cells": result["occupied_cells"],
        "cells_meeting_minimum_support": result["supported_cells"],
        "detections_in_supported_cells": result["detections_in_supported_cells"],
        "fraction_detections_contributing_to_supported_cells": result[
            "fraction_detections_in_supported_cells"
        ],
        "supported_cell_size_median": result["supported_cell_size_median"],
        "supported_cell_size_min": result["supported_cell_size_min"],
        "supported_cell_size_max": result["supported_cell_size_max"],
        "detection_ece": result["detection_ece"],
        "detection_ece_defined": result["detection_ece_defined"],
        "prediction_score_floor": row["prediction_score_floor"],
        "bins_per_dimension": row["bins_per_dimension"],
        "minimum_samples_per_cell": row["minimum_samples_per_cell"],
    }


def _sensitivity_row(
    *,
    sensitivity_type: str,
    detector: str,
    seed: int,
    bins: tuple[int, ...],
    minimum_samples: int,
    confidence_floor: float,
    is_original: bool,
    result: Mapping[str, Any],
    baseline_count: int,
) -> dict[str, Any]:
    count = int(result["prediction_count"])
    return {
        "sensitivity_type": sensitivity_type,
        "detector": detector,
        "seed": seed,
        "equal_bins_per_dimension": bins[0] if len(set(bins)) == 1 else "",
        "bins_per_dimension": "|".join(str(value) for value in bins),
        "minimum_samples_per_cell": minimum_samples,
        "confidence_floor": confidence_floor,
        "is_original_setting": is_original,
        "detection_ece": result["detection_ece"],
        "detection_ece_defined": result["detection_ece_defined"],
        "undefined_reason": result["undefined_reason"],
        "total_detections_entering_calibration": count,
        "baseline_floor_detection_count": baseline_count,
        "detections_removed_from_baseline_floor": baseline_count - count,
        "fraction_baseline_detections_retained": (
            float(count / baseline_count) if baseline_count else None
        ),
        "total_possible_multidimensional_cells": result["total_possible_multidimensional_cells"],
        "occupied_cells": result["occupied_cells"],
        "cells_meeting_minimum_support": result["supported_cells"],
        "detections_in_supported_cells": result["detections_in_supported_cells"],
        "fraction_detections_contributing_to_supported_cells": result[
            "fraction_detections_in_supported_cells"
        ],
        "supported_cell_size_median": result["supported_cell_size_median"],
        "supported_cell_size_min": result["supported_cell_size_min"],
        "supported_cell_size_max": result["supported_cell_size_max"],
    }


def _save_figure(path: Path, figure: Any, *, dpi: int) -> Path:
    from matplotlib import pyplot as plt

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


def _plot_setup() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _detector_style(detectors: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    palette = ("#2f6f9f", "#d97706", "#4f7c35", "#7c3f8c")
    colors = {detector: palette[index % len(palette)] for index, detector in enumerate(detectors)}
    display_names = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    titles = {
        detector: display_names.get(detector, detector.replace("_", " ")) for detector in detectors
    }
    return colors, titles


def _atomic_reliability_figure(
    path: Path,
    reliability: Mapping[tuple[str, int], list[dict[str, Any]]],
    pooled: Mapping[str, list[dict[str, Any]]],
    *,
    dpi: int,
) -> Path:
    plt = _plot_setup()
    detectors = sorted(pooled)
    colors, titles = _detector_style(detectors)
    figure, axes = plt.subplots(1, len(detectors), figsize=(6 * len(detectors), 5.4), squeeze=False)
    for axis, detector in zip(axes[0], detectors, strict=True):
        axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.2, label="Ideal")
        for (row_detector, seed), points in sorted(reliability.items()):
            if row_detector != detector or not points:
                continue
            axis.plot(
                [point["mean_confidence"] for point in points],
                [point["fraction_true_positives"] for point in points],
                marker="o",
                markersize=3,
                linewidth=0.9,
                color=colors[detector],
                alpha=0.32,
                label=f"Run {seed}",
            )
        pooled_points = pooled[detector]
        if pooled_points:
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
        axis.set_xlabel("Mean emitted-detection confidence")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.20)
        axis.legend(loc="lower right", fontsize=8)
    axes[0, 0].set_ylabel("Matched fraction (confidence-only marginal)")
    figure.suptitle(
        "Confidence-only marginal reliability (not a visualization of all D-ECE dimensions)"
    )
    figure.tight_layout()
    return _save_figure(path, figure, dpi=dpi)


def _atomic_support_figure(
    path: Path, support_rows: Sequence[Mapping[str, Any]], *, dpi: int
) -> Path:
    plt = _plot_setup()
    detectors = sorted({str(row["detector"]) for row in support_rows})
    colors, titles = _detector_style(detectors)
    figure, axes = plt.subplots(
        len(detectors), 2, figsize=(12, 4.2 * len(detectors)), squeeze=False
    )
    for row_index, detector in enumerate(detectors):
        rows = sorted(
            (row for row in support_rows if row["detector"] == detector),
            key=lambda row: int(row["seed"]),
        )
        labels = [str(row["seed"]) for row in rows]
        positions = np.arange(len(rows))
        fractions = [row["fraction_detections_contributing_to_supported_cells"] for row in rows]
        axes[row_index, 0].bar(positions, fractions, color=colors[detector])
        axes[row_index, 0].set_ylim(0, 1.05)
        axes[row_index, 0].set_ylabel("Fraction of detections")
        axes[row_index, 0].set_title(f"{titles[detector]}: supported-cell detection fraction")
        occupied = [row["occupied_cells"] for row in rows]
        supported = [row["cells_meeting_minimum_support"] for row in rows]
        width = 0.38
        axes[row_index, 1].bar(
            positions - width / 2, occupied, width, label="Occupied", color="#94a3b8"
        )
        axes[row_index, 1].bar(
            positions + width / 2, supported, width, label="Supported", color=colors[detector]
        )
        possible = rows[0]["total_possible_multidimensional_cells"]
        axes[row_index, 1].set_title(f"{titles[detector]}: cells out of {possible} possible")
        axes[row_index, 1].set_ylabel("Cell count")
        axes[row_index, 1].legend()
        for axis in axes[row_index]:
            axis.set_xticks(positions, labels)
            axis.set_xlabel("Run seed")
            axis.grid(axis="y", alpha=0.20)
    figure.suptitle("Primary D-ECE support and occupancy diagnostics")
    figure.tight_layout()
    return _save_figure(path, figure, dpi=dpi)


def _atomic_binning_sensitivity_figure(
    path: Path, rows: Sequence[Mapping[str, Any]], *, dpi: int
) -> Path:
    plt = _plot_setup()
    selected = [row for row in rows if row["sensitivity_type"] == "binning_minimum_support"]
    detectors = sorted({str(row["detector"]) for row in selected})
    bin_values = sorted({int(row["equal_bins_per_dimension"]) for row in selected})
    minimum_values = sorted({int(row["minimum_samples_per_cell"]) for row in selected})
    original = next(row for row in selected if row["is_original_setting"])
    original_x = bin_values.index(int(original["equal_bins_per_dimension"]))
    original_y = minimum_values.index(int(original["minimum_samples_per_cell"]))
    figure, axes = plt.subplots(
        len(detectors), 2, figsize=(12, 4.3 * len(detectors)), squeeze=False
    )
    for row_index, detector in enumerate(detectors):
        for column, (field, title, value_format, limits) in enumerate(
            (
                ("detection_ece", "descriptive mean D-ECE", ".3f", None),
                (
                    "fraction_detections_contributing_to_supported_cells",
                    "mean supported-detection fraction",
                    ".2f",
                    (0.0, 1.0),
                ),
            )
        ):
            matrix = np.full((len(minimum_values), len(bin_values)), np.nan)
            for minimum_index, minimum in enumerate(minimum_values):
                for bin_index, bin_count in enumerate(bin_values):
                    values = [
                        float(item[field])
                        for item in selected
                        if item["detector"] == detector
                        and int(item["minimum_samples_per_cell"]) == minimum
                        and int(item["equal_bins_per_dimension"]) == bin_count
                        and item[field] is not None
                    ]
                    if values:
                        matrix[minimum_index, bin_index] = float(np.mean(values))
            image = axes[row_index, column].imshow(
                matrix,
                aspect="auto",
                cmap="viridis",
                vmin=None if limits is None else limits[0],
                vmax=None if limits is None else limits[1],
            )
            for y_index in range(matrix.shape[0]):
                for x_index in range(matrix.shape[1]):
                    value = matrix[y_index, x_index]
                    label = "NA" if np.isnan(value) else format(value, value_format)
                    axes[row_index, column].text(
                        x_index, y_index, label, ha="center", va="center", color="white"
                    )
            from matplotlib.patches import Rectangle

            axes[row_index, column].add_patch(
                Rectangle(
                    (original_x - 0.5, original_y - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#dc2626",
                    linewidth=3,
                    label="Original 5-bin / minimum-8 setting",
                )
            )
            axes[row_index, column].set_xticks(range(len(bin_values)), bin_values)
            axes[row_index, column].set_yticks(range(len(minimum_values)), minimum_values)
            axes[row_index, column].set_xlabel("Equal bins per numeric dimension")
            axes[row_index, column].set_ylabel("Minimum detections per cell")
            axes[row_index, column].set_title(f"{detector}: {title}")
            axes[row_index, column].legend(loc="lower left", fontsize=7)
            figure.colorbar(image, ax=axes[row_index, column], shrink=0.78)
    figure.suptitle("Predeclared descriptive D-ECE binning/support sensitivity")
    figure.tight_layout()
    return _save_figure(path, figure, dpi=dpi)


def _atomic_floor_sensitivity_figure(
    path: Path, rows: Sequence[Mapping[str, Any]], *, dpi: int
) -> Path:
    plt = _plot_setup()
    selected = [row for row in rows if row["sensitivity_type"] == "confidence_floor"]
    detectors = sorted({str(row["detector"]) for row in selected})
    original_floor = float(
        next(row for row in selected if row["is_original_setting"])["confidence_floor"]
    )
    colors, titles = _detector_style(detectors)
    figure, axes = plt.subplots(
        len(detectors), 2, figsize=(12, 4.3 * len(detectors)), squeeze=False
    )
    for row_index, detector in enumerate(detectors):
        detector_rows = [row for row in selected if row["detector"] == detector]
        run_ids = sorted({int(row["seed"]) for row in detector_rows})
        for run_id in run_ids:
            run_rows = sorted(
                (row for row in detector_rows if int(row["seed"]) == run_id),
                key=lambda row: float(row["confidence_floor"]),
            )
            floors = [float(row["confidence_floor"]) for row in run_rows]
            d_ece = [
                np.nan if row["detection_ece"] is None else float(row["detection_ece"])
                for row in run_rows
            ]
            retained = [row["fraction_baseline_detections_retained"] for row in run_rows]
            axes[row_index, 0].plot(
                floors, d_ece, marker="o", color=colors[detector], alpha=0.48, label=str(run_id)
            )
            axes[row_index, 1].plot(
                floors, retained, marker="o", color=colors[detector], alpha=0.48, label=str(run_id)
            )
        axes[row_index, 0].set_title(f"{titles[detector]}: primary-setting D-ECE")
        axes[row_index, 0].set_ylabel("D-ECE")
        axes[row_index, 1].set_title(f"{titles[detector]}: retained emitted population")
        axes[row_index, 1].set_ylabel("Fraction retained vs 0.001 bundle floor")
        axes[row_index, 1].set_ylim(0, 1.05)
        for axis in axes[row_index]:
            axis.set_xscale("log")
            axis.set_xlabel("Confidence floor")
            axis.axvline(
                original_floor,
                color="#dc2626",
                linestyle="--",
                linewidth=1.4,
                label="Original floor",
            )
            axis.grid(alpha=0.20)
            axis.legend(title="Run seed", fontsize=8, ncol=2)
    figure.suptitle("Predeclared descriptive confidence-floor sensitivity")
    figure.tight_layout()
    return _save_figure(path, figure, dpi=dpi)


def _class_universe(
    bundles: Sequence[Mapping[str, Any]], targets: Sequence[ImageTarget]
) -> tuple[int, ...]:
    class_ids = {int(value) for target in targets for value in target.labels}
    class_ids.update(
        int(value)
        for bundle in bundles
        for prediction in bundle["predictions"]
        for value in prediction.labels
    )
    if not class_ids:
        raise ValueError("calibration requires at least one declared target or prediction class")
    return tuple(sorted(class_ids))


def preflight(config: CalibrationConfig) -> dict[str, Any]:
    """Validate immutable evidence and the predeclared grid without outputs."""

    phase5, summary, bundles, targets = _validate_upstream(config)
    return {
        "status": "preflight_passed",
        "experiment_id": config.experiment_id,
        "detectors": sorted({str(bundle["detector"]) for bundle in bundles}),
        "seeds": list(phase5.seeds),
        "bundle_count": len(bundles),
        "image_count": len(targets),
        "annotation_count": int(summary["annotation_count"]),
        "class_ids": list(_class_universe(bundles, targets)),
        "original_setting": {
            "prediction_score_floor": config.calibration.prediction_score_floor,
            "bins_per_dimension": list(config.calibration.bins_per_dimension),
            "minimum_samples_per_cell": config.calibration.minimum_samples_per_cell,
        },
        "sensitivity": config.calibration.sensitivity.model_dump(mode="json"),
    }


def run_calibration(config: CalibrationConfig) -> dict[str, Any]:
    """Generate primary D-ECE, support, sensitivity, figures, and provenance."""

    phase5, phase5_summary, bundles, targets = _validate_upstream(config)
    settings = config.calibration
    class_universe = _class_universe(bundles, targets)
    run_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    samples_by_run: dict[tuple[str, int], DetectionCalibrationSamples] = {}
    reliability: dict[tuple[str, int], list[dict[str, Any]]] = {}
    multivariate_cells: dict[str, Any] = {}

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
        primary = detection_expected_calibration_error(
            samples.features,
            samples.matched,
            class_ids=samples.class_ids,
            possible_class_ids=class_universe,
            bins=settings.bins_per_dimension,
            minimum_samples_per_cell=settings.minimum_samples_per_cell,
        )
        if not primary["detection_ece_defined"]:
            raise ValueError(f"primary D-ECE is undefined for configured run {(detector, seed)}")
        row = _run_summary_row(
            detector=detector,
            seed=seed,
            samples=samples,
            result=primary,
            settings=settings,
            class_count=len(class_universe),
        )
        run_rows.append(row)
        support_rows.append(_support_row(row, primary))
        key = detector, seed
        samples_by_run[key] = samples
        reliability[key] = reliability_bins(
            samples.confidence, samples.matched, bins=settings.reliability_bins
        )
        multivariate_cells[f"{detector}_seed{seed}"] = primary["bins"]

        for bin_count in settings.sensitivity.equal_bins_per_dimension:
            bins = (bin_count,) * len(FEATURE_NAMES)
            for minimum_samples in settings.sensitivity.minimum_samples_per_cell:
                result = detection_expected_calibration_error(
                    samples.features,
                    samples.matched,
                    class_ids=samples.class_ids,
                    possible_class_ids=class_universe,
                    bins=bins,
                    minimum_samples_per_cell=minimum_samples,
                )
                sensitivity_rows.append(
                    _sensitivity_row(
                        sensitivity_type="binning_minimum_support",
                        detector=detector,
                        seed=seed,
                        bins=bins,
                        minimum_samples=minimum_samples,
                        confidence_floor=settings.prediction_score_floor,
                        is_original=(
                            bins == settings.bins_per_dimension
                            and minimum_samples == settings.minimum_samples_per_cell
                        ),
                        result=result,
                        baseline_count=len(samples.matched),
                    )
                )

        for confidence_floor in settings.sensitivity.confidence_floors:
            floor_samples = build_detection_calibration_samples(
                bundle["predictions"],
                targets,
                score_floor=confidence_floor,
                iou_threshold=settings.match_iou_threshold,
                max_detections=settings.max_detections_per_image,
            )
            result = detection_expected_calibration_error(
                floor_samples.features,
                floor_samples.matched,
                class_ids=floor_samples.class_ids,
                possible_class_ids=class_universe,
                bins=settings.bins_per_dimension,
                minimum_samples_per_cell=settings.minimum_samples_per_cell,
            )
            sensitivity_rows.append(
                _sensitivity_row(
                    sensitivity_type="confidence_floor",
                    detector=detector,
                    seed=seed,
                    bins=settings.bins_per_dimension,
                    minimum_samples=settings.minimum_samples_per_cell,
                    confidence_floor=confidence_floor,
                    is_original=confidence_floor == settings.prediction_score_floor,
                    result=result,
                    baseline_count=len(samples.matched),
                )
            )

    output_rows = _add_detector_summaries(run_rows)
    detectors = sorted({str(row["detector"]) for row in run_rows})
    pooled_reliability: dict[str, list[dict[str, Any]]] = {}
    for detector in detectors:
        detector_samples = [
            samples for (name, _), samples in samples_by_run.items() if name == detector
        ]
        pooled_reliability[detector] = reliability_bins(
            np.concatenate([samples.confidence for samples in detector_samples]),
            np.concatenate([samples.matched for samples in detector_samples]),
            bins=settings.reliability_bins,
        )

    artifact_paths = [
        _atomic_csv(config.resolve(config.outputs.summary_table), output_rows, SUMMARY_FIELDS),
        _atomic_csv(config.resolve(config.outputs.support_table), support_rows, SUPPORT_FIELDS),
        _atomic_csv(
            config.resolve(config.outputs.sensitivity_table),
            sensitivity_rows,
            SENSITIVITY_FIELDS,
        ),
        _atomic_reliability_figure(
            config.resolve(config.outputs.reliability_figure),
            reliability,
            pooled_reliability,
            dpi=config.plots.dpi,
        ),
        _atomic_support_figure(
            config.resolve(config.outputs.support_figure), support_rows, dpi=config.plots.dpi
        ),
        _atomic_binning_sensitivity_figure(
            config.resolve(config.outputs.binning_sensitivity_figure),
            sensitivity_rows,
            dpi=config.plots.dpi,
        ),
        _atomic_floor_sensitivity_figure(
            config.resolve(config.outputs.confidence_floor_sensitivity_figure),
            sensitivity_rows,
            dpi=config.plots.dpi,
        ),
    ]

    detector_summaries = {
        str(row["detector"]): {
            "comparison_status": "descriptive_no_between_detector_inference",
            "run_count": row["run_count"],
            "mean_detection_ece": row["detection_ece"],
            "sample_std_detection_ece": row["detection_ece_std"],
        }
        for row in output_rows
        if row["row_type"] == "detector_descriptive_mean"
    }
    floor_rows = [row for row in sensitivity_rows if row["sensitivity_type"] == "confidence_floor"]
    floor_population_summary: dict[str, dict[str, Any]] = {}
    for detector in detectors:
        floor_population_summary[detector] = {}
        for floor in settings.sensitivity.confidence_floors:
            selected = [
                row
                for row in floor_rows
                if row["detector"] == detector and row["confidence_floor"] == floor
            ]
            retained = [
                float(row["fraction_baseline_detections_retained"])
                for row in selected
                if row["fraction_baseline_detections_retained"] is not None
            ]
            floor_population_summary[detector][str(floor)] = {
                "run_count": len(selected),
                "d_ece_defined_run_count": sum(
                    bool(row["detection_ece_defined"]) for row in selected
                ),
                "zero_detection_run_count": sum(
                    int(row["total_detections_entering_calibration"]) == 0 for row in selected
                ),
                "equal_run_mean_fraction_baseline_detections_retained": (
                    float(np.mean(retained)) if retained else None
                ),
            }
    higher_floor_rows = [
        row
        for row in floor_rows
        if float(row["confidence_floor"]) > settings.prediction_score_floor
    ]
    floor_materially_changes_population = any(
        row["fraction_baseline_detections_retained"] is not None
        and float(row["fraction_baseline_detections_retained"]) < 0.9
        for row in higher_floor_rows
    )
    summary = {
        "schema_version": 2,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "method": {
            "name": "class_conditioned_multivariate_detection_expected_calibration_error",
            "paper": (
                "Kuppers et al., Multivariate Confidence Calibration for Object Detection, "
                "CVPR Workshops 2020"
            ),
            "doi": "10.1109/CVPRW50498.2020.00171",
            "numeric_dimensions": list(FEATURE_NAMES),
            "class_handling": (
                "Predicted class is a categorical stratum; RSNA has one foreground class. "
                "Class is not treated as an ordered numeric histogram dimension."
            ),
            "class_ids": list(class_universe),
            "binning": (
                "Equal-width bins on [0,1]; floor(value*bins), with 1.0 clipped into the final bin."
            ),
            "weighting": (
                "Each supported cell contributes cell_count / all_emitted_detection_count "
                "times absolute(empirical_precision - mean_confidence)."
            ),
            "minimum_cell_rule": (
                "Cells with fewer than the configured minimum contribute zero; all emitted "
                "detections remain in the weighting denominator."
            ),
            "original_setting": {
                "bins_per_dimension": list(settings.bins_per_dimension),
                "minimum_samples_per_cell": settings.minimum_samples_per_cell,
                "prediction_score_floor": settings.prediction_score_floor,
            },
            "matching_rule": {
                "order": "stable_descending_confidence_with_per_image_cap",
                "class_rule": "same_predicted_and_target_class",
                "assignment": "highest_IoU_currently_unmatched_target",
                "target_reuse": False,
                "comparison": "greater_than_or_equal",
                "iou_threshold": settings.match_iou_threshold,
            },
            "confidence_floor": {
                "comparison": "greater_than_or_equal",
                "value": settings.prediction_score_floor,
                "source": "frozen_phase5_prediction_bundle_minimum",
                "lower_floor_not_observable_from_bundles": True,
            },
            "paper_experiment_alignment": {
                "full_numeric_dimension_bins_per_dimension": 5,
                "minimum_cell_size": 8,
                "paper_detector_demonstration_protocol": {
                    "probability_threshold": 0.3,
                    "nms_iou_threshold": 0.6,
                    "reported_correctness_iou_thresholds": [0.6, 0.75],
                },
                "note": (
                    "The project IoU threshold and confidence floor are frozen project "
                    "evaluation choices, not universal constants in the D-ECE definition "
                    "and not a replication of the paper's detector demonstration protocol."
                ),
            },
            "population": "emitted_post_NMS_detections_at_or_above_the_confidence_floor",
            "missed_targets_included": False,
            "missed_target_scope": (
                "A missed ground-truth object has no emitted confidence and therefore lies "
                "outside D-ECE's emitted-detection calibration population."
            ),
            "interpretation": "descriptive_detection_confidence_calibration_only",
            "clinical_risk_calibration": False,
            "reliability_diagram_scope": "confidence_only_marginal_not_full_multivariate_D_ECE",
        },
        "sensitivity": {
            "status": "predeclared_descriptive_robustness_grid_not_model_selection",
            "grid": settings.sensitivity.model_dump(mode="json"),
            "original_setting_clearly_flagged_in_table": True,
            "low_confidence_detections_or_runs_removed_for_stability": False,
            "confidence_floor_materially_changes_emitted_population": (
                floor_materially_changes_population
            ),
            "materiality_rule": (
                "True if any predeclared higher-floor run retains less than 90% of its "
                "0.001-floor emitted detections. This is a descriptive flag, not a test."
            ),
            "confidence_floor_population_summary": floor_population_summary,
        },
        "exam_level_probability_calibration": {
            "separate_from_detection_level_d_ece": True,
            "feeds_d_ece": False,
            "receives_d_ece": False,
            "standard_decision_curve_analysis_created_in_batch30": False,
            "reason": (
                "Complete validation-frozen exam-level probability calibration inputs were "
                "unavailable; Batch 30 retained a probability-semantic DCA guard only."
            ),
        },
        "implementation": {
            "direct_netcal_dependency": False,
            "implementation_note": "Independent NumPy implementation of the paper definition.",
            "run_specific_branches": False,
            "all_configured_runs_use_same_pipeline": True,
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
        "detector_summaries": detector_summaries,
        "run_rows": run_rows,
        "support_rows": support_rows,
        "sensitivity_rows": sensitivity_rows,
        "reliability_bins_confidence_only_marginal": {
            f"{detector}_seed{seed}": points for (detector, seed), points in reliability.items()
        },
        "pooled_reliability_bins_confidence_only_marginal": pooled_reliability,
        "primary_multivariate_cells": multivariate_cells,
        "artifacts": {
            path.relative_to(config.project_root).as_posix(): sha256_file(path)
            for path in artifact_paths
        },
        "seeds": list(phase5.seeds),
    }
    summary_path = config.resolve(config.outputs.summary_json)
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
