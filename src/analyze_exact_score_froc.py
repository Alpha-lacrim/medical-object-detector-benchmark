"""Build exact-score FROC frontiers from frozen Phase 5 prediction bundles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import evaluate_operating_point, load_phase5_config, sha256_file
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget, match_image
from src.plot_froc_curves import load_froc_config
from src.stats.run_statistics import load_coco_targets

EXACT_PER_SEED_FIELDS = (
    "detector",
    "seed",
    "threshold_relaxation_rank",
    "threshold",
    "threshold_kind",
    "is_candidate_floor_boundary",
    "sensitivity",
    "true_positives",
    "false_positives",
    "false_negatives",
    "prediction_count",
    "target_count",
    "fp_per_image",
)
EXACT_AGGREGATE_CURVE_FIELDS = (
    "detector",
    "fp_per_image_query",
    "seed_count",
    "sensitivity",
    "sensitivity_std",
    "achieved_fp_per_image",
    "achieved_fp_per_image_std",
    "candidate_floor_limited_run_count",
    "candidate_floor_limited_seeds",
)
OPERATING_PER_SEED_FIELDS = (
    "detector",
    "seed",
    "fp_per_image_budget",
    "sensitivity",
    "achieved_fp_per_image",
    "selected_threshold",
    "selected_threshold_kind",
    "selected_at_candidate_floor_boundary",
    "candidate_score_floor",
    "candidate_floor_endpoint_sensitivity",
    "candidate_floor_endpoint_fp_per_image",
    "candidate_floor_limited",
)
OPERATING_FIELDS = (
    "detector",
    "fp_per_image_budget",
    "seed_count",
    "sensitivity",
    "sensitivity_std",
    "sensitivity_mean_plus_minus_std",
    "achieved_fp_per_image",
    "achieved_fp_per_image_std",
    "selected_threshold_mean",
    "selected_threshold_min",
    "selected_threshold_max",
    "selected_at_candidate_floor_count",
    "candidate_floor_limited_run_count",
    "candidate_floor_limited_seeds",
    "selection_rule",
)
COMPARISON_FIELDS = (
    "fp_per_image_budget",
    "historical_faster_sensitivity",
    "historical_yolo_sensitivity",
    "historical_faster_minus_yolo_gap",
    "exact_faster_sensitivity",
    "exact_yolo_sensitivity",
    "exact_faster_minus_yolo_gap",
    "gap_change_exact_minus_historical",
    "gap_change_classification",
    "direction_status",
    "faster_candidate_floor_limited_run_count",
    "faster_candidate_floor_limited_seeds",
    "yolo_candidate_floor_limited_run_count",
    "yolo_candidate_floor_limited_seeds",
)
PRIOR_COMPARISON_FIELDS = (
    "fp_per_image_budget",
    "prior_faster_sensitivity",
    "prior_yolo_sensitivity",
    "prior_faster_minus_yolo_gap",
    "current_faster_sensitivity",
    "current_yolo_sensitivity",
    "current_faster_minus_yolo_gap",
    "gap_change_current_minus_prior",
    "gap_change_classification",
    "direction_status",
)
PRIOR_RUN_COMPARISON_FIELDS = (
    "detector",
    "seed",
    "fp_per_image_budget",
    "prior_sensitivity",
    "current_sensitivity",
    "sensitivity_change_current_minus_prior",
    "sensitivity_change_status",
    "prior_candidate_floor_limited",
    "current_candidate_floor_limited",
)
FRONTIER_BOUND_FIELDS = (
    "detector",
    "fp_per_image_budget",
    "observed_aggregate_sensitivity",
    "conservative_upper_aggregate_sensitivity",
    "maximum_possible_increase",
    "candidate_floor_limited_run_count",
    "candidate_floor_limited_seeds",
    "observed_higher_detector",
    "ordering_can_theoretically_reverse",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen prediction and historical-grid inputs."""

    phase5_config: Path
    phase5_summary: Path
    test_annotations: Path
    historical_froc_config: Path
    historical_froc_summary: Path
    historical_curve_table: Path
    historical_operating_points_table: Path
    prior_exact_summary: Path | None = None
    prior_operating_points_per_seed_table: Path | None = None
    prior_operating_points_table: Path | None = None


class AnalysisSettings(StrictModel):
    """Configured exact-score evaluator contract."""

    expected_seeds: tuple[int, ...]
    class_names: dict[int, str]
    candidate_score_floor: float = Field(ge=0, le=1)
    historical_candidate_score_floor: float | None = Field(default=None, ge=0, le=1)
    match_iou_threshold: float = Field(gt=0, le=1)
    nms_iou_threshold: float = Field(gt=0, le=1)
    max_detections_per_image: int = Field(gt=0)
    fp_per_image_budgets: tuple[float, ...]
    numeric_tolerance: float = Field(gt=0)
    score_comparison: Literal["greater_than_or_equal"]
    include_upper_empty_sentinel: bool
    include_candidate_floor_endpoint: bool

    @model_validator(mode="after")
    def validate_contract(self) -> AnalysisSettings:
        if (
            not self.expected_seeds
            or tuple(sorted(set(self.expected_seeds))) != self.expected_seeds
        ):
            raise ValueError("expected_seeds must be unique and increasing")
        if not self.class_names or any(not value for value in self.class_names.values()):
            raise ValueError("class_names must contain non-empty configured labels")
        if (
            not self.fp_per_image_budgets
            or tuple(sorted(set(self.fp_per_image_budgets))) != self.fp_per_image_budgets
            or any(value <= 0 for value in self.fp_per_image_budgets)
        ):
            raise ValueError("FP/image budgets must be positive, unique, and increasing")
        if not self.include_upper_empty_sentinel or not self.include_candidate_floor_endpoint:
            raise ValueError("both exact-score endpoint sentinels are required")
        return self


class InferenceSensitivityProposal(StrictModel):
    """Review-only lower-floor proposal; this command never performs inference."""

    proposed_candidate_score_floor: float = Field(ge=0, le=1)
    requires_user_approval: bool
    selection_basis: str = Field(min_length=1)
    expected_images_per_model: int = Field(gt=0)
    expected_model_count: int = Field(gt=0)
    expected_artifacts: tuple[str, ...] = Field(min_length=1)


class DetectorSettings(StrictModel):
    """Display settings for one configured detector."""

    label: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class PlotSettings(StrictModel):
    """Deterministic comparison-figure settings."""

    dpi: int = Field(ge=72, le=600)
    figure_width: float = Field(gt=0)
    figure_height: float = Field(gt=0)
    minimum_x: float = Field(gt=0)
    maximum_x: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_extent(self) -> PlotSettings:
        if self.maximum_x <= self.minimum_x:
            raise ValueError("plot maximum_x must exceed minimum_x")
        return self


class OutputSettings(StrictModel):
    """Versioned exact-score artifacts."""

    exact_per_seed_table: Path
    exact_aggregate_curve_table: Path
    operating_points_per_seed_table: Path
    operating_points_table: Path
    historical_comparison_table: Path
    figure: Path
    summary_json: Path
    prior_comparison_table: Path | None = None
    prior_run_comparison_table: Path | None = None
    frontier_bounds_table: Path | None = None


class ExactFrocConfig(StrictModel):
    """Strict Batch 42 exact-score FROC contract."""

    schema_version: Literal[2, 3, 4]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    analysis: AnalysisSettings
    inference_sensitivity_proposal: InferenceSensitivityProposal | None = None
    detectors: dict[str, DetectorSettings]
    plot: PlotSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_top_level(self) -> ExactFrocConfig:
        if len(self.detectors) != 2:
            raise ValueError("exact-score FROC must declare exactly two detectors")
        proposal = self.inference_sensitivity_proposal
        if self.schema_version == 2 and proposal is None:
            raise ValueError("schema v2 requires a lower-floor inference proposal")
        if self.schema_version == 3 and proposal is not None:
            raise ValueError("schema v3 records completed inference and must omit a proposal")
        if self.schema_version == 4:
            if proposal is not None:
                raise ValueError("schema v4 records completed inference and must omit a proposal")
            required_inputs = (
                self.inputs.prior_exact_summary,
                self.inputs.prior_operating_points_per_seed_table,
                self.inputs.prior_operating_points_table,
            )
            required_outputs = (
                self.outputs.prior_comparison_table,
                self.outputs.prior_run_comparison_table,
                self.outputs.frontier_bounds_table,
            )
            if any(value is None for value in (*required_inputs, *required_outputs)):
                raise ValueError("schema v4 requires prior-comparison and frontier-bound paths")
        if proposal is not None:
            if proposal.proposed_candidate_score_floor >= self.analysis.candidate_score_floor:
                raise ValueError("proposed inference floor must be lower than the frozen floor")
            if not proposal.requires_user_approval:
                raise ValueError("lower-floor inference proposal must require user approval")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_exact_froc_config(path: str | Path) -> ExactFrocConfig:
    """Load and strictly validate the exact-score YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exact-score FROC config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return ExactFrocConfig.model_validate(payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected gzip JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required exact-score FROC input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required exact-score FROC input has no rows: {path}")
    return rows


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


def _resolved_input(config: ExactFrocConfig, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.resolve(path)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("sample standard deviation requires at least two runs")
    return 0.0 if np.all(array == array[0]) else float(np.std(array, ddof=1))


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["sensitivity"]),
        -float(row["fp_per_image"]),
        float(row["threshold"]),
    )


def load_exact_inputs(
    config: ExactFrocConfig,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[ImageTarget], dict[int, str]]:
    """Hash-check all frozen bundles and the unchanged evaluator contract."""

    phase5_path = config.resolve(config.inputs.phase5_config)
    summary_path = config.resolve(config.inputs.phase5_summary)
    annotation_path = config.resolve(config.inputs.test_annotations)
    phase5 = load_phase5_config(phase5_path)
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Phase 5 summary is not complete")
    if summary.get("config_sha256") != sha256_file(phase5_path):
        raise ValueError("Phase 5 config hash differs from its frozen summary")
    if summary.get("experiment_id") != phase5.experiment_id:
        raise ValueError("Phase 5 experiment identity differs from its frozen summary")
    if summary.get("test_annotation_sha256") != sha256_file(annotation_path):
        raise ValueError("test annotations differ from the Phase 5 source")
    if summary.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("Phase 5 evaluator settings differ from the frozen summary")

    analysis = config.analysis
    expected_evaluation = {
        "coco_minimum_score": analysis.candidate_score_floor,
        "match_iou_threshold": analysis.match_iou_threshold,
        "nms_iou_threshold": analysis.nms_iou_threshold,
        "max_detections": analysis.max_detections_per_image,
    }
    for field, expected_value in expected_evaluation.items():
        if getattr(phase5.evaluation, field) != expected_value:
            raise ValueError(f"configured {field} differs from the frozen Phase 5 contract")
    if phase5.seeds != analysis.expected_seeds:
        raise ValueError("configured exact-score seeds differ from the frozen Phase 5 seeds")
    if set(config.detectors) != {run.detector for run in phase5.runs}:
        raise ValueError("configured detector set differs from the frozen Phase 5 runs")

    targets, category_names = load_coco_targets(annotation_path)
    if category_names != analysis.class_names:
        raise ValueError("configured class contract differs from the frozen annotations")
    expected = {
        (detector, seed) for detector in config.detectors for seed in analysis.expected_seeds
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
        bundle_path = _resolved_input(config, comparison["prediction_bundle"])
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
        if payload.get("annotation_sha256") != summary["test_annotation_sha256"]:
            raise ValueError(f"Phase 5 bundle annotation mismatch: {bundle_path}")
        if payload.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
            raise ValueError(f"Phase 5 bundle evaluator mismatch: {bundle_path}")
        if payload.get("checkpoint_sha256") != comparison.get("checkpoint_sha256"):
            raise ValueError(f"Phase 5 bundle checkpoint mismatch: {bundle_path}")
        predictions = _deserialize_predictions(payload)
        if len(predictions) != len(targets):
            raise ValueError(f"Phase 5 bundle image count mismatch: {bundle_path}")
        all_scores = [item.scores for item in predictions if len(item.scores)]
        if (
            all_scores
            and min(float(np.min(item)) for item in all_scores) < analysis.candidate_score_floor
        ):
            raise ValueError(
                f"Phase 5 bundle contains a score below its frozen floor: {bundle_path}"
            )
        bundles.append(
            {
                "detector": detector,
                "seed": seed,
                "path": bundle_path,
                "sha256": expected_hash,
                "payload": payload,
                "predictions": predictions,
            }
        )
    if seen != expected:
        raise ValueError("Phase 5 summary lacks the complete configured detector/seed grid")
    bundles.sort(key=lambda item: (str(item["detector"]), int(item["seed"])))
    proposal = config.inference_sensitivity_proposal
    if proposal is not None:
        if len(targets) != proposal.expected_images_per_model:
            raise ValueError("proposed inference image count differs from the frozen test split")
        if len(bundles) != proposal.expected_model_count:
            raise ValueError("proposed inference model count differs from the frozen run grid")
    return phase5, summary, bundles, targets, category_names


def load_historical_grid(
    config: ExactFrocConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate the frozen five-run grid FROC inputs used for comparison."""

    froc_config_path = config.resolve(config.inputs.historical_froc_config)
    summary_path = config.resolve(config.inputs.historical_froc_summary)
    curve_path = config.resolve(config.inputs.historical_curve_table)
    operating_path = config.resolve(config.inputs.historical_operating_points_table)
    historical_config = load_froc_config(froc_config_path)
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("historical FROC summary is not complete")
    if summary.get("config_sha256") != sha256_file(froc_config_path):
        raise ValueError("historical FROC config hash mismatch")
    if historical_config.analysis.fp_per_image_budgets != config.analysis.fp_per_image_budgets:
        raise ValueError("historical and exact-score FROC budgets differ")
    if set(historical_config.detectors) != set(config.detectors):
        raise ValueError("historical and exact-score detector sets differ")
    for artifact_name, path in (
        ("curve_table", curve_path),
        ("operating_points_table", operating_path),
    ):
        if summary.get("artifacts", {}).get(artifact_name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"historical FROC {artifact_name} hash mismatch")
    if int(summary.get("counts", {}).get("seeds_per_detector", 0)) != len(
        config.analysis.expected_seeds
    ):
        raise ValueError("historical FROC run count differs from exact-score scope")

    threshold_summary_path = historical_config.resolve(historical_config.inputs.threshold_summary)
    threshold_summary = _read_json(threshold_summary_path)
    thresholds = threshold_summary.get("analysis", {}).get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("historical threshold summary lacks its configured grid")
    curve_rows = [
        {
            **row,
            "threshold": float(row["threshold"]),
            "seed_count": int(row["seed_count"]),
            "sensitivity": float(row["sensitivity"]),
            "sensitivity_std": float(row["sensitivity_std"]),
            "fp_per_image": float(row["fp_per_image"]),
            "fp_per_image_std": float(row["fp_per_image_std"]),
        }
        for row in _read_csv(curve_path)
    ]
    operating_rows = [
        {
            **row,
            "fp_per_image_budget": float(row["fp_per_image_budget"]),
            "seed_count": int(row["seed_count"]),
            "sensitivity": float(row["sensitivity"]),
            "sensitivity_std": float(row["sensitivity_std"]),
            "achieved_fp_per_image": float(row["achieved_fp_per_image"]),
        }
        for row in _read_csv(operating_path)
    ]
    grid = {
        "threshold_count": len(thresholds),
        "lower_limit": float(min(thresholds)),
        "upper_limit": float(max(thresholds)),
        "thresholds": [float(value) for value in thresholds],
    }
    return curve_rows, operating_rows, summary, grid


def exact_score_froc_rows(
    predictions: Sequence[ImagePrediction],
    targets: Sequence[ImageTarget],
    *,
    detector: str,
    seed: int,
    class_ids: tuple[int, ...],
    candidate_score_floor: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every distinct retained score using one equivalent greedy-match pass."""

    prediction_map = {item.image_id: item for item in predictions}
    target_map = {item.image_id: item for item in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("exact-score FROC requires unique image identifiers")
    if set(prediction_map) != set(target_map):
        raise ValueError("prediction and target image identifiers differ")
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must be non-empty and unique")

    allowed = set(class_ids)
    events: defaultdict[float, list[int]] = defaultdict(lambda: [0, 0])
    target_count = 0
    raw_emitted_count = 0
    evaluated_prediction_count = 0
    for image_id in sorted(target_map):
        prediction = prediction_map[image_id]
        target = target_map[image_id]
        if not set(prediction.labels).issubset(allowed) or not set(target.labels).issubset(allowed):
            raise ValueError("record contains a label outside the configured class contract")
        if len(prediction.scores) and float(np.min(prediction.scores)) < candidate_score_floor:
            raise ValueError("prediction record contains a score below the candidate floor")
        result = match_image(
            prediction,
            target,
            score_threshold=candidate_score_floor,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        matched = {match.prediction_index for match in result.matches}
        for prediction_index in result.prediction_indices:
            score = float(prediction.scores[prediction_index])
            events[score][0] += 1
            events[score][1] += int(prediction_index in matched)
        target_count += len(target.boxes_xyxy)
        raw_emitted_count += len(prediction.scores)
        evaluated_prediction_count += len(result.prediction_indices)
    if target_count <= 0:
        raise ValueError("FROC sensitivity requires at least one target")

    maximum_score = max(events, default=candidate_score_floor)
    upper_sentinel = float(np.nextafter(maximum_score, np.inf))
    rows: list[dict[str, Any]] = [
        {
            "detector": detector,
            "seed": seed,
            "threshold_relaxation_rank": 0,
            "threshold": upper_sentinel,
            "threshold_kind": "upper_empty_sentinel",
            "is_candidate_floor_boundary": False,
            "sensitivity": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": target_count,
            "prediction_count": 0,
            "target_count": target_count,
            "fp_per_image": 0.0,
        }
    ]
    cumulative_predictions = 0
    cumulative_true_positives = 0
    image_count = len(targets)
    for score in sorted(events, reverse=True):
        event_predictions, event_true_positives = events[score]
        cumulative_predictions += event_predictions
        cumulative_true_positives += event_true_positives
        false_positives = cumulative_predictions - cumulative_true_positives
        is_floor = score == candidate_score_floor
        rows.append(
            {
                "detector": detector,
                "seed": seed,
                "threshold_relaxation_rank": len(rows),
                "threshold": score,
                "threshold_kind": (
                    "candidate_floor_emitted_score" if is_floor else "emitted_score"
                ),
                "is_candidate_floor_boundary": is_floor,
                "sensitivity": cumulative_true_positives / target_count,
                "true_positives": cumulative_true_positives,
                "false_positives": false_positives,
                "false_negatives": target_count - cumulative_true_positives,
                "prediction_count": cumulative_predictions,
                "target_count": target_count,
                "fp_per_image": false_positives / image_count,
            }
        )
    if rows[-1]["threshold"] != candidate_score_floor:
        floor_row = dict(rows[-1])
        floor_row.update(
            {
                "threshold_relaxation_rank": len(rows),
                "threshold": candidate_score_floor,
                "threshold_kind": "candidate_floor_sentinel",
                "is_candidate_floor_boundary": True,
            }
        )
        rows.append(floor_row)

    audit_exact_score_monotonicity(rows, tolerance=0.0)
    emitted_scores = [score for score in events]
    run_summary = {
        "detector": detector,
        "seed": seed,
        "images": image_count,
        "targets": target_count,
        "raw_emitted_predictions": raw_emitted_count,
        "evaluated_predictions_after_image_cap": evaluated_prediction_count,
        "unique_evaluated_scores": len(events),
        "minimum_emitted_score": min(emitted_scores) if emitted_scores else None,
        "maximum_emitted_score": max(emitted_scores) if emitted_scores else None,
        "upper_empty_sentinel": upper_sentinel,
        "candidate_score_floor": candidate_score_floor,
        "candidate_floor_endpoint_sensitivity": float(rows[-1]["sensitivity"]),
        "candidate_floor_endpoint_fp_per_image": float(rows[-1]["fp_per_image"]),
        "threshold_rows": len(rows),
        "monotonic_sensitivity": True,
        "monotonic_fp_per_image": True,
    }
    return rows, run_summary


def audit_exact_score_monotonicity(rows: Sequence[Mapping[str, Any]], *, tolerance: float) -> None:
    """Reject threshold-order, endpoint, sensitivity, or FP monotonicity failures."""

    if len(rows) < 2:
        raise ValueError("exact-score frontier must contain both endpoint sentinels")
    thresholds = [float(row["threshold"]) for row in rows]
    if any(left <= right for left, right in pairwise(thresholds)):
        raise ValueError("exact-score thresholds are not strictly descending")
    first = rows[0]
    if (
        first.get("threshold_kind") != "upper_empty_sentinel"
        or int(first["prediction_count"]) != 0
        or int(first["true_positives"]) != 0
        or int(first["false_positives"]) != 0
    ):
        raise ValueError("exact-score frontier lacks a valid empty upper endpoint")
    if not bool(rows[-1].get("is_candidate_floor_boundary")):
        raise ValueError("exact-score frontier lacks a candidate-floor endpoint")
    sensitivity = [float(row["sensitivity"]) for row in rows]
    fp_per_image = [float(row["fp_per_image"]) for row in rows]
    if any(right + tolerance < left for left, right in pairwise(sensitivity)):
        raise ValueError("sensitivity decreased as the exact threshold relaxed")
    if any(right + tolerance < left for left, right in pairwise(fp_per_image)):
        raise ValueError("FP/image decreased as the exact threshold relaxed")


def select_exact_operating_points(
    rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
    *,
    candidate_score_floor: float,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select per-run exact-score points and aggregate equally across runs."""

    grouped: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["detector"]), int(row["seed"]))].append(row)
    per_seed: list[dict[str, Any]] = []
    for (detector, seed), run_rows in sorted(grouped.items()):
        floor_rows = [row for row in run_rows if bool(row["is_candidate_floor_boundary"])]
        if len(floor_rows) != 1:
            raise ValueError(f"expected one candidate-floor endpoint for {detector} seed {seed}")
        floor_row = floor_rows[0]
        for budget in budgets:
            candidates = [
                row for row in run_rows if float(row["fp_per_image"]) <= budget + tolerance
            ]
            if not candidates:
                raise ValueError(f"no exact-score point below {budget} FP/image")
            selected = max(candidates, key=_selection_key)
            floor_limited = float(floor_row["fp_per_image"]) < budget - tolerance
            per_seed.append(
                {
                    "detector": detector,
                    "seed": seed,
                    "fp_per_image_budget": float(budget),
                    "sensitivity": float(selected["sensitivity"]),
                    "achieved_fp_per_image": float(selected["fp_per_image"]),
                    "selected_threshold": float(selected["threshold"]),
                    "selected_threshold_kind": str(selected["threshold_kind"]),
                    "selected_at_candidate_floor_boundary": bool(
                        selected["is_candidate_floor_boundary"]
                    ),
                    "candidate_score_floor": candidate_score_floor,
                    "candidate_floor_endpoint_sensitivity": float(floor_row["sensitivity"]),
                    "candidate_floor_endpoint_fp_per_image": float(floor_row["fp_per_image"]),
                    "candidate_floor_limited": floor_limited,
                }
            )

    selected_groups: defaultdict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        selected_groups[(str(row["detector"]), float(row["fp_per_image_budget"]))].append(row)
    aggregate: list[dict[str, Any]] = []
    selection_rule = (
        "per run: maximum observed exact-score sensitivity with FP/image <= budget; "
        "no interpolation; ties use fewer FP/image then higher threshold"
    )
    for (detector, budget), group in sorted(selected_groups.items()):
        sensitivities = [float(row["sensitivity"]) for row in group]
        achieved = [float(row["achieved_fp_per_image"]) for row in group]
        thresholds = [float(row["selected_threshold"]) for row in group]
        limited_seeds = sorted(
            int(row["seed"]) for row in group if bool(row["candidate_floor_limited"])
        )
        sensitivity_mean = float(np.mean(sensitivities))
        sensitivity_std = _sample_std(sensitivities)
        aggregate.append(
            {
                "detector": detector,
                "fp_per_image_budget": budget,
                "seed_count": len(group),
                "sensitivity": sensitivity_mean,
                "sensitivity_std": sensitivity_std,
                "sensitivity_mean_plus_minus_std": (
                    f"{sensitivity_mean:.6g} +/- {sensitivity_std:.6g}"
                ),
                "achieved_fp_per_image": float(np.mean(achieved)),
                "achieved_fp_per_image_std": _sample_std(achieved),
                "selected_threshold_mean": float(np.mean(thresholds)),
                "selected_threshold_min": min(thresholds),
                "selected_threshold_max": max(thresholds),
                "selected_at_candidate_floor_count": sum(
                    bool(row["selected_at_candidate_floor_boundary"]) for row in group
                ),
                "candidate_floor_limited_run_count": len(limited_seeds),
                "candidate_floor_limited_seeds": ";".join(map(str, limited_seeds)),
                "selection_rule": selection_rule,
            }
        )
    return aggregate, per_seed


def aggregate_exact_frontier(
    rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Evaluate the equal-run exact frontier on the union of observed FP coordinates."""

    by_detector: defaultdict[str, dict[int, list[Mapping[str, Any]]]] = defaultdict(dict)
    for row in rows:
        detector = str(row["detector"])
        seed = int(row["seed"])
        by_detector.setdefault(detector, {}).setdefault(seed, []).append(row)

    result: list[dict[str, Any]] = []
    for detector, seed_rows in sorted(by_detector.items()):
        for run_rows in seed_rows.values():
            run_rows.sort(key=lambda row: int(row["threshold_relaxation_rank"]))
        queries = sorted(
            {
                0.0,
                *(float(value) for value in budgets),
                *(
                    float(row["fp_per_image"])
                    for run_rows in seed_rows.values()
                    for row in run_rows
                ),
            }
        )
        states = {
            seed: {"fp": [float(row["fp_per_image"]) for row in run_rows], "rows": run_rows}
            for seed, run_rows in seed_rows.items()
        }
        floor_fp = {
            seed: float(
                next(row for row in run_rows if bool(row["is_candidate_floor_boundary"]))[
                    "fp_per_image"
                ]
            )
            for seed, run_rows in seed_rows.items()
        }
        for query in queries:
            selected: list[Mapping[str, Any]] = []
            limited: list[int] = []
            for seed, state in states.items():
                end = bisect_right(state["fp"], query + tolerance)
                candidates = state["rows"][:end]
                if not candidates:
                    raise ValueError("upper empty endpoint is missing from an exact frontier")
                selected.append(max(candidates, key=_selection_key))
                if floor_fp[seed] < query - tolerance:
                    limited.append(seed)
            sensitivities = [float(row["sensitivity"]) for row in selected]
            achieved = [float(row["fp_per_image"]) for row in selected]
            result.append(
                {
                    "detector": detector,
                    "fp_per_image_query": query,
                    "seed_count": len(selected),
                    "sensitivity": float(np.mean(sensitivities)),
                    "sensitivity_std": _sample_std(sensitivities),
                    "achieved_fp_per_image": float(np.mean(achieved)),
                    "achieved_fp_per_image_std": _sample_std(achieved),
                    "candidate_floor_limited_run_count": len(limited),
                    "candidate_floor_limited_seeds": ";".join(map(str, sorted(limited))),
                }
            )
    return result


def compare_historical_and_exact(
    historical: Sequence[Mapping[str, Any]],
    exact: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compare detector gaps at the prespecified budgets without changing them."""

    old = {(str(row["detector"]), float(row["fp_per_image_budget"])): row for row in historical}
    new = {(str(row["detector"]), float(row["fp_per_image_budget"])): row for row in exact}
    result: list[dict[str, Any]] = []
    for budget in budgets:
        old_faster = old[("faster_rcnn", float(budget))]
        old_yolo = old[("yolo11s", float(budget))]
        new_faster = new[("faster_rcnn", float(budget))]
        new_yolo = new[("yolo11s", float(budget))]
        historical_gap = float(old_faster["sensitivity"]) - float(old_yolo["sensitivity"])
        exact_gap = float(new_faster["sensitivity"]) - float(new_yolo["sensitivity"])
        change = exact_gap - historical_gap
        if change > tolerance:
            change_classification = "strengthened"
        elif change < -tolerance:
            change_classification = "weakened"
        else:
            change_classification = "unchanged"
        if historical_gap > tolerance and exact_gap > tolerance:
            direction = "unchanged_faster_higher"
        elif historical_gap < -tolerance and exact_gap < -tolerance:
            direction = "unchanged_yolo_higher"
        elif historical_gap * exact_gap < -(tolerance**2):
            direction = "reversed"
        else:
            direction = "changed_to_or_from_tie"
        result.append(
            {
                "fp_per_image_budget": float(budget),
                "historical_faster_sensitivity": float(old_faster["sensitivity"]),
                "historical_yolo_sensitivity": float(old_yolo["sensitivity"]),
                "historical_faster_minus_yolo_gap": historical_gap,
                "exact_faster_sensitivity": float(new_faster["sensitivity"]),
                "exact_yolo_sensitivity": float(new_yolo["sensitivity"]),
                "exact_faster_minus_yolo_gap": exact_gap,
                "gap_change_exact_minus_historical": change,
                "gap_change_classification": change_classification,
                "direction_status": direction,
                "faster_candidate_floor_limited_run_count": int(
                    new_faster["candidate_floor_limited_run_count"]
                ),
                "faster_candidate_floor_limited_seeds": str(
                    new_faster["candidate_floor_limited_seeds"]
                ),
                "yolo_candidate_floor_limited_run_count": int(
                    new_yolo["candidate_floor_limited_run_count"]
                ),
                "yolo_candidate_floor_limited_seeds": str(
                    new_yolo["candidate_floor_limited_seeds"]
                ),
            }
        )
    return result


def load_prior_exact_operating_points(
    config: ExactFrocConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Hash-check the immediately preceding exact-score operating evidence."""

    values = (
        config.inputs.prior_exact_summary,
        config.inputs.prior_operating_points_per_seed_table,
        config.inputs.prior_operating_points_table,
    )
    if any(value is None for value in values):
        raise ValueError("prior exact-score inputs are not configured")
    summary_path = config.resolve(Path(values[0]))
    per_seed_path = config.resolve(Path(values[1]))
    aggregate_path = config.resolve(Path(values[2]))
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("prior exact-score summary is not complete")
    if summary.get("protocol", {}).get("fp_per_image_budgets") != list(
        config.analysis.fp_per_image_budgets
    ):
        raise ValueError("prior and current exact-score budgets differ")
    for artifact_name, path in (
        ("operating_points_per_seed_table", per_seed_path),
        ("operating_points_table", aggregate_path),
    ):
        if summary.get("artifacts", {}).get(artifact_name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"prior exact-score {artifact_name} hash mismatch")

    per_seed = [
        {
            **row,
            "seed": int(row["seed"]),
            "fp_per_image_budget": float(row["fp_per_image_budget"]),
            "sensitivity": float(row["sensitivity"]),
            "candidate_floor_limited": str(row["candidate_floor_limited"]).lower() == "true",
        }
        for row in _read_csv(per_seed_path)
    ]
    aggregate = [
        {
            **row,
            "fp_per_image_budget": float(row["fp_per_image_budget"]),
            "sensitivity": float(row["sensitivity"]),
        }
        for row in _read_csv(aggregate_path)
    ]
    return per_seed, aggregate, summary


def compare_prior_and_current_exact(
    prior: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Classify how the detector gap changed from the preceding exact frontier."""

    old = {(str(row["detector"]), float(row["fp_per_image_budget"])): row for row in prior}
    new = {(str(row["detector"]), float(row["fp_per_image_budget"])): row for row in current}
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        prior_faster = float(old[("faster_rcnn", float(budget))]["sensitivity"])
        prior_yolo = float(old[("yolo11s", float(budget))]["sensitivity"])
        current_faster = float(new[("faster_rcnn", float(budget))]["sensitivity"])
        current_yolo = float(new[("yolo11s", float(budget))]["sensitivity"])
        prior_gap = prior_faster - prior_yolo
        current_gap = current_faster - current_yolo
        if prior_gap * current_gap < -(tolerance**2):
            classification = "reversed"
            direction = "reversed"
        elif prior_gap > tolerance and current_gap > tolerance:
            change = abs(current_gap) - abs(prior_gap)
            classification = (
                "strengthened"
                if change > tolerance
                else "weakened"
                if change < -tolerance
                else "unchanged"
            )
            direction = "unchanged_faster_higher"
        elif prior_gap < -tolerance and current_gap < -tolerance:
            change = abs(current_gap) - abs(prior_gap)
            classification = (
                "strengthened"
                if change > tolerance
                else "weakened"
                if change < -tolerance
                else "unchanged"
            )
            direction = "unchanged_yolo_higher"
        else:
            classification = (
                "unchanged" if abs(current_gap - prior_gap) <= tolerance else "tie_changed"
            )
            direction = "changed_to_or_from_tie"
        rows.append(
            {
                "fp_per_image_budget": float(budget),
                "prior_faster_sensitivity": prior_faster,
                "prior_yolo_sensitivity": prior_yolo,
                "prior_faster_minus_yolo_gap": prior_gap,
                "current_faster_sensitivity": current_faster,
                "current_yolo_sensitivity": current_yolo,
                "current_faster_minus_yolo_gap": current_gap,
                "gap_change_current_minus_prior": current_gap - prior_gap,
                "gap_change_classification": classification,
                "direction_status": direction,
            }
        )
    return rows


def compare_prior_and_current_runs(
    prior: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Record every run-budget sensitivity change from the prior frontier."""

    old = {
        (str(row["detector"]), int(row["seed"]), float(row["fp_per_image_budget"])): row
        for row in prior
    }
    rows: list[dict[str, Any]] = []
    for row in current:
        key = str(row["detector"]), int(row["seed"]), float(row["fp_per_image_budget"])
        previous = old[key]
        change = float(row["sensitivity"]) - float(previous["sensitivity"])
        status = (
            "increased"
            if change > tolerance
            else "decreased"
            if change < -tolerance
            else "unchanged"
        )
        rows.append(
            {
                "detector": key[0],
                "seed": key[1],
                "fp_per_image_budget": key[2],
                "prior_sensitivity": float(previous["sensitivity"]),
                "current_sensitivity": float(row["sensitivity"]),
                "sensitivity_change_current_minus_prior": change,
                "sensitivity_change_status": status,
                "prior_candidate_floor_limited": bool(previous["candidate_floor_limited"]),
                "current_candidate_floor_limited": bool(row["candidate_floor_limited"]),
            }
        )
    return rows


def conservative_frontier_bounds(
    aggregate: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
    detectors: Sequence[str],
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bound missing aggregate sensitivity by assigning every limited run sensitivity one."""

    aggregate_map = {
        (str(row["detector"]), float(row["fp_per_image_budget"])): row for row in aggregate
    }
    bound_map: dict[tuple[str, float], dict[str, Any]] = {}
    for detector in detectors:
        for budget in budgets:
            selected = [
                row
                for row in per_seed
                if str(row["detector"]) == detector
                and float(row["fp_per_image_budget"]) == float(budget)
            ]
            if not selected:
                raise ValueError(f"missing per-seed operating points for {detector} at {budget}")
            limited = [row for row in selected if bool(row["candidate_floor_limited"])]
            observed = float(aggregate_map[(detector, float(budget))]["sensitivity"])
            upper = float(
                np.mean(
                    [
                        1.0 if bool(row["candidate_floor_limited"]) else float(row["sensitivity"])
                        for row in selected
                    ]
                )
            )
            bound_map[(detector, float(budget))] = {
                "observed": observed,
                "upper": upper,
                "limited": limited,
            }

    rows: list[dict[str, Any]] = []
    ordering: list[dict[str, Any]] = []
    for budget in budgets:
        states = {detector: bound_map[(detector, float(budget))] for detector in detectors}
        ranked = sorted(detectors, key=lambda detector: states[detector]["observed"], reverse=True)
        higher, lower = ranked[0], ranked[1]
        can_reverse = states[lower]["upper"] > states[higher]["observed"] + tolerance
        ordering.append(
            {
                "fp_per_image_budget": float(budget),
                "observed_higher_detector": higher,
                "observed_gap": states[higher]["observed"] - states[lower]["observed"],
                "lower_detector_conservative_upper_sensitivity": states[lower]["upper"],
                "minimum_possible_gap_preserving_observed_order": (
                    states[higher]["observed"] - states[lower]["upper"]
                ),
                "ordering_can_theoretically_reverse": can_reverse,
            }
        )
        for detector in detectors:
            state = states[detector]
            limited = state["limited"]
            rows.append(
                {
                    "detector": detector,
                    "fp_per_image_budget": float(budget),
                    "observed_aggregate_sensitivity": state["observed"],
                    "conservative_upper_aggregate_sensitivity": state["upper"],
                    "maximum_possible_increase": state["upper"] - state["observed"],
                    "candidate_floor_limited_run_count": len(limited),
                    "candidate_floor_limited_seeds": ";".join(
                        str(row["seed"])
                        for row in sorted(limited, key=lambda item: int(item["seed"]))
                    ),
                    "observed_higher_detector": higher,
                    "ordering_can_theoretically_reverse": can_reverse,
                }
            )
    return rows, ordering


def build_run_reachability(
    run_summaries: Sequence[Mapping[str, Any]],
    operating_per_seed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize endpoint reachability and all configured budgets for every run."""

    endpoints = {(str(row["detector"]), int(row["seed"])): row for row in run_summaries}
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in operating_per_seed:
        grouped[(str(row["detector"]), int(row["seed"]))].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(endpoints):
        endpoint = endpoints[key]
        result.append(
            {
                "detector": key[0],
                "seed": key[1],
                "maximum_reachable_fp_per_image": float(
                    endpoint["candidate_floor_endpoint_fp_per_image"]
                ),
                "candidate_floor_endpoint_sensitivity": float(
                    endpoint["candidate_floor_endpoint_sensitivity"]
                ),
                "budgets": [
                    {
                        "fp_per_image_budget": float(row["fp_per_image_budget"]),
                        "sensitivity": float(row["sensitivity"]),
                        "achieved_fp_per_image": float(row["achieved_fp_per_image"]),
                        "status": (
                            "floor_limited_lower_bound"
                            if bool(row["candidate_floor_limited"])
                            else "observed_within_reachable_frontier"
                        ),
                    }
                    for row in sorted(
                        grouped[key], key=lambda item: float(item["fp_per_image_budget"])
                    )
                ],
            }
        )
    return result


def audit_selected_points_against_canonical_matcher(
    bundles: Sequence[Mapping[str, Any]],
    targets: list[ImageTarget],
    operating_per_seed: Sequence[Mapping[str, Any]],
    *,
    class_ids: tuple[int, ...],
    iou_threshold: float,
    max_detections: int,
    tolerance: float,
) -> int:
    """Cross-check every reported budget point through the canonical evaluator."""

    bundle_map = {(str(bundle["detector"]), int(bundle["seed"])): bundle for bundle in bundles}
    checked = 0
    for row in operating_per_seed:
        key = str(row["detector"]), int(row["seed"])
        bundle = bundle_map[key]
        overall = evaluate_operating_point(
            bundle["predictions"],
            targets,
            class_ids=class_ids,
            score_threshold=float(row["selected_threshold"]),
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )["overall"]
        if not np.isclose(
            float(overall["recall"]),
            float(row["sensitivity"]),
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(f"exact-score sensitivity cross-check failed for {key}")
        achieved_fp = int(overall["fp"]) / len(targets)
        if not np.isclose(
            achieved_fp,
            float(row["achieved_fp_per_image"]),
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(f"exact-score FP/image cross-check failed for {key}")
        checked += 1
    return checked


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
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
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_figure(path: Path, figure: Any, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def plot_exact_froc_comparison(
    exact_rows: Sequence[Mapping[str, Any]],
    aggregate_curve_rows: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]],
    historical_curve_rows: Sequence[Mapping[str, Any]],
    config: ExactFrocConfig,
) -> Path:
    """Plot the historical grid and observed exact-score frontier."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure, axis = plt.subplots(
        figsize=(config.plot.figure_width, config.plot.figure_height), constrained_layout=True
    )
    try:
        for detector, settings in config.detectors.items():
            historical = sorted(
                (
                    row
                    for row in historical_curve_rows
                    if row["detector"] == detector and float(row["fp_per_image"]) > 0
                ),
                key=lambda row: float(row["fp_per_image"]),
            )
            axis.plot(
                [float(row["fp_per_image"]) for row in historical],
                [float(row["sensitivity"]) for row in historical],
                color=settings.color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.65,
                label=f"{settings.label}: historical 0.01-0.99 grid",
            )
            exact = sorted(
                (
                    row
                    for row in aggregate_curve_rows
                    if row["detector"] == detector
                    and config.plot.minimum_x
                    <= float(row["fp_per_image_query"])
                    <= config.plot.maximum_x
                ),
                key=lambda row: float(row["fp_per_image_query"]),
            )
            x = np.asarray([row["fp_per_image_query"] for row in exact], dtype=np.float64)
            sensitivity = np.asarray([row["sensitivity"] for row in exact], dtype=np.float64)
            sensitivity_std = np.asarray(
                [row["sensitivity_std"] for row in exact], dtype=np.float64
            )
            axis.step(
                x,
                sensitivity,
                where="post",
                color=settings.color,
                linewidth=2.2,
                label=f"{settings.label}: observed exact-score frontier",
            )
            axis.fill_between(
                x,
                np.clip(sensitivity - sensitivity_std, 0, 1),
                np.clip(sensitivity + sensitivity_std, 0, 1),
                step="post",
                color=settings.color,
                alpha=0.14,
                linewidth=0,
            )
            summaries = [row for row in operating_rows if row["detector"] == detector]
            axis.scatter(
                [row["fp_per_image_budget"] for row in summaries],
                [row["sensitivity"] for row in summaries],
                marker="D",
                s=42,
                facecolor="white",
                edgecolor=settings.color,
                linewidth=1.3,
                zorder=5,
            )
            floor_rows = [
                row
                for row in exact_rows
                if row["detector"] == detector and bool(row["is_candidate_floor_boundary"])
            ]
            axis.scatter(
                [row["fp_per_image"] for row in floor_rows],
                [row["sensitivity"] for row in floor_rows],
                marker="v",
                s=32,
                color=settings.color,
                alpha=0.55,
                zorder=4,
            )

        power_ticks = 2.0 ** np.arange(
            np.ceil(np.log2(config.plot.minimum_x)),
            np.floor(np.log2(config.plot.maximum_x)) + 1,
        )
        axis.set_xscale("log", base=2)
        axis.set_xlim(config.plot.minimum_x, config.plot.maximum_x)
        axis.set_ylim(0, 1)
        axis.set_xticks(power_ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        axis.set_xlabel("False positives per image (log₂ scale)")
        axis.set_ylabel(f"Sensitivity (recall at IoU = {config.analysis.match_iou_threshold:.2f})")
        axis.set_title("Historical grid versus observed exact-score FROC (5 runs/detector)")
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="lower right", fontsize=8.5)
        axis.text(
            0.02,
            0.98,
            "Solid/bands: equal-run exact-score mean +/- sample SD\n"
            "Diamonds: prespecified budgets; triangles: frozen candidate-floor endpoints\n"
            "Floor-limited budgets remain lower-bound observations",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
        return _atomic_figure(config.resolve(config.outputs.figure), figure, dpi=config.plot.dpi)
    finally:
        plt.close(figure)


def _candidate_floor_assessment(
    operating_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    config: ExactFrocConfig,
) -> dict[str, Any]:
    by_detector: dict[str, Any] = {}
    for detector in config.detectors:
        selected = [row for row in operating_rows if row["detector"] == detector]
        by_detector[detector] = {
            "floor_limited_budgets": [
                float(row["fp_per_image_budget"])
                for row in selected
                if int(row["candidate_floor_limited_run_count"]) > 0
            ],
            "all_runs_floor_limited_budgets": [
                float(row["fp_per_image_budget"])
                for row in selected
                if int(row["candidate_floor_limited_run_count"]) == int(row["seed_count"])
            ],
            "limited_runs_by_budget": {
                str(row["fp_per_image_budget"]): {
                    "count": int(row["candidate_floor_limited_run_count"]),
                    "seeds": str(row["candidate_floor_limited_seeds"]),
                }
                for row in selected
            },
        }
    limited = any(int(row["candidate_floor_limited_run_count"]) > 0 for row in operating_rows)
    proposal = config.inference_sensitivity_proposal
    if limited and proposal is not None:
        inference_status = "proposed_only_user_approval_required"
    elif limited:
        inference_status = "additional_inference_not_automatically_authorized"
    else:
        inference_status = "not_required"
    return {
        "observable_frontier_truncated_by_candidate_floor": limited,
        "candidate_score_floor": config.analysis.candidate_score_floor,
        "criterion": (
            "a run is floor-limited at a budget when its all-retained-candidate "
            "FP/image endpoint remains strictly below that budget"
        ),
        "by_detector": by_detector,
        "detector_gap_reversal_observed": any(
            row["direction_status"] == "reversed" for row in comparison_rows
        ),
        "lower_floor_checkpoint_inference_status": inference_status,
        "proposed_candidate_score_floor": (
            proposal.proposed_candidate_score_floor if limited and proposal is not None else None
        ),
        "proposal_selection_basis": proposal.selection_basis if proposal is not None else None,
        "performs_checkpoint_inference": False,
    }


def _inference_sensitivity_proposal(
    config: ExactFrocConfig,
    phase5: Any,
    bundles: Sequence[Mapping[str, Any]],
    *,
    image_count: int,
    needed: bool,
) -> dict[str, Any]:
    """Describe, but never execute, the approval-gated lower-floor inference."""

    proposal = config.inference_sensitivity_proposal
    checkpoint_by_run = {
        (run.detector, run.seed): config.resolve(run.checkpoint) for run in phase5.runs
    }
    models = [
        {
            "detector": str(bundle["detector"]),
            "seed": int(bundle["seed"]),
            "checkpoint": checkpoint_by_run[(str(bundle["detector"]), int(bundle["seed"]))]
            .relative_to(config.project_root)
            .as_posix(),
            "checkpoint_sha256": str(bundle["payload"]["checkpoint_sha256"]),
        }
        for bundle in bundles
    ]
    if proposal is None:
        return {
            "status": "completed_by_approved_upstream_inference",
            "executed": True,
            "requires_user_approval": False,
            "current_candidate_score_floor": config.analysis.candidate_score_floor,
            "proposed_candidate_score_floor": None,
            "models": models,
            "model_count": len(models),
            "images_per_model": image_count,
            "model_image_evaluations": len(models) * image_count,
        }
    return {
        "status": "proposed_only_user_approval_required" if needed else "not_required",
        "executed": False,
        "requires_user_approval": proposal.requires_user_approval,
        "current_candidate_score_floor": config.analysis.candidate_score_floor,
        "proposed_candidate_score_floor": proposal.proposed_candidate_score_floor,
        "selection_basis": proposal.selection_basis,
        "technical_support": {
            "shared_evaluation_config": "src/evaluate.py accepts coco_minimum_score in [0, 1]",
            "faster_rcnn_adapter": (
                "src/models/faster_rcnn_config.py accepts box_score_threshold in [0, 1]"
            ),
            "yolo11s_adapter": (
                "src/models/yolo_config.py accepts inference_minimum_score in [0, 1]"
            ),
        },
        "scope": {
            "models": models,
            "model_count": len(models),
            "images_per_model": image_count,
            "model_image_evaluations": len(models) * image_count,
            "annotations": config.inputs.test_annotations.as_posix(),
            "unchanged": [
                "test images",
                "annotations",
                "Lung Opacity class",
                "IoU 0.50 matching",
                "NMS IoU 0.50",
                "maximum 100 detections per image",
            ],
        },
        "expected_artifacts": list(proposal.expected_artifacts),
    }


def run_exact_froc(config: ExactFrocConfig) -> dict[str, Any]:
    """Generate versioned exact-score FROC artifacts without loading checkpoints."""

    phase5, phase5_summary, bundles, targets, category_names = load_exact_inputs(config)
    historical_curve, historical_operating, historical_summary, historical_grid = (
        load_historical_grid(config)
    )
    exact_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        rows, run_summary = exact_score_froc_rows(
            bundle["predictions"],
            targets,
            detector=str(bundle["detector"]),
            seed=int(bundle["seed"]),
            class_ids=tuple(sorted(category_names)),
            candidate_score_floor=config.analysis.candidate_score_floor,
            iou_threshold=config.analysis.match_iou_threshold,
            max_detections=config.analysis.max_detections_per_image,
        )
        exact_rows.extend(rows)
        run_summaries.append(run_summary)
    operating_rows, operating_per_seed = select_exact_operating_points(
        exact_rows,
        config.analysis.fp_per_image_budgets,
        candidate_score_floor=config.analysis.candidate_score_floor,
        tolerance=config.analysis.numeric_tolerance,
    )
    aggregate_curve = aggregate_exact_frontier(
        exact_rows,
        config.analysis.fp_per_image_budgets,
        tolerance=config.analysis.numeric_tolerance,
    )
    comparison_rows = compare_historical_and_exact(
        historical_operating,
        operating_rows,
        config.analysis.fp_per_image_budgets,
        tolerance=config.analysis.numeric_tolerance,
    )
    prior_operating: list[dict[str, Any]] = []
    prior_summary: dict[str, Any] | None = None
    prior_comparison_rows: list[dict[str, Any]] = []
    prior_run_comparison_rows: list[dict[str, Any]] = []
    frontier_bound_rows: list[dict[str, Any]] = []
    ordering_bounds: list[dict[str, Any]] = []
    if config.schema_version == 4:
        prior_per_seed, prior_operating, prior_summary = load_prior_exact_operating_points(config)
        prior_comparison_rows = compare_prior_and_current_exact(
            prior_operating,
            operating_rows,
            config.analysis.fp_per_image_budgets,
            tolerance=config.analysis.numeric_tolerance,
        )
        prior_run_comparison_rows = compare_prior_and_current_runs(
            prior_per_seed,
            operating_per_seed,
            tolerance=config.analysis.numeric_tolerance,
        )
        frontier_bound_rows, ordering_bounds = conservative_frontier_bounds(
            operating_rows,
            operating_per_seed,
            config.analysis.fp_per_image_budgets,
            tuple(config.detectors),
            tolerance=config.analysis.numeric_tolerance,
        )
    canonical_crosschecks = audit_selected_points_against_canonical_matcher(
        bundles,
        targets,
        operating_per_seed,
        class_ids=tuple(sorted(category_names)),
        iou_threshold=config.analysis.match_iou_threshold,
        max_detections=config.analysis.max_detections_per_image,
        tolerance=config.analysis.numeric_tolerance,
    )

    exact_path = _atomic_csv(
        config.resolve(config.outputs.exact_per_seed_table), EXACT_PER_SEED_FIELDS, exact_rows
    )
    aggregate_curve_path = _atomic_csv(
        config.resolve(config.outputs.exact_aggregate_curve_table),
        EXACT_AGGREGATE_CURVE_FIELDS,
        aggregate_curve,
    )
    operating_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.operating_points_per_seed_table),
        OPERATING_PER_SEED_FIELDS,
        operating_per_seed,
    )
    operating_path = _atomic_csv(
        config.resolve(config.outputs.operating_points_table), OPERATING_FIELDS, operating_rows
    )
    comparison_path = _atomic_csv(
        config.resolve(config.outputs.historical_comparison_table),
        COMPARISON_FIELDS,
        comparison_rows,
    )
    prior_comparison_path: Path | None = None
    prior_run_comparison_path: Path | None = None
    frontier_bounds_path: Path | None = None
    if config.schema_version == 4:
        if (
            config.outputs.prior_comparison_table is None
            or config.outputs.prior_run_comparison_table is None
            or config.outputs.frontier_bounds_table is None
        ):
            raise ValueError("schema v4 output paths are missing")
        prior_comparison_path = _atomic_csv(
            config.resolve(config.outputs.prior_comparison_table),
            PRIOR_COMPARISON_FIELDS,
            prior_comparison_rows,
        )
        prior_run_comparison_path = _atomic_csv(
            config.resolve(config.outputs.prior_run_comparison_table),
            PRIOR_RUN_COMPARISON_FIELDS,
            prior_run_comparison_rows,
        )
        frontier_bounds_path = _atomic_csv(
            config.resolve(config.outputs.frontier_bounds_table),
            FRONTIER_BOUND_FIELDS,
            frontier_bound_rows,
        )
    figure_path = plot_exact_froc_comparison(
        exact_rows, aggregate_curve, operating_rows, historical_curve, config
    )
    floor_assessment = _candidate_floor_assessment(operating_rows, comparison_rows, config)
    run_reachability = build_run_reachability(run_summaries, operating_per_seed)
    inference_proposal = _inference_sensitivity_proposal(
        config,
        phase5,
        bundles,
        image_count=len(targets),
        needed=bool(floor_assessment["observable_frontier_truncated_by_candidate_floor"]),
    )

    source_path = Path(__file__).resolve()
    summary = {
        "schema_version": config.schema_version,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            source_path.relative_to(config.project_root).as_posix(): sha256_file(source_path),
            "src/evaluate.py": sha256_file(config.project_root / "src/evaluate.py"),
            "src/meddet_benchmark/evaluation.py": sha256_file(
                config.project_root / "src/meddet_benchmark/evaluation.py"
            ),
        },
        "upstream": {
            "phase5_config": _artifact(
                config.resolve(config.inputs.phase5_config), config.project_root
            ),
            "phase5_summary": _artifact(
                config.resolve(config.inputs.phase5_summary), config.project_root
            ),
            "test_annotations": _artifact(
                config.resolve(config.inputs.test_annotations), config.project_root
            ),
            "prediction_bundles": [
                {
                    "detector": bundle["detector"],
                    "seed": bundle["seed"],
                    **_artifact(bundle["path"], config.project_root),
                }
                for bundle in bundles
            ],
            "historical_froc_config": _artifact(
                config.resolve(config.inputs.historical_froc_config), config.project_root
            ),
            "historical_froc_summary": _artifact(
                config.resolve(config.inputs.historical_froc_summary), config.project_root
            ),
            "historical_curve_table": _artifact(
                config.resolve(config.inputs.historical_curve_table), config.project_root
            ),
            "historical_operating_points_table": _artifact(
                config.resolve(config.inputs.historical_operating_points_table),
                config.project_root,
            ),
            **(
                {
                    "prior_exact_summary": _artifact(
                        config.resolve(Path(config.inputs.prior_exact_summary)),
                        config.project_root,
                    ),
                    "prior_operating_points_per_seed_table": _artifact(
                        config.resolve(Path(config.inputs.prior_operating_points_per_seed_table)),
                        config.project_root,
                    ),
                    "prior_operating_points_table": _artifact(
                        config.resolve(Path(config.inputs.prior_operating_points_table)),
                        config.project_root,
                    ),
                }
                if config.schema_version == 4
                else {}
            ),
        },
        "protocol": {
            "thresholds": (
                "unique emitted confidence scores within each frozen bundle, evaluated in "
                "strictly descending order"
            ),
            "score_comparison": config.analysis.score_comparison,
            "upper_endpoint": "next representable float above each run's maximum emitted score",
            "lower_endpoint": "configured frozen candidate-score floor",
            "candidate_score_floor": config.analysis.candidate_score_floor,
            "matching_iou": config.analysis.match_iou_threshold,
            "matching": (
                "stable descending score, same configured class, highest-IoU unmatched target, "
                "no target reuse"
            ),
            "nms_iou": config.analysis.nms_iou_threshold,
            "max_detections_per_image": config.analysis.max_detections_per_image,
            "class_names": {str(key): value for key, value in sorted(category_names.items())},
            "seeds": list(config.analysis.expected_seeds),
            "fp_per_image_budgets": list(config.analysis.fp_per_image_budgets),
            "budget_selection": (
                "within each run, maximum observed exact-score sensitivity at or below the "
                "budget; no interpolation; ties use fewer FP/image then higher threshold"
            ),
            "aggregation": "arithmetic mean and sample standard deviation across five runs",
            "images_annotations_predictions_nms_unchanged": True,
            "threshold_selection_for_deployment": False,
            "performs_training": False,
            "performs_checkpoint_loading": False,
            "performs_model_inference": False,
        },
        "historical_grid": {
            **historical_grid,
            "matching_iou": config.analysis.match_iou_threshold,
            "sensitivity_definition": historical_summary["protocol"]["sensitivity"],
            "max_detections_per_image": phase5.evaluation.max_detections,
            "candidate_score_floor": (
                config.analysis.historical_candidate_score_floor
                if config.analysis.historical_candidate_score_floor is not None
                else phase5.evaluation.coco_minimum_score
            ),
            "seeds": list(config.analysis.expected_seeds),
            "fp_per_image_budgets": list(config.analysis.fp_per_image_budgets),
        },
        "run_score_boundaries": run_summaries,
        "run_reachability": run_reachability,
        "monotonicity_audit": {
            "runs_checked": len(run_summaries),
            "sensitivity_failures": 0,
            "fp_per_image_failures": 0,
            "status": "pass",
        },
        "canonical_matcher_crosscheck": {
            "operating_points_checked": canonical_crosschecks,
            "sensitivity_failures": 0,
            "fp_per_image_failures": 0,
            "status": "pass",
        },
        "counts": {
            "images_per_run": len(targets),
            "annotations": phase5_summary["annotation_count"],
            "detectors": len(config.detectors),
            "runs_per_detector": len(config.analysis.expected_seeds),
            "prediction_bundles": len(bundles),
            "exact_per_seed_rows": len(exact_rows),
            "exact_aggregate_curve_rows": len(aggregate_curve),
            "operating_point_rows_per_seed": len(operating_per_seed),
            "operating_point_rows": len(operating_rows),
            "historical_comparison_rows": len(comparison_rows),
            "prior_comparison_rows": len(prior_comparison_rows),
            "prior_run_comparison_rows": len(prior_run_comparison_rows),
            "frontier_bound_rows": len(frontier_bound_rows),
        },
        "operating_points": operating_rows,
        "operating_points_per_seed": operating_per_seed,
        "historical_comparison": comparison_rows,
        "prior_exact_analysis_id": (
            prior_summary.get("analysis_id") if prior_summary is not None else None
        ),
        "prior_exact_comparison": prior_comparison_rows,
        "prior_exact_run_comparison": prior_run_comparison_rows,
        "conservative_missing_frontier_bounds": frontier_bound_rows,
        "detector_ordering_bounds": ordering_bounds,
        "candidate_floor_assessment": floor_assessment,
        "inference_only_sensitivity_proposal": inference_proposal,
        "artifacts": {
            "exact_per_seed_table": _artifact(exact_path, config.project_root),
            "exact_aggregate_curve_table": _artifact(aggregate_curve_path, config.project_root),
            "operating_points_per_seed_table": _artifact(
                operating_per_seed_path, config.project_root
            ),
            "operating_points_table": _artifact(operating_path, config.project_root),
            "historical_comparison_table": _artifact(comparison_path, config.project_root),
            "figure": _artifact(figure_path, config.project_root),
            **(
                {
                    "prior_comparison_table": _artifact(prior_comparison_path, config.project_root),
                    "prior_run_comparison_table": _artifact(
                        prior_run_comparison_path, config.project_root
                    ),
                    "frontier_bounds_table": _artifact(frontier_bounds_path, config.project_root),
                }
                if prior_comparison_path is not None
                and prior_run_comparison_path is not None
                and frontier_bounds_path is not None
                else {}
            ),
        },
    }
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def preflight(config: ExactFrocConfig) -> dict[str, Any]:
    """Validate every frozen input without generating an artifact."""

    _phase5, _summary, bundles, targets, category_names = load_exact_inputs(config)
    _curve, _operating, _historical_summary, historical_grid = load_historical_grid(config)
    if config.schema_version == 4:
        load_prior_exact_operating_points(config)
    return {
        "status": "ready",
        "analysis_id": config.analysis_id,
        "prediction_bundles": len(bundles),
        "images_per_bundle": len(targets),
        "class_names": category_names,
        "seeds": list(config.analysis.expected_seeds),
        "historical_grid": historical_grid,
        "candidate_score_floor": config.analysis.candidate_score_floor,
        "performs_training": False,
        "performs_checkpoint_loading": False,
        "performs_model_inference": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the exact-score FROC command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/froc_exact_score_v2.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs or generate exact-score FROC artifacts."""

    args = build_parser().parse_args(argv)
    config = load_exact_froc_config(args.config)
    if args.mode == "run":
        run_exact_froc(config)
    else:
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
