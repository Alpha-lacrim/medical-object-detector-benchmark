"""Recall-weighted F-beta validation-threshold sensitivity.

This module operates only on the frozen validation prediction bundles created
by Phase 14. It never loads a checkpoint, performs model inference, or reads
test labels. Beta is a recall-versus-precision preference parameter, not a
measured clinical-harm ratio.
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

import matplotlib
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import sha256_file
from src.evaluate_threshold_selection import load_threshold_selection_config
from src.meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    evaluate_operating_point,
)
from src.stats.paired import (
    PatientClusters,
    build_patient_clusters,
    draw_hierarchical_bootstrap_multiplicities,
    stable_rng_seed,
)
from src.stats.run_statistics import load_coco_targets, load_patient_group_map

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SUMMARY_FIELDS = (
    "detector",
    "selection_split",
    "beta",
    "recall_to_precision_weight",
    "beta_interpretation",
    "selected_threshold",
    "selection_boundary",
    "primary_batch14_threshold",
    "threshold_change_from_batch14",
    "validation_precision",
    "validation_recall",
    "validation_f_beta",
    "f_beta_ci_lower",
    "f_beta_ci_upper",
    "near_optimal_lcb_tolerance",
    "near_optimal_plateau_start",
    "near_optimal_plateau_end",
    "near_optimal_plateau_width",
    "near_optimal_plateau_candidate_count",
    "bootstrap_selected_tau_ci_lower",
    "bootstrap_selected_tau_median",
    "bootstrap_selected_tau_ci_upper",
    "bootstrap_modal_selected_tau",
    "bootstrap_modal_selection_frequency",
    "canonical_tau_bootstrap_selection_frequency",
    "confidence_level",
    "bootstrap_resamples",
    "bootstrap_valid_resamples",
    "patient_group_count",
    "validation_image_count",
    "seed_count",
    "bootstrap_method",
    "seed_aggregation",
    "selection_rule",
    "tie_breaker",
    "relationship_to_primary_threshold",
)

STABILITY_FIELDS = (
    "detector",
    "selection_split",
    "beta",
    "recall_to_precision_weight",
    "candidate_threshold",
    "canonical_selected_threshold",
    "canonical_selection_rule",
    "f_beta_ci_lower",
    "near_optimal_lcb_tolerance",
    "in_near_optimal_plateau",
    "near_optimal_plateau_start",
    "near_optimal_plateau_end",
    "near_optimal_plateau_width",
    "near_optimal_plateau_candidate_count",
    "bootstrap_selection_count",
    "bootstrap_selection_frequency",
    "bootstrap_resamples",
    "bootstrap_selection_rule",
    "tie_breaker",
)

HYPOTHETICAL_LOSS_FIELDS = (
    "detector",
    "selection_split",
    "hypothetical_fn_to_fp_loss_ratio",
    "assumption_status",
    "selected_threshold",
    "selection_boundary",
    "validation_precision",
    "validation_recall",
    "validation_false_negatives_per_image",
    "validation_false_positives_per_image",
    "validation_hypothetical_loss_per_image",
    "loss_ci_lower",
    "loss_ci_upper",
    "confidence_level",
    "bootstrap_resamples",
    "bootstrap_valid_resamples",
    "patient_group_count",
    "validation_image_count",
    "seed_count",
    "normalization_unit",
    "selection_rule",
    "tie_breaker",
    "relationship_to_f_beta",
    "relationship_to_primary_threshold",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen validation inputs and the existing primary threshold table."""

    selection_data_role: Literal["model_development_validation"]
    validation_split_name: str = Field(pattern=r"^[a-z0-9_-]+$")
    threshold_selection_config: Path
    validation_split_manifest: Path
    primary_operating_points: Path


class AnalysisSettings(StrictModel):
    """F-beta sweep, hierarchical bootstrap, and selection contract."""

    threshold_start: float = Field(ge=0, le=1)
    threshold_stop: float = Field(ge=0, le=1)
    threshold_steps: int = Field(ge=2, le=1001)
    beta_values: tuple[float, ...] = Field(min_length=1)
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_stream_label: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    bootstrap_method: Literal["paired_hierarchical_patient_cluster_percentile"]
    seed_aggregation: Literal["arithmetic_mean_across_frozen_validation_seeds"]
    selection_rule: Literal["maximum_lower_confidence_bound_f_beta"]
    tie_breaker: Literal["highest_threshold"]
    near_optimal_absolute_tolerance: float = Field(ge=0, le=1)
    bootstrap_selection_rule: Literal["maximum_bootstrap_mean_f_beta"]
    manifest_image_column: str = Field(min_length=1)
    patient_group_column: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_analysis(self) -> AnalysisSettings:
        if self.threshold_stop <= self.threshold_start:
            raise ValueError("threshold_stop must be greater than threshold_start")
        if any(not np.isfinite(value) or value <= 0 for value in self.beta_values):
            raise ValueError("beta values must be finite and positive")
        if len(set(self.beta_values)) != len(self.beta_values):
            raise ValueError("beta values must be unique")
        return self

    def thresholds(self) -> tuple[float, ...]:
        """Return the inclusive, numerically stable threshold grid."""

        return tuple(
            float(value)
            for value in np.round(
                np.linspace(self.threshold_start, self.threshold_stop, self.threshold_steps),
                12,
            )
        )


class HypotheticalLossSettings(StrictModel):
    """Separate validation-only linear detection-error loss sensitivity."""

    enabled: bool
    fn_to_fp_loss_ratios: tuple[float, ...] = Field(min_length=1)
    normalization_unit: Literal["validation_image"]
    selection_rule: Literal["minimum_mean_hypothetical_detection_error_loss"]
    tie_breaker: Literal["highest_threshold"]

    @model_validator(mode="after")
    def validate_ratios(self) -> HypotheticalLossSettings:
        if any(not np.isfinite(value) or value <= 0 for value in self.fn_to_fp_loss_ratios):
            raise ValueError("hypothetical loss ratios must be finite and positive")
        if len(set(self.fn_to_fp_loss_ratios)) != len(self.fn_to_fp_loss_ratios):
            raise ValueError("hypothetical loss ratios must be unique")
        return self


class PlotSettings(StrictModel):
    """Static sensitivity-figure dimensions."""

    width_inches: float = Field(gt=0)
    height_inches: float = Field(gt=0)
    dpi: int = Field(ge=72, le=600)


class OutputSettings(StrictModel):
    """Required table/figure plus a provenance summary."""

    summary_table: Path
    stability_table: Path
    hypothetical_loss_table: Path
    sensitivity_figure: Path
    log_dir: Path
    summary_json: Path


class ThresholdCalibrationConfig(StrictModel):
    """Strict Batch 29 threshold-sensitivity contract."""

    schema_version: Literal[2]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0)
    inputs: InputSettings
    analysis: AnalysisSettings
    hypothetical_loss: HypotheticalLossSettings
    plot: PlotSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True)
class FrozenValidationData:
    """Verified frozen validation records and patient grouping."""

    bundles: tuple[dict[str, Any], ...]
    targets: tuple[ImageTarget, ...]
    class_ids: tuple[int, ...]
    evaluation: dict[str, Any]
    patient_clusters: PatientClusters
    primary_thresholds: dict[str, float]
    manifest_path: Path
    annotation_path: Path
    selection_config_path: Path
    primary_table_path: Path

    @property
    def detectors(self) -> tuple[str, ...]:
        return tuple(sorted({str(bundle["detector"]) for bundle in self.bundles}))

    @property
    def seeds(self) -> tuple[int, ...]:
        grids = {
            detector: tuple(
                sorted(
                    int(bundle["seed"]) for bundle in self.bundles if bundle["detector"] == detector
                )
            )
            for detector in self.detectors
        }
        unique = set(grids.values())
        if len(unique) != 1:
            raise ValueError(f"detectors have different validation seed grids: {grids}")
        return next(iter(unique))


@dataclass(frozen=True)
class BootstrapPlan:
    """Common random patient/seed multiplicities used at every threshold."""

    image_multiplicities: IntArray
    seed_multiplicities: IntArray


def load_threshold_calibration_config(path: str | Path) -> ThresholdCalibrationConfig:
    """Load and strictly validate the Batch 29 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold-calibration config must contain a mapping")
    payload["project_root"] = source.parent.parent.resolve()
    payload["source_path"] = source
    return ThresholdCalibrationConfig.model_validate(payload)


def f_beta(
    precision: float | FloatArray,
    recall: float | FloatArray,
    beta: float,
) -> float | FloatArray:
    """Compute F-beta with beta as a recall-versus-precision preference parameter.

    This is the weighted harmonic mean whose relative recall weight is beta
    squared. That algebra does not identify beta squared with empirical harm.
    """

    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    precision_array = np.asarray(precision, dtype=np.float64)
    recall_array = np.asarray(recall, dtype=np.float64)
    if np.any(~np.isfinite(precision_array)) or np.any(~np.isfinite(recall_array)):
        raise ValueError("precision and recall must be finite")
    if np.any((precision_array < 0) | (precision_array > 1)) or np.any(
        (recall_array < 0) | (recall_array > 1)
    ):
        raise ValueError("precision and recall must lie in [0, 1]")
    beta_squared = beta**2
    denominator = beta_squared * precision_array + recall_array
    result = np.divide(
        (1 + beta_squared) * precision_array * recall_array,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(result) if result.ndim == 0 else result


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


def _read_primary_thresholds(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"primary operating-point table is empty: {path}")
    thresholds: dict[str, float] = {}
    for row in rows:
        detector = str(row.get("detector", "")).strip()
        if not detector or detector in thresholds:
            raise ValueError("primary operating-point detector identities must be unique")
        if row.get("selection_split") != "validation":
            raise ValueError("primary operating points must be validation-selected")
        threshold = float(row["selected_threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("primary operating-point threshold lies outside [0, 1]")
        thresholds[detector] = threshold
    return thresholds


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


def _atomic_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, raw)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_validation_only_selection_contract(
    *,
    selection_data_role: str,
    upstream_split_name: str,
    expected_validation_split_name: str,
    test_split_accessed: Any,
    annotation_image_ids: Sequence[str],
    split_image_ids: Sequence[str],
) -> None:
    """Reject any selection evidence that is not the declared validation partition."""

    if selection_data_role != "model_development_validation":
        raise ValueError("threshold selection must use model-development validation data")
    if upstream_split_name != expected_validation_split_name:
        raise ValueError("upstream threshold-selection split is not the declared validation split")
    if test_split_accessed is not False:
        raise ValueError("threshold selection manifest does not prove test isolation")
    if tuple(sorted(annotation_image_ids)) != tuple(sorted(split_image_ids)):
        raise ValueError("selection labels and validation split manifest cover different images")


def _load_frozen_validation(config: ThresholdCalibrationConfig) -> FrozenValidationData:
    selection_config_path = config.resolve(config.inputs.threshold_selection_config)
    selection_config = load_threshold_selection_config(selection_config_path)
    if selection_config.project_root != config.project_root:
        raise ValueError("threshold-selection config belongs to a different repository root")
    if not np.allclose(
        selection_config.selection.thresholds(), config.analysis.thresholds(), atol=0, rtol=0
    ):
        raise ValueError("F-beta sweep must reuse the frozen Phase 14 threshold grid")

    manifest_path = selection_config.resolve(selection_config.outputs.validation_manifest)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("analysis_id") != (
        selection_config.analysis_id
    ):
        raise ValueError("validation prediction manifest identity or status is invalid")
    if manifest.get("config_sha256") != sha256_file(selection_config_path):
        raise ValueError("validation prediction manifest used a different selection config")
    if manifest.get("performs_training") is not False:
        raise ValueError("validation manifest violates the no-training/test-isolation contract")

    annotation_path = selection_config.resolve(selection_config.inputs.validation_annotations)
    annotation_hash = sha256_file(annotation_path)
    if (
        manifest.get("upstream", {}).get("validation_annotations", {}).get("sha256")
        != annotation_hash
    ):
        raise ValueError("validation annotations differ from the frozen manifest")
    targets, category_names = load_coco_targets(annotation_path)
    target_ids = tuple(target.image_id for target in targets)
    patient_manifest_path = config.resolve(config.inputs.validation_split_manifest)
    patient_map = load_patient_group_map(
        patient_manifest_path,
        image_column=config.analysis.manifest_image_column,
        patient_group_column=config.analysis.patient_group_column,
    )
    validate_validation_only_selection_contract(
        selection_data_role=config.inputs.selection_data_role,
        upstream_split_name=selection_config.inputs.validation_split,
        expected_validation_split_name=config.inputs.validation_split_name,
        test_split_accessed=manifest.get("test_split_accessed"),
        annotation_image_ids=target_ids,
        split_image_ids=tuple(patient_map),
    )
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("validation manifest lacks evaluator settings")
    if float(evaluation["coco_minimum_score"]) > config.analysis.threshold_start:
        raise ValueError("validation bundles discarded scores required by the sweep")

    primary_table_path = config.resolve(config.inputs.primary_operating_points)
    expected_primary_path = selection_config.resolve(
        selection_config.outputs.selected_operating_points_table
    )
    if primary_table_path != expected_primary_path:
        raise ValueError("configured primary thresholds are not Phase 14's aggregate table")
    primary_thresholds = _read_primary_thresholds(primary_table_path)

    records = manifest.get("runs")
    if not isinstance(records, list) or len(records) != int(manifest["counts"]["bundles"]):
        raise ValueError("validation manifest has an incomplete run list")
    bundles: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        detector, seed = str(record["detector"]), int(record["seed"])
        key = detector, seed
        if key in seen:
            raise ValueError(f"duplicate validation bundle identity: {key}")
        seen.add(key)
        bundle_info = record["validation_bundle"]
        bundle_path = config.project_root / str(bundle_info["path"])
        if sha256_file(bundle_path) != bundle_info["sha256"]:
            raise ValueError(f"validation bundle hash mismatch: {bundle_path}")
        payload = _read_bundle(bundle_path)
        if (
            payload.get("schema_version") != 1
            or payload.get("detector") != detector
            or int(payload.get("seed")) != seed
            or payload.get("split") != "validation"
        ):
            raise ValueError(f"validation bundle identity mismatch: {bundle_path}")
        if payload.get("annotation_sha256") != annotation_hash:
            raise ValueError(f"validation bundle annotation mismatch: {bundle_path}")
        if payload.get("evaluation") != evaluation:
            raise ValueError(f"validation bundle evaluator mismatch: {bundle_path}")
        predictions = _deserialize_predictions(payload)
        if tuple(prediction.image_id for prediction in predictions) != target_ids:
            raise ValueError(f"validation bundle image grid mismatch: {bundle_path}")
        bundles.append(
            {
                "detector": detector,
                "seed": seed,
                "path": bundle_path,
                "sha256": str(bundle_info["sha256"]),
                "predictions": predictions,
            }
        )
    bundles.sort(key=lambda item: (str(item["detector"]), int(item["seed"])))
    detectors = {str(bundle["detector"]) for bundle in bundles}
    if detectors != set(primary_thresholds):
        raise ValueError("validation bundles and primary threshold table cover different detectors")
    seed_grids = {
        detector: tuple(
            sorted(int(bundle["seed"]) for bundle in bundles if bundle["detector"] == detector)
        )
        for detector in detectors
    }
    if len(set(seed_grids.values())) != 1 or not next(iter(seed_grids.values())):
        raise ValueError(f"detectors have incomplete or unequal seed grids: {seed_grids}")

    patient_clusters = build_patient_clusters(target_ids, patient_map)
    return FrozenValidationData(
        bundles=tuple(bundles),
        targets=tuple(targets),
        class_ids=tuple(sorted(category_names)),
        evaluation=dict(evaluation),
        patient_clusters=patient_clusters,
        primary_thresholds=primary_thresholds,
        manifest_path=manifest_path,
        annotation_path=annotation_path,
        selection_config_path=selection_config_path,
        primary_table_path=primary_table_path,
    )


def build_bootstrap_plan(
    patient_clusters: PatientClusters,
    *,
    seed_count: int,
    resamples: int,
    base_seed: int,
    label: str,
) -> BootstrapPlan:
    """Materialize common draws via the established Phase 8 resampling helper."""

    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    rng = np.random.default_rng(stable_rng_seed(base_seed, label))
    image = np.empty((resamples, len(patient_clusters.image_ids)), dtype=np.int64)
    seeds = np.empty((resamples, seed_count), dtype=np.int64)
    for index in range(resamples):
        image[index], seeds[index] = draw_hierarchical_bootstrap_multiplicities(
            rng,
            patient_clusters,
            seed_count=seed_count,
        )
    return BootstrapPlan(image_multiplicities=image, seed_multiplicities=seeds)


def _ratios_from_counts(
    tp: FloatArray | IntArray,
    fp: FloatArray | IntArray,
    fn: FloatArray | IntArray,
) -> tuple[FloatArray, FloatArray]:
    tp_array = np.asarray(tp, dtype=np.float64)
    precision_denominator = tp_array + np.asarray(fp, dtype=np.float64)
    recall_denominator = tp_array + np.asarray(fn, dtype=np.float64)
    precision = np.divide(
        tp_array,
        precision_denominator,
        out=np.zeros_like(tp_array),
        where=precision_denominator > 0,
    )
    recall = np.divide(
        tp_array,
        recall_denominator,
        out=np.zeros_like(tp_array),
        where=recall_denominator > 0,
    )
    return precision, recall


def summarize_threshold_counts(
    tp_by_seed_image: IntArray,
    fp_by_seed_image: IntArray,
    fn_by_seed_image: IntArray,
    *,
    beta: float,
    bootstrap_plan: BootstrapPlan,
    confidence_level: float,
) -> dict[str, Any]:
    """Estimate mean-across-seed F-beta and its hierarchical percentile CI."""

    summary, _distribution = _summarize_threshold_counts_with_distribution(
        tp_by_seed_image,
        fp_by_seed_image,
        fn_by_seed_image,
        beta=beta,
        bootstrap_plan=bootstrap_plan,
        confidence_level=confidence_level,
    )
    return summary


def _summarize_threshold_counts_with_distribution(
    tp_by_seed_image: IntArray,
    fp_by_seed_image: IntArray,
    fn_by_seed_image: IntArray,
    *,
    beta: float,
    bootstrap_plan: BootstrapPlan,
    confidence_level: float,
) -> tuple[dict[str, Any], FloatArray]:
    """Return the F-beta summary and draw-level mean used for stability checks."""

    if not (
        tp_by_seed_image.shape == fp_by_seed_image.shape == fn_by_seed_image.shape
        and tp_by_seed_image.ndim == 2
    ):
        raise ValueError("TP/FP/FN arrays must share shape (seed, image)")
    seed_count, image_count = tp_by_seed_image.shape
    if bootstrap_plan.image_multiplicities.shape[1] != image_count:
        raise ValueError("bootstrap image multiplicities do not match evidence")
    if bootstrap_plan.seed_multiplicities.shape != (
        len(bootstrap_plan.image_multiplicities),
        seed_count,
    ):
        raise ValueError("bootstrap seed multiplicities do not match evidence")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")

    point_precision, point_recall = _ratios_from_counts(
        np.sum(tp_by_seed_image, axis=1),
        np.sum(fp_by_seed_image, axis=1),
        np.sum(fn_by_seed_image, axis=1),
    )
    point_f_beta = np.asarray(f_beta(point_precision, point_recall, beta), dtype=np.float64)

    weights = bootstrap_plan.image_multiplicities
    bootstrap_tp = weights @ tp_by_seed_image.T
    bootstrap_fp = weights @ fp_by_seed_image.T
    bootstrap_fn = weights @ fn_by_seed_image.T
    bootstrap_precision, bootstrap_recall = _ratios_from_counts(
        bootstrap_tp, bootstrap_fp, bootstrap_fn
    )
    bootstrap_f_beta = np.asarray(
        f_beta(bootstrap_precision, bootstrap_recall, beta), dtype=np.float64
    )
    seed_weights = bootstrap_plan.seed_multiplicities
    seed_weight_sums = np.sum(seed_weights, axis=1)
    if np.any(seed_weight_sums <= 0):
        raise ValueError("every bootstrap draw must retain at least one seed")
    bootstrap_mean = np.sum(bootstrap_f_beta * seed_weights, axis=1) / seed_weight_sums
    valid = bootstrap_mean[np.isfinite(bootstrap_mean)]
    if len(valid) != len(bootstrap_mean):
        raise ValueError("every bootstrap F-beta estimate must be finite")
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(valid, [tail, 1 - tail])
    return (
        {
            "precision": float(np.mean(point_precision)),
            "recall": float(np.mean(point_recall)),
            "f_beta": float(np.mean(point_f_beta)),
            "f_beta_ci_lower": float(lower),
            "f_beta_ci_upper": float(upper),
            "bootstrap_valid_resamples": len(valid),
        },
        np.asarray(bootstrap_mean, dtype=np.float64),
    )


def select_calibrated_thresholds(
    curve_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select each detector/beta maximum lower CI bound with a high-threshold tie break."""

    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in curve_rows:
        key = str(row["detector"]), float(row["beta"])
        groups.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for _key, candidates in sorted(groups.items()):
        best = max(
            candidates,
            key=lambda row: (float(row["f_beta_ci_lower"]), float(row["threshold"])),
        )
        selected.append(dict(best))
    return selected


def threshold_stability_diagnostics(
    curve_rows: Sequence[Mapping[str, Any]],
    bootstrap_distributions: Mapping[tuple[str, float, float], FloatArray],
    *,
    near_optimal_absolute_tolerance: float,
    confidence_level: float,
    bootstrap_selection_rule: str,
    tie_breaker: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], dict[str, Any]]]:
    """Describe LCB plateaus and bootstrap argmax frequencies without retuning."""

    if not 0 <= near_optimal_absolute_tolerance <= 1:
        raise ValueError("near-optimal tolerance must lie in [0, 1]")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")

    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in curve_rows:
        key = str(row["detector"]), float(row["beta"])
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, float], dict[str, Any]] = {}
    tail = (1 - confidence_level) / 2
    for key, unsorted_series in sorted(groups.items()):
        detector, beta = key
        series = sorted(unsorted_series, key=lambda row: float(row["threshold"]))
        thresholds = np.asarray([row["threshold"] for row in series], dtype=np.float64)
        objective = np.asarray([row["f_beta_ci_lower"] for row in series], dtype=np.float64)
        canonical_index = int(
            max(
                range(len(series)),
                key=lambda index: (objective[index], thresholds[index]),
            )
        )
        near = objective >= objective[canonical_index] - near_optimal_absolute_tolerance
        plateau_start_index = canonical_index
        while plateau_start_index > 0 and near[plateau_start_index - 1]:
            plateau_start_index -= 1
        plateau_end_index = canonical_index
        while plateau_end_index + 1 < len(series) and near[plateau_end_index + 1]:
            plateau_end_index += 1
        plateau_mask = np.zeros(len(series), dtype=bool)
        plateau_mask[plateau_start_index : plateau_end_index + 1] = True

        distribution_matrix = np.stack(
            [
                np.asarray(
                    bootstrap_distributions[(detector, beta, float(threshold))],
                    dtype=np.float64,
                )
                for threshold in thresholds
            ]
        )
        if np.any(~np.isfinite(distribution_matrix)):
            raise ValueError("bootstrap threshold-stability evidence must be finite")
        reverse_argmax = np.argmax(distribution_matrix[::-1], axis=0)
        selected_indices = len(series) - 1 - reverse_argmax
        selection_counts = np.bincount(selected_indices, minlength=len(series))
        selection_frequencies = selection_counts / len(selected_indices)
        selected_taus = thresholds[selected_indices]
        lower, median, upper = np.quantile(selected_taus, [tail, 0.5, 1 - tail])
        modal_count = int(np.max(selection_counts))
        modal_index = int(np.flatnonzero(selection_counts == modal_count)[-1])

        plateau_start = float(thresholds[plateau_start_index])
        plateau_end = float(thresholds[plateau_end_index])
        plateau_width = float(np.round(plateau_end - plateau_start, 12))
        summary = {
            "near_optimal_lcb_tolerance": near_optimal_absolute_tolerance,
            "near_optimal_plateau_start": plateau_start,
            "near_optimal_plateau_end": plateau_end,
            "near_optimal_plateau_width": plateau_width,
            "near_optimal_plateau_candidate_count": int(np.sum(plateau_mask)),
            "bootstrap_selected_tau_ci_lower": float(lower),
            "bootstrap_selected_tau_median": float(median),
            "bootstrap_selected_tau_ci_upper": float(upper),
            "bootstrap_modal_selected_tau": float(thresholds[modal_index]),
            "bootstrap_modal_selection_frequency": float(selection_frequencies[modal_index]),
            "canonical_tau_bootstrap_selection_frequency": float(
                selection_frequencies[canonical_index]
            ),
        }
        summaries[key] = summary

        for index, threshold in enumerate(thresholds):
            rows.append(
                {
                    "detector": detector,
                    "selection_split": "validation",
                    "beta": beta,
                    "recall_to_precision_weight": beta**2,
                    "candidate_threshold": float(threshold),
                    "canonical_selected_threshold": float(thresholds[canonical_index]),
                    "canonical_selection_rule": "maximum_lower_confidence_bound_f_beta",
                    "f_beta_ci_lower": float(objective[index]),
                    "near_optimal_lcb_tolerance": near_optimal_absolute_tolerance,
                    "in_near_optimal_plateau": bool(plateau_mask[index]),
                    "near_optimal_plateau_start": plateau_start,
                    "near_optimal_plateau_end": plateau_end,
                    "near_optimal_plateau_width": plateau_width,
                    "near_optimal_plateau_candidate_count": int(np.sum(plateau_mask)),
                    "bootstrap_selection_count": int(selection_counts[index]),
                    "bootstrap_selection_frequency": float(selection_frequencies[index]),
                    "bootstrap_resamples": len(selected_indices),
                    "bootstrap_selection_rule": bootstrap_selection_rule,
                    "tie_breaker": tie_breaker,
                }
            )
    return rows, summaries


def summarize_hypothetical_detection_error_loss(
    tp_by_seed_image: IntArray,
    fp_by_seed_image: IntArray,
    fn_by_seed_image: IntArray,
    *,
    fn_to_fp_loss_ratio: float,
    bootstrap_plan: BootstrapPlan,
    confidence_level: float,
) -> dict[str, Any]:
    """Compute r*FN/N + FP/N for an explicitly hypothetical error-loss ratio."""

    if not np.isfinite(fn_to_fp_loss_ratio) or fn_to_fp_loss_ratio <= 0:
        raise ValueError("hypothetical FN-to-FP loss ratio must be finite and positive")
    if not (
        tp_by_seed_image.shape == fp_by_seed_image.shape == fn_by_seed_image.shape
        and tp_by_seed_image.ndim == 2
    ):
        raise ValueError("TP/FP/FN arrays must share shape (seed, image)")
    seed_count, image_count = tp_by_seed_image.shape
    if bootstrap_plan.image_multiplicities.shape[1] != image_count:
        raise ValueError("bootstrap image multiplicities do not match evidence")
    if bootstrap_plan.seed_multiplicities.shape != (
        len(bootstrap_plan.image_multiplicities),
        seed_count,
    ):
        raise ValueError("bootstrap seed multiplicities do not match evidence")

    point_tp = np.sum(tp_by_seed_image, axis=1)
    point_fp = np.sum(fp_by_seed_image, axis=1)
    point_fn = np.sum(fn_by_seed_image, axis=1)
    point_precision, point_recall = _ratios_from_counts(point_tp, point_fp, point_fn)
    fp_per_image = point_fp / image_count
    fn_per_image = point_fn / image_count
    point_loss = fn_to_fp_loss_ratio * fn_per_image + fp_per_image

    weights = bootstrap_plan.image_multiplicities
    bootstrap_image_counts = np.sum(weights, axis=1)
    if np.any(bootstrap_image_counts <= 0):
        raise ValueError("every bootstrap draw must retain at least one validation image")
    bootstrap_fp = (weights @ fp_by_seed_image.T) / bootstrap_image_counts[:, None]
    bootstrap_fn = (weights @ fn_by_seed_image.T) / bootstrap_image_counts[:, None]
    bootstrap_loss = fn_to_fp_loss_ratio * bootstrap_fn + bootstrap_fp
    seed_weights = bootstrap_plan.seed_multiplicities
    seed_weight_sums = np.sum(seed_weights, axis=1)
    if np.any(seed_weight_sums <= 0):
        raise ValueError("every bootstrap draw must retain at least one seed")
    bootstrap_mean = np.sum(bootstrap_loss * seed_weights, axis=1) / seed_weight_sums
    if np.any(~np.isfinite(bootstrap_mean)):
        raise ValueError("every bootstrap hypothetical-loss estimate must be finite")
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(bootstrap_mean, [tail, 1 - tail])
    return {
        "precision": float(np.mean(point_precision)),
        "recall": float(np.mean(point_recall)),
        "false_negatives_per_image": float(np.mean(fn_per_image)),
        "false_positives_per_image": float(np.mean(fp_per_image)),
        "hypothetical_loss_per_image": float(np.mean(point_loss)),
        "loss_ci_lower": float(lower),
        "loss_ci_upper": float(upper),
        "bootstrap_valid_resamples": len(bootstrap_mean),
    }


def _threshold_counts(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    class_ids: tuple[int, ...],
    threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[IntArray, IntArray, IntArray]:
    result = evaluate_operating_point(
        predictions,
        targets,
        class_ids=class_ids,
        score_threshold=threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    per_image = result["per_image"]
    return tuple(
        np.asarray([int(row[field]) for row in per_image], dtype=np.int64)
        for field in ("tp", "fp", "fn")
    )


def compute_calibration_curve(
    data: FrozenValidationData,
    config: ThresholdCalibrationConfig,
    bootstrap_plan: BootstrapPlan,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, float, float], FloatArray],
    dict[tuple[str, float], tuple[IntArray, IntArray, IntArray]],
]:
    """Run the threshold outer loop and patient-cluster bootstrap inner analysis."""

    rows: list[dict[str, Any]] = []
    bootstrap_distributions: dict[tuple[str, float, float], FloatArray] = {}
    count_evidence: dict[tuple[str, float], tuple[IntArray, IntArray, IntArray]] = {}
    targets = list(data.targets)
    for detector in data.detectors:
        detector_bundles = [bundle for bundle in data.bundles if bundle["detector"] == detector]
        for threshold in config.analysis.thresholds():
            tp_parts: list[IntArray] = []
            fp_parts: list[IntArray] = []
            fn_parts: list[IntArray] = []
            for bundle in detector_bundles:
                tp, fp, fn = _threshold_counts(
                    bundle["predictions"],
                    targets,
                    class_ids=data.class_ids,
                    threshold=threshold,
                    iou_threshold=float(data.evaluation["match_iou_threshold"]),
                    max_detections=int(data.evaluation["max_detections"]),
                )
                tp_parts.append(tp)
                fp_parts.append(fp)
                fn_parts.append(fn)
            tp_by_seed_image = np.stack(tp_parts)
            fp_by_seed_image = np.stack(fp_parts)
            fn_by_seed_image = np.stack(fn_parts)
            count_evidence[(detector, threshold)] = (
                tp_by_seed_image,
                fp_by_seed_image,
                fn_by_seed_image,
            )
            for beta in config.analysis.beta_values:
                estimate, distribution = _summarize_threshold_counts_with_distribution(
                    tp_by_seed_image,
                    fp_by_seed_image,
                    fn_by_seed_image,
                    beta=beta,
                    bootstrap_plan=bootstrap_plan,
                    confidence_level=config.analysis.confidence_level,
                )
                rows.append(
                    {
                        "detector": detector,
                        "beta": beta,
                        "recall_to_precision_weight": beta**2,
                        "threshold": threshold,
                        **estimate,
                    }
                )
                bootstrap_distributions[(detector, beta, threshold)] = distribution
    return rows, bootstrap_distributions, count_evidence


def compute_hypothetical_loss_curve(
    count_evidence: Mapping[tuple[str, float], tuple[IntArray, IntArray, IntArray]],
    config: ThresholdCalibrationConfig,
    bootstrap_plan: BootstrapPlan,
) -> list[dict[str, Any]]:
    """Compute the separate linear detection-error loss sweep on validation."""

    rows: list[dict[str, Any]] = []
    if not config.hypothetical_loss.enabled:
        return rows
    for (detector, threshold), (tp, fp, fn) in sorted(count_evidence.items()):
        for ratio in config.hypothetical_loss.fn_to_fp_loss_ratios:
            rows.append(
                {
                    "detector": detector,
                    "hypothetical_fn_to_fp_loss_ratio": ratio,
                    "threshold": threshold,
                    **summarize_hypothetical_detection_error_loss(
                        tp,
                        fp,
                        fn,
                        fn_to_fp_loss_ratio=ratio,
                        bootstrap_plan=bootstrap_plan,
                        confidence_level=config.analysis.confidence_level,
                    ),
                }
            )
    return rows


def select_hypothetical_loss_thresholds(
    curve_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Minimize validation mean hypothetical loss with the declared tie break."""

    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in curve_rows:
        key = str(row["detector"]), float(row["hypothetical_fn_to_fp_loss_ratio"])
        groups.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for _key, candidates in sorted(groups.items()):
        best = min(
            candidates,
            key=lambda row: (
                float(row["hypothetical_loss_per_image"]),
                -float(row["threshold"]),
            ),
        )
        selected.append(dict(best))
    return selected


def _summary_rows(
    selected: Sequence[Mapping[str, Any]],
    stability_summaries: Mapping[tuple[str, float], Mapping[str, Any]],
    data: FrozenValidationData,
    config: ThresholdCalibrationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        detector = str(item["detector"])
        selected_threshold = float(item["threshold"])
        primary_threshold = data.primary_thresholds[detector]
        rows.append(
            {
                "detector": detector,
                "selection_split": "validation",
                "beta": item["beta"],
                "recall_to_precision_weight": item["recall_to_precision_weight"],
                "beta_interpretation": "recall_vs_precision_preference_parameter",
                "selected_threshold": selected_threshold,
                "selection_boundary": (
                    "lower"
                    if np.isclose(selected_threshold, config.analysis.threshold_start)
                    else (
                        "upper"
                        if np.isclose(selected_threshold, config.analysis.threshold_stop)
                        else "none"
                    )
                ),
                "primary_batch14_threshold": primary_threshold,
                "threshold_change_from_batch14": float(
                    np.round(selected_threshold - primary_threshold, 12)
                ),
                "validation_precision": item["precision"],
                "validation_recall": item["recall"],
                "validation_f_beta": item["f_beta"],
                "f_beta_ci_lower": item["f_beta_ci_lower"],
                "f_beta_ci_upper": item["f_beta_ci_upper"],
                **stability_summaries[(detector, float(item["beta"]))],
                "confidence_level": config.analysis.confidence_level,
                "bootstrap_resamples": config.analysis.bootstrap_resamples,
                "bootstrap_valid_resamples": item["bootstrap_valid_resamples"],
                "patient_group_count": data.patient_clusters.patient_group_count,
                "validation_image_count": len(data.targets),
                "seed_count": len(data.seeds),
                "bootstrap_method": config.analysis.bootstrap_method,
                "seed_aggregation": config.analysis.seed_aggregation,
                "selection_rule": config.analysis.selection_rule,
                "tie_breaker": config.analysis.tie_breaker,
                "relationship_to_primary_threshold": "sensitivity_extension_not_replacement",
            }
        )
    return rows


def _hypothetical_loss_summary_rows(
    selected: Sequence[Mapping[str, Any]],
    data: FrozenValidationData,
    config: ThresholdCalibrationConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        selected_threshold = float(item["threshold"])
        rows.append(
            {
                "detector": item["detector"],
                "selection_split": "validation",
                "hypothetical_fn_to_fp_loss_ratio": item["hypothetical_fn_to_fp_loss_ratio"],
                "assumption_status": "hypothetical_not_empirically_valued",
                "selected_threshold": selected_threshold,
                "selection_boundary": (
                    "lower"
                    if np.isclose(selected_threshold, config.analysis.threshold_start)
                    else (
                        "upper"
                        if np.isclose(selected_threshold, config.analysis.threshold_stop)
                        else "none"
                    )
                ),
                "validation_precision": item["precision"],
                "validation_recall": item["recall"],
                "validation_false_negatives_per_image": item["false_negatives_per_image"],
                "validation_false_positives_per_image": item["false_positives_per_image"],
                "validation_hypothetical_loss_per_image": item["hypothetical_loss_per_image"],
                "loss_ci_lower": item["loss_ci_lower"],
                "loss_ci_upper": item["loss_ci_upper"],
                "confidence_level": config.analysis.confidence_level,
                "bootstrap_resamples": config.analysis.bootstrap_resamples,
                "bootstrap_valid_resamples": item["bootstrap_valid_resamples"],
                "patient_group_count": data.patient_clusters.patient_group_count,
                "validation_image_count": len(data.targets),
                "seed_count": len(data.seeds),
                "normalization_unit": config.hypothetical_loss.normalization_unit,
                "selection_rule": config.hypothetical_loss.selection_rule,
                "tie_breaker": config.hypothetical_loss.tie_breaker,
                "relationship_to_f_beta": "separate_linear_error_loss_not_f_beta",
                "relationship_to_primary_threshold": "sensitivity_extension_not_replacement",
            }
        )
    return rows


def _atomic_sensitivity_figure(
    path: Path,
    curve_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    data: FrozenValidationData,
    config: ThresholdCalibrationConfig,
) -> Path:
    figure, axes = plt.subplots(
        1,
        len(data.detectors),
        figsize=(config.plot.width_inches, config.plot.height_inches),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(config.analysis.beta_values)))
    for axis, detector in zip(axes[0], data.detectors, strict=True):
        for color, beta in zip(colors, config.analysis.beta_values, strict=True):
            series = sorted(
                (
                    row
                    for row in curve_rows
                    if row["detector"] == detector and float(row["beta"]) == beta
                ),
                key=lambda row: float(row["threshold"]),
            )
            thresholds = np.asarray([row["threshold"] for row in series], dtype=np.float64)
            lower_bounds = np.asarray([row["f_beta_ci_lower"] for row in series], dtype=np.float64)
            selected = next(
                row
                for row in selected_rows
                if row["detector"] == detector and float(row["beta"]) == beta
            )
            axis.plot(
                thresholds,
                lower_bounds,
                color=color,
                linewidth=1.8,
                label=rf"$\beta$={beta:g} (recall weight $\beta^2$={beta**2:g})",
            )
            axis.scatter(
                [selected["threshold"]],
                [selected["f_beta_ci_lower"]],
                color=color,
                edgecolor="white",
                linewidth=0.7,
                s=45,
                zorder=4,
            )
        axis.axvline(
            data.primary_thresholds[detector],
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            label="Batch 14 primary threshold",
        )
        display_name = {
            "faster_rcnn": "Faster R-CNN",
            "yolo11s": "YOLO11s",
        }.get(detector, detector.replace("_", " ").title())
        axis.set_title(display_name)
        axis.set_xlabel(r"Confidence threshold $\tau$")
        axis.set_ylabel(r"Lower 95% bootstrap bound of mean $F_\beta$")
        axis.set_xlim(
            max(0.0, config.analysis.threshold_start - 0.01),
            min(1.0, config.analysis.threshold_stop + 0.01),
        )
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.legend(fontsize=8, frameon=False, loc="best")
    figure.suptitle("Recall-weighted F-beta validation-threshold sensitivity")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=config.plot.dpi, bbox_inches="tight", metadata={})
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return path


def preflight(config: ThresholdCalibrationConfig) -> dict[str, Any]:
    """Verify the complete frozen validation evidence without computing the sweep."""

    data = _load_frozen_validation(config)
    return {
        "status": "ready",
        "detectors": list(data.detectors),
        "seeds": list(data.seeds),
        "validation_images": len(data.targets),
        "patient_groups": data.patient_clusters.patient_group_count,
        "prediction_bundles": len(data.bundles),
        "thresholds": len(config.analysis.thresholds()),
        "betas": list(config.analysis.beta_values),
        "beta_interpretation": "recall_vs_precision_preference_parameter",
        "hypothetical_fn_to_fp_loss_ratios": (
            list(config.hypothetical_loss.fn_to_fp_loss_ratios)
            if config.hypothetical_loss.enabled
            else []
        ),
        "bootstrap_resamples": config.analysis.bootstrap_resamples,
        "performs_training": False,
        "performs_inference": False,
        "accesses_test_split": False,
    }


def run_threshold_calibration(config: ThresholdCalibrationConfig) -> dict[str, Any]:
    """Run the offline validation sensitivity analyses and write artifacts."""

    data = _load_frozen_validation(config)
    bootstrap_plan = build_bootstrap_plan(
        data.patient_clusters,
        seed_count=len(data.seeds),
        resamples=config.analysis.bootstrap_resamples,
        base_seed=config.seed,
        label=config.analysis.bootstrap_stream_label,
    )
    curve_rows, bootstrap_distributions, count_evidence = compute_calibration_curve(
        data, config, bootstrap_plan
    )
    selected = select_calibrated_thresholds(curve_rows)
    stability_rows, stability_summaries = threshold_stability_diagnostics(
        curve_rows,
        bootstrap_distributions,
        near_optimal_absolute_tolerance=config.analysis.near_optimal_absolute_tolerance,
        confidence_level=config.analysis.confidence_level,
        bootstrap_selection_rule=config.analysis.bootstrap_selection_rule,
        tie_breaker=config.analysis.tie_breaker,
    )
    summary_rows = _summary_rows(selected, stability_summaries, data, config)
    loss_curve_rows = compute_hypothetical_loss_curve(count_evidence, config, bootstrap_plan)
    loss_selected = select_hypothetical_loss_thresholds(loss_curve_rows)
    loss_summary_rows = _hypothetical_loss_summary_rows(loss_selected, data, config)
    table_path = _atomic_csv(
        config.resolve(config.outputs.summary_table), SUMMARY_FIELDS, summary_rows
    )
    stability_path = _atomic_csv(
        config.resolve(config.outputs.stability_table), STABILITY_FIELDS, stability_rows
    )
    loss_table_path = _atomic_csv(
        config.resolve(config.outputs.hypothetical_loss_table),
        HYPOTHETICAL_LOSS_FIELDS,
        loss_summary_rows,
    )
    figure_path = _atomic_sensitivity_figure(
        config.resolve(config.outputs.sensitivity_figure),
        curve_rows,
        selected,
        data,
        config,
    )

    summary = {
        "schema_version": 2,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            Path(__file__).resolve().relative_to(config.project_root).as_posix(): sha256_file(
                Path(__file__).resolve()
            ),
            "src/stats/paired.py": sha256_file(config.project_root / "src/stats/paired.py"),
            "src/meddet_benchmark/evaluation.py": sha256_file(
                config.project_root / "src/meddet_benchmark/evaluation.py"
            ),
        },
        "method": {
            **config.analysis.model_dump(mode="json"),
            "beta_interpretation": (
                "beta is a recall-versus-precision preference parameter; beta^2 is the "
                "relative recall weight in the weighted harmonic mean"
            ),
            "objective_derivation": ("F_beta=(1+beta^2)*TP/((1+beta^2)*TP+beta^2*FN+FP)"),
            "beta_is_measured_clinical_harm_ratio": False,
            "confidence_interval": "two-sided percentile interval",
            "bootstrap_random_numbers": "common across every threshold and beta",
            "selection_split": "validation",
            "relationship_to_primary_threshold": "sensitivity_extension_not_replacement",
        },
        "hypothetical_detection_error_loss": {
            **config.hypothetical_loss.model_dump(mode="json"),
            "formula": "L(tau;r)=r*FN(tau)/N+FP(tau)/N",
            "N_definition": "number of validation images in the observed or bootstrap draw",
            "assumption_status": "hypothetical_not_empirically_valued",
            "selection_split": "validation",
            "relationship_to_f_beta": "separate_linear_error_loss_not_f_beta",
            "deployment_utility_claimed": False,
        },
        "counts": {
            "detectors": len(data.detectors),
            "seeds_per_detector": len(data.seeds),
            "validation_images": len(data.targets),
            "patient_groups": data.patient_clusters.patient_group_count,
            "thresholds": len(config.analysis.thresholds()),
            "betas": len(config.analysis.beta_values),
            "curve_rows": len(curve_rows),
            "selected_rows": len(summary_rows),
            "stability_rows": len(stability_rows),
            "hypothetical_loss_curve_rows": len(loss_curve_rows),
            "hypothetical_loss_selected_rows": len(loss_summary_rows),
        },
        "upstream": {
            "threshold_selection_config": _artifact(
                data.selection_config_path, config.project_root
            ),
            "validation_manifest": _artifact(data.manifest_path, config.project_root),
            "validation_annotations": _artifact(data.annotation_path, config.project_root),
            "validation_split_manifest": _artifact(
                config.resolve(config.inputs.validation_split_manifest), config.project_root
            ),
            "primary_operating_points": _artifact(data.primary_table_path, config.project_root),
            "prediction_bundles": [
                {
                    "detector": bundle["detector"],
                    "seed": bundle["seed"],
                    "path": bundle["path"].relative_to(config.project_root).as_posix(),
                    "sha256": bundle["sha256"],
                }
                for bundle in data.bundles
            ],
        },
        "selected_operating_points": summary_rows,
        "threshold_stability": stability_rows,
        "threshold_curves": curve_rows,
        "hypothetical_loss_selected_operating_points": loss_summary_rows,
        "hypothetical_loss_curves": loss_curve_rows,
        "artifacts": {
            "summary_table": _artifact(table_path, config.project_root),
            "stability_table": _artifact(stability_path, config.project_root),
            "hypothetical_loss_table": _artifact(loss_table_path, config.project_root),
            "sensitivity_figure": _artifact(figure_path, config.project_root),
        },
        "performs_training": False,
        "performs_inference": False,
        "accesses_test_split": False,
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    result = {
        "status": "complete",
        "summary_table": table_path.as_posix(),
        "stability_table": stability_path.as_posix(),
        "hypothetical_loss_table": loss_table_path.as_posix(),
        "sensitivity_figure": figure_path.as_posix(),
        "summary_json": summary_path.as_posix(),
        "selected_operating_points": summary_rows,
        "hypothetical_loss_selected_operating_points": loss_summary_rows,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "run"), default="run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_threshold_calibration_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
    else:
        run_threshold_calibration(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
