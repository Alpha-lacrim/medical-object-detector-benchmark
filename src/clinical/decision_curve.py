"""Patient-cluster-aware decision-curve analysis of frozen test predictions.

The analysis reduces each object detector to an exam-level action rule: an
exam is flagged when its maximum emitted box confidence reaches the threshold.
It reads frozen Phase 5 bundles only; it never loads checkpoints, performs
inference, or trains a model.
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

from src.evaluate import load_phase5_config, sha256_file
from src.stats.paired import PatientClusters, build_patient_clusters
from src.stats.run_statistics import load_coco_targets
from src.stats.threshold_calibration import BootstrapPlan, build_bootstrap_plan

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SUMMARY_FIELDS = (
    "detector",
    "threshold_probability",
    "net_benefit",
    "net_benefit_ci_lower",
    "net_benefit_ci_upper",
    "mean_true_positive_images",
    "mean_false_positive_images",
    "mean_flagged_images",
    "treat_all_net_benefit",
    "treat_all_ci_lower",
    "treat_all_ci_upper",
    "treat_none_net_benefit",
    "treat_none_ci_lower",
    "treat_none_ci_upper",
    "net_benefit_difference_vs_other",
    "difference_ci_lower",
    "difference_ci_upper",
    "other_detector",
    "point_estimate_preferred_detector",
    "paired_ci_preferred_detector",
    "point_estimate_best_strategy",
    "population",
    "population_unit",
    "image_count",
    "positive_image_count",
    "negative_image_count",
    "disease_prevalence",
    "disease_prevalence_percent",
    "patient_group_count",
    "seed_count",
    "seed_ids",
    "confidence_level",
    "bootstrap_resamples",
    "bootstrap_valid_resamples",
    "bootstrap_method",
    "seed_aggregation",
    "decision_rule",
    "outcome_definition",
    "score_interpretation",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen test evidence used by the offline analysis."""

    phase5_config: Path
    phase5_summary: Path
    test_annotations: Path
    test_split_manifest: Path


class AnalysisSettings(StrictModel):
    """Decision threshold, population, and resampling contract."""

    detectors: tuple[str, str]
    threshold_start: float = Field(gt=0, lt=1)
    threshold_stop: float = Field(gt=0, lt=1)
    threshold_steps: int = Field(ge=2, le=1001)
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_method: Literal["paired_hierarchical_patient_cluster_percentile"]
    seed_aggregation: Literal["arithmetic_mean_across_frozen_test_seeds"]
    decision_unit: Literal["exam_image"]
    decision_score: Literal["maximum_emitted_box_confidence"]
    outcome_definition: Literal["at_least_one_lung_opacity_annotation"]
    score_interpretation: Literal["raw_detector_score_used_as_nominal_threshold_probability"]
    population_label: str = Field(min_length=1)
    expected_image_count: int = Field(ge=1)
    manifest_split_column: str = Field(min_length=1)
    expected_split: str = Field(min_length=1)
    manifest_image_column: str = Field(min_length=1)
    patient_group_column: str = Field(min_length=1)
    positive_column: str = Field(min_length=1)
    positive_value: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_contract(self) -> AnalysisSettings:
        if self.threshold_stop <= self.threshold_start:
            raise ValueError("threshold_stop must be greater than threshold_start")
        if len(set(self.detectors)) != 2 or any(not name.strip() for name in self.detectors):
            raise ValueError("detectors must contain two distinct non-empty names")
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


class PlotSettings(StrictModel):
    """Static decision-curve figure settings."""

    width_inches: float = Field(gt=0)
    height_inches: float = Field(gt=0)
    dpi: int = Field(ge=72, le=600)
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def validate_limits(self) -> PlotSettings:
        if not np.isfinite(self.y_min) or not np.isfinite(self.y_max):
            raise ValueError("plot y limits must be finite")
        if self.y_max <= self.y_min:
            raise ValueError("plot y_max must be greater than y_min")
        return self


class OutputSettings(StrictModel):
    """Required result artifacts and the provenance record."""

    summary_table: Path
    decision_curve_figure: Path
    log_dir: Path
    summary_json: Path


class DecisionCurveConfig(StrictModel):
    """Strict Batch 20 decision-curve contract."""

    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    inputs: InputSettings
    analysis: AnalysisSettings
    plot: PlotSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True)
class FrozenScoreBundle:
    """One verified frozen detector/seed maximum-score vector."""

    detector: str
    seed: int
    path: Path
    sha256: str
    maximum_scores: FloatArray


@dataclass(frozen=True)
class FrozenDecisionData:
    """Verified Phase 5 score evidence and full-test population labels."""

    bundles: tuple[FrozenScoreBundle, ...]
    image_ids: tuple[str, ...]
    outcome_positive: BoolArray
    patient_clusters: PatientClusters
    phase5_config_path: Path
    phase5_summary_path: Path
    annotation_path: Path
    split_manifest_path: Path

    @property
    def detectors(self) -> tuple[str, ...]:
        """Return detector names in deterministic order."""

        return tuple(sorted({bundle.detector for bundle in self.bundles}))

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the common detector seed grid, rejecting incomplete grids."""

        grids = {
            detector: tuple(
                sorted(bundle.seed for bundle in self.bundles if bundle.detector == detector)
            )
            for detector in self.detectors
        }
        if len(set(grids.values())) != 1:
            raise ValueError(f"detectors have different frozen seed grids: {grids}")
        return next(iter(grids.values()))


@dataclass(frozen=True)
class NetBenefitEstimate:
    """Point estimate, percentile interval, and retained bootstrap draws."""

    net_benefit: float
    ci_lower: float
    ci_upper: float
    valid_resamples: int
    mean_true_positives: float
    mean_false_positives: float
    bootstrap_values: FloatArray


def load_decision_curve_config(path: str | Path) -> DecisionCurveConfig:
    """Load and strictly validate the Batch 20 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decision-curve config must contain a mapping")
    payload["project_root"] = source.parent.parent.resolve()
    payload["source_path"] = source
    return DecisionCurveConfig.model_validate(payload)


def net_benefit(
    true_positives: float | FloatArray,
    false_positives: float | FloatArray,
    sample_size: float | FloatArray,
    threshold_probability: float,
) -> float | FloatArray:
    """Compute TP/N - FP/N * threshold odds for binary decisions."""

    if not np.isfinite(threshold_probability) or not 0 < threshold_probability < 1:
        raise ValueError("threshold_probability must lie strictly inside (0, 1)")
    tp, fp, size = np.broadcast_arrays(
        np.asarray(true_positives, dtype=np.float64),
        np.asarray(false_positives, dtype=np.float64),
        np.asarray(sample_size, dtype=np.float64),
    )
    if not np.isfinite(tp).all() or not np.isfinite(fp).all() or not np.isfinite(size).all():
        raise ValueError("counts and sample_size must be finite")
    if np.any(tp < 0) or np.any(fp < 0) or np.any(size <= 0) or np.any(tp + fp > size):
        raise ValueError("counts must be non-negative, disjoint, and within sample_size")
    threshold_odds = threshold_probability / (1 - threshold_probability)
    result = tp / size - (fp / size) * threshold_odds
    return float(result) if result.ndim == 0 else result


def treat_all_net_benefit(
    prevalence: float | FloatArray, threshold_probability: float
) -> float | FloatArray:
    """Compute the decision-curve treat-all reference from empirical prevalence."""

    prevalence_array = np.asarray(prevalence, dtype=np.float64)
    if not np.isfinite(prevalence_array).all() or np.any(
        (prevalence_array < 0) | (prevalence_array > 1)
    ):
        raise ValueError("prevalence must be finite and lie in [0, 1]")
    result = net_benefit(
        prevalence_array,
        1 - prevalence_array,
        np.ones_like(prevalence_array),
        threshold_probability,
    )
    return result


def maximum_scores_by_image(
    prediction_records: Sequence[Mapping[str, Any]], image_ids: tuple[str, ...]
) -> FloatArray:
    """Return one maximum emitted box confidence per expected image."""

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("expected image IDs must be unique")
    score_map: dict[str, float] = {}
    for record in prediction_records:
        image_id = str(record.get("image_id", ""))
        if not image_id or image_id in score_map:
            raise ValueError("prediction bundle image IDs must be unique and non-empty")
        raw_scores = np.asarray(record.get("scores", []), dtype=np.float64)
        if (
            raw_scores.ndim != 1
            or not np.isfinite(raw_scores).all()
            or np.any((raw_scores < 0) | (raw_scores > 1))
        ):
            raise ValueError(f"invalid prediction scores for image {image_id}")
        score_map[image_id] = float(np.max(raw_scores)) if len(raw_scores) else 0.0
    if set(score_map) != set(image_ids):
        missing = sorted(set(image_ids) - set(score_map))
        extra = sorted(set(score_map) - set(image_ids))
        raise ValueError(
            f"prediction image grid mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return np.asarray([score_map[image_id] for image_id in image_ids], dtype=np.float64)


def summarize_net_benefit(
    predicted_positive_by_seed_image: BoolArray,
    outcome_positive: BoolArray,
    *,
    threshold_probability: float,
    bootstrap_plan: BootstrapPlan,
    confidence_level: float,
) -> NetBenefitEstimate:
    """Estimate mean-across-seed net benefit with a hierarchical percentile CI."""

    predicted = np.asarray(predicted_positive_by_seed_image, dtype=np.bool_)
    outcome = np.asarray(outcome_positive, dtype=np.bool_)
    if predicted.ndim != 2 or outcome.ndim != 1 or predicted.shape[1] != len(outcome):
        raise ValueError("predictions must have shape (seed, image) aligned with outcomes")
    seed_count, image_count = predicted.shape
    if bootstrap_plan.image_multiplicities.shape[1] != image_count:
        raise ValueError("bootstrap image multiplicities do not match decisions")
    if bootstrap_plan.seed_multiplicities.shape != (
        len(bootstrap_plan.image_multiplicities),
        seed_count,
    ):
        raise ValueError("bootstrap seed multiplicities do not match decisions")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")

    true_positive = predicted & outcome[None, :]
    false_positive = predicted & ~outcome[None, :]
    point_tp = np.sum(true_positive, axis=1, dtype=np.int64)
    point_fp = np.sum(false_positive, axis=1, dtype=np.int64)
    point_by_seed = np.asarray(
        net_benefit(point_tp, point_fp, image_count, threshold_probability), dtype=np.float64
    )

    image_weights = bootstrap_plan.image_multiplicities
    bootstrap_size = np.sum(image_weights, axis=1, dtype=np.int64)
    bootstrap_tp = image_weights @ true_positive.T.astype(np.int64)
    bootstrap_fp = image_weights @ false_positive.T.astype(np.int64)
    bootstrap_by_seed = np.asarray(
        net_benefit(
            bootstrap_tp,
            bootstrap_fp,
            bootstrap_size[:, None],
            threshold_probability,
        ),
        dtype=np.float64,
    )
    seed_weights = bootstrap_plan.seed_multiplicities
    seed_weight_sums = np.sum(seed_weights, axis=1, dtype=np.int64)
    if np.any(seed_weight_sums <= 0):
        raise ValueError("every bootstrap draw must retain at least one seed")
    bootstrap_values = np.sum(bootstrap_by_seed * seed_weights, axis=1) / seed_weight_sums
    valid = bootstrap_values[np.isfinite(bootstrap_values)]
    if not len(valid):
        raise ValueError("no finite bootstrap net-benefit estimates")
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(valid, [tail, 1 - tail])
    return NetBenefitEstimate(
        net_benefit=float(np.mean(point_by_seed)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        valid_resamples=len(valid),
        mean_true_positives=float(np.mean(point_tp)),
        mean_false_positives=float(np.mean(point_fp)),
        bootstrap_values=bootstrap_values,
    )


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


def _load_population(
    path: Path,
    config: DecisionCurveConfig,
    image_ids: tuple[str, ...],
    annotation_positive: BoolArray,
) -> tuple[BoolArray, PatientClusters]:
    settings = config.analysis
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != settings.expected_image_count:
        raise ValueError(
            f"full test manifest must contain {settings.expected_image_count} images, "
            f"found {len(rows)}"
        )
    required = {
        settings.manifest_split_column,
        settings.manifest_image_column,
        settings.patient_group_column,
        settings.positive_column,
    }
    missing_columns = required - set(rows[0] if rows else ())
    if missing_columns:
        raise ValueError(f"test manifest lacks columns: {sorted(missing_columns)}")
    if {str(row[settings.manifest_split_column]) for row in rows} != {settings.expected_split}:
        raise ValueError("test manifest contains an unexpected split identity")

    by_image: dict[str, Mapping[str, str]] = {}
    for row in rows:
        image_id = str(row[settings.manifest_image_column]).strip()
        if not image_id or image_id in by_image:
            raise ValueError("test manifest image IDs must be unique and non-empty")
        by_image[image_id] = row
    if set(by_image) != set(image_ids):
        raise ValueError("test manifest and annotation image grids differ")

    outcomes = np.asarray(
        [
            str(by_image[image_id][settings.positive_column]) == settings.positive_value
            for image_id in image_ids
        ],
        dtype=np.bool_,
    )
    if not np.array_equal(outcomes, annotation_positive):
        raise ValueError("manifest positivity differs from the presence of COCO annotations")
    group_map = {
        image_id: str(by_image[image_id][settings.patient_group_column]).strip()
        for image_id in image_ids
    }
    return outcomes, build_patient_clusters(image_ids, group_map)


def _load_frozen_data(config: DecisionCurveConfig) -> FrozenDecisionData:
    phase5_config_path = config.resolve(config.inputs.phase5_config)
    phase5_summary_path = config.resolve(config.inputs.phase5_summary)
    annotation_path = config.resolve(config.inputs.test_annotations)
    split_manifest_path = config.resolve(config.inputs.test_split_manifest)
    phase5 = load_phase5_config(phase5_config_path)
    summary = _read_json(phase5_summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Phase 5 summary is not complete")
    if summary.get("config_sha256") != sha256_file(phase5_config_path):
        raise ValueError("Phase 5 config differs from its frozen summary")
    annotation_sha256 = sha256_file(annotation_path)
    if summary.get("test_annotation_sha256") != annotation_sha256:
        raise ValueError("test annotations differ from the frozen Phase 5 source")
    if summary.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("Phase 5 evaluator settings differ from the frozen summary")
    if phase5.split != config.analysis.expected_split:
        raise ValueError("Phase 5 config is not bound to the requested test split")
    if phase5.evaluation.coco_minimum_score > config.analysis.threshold_start:
        raise ValueError("frozen bundles discarded scores required by the DCA grid")

    targets, _ = load_coco_targets(annotation_path)
    targets = sorted(targets, key=lambda target: target.image_id)
    image_ids = tuple(target.image_id for target in targets)
    if len(image_ids) != config.analysis.expected_image_count:
        raise ValueError("test annotations do not contain the full configured population")
    annotation_positive = np.asarray(
        [len(target.boxes_xyxy) > 0 for target in targets], dtype=np.bool_
    )
    outcomes, patient_clusters = _load_population(
        split_manifest_path,
        config,
        image_ids,
        annotation_positive,
    )

    expected = {(detector, seed) for detector in config.analysis.detectors for seed in phase5.seeds}
    seen: set[tuple[str, int]] = set()
    bundles: list[FrozenScoreBundle] = []
    for run in summary.get("runs", []):
        detector, seed = str(run.get("detector")), int(run.get("seed"))
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
        bundle_hash = str(run.get("prediction_bundle_sha256"))
        if sha256_file(bundle_path) != bundle_hash:
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
        records = payload.get("predictions")
        if not isinstance(records, list):
            raise ValueError(f"Phase 5 bundle lacks prediction records: {bundle_path}")
        bundles.append(
            FrozenScoreBundle(
                detector=detector,
                seed=seed,
                path=bundle_path,
                sha256=bundle_hash,
                maximum_scores=maximum_scores_by_image(records, image_ids),
            )
        )
    if seen != expected:
        raise ValueError("Phase 5 summary does not contain the configured detector/seed grid")
    bundles.sort(key=lambda bundle: (bundle.detector, bundle.seed))
    data = FrozenDecisionData(
        bundles=tuple(bundles),
        image_ids=image_ids,
        outcome_positive=outcomes,
        patient_clusters=patient_clusters,
        phase5_config_path=phase5_config_path,
        phase5_summary_path=phase5_summary_path,
        annotation_path=annotation_path,
        split_manifest_path=split_manifest_path,
    )
    if data.detectors != tuple(sorted(config.analysis.detectors)):
        raise ValueError("loaded detector identities differ from the configured pair")
    if data.seeds != tuple(phase5.seeds):
        raise ValueError("loaded seed grid differs from the frozen Phase 5 config")
    return data


def _reference_estimates(
    data: FrozenDecisionData,
    bootstrap_plan: BootstrapPlan,
    *,
    threshold_probability: float,
    confidence_level: float,
) -> dict[str, float | int]:
    image_weights = bootstrap_plan.image_multiplicities
    bootstrap_size = np.sum(image_weights, axis=1, dtype=np.int64)
    bootstrap_positive = image_weights @ data.outcome_positive.astype(np.int64)
    bootstrap_prevalence = bootstrap_positive / bootstrap_size
    treat_all_bootstrap = np.asarray(
        treat_all_net_benefit(bootstrap_prevalence, threshold_probability),
        dtype=np.float64,
    )
    valid = treat_all_bootstrap[np.isfinite(treat_all_bootstrap)]
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(valid, [tail, 1 - tail])
    prevalence = float(np.mean(data.outcome_positive))
    return {
        "treat_all_net_benefit": float(treat_all_net_benefit(prevalence, threshold_probability)),
        "treat_all_ci_lower": float(lower),
        "treat_all_ci_upper": float(upper),
        "treat_none_net_benefit": 0.0,
        "treat_none_ci_lower": 0.0,
        "treat_none_ci_upper": 0.0,
        "reference_valid_resamples": len(valid),
    }


def compute_decision_curves(
    data: FrozenDecisionData,
    config: DecisionCurveConfig,
    bootstrap_plan: BootstrapPlan,
) -> list[dict[str, Any]]:
    """Compute both detector curves and paired pointwise differences."""

    detector_scores = {
        detector: np.stack(
            [bundle.maximum_scores for bundle in data.bundles if bundle.detector == detector]
        )
        for detector in data.detectors
    }
    image_count = len(data.image_ids)
    positive_count = int(np.sum(data.outcome_positive))
    prevalence = positive_count / image_count
    rows: list[dict[str, Any]] = []
    for threshold in config.analysis.thresholds():
        references = _reference_estimates(
            data,
            bootstrap_plan,
            threshold_probability=threshold,
            confidence_level=config.analysis.confidence_level,
        )
        estimates = {
            detector: summarize_net_benefit(
                detector_scores[detector] >= threshold,
                data.outcome_positive,
                threshold_probability=threshold,
                bootstrap_plan=bootstrap_plan,
                confidence_level=config.analysis.confidence_level,
            )
            for detector in data.detectors
        }
        first, second = data.detectors
        if np.isclose(
            estimates[first].net_benefit,
            estimates[second].net_benefit,
            atol=1e-15,
            rtol=0,
        ):
            point_preferred = "tie"
        else:
            point_preferred = max(data.detectors, key=lambda name: estimates[name].net_benefit)
        strategy_values = {
            **{name: estimates[name].net_benefit for name in data.detectors},
            "treat_all": float(references["treat_all_net_benefit"]),
            "treat_none": 0.0,
        }
        best_value = max(strategy_values.values())
        best_strategy = "|".join(
            name
            for name, value in strategy_values.items()
            if np.isclose(value, best_value, atol=1e-15, rtol=0)
        )
        for detector in data.detectors:
            other = next(name for name in data.detectors if name != detector)
            difference = estimates[detector].bootstrap_values - estimates[other].bootstrap_values
            tail = (1 - config.analysis.confidence_level) / 2
            difference_lower, difference_upper = np.quantile(difference, [tail, 1 - tail])
            if difference_lower > 0:
                paired_preferred = detector
            elif difference_upper < 0:
                paired_preferred = other
            else:
                paired_preferred = "not_distinguished"
            estimate = estimates[detector]
            rows.append(
                {
                    "detector": detector,
                    "threshold_probability": threshold,
                    "net_benefit": estimate.net_benefit,
                    "net_benefit_ci_lower": estimate.ci_lower,
                    "net_benefit_ci_upper": estimate.ci_upper,
                    "mean_true_positive_images": estimate.mean_true_positives,
                    "mean_false_positive_images": estimate.mean_false_positives,
                    "mean_flagged_images": (
                        estimate.mean_true_positives + estimate.mean_false_positives
                    ),
                    **{
                        key: value
                        for key, value in references.items()
                        if key != "reference_valid_resamples"
                    },
                    "net_benefit_difference_vs_other": (
                        estimate.net_benefit - estimates[other].net_benefit
                    ),
                    "difference_ci_lower": float(difference_lower),
                    "difference_ci_upper": float(difference_upper),
                    "other_detector": other,
                    "point_estimate_preferred_detector": point_preferred,
                    "paired_ci_preferred_detector": paired_preferred,
                    "point_estimate_best_strategy": best_strategy,
                    "population": config.analysis.population_label,
                    "population_unit": config.analysis.decision_unit,
                    "image_count": image_count,
                    "positive_image_count": positive_count,
                    "negative_image_count": image_count - positive_count,
                    "disease_prevalence": prevalence,
                    "disease_prevalence_percent": 100 * prevalence,
                    "patient_group_count": data.patient_clusters.patient_group_count,
                    "seed_count": len(data.seeds),
                    "seed_ids": "|".join(str(seed) for seed in data.seeds),
                    "confidence_level": config.analysis.confidence_level,
                    "bootstrap_resamples": config.analysis.bootstrap_resamples,
                    "bootstrap_valid_resamples": estimate.valid_resamples,
                    "bootstrap_method": config.analysis.bootstrap_method,
                    "seed_aggregation": config.analysis.seed_aggregation,
                    "decision_rule": "maximum emitted box confidence >= threshold",
                    "outcome_definition": config.analysis.outcome_definition,
                    "score_interpretation": config.analysis.score_interpretation,
                }
            )
    return rows


def contiguous_ranges(rows: Sequence[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    """Collapse adjacent thresholds sharing one categorical value into ranges."""

    by_threshold: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        threshold = float(row["threshold_probability"])
        by_threshold.setdefault(threshold, row)
    ordered = sorted(by_threshold.items())
    ranges: list[dict[str, Any]] = []
    for threshold, row in ordered:
        value = row[key]
        if not ranges or ranges[-1]["value"] != value:
            ranges.append({"start": threshold, "stop": threshold, "value": value})
        else:
            ranges[-1]["stop"] = threshold
    return ranges


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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, raw)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _display_name(detector: str) -> str:
    return {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}.get(
        detector, detector.replace("_", " ").title()
    )


def _atomic_figure(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    data: FrozenDecisionData,
    config: DecisionCurveConfig,
) -> Path:
    colors = {data.detectors[0]: "#2f6f9f", data.detectors[1]: "#d97706"}
    figure, axis = plt.subplots(
        figsize=(config.plot.width_inches, config.plot.height_inches), constrained_layout=True
    )
    for detector in data.detectors:
        series = sorted(
            (row for row in rows if row["detector"] == detector),
            key=lambda row: float(row["threshold_probability"]),
        )
        thresholds = np.asarray([row["threshold_probability"] for row in series], dtype=np.float64)
        values = np.asarray([row["net_benefit"] for row in series], dtype=np.float64)
        lower = np.asarray([row["net_benefit_ci_lower"] for row in series], dtype=np.float64)
        upper = np.asarray([row["net_benefit_ci_upper"] for row in series], dtype=np.float64)
        axis.plot(
            thresholds,
            values,
            color=colors[detector],
            linewidth=2.2,
            label=_display_name(detector),
        )
        axis.fill_between(thresholds, lower, upper, color=colors[detector], alpha=0.18)

    reference = sorted(
        (row for row in rows if row["detector"] == data.detectors[0]),
        key=lambda row: float(row["threshold_probability"]),
    )
    thresholds = np.asarray([row["threshold_probability"] for row in reference], dtype=np.float64)
    axis.plot(
        thresholds,
        [row["treat_all_net_benefit"] for row in reference],
        color="#444444",
        linestyle="--",
        linewidth=1.6,
        label="Treat all",
    )
    axis.axhline(0.0, color="#111111", linestyle=":", linewidth=1.5, label="Treat none")
    prevalence = float(np.mean(data.outcome_positive))
    axis.set_title(
        "Decision curves on the full held-out test split\n"
        f"Empirical exam prevalence: {int(np.sum(data.outcome_positive))}/"
        f"{len(data.image_ids)} = {prevalence:.3%}"
    )
    axis.set_xlabel(r"Nominal threshold probability / raw detector score $\tau$")
    axis.set_ylabel("Net benefit")
    axis.set_xlim(config.analysis.threshold_start, config.analysis.threshold_stop)
    axis.set_ylim(config.plot.y_min, config.plot.y_max)
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, loc="best")

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


def preflight(config: DecisionCurveConfig) -> dict[str, Any]:
    """Verify the full frozen test evidence without computing the curves."""

    data = _load_frozen_data(config)
    positive_count = int(np.sum(data.outcome_positive))
    image_count = len(data.image_ids)
    return {
        "status": "ready",
        "analysis_id": config.analysis_id,
        "population": config.analysis.population_label,
        "population_unit": config.analysis.decision_unit,
        "image_count": image_count,
        "positive_image_count": positive_count,
        "negative_image_count": image_count - positive_count,
        "empirical_disease_prevalence": positive_count / image_count,
        "patient_group_count": data.patient_clusters.patient_group_count,
        "detectors": list(data.detectors),
        "seeds": list(data.seeds),
        "prediction_bundles": len(data.bundles),
        "thresholds": len(config.analysis.thresholds()),
        "bootstrap_resamples": config.analysis.bootstrap_resamples,
        "performs_training": False,
        "performs_inference": False,
        "loads_checkpoints": False,
        "uses_robustness_subsample": False,
    }


def run_decision_curve(config: DecisionCurveConfig) -> dict[str, Any]:
    """Compute decision curves and write deterministic result artifacts."""

    data = _load_frozen_data(config)
    bootstrap_plan = build_bootstrap_plan(
        data.patient_clusters,
        seed_count=len(data.seeds),
        resamples=config.analysis.bootstrap_resamples,
        base_seed=config.seed,
        label=config.analysis_id,
    )
    rows = compute_decision_curves(data, config, bootstrap_plan)
    table_path = _atomic_csv(config.resolve(config.outputs.summary_table), rows)
    figure_path = _atomic_figure(
        config.resolve(config.outputs.decision_curve_figure), rows, data, config
    )
    positive_count = int(np.sum(data.outcome_positive))
    image_count = len(data.image_ids)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "population": {
            "label": config.analysis.population_label,
            "unit": config.analysis.decision_unit,
            "image_count": image_count,
            "positive_image_count": positive_count,
            "negative_image_count": image_count - positive_count,
            "empirical_disease_prevalence": positive_count / image_count,
            "empirical_disease_prevalence_percent": 100 * positive_count / image_count,
            "patient_group_count": data.patient_clusters.patient_group_count,
            "source": data.split_manifest_path.relative_to(config.project_root).as_posix(),
            "robustness_subsample_used": False,
        },
        "method": {
            **config.analysis.model_dump(mode="json"),
            "net_benefit_formula": "TP/N - (FP/N) * tau/(1-tau)",
            "decision_rule": "flag exam when maximum emitted box confidence >= tau",
            "confidence_interval": "pointwise two-sided percentile interval",
            "common_draws_across_detectors_and_thresholds": True,
            "bootstrap_random_unit": "NIH patient group plus frozen training seed",
            "localization_used_in_decision": False,
        },
        "counts": {
            "detectors": len(data.detectors),
            "seeds_per_detector": len(data.seeds),
            "prediction_bundles": len(data.bundles),
            "thresholds": len(config.analysis.thresholds()),
            "rows": len(rows),
        },
        "point_estimate_preference_ranges": contiguous_ranges(
            rows, key="point_estimate_preferred_detector"
        ),
        "paired_difference_ci_preference_ranges": contiguous_ranges(
            rows, key="paired_ci_preferred_detector"
        ),
        "point_estimate_best_strategy_ranges": contiguous_ranges(
            rows, key="point_estimate_best_strategy"
        ),
        "source": {
            "config": _artifact(config.source_path, config.project_root),
            "analysis_source": _artifact(Path(__file__).resolve(), config.project_root),
            "bootstrap_source": _artifact(
                config.project_root / "src/stats/paired.py", config.project_root
            ),
            "bootstrap_plan_source": _artifact(
                config.project_root / "src/stats/threshold_calibration.py",
                config.project_root,
            ),
            "phase5_config": _artifact(data.phase5_config_path, config.project_root),
            "phase5_summary": _artifact(data.phase5_summary_path, config.project_root),
            "test_annotations": _artifact(data.annotation_path, config.project_root),
            "test_split_manifest": _artifact(data.split_manifest_path, config.project_root),
            "prediction_bundles": [
                {
                    "detector": bundle.detector,
                    "seed": bundle.seed,
                    "path": bundle.path.relative_to(config.project_root).as_posix(),
                    "sha256": bundle.sha256,
                }
                for bundle in data.bundles
            ],
        },
        "artifacts": {
            "summary_table": _artifact(table_path, config.project_root),
            "decision_curve_figure": _artifact(figure_path, config.project_root),
        },
        "performs_training": False,
        "performs_inference": False,
        "loads_checkpoints": False,
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    result = {
        "status": "complete",
        "population": summary["population"],
        "summary_table": table_path.as_posix(),
        "decision_curve_figure": figure_path.as_posix(),
        "summary_json": summary_path.as_posix(),
        "point_estimate_preference_ranges": summary["point_estimate_preference_ranges"],
        "paired_difference_ci_preference_ranges": summary["paired_difference_ci_preference_ranges"],
        "point_estimate_best_strategy_ranges": summary["point_estimate_best_strategy_ranges"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/decision_curve.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run decision-curve preflight or the complete offline analysis."""

    args = _parse_args(argv)
    config = load_decision_curve_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True, allow_nan=False))
    else:
        run_decision_curve(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
