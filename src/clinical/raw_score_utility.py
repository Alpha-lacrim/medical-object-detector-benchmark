"""Relabel the historical raw-score calculation as a non-standard utility curve.

The exact pre-Batch-30 implementation, config, table, figure, and provenance
remain archived. This module recomputes the same arithmetic from the same frozen
test bundles, verifies numerical identity against the archived table, and emits
artifacts whose terminology cannot be mistaken for conventional DCA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.clinical.archive.decision_curve_pre_batch30_nonstandard import (
    SUMMARY_FIELDS as HISTORICAL_SUMMARY_FIELDS,
)
from src.clinical.archive.decision_curve_pre_batch30_nonstandard import (
    DecisionCurveConfig as HistoricalDecisionCurveConfig,
)
from src.clinical.archive.decision_curve_pre_batch30_nonstandard import (
    FrozenDecisionData,
    _load_frozen_data,
    compute_decision_curves,
    contiguous_ranges,
    load_decision_curve_config,
)
from src.clinical.archive.decision_curve_pre_batch30_nonstandard import (
    maximum_scores_by_image as _historical_maximum_scores_by_image,
)
from src.evaluate import sha256_file
from src.stats.threshold_calibration import build_bootstrap_plan

FloatArray = NDArray[np.float64]

CLASSIFICATION = "NON_STANDARD_RAW_SCORE_THRESHOLD_UTILITY"
INVALID_DCA_REASON = "raw_detector_confidence_is_not_a_validated_exam_outcome_probability"
UTILITY_WEIGHT_FORMULA = "raw_score_threshold/(1-raw_score_threshold)"


def maximum_scores_by_image(
    records: Sequence[Mapping[str, Any]], image_ids: Sequence[str]
) -> FloatArray:
    """Expose the exact historical exam-marker construction for audit and tests."""

    return _historical_maximum_scores_by_image(records, image_ids)


RAW_SCORE_FIELDS = (
    "detector",
    "raw_score_threshold",
    "raw_score_utility_index",
    "raw_score_utility_ci_lower",
    "raw_score_utility_ci_upper",
    "mean_true_positive_images",
    "mean_false_positive_images",
    "mean_flagged_images",
    "all_action_raw_score_utility_index",
    "all_action_ci_lower",
    "all_action_ci_upper",
    "no_action_raw_score_utility_index",
    "no_action_ci_lower",
    "no_action_ci_upper",
    "raw_score_utility_difference_vs_other",
    "difference_ci_lower",
    "difference_ci_upper",
    "other_detector",
    "point_estimate_preferred_detector",
    "paired_ci_preferred_detector",
    "point_estimate_best_raw_score_strategy",
    "population",
    "population_unit",
    "image_count",
    "positive_image_count",
    "negative_image_count",
    "study_prevalence",
    "study_prevalence_percent",
    "patient_group_count",
    "seed_count",
    "seed_ids",
    "confidence_level",
    "bootstrap_resamples",
    "bootstrap_valid_resamples",
    "bootstrap_method",
    "seed_aggregation",
    "analysis_classification",
    "conventional_dca_interpretation_valid",
    "invalid_standard_dca_reason",
    "decision_rule",
    "score_interpretation",
    "utility_weight_formula",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    historical_config: Path
    validation_prediction_manifest: Path


class AnalysisSettings(StrictModel):
    classification: Literal["NON_STANDARD_RAW_SCORE_THRESHOLD_UTILITY"]
    conventional_dca_interpretation_valid: Literal[False]
    required_calibration_scope: Literal[
        "separate_validation_frozen_mapping_for_every_detector_test_run"
    ]
    expected_salvage_outcome: Literal["incomplete_run_specific_validation_predictions"]
    invalid_standard_dca_reason: Literal[
        "raw_detector_confidence_is_not_a_validated_exam_outcome_probability"
    ]


class PlotSettings(StrictModel):
    width_inches: float = Field(gt=0)
    height_inches: float = Field(gt=0)
    dpi: int = Field(ge=72, le=600)
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def validate_limits(self) -> PlotSettings:
        if not np.isfinite(self.y_min) or not np.isfinite(self.y_max):
            raise ValueError("plot limits must be finite")
        if self.y_max <= self.y_min:
            raise ValueError("plot y_max must be greater than y_min")
        return self


class OutputSettings(StrictModel):
    summary_table: Path
    utility_figure: Path
    summary_json: Path


class HistoricalArchive(StrictModel):
    role: str = Field(min_length=1)
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RawScoreUtilityConfig(StrictModel):
    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    analysis: AnalysisSettings
    plot: PlotSettings
    outputs: OutputSettings
    historical_archives: tuple[HistoricalArchive, ...] = Field(min_length=5)
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_raw_score_utility_config(path: str | Path) -> RawScoreUtilityConfig:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("raw-score utility config must contain a mapping")
    payload["project_root"] = source.parent.parent.resolve()
    payload["source_path"] = source
    return RawScoreUtilityConfig.model_validate(payload)


def raw_score_utility_index(
    true_positives: float | FloatArray,
    false_positives: float | FloatArray,
    sample_size: float | FloatArray,
    raw_score_threshold: float,
) -> float | FloatArray:
    """Reproduce the historical DCA-formula arithmetic without DCA semantics."""

    threshold = float(raw_score_threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("raw_score_threshold must lie strictly inside (0, 1)")
    tp, fp, size = np.broadcast_arrays(
        np.asarray(true_positives, dtype=np.float64),
        np.asarray(false_positives, dtype=np.float64),
        np.asarray(sample_size, dtype=np.float64),
    )
    if not np.isfinite(tp).all() or not np.isfinite(fp).all() or not np.isfinite(size).all():
        raise ValueError("counts and sample_size must be finite")
    if np.any(tp < 0) or np.any(fp < 0) or np.any(size <= 0) or np.any(tp + fp > size):
        raise ValueError("counts must be non-negative, disjoint, and within sample_size")
    result = tp / size - (fp / size) * threshold / (1 - threshold)
    return float(result) if result.ndim == 0 else result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _audit_archives(config: RawScoreUtilityConfig) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for archive in config.historical_archives:
        path = config.resolve(archive.path)
        actual = sha256_file(path)
        if actual != archive.sha256:
            raise ValueError(f"historical archive hash mismatch: {path}")
        audited.append({"role": archive.role, **_artifact(path, config.project_root)})
    return audited


def _historical_config(config: RawScoreUtilityConfig) -> HistoricalDecisionCurveConfig:
    historical = load_decision_curve_config(config.resolve(config.inputs.historical_config))
    if historical.analysis.score_interpretation != (
        "raw_detector_score_used_as_nominal_threshold_probability"
    ):
        raise ValueError("historical config no longer declares its raw-score construction")
    return historical


def _salvage_assessment(
    config: RawScoreUtilityConfig,
    data: FrozenDecisionData,
) -> dict[str, Any]:
    path = config.resolve(config.inputs.validation_prediction_manifest)
    manifest = _read_json(path)
    if manifest.get("status") != "complete":
        raise ValueError("validation prediction manifest is not complete")
    validation_grid: set[tuple[str, int]] = set()
    for record in manifest.get("runs", []):
        key = str(record.get("detector")), int(record.get("seed"))
        if key in validation_grid:
            raise ValueError(f"duplicate validation prediction identity: {key}")
        validation_grid.add(key)
        bundle = record.get("validation_bundle")
        if not isinstance(bundle, dict):
            raise ValueError(f"validation record lacks bundle provenance: {key}")
        bundle_path = config.resolve(Path(str(bundle.get("path", ""))))
        if sha256_file(bundle_path) != bundle.get("sha256"):
            raise ValueError(f"validation bundle hash mismatch: {bundle_path}")
    test_grid = {(bundle.detector, bundle.seed) for bundle in data.bundles}
    missing = sorted(test_grid - validation_grid)
    extra = sorted(validation_grid - test_grid)
    if not missing:
        raise ValueError(
            "configured removal path expected incomplete validation coverage, but every "
            "test run now has a frozen validation bundle; reassess before proceeding"
        )
    return {
        "required_scope": config.analysis.required_calibration_scope,
        "test_run_count": len(test_grid),
        "frozen_validation_run_count": len(validation_grid),
        "test_run_grid": [
            {"detector": detector, "seed": seed} for detector, seed in sorted(test_grid)
        ],
        "frozen_validation_run_grid": [
            {"detector": detector, "seed": seed} for detector, seed in sorted(validation_grid)
        ],
        "test_runs_without_frozen_validation_predictions": [
            {"detector": detector, "seed": seed} for detector, seed in missing
        ],
        "validation_runs_without_test_predictions": [
            {"detector": detector, "seed": seed} for detector, seed in extra
        ],
        "probability_calibration_salvage_valid_for_all_retained_runs": False,
        "calibrator_family_or_hyperparameters_selected": False,
        "test_dca_used_for_calibrator_selection": False,
        "path_taken": "preferred_removal_and_supplementary_demotion",
        "reason": (
            "Four retained detector/test runs lack frozen run-specific validation "
            "predictions, so the required validation-only probability mappings cannot "
            "be fitted and frozen for the complete five-seed analysis."
        ),
        "validation_prediction_manifest": _artifact(path, config.project_root),
    }


def _strategy_label(value: object) -> str:
    replacements = {
        "treat_all": "all_action_reference",
        "treat_none": "no_action_reference",
    }
    return "|".join(replacements.get(part, part) for part in str(value).split("|"))


def _relabel_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "detector": row["detector"],
        "raw_score_threshold": row["threshold_probability"],
        "raw_score_utility_index": row["net_benefit"],
        "raw_score_utility_ci_lower": row["net_benefit_ci_lower"],
        "raw_score_utility_ci_upper": row["net_benefit_ci_upper"],
        "mean_true_positive_images": row["mean_true_positive_images"],
        "mean_false_positive_images": row["mean_false_positive_images"],
        "mean_flagged_images": row["mean_flagged_images"],
        "all_action_raw_score_utility_index": row["treat_all_net_benefit"],
        "all_action_ci_lower": row["treat_all_ci_lower"],
        "all_action_ci_upper": row["treat_all_ci_upper"],
        "no_action_raw_score_utility_index": row["treat_none_net_benefit"],
        "no_action_ci_lower": row["treat_none_ci_lower"],
        "no_action_ci_upper": row["treat_none_ci_upper"],
        "raw_score_utility_difference_vs_other": row["net_benefit_difference_vs_other"],
        "difference_ci_lower": row["difference_ci_lower"],
        "difference_ci_upper": row["difference_ci_upper"],
        "other_detector": row["other_detector"],
        "point_estimate_preferred_detector": row["point_estimate_preferred_detector"],
        "paired_ci_preferred_detector": row["paired_ci_preferred_detector"],
        "point_estimate_best_raw_score_strategy": _strategy_label(
            row["point_estimate_best_strategy"]
        ),
        "population": row["population"],
        "population_unit": row["population_unit"],
        "image_count": row["image_count"],
        "positive_image_count": row["positive_image_count"],
        "negative_image_count": row["negative_image_count"],
        "study_prevalence": row["disease_prevalence"],
        "study_prevalence_percent": row["disease_prevalence_percent"],
        "patient_group_count": row["patient_group_count"],
        "seed_count": row["seed_count"],
        "seed_ids": row["seed_ids"],
        "confidence_level": row["confidence_level"],
        "bootstrap_resamples": row["bootstrap_resamples"],
        "bootstrap_valid_resamples": row["bootstrap_valid_resamples"],
        "bootstrap_method": row["bootstrap_method"],
        "seed_aggregation": row["seed_aggregation"],
        "analysis_classification": CLASSIFICATION,
        "conventional_dca_interpretation_valid": False,
        "invalid_standard_dca_reason": INVALID_DCA_REASON,
        "decision_rule": "maximum emitted box confidence >= raw score threshold",
        "score_interpretation": ("raw_detector_confidence_not_validated_exam_outcome_probability"),
        "utility_weight_formula": UTILITY_WEIGHT_FORMULA,
    }


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_SCORE_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in RAW_SCORE_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _display_name(detector: str) -> str:
    return {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}.get(
        detector, detector.replace("_", " ").title()
    )


def _atomic_figure(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    data: FrozenDecisionData,
    config: RawScoreUtilityConfig,
) -> Path:
    colors = {data.detectors[0]: "#2f6f9f", data.detectors[1]: "#d97706"}
    figure, axis = plt.subplots(
        figsize=(config.plot.width_inches, config.plot.height_inches), constrained_layout=True
    )
    for detector in data.detectors:
        series = sorted(
            (row for row in rows if row["detector"] == detector),
            key=lambda row: float(row["raw_score_threshold"]),
        )
        thresholds = np.asarray([row["raw_score_threshold"] for row in series])
        values = np.asarray([row["raw_score_utility_index"] for row in series])
        lower = np.asarray([row["raw_score_utility_ci_lower"] for row in series])
        upper = np.asarray([row["raw_score_utility_ci_upper"] for row in series])
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
        key=lambda row: float(row["raw_score_threshold"]),
    )
    thresholds = np.asarray([row["raw_score_threshold"] for row in reference])
    axis.plot(
        thresholds,
        [row["all_action_raw_score_utility_index"] for row in reference],
        color="#444444",
        linestyle="--",
        linewidth=1.6,
        label="All-action reference",
    )
    axis.axhline(0, color="#111111", linestyle=":", linewidth=1.5, label="No-action reference")
    axis.set_title(
        "Exploratory raw-score threshold utility/sensitivity\n"
        "NON-STANDARD: detector confidence is not an outcome probability"
    )
    axis.set_xlabel("Raw maximum detector-confidence threshold")
    axis.set_ylabel("Raw-score utility index (arbitrary score scale)")
    historical = _historical_config(config)
    axis.set_xlim(historical.analysis.threshold_start, historical.analysis.threshold_stop)
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


def _historical_numeric_audit(
    config: RawScoreUtilityConfig,
    generated_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    archive = next(
        item for item in config.historical_archives if item.role == "original_batch20_table"
    )
    with config.resolve(archive.path).open("r", encoding="utf-8", newline="") as handle:
        archived_rows = list(csv.DictReader(handle))
    if len(archived_rows) != len(generated_rows):
        raise ValueError("historical and regenerated raw-score row counts differ")
    numeric_fields = set(HISTORICAL_SUMMARY_FIELDS) - {
        "detector",
        "other_detector",
        "point_estimate_preferred_detector",
        "paired_ci_preferred_detector",
        "point_estimate_best_strategy",
        "population",
        "population_unit",
        "seed_ids",
        "bootstrap_method",
        "seed_aggregation",
        "decision_rule",
        "outcome_definition",
        "score_interpretation",
    }
    maximum_error = 0.0
    for archived, generated in zip(archived_rows, generated_rows, strict=True):
        for field in HISTORICAL_SUMMARY_FIELDS:
            if field in numeric_fields:
                maximum_error = max(
                    maximum_error,
                    abs(float(archived[field]) - float(generated[field])),
                )
            elif str(archived[field]) != str(generated[field]):
                raise ValueError(f"historical categorical field changed: {field}")
    if maximum_error > 1e-15:
        raise ValueError(f"historical raw-score arithmetic changed: {maximum_error}")
    return {
        "rows_compared": len(archived_rows),
        "fields_compared": len(HISTORICAL_SUMMARY_FIELDS),
        "maximum_absolute_numeric_difference": maximum_error,
        "categorical_fields_identical": True,
    }


def preflight(config: RawScoreUtilityConfig) -> dict[str, Any]:
    archives = _audit_archives(config)
    historical = _historical_config(config)
    data = _load_frozen_data(historical)
    salvage = _salvage_assessment(config, data)
    return {
        "status": "ready",
        "analysis_id": config.analysis_id,
        "analysis_classification": config.analysis.classification,
        "conventional_dca_interpretation_valid": False,
        "test_images": len(data.image_ids),
        "test_patient_groups": data.patient_clusters.patient_group_count,
        "test_positive_images": int(np.sum(data.outcome_positive)),
        "test_study_prevalence": float(np.mean(data.outcome_positive)),
        "test_prediction_bundles": len(data.bundles),
        "salvage_assessment": salvage,
        "historical_archives_verified": len(archives),
        "performs_training": False,
        "performs_inference": False,
        "fits_probability_calibrator": False,
        "accesses_test_outcomes_for_selection": False,
    }


def run_raw_score_utility(config: RawScoreUtilityConfig) -> dict[str, Any]:
    archives = _audit_archives(config)
    historical = _historical_config(config)
    data = _load_frozen_data(historical)
    salvage = _salvage_assessment(config, data)
    bootstrap_plan = build_bootstrap_plan(
        data.patient_clusters,
        seed_count=len(data.seeds),
        resamples=historical.analysis.bootstrap_resamples,
        base_seed=historical.seed,
        label=historical.analysis_id,
    )
    historical_rows = compute_decision_curves(data, historical, bootstrap_plan)
    numeric_audit = _historical_numeric_audit(config, historical_rows)
    rows = [_relabel_row(row) for row in historical_rows]
    table_path = _atomic_csv(config.resolve(config.outputs.summary_table), rows)
    figure_path = _atomic_figure(config.resolve(config.outputs.utility_figure), rows, data, config)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "analysis_classification": config.analysis.classification,
        "conventional_dca_interpretation_valid": False,
        "invalid_standard_dca_reason": config.analysis.invalid_standard_dca_reason,
        "paper_path": "removed_from_main_results_and_demoted_to_supplementary_limitations",
        "population": {
            "label": historical.analysis.population_label,
            "unit": historical.analysis.decision_unit,
            "image_count": len(data.image_ids),
            "positive_image_count": int(np.sum(data.outcome_positive)),
            "negative_image_count": int(np.sum(~data.outcome_positive)),
            "study_prevalence": float(np.mean(data.outcome_positive)),
            "patient_group_count": data.patient_clusters.patient_group_count,
            "enriched_internal_test_subset": True,
            "deployment_prevalence_sample": False,
            "external_clinical_utility_inference_valid": False,
        },
        "method": {
            "decision_rule": "maximum emitted box confidence >= raw score threshold",
            "score_interpretation": (
                "raw detector confidence; not a validated exam outcome probability"
            ),
            "utility_index_formula": "TP/N - (FP/N) * raw_tau/(1-raw_tau)",
            "utility_weight_formula": UTILITY_WEIGHT_FORMULA,
            "why_nonstandard": (
                "The action threshold and harm weight reuse the same raw detector score, "
                "although conventional DCA requires a threshold probability representing "
                "the decision-maker's harm/benefit trade-off."
            ),
            "bootstrap_method": historical.analysis.bootstrap_method,
            "bootstrap_resamples": historical.analysis.bootstrap_resamples,
            "confidence_interval": "pointwise two-sided percentile interval",
            "localization_used_in_action": False,
        },
        "salvage_assessment": salvage,
        "historical_arithmetic_audit": numeric_audit,
        "historical_archives": archives,
        "raw_score_point_estimate_preference_ranges": contiguous_ranges(
            historical_rows, key="point_estimate_preferred_detector"
        ),
        "raw_score_paired_difference_ranges": contiguous_ranges(
            historical_rows, key="paired_ci_preferred_detector"
        ),
        "raw_score_best_strategy_ranges": [
            {**item, "value": _strategy_label(item["value"])}
            for item in contiguous_ranges(historical_rows, key="point_estimate_best_strategy")
        ],
        "source": {
            "config": _artifact(config.source_path, config.project_root),
            "analysis_source": _artifact(Path(__file__).resolve(), config.project_root),
            "historical_config": _artifact(historical.source_path, config.project_root),
            "historical_analysis_source": _artifact(
                config.project_root
                / "src/clinical/archive/decision_curve_pre_batch30_nonstandard.py",
                config.project_root,
            ),
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
            "raw_score_utility_figure": _artifact(figure_path, config.project_root),
        },
        "performs_training": False,
        "performs_inference": False,
        "fits_probability_calibrator": False,
        "performs_standard_dca": False,
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    result = {
        "status": "complete",
        "analysis_classification": config.analysis.classification,
        "conventional_dca_interpretation_valid": False,
        "summary_table": table_path.as_posix(),
        "utility_figure": figure_path.as_posix(),
        "summary_json": summary_path.as_posix(),
        "salvage_path_taken": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/raw_score_utility.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_raw_score_utility_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True, allow_nan=False))
    else:
        run_raw_score_utility(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
