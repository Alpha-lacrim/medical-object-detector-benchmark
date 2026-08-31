"""Audit n=3 versus n=5 operating-regime evidence from frozen predictions only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import sha256_file
from src.plot_pareto_frontier import dominance_by_panel, load_pareto_config, load_pareto_points

FIXED_FIELDS = (
    "detector",
    "selection_split",
    "selection_rule",
    "tie_breaker",
    "selected_threshold",
    "threshold_selection_run_count",
    "test_run_count",
    "validation_precision",
    "validation_precision_std",
    "validation_recall",
    "validation_recall_std",
    "validation_f1",
    "validation_f1_std",
    "test_precision",
    "test_precision_std",
    "test_recall",
    "test_recall_std",
    "test_f1",
    "test_f1_std",
)
FIXED_PER_RUN_FIELDS = (
    "detector",
    "seed",
    "selection_split",
    "selection_rule",
    "selected_threshold",
    "threshold_selection_run_count",
    "test_run_count",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_true_positives",
    "test_false_positives",
    "test_false_negatives",
    "test_prediction_count",
    "test_target_count",
)
INVENTORY_FIELDS = (
    "analysis",
    "n5_recomputable",
    "frozen_input_scope",
    "historical_n3_artifact",
    "n5_sensitivity_artifact",
    "threshold_selection_run_count",
    "test_run_count",
    "performs_training",
    "reselects_threshold",
    "uses_test_for_tuning",
    "seed_271_policy",
)
CONCLUSION_FIELDS = (
    "analysis",
    "conclusion",
    "n3_run_count",
    "n5_run_count",
    "n3_result",
    "n5_result",
    "n3_directional_margin",
    "n5_directional_margin",
    "classification",
    "seed_271_effect",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Historical and n=5 frozen artifacts consumed by the audit."""

    decision_log: Path
    historical_selection_summary: Path
    historical_selected_table: Path
    historical_selected_per_seed_table: Path
    historical_threshold_summary: Path
    historical_threshold_table: Path
    historical_threshold_per_seed_table: Path
    historical_pr_table: Path
    historical_pr_per_seed_table: Path
    n5_threshold_config: Path
    n5_threshold_summary: Path
    n5_threshold_table: Path
    n5_threshold_per_seed_table: Path
    n5_pr_table: Path
    n5_pr_per_seed_table: Path
    historical_froc_summary: Path
    historical_froc_table: Path
    n5_froc_config: Path
    n5_froc_summary: Path
    n5_froc_table: Path
    historical_pareto_config: Path
    n5_pareto_config: Path
    n5_pareto_summary: Path


class AnalysisSettings(StrictModel):
    """Declared run grids and comparison tolerance."""

    expected_historical_seeds: tuple[int, ...]
    expected_sensitivity_seeds: tuple[int, ...]
    influence_seed: int
    numeric_tolerance: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_seed_sets(self) -> AnalysisSettings:
        for name, seeds in (
            ("expected_historical_seeds", self.expected_historical_seeds),
            ("expected_sensitivity_seeds", self.expected_sensitivity_seeds),
        ):
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError(f"analysis.{name} must be unique and non-empty")
        if not set(self.expected_historical_seeds) < set(self.expected_sensitivity_seeds):
            raise ValueError("historical seeds must be a strict subset of sensitivity seeds")
        if self.influence_seed not in self.expected_sensitivity_seeds:
            raise ValueError("influence_seed must be in expected_sensitivity_seeds")
        return self


class OutputSettings(StrictModel):
    """Versioned Batch 35 outputs."""

    fixed_threshold_table: Path
    fixed_threshold_per_seed_table: Path
    inventory_table: Path
    conclusion_table: Path
    summary_json: Path


class SensitivityConfig(StrictModel):
    """Strict Batch 35 operating-regime audit contract."""

    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    analysis: AnalysisSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_sensitivity_config(path: str | Path) -> SensitivityConfig:
    """Load and strictly validate the Batch 35 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("operating-regime sensitivity config must contain a mapping")
    payload["project_root"] = source.parent.parent.resolve()
    payload["source_path"] = source
    return SensitivityConfig.model_validate(payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required operating-regime artifact is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required operating-regime artifact has no rows: {path}")
    return rows


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
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


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _float(row: Mapping[str, str], field: str, *, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid {field!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"{source}: {field!r} must be finite")
    return value


def _integer(row: Mapping[str, str], field: str, *, source: Path) -> int:
    value = _float(row, field, source=source)
    integer = int(value)
    if integer != value:
        raise ValueError(f"{source}: {field!r} must be an integer")
    return integer


def _verify_summary_artifact(summary: Mapping[str, Any], name: str, path: Path) -> None:
    artifact = summary.get("artifacts", {}).get(name, {})
    if artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"summary hash mismatch for {name}: {path}")


def _seed_grid(
    rows: Sequence[Mapping[str, str]], *, seed_field: str = "seed"
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for row in rows:
        result.setdefault(str(row["detector"]), set()).add(int(row[seed_field]))
    return result


def load_fixed_threshold_inputs(
    config: SensitivityConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Validate historical threshold selection and the complete n=5 test sweep."""

    historical_summary_path = config.resolve(config.inputs.historical_selection_summary)
    historical_selected_path = config.resolve(config.inputs.historical_selected_table)
    historical_selected_per_seed_path = config.resolve(
        config.inputs.historical_selected_per_seed_table
    )
    historical_summary = _read_json(historical_summary_path)
    if historical_summary.get("status") != "complete":
        raise ValueError("historical threshold-selection summary is not complete")
    _verify_summary_artifact(
        historical_summary, "selected_operating_points_table", historical_selected_path
    )
    _verify_summary_artifact(
        historical_summary,
        "selected_operating_points_per_seed_table",
        historical_selected_per_seed_path,
    )
    historical_rows = _read_csv(historical_selected_path)
    historical_per_seed = _read_csv(historical_selected_per_seed_path)
    expected_historical = set(config.analysis.expected_historical_seeds)
    if any(seeds != expected_historical for seeds in _seed_grid(historical_per_seed).values()):
        raise ValueError("historical selected-threshold table has an unexpected seed grid")

    n5_config_path = config.resolve(config.inputs.n5_threshold_config)
    n5_summary_path = config.resolve(config.inputs.n5_threshold_summary)
    n5_per_seed_path = config.resolve(config.inputs.n5_threshold_per_seed_table)
    n5_summary = _read_json(n5_summary_path)
    if n5_summary.get("status") != "complete":
        raise ValueError("n=5 threshold-sweep summary is not complete")
    if n5_summary.get("config_sha256") != sha256_file(n5_config_path):
        raise ValueError("n=5 threshold config differs from its summary")
    _verify_summary_artifact(n5_summary, "threshold_per_seed_table", n5_per_seed_path)
    n5_per_seed = _read_csv(n5_per_seed_path)
    expected_n5 = set(config.analysis.expected_sensitivity_seeds)
    seed_grid = _seed_grid(n5_per_seed)
    if set(seed_grid) != {"faster_rcnn", "yolo11s"} or any(
        seeds != expected_n5 for seeds in seed_grid.values()
    ):
        raise ValueError(f"n=5 threshold sweep has an unexpected detector/run grid: {seed_grid}")
    return historical_rows, n5_per_seed, historical_summary, n5_summary


def build_fixed_threshold_tables(
    historical_rows: Sequence[Mapping[str, str]],
    n5_per_seed: Sequence[Mapping[str, str]],
    *,
    expected_test_seeds: Sequence[int],
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply historical validation-selected thresholds to every n=5 test run row."""

    expected = set(expected_test_seeds)
    per_run: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for historical in sorted(historical_rows, key=lambda row: row["detector"]):
        detector = str(historical["detector"])
        if historical.get("selection_split") != "validation":
            raise ValueError("frozen thresholds must have been selected on validation")
        threshold = float(historical["selected_threshold"])
        selected = [
            row
            for row in n5_per_seed
            if row["detector"] == detector
            and np.isclose(float(row["threshold"]), threshold, rtol=0, atol=tolerance)
        ]
        seeds = {int(row["seed"]) for row in selected}
        if seeds != expected or len(selected) != len(expected):
            raise ValueError(
                f"fixed threshold {threshold} does not retain every expected {detector} run"
            )
        selection_n = int(historical["seed_count"])
        for row in sorted(selected, key=lambda item: int(item["seed"])):
            per_run.append(
                {
                    "detector": detector,
                    "seed": int(row["seed"]),
                    "selection_split": "validation",
                    "selection_rule": historical["selection_rule"],
                    "selected_threshold": threshold,
                    "threshold_selection_run_count": selection_n,
                    "test_run_count": len(selected),
                    "test_precision": float(row["precision"]),
                    "test_recall": float(row["recall"]),
                    "test_f1": float(row["f1"]),
                    "test_true_positives": int(row["true_positives"]),
                    "test_false_positives": int(row["false_positives"]),
                    "test_false_negatives": int(row["false_negatives"]),
                    "test_prediction_count": int(row["prediction_count"]),
                    "test_target_count": int(row["target_count"]),
                }
            )
        metrics: dict[str, tuple[float, float]] = {}
        for metric in ("precision", "recall", "f1"):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            std = 0.0 if np.all(values == values[0]) else float(np.std(values, ddof=1))
            metrics[metric] = float(np.mean(values)), std
        aggregate.append(
            {
                "detector": detector,
                "selection_split": "validation",
                "selection_rule": historical["selection_rule"],
                "tie_breaker": historical["tie_breaker"],
                "selected_threshold": threshold,
                "threshold_selection_run_count": selection_n,
                "test_run_count": len(selected),
                **{
                    field: float(historical[field])
                    for field in (
                        "validation_precision",
                        "validation_precision_std",
                        "validation_recall",
                        "validation_recall_std",
                        "validation_f1",
                        "validation_f1_std",
                    )
                },
                **{f"test_{metric}": values[0] for metric, values in metrics.items()},
                **{f"test_{metric}_std": values[1] for metric, values in metrics.items()},
            }
        )
    return aggregate, per_run


def write_fixed_threshold_outputs(config: SensitivityConfig) -> dict[str, Any]:
    """Write n=5 test performance at unchanged n=3 validation-selected thresholds."""

    historical, n5_per_seed, historical_summary, n5_summary = load_fixed_threshold_inputs(config)
    aggregate, per_run = build_fixed_threshold_tables(
        historical,
        n5_per_seed,
        expected_test_seeds=config.analysis.expected_sensitivity_seeds,
        tolerance=config.analysis.numeric_tolerance,
    )
    aggregate_path = _atomic_csv(
        config.resolve(config.outputs.fixed_threshold_table), FIXED_FIELDS, aggregate
    )
    per_run_path = _atomic_csv(
        config.resolve(config.outputs.fixed_threshold_per_seed_table),
        FIXED_PER_RUN_FIELDS,
        per_run,
    )
    return {
        "aggregate": aggregate,
        "per_run": per_run,
        "historical_summary": historical_summary,
        "n5_summary": n5_summary,
        "artifacts": {
            "fixed_threshold_table": _artifact(aggregate_path, config.project_root),
            "fixed_threshold_per_seed_table": _artifact(per_run_path, config.project_root),
        },
    }


def classify_margin_change(old: float, new: float, *, tolerance: float) -> str:
    """Classify a prespecified directional margin from n=3 to n=5."""

    if (old > tolerance and new < -tolerance) or (old < -tolerance and new > tolerance):
        return "reversed"
    if np.isclose(old, new, rtol=0, atol=tolerance):
        return "unchanged"
    if abs(new) > abs(old) + tolerance and old * new > 0:
        return "strengthened"
    return "weakened"


def _metric_by_detector(
    rows: Sequence[Mapping[str, str]], field: str, *, threshold: float | None = None
) -> dict[str, float]:
    selected = rows
    if threshold is not None:
        selected = [
            row
            for row in rows
            if np.isclose(float(row["threshold"]), threshold, rtol=0, atol=1e-12)
        ]
    result = {str(row["detector"]): float(row[field]) for row in selected}
    if set(result) != {"faster_rcnn", "yolo11s"}:
        raise ValueError(f"missing detector values for {field}")
    return result


def _result_text(values: Mapping[str, float], *, digits: int = 6) -> str:
    return (
        f"Faster R-CNN={values['faster_rcnn']:.{digits}f}; YOLO11s={values['yolo11s']:.{digits}f}"
    )


def _conclusion_row(
    *,
    analysis: str,
    conclusion: str,
    old_values: Mapping[str, float],
    new_values: Mapping[str, float],
    preferred: str,
    n3: int,
    n5: int,
    seed_effect: str,
    tolerance: float,
) -> dict[str, Any]:
    alternative = "yolo11s" if preferred == "faster_rcnn" else "faster_rcnn"
    old_margin = old_values[preferred] - old_values[alternative]
    new_margin = new_values[preferred] - new_values[alternative]
    return {
        "analysis": analysis,
        "conclusion": conclusion,
        "n3_run_count": n3,
        "n5_run_count": n5,
        "n3_result": _result_text(old_values),
        "n5_result": _result_text(new_values),
        "n3_directional_margin": old_margin,
        "n5_directional_margin": new_margin,
        "classification": classify_margin_change(old_margin, new_margin, tolerance=tolerance),
        "seed_271_effect": seed_effect,
    }


def build_conclusion_rows(
    config: SensitivityConfig,
    fixed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the complete requested n=3-versus-n=5 conclusion audit."""

    n3 = len(config.analysis.expected_historical_seeds)
    n5 = len(config.analysis.expected_sensitivity_seeds)
    tolerance = config.analysis.numeric_tolerance
    n3_threshold_summary = _read_json(config.resolve(config.inputs.historical_threshold_summary))
    n5_threshold_summary = _read_json(config.resolve(config.inputs.n5_threshold_summary))
    n3_threshold = _read_csv(config.resolve(config.inputs.historical_threshold_table))
    n5_threshold = _read_csv(config.resolve(config.inputs.n5_threshold_table))
    reference = float(
        n3_threshold_summary["finding_summary"]["detectors"]["faster_rcnn"][
            "reference_operating_point"
        ]["threshold"]
    )
    rows: list[dict[str, Any]] = []
    seed_zero_note = (
        "YOLO11s seed 271 is retained with zero detections and defined zero metrics at 0.25."
    )
    for metric, preferred in (
        ("precision", "yolo11s"),
        ("recall", "faster_rcnn"),
        ("f1", "faster_rcnn"),
    ):
        preferred_label = "YOLO11s" if preferred == "yolo11s" else "Faster R-CNN"
        rows.append(
            _conclusion_row(
                analysis="shared-threshold score scale",
                conclusion=f"{preferred_label} has higher {metric} at score {reference:g}",
                old_values=_metric_by_detector(n3_threshold, metric, threshold=reference),
                new_values=_metric_by_detector(n5_threshold, metric, threshold=reference),
                preferred=preferred,
                n3=n3,
                n5=n5,
                seed_effect=seed_zero_note,
                tolerance=tolerance,
            )
        )

    n3_pr = _read_csv(config.resolve(config.inputs.historical_pr_table))
    n5_pr = _read_csv(config.resolve(config.inputs.n5_pr_table))
    for curve in ("ap50", "ap50_95"):
        old_values = {
            detector: next(
                float(row["average_precision"])
                for row in n3_pr
                if row["detector"] == detector and row["curve"] == curve
            )
            for detector in ("faster_rcnn", "yolo11s")
        }
        new_values = {
            detector: next(
                float(row["average_precision"])
                for row in n5_pr
                if row["detector"] == detector and row["curve"] == curve
            )
            for detector in ("faster_rcnn", "yolo11s")
        }
        rows.append(
            _conclusion_row(
                analysis="official precision-recall",
                conclusion=f"Faster R-CNN has higher mean {curve} (threshold-free ranking)",
                old_values=old_values,
                new_values=new_values,
                preferred="faster_rcnn",
                n3=n3,
                n5=n5,
                seed_effect=(
                    "YOLO11s seed 271 contributes its observed nonzero AP despite zero "
                    "detections at 0.25; no curve row is filtered."
                ),
                tolerance=tolerance,
            )
        )
    for curve in ("ap50", "ap50_95"):
        old_counts = n3_threshold_summary["finding_summary"]["official_curve_comparison"][curve]
        new_counts = n5_threshold_summary["finding_summary"]["official_curve_comparison"][curve]
        old_margin = float(
            old_counts["faster_higher_precision_points"]
            - old_counts["yolo_higher_precision_points"]
        )
        new_margin = float(
            new_counts["faster_higher_precision_points"]
            - new_counts["yolo_higher_precision_points"]
        )
        rows.append(
            {
                "analysis": "official precision-recall",
                "conclusion": f"Faster R-CNN leads more official {curve} recall positions",
                "n3_run_count": n3,
                "n5_run_count": n5,
                "n3_result": (
                    f"Faster={old_counts['faster_higher_precision_points']}; "
                    f"YOLO={old_counts['yolo_higher_precision_points']}; "
                    f"ties={old_counts['equal_precision_points']}"
                ),
                "n5_result": (
                    f"Faster={new_counts['faster_higher_precision_points']}; "
                    f"YOLO={new_counts['yolo_higher_precision_points']}; "
                    f"ties={new_counts['equal_precision_points']}"
                ),
                "n3_directional_margin": old_margin,
                "n5_directional_margin": new_margin,
                "classification": classify_margin_change(
                    old_margin, new_margin, tolerance=tolerance
                ),
                "seed_271_effect": "Seed 271 contributes its full observed 101-point curve.",
            }
        )

    n3_froc = _read_csv(config.resolve(config.inputs.historical_froc_table))
    n5_froc = _read_csv(config.resolve(config.inputs.n5_froc_table))
    budgets = sorted({float(row["fp_per_image_budget"]) for row in n3_froc})
    if budgets != sorted({float(row["fp_per_image_budget"]) for row in n5_froc}):
        raise ValueError("n=3 and n=5 FROC budget grids differ")
    for budget in budgets:
        rows.append(
            _conclusion_row(
                analysis="FROC",
                conclusion=f"Faster R-CNN has higher sensitivity at {budget:g} FP/image",
                old_values=_metric_by_detector(
                    [row for row in n3_froc if float(row["fp_per_image_budget"]) == budget],
                    "sensitivity",
                ),
                new_values=_metric_by_detector(
                    [row for row in n5_froc if float(row["fp_per_image_budget"]) == budget],
                    "sensitivity",
                ),
                preferred="faster_rcnn",
                n3=n3,
                n5=n5,
                seed_effect=(
                    "Seed 271 is retained; its best observed point at this budget is selected "
                    "from the same 0.01--0.99 grid."
                ),
                tolerance=tolerance,
            )
        )

    n3_fixed = _read_csv(config.resolve(config.inputs.historical_selected_table))
    n5_fixed = fixed["aggregate"]
    for metric in ("precision", "recall", "f1"):
        rows.append(
            _conclusion_row(
                analysis="frozen validation-selected threshold",
                conclusion=f"Faster R-CNN has higher test {metric} at 0.69 versus 0.05",
                old_values={row["detector"]: float(row[f"test_{metric}"]) for row in n3_fixed},
                new_values={row["detector"]: float(row[f"test_{metric}"]) for row in n5_fixed},
                preferred="faster_rcnn",
                n3=n3,
                n5=n5,
                seed_effect=(
                    "YOLO11s seed 271 is retained with zero detections and defined zero "
                    "precision/recall/F1 at the unchanged threshold 0.05."
                ),
                tolerance=tolerance,
            )
        )

    historical_pareto = dominance_by_panel(
        load_pareto_points(
            load_pareto_config(config.resolve(config.inputs.historical_pareto_config))
        )
    )
    sensitivity_pareto = dominance_by_panel(
        load_pareto_points(load_pareto_config(config.resolve(config.inputs.n5_pareto_config)))
    )
    for panel in sorted(historical_pareto):
        old = historical_pareto[panel]
        new = sensitivity_pareto[panel]
        rows.append(
            {
                "analysis": "Pareto",
                "conclusion": f"panel {panel}: neither detector strictly dominates",
                "n3_run_count": n3,
                "n5_run_count": n5,
                "n3_result": "neither" if old is None else str(old),
                "n5_result": "neither" if new is None else str(new),
                "n3_directional_margin": "",
                "n5_directional_margin": "",
                "classification": "unchanged" if old == new else "reversed",
                "seed_271_effect": (
                    "Seed 271 contributes observed AP, fixed-threshold recall, and same-run "
                    "hardware metrics to the n=5 cloud."
                ),
            }
        )
    return rows


def build_inventory_rows(config: SensitivityConfig) -> list[dict[str, Any]]:
    """Inventory the four offline n=5 recomputations and their historical counterparts."""

    common = {
        "n5_recomputable": True,
        "frozen_input_scope": "10 hash-bound test prediction bundles; 5 runs per detector",
        "threshold_selection_run_count": 3,
        "test_run_count": 5,
        "performs_training": False,
        "reselects_threshold": False,
        "uses_test_for_tuning": False,
        "seed_271_policy": (
            "included exactly as observed; zero-at-threshold metrics stay zero and "
            "threshold-free or low-threshold outputs stay nonzero where observed"
        ),
    }
    return [
        {
            "analysis": "PR",
            **common,
            "historical_n3_artifact": config.inputs.historical_pr_table.as_posix(),
            "n5_sensitivity_artifact": config.inputs.n5_pr_table.as_posix(),
        },
        {
            "analysis": "FROC",
            **common,
            "historical_n3_artifact": config.inputs.historical_froc_table.as_posix(),
            "n5_sensitivity_artifact": config.inputs.n5_froc_table.as_posix(),
        },
        {
            "analysis": "Pareto",
            **common,
            "historical_n3_artifact": "results/figures/pareto_frontier.png",
            "n5_sensitivity_artifact": "results/tables/pareto_points_n5_sensitivity.csv",
        },
        {
            "analysis": "fixed validation-selected threshold",
            **common,
            "historical_n3_artifact": config.inputs.historical_selected_table.as_posix(),
            "n5_sensitivity_artifact": config.outputs.fixed_threshold_table.as_posix(),
        },
    ]


def run_sensitivity(config: SensitivityConfig) -> dict[str, Any]:
    """Finalize the n=5 audit, conclusion comparison, and provenance summary."""

    fixed = write_fixed_threshold_outputs(config)
    conclusions = build_conclusion_rows(config, fixed)
    inventory = build_inventory_rows(config)
    inventory_path = _atomic_csv(
        config.resolve(config.outputs.inventory_table), INVENTORY_FIELDS, inventory
    )
    conclusion_path = _atomic_csv(
        config.resolve(config.outputs.conclusion_table), CONCLUSION_FIELDS, conclusions
    )
    required_inputs = {
        name: config.resolve(path)
        for name, path in config.inputs.model_dump().items()
        if name not in {"historical_pareto_config", "n5_pareto_config"}
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            Path(__file__).resolve().relative_to(config.project_root).as_posix(): sha256_file(
                Path(__file__).resolve()
            ),
            "src/evaluate_threshold_sweep.py": sha256_file(
                config.project_root / "src/evaluate_threshold_sweep.py"
            ),
            "src/plot_froc_curves.py": sha256_file(config.project_root / "src/plot_froc_curves.py"),
            "src/plot_pareto_frontier.py": sha256_file(
                config.project_root / "src/plot_pareto_frontier.py"
            ),
        },
        "protocol": {
            "historical_run_count": len(config.analysis.expected_historical_seeds),
            "sensitivity_run_count": len(config.analysis.expected_sensitivity_seeds),
            "historical_seeds": list(config.analysis.expected_historical_seeds),
            "sensitivity_seeds": list(config.analysis.expected_sensitivity_seeds),
            "influence_seed": config.analysis.influence_seed,
            "threshold_policy": (
                "retain 0.69/0.05 selected by maximum arithmetic mean validation F1 across "
                "the original three runs; apply without reselection to all five frozen test runs"
            ),
            "performs_training": False,
            "performs_checkpoint_loading": False,
            "performs_model_inference": False,
            "uses_test_for_threshold_selection": False,
        },
        "inventory": inventory,
        "conclusions": conclusions,
        "upstream": {
            **{
                name: _artifact(path, config.project_root) for name, path in required_inputs.items()
            },
            "historical_pareto_config": _artifact(
                config.resolve(config.inputs.historical_pareto_config), config.project_root
            ),
            "n5_pareto_config": _artifact(
                config.resolve(config.inputs.n5_pareto_config), config.project_root
            ),
        },
        "artifacts": {
            **fixed["artifacts"],
            "inventory_table": _artifact(inventory_path, config.project_root),
            "conclusion_table": _artifact(conclusion_path, config.project_root),
        },
    }
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Batch 35 command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/operating_regime_n5_sensitivity.yaml"),
    )
    parser.add_argument(
        "--mode", choices=("preflight", "fixed-thresholds", "run"), default="preflight"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, write fixed-threshold outputs, or finalize the complete audit."""

    args = build_parser().parse_args(argv)
    config = load_sensitivity_config(args.config)
    historical, n5_per_seed, _historical_summary, _n5_summary = load_fixed_threshold_inputs(config)
    ready = {
        "status": "ready",
        "historical_selected_rows": len(historical),
        "n5_threshold_rows_per_seed": len(n5_per_seed),
        "performs_training": False,
        "performs_inference": False,
    }
    if args.mode == "preflight":
        print(json.dumps(ready, indent=2, sort_keys=True))
    elif args.mode == "fixed-thresholds":
        result = write_fixed_threshold_outputs(config)
        print(json.dumps({"status": "complete", "artifacts": result["artifacts"]}, indent=2))
    else:
        run_sensitivity(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
