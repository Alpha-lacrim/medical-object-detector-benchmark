"""Reparameterize the frozen test threshold sweep as free-response ROC curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import sha256_file

FROC_OPERATING_FIELDS = (
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
    "selection_rule",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen Batch 10 artifacts consumed by the FROC redescription."""

    threshold_config: Path
    threshold_summary: Path
    threshold_per_seed_table: Path


class AnalysisSettings(StrictModel):
    """Predeclared false-positive budgets and numeric checks."""

    fp_per_image_budgets: tuple[float, ...]
    numeric_tolerance: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_budgets(self) -> AnalysisSettings:
        if not self.fp_per_image_budgets:
            raise ValueError("at least one FP/image budget is required")
        if any(value <= 0 for value in self.fp_per_image_budgets):
            raise ValueError("FP/image budgets must be positive")
        if tuple(sorted(set(self.fp_per_image_budgets))) != self.fp_per_image_budgets:
            raise ValueError("FP/image budgets must be unique and increasing")
        return self


class DetectorSettings(StrictModel):
    """Display settings for one detector."""

    label: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class PlotSettings(StrictModel):
    """Deterministic FROC figure settings."""

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
    """Required FROC artifacts."""

    operating_points_table: Path
    figure: Path
    summary_json: Path


class FrocConfig(StrictModel):
    """Strict top-level FROC analysis contract."""

    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    analysis: AnalysisSettings
    detectors: dict[str, DetectorSettings]
    plot: PlotSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def require_two_detectors(self) -> FrocConfig:
        if len(self.detectors) != 2:
            raise ValueError("FROC config must declare exactly two detectors")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_froc_config(path: str | Path) -> FrocConfig:
    """Load and strictly validate the FROC YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FROC config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return FrocConfig.model_validate(payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required FROC input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required FROC input has no rows: {path}")
    return rows


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
    if value != integer:
        raise ValueError(f"{source}: {field!r} must be an integer")
    return integer


def _sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("sample standard deviation requires at least two seeds")
    return 0.0 if np.all(array == array[0]) else float(np.std(array, ddof=1))


def load_froc_rows(config: FrocConfig) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Validate Batch 10 provenance and add FP/image to every seed-threshold row."""

    threshold_config_path = config.resolve(config.inputs.threshold_config)
    threshold_summary_path = config.resolve(config.inputs.threshold_summary)
    table_path = config.resolve(config.inputs.threshold_per_seed_table)
    summary = _read_json(threshold_summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Batch 10 threshold summary is not complete")
    if summary.get("config_sha256") != sha256_file(threshold_config_path):
        raise ValueError("Batch 10 config hash differs from its frozen summary")
    table_artifact = summary.get("artifacts", {}).get("threshold_per_seed_table", {})
    if table_artifact.get("sha256") != sha256_file(table_path):
        raise ValueError("Batch 10 per-seed threshold table hash mismatch")

    image_count = int(summary["counts"]["images_per_bundle"])
    expected_rows = int(summary["counts"]["threshold_rows_per_seed"])
    raw_rows = _read_csv(table_path)
    if len(raw_rows) != expected_rows:
        raise ValueError("Batch 10 per-seed threshold row count mismatch")
    converted: list[dict[str, Any]] = []
    for row in raw_rows:
        detector = row.get("detector", "")
        if detector not in config.detectors:
            raise ValueError(f"unexpected detector in FROC input: {detector!r}")
        false_positives = _integer(row, "false_positives", source=table_path)
        recall = _float(row, "recall", source=table_path)
        true_positives = _integer(row, "true_positives", source=table_path)
        false_negatives = _integer(row, "false_negatives", source=table_path)
        target_count = _integer(row, "target_count", source=table_path)
        if true_positives + false_negatives != target_count:
            raise ValueError("FROC source row has inconsistent target counts")
        if not np.isclose(
            recall,
            true_positives / target_count,
            atol=config.analysis.numeric_tolerance,
            rtol=0,
        ):
            raise ValueError("FROC sensitivity does not reproduce the source recall")
        converted.append(
            {
                "detector": detector,
                "seed": _integer(row, "seed", source=table_path),
                "threshold": _float(row, "threshold", source=table_path),
                "sensitivity": recall,
                "false_positives": false_positives,
                "fp_per_image": false_positives / image_count,
            }
        )

    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in converted:
        grouped[(row["detector"], row["seed"])].append(row)
    expected_thresholds = int(summary["analysis"]["threshold_count"])
    detector_seeds: dict[str, set[int]] = defaultdict(set)
    for (detector, seed), rows in grouped.items():
        detector_seeds[detector].add(seed)
        thresholds = {float(row["threshold"]) for row in rows}
        if len(rows) != expected_thresholds or len(thresholds) != expected_thresholds:
            raise ValueError(f"incomplete FROC threshold grid for {detector} seed {seed}")
    seed_sets = list(detector_seeds.values())
    if set(detector_seeds) != set(config.detectors) or any(
        seeds != seed_sets[0] for seeds in seed_sets[1:]
    ):
        raise ValueError("FROC detector seed grids differ")
    return converted, image_count, summary


def aggregate_froc_curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate common-threshold FROC coordinates across seeds."""

    grouped: defaultdict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["detector"]), float(row["threshold"]))].append(row)
    result: list[dict[str, Any]] = []
    for (detector, threshold), group in sorted(grouped.items()):
        sensitivity = [float(row["sensitivity"]) for row in group]
        fp_per_image = [float(row["fp_per_image"]) for row in group]
        result.append(
            {
                "detector": detector,
                "threshold": threshold,
                "seed_count": len(group),
                "sensitivity": float(np.mean(sensitivity)),
                "sensitivity_std": _sample_std(sensitivity),
                "fp_per_image": float(np.mean(fp_per_image)),
                "fp_per_image_std": _sample_std(fp_per_image),
            }
        )
    return result


def select_froc_operating_points(
    rows: Sequence[Mapping[str, Any]], budgets: Sequence[float], *, tolerance: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the highest seed-wise sensitivity at or below each FP/image budget."""

    grouped: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["detector"]), int(row["seed"]))].append(row)
    per_seed: list[dict[str, Any]] = []
    for (detector, seed), seed_rows in sorted(grouped.items()):
        for budget in budgets:
            candidates = [
                row for row in seed_rows if float(row["fp_per_image"]) <= budget + tolerance
            ]
            if not candidates:
                raise ValueError(f"no FROC point at or below {budget} FP/image for {detector}")
            best = max(
                candidates,
                key=lambda row: (
                    float(row["sensitivity"]),
                    -float(row["fp_per_image"]),
                    float(row["threshold"]),
                ),
            )
            per_seed.append(
                {
                    "detector": detector,
                    "seed": seed,
                    "fp_per_image_budget": float(budget),
                    "sensitivity": float(best["sensitivity"]),
                    "achieved_fp_per_image": float(best["fp_per_image"]),
                    "selected_threshold": float(best["threshold"]),
                }
            )

    grouped_selected: defaultdict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped_selected[(str(row["detector"]), float(row["fp_per_image_budget"]))].append(row)
    aggregate: list[dict[str, Any]] = []
    rule = (
        "per seed: maximum observed sensitivity with FP/image <= budget; no interpolation; "
        "ties use fewer FP/image then higher threshold"
    )
    for (detector, budget), group in sorted(grouped_selected.items()):
        sensitivities = [float(row["sensitivity"]) for row in group]
        achieved = [float(row["achieved_fp_per_image"]) for row in group]
        thresholds = [float(row["selected_threshold"]) for row in group]
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
                    f"{sensitivity_mean:.6g} ± {sensitivity_std:.6g}"
                ),
                "achieved_fp_per_image": float(np.mean(achieved)),
                "achieved_fp_per_image_std": _sample_std(achieved),
                "selected_threshold_mean": float(np.mean(thresholds)),
                "selected_threshold_min": min(thresholds),
                "selected_threshold_max": max(thresholds),
                "selection_rule": rule,
            }
        )
    return aggregate, per_seed


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
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
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


def plot_froc(
    curve_rows: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]],
    config: FrocConfig,
) -> Path:
    """Render mean test FROC curves and predeclared FP/image budget summaries."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure, axis = plt.subplots(
        figsize=(config.plot.figure_width, config.plot.figure_height), constrained_layout=True
    )
    try:
        for detector, settings in config.detectors.items():
            selected = sorted(
                (
                    row
                    for row in curve_rows
                    if row["detector"] == detector and float(row["fp_per_image"]) > 0
                ),
                key=lambda row: float(row["fp_per_image"]),
            )
            x = np.asarray([row["fp_per_image"] for row in selected], dtype=np.float64)
            sensitivity = np.asarray([row["sensitivity"] for row in selected], dtype=np.float64)
            sensitivity_std = np.asarray(
                [row["sensitivity_std"] for row in selected], dtype=np.float64
            )
            axis.plot(x, sensitivity, color=settings.color, linewidth=2, label=settings.label)
            axis.fill_between(
                x,
                np.clip(sensitivity - sensitivity_std, 0, 1),
                np.clip(sensitivity + sensitivity_std, 0, 1),
                color=settings.color,
                alpha=0.17,
                linewidth=0,
            )
            summaries = [row for row in operating_rows if row["detector"] == detector]
            axis.scatter(
                [row["fp_per_image_budget"] for row in summaries],
                [row["sensitivity"] for row in summaries],
                marker="D",
                s=45,
                facecolor="white",
                edgecolor=settings.color,
                linewidth=1.3,
                zorder=4,
            )

        power_ticks = 2.0 ** np.arange(
            np.ceil(np.log2(config.plot.minimum_x)),
            np.floor(np.log2(config.plot.maximum_x)) + 1,
        )
        axis.set_xscale("log", base=2)
        axis.set_xlim(config.plot.minimum_x, config.plot.maximum_x)
        axis.set_ylim(0, 1.0)
        axis.set_xticks(power_ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        axis.set_xlabel("Average false positives per image (log₂ scale)")
        axis.set_ylabel("Sensitivity (recall at IoU = 0.50)")
        axis.set_title("Free-response ROC from the frozen test threshold sweep")
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="lower right")
        axis.text(
            0.02,
            0.98,
            "Lines/bands: common-threshold mean ± sample SD\n"
            "Diamonds: non-interpolated per-seed budget summaries\n"
            "(x = budget; achieved FP/image may be lower)",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
        return _atomic_figure(config.resolve(config.outputs.figure), figure, dpi=config.plot.dpi)
    finally:
        plt.close(figure)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_froc(config: FrocConfig) -> dict[str, Any]:
    """Create FROC table, figure, and provenance summary without new inference."""

    rows, image_count, threshold_summary = load_froc_rows(config)
    curve_rows = aggregate_froc_curve(rows)
    operating_rows, per_seed_operating = select_froc_operating_points(
        rows,
        config.analysis.fp_per_image_budgets,
        tolerance=config.analysis.numeric_tolerance,
    )
    table_path = _atomic_csv(
        config.resolve(config.outputs.operating_points_table),
        FROC_OPERATING_FIELDS,
        operating_rows,
    )
    figure_path = plot_froc(curve_rows, operating_rows, config)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            Path(__file__).resolve().relative_to(config.project_root).as_posix(): sha256_file(
                Path(__file__).resolve()
            )
        },
        "upstream": {
            "threshold_config": _artifact(
                config.resolve(config.inputs.threshold_config), config.project_root
            ),
            "threshold_summary": _artifact(
                config.resolve(config.inputs.threshold_summary), config.project_root
            ),
            "threshold_per_seed_table": _artifact(
                config.resolve(config.inputs.threshold_per_seed_table), config.project_root
            ),
        },
        "protocol": {
            "source": "frozen Batch 10 test-set threshold sweep",
            "sensitivity": "operating-point recall at match IoU 0.50",
            "false_positives_per_image": "false-positive detections divided by all test images",
            "budgets": list(config.analysis.fp_per_image_budgets),
            "budget_selection": (
                "within each seed, maximum observed sensitivity at or below the budget; "
                "no interpolation or extrapolation"
            ),
            "aggregation": "arithmetic mean and sample standard deviation across seeds",
            "threshold_selection_for_deployment": False,
            "performs_training": False,
            "performs_checkpoint_loading": False,
            "performs_model_inference": False,
        },
        "counts": {
            "images_per_seed": image_count,
            "detectors": len(config.detectors),
            "seeds_per_detector": threshold_summary["counts"]["seeds_per_detector"],
            "source_rows": len(rows),
            "curve_rows": len(curve_rows),
            "operating_point_rows": len(operating_rows),
        },
        "operating_points": operating_rows,
        "operating_points_per_seed": per_seed_operating,
        "artifacts": {
            "operating_points_table": _artifact(table_path, config.project_root),
            "figure": _artifact(figure_path, config.project_root),
        },
    }
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the FROC command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/froc.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs or render the FROC artifacts."""

    args = build_parser().parse_args(argv)
    config = load_froc_config(args.config)
    rows, image_count, _summary = load_froc_rows(config)
    result = {
        "status": "ready",
        "rows": len(rows),
        "images_per_seed": image_count,
        "performs_training": False,
        "performs_inference": False,
    }
    if args.mode == "run":
        run_froc(config)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
