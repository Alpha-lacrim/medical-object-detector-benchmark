"""Render the offline accuracy-efficiency Pareto comparison from frozen CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Direction = Literal["higher", "lower"]


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen CSV inputs consumed by the Pareto analysis."""

    comparison_summary: Path
    comparison_per_seed: Path
    threshold_summary: Path
    threshold_per_seed: Path
    compute_tables: tuple[Path, ...]

    @model_validator(mode="after")
    def require_compute_tables(self) -> InputSettings:
        """Require at least one declared compute table."""
        if not self.compute_tables:
            raise ValueError("inputs.compute_tables must not be empty")
        return self


class AnalysisSettings(StrictModel):
    """Operating-point selection and numeric validation settings."""

    recall_operating_point: Literal["peak_mean_f1"]
    numeric_tolerance: float = Field(gt=0)


class DetectorSettings(StrictModel):
    """Display and publication-table mapping for one detector."""

    label: str = Field(min_length=1)
    summary_prefix: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class PlotSettings(StrictModel):
    """Figure dimensions and rendering settings."""

    dpi: int = Field(ge=72)
    figure_width: float = Field(gt=0)
    figure_height: float = Field(gt=0)
    marker_size: float = Field(gt=0)


class OutputSettings(StrictModel):
    """Declared Pareto output path."""

    figure: Path


class ParetoConfig(StrictModel):
    """Strict schema for the frozen-data Pareto plot."""

    schema_version: Literal[1]
    analysis_id: str = Field(min_length=1)
    inputs: InputSettings
    analysis: AnalysisSettings
    detectors: dict[str, DetectorSettings]
    plot: PlotSettings
    outputs: OutputSettings

    @model_validator(mode="after")
    def require_two_detectors(self) -> ParetoConfig:
        """Enforce the project's exactly-two-detector comparison contract."""
        if len(self.detectors) != 2:
            raise ValueError("detectors must declare exactly two entries")
        return self


@dataclass(frozen=True)
class ParetoPoint:
    """One seed-specific accuracy, operating-point, and compute observation."""

    detector: str
    seed: int
    run_id: str
    map_50_95: float
    recall: float
    recall_threshold: float
    throughput_fps: float
    mean_latency_ms: float
    total_parameters: int
    estimated_gflops: float


@dataclass(frozen=True)
class PanelSpec:
    """Axis definition for one Pareto panel."""

    letter: str
    title: str
    x_field: str
    y_field: str
    x_label: str
    y_label: str
    x_direction: Direction
    y_direction: Direction
    x_scale: float = 1.0


def load_pareto_config(path: str | Path) -> ParetoConfig:
    """Load and validate the Pareto YAML configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ParetoConfig.model_validate(raw)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and reject missing or empty artifacts."""
    if not path.is_file():
        raise FileNotFoundError(f"required Pareto input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required Pareto input has no data rows: {path}")
    return rows


def _float(row: Mapping[str, str], field: str, *, source: Path) -> float:
    """Parse a finite float from a CSV row."""
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid {field!r} value") from error
    if not np.isfinite(value):
        raise ValueError(f"{source}: {field!r} must be finite")
    return value


def _integer(row: Mapping[str, str], field: str, *, source: Path) -> int:
    """Parse an integer from a CSV row."""
    value = _float(row, field, source=source)
    integer = int(value)
    if value != integer:
        raise ValueError(f"{source}: {field!r} must be an integer")
    return integer


def _assert_close(
    actual: float,
    expected: float,
    *,
    tolerance: float,
    context: str,
) -> None:
    """Reject disagreements between frozen companion artifacts."""
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"{context}: {actual:.17g} != {expected:.17g}")


def _select_recall_thresholds(
    rows: Sequence[Mapping[str, str]],
    *,
    detectors: set[str],
    source: Path,
) -> dict[str, float]:
    """Select each detector's best observed mean-F1 threshold."""
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        detector = row.get("detector", "")
        if detector in detectors:
            grouped[detector].append(row)
    if set(grouped) != detectors:
        raise ValueError(f"{source}: threshold detectors do not match config")

    selected: dict[str, float] = {}
    for detector, detector_rows in grouped.items():
        best = min(
            detector_rows,
            key=lambda row: (
                -_float(row, "f1", source=source),
                _float(row, "threshold", source=source),
            ),
        )
        selected[detector] = _float(best, "threshold", source=source)
    return selected


def load_pareto_points(config: ParetoConfig) -> list[ParetoPoint]:
    """Join and cross-check all frozen accuracy, threshold, and compute artifacts."""
    detector_names = set(config.detectors)
    tolerance = config.analysis.numeric_tolerance
    comparison_rows = _read_csv(config.inputs.comparison_per_seed)
    comparison_by_run: dict[str, Mapping[str, str]] = {}
    for row in comparison_rows:
        detector = row.get("detector", "")
        if detector not in detector_names:
            raise ValueError(
                f"{config.inputs.comparison_per_seed}: unexpected detector {detector!r}"
            )
        run_id = row.get("run_id", "")
        if not run_id or run_id in comparison_by_run:
            raise ValueError(
                f"{config.inputs.comparison_per_seed}: duplicate or empty run_id {run_id!r}"
            )
        comparison_by_run[run_id] = row

    threshold_summary = _read_csv(config.inputs.threshold_summary)
    thresholds = _select_recall_thresholds(
        threshold_summary,
        detectors=detector_names,
        source=config.inputs.threshold_summary,
    )
    threshold_per_seed = _read_csv(config.inputs.threshold_per_seed)
    recall_by_key: dict[tuple[str, int], float] = {}
    for row in threshold_per_seed:
        detector = row.get("detector", "")
        if detector not in thresholds:
            continue
        threshold = _float(row, "threshold", source=config.inputs.threshold_per_seed)
        if not np.isclose(threshold, thresholds[detector], rtol=0.0, atol=tolerance):
            continue
        key = (detector, _integer(row, "seed", source=config.inputs.threshold_per_seed))
        if key in recall_by_key:
            raise ValueError(f"{config.inputs.threshold_per_seed}: duplicate row for {key}")
        recall_by_key[key] = _float(row, "recall", source=config.inputs.threshold_per_seed)

    points: list[ParetoPoint] = []
    seen_runs: set[str] = set()
    for compute_path in config.inputs.compute_tables:
        for compute_row in _read_csv(compute_path):
            run_id = compute_row.get("run_id", "")
            if run_id not in comparison_by_run:
                raise ValueError(f"{compute_path}: run_id {run_id!r} has no accuracy row")
            if run_id in seen_runs:
                raise ValueError(f"compute run_id appears more than once: {run_id}")
            seen_runs.add(run_id)
            comparison = comparison_by_run[run_id]
            detector = comparison["detector"]
            seed = _integer(comparison, "seed", source=config.inputs.comparison_per_seed)
            compute_seed = _integer(compute_row, "seed", source=compute_path)
            if seed != compute_seed:
                raise ValueError(f"{compute_path}: seed does not match {run_id}")
            key = (detector, seed)
            if key not in recall_by_key:
                raise ValueError(f"missing selected-threshold recall for {key}")
            points.append(
                ParetoPoint(
                    detector=detector,
                    seed=seed,
                    run_id=run_id,
                    map_50_95=_float(
                        comparison,
                        "map_50_95",
                        source=config.inputs.comparison_per_seed,
                    ),
                    recall=recall_by_key[key],
                    recall_threshold=thresholds[detector],
                    throughput_fps=_float(compute_row, "throughput_fps", source=compute_path),
                    mean_latency_ms=_float(compute_row, "mean_latency_ms", source=compute_path),
                    total_parameters=_integer(compute_row, "total_parameters", source=compute_path),
                    estimated_gflops=_float(compute_row, "estimated_gflops", source=compute_path),
                )
            )

    if seen_runs != set(comparison_by_run):
        missing = sorted(set(comparison_by_run) - seen_runs)
        raise ValueError(f"compute tables do not cover comparison runs: {missing}")
    _validate_seed_grid(points, detector_names)
    _validate_publication_summary(points, config)
    _validate_threshold_summary(points, threshold_summary, config)
    return sorted(points, key=lambda point: (point.detector, point.seed))


def _validate_seed_grid(points: Sequence[ParetoPoint], detectors: set[str]) -> None:
    """Require an identical seed grid for the two detector arms."""
    seeds_by_detector = {
        detector: {point.seed for point in points if point.detector == detector}
        for detector in detectors
    }
    seed_sets = list(seeds_by_detector.values())
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError(f"detector seed grids differ or are empty: {seeds_by_detector}")


def _validate_publication_summary(
    points: Sequence[ParetoPoint],
    config: ParetoConfig,
) -> None:
    """Verify the plotted seed-level AP values reproduce the publication table."""
    summary_rows = _read_csv(config.inputs.comparison_summary)
    map_rows = [row for row in summary_rows if row.get("metric") == "map_50_95"]
    if len(map_rows) != 1:
        raise ValueError(
            f"{config.inputs.comparison_summary}: expected one map_50_95 summary row"
        )
    row = map_rows[0]
    for detector, settings in config.detectors.items():
        observed = np.mean([point.map_50_95 for point in points if point.detector == detector])
        expected = _float(
            row,
            f"{settings.summary_prefix}_mean",
            source=config.inputs.comparison_summary,
        )
        _assert_close(
            float(observed),
            expected,
            tolerance=config.analysis.numeric_tolerance,
            context=f"{detector} mean mAP@0.5:0.95",
        )


def _validate_threshold_summary(
    points: Sequence[ParetoPoint],
    summary_rows: Sequence[Mapping[str, str]],
    config: ParetoConfig,
) -> None:
    """Verify selected seed recalls reproduce the threshold-sweep mean."""
    for detector in config.detectors:
        detector_points = [point for point in points if point.detector == detector]
        threshold = detector_points[0].recall_threshold
        matching = [
            row
            for row in summary_rows
            if row.get("detector") == detector
            and np.isclose(
                _float(row, "threshold", source=config.inputs.threshold_summary),
                threshold,
                rtol=0.0,
                atol=config.analysis.numeric_tolerance,
            )
        ]
        if len(matching) != 1:
            raise ValueError(f"missing unique threshold summary for {detector} at {threshold}")
        expected = _float(matching[0], "recall", source=config.inputs.threshold_summary)
        observed = float(np.mean([point.recall for point in detector_points]))
        _assert_close(
            observed,
            expected,
            tolerance=config.analysis.numeric_tolerance,
            context=f"{detector} selected-threshold mean recall",
        )


def _directed(value: float, direction: Direction) -> float:
    """Map an objective onto a higher-is-better scale."""
    return value if direction == "higher" else -value


def strict_detector_dominance(
    points: Sequence[ParetoPoint],
    *,
    x_field: str,
    y_field: str,
    x_direction: Direction,
    y_direction: Direction,
) -> str | None:
    """Return a detector only when both seed clouds are strictly ordered.

    Detector A strictly dominates B when every A seed is better than every B
    seed on both directed objectives. This intentionally conservative rule
    exposes seed overlap instead of declaring dominance from means alone.
    """
    detectors = sorted({point.detector for point in points})
    if len(detectors) != 2:
        raise ValueError("strict dominance requires exactly two detectors")
    for candidate, alternative in (detectors, detectors[::-1]):
        candidate_points = [point for point in points if point.detector == candidate]
        alternative_points = [point for point in points if point.detector == alternative]
        candidate_x = min(
            _directed(float(getattr(point, x_field)), x_direction)
            for point in candidate_points
        )
        alternative_x = max(
            _directed(float(getattr(point, x_field)), x_direction)
            for point in alternative_points
        )
        candidate_y = min(
            _directed(float(getattr(point, y_field)), y_direction)
            for point in candidate_points
        )
        alternative_y = max(
            _directed(float(getattr(point, y_field)), y_direction)
            for point in alternative_points
        )
        if candidate_x > alternative_x and candidate_y > alternative_y:
            return candidate
    return None


def panel_specs() -> tuple[PanelSpec, ...]:
    """Return the four requested benefit-resource panel definitions."""
    return (
        PanelSpec(
            letter="a",
            title="mAP@0.5:0.95 vs throughput",
            x_field="throughput_fps",
            y_field="map_50_95",
            x_label="Throughput (FPS; higher is better)",
            y_label="mAP@0.5:0.95 (higher is better)",
            x_direction="higher",
            y_direction="higher",
        ),
        PanelSpec(
            letter="b",
            title="Threshold-aware recall vs latency",
            x_field="mean_latency_ms",
            y_field="recall",
            x_label="Mean latency (ms/image; lower is better)",
            y_label="Recall at peak mean-F1 threshold (higher is better)",
            x_direction="lower",
            y_direction="higher",
        ),
        PanelSpec(
            letter="c",
            title="mAP@0.5:0.95 vs parameters",
            x_field="total_parameters",
            y_field="map_50_95",
            x_label="Parameters (millions; lower is better)",
            y_label="mAP@0.5:0.95 (higher is better)",
            x_direction="lower",
            y_direction="higher",
            x_scale=1_000_000.0,
        ),
        PanelSpec(
            letter="d",
            title="Threshold-aware recall vs estimated GFLOPs",
            x_field="estimated_gflops",
            y_field="recall",
            x_label="Estimated GFLOPs/image (lower is better)",
            y_label="Recall at peak mean-F1 threshold (higher is better)",
            x_direction="lower",
            y_direction="higher",
        ),
    )


def dominance_by_panel(points: Sequence[ParetoPoint]) -> dict[str, str | None]:
    """Classify strict detector-level dominance in all four panels."""
    return {
        panel.letter: strict_detector_dominance(
            points,
            x_field=panel.x_field,
            y_field=panel.y_field,
            x_direction=panel.x_direction,
            y_direction=panel.y_direction,
        )
        for panel in panel_specs()
    }


def _atomic_figure(path: Path, figure: Any, *, dpi: int) -> Path:
    """Save a figure atomically so partial files are never published."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def plot_pareto_frontier(
    points: Sequence[ParetoPoint],
    config: ParetoConfig,
) -> Path:
    """Render the requested four-panel seed-level Pareto figure."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    seeds = sorted({point.seed for point in points})
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    if len(seeds) > len(markers):
        raise ValueError(f"plot supports at most {len(markers)} distinct seeds")
    marker_by_seed = dict(zip(seeds, markers, strict=False))
    annotation_offsets = ((5, 5), (5, 5), (5, -12), (5, -12), (5, 5))
    offset_by_seed = {
        seed: annotation_offsets[index % len(annotation_offsets)]
        for index, seed in enumerate(seeds)
    }
    dominance = dominance_by_panel(points)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(config.plot.figure_width, config.plot.figure_height),
        constrained_layout=True,
    )
    try:
        for axis, panel in zip(axes.flat, panel_specs(), strict=True):
            for point in points:
                settings = config.detectors[point.detector]
                x_value = float(getattr(point, panel.x_field)) / panel.x_scale
                y_value = float(getattr(point, panel.y_field))
                axis.scatter(
                    x_value,
                    y_value,
                    s=config.plot.marker_size,
                    marker=marker_by_seed[point.seed],
                    facecolor=settings.color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.92,
                    zorder=3,
                )
                axis.annotate(
                    str(point.seed),
                    (x_value, y_value),
                    xytext=offset_by_seed[point.seed],
                    textcoords="offset points",
                    fontsize=8,
                    color=settings.color,
                )

            dominant = dominance[panel.letter]
            if dominant is None:
                dominance_text = "Pareto: neither detector dominates"
            else:
                dominance_text = f"Pareto: {config.detectors[dominant].label} dominates"
            axis.text(
                0.03,
                0.97,
                dominance_text,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88},
            )
            axis.set_title(f"({panel.letter}) {panel.title}", fontweight="bold")
            axis.set_xlabel(panel.x_label)
            axis.set_ylabel(panel.y_label)
            axis.grid(alpha=0.25, linewidth=0.8)
            axis.margins(x=0.16, y=0.16)

        detector_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=settings.color,
                markeredgecolor="white",
                markersize=9,
                label=settings.label,
            )
            for settings in config.detectors.values()
        ]
        seed_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_by_seed[seed],
                linestyle="none",
                color="#555555",
                markersize=8,
                label=f"seed {seed}",
            )
            for seed in seeds
        ]
        figure.legend(
            handles=[*detector_handles, *seed_handles],
            loc="outside lower center",
            ncol=len(detector_handles) + len(seed_handles),
            frameon=False,
        )
        thresholds = {
            config.detectors[detector].label: next(
                point.recall_threshold for point in points if point.detector == detector
            )
            for detector in config.detectors
        }
        threshold_text = "; ".join(
            f"{label} threshold={threshold:.2f}" for label, threshold in thresholds.items()
        )
        figure.suptitle(
            f"Accuracy-efficiency trade-offs across {len(seeds)} training seeds\n"
            f"Recall panels use each detector's peak mean-F1 sweep point ({threshold_text})",
            fontsize=14,
            fontweight="bold",
        )
        return _atomic_figure(config.outputs.figure, figure, dpi=config.plot.dpi)
    finally:
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the Pareto YAML config")
    parser.add_argument(
        "--mode",
        choices=("preflight", "run"),
        default="run",
        help="Validate inputs only or validate and render the figure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate frozen inputs and optionally render the Pareto figure."""
    args = build_parser().parse_args(argv)
    config = load_pareto_config(args.config)
    points = load_pareto_points(config)
    dominance = dominance_by_panel(points)
    thresholds = {
        detector: next(point.recall_threshold for point in points if point.detector == detector)
        for detector in config.detectors
    }
    print(
        f"Validated {len(points)} frozen seed rows; "
        f"thresholds={thresholds}; dominance={dominance}"
    )
    if args.mode == "run":
        output = plot_pareto_frontier(points, config)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
