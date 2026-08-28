"""Render audited seed-level raincloud plots for predictive and compute metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject undeclared configuration fields."""

    model_config = ConfigDict(extra="forbid")


class InputConfig(StrictModel):
    """Input table paths."""

    summary_table: Path
    per_seed_table: Path


class DetectorConfig(StrictModel):
    """Detector identity and presentation metadata."""

    key: str
    summary_prefix: str
    label: str
    color: str


class MetricConfig(StrictModel):
    """One source metric and its display transform."""

    key: str
    label: str
    group: Literal["predictive", "compute"]
    panel_index: int = Field(ge=0)
    display_unit: str
    scale: float = Field(gt=0.0)


class PlotConfig(StrictModel):
    """Raincloud layout and styling parameters."""

    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    width_inches: float = Field(gt=0.0)
    height_inches: float = Field(gt=0.0)
    dpi: int = Field(gt=0)
    violin_bandwidth_adjust: float = Field(gt=0.0)
    violin_width: float = Field(gt=0.0)
    box_width: float = Field(gt=0.0)
    point_size: float = Field(gt=0.0)
    point_jitter: float = Field(ge=0.0)
    panel_title_size: float = Field(gt=0.0)
    axis_label_size: float = Field(gt=0.0)
    tick_label_size: float = Field(gt=0.0)
    sample_label_size: float = Field(gt=0.0)


class OutputConfig(StrictModel):
    """Generated artifact paths."""

    figure: Path
    summary_json: Path


class RaincloudConfig(StrictModel):
    """Complete Batch 23 raincloud configuration."""

    schema_version: Literal[1]
    analysis_id: str
    seed: int
    inputs: InputConfig
    detectors: list[DetectorConfig]
    metrics: list[MetricConfig]
    plot: PlotConfig
    outputs: OutputConfig

    @model_validator(mode="after")
    def validate_identity(self) -> RaincloudConfig:
        """Require unique identities and enough subplot cells."""

        detector_keys = [detector.key for detector in self.detectors]
        prefixes = [detector.summary_prefix for detector in self.detectors]
        metric_keys = [metric.key for metric in self.metrics]
        panel_indices = [metric.panel_index for metric in self.metrics]
        if len(self.detectors) != 2:
            raise ValueError("exactly two detector entries are required")
        if len(detector_keys) != len(set(detector_keys)):
            raise ValueError("detector keys must be unique")
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("detector summary prefixes must be unique")
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("metric keys must be unique")
        if len(panel_indices) != len(set(panel_indices)):
            raise ValueError("metric panel indices must be unique")
        if max(panel_indices) >= self.plot.rows * self.plot.columns:
            raise ValueError("metric panel index lies outside the configured grid")
        return self


@dataclass(frozen=True)
class Observation:
    """One finite seed-level value used in the figure."""

    metric: str
    detector: str
    detector_label: str
    seed: int
    value: float


@dataclass(frozen=True)
class MetricAudit:
    """Aggregate-to-seed validation result for one metric."""

    metric: str
    counts: dict[str, int]
    attempted_counts: dict[str, int]
    undefined_seeds: dict[str, str]
    undefined_reasons: dict[str, str]
    sample_size_note: str


def load_config(path: Path) -> RaincloudConfig:
    """Load and strictly validate a raincloud YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return RaincloudConfig.model_validate(payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as stripped string dictionaries."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {key: value.strip() for key, value in row.items()} for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"CSV contains no records: {path}")
    return rows


def _finite_or_none(value: str, *, context: str) -> float | None:
    """Parse a finite float, preserving an explicitly blank undefined value."""

    if value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite value for {context}: {value!r}")
    return parsed


def _assert_close(actual: float, expected: float, *, context: str) -> None:
    """Reject aggregate values that drift from their seed-level source."""

    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"aggregate mismatch for {context}: table={actual}, seeds={expected}")


def load_and_audit_observations(
    config: RaincloudConfig,
) -> tuple[list[Observation], list[MetricAudit]]:
    """Load seed records and prove that their reductions match the publication table."""

    summary_rows = _read_csv(config.inputs.summary_table)
    seed_rows = _read_csv(config.inputs.per_seed_table)
    summary_by_metric = {row.get("metric", ""): row for row in summary_rows}
    if len(summary_by_metric) != len(summary_rows) or "" in summary_by_metric:
        raise ValueError("summary table metric keys must be present and unique")

    detector_by_key = {detector.key: detector for detector in config.detectors}
    observed_pairs: set[tuple[str, int]] = set()
    for row in seed_rows:
        detector = row.get("detector", "")
        if detector not in detector_by_key:
            raise ValueError(f"unexpected detector in seed table: {detector!r}")
        seed = int(row.get("seed", ""))
        pair = (detector, seed)
        if pair in observed_pairs:
            raise ValueError(f"duplicate detector/seed row: {pair}")
        observed_pairs.add(pair)

    observations: list[Observation] = []
    audits: list[MetricAudit] = []
    for metric in config.metrics:
        summary = summary_by_metric.get(metric.key)
        if summary is None:
            raise ValueError(f"configured metric missing from summary table: {metric.key}")
        counts: dict[str, int] = {}
        attempted_counts: dict[str, int] = {}
        undefined_seeds: dict[str, str] = {}
        undefined_reasons: dict[str, str] = {}
        for detector in config.detectors:
            detector_rows = [row for row in seed_rows if row["detector"] == detector.key]
            values: list[float] = []
            for row in detector_rows:
                if metric.key not in row:
                    raise ValueError(f"seed table lacks configured metric column: {metric.key}")
                value = _finite_or_none(
                    row[metric.key],
                    context=f"{metric.key}/{detector.key}/seed-{row['seed']}",
                )
                if value is None:
                    continue
                values.append(value)
                observations.append(
                    Observation(
                        metric=metric.key,
                        detector=detector.key,
                        detector_label=detector.label,
                        seed=int(row["seed"]),
                        value=value,
                    )
                )
            if not values:
                raise ValueError(f"metric has no finite values: {metric.key}/{detector.key}")

            prefix = detector.summary_prefix
            reported_n = int(summary[f"{prefix}_n"])
            attempted_n = int(summary[f"{prefix}_attempted_n"])
            if reported_n != len(values):
                raise ValueError(
                    f"sample-size mismatch for {metric.key}/{detector.key}: "
                    f"summary={reported_n}, finite seeds={len(values)}"
                )
            if attempted_n != len(detector_rows):
                raise ValueError(
                    f"attempted-n mismatch for {metric.key}/{detector.key}: "
                    f"summary={attempted_n}, rows={len(detector_rows)}"
                )
            seed_mean = statistics.fmean(values)
            seed_std = statistics.stdev(values) if len(values) > 1 else 0.0
            _assert_close(
                float(summary[f"{prefix}_mean"]),
                seed_mean,
                context=f"{metric.key}/{detector.key}/mean",
            )
            _assert_close(
                float(summary[f"{prefix}_std"]),
                seed_std,
                context=f"{metric.key}/{detector.key}/std",
            )
            counts[detector.key] = reported_n
            attempted_counts[detector.key] = attempted_n
            undefined_seeds[detector.key] = summary.get(f"{prefix}_undefined_seeds", "")
            undefined_reasons[detector.key] = summary.get(f"{prefix}_undefined_reason", "")

        audits.append(
            MetricAudit(
                metric=metric.key,
                counts=counts,
                attempted_counts=attempted_counts,
                undefined_seeds=undefined_seeds,
                undefined_reasons=undefined_reasons,
                sample_size_note=summary.get("sample_size_note", ""),
            )
        )
    return observations, audits


def _sample_label(audit: MetricAudit, config: RaincloudConfig) -> str:
    """Build an explicit per-panel sample-size label."""

    labels = [f"{detector.label} n={audit.counts[detector.key]}" for detector in config.detectors]
    suffixes: list[str] = []
    for detector in config.detectors:
        seed_text = audit.undefined_seeds[detector.key]
        reason = audit.undefined_reasons[detector.key]
        if seed_text:
            suffix = f"{detector.label}: seed {seed_text} undefined"
            if reason:
                suffix = f"{suffix} ({reason.replace('_', ' ')})"
            suffixes.append(suffix)
    first_line = " | ".join(labels)
    return first_line if not suffixes else f"{first_line}\n" + "; ".join(suffixes)


def render_raincloud(
    observations: list[Observation],
    audits: list[MetricAudit],
    config: RaincloudConfig,
) -> Path:
    """Render cloud, box, and seed-point layers for every configured metric."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    metric_by_key = {metric.key: metric for metric in config.metrics}
    audit_by_key = {audit.metric: audit for audit in audits}
    detector_order = [detector.label for detector in config.detectors]
    palette = {detector.label: detector.color for detector in config.detectors}
    records = [
        {
            "metric": observation.metric,
            "detector": observation.detector_label,
            "seed": observation.seed,
            "value": observation.value * metric_by_key[observation.metric].scale,
        }
        for observation in observations
    ]
    frame = pd.DataFrame.from_records(records)
    np.random.seed(config.seed)
    sns.set_theme(style="whitegrid", context="paper")

    figure, axes = plt.subplots(
        config.plot.rows,
        config.plot.columns,
        figsize=(config.plot.width_inches, config.plot.height_inches),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    used_panel_indices: set[int] = set()
    for metric in config.metrics:
        axis = flat_axes[metric.panel_index]
        used_panel_indices.add(metric.panel_index)
        metric_frame = frame.loc[frame["metric"] == metric.key]
        sns.violinplot(
            data=metric_frame,
            x="detector",
            y="value",
            hue="detector",
            order=detector_order,
            hue_order=detector_order,
            palette=palette,
            inner=None,
            cut=0,
            density_norm="width",
            common_norm=False,
            bw_adjust=config.plot.violin_bandwidth_adjust,
            width=config.plot.violin_width,
            linewidth=0.8,
            legend=False,
            ax=axis,
        )
        sns.boxplot(
            data=metric_frame,
            x="detector",
            y="value",
            hue="detector",
            order=detector_order,
            hue_order=detector_order,
            palette=palette,
            width=config.plot.box_width,
            whis=(0, 100),
            showfliers=False,
            fill=False,
            linewidth=1.1,
            legend=False,
            ax=axis,
        )
        sns.stripplot(
            data=metric_frame,
            x="detector",
            y="value",
            hue="detector",
            order=detector_order,
            hue_order=detector_order,
            palette=palette,
            size=config.plot.point_size,
            jitter=config.plot.point_jitter,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            legend=False,
            ax=axis,
        )
        axis.margins(y=0.18)
        axis.set_title(
            metric.label,
            fontsize=config.plot.panel_title_size,
            fontweight="bold",
            pad=18,
        )
        axis.set_xlabel("")
        axis.set_ylabel(metric.display_unit, fontsize=config.plot.axis_label_size)
        axis.tick_params(axis="both", labelsize=config.plot.tick_label_size)
        axis.text(
            0.5,
            0.985,
            _sample_label(audit_by_key[metric.key], config),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=config.plot.sample_label_size,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
        )
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for index, axis in enumerate(flat_axes):
        if index not in used_panel_indices:
            axis.set_visible(False)
    figure.suptitle(
        "Seed-level clean-test predictive and compute distributions",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.974,
        "Cloud = seed distribution; box = median and range; rain = individual seeds. "
        "Zero-variance clouds collapse to their box/points; every panel states actual finite n.",
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.948), h_pad=2.1, w_pad=1.4)

    destination = config.outputs.figure
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    figure.savefig(
        temporary,
        dpi=config.plot.dpi,
        bbox_inches="tight",
        metadata={"Software": "matplotlib/seaborn"},
    )
    plt.close(figure)
    temporary.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(
    config: RaincloudConfig,
    audits: list[MetricAudit],
    figure_path: Path,
) -> Path:
    """Write deterministic provenance for the audited figure."""

    payload = {
        "schema_version": 1,
        "analysis_id": config.analysis_id,
        "inputs": {
            "summary_table": str(config.inputs.summary_table).replace("\\", "/"),
            "summary_table_sha256": _sha256(config.inputs.summary_table),
            "per_seed_table": str(config.inputs.per_seed_table).replace("\\", "/"),
            "per_seed_table_sha256": _sha256(config.inputs.per_seed_table),
        },
        "figure": str(figure_path).replace("\\", "/"),
        "figure_sha256": _sha256(figure_path),
        "metrics": [
            {
                "metric": audit.metric,
                "counts": audit.counts,
                "attempted_counts": audit.attempted_counts,
                "undefined_seeds": audit.undefined_seeds,
                "undefined_reasons": audit.undefined_reasons,
                "sample_size_note": audit.sample_size_note,
            }
            for audit in audits
        ],
    }
    destination = config.outputs.summary_json
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def run(config: RaincloudConfig) -> tuple[Path, Path]:
    """Audit the source tables, render the figure, and write provenance."""

    observations, audits = load_and_audit_observations(config)
    figure_path = render_raincloud(observations, audits, config)
    summary_path = write_summary(config, audits, figure_path)
    return figure_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), default="run")
    return parser


def main() -> None:
    """Run the configured Batch 23 raincloud workflow."""

    args = build_parser().parse_args()
    config = load_config(args.config)
    observations, audits = load_and_audit_observations(config)
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "analysis_id": config.analysis_id,
                    "observation_count": len(observations),
                    "metric_counts": {audit.metric: audit.counts for audit in audits},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    figure_path = render_raincloud(observations, audits, config)
    summary_path = write_summary(config, audits, figure_path)
    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
