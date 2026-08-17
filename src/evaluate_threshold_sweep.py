"""Offline confidence-threshold and COCO precision-recall analysis of Phase 5 bundles."""

from __future__ import annotations

import argparse
import csv
import gzip
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

from src.evaluate import evaluate_operating_point, load_phase5_config, sha256_file
from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import ImagePrediction
from src.stats.run_statistics import load_coco_targets

DetectorName = Literal["faster_rcnn", "yolo11s"]
CurveName = Literal["ap50", "ap50_95"]

THRESHOLD_PER_SEED_FIELDS = (
    "detector",
    "seed",
    "threshold",
    "precision",
    "recall",
    "f1",
    "true_positives",
    "false_positives",
    "false_negatives",
    "prediction_count",
    "target_count",
)
THRESHOLD_FIELDS = (
    "detector",
    "threshold",
    "seed_count",
    "precision",
    "precision_std",
    "precision_mean_plus_minus_std",
    "recall",
    "recall_std",
    "recall_mean_plus_minus_std",
    "f1",
    "f1_std",
    "f1_mean_plus_minus_std",
)
PR_PER_SEED_FIELDS = (
    "detector",
    "seed",
    "curve",
    "iou_definition",
    "recall",
    "precision",
    "average_precision",
)
PR_FIELDS = (
    "detector",
    "curve",
    "iou_definition",
    "recall",
    "seed_count",
    "precision",
    "precision_std",
    "average_precision",
    "average_precision_std",
)
OPERATING_TARGET_FIELDS = (
    "detector",
    "target_metric",
    "target_value",
    "response_metric",
    "status",
    "threshold",
    "response",
    "response_std",
    "seeds_reaching_target",
    "seed_count",
    "selection_rule",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen Phase 5 inputs consumed by the offline analysis."""

    phase5_config: Path
    phase5_summary: Path
    test_annotations: Path


class SweepSettings(StrictModel):
    """Confidence grid and predeclared operating targets."""

    start: float = Field(ge=0, le=1)
    stop: float = Field(ge=0, le=1)
    steps: int = Field(ge=2, le=1001)
    precision_targets: tuple[float, ...]
    recall_targets: tuple[float, ...]

    @model_validator(mode="after")
    def validate_grid_and_targets(self) -> SweepSettings:
        if self.stop <= self.start:
            raise ValueError("sweep stop must be greater than start")
        for name, targets in (
            ("precision_targets", self.precision_targets),
            ("recall_targets", self.recall_targets),
        ):
            if not targets or len(set(targets)) != len(targets):
                raise ValueError(f"{name} must contain unique values")
            if any(not 0 < value <= 1 for value in targets):
                raise ValueError(f"{name} values must be in (0, 1]")
        return self

    def thresholds(self) -> tuple[float, ...]:
        """Return the inclusive, numerically stable threshold grid."""

        return tuple(
            float(value) for value in np.round(np.linspace(self.start, self.stop, self.steps), 12)
        )


class PlotSettings(StrictModel):
    """Deterministic figure rendering settings."""

    dpi: int = Field(ge=72, le=600)


class OutputSettings(StrictModel):
    """Generated tables, figures, and provenance summary."""

    log_dir: Path
    summary_json: Path
    threshold_table: Path
    threshold_per_seed_table: Path
    precision_recall_table: Path
    precision_recall_per_seed_table: Path
    operating_targets_table: Path
    precision_recall_figure: Path
    f1_figure: Path


class ThresholdSweepConfig(StrictModel):
    """Strict Batch 10 offline-analysis contract."""

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    sweep: SweepSettings
    plots: PlotSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_threshold_sweep_config(path: str | Path) -> ThresholdSweepConfig:
    """Load and strictly validate the Batch 10 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold-sweep config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return ThresholdSweepConfig.model_validate(payload)


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
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, raw)


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
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


def _resolved_input(config: ThresholdSweepConfig, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config.resolve(path)


def _validate_upstream(
    config: ThresholdSweepConfig,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], Any, dict[int, str]]:
    phase5_config_path = config.resolve(config.inputs.phase5_config)
    phase5_summary_path = config.resolve(config.inputs.phase5_summary)
    annotation_path = config.resolve(config.inputs.test_annotations)
    phase5 = load_phase5_config(phase5_config_path)
    summary = _read_json(phase5_summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Phase 5 summary is not complete")
    if summary.get("config_sha256") != sha256_file(phase5_config_path):
        raise ValueError("Phase 5 config hash differs from its frozen summary")
    if summary.get("experiment_id") != phase5.experiment_id:
        raise ValueError("Phase 5 experiment identity differs from its frozen summary")
    if summary.get("test_annotation_sha256") != sha256_file(annotation_path):
        raise ValueError("test annotations differ from the Phase 5 evaluation source")
    if summary.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("Phase 5 evaluator settings differ from its frozen summary")
    if phase5.evaluation.coco_minimum_score > config.sweep.start:
        raise ValueError("frozen bundles discard scores required by the configured sweep")
    if not any(
        np.isclose(value, phase5.evaluation.score_threshold) for value in config.sweep.thresholds()
    ):
        raise ValueError("threshold grid must include the frozen Phase 5 operating point")

    targets, category_names = load_coco_targets(annotation_path)
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
        predictions = _deserialize_predictions(payload)
        if len(predictions) != len(targets):
            raise ValueError(f"Phase 5 bundle image count mismatch: {bundle_path}")
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
        raise ValueError("Phase 5 summary does not contain the complete detector/seed grid")
    bundles.sort(key=lambda item: (str(item["detector"]), int(item["seed"])))
    return phase5, summary, bundles, targets, category_names


def sweep_prediction_records(
    predictions: list[ImagePrediction],
    targets: Any,
    *,
    detector: str,
    seed: int,
    thresholds: Sequence[float],
    class_ids: tuple[int, ...],
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Evaluate a frozen prediction set through the canonical operating-point matcher."""

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        overall = evaluate_operating_point(
            predictions,
            targets,
            class_ids=class_ids,
            score_threshold=float(threshold),
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )["overall"]
        rows.append(
            {
                "detector": detector,
                "seed": seed,
                "threshold": float(threshold),
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "true_positives": overall["tp"],
                "false_positives": overall["fp"],
                "false_negatives": overall["fn"],
                "prediction_count": overall["prediction_count"],
                "target_count": overall["target_count"],
            }
        )
    return rows


def _sample_std(values: np.ndarray, ddof: int) -> float:
    if len(values) <= ddof or not np.isfinite(values).all():
        raise ValueError("cannot compute sample standard deviation")
    return 0.0 if np.all(values == values[0]) else float(np.std(values, ddof=ddof))


def aggregate_threshold_rows(
    rows: Sequence[Mapping[str, Any]], *, ddof: int
) -> list[dict[str, Any]]:
    """Aggregate threshold metrics across seeds with mean and sample SD."""

    grouped: defaultdict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["detector"]), float(row["threshold"]))].append(row)
    result: list[dict[str, Any]] = []
    for (detector, threshold), group in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "detector": detector,
            "threshold": threshold,
            "seed_count": len(group),
        }
        for metric in ("precision", "recall", "f1"):
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            mean = float(np.mean(values))
            std = _sample_std(values, ddof)
            entry[metric] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_mean_plus_minus_std"] = f"{mean:.6g} ± {std:.6g}"
        result.append(entry)
    return result


def _precision_recall_rows(
    bundles: Sequence[Mapping[str, Any]],
    targets: Any,
    category_names: dict[int, str],
    *,
    minimum_score: float,
    max_detections: int,
    ddof: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    class_ids = tuple(sorted(category_names))
    per_seed_rows: list[dict[str, Any]] = []
    coco_runs: list[dict[str, Any]] = []
    definitions = {
        "ap50": ("precision_iou_50", "IoU=0.50"),
        "ap50_95": ("precision_iou_50_95", "mean IoU=0.50:0.95"),
    }
    for bundle in bundles:
        result = evaluate_coco(
            bundle["predictions"],
            targets,
            class_ids=class_ids,
            class_names=category_names,
            minimum_score=minimum_score,
            max_detections=max_detections,
            include_precision_recall=True,
        )
        original = bundle["payload"]["coco"]
        for metric in ("ap50", "ap50_95"):
            if not np.isclose(result[metric], original[metric], atol=5e-12, rtol=0):
                raise ValueError(
                    f"COCO {metric} did not reproduce for {bundle['detector']} "
                    f"seed {bundle['seed']}"
                )
        curve = result["precision_recall"]
        for curve_name, (field, definition) in definitions.items():
            for recall, precision in zip(curve["recall"], curve[field], strict=True):
                per_seed_rows.append(
                    {
                        "detector": bundle["detector"],
                        "seed": bundle["seed"],
                        "curve": curve_name,
                        "iou_definition": definition,
                        "recall": recall,
                        "precision": precision,
                        "average_precision": result[curve_name],
                    }
                )
        coco_runs.append(
            {
                "detector": bundle["detector"],
                "seed": bundle["seed"],
                "ap50": result["ap50"],
                "ap50_95": result["ap50_95"],
                "prediction_count": result["prediction_count"],
            }
        )

    grouped: defaultdict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[
            (
                str(row["detector"]),
                str(row["curve"]),
                str(row["iou_definition"]),
                float(row["recall"]),
            )
        ].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for (detector, curve_name, definition, recall), group in sorted(grouped.items()):
        precision = np.asarray([float(row["precision"]) for row in group], dtype=np.float64)
        average_precision = np.asarray(
            [float(row["average_precision"]) for row in group], dtype=np.float64
        )
        aggregate_rows.append(
            {
                "detector": detector,
                "curve": curve_name,
                "iou_definition": definition,
                "recall": recall,
                "seed_count": len(group),
                "precision": float(np.mean(precision)),
                "precision_std": _sample_std(precision, ddof),
                "average_precision": float(np.mean(average_precision)),
                "average_precision_std": _sample_std(average_precision, ddof),
            }
        )
    return per_seed_rows, aggregate_rows, coco_runs


def select_operating_targets(
    aggregate_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    precision_targets: Sequence[float],
    recall_targets: Sequence[float],
) -> list[dict[str, Any]]:
    """Select non-interpolated fixed-target points from each detector's mean curve."""

    detectors = sorted({str(row["detector"]) for row in aggregate_rows})
    result: list[dict[str, Any]] = []
    for detector in detectors:
        mean_rows = [row for row in aggregate_rows if row["detector"] == detector]
        seed_rows = [row for row in per_seed_rows if row["detector"] == detector]
        seed_count = len({int(row["seed"]) for row in seed_rows})
        requests = [
            *(("precision", value, "recall") for value in precision_targets),
            *(("recall", value, "precision") for value in recall_targets),
        ]
        for target_metric, target_value, response_metric in requests:
            candidates = [row for row in mean_rows if float(row[target_metric]) >= target_value]
            seed_reachability = {
                int(row["seed"]) for row in seed_rows if float(row[target_metric]) >= target_value
            }
            selection_rule = (
                f"maximum mean {response_metric} among swept thresholds with mean "
                f"{target_metric} >= target; no interpolation"
            )
            if not candidates:
                result.append(
                    {
                        "detector": detector,
                        "target_metric": target_metric,
                        "target_value": target_value,
                        "response_metric": response_metric,
                        "status": "not_reachable_at_any_swept_threshold",
                        "threshold": None,
                        "response": None,
                        "response_std": None,
                        "seeds_reaching_target": len(seed_reachability),
                        "seed_count": seed_count,
                        "selection_rule": selection_rule,
                    }
                )
                continue
            best = max(
                candidates,
                key=lambda row: (
                    float(row[response_metric]),
                    float(row[target_metric]),
                    float(row["threshold"]),
                ),
            )
            result.append(
                {
                    "detector": detector,
                    "target_metric": target_metric,
                    "target_value": target_value,
                    "response_metric": response_metric,
                    "status": "reachable",
                    "threshold": best["threshold"],
                    "response": best[response_metric],
                    "response_std": best[f"{response_metric}_std"],
                    "seeds_reaching_target": len(seed_reachability),
                    "seed_count": seed_count,
                    "selection_rule": selection_rule,
                }
            )
    return result


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


def _plot_precision_recall(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    dpi: int,
    seed_count: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    colors = {"faster_rcnn": "#1f77b4", "yolo11s": "#d62728"}
    labels = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    panels = (
        ("ap50", "IoU = 0.50 (underlying AP@0.5)"),
        ("ap50_95", "Mean IoU = 0.50:0.95 (underlying AP@0.5:0.95)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    try:
        for axis, (curve_name, title) in zip(axes, panels, strict=True):
            for detector in ("faster_rcnn", "yolo11s"):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["detector"] == detector and row["curve"] == curve_name
                    ),
                    key=lambda row: float(row["recall"]),
                )
                recall = np.asarray([row["recall"] for row in selected], dtype=np.float64)
                precision = np.asarray([row["precision"] for row in selected], dtype=np.float64)
                std = np.asarray([row["precision_std"] for row in selected], dtype=np.float64)
                ap = float(selected[0]["average_precision"])
                ap_std = float(selected[0]["average_precision_std"])
                axis.plot(
                    recall,
                    precision,
                    color=colors[detector],
                    linewidth=2,
                    label=f"{labels[detector]} (AP {ap:.3f} ± {ap_std:.3f})",
                )
                axis.fill_between(
                    recall,
                    np.clip(precision - std, 0, 1),
                    np.clip(precision + std, 0, 1),
                    color=colors[detector],
                    alpha=0.18,
                    linewidth=0,
                )
            axis.set_title(title)
            axis.set_xlabel("Recall")
            axis.set_ylabel("Interpolated precision")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right")
        figure.suptitle(
            f"Official COCO precision-recall curves (mean ± sample SD, {seed_count} seeds)"
        )
        return _atomic_figure(path, figure, dpi=dpi)
    finally:
        plt.close(figure)


def _plot_f1(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    dpi: int,
    reference_threshold: float,
    threshold_start: float,
    threshold_stop: float,
    seed_count: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    colors = {"faster_rcnn": "#1f77b4", "yolo11s": "#d62728"}
    labels = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    figure, axis = plt.subplots(figsize=(8.7, 5.6), constrained_layout=True)
    try:
        upper = 0.0
        for detector in ("faster_rcnn", "yolo11s"):
            selected = sorted(
                (row for row in rows if row["detector"] == detector),
                key=lambda row: float(row["threshold"]),
            )
            thresholds = np.asarray([row["threshold"] for row in selected], dtype=np.float64)
            f1 = np.asarray([row["f1"] for row in selected], dtype=np.float64)
            std = np.asarray([row["f1_std"] for row in selected], dtype=np.float64)
            axis.plot(
                thresholds,
                f1,
                color=colors[detector],
                linewidth=2,
                label=labels[detector],
            )
            axis.fill_between(
                thresholds,
                np.clip(f1 - std, 0, 1),
                np.clip(f1 + std, 0, 1),
                color=colors[detector],
                alpha=0.18,
                linewidth=0,
            )
            best_index = int(np.argmax(f1))
            axis.scatter(
                [thresholds[best_index]],
                [f1[best_index]],
                color=colors[detector],
                marker="o",
                s=42,
                zorder=3,
            )
            upper = max(upper, float(np.max(f1 + std)))
        axis.axvline(
            reference_threshold,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"Main threshold {reference_threshold:g}",
        )
        axis.set_title(
            f"F1 across frozen confidence thresholds (mean ± sample SD, {seed_count} seeds)"
        )
        axis.set_xlabel("Confidence threshold")
        axis.set_ylabel("F1 at match IoU = 0.50")
        axis.set_xlim(threshold_start, threshold_stop)
        axis.set_ylim(0, min(1.0, max(0.1, upper * 1.12)))
        axis.grid(alpha=0.25)
        axis.legend()
        return _atomic_figure(path, figure, dpi=dpi)
    finally:
        plt.close(figure)


def _artifact(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _finding_summary(
    threshold_rows: Sequence[Mapping[str, Any]],
    pr_rows: Sequence[Mapping[str, Any]],
    *,
    reference_threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"detectors": {}}
    for detector in ("faster_rcnn", "yolo11s"):
        selected = [row for row in threshold_rows if row["detector"] == detector]
        fixed = next(
            row for row in selected if np.isclose(float(row["threshold"]), reference_threshold)
        )
        peak = max(selected, key=lambda row: float(row["f1"]))
        result["detectors"][detector] = {
            "reference_operating_point": {
                "threshold": reference_threshold,
                **{metric: fixed[metric] for metric in ("precision", "recall", "f1")},
            },
            "maximum_mean_recall": max(float(row["recall"]) for row in selected),
            "peak_mean_f1": peak["f1"],
            "peak_mean_f1_threshold": peak["threshold"],
        }
    tolerance = 1e-12
    result["official_curve_comparison"] = {}
    for curve_name in ("ap50", "ap50_95"):
        by_detector = {
            detector: {
                float(row["recall"]): float(row["precision"])
                for row in pr_rows
                if row["detector"] == detector and row["curve"] == curve_name
            }
            for detector in ("faster_rcnn", "yolo11s")
        }
        faster = by_detector["faster_rcnn"]
        yolo = by_detector["yolo11s"]
        common = sorted(set(faster) & set(yolo))
        result["official_curve_comparison"][curve_name] = {
            "recall_grid_points": len(common),
            "faster_higher_precision_points": sum(
                faster[value] > yolo[value] + tolerance for value in common
            ),
            "equal_precision_points": sum(
                abs(faster[value] - yolo[value]) <= tolerance for value in common
            ),
            "yolo_higher_precision_points": sum(
                yolo[value] > faster[value] + tolerance for value in common
            ),
            "maximum_recall_with_positive_mean_precision": {
                detector: max(
                    recall for recall, precision in values.items() if precision > tolerance
                )
                for detector, values in by_detector.items()
            },
        }
    return result


def preflight(config: ThresholdSweepConfig) -> dict[str, Any]:
    """Verify all frozen inputs and the complete six-bundle grid."""

    phase5, _summary, bundles, targets, _category_names = _validate_upstream(config)
    return {
        "status": "ready",
        "phase5_experiment_id": phase5.experiment_id,
        "bundle_count": len(bundles),
        "image_count_per_bundle": len(targets),
        "threshold_count": len(config.sweep.thresholds()),
        "performs_training": False,
        "performs_inference": False,
    }


def run_threshold_sweep(config: ThresholdSweepConfig) -> dict[str, Any]:
    """Reprocess six frozen bundles into threshold and official COCO PR artifacts."""

    phase5, phase5_summary, bundles, targets, category_names = _validate_upstream(config)
    thresholds = config.sweep.thresholds()
    class_ids = tuple(sorted(category_names))
    per_seed_threshold: list[dict[str, Any]] = []
    for bundle in bundles:
        print(
            f"sweeping {bundle['detector']} seed {bundle['seed']} "
            f"over {len(thresholds)} thresholds",
            flush=True,
        )
        rows = sweep_prediction_records(
            bundle["predictions"],
            targets,
            detector=str(bundle["detector"]),
            seed=int(bundle["seed"]),
            thresholds=thresholds,
            class_ids=class_ids,
            iou_threshold=phase5.evaluation.match_iou_threshold,
            max_detections=phase5.evaluation.max_detections,
        )
        original = bundle["payload"]["operating_point"]["overall"]
        fixed = next(row for row in rows if np.isclose(row["threshold"], 0.25))
        for metric in ("precision", "recall", "f1"):
            if not np.isclose(fixed[metric], original[metric], atol=5e-12, rtol=0):
                raise ValueError(
                    f"fixed-threshold {metric} did not reproduce for "
                    f"{bundle['detector']} seed {bundle['seed']}"
                )
        per_seed_threshold.extend(rows)
    threshold_rows = aggregate_threshold_rows(
        per_seed_threshold, ddof=phase5.runtime.statistics_ddof
    )
    pr_per_seed, pr_rows, coco_runs = _precision_recall_rows(
        bundles,
        targets,
        category_names,
        minimum_score=phase5.evaluation.coco_minimum_score,
        max_detections=phase5.evaluation.max_detections,
        ddof=phase5.runtime.statistics_ddof,
    )
    operating_targets = select_operating_targets(
        threshold_rows,
        per_seed_threshold,
        precision_targets=config.sweep.precision_targets,
        recall_targets=config.sweep.recall_targets,
    )

    threshold_path = _atomic_csv(
        config.resolve(config.outputs.threshold_table), THRESHOLD_FIELDS, threshold_rows
    )
    threshold_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.threshold_per_seed_table),
        THRESHOLD_PER_SEED_FIELDS,
        per_seed_threshold,
    )
    pr_path = _atomic_csv(config.resolve(config.outputs.precision_recall_table), PR_FIELDS, pr_rows)
    pr_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.precision_recall_per_seed_table),
        PR_PER_SEED_FIELDS,
        pr_per_seed,
    )
    targets_path = _atomic_csv(
        config.resolve(config.outputs.operating_targets_table),
        OPERATING_TARGET_FIELDS,
        operating_targets,
    )
    pr_figure_path = _plot_precision_recall(
        config.resolve(config.outputs.precision_recall_figure),
        pr_rows,
        dpi=config.plots.dpi,
        seed_count=len(phase5.seeds),
    )
    f1_figure_path = _plot_f1(
        config.resolve(config.outputs.f1_figure),
        threshold_rows,
        dpi=config.plots.dpi,
        reference_threshold=phase5.evaluation.score_threshold,
        threshold_start=config.sweep.start,
        threshold_stop=config.sweep.stop,
        seed_count=len(phase5.seeds),
    )

    source_paths = (
        Path(__file__).resolve(),
        config.project_root / "src" / "evaluate.py",
        config.project_root / "src" / "meddet_benchmark" / "evaluation.py",
        config.project_root / "src" / "meddet_benchmark" / "coco_evaluation.py",
        config.project_root / "src" / "stats" / "run_statistics.py",
    )
    summary = {
        "schema_version": config.schema_version,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            path.relative_to(config.project_root).as_posix(): sha256_file(path)
            for path in source_paths
        },
        "upstream": {
            "phase5_summary": _artifact(
                config.resolve(config.inputs.phase5_summary), config.project_root
            ),
            "phase5_config": _artifact(
                config.resolve(config.inputs.phase5_config), config.project_root
            ),
            "test_annotations": _artifact(
                config.resolve(config.inputs.test_annotations), config.project_root
            ),
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
        "analysis": {
            "thresholds": list(thresholds),
            "threshold_count": len(thresholds),
            "aggregation": "arithmetic mean and sample standard deviation across seeds",
            "statistics_ddof": phase5.runtime.statistics_ddof,
            "matcher": (
                "src.evaluate.evaluate_operating_point, imported there from "
                "src.meddet_benchmark.evaluation"
            ),
            "match_iou_threshold": phase5.evaluation.match_iou_threshold,
            "max_detections_per_image": phase5.evaluation.max_detections,
            "coco_precision_recall": (
                "official pycocotools 101-point interpolated precision tensor; "
                "AP@0.5 and IoU-averaged AP@0.5:0.95 views"
            ),
            "precision_targets": list(config.sweep.precision_targets),
            "recall_targets": list(config.sweep.recall_targets),
            "performs_training": False,
            "performs_inference": False,
        },
        "counts": {
            "detectors": 2,
            "seeds_per_detector": len(phase5.seeds),
            "prediction_bundles": len(bundles),
            "images_per_bundle": len(targets),
            "annotations": phase5_summary["annotation_count"],
            "threshold_rows_per_seed": len(per_seed_threshold),
            "threshold_rows_aggregate": len(threshold_rows),
            "precision_recall_rows_per_seed": len(pr_per_seed),
            "precision_recall_rows_aggregate": len(pr_rows),
        },
        "coco_reproduction": coco_runs,
        "operating_targets": operating_targets,
        "finding_summary": _finding_summary(
            threshold_rows,
            pr_rows,
            reference_threshold=phase5.evaluation.score_threshold,
        ),
        "artifacts": {
            "threshold_table": _artifact(threshold_path, config.project_root),
            "threshold_per_seed_table": _artifact(threshold_per_seed_path, config.project_root),
            "precision_recall_table": _artifact(pr_path, config.project_root),
            "precision_recall_per_seed_table": _artifact(pr_per_seed_path, config.project_root),
            "operating_targets_table": _artifact(targets_path, config.project_root),
            "precision_recall_figure": _artifact(pr_figure_path, config.project_root),
            "f1_figure": _artifact(f1_figure_path, config.project_root),
        },
    }
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the offline threshold-analysis command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/threshold_sweep.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight or the complete offline threshold analysis."""

    args = build_parser().parse_args(argv)
    config = load_threshold_sweep_config(args.config)
    result = preflight(config) if args.mode == "preflight" else run_threshold_sweep(config)
    if args.mode == "preflight":
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
