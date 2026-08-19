"""Audit YOLO seed convergence and score-scale stability from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import load_phase5_config, sha256_file
from src.meddet_benchmark.evaluation import ImagePrediction, evaluate_operating_point
from src.stats.run_statistics import load_coco_targets


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    phase5_config: Path
    phase5_summary: Path
    test_annotations: Path
    threshold_selection_summary: Path


class OutputSettings(StrictModel):
    table: Path
    summary_json: Path


class StabilityConfig(StrictModel):
    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    detector: Literal["yolo11s"]
    expected_seeds: tuple[int, ...]
    expected_confidence_degeneracy_seeds: tuple[int, ...]
    inputs: InputSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_contract(self) -> StabilityConfig:
        if not self.expected_seeds or len(set(self.expected_seeds)) != len(self.expected_seeds):
            raise ValueError("expected seeds must be non-empty and unique")
        if not set(self.expected_confidence_degeneracy_seeds) <= set(self.expected_seeds):
            raise ValueError("expected confidence-degeneracy seeds must be expected seeds")
        return self

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path).resolve()


TABLE_FIELDS = (
    "detector",
    "seed",
    "run_id",
    "training_status",
    "stop_reason",
    "completed_epochs",
    "best_epoch",
    "curve_nonfinite_value_count",
    "curve_all_zero_loss_epoch_count",
    "positive_val_map_50_95_epoch_count",
    "best_val_map_50",
    "best_val_map_50_95",
    "final_train_box_loss",
    "final_train_cls_loss",
    "final_train_dfl_loss",
    "final_val_map_50",
    "final_val_map_50_95",
    "convergence_classification",
    "maximum_test_score",
    "frozen_threshold",
    "frozen_prediction_count",
    "frozen_true_positives",
    "frozen_false_positives",
    "frozen_false_negatives",
    "frozen_precision",
    "frozen_recall",
    "frozen_f1",
    "test_ap_50",
    "test_ap_50_95",
    "historical_selected_threshold",
    "historical_selected_threshold_source_seed_count",
    "selected_prediction_count",
    "selected_true_positives",
    "selected_false_positives",
    "selected_false_negatives",
    "selected_precision",
    "selected_recall",
    "selected_f1",
    "coco_floor_threshold",
    "coco_floor_prediction_count",
    "coco_floor_true_positives",
    "coco_floor_false_positives",
    "coco_floor_false_negatives",
    "coco_floor_precision",
    "coco_floor_recall",
    "coco_floor_f1",
    "coco_floor_matched_mean_iou",
    "coco_floor_matched_mean_dice",
    "conditional_iou_dice_defined",
    "all_attempt_aggregation",
    "conditional_aggregation",
    "operational_diagnostic",
)

CURVE_FIELDS = (
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
)


def load_config(path: str | Path) -> StabilityConfig:
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return StabilityConfig.model_validate(
        {**payload, "project_root": source.parent.parent, "source_path": source}
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


def _predictions(payload: Mapping[str, Any]) -> list[ImagePrediction]:
    records = payload.get("predictions")
    if not isinstance(records, list):
        raise ValueError("prediction bundle lacks prediction records")
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


def summarize_training_curve(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Return fail-closed convergence indicators from Ultralytics epoch rows."""

    if not rows:
        raise ValueError("training curve is empty")
    values: list[dict[str, float]] = []
    nonfinite_count = 0
    for row in rows:
        parsed: dict[str, float] = {}
        for field in CURVE_FIELDS:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"training curve lacks numeric field {field}") from error
            if not math.isfinite(value):
                nonfinite_count += 1
            parsed[field] = value
        values.append(parsed)
    zero_loss_epochs = sum(all(item[field] == 0 for field in CURVE_FIELDS[:3]) for item in values)
    positive_map_epochs = sum(item["metrics/mAP50-95(B)"] > 0 for item in values)
    final = values[-1]
    return {
        "completed_epochs": len(values),
        "curve_nonfinite_value_count": nonfinite_count,
        "curve_all_zero_loss_epoch_count": zero_loss_epochs,
        "positive_val_map_50_95_epoch_count": positive_map_epochs,
        "best_val_map_50": max(item["metrics/mAP50(B)"] for item in values),
        "best_val_map_50_95": max(item["metrics/mAP50-95(B)"] for item in values),
        "final_train_box_loss": final["train/box_loss"],
        "final_train_cls_loss": final["train/cls_loss"],
        "final_train_dfl_loss": final["train/dfl_loss"],
        "final_val_map_50": final["metrics/mAP50(B)"],
        "final_val_map_50_95": final["metrics/mAP50-95(B)"],
    }


def classify_convergence(training_status: str, curve: Mapping[str, Any]) -> str:
    normal = (
        training_status == "complete"
        and int(curve["curve_nonfinite_value_count"]) == 0
        and int(curve["curve_all_zero_loss_epoch_count"]) == 0
        and int(curve["positive_val_map_50_95_epoch_count"]) > 0
        and float(curve["best_val_map_50_95"]) > 0
    )
    return "normal_converged" if normal else "collapse_or_invalid"


def classify_operational(
    *, convergence: str, frozen_prediction_count: int, frozen_true_positives: int, floor_tp: int
) -> str:
    if convergence != "normal_converged":
        return "training_collapse_or_invalid"
    if frozen_prediction_count == 0 and floor_tp > 0:
        return "confidence_score_degeneracy_at_frozen_threshold"
    if frozen_prediction_count > 0 and frozen_true_positives == 0:
        return "detections_without_iou_qualified_match"
    return "operational_detections_present"


def _operating(
    predictions: list[ImagePrediction],
    targets: Any,
    *,
    class_ids: tuple[int, ...],
    threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    return evaluate_operating_point(
        predictions,
        targets,
        class_ids=class_ids,
        score_threshold=threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )["overall"]


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
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
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
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


def run(config: StabilityConfig) -> dict[str, Any]:
    phase5_path = config.resolve(config.inputs.phase5_config)
    phase5_summary_path = config.resolve(config.inputs.phase5_summary)
    annotation_path = config.resolve(config.inputs.test_annotations)
    selection_path = config.resolve(config.inputs.threshold_selection_summary)
    phase5 = load_phase5_config(phase5_path)
    phase5_summary = _read_json(phase5_summary_path)
    selection = _read_json(selection_path)
    if phase5_summary.get("status") != "complete":
        raise ValueError("Phase 5 summary is not complete")
    if phase5_summary.get("config_sha256") != sha256_file(phase5_path):
        raise ValueError("Phase 5 config differs from its summary")
    if phase5_summary.get("test_annotation_sha256") != sha256_file(annotation_path):
        raise ValueError("test annotations differ from Phase 5")
    if selection.get("status") != "complete":
        raise ValueError("threshold-selection summary is not complete")

    selected_threshold = float(selection["selection"]["selected"][config.detector]["threshold"])
    selection_bundles = [
        item
        for item in selection["upstream"]["test_prediction_bundles"]
        if item["detector"] == config.detector
    ]
    selected_source_n = len(selection_bundles)
    run_specs = {run.seed: run for run in phase5.runs if run.detector == config.detector}
    summary_runs = {
        int(item["seed"]): item
        for item in phase5_summary["runs"]
        if item["detector"] == config.detector
    }
    if tuple(run_specs) != config.expected_seeds or set(summary_runs) != set(config.expected_seeds):
        raise ValueError("YOLO seed grid differs from the diagnostic contract")

    targets, category_names = load_coco_targets(annotation_path)
    class_ids = tuple(sorted(category_names))
    rows: list[dict[str, Any]] = []
    bundle_artifacts: list[dict[str, Any]] = []
    for seed in config.expected_seeds:
        spec = run_specs[seed]
        summary_run = summary_runs[seed]
        comparison = summary_run["comparison_row"]
        bundle_path = config.resolve(Path(comparison["prediction_bundle"]))
        if sha256_file(bundle_path) != summary_run["prediction_bundle_sha256"]:
            raise ValueError(f"prediction bundle hash differs for seed {seed}")
        bundle = _read_bundle(bundle_path)
        if bundle.get("detector") != config.detector or int(bundle.get("seed")) != seed:
            raise ValueError(f"prediction bundle identity differs for seed {seed}")
        predictions = _predictions(bundle)
        maximum_score = max(
            (float(np.max(item.scores)) for item in predictions if len(item.scores)),
            default=None,
        )
        if maximum_score is None or not math.isclose(
            maximum_score, float(comparison["maximum_prediction_score"]), rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError(f"maximum score differs from Phase 5 for seed {seed}")

        frozen = _operating(
            predictions,
            targets,
            class_ids=class_ids,
            threshold=phase5.evaluation.score_threshold,
            iou_threshold=phase5.evaluation.match_iou_threshold,
            max_detections=phase5.evaluation.max_detections,
        )
        for field, source_field in (
            ("prediction_count", "operating_point_prediction_count"),
            ("tp", "true_positives"),
            ("fp", "false_positives"),
            ("fn", "false_negatives"),
        ):
            if int(frozen[field]) != int(comparison[source_field]):
                raise ValueError(f"frozen metric differs from Phase 5 for seed {seed}: {field}")
        selected = _operating(
            predictions,
            targets,
            class_ids=class_ids,
            threshold=selected_threshold,
            iou_threshold=phase5.evaluation.match_iou_threshold,
            max_detections=phase5.evaluation.max_detections,
        )
        floor = _operating(
            predictions,
            targets,
            class_ids=class_ids,
            threshold=phase5.evaluation.coco_minimum_score,
            iou_threshold=phase5.evaluation.match_iou_threshold,
            max_detections=phase5.evaluation.max_detections,
        )

        training_summary_path = config.resolve(spec.training_summary)
        training = _read_json(training_summary_path)
        curve_path = training_summary_path.parent / "results.csv"
        with curve_path.open(newline="", encoding="utf-8") as handle:
            curve = summarize_training_curve(list(csv.DictReader(handle)))
        if int(training["completed_epochs"]) != int(curve["completed_epochs"]):
            raise ValueError(f"curve epoch count differs from summary for seed {seed}")
        convergence = classify_convergence(str(training.get("status")), curve)
        operational = classify_operational(
            convergence=convergence,
            frozen_prediction_count=int(frozen["prediction_count"]),
            frozen_true_positives=int(frozen["tp"]),
            floor_tp=int(floor["tp"]),
        )
        conditional_defined = int(frozen["tp"]) > 0
        rows.append(
            {
                "detector": config.detector,
                "seed": seed,
                "run_id": comparison["run_id"],
                "training_status": training["status"],
                "stop_reason": training["stop_reason"],
                **curve,
                "best_epoch": training["best_epoch"],
                "convergence_classification": convergence,
                "maximum_test_score": maximum_score,
                "frozen_threshold": phase5.evaluation.score_threshold,
                "frozen_prediction_count": frozen["prediction_count"],
                "frozen_true_positives": frozen["tp"],
                "frozen_false_positives": frozen["fp"],
                "frozen_false_negatives": frozen["fn"],
                "frozen_precision": frozen["precision"],
                "frozen_recall": frozen["recall"],
                "frozen_f1": frozen["f1"],
                "test_ap_50": comparison["map_50"],
                "test_ap_50_95": comparison["map_50_95"],
                "historical_selected_threshold": selected_threshold,
                "historical_selected_threshold_source_seed_count": selected_source_n,
                "selected_prediction_count": selected["prediction_count"],
                "selected_true_positives": selected["tp"],
                "selected_false_positives": selected["fp"],
                "selected_false_negatives": selected["fn"],
                "selected_precision": selected["precision"],
                "selected_recall": selected["recall"],
                "selected_f1": selected["f1"],
                "coco_floor_threshold": phase5.evaluation.coco_minimum_score,
                "coco_floor_prediction_count": floor["prediction_count"],
                "coco_floor_true_positives": floor["tp"],
                "coco_floor_false_positives": floor["fp"],
                "coco_floor_false_negatives": floor["fn"],
                "coco_floor_precision": floor["precision"],
                "coco_floor_recall": floor["recall"],
                "coco_floor_f1": floor["f1"],
                "coco_floor_matched_mean_iou": floor["matched_mean_iou"],
                "coco_floor_matched_mean_dice": floor["matched_mean_box_dice"],
                "conditional_iou_dice_defined": conditional_defined,
                "all_attempt_aggregation": "included",
                "conditional_aggregation": (
                    "included" if conditional_defined else "excluded_from_iou_dice_only"
                ),
                "operational_diagnostic": operational,
            }
        )
        bundle_artifacts.append(
            {
                "detector": config.detector,
                "seed": seed,
                **_artifact(bundle_path, config.project_root),
            }
        )

    observed_degenerate = tuple(
        int(row["seed"])
        for row in rows
        if row["operational_diagnostic"] == "confidence_score_degeneracy_at_frozen_threshold"
    )
    if observed_degenerate != config.expected_confidence_degeneracy_seeds:
        raise ValueError(
            "observed confidence-degeneracy seeds differ from the diagnostic contract: "
            f"observed={observed_degenerate}, "
            f"expected={config.expected_confidence_degeneracy_seeds}"
        )

    table_path = _atomic_csv(config.resolve(config.outputs.table), rows)
    summary = {
        "schema_version": config.schema_version,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "config": _artifact(config.source_path, config.project_root),
        "source": _artifact(
            config.project_root / "src" / "analyze_yolo_seed_stability.py",
            config.project_root,
        ),
        "inputs": {
            "phase5_config": _artifact(phase5_path, config.project_root),
            "phase5_summary": _artifact(phase5_summary_path, config.project_root),
            "test_annotations": _artifact(annotation_path, config.project_root),
            "threshold_selection_summary": _artifact(selection_path, config.project_root),
            "prediction_bundles": bundle_artifacts,
        },
        "thresholds": {
            "frozen": phase5.evaluation.score_threshold,
            "historical_validation_selected": selected_threshold,
            "historical_validation_selected_seed_count": selected_source_n,
            "coco_floor": phase5.evaluation.coco_minimum_score,
        },
        "expected_confidence_degeneracy_seeds": list(config.expected_confidence_degeneracy_seeds),
        "observed_confidence_degeneracy_seeds": list(observed_degenerate),
        "rows": rows,
        "artifact": _artifact(table_path, config.project_root),
        "interpretation": (
            "Confidence degeneracy requires normal convergence, no predictions at the frozen "
            "threshold, and at least one IoU-qualified true positive at the COCO score floor."
        ),
    }
    summary_path = config.resolve(config.outputs.summary_json)
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    _atomic_bytes(summary_path, encoded)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/yolo_seed_stability.yaml"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(load_config(args.config))
    print(
        json.dumps(
            {"status": result["status"], "artifact": result["artifact"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
