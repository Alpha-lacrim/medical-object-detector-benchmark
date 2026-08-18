"""Select detector thresholds on validation and evaluate them once on frozen test bundles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import (
    _collect_faster_rcnn_predictions,
    _collect_yolo_predictions,
    _targets_from_dataset,
    evaluate_prediction_records,
    load_and_validate_training_configs,
    load_phase5_config,
    sha256_file,
)
from src.evaluate_threshold_sweep import (
    _deserialize_predictions,
    _read_bundle,
    _validate_upstream,
    aggregate_threshold_rows,
    load_threshold_sweep_config,
    sweep_prediction_records,
)
from src.meddet_benchmark.evaluation import ImagePrediction

VALIDATION_PER_SEED_FIELDS = (
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
VALIDATION_FIELDS = (
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
SELECTED_FIELDS = (
    "detector",
    "selection_split",
    "selection_rule",
    "tie_breaker",
    "selected_threshold",
    "seed_count",
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
SELECTED_PER_SEED_FIELDS = (
    "detector",
    "seed",
    "selection_split",
    "selection_rule",
    "selected_threshold",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_true_positives",
    "test_false_positives",
    "test_false_negatives",
    "test_prediction_count",
    "test_target_count",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen checkpoints, split metadata, and test bundles used by the correction."""

    phase5_config: Path
    phase5_summary: Path
    validation_split: str = Field(pattern=r"^[a-z0-9_-]+$")
    validation_annotations: Path
    test_threshold_config: Path


class SelectionSettings(StrictModel):
    """Predeclared validation-only threshold-selection rule."""

    start: float = Field(ge=0, le=1)
    stop: float = Field(ge=0, le=1)
    steps: int = Field(ge=2, le=1001)
    rule: Literal["maximum_mean_f1"]
    tie_breaker: Literal["highest_threshold"]

    @model_validator(mode="after")
    def validate_grid(self) -> SelectionSettings:
        if self.stop <= self.start:
            raise ValueError("selection stop must be greater than start")
        return self

    def thresholds(self) -> tuple[float, ...]:
        """Return the inclusive threshold grid used on validation."""

        return tuple(
            float(value) for value in np.round(np.linspace(self.start, self.stop, self.steps), 12)
        )


class OutputSettings(StrictModel):
    """Validation bundles and final selection artifacts."""

    log_dir: Path
    validation_bundles_dir: Path
    validation_manifest: Path
    validation_threshold_table: Path
    validation_threshold_per_seed_table: Path
    selected_operating_points_table: Path
    selected_operating_points_per_seed_table: Path
    summary_json: Path


class ThresholdSelectionConfig(StrictModel):
    """Strict Phase 14 validation-selection contract."""

    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    selection: SelectionSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_threshold_selection_config(path: str | Path) -> ThresholdSelectionConfig:
    """Load and strictly validate the Phase 14 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold-selection config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return ThresholdSelectionConfig.model_validate(payload)


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_one_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one validation row: {path}")
    return rows[0]


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _serialize_prediction(prediction: ImagePrediction) -> dict[str, Any]:
    return {
        "image_id": prediction.image_id,
        "image_size": list(prediction.image_size),
        "boxes_xyxy": prediction.boxes_xyxy.tolist(),
        "labels": prediction.labels.tolist(),
        "scores": prediction.scores.tolist(),
    }


def _write_validation_bundle(
    path: Path,
    *,
    detector: str,
    seed: int,
    checkpoint_sha256: str,
    annotation_path: Path,
    predictions: list[ImagePrediction],
    metrics: Mapping[str, Any],
    evaluation: Any,
    inference_seconds: float,
) -> Path:
    payload = {
        "schema_version": 1,
        "detector": detector,
        "seed": seed,
        "split": "validation",
        "checkpoint_sha256": checkpoint_sha256,
        "annotation_path": annotation_path.as_posix(),
        "annotation_sha256": sha256_file(annotation_path),
        "evaluation": evaluation.model_dump(mode="json"),
        "inference_seconds": inference_seconds,
        "predictions": [_serialize_prediction(item) for item in predictions],
        "operating_point": metrics["operating_point"],
        "coco": metrics["coco"],
    }
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return _atomic_bytes(path, gzip.compress(raw, compresslevel=9, mtime=0))


def _load_contract(
    config: ThresholdSelectionConfig,
) -> tuple[Any, dict[str, Any], dict[tuple[str, int], Any]]:
    phase5_path = config.resolve(config.inputs.phase5_config)
    summary_path = config.resolve(config.inputs.phase5_summary)
    phase5 = load_phase5_config(phase5_path)
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Phase 5 summary is not complete")
    if summary.get("config_sha256") != sha256_file(phase5_path):
        raise ValueError("Phase 5 config hash differs from its frozen summary")
    if summary.get("experiment_id") != phase5.experiment_id:
        raise ValueError("Phase 5 experiment identity differs from its frozen summary")
    if summary.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("Phase 5 evaluator settings differ from its frozen summary")
    if phase5.evaluation.coco_minimum_score > config.selection.start:
        raise ValueError("checkpoint inference minimum score would discard sweep values")
    if not any(
        np.isclose(value, phase5.evaluation.score_threshold)
        for value in config.selection.thresholds()
    ):
        raise ValueError("validation grid must include the original operating threshold")

    training_configs = load_and_validate_training_configs(phase5)
    run_summary = {
        (str(item["detector"]), int(item["seed"])): item for item in summary.get("runs", [])
    }
    expected = {(run.detector, run.seed) for run in phase5.runs}
    if set(run_summary) != expected:
        raise ValueError("Phase 5 summary lacks the complete detector/seed grid")
    for run in phase5.runs:
        actual_hash = sha256_file(phase5.resolve(run.checkpoint))
        recorded_hash = str(
            run_summary[(run.detector, run.seed)]["comparison_row"]["checkpoint_sha256"]
        )
        if actual_hash != recorded_hash:
            raise ValueError(
                f"checkpoint hash differs from Phase 5 for {run.detector} seed {run.seed}"
            )
    return phase5, summary, training_configs


def _load_validation_dataset(
    config: ThresholdSelectionConfig, phase5: Any, training_configs: Mapping[Any, Any]
) -> Any:
    from src.models.faster_rcnn_data import CocoDetectionDataset

    faster = training_configs[("faster_rcnn", phase5.seeds[0])]
    dataset = CocoDetectionDataset(
        faster.resolve(faster.data.dataset_config),
        config.inputs.validation_split,
        mode="full",
        project_root=config.project_root,
    )
    expected = config.resolve(config.inputs.validation_annotations)
    if dataset.annotation_file.resolve() != expected:
        raise ValueError("configured validation annotations disagree with the dataset")
    if faster.resolve(faster.data.val_annotations).resolve() != expected:
        raise ValueError("validation annotations disagree with the training contract")
    return dataset


def _validation_reference_table(detector: str, model_config: Any) -> Path:
    if detector == "faster_rcnn":
        return model_config.resolve(model_config.outputs.validation_table_path)
    return model_config.resolve(model_config.outputs.validation_table)


def _frozen_faster_config_sha256(run: Any, phase5: Any, model_config: Any) -> str:
    """Validate semantic config equality when historical YAML byte formatting changed."""

    from src.models.faster_rcnn_config import serializable_config

    training_summary_path = phase5.resolve(run.training_summary)
    training_summary = _read_json(training_summary_path)
    frozen_sha256 = str(training_summary.get("config_sha256", ""))
    if not frozen_sha256:
        raise ValueError(f"training summary lacks config hash: {training_summary_path}")
    resolved_config_path = training_summary_path.parent / model_config.outputs.resolved_config_path
    frozen_config = _read_json(resolved_config_path)
    if serializable_config(model_config) != frozen_config:
        raise ValueError(f"current config differs semantically from {resolved_config_path}")
    return frozen_sha256


def _verify_reference_operating_point(
    metrics: Mapping[str, Any], reference_path: Path, *, detector: str, seed: int
) -> dict[str, Any]:
    reference = _read_one_csv_row(reference_path)
    observed = metrics["operating_point"]["overall"]
    checked: dict[str, Any] = {}
    for field in ("precision", "recall", "f1"):
        expected = float(reference[field])
        actual = float(observed[field])
        if not np.isclose(actual, expected, atol=5e-12, rtol=0):
            raise ValueError(
                f"validation {field} does not reproduce for {detector} seed {seed}: "
                f"{actual} != {expected}"
            )
        checked[field] = actual
    return checked


def preflight(config: ThresholdSelectionConfig) -> dict[str, Any]:
    """Validate immutable sources without loading checkpoints or initializing CUDA."""

    phase5, _summary, training_configs = _load_contract(config)
    dataset = _load_validation_dataset(config, phase5, training_configs)
    missing = [
        config.resolve(config.outputs.validation_manifest),
        *[
            config.resolve(config.outputs.validation_bundles_dir)
            / f"{run.detector}_seed{run.seed}_validation_predictions.json.gz"
            for run in phase5.runs
        ],
    ]
    return {
        "status": "ready",
        "detectors": 2,
        "seeds_per_detector": len(phase5.seeds),
        "validation_images": len(dataset),
        "validation_annotations": sum(len(record.annotations) for record in dataset.records),
        "threshold_count": len(config.selection.thresholds()),
        "validation_bundles_ready": all(path.is_file() for path in missing),
        "performs_training": False,
        "preflight_performs_inference": False,
    }


def collect_validation_predictions(config: ThresholdSelectionConfig) -> dict[str, Any]:
    """Materialize raw validation predictions from the six immutable best checkpoints."""

    from src.utils.seed import initialize_reproducibility, seed_everything

    phase5, phase5_summary, training_configs = _load_contract(config)
    dataset = _load_validation_dataset(config, phase5, training_configs)
    initialize_reproducibility(phase5.seeds[0], config.resolve(config.outputs.log_dir))
    targets = _targets_from_dataset(dataset)
    annotation_path = dataset.annotation_file.resolve()
    records: list[dict[str, Any]] = []
    for run in sorted(phase5.runs, key=lambda item: (item.detector, item.seed)):
        seed_everything(run.seed)
        print(
            f"materializing validation predictions for {run.detector} seed {run.seed}", flush=True
        )
        model_config = training_configs[(run.detector, run.seed)]
        if run.detector == "faster_rcnn":
            predictions, inference_seconds = _collect_faster_rcnn_predictions(
                run,
                phase5,
                model_config,
                dataset,
                expected_config_sha256=_frozen_faster_config_sha256(run, phase5, model_config),
            )
        else:
            predictions, inference_seconds = _collect_yolo_predictions(
                run, phase5, model_config, dataset
            )
        metrics = evaluate_prediction_records(
            predictions,
            targets,
            category_names=dataset.category_names,
            settings=phase5.evaluation,
        )
        reference_path = _validation_reference_table(run.detector, model_config)
        reproduced = _verify_reference_operating_point(
            metrics, reference_path, detector=run.detector, seed=run.seed
        )
        checkpoint_path = phase5.resolve(run.checkpoint)
        bundle_path = config.resolve(config.outputs.validation_bundles_dir) / (
            f"{run.detector}_seed{run.seed}_validation_predictions.json.gz"
        )
        _write_validation_bundle(
            bundle_path,
            detector=run.detector,
            seed=run.seed,
            checkpoint_sha256=sha256_file(checkpoint_path),
            annotation_path=annotation_path,
            predictions=predictions,
            metrics=metrics,
            evaluation=phase5.evaluation,
            inference_seconds=inference_seconds,
        )
        records.append(
            {
                "detector": run.detector,
                "seed": run.seed,
                "checkpoint": _artifact(checkpoint_path, config.project_root),
                "validation_reference_table": _artifact(reference_path, config.project_root),
                "validation_bundle": _artifact(bundle_path, config.project_root),
                "image_count": len(predictions),
                "inference_seconds": inference_seconds,
                "reference_operating_point_reproduction": reproduced,
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            Path(__file__).resolve().relative_to(config.project_root).as_posix(): sha256_file(
                Path(__file__).resolve()
            ),
            "src/evaluate.py": sha256_file(config.project_root / "src/evaluate.py"),
        },
        "upstream": {
            "phase5_config": _artifact(
                config.resolve(config.inputs.phase5_config), config.project_root
            ),
            "phase5_summary": _artifact(
                config.resolve(config.inputs.phase5_summary), config.project_root
            ),
            "validation_annotations": _artifact(annotation_path, config.project_root),
        },
        "environment": {
            "run_environment": _artifact(
                config.resolve(config.outputs.log_dir) / "run_environment.json",
                config.project_root,
            ),
            "pip_freeze": _artifact(
                config.resolve(config.outputs.log_dir) / "pip_freeze.txt",
                config.project_root,
            ),
        },
        "evaluation": phase5.evaluation.model_dump(mode="json"),
        "counts": {
            "images_per_bundle": len(dataset),
            "annotations": sum(len(record.annotations) for record in dataset.records),
            "bundles": len(records),
        },
        "runs": records,
        "performs_training": False,
        "performs_inference": True,
        "test_split_accessed": False,
        "phase5_test_summary_status": phase5_summary["status"],
    }
    manifest_path = config.resolve(config.outputs.validation_manifest)
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "manifest": manifest_path.as_posix()}, indent=2))
    return manifest


def _load_validation_bundles(
    config: ThresholdSelectionConfig,
) -> tuple[Any, list[dict[str, Any]], Any, dict[int, str]]:
    phase5, _phase5_summary, training_configs = _load_contract(config)
    dataset = _load_validation_dataset(config, phase5, training_configs)
    targets = _targets_from_dataset(dataset)
    manifest_path = config.resolve(config.outputs.validation_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError("validation manifest is missing; run --mode collect-validation")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("analysis_id") != config.analysis_id:
        raise ValueError("validation manifest identity or status is invalid")
    if manifest.get("config_sha256") != sha256_file(config.source_path):
        raise ValueError("validation manifest was created from a different config")
    if manifest.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("validation bundle evaluation settings differ from Phase 5")
    if manifest.get("upstream", {}).get("validation_annotations", {}).get("sha256") != sha256_file(
        dataset.annotation_file
    ):
        raise ValueError("validation annotations differ from the bundle manifest")

    expected = {(run.detector, run.seed) for run in phase5.runs}
    bundles: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in manifest.get("runs", []):
        detector = str(record["detector"])
        seed = int(record["seed"])
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
        if payload.get("annotation_sha256") != sha256_file(dataset.annotation_file):
            raise ValueError(f"validation bundle annotation mismatch: {bundle_path}")
        if payload.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
            raise ValueError(f"validation bundle evaluator mismatch: {bundle_path}")
        predictions = _deserialize_predictions(payload)
        if len(predictions) != len(targets):
            raise ValueError(f"validation bundle image count mismatch: {bundle_path}")
        bundles.append(
            {
                "detector": detector,
                "seed": seed,
                "path": bundle_path,
                "sha256": bundle_info["sha256"],
                "payload": payload,
                "predictions": predictions,
            }
        )
    if seen != expected:
        raise ValueError("validation manifest lacks the complete detector/seed grid")
    bundles.sort(key=lambda item: (str(item["detector"]), int(item["seed"])))
    return phase5, bundles, targets, dataset.category_names


def select_validation_thresholds(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Select maximum mean validation F1, breaking exact ties toward selectivity."""

    selected: dict[str, Mapping[str, Any]] = {}
    detectors = sorted({str(row["detector"]) for row in aggregate_rows})
    for detector in detectors:
        candidates = [row for row in aggregate_rows if row["detector"] == detector]
        if not candidates:
            raise ValueError(f"no validation threshold rows for {detector}")
        selected[detector] = max(
            candidates,
            key=lambda row: (float(row["f1"]), float(row["threshold"])),
        )
    return selected


def _selected_tables(
    validation_per_seed: Sequence[Mapping[str, Any]],
    test_per_seed: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Mapping[str, Any]],
    *,
    ddof: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_aggregate = aggregate_threshold_rows(test_per_seed, ddof=ddof)
    selected_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    for detector, validation_row in sorted(selected.items()):
        threshold = float(validation_row["threshold"])
        test_row = next(row for row in test_aggregate if row["detector"] == detector)
        selected_rows.append(
            {
                "detector": detector,
                "selection_split": "validation",
                "selection_rule": "maximum arithmetic mean F1 across three validation seeds",
                "tie_breaker": "highest threshold among exact mean-F1 ties",
                "selected_threshold": threshold,
                "seed_count": validation_row["seed_count"],
                **{
                    f"validation_{metric}": validation_row[metric]
                    for metric in ("precision", "recall", "f1")
                },
                **{
                    f"validation_{metric}_std": validation_row[f"{metric}_std"]
                    for metric in ("precision", "recall", "f1")
                },
                **{f"test_{metric}": test_row[metric] for metric in ("precision", "recall", "f1")},
                **{
                    f"test_{metric}_std": test_row[f"{metric}_std"]
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
        detector_validation = {
            int(row["seed"]): row
            for row in validation_per_seed
            if row["detector"] == detector and np.isclose(float(row["threshold"]), threshold)
        }
        detector_test = {
            int(row["seed"]): row for row in test_per_seed if row["detector"] == detector
        }
        if set(detector_validation) != set(detector_test):
            raise ValueError(f"validation/test seed grids differ for {detector}")
        for seed in sorted(detector_validation):
            validation_seed = detector_validation[seed]
            test_seed = detector_test[seed]
            per_seed_rows.append(
                {
                    "detector": detector,
                    "seed": seed,
                    "selection_split": "validation",
                    "selection_rule": "maximum arithmetic mean F1 across three validation seeds",
                    "selected_threshold": threshold,
                    "validation_precision": validation_seed["precision"],
                    "validation_recall": validation_seed["recall"],
                    "validation_f1": validation_seed["f1"],
                    "test_precision": test_seed["precision"],
                    "test_recall": test_seed["recall"],
                    "test_f1": test_seed["f1"],
                    "test_true_positives": test_seed["true_positives"],
                    "test_false_positives": test_seed["false_positives"],
                    "test_false_negatives": test_seed["false_negatives"],
                    "test_prediction_count": test_seed["prediction_count"],
                    "test_target_count": test_seed["target_count"],
                }
            )
    return selected_rows, per_seed_rows


def run_threshold_selection(config: ThresholdSelectionConfig) -> dict[str, Any]:
    """Sweep validation, freeze thresholds, and evaluate each once on frozen test bundles."""

    phase5, validation_bundles, validation_targets, category_names = _load_validation_bundles(
        config
    )
    thresholds = config.selection.thresholds()
    class_ids = tuple(sorted(category_names))
    validation_per_seed: list[dict[str, Any]] = []
    for bundle in validation_bundles:
        validation_per_seed.extend(
            sweep_prediction_records(
                bundle["predictions"],
                validation_targets,
                detector=str(bundle["detector"]),
                seed=int(bundle["seed"]),
                thresholds=thresholds,
                class_ids=class_ids,
                iou_threshold=phase5.evaluation.match_iou_threshold,
                max_detections=phase5.evaluation.max_detections,
            )
        )
    validation_aggregate = aggregate_threshold_rows(
        validation_per_seed, ddof=phase5.runtime.statistics_ddof
    )
    selected = select_validation_thresholds(validation_aggregate)

    test_config = load_threshold_sweep_config(config.resolve(config.inputs.test_threshold_config))
    test_phase5, _summary, test_bundles, test_targets, test_categories = _validate_upstream(
        test_config
    )
    if test_phase5.evaluation != phase5.evaluation or test_categories != category_names:
        raise ValueError("validation and test evaluator/category contracts differ")
    test_per_seed: list[dict[str, Any]] = []
    for bundle in test_bundles:
        threshold = float(selected[str(bundle["detector"])]["threshold"])
        rows = sweep_prediction_records(
            bundle["predictions"],
            test_targets,
            detector=str(bundle["detector"]),
            seed=int(bundle["seed"]),
            thresholds=(threshold,),
            class_ids=class_ids,
            iou_threshold=phase5.evaluation.match_iou_threshold,
            max_detections=phase5.evaluation.max_detections,
        )
        if len(rows) != 1:
            raise AssertionError("selected test threshold must be evaluated exactly once")
        test_per_seed.extend(rows)

    selected_rows, selected_per_seed = _selected_tables(
        validation_per_seed,
        test_per_seed,
        selected,
        ddof=phase5.runtime.statistics_ddof,
    )
    validation_path = _atomic_csv(
        config.resolve(config.outputs.validation_threshold_table),
        VALIDATION_FIELDS,
        validation_aggregate,
    )
    validation_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.validation_threshold_per_seed_table),
        VALIDATION_PER_SEED_FIELDS,
        validation_per_seed,
    )
    selected_path = _atomic_csv(
        config.resolve(config.outputs.selected_operating_points_table),
        SELECTED_FIELDS,
        selected_rows,
    )
    selected_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.selected_operating_points_per_seed_table),
        SELECTED_PER_SEED_FIELDS,
        selected_per_seed,
    )

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
            "src/evaluate.py": sha256_file(config.project_root / "src/evaluate.py"),
            "src/evaluate_threshold_sweep.py": sha256_file(
                config.project_root / "src/evaluate_threshold_sweep.py"
            ),
        },
        "upstream": {
            "validation_manifest": _artifact(
                config.resolve(config.outputs.validation_manifest), config.project_root
            ),
            "test_threshold_config": _artifact(
                config.resolve(config.inputs.test_threshold_config), config.project_root
            ),
            "test_prediction_bundles": [
                {
                    "detector": bundle["detector"],
                    "seed": bundle["seed"],
                    "path": bundle["path"].relative_to(config.project_root).as_posix(),
                    "sha256": bundle["sha256"],
                }
                for bundle in test_bundles
            ],
        },
        "selection": {
            "split": "validation",
            "rule": "maximum arithmetic mean F1 across the three validation seeds",
            "reason": (
                "F1 gives precision and recall equal weight without inventing an unvalidated "
                "clinical cost ratio and preserves the predeclared Batch 10 grid"
            ),
            "tie_breaker": "highest threshold among exact mean-F1 ties",
            "thresholds": list(thresholds),
            "selected": {
                detector: {
                    "threshold": row["threshold"],
                    "validation_precision": row["precision"],
                    "validation_recall": row["recall"],
                    "validation_f1": row["f1"],
                }
                for detector, row in selected.items()
            },
        },
        "test_evaluation": {
            "policy": "apply each frozen detector threshold once to each frozen test bundle",
            "threshold_evaluations_per_bundle": 1,
            "selected_operating_points": selected_rows,
            "performs_training": False,
            "performs_checkpoint_loading": False,
            "performs_model_inference": False,
        },
        "counts": {
            "validation_images_per_bundle": len(validation_targets),
            "test_images_per_bundle": len(test_targets),
            "validation_threshold_rows_per_seed": len(validation_per_seed),
            "validation_threshold_rows_aggregate": len(validation_aggregate),
            "selected_test_rows_per_seed": len(test_per_seed),
        },
        "artifacts": {
            "validation_threshold_table": _artifact(validation_path, config.project_root),
            "validation_threshold_per_seed_table": _artifact(
                validation_per_seed_path, config.project_root
            ),
            "selected_operating_points_table": _artifact(selected_path, config.project_root),
            "selected_operating_points_per_seed_table": _artifact(
                selected_per_seed_path, config.project_root
            ),
        },
    }
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the validation-selection command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/threshold_selection.yaml"))
    parser.add_argument(
        "--mode",
        choices=("preflight", "collect-validation", "run"),
        default="preflight",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight, one-time validation inference, or offline selection."""

    args = build_parser().parse_args(argv)
    config = load_threshold_selection_config(args.config)
    if args.mode == "preflight":
        result = preflight(config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.mode == "collect-validation":
        collect_validation_predictions(config)
    else:
        run_threshold_selection(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
