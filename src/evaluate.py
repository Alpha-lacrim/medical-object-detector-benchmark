"""Unified held-out evaluation and multi-seed detector comparison harness."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    evaluate_operating_point,
)

DetectorName = Literal["faster_rcnn", "yolo11s"]


class StrictModel(BaseModel):
    """Reject undeclared config keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationSettings(StrictModel):
    """Thresholds shared by every detector and seed."""

    score_threshold: float = Field(ge=0, le=1)
    match_iou_threshold: float = Field(gt=0, le=1)
    coco_minimum_score: float = Field(ge=0, le=1)
    nms_iou_threshold: float = Field(gt=0, le=1)
    max_detections: int = Field(ge=10)

    @model_validator(mode="after")
    def minimum_score_precedes_operating_point(self) -> EvaluationSettings:
        if self.coco_minimum_score > self.score_threshold:
            raise ValueError("coco_minimum_score must not exceed score_threshold")
        return self


class RuntimeSettings(StrictModel):
    """Common evaluation runtime and aggregation settings."""

    device: Literal["cuda"]
    inference_batch_size: Literal[1]
    num_workers: int = Field(ge=0)
    pin_memory: bool
    statistics_ddof: Literal[1]


class RunSpec(StrictModel):
    """Paths tying one trained seed to its immutable evidence."""

    detector: DetectorName
    seed: int = Field(ge=0)
    training_config: Path
    checkpoint: Path
    training_summary: Path
    compute_table: Path


class OutputSettings(StrictModel):
    """Generated prediction, audit, and comparison artifacts."""

    log_dir: Path
    prediction_bundles_dir: Path
    summary_json: Path
    per_seed_table: Path
    mean_std_table: Path
    publication_table: Path


class TimingSources(StrictModel):
    """Accepted measurements reused when only the training RNG seed changes."""

    faster_rcnn: Path
    yolo11s: Path


class TimingReuseReview(StrictModel):
    """Exact reviewed source drift between a historical timing and the current tree."""

    timing_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_changes: tuple[Path, ...]
    reviewed_additions: tuple[Path, ...]
    rationale: str = Field(min_length=1)


class TimingReuseReviews(StrictModel):
    """Detector-specific audits authorizing exact, hash-bound timing reuse."""

    faster_rcnn: TimingReuseReview
    yolo11s: TimingReuseReview


class Phase5Config(StrictModel):
    """Strict top-level Phase 5 experiment contract."""

    schema_version: Literal[1]
    experiment_id: str
    seeds: tuple[int, ...]
    split: Literal["test"]
    evaluation: EvaluationSettings
    runtime: RuntimeSettings
    timing_sources: TimingSources
    timing_reuse_reviews: TimingReuseReviews | None = None
    runs: tuple[RunSpec, ...]
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def complete_factorial_run_grid(self) -> Phase5Config:
        if len(self.seeds) < 2:
            raise ValueError("seeds must contain at least two values")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain unique values")
        expected = {
            (detector, seed) for detector in ("faster_rcnn", "yolo11s") for seed in self.seeds
        }
        actual = {(run.detector, run.seed) for run in self.runs}
        if len(actual) != len(self.runs):
            raise ValueError("runs must not repeat a detector/seed pair")
        if actual != expected:
            raise ValueError("runs must contain both detectors for every configured seed")
        return self

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path).resolve()


PER_SEED_FIELDS = (
    "detector",
    "seed",
    "run_id",
    "best_epoch",
    "completed_epochs",
    "split",
    "image_count",
    "target_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "operating_point_prediction_count",
    "coco_prediction_count",
    "maximum_prediction_score",
    "score_threshold",
    "match_iou_threshold",
    "coco_minimum_score",
    "max_detections",
    "precision",
    "recall",
    "f1",
    "iou",
    "dice",
    "conditional_localization_defined",
    "conditional_localization_undefined_reason",
    "map_50",
    "map_50_95",
    "evaluation_inference_seconds",
    "evaluation_fps",
    "profile_fps",
    "inference_time_ms",
    "p50_inference_time_ms",
    "p95_inference_time_ms",
    "total_parameters",
    "trainable_parameters",
    "gflops",
    "peak_gpu_memory_mib",
    "training_time_seconds",
    "checkpoint_size_mib",
    "checkpoint_sha256",
    "prediction_bundle",
)

METRICS = (
    ("precision", "ratio"),
    ("recall", "ratio"),
    ("f1", "ratio"),
    ("iou", "ratio"),
    ("dice", "ratio"),
    ("map_50", "ratio"),
    ("map_50_95", "ratio"),
    ("profile_fps", "images/second"),
    ("inference_time_ms", "milliseconds/image"),
    ("total_parameters", "parameters"),
    ("trainable_parameters", "parameters"),
    ("gflops", "GFLOPs/image"),
    ("peak_gpu_memory_mib", "MiB"),
    ("training_time_seconds", "seconds"),
)

CONDITIONAL_METRICS = frozenset({"iou", "dice"})


def load_phase5_config(path: str | Path) -> Phase5Config:
    """Load the strict evaluator YAML without importing either detector framework."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return Phase5Config.model_validate(payload)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _atomic_bytes(path, text.encode("utf-8"))


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
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_one_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one compute row in {path}, found {len(rows)}")
    return rows[0]


def _json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _faster_contract(config: Any) -> dict[str, Any]:
    contract = asdict(config)
    for key in ("seed", "experiment_id", "outputs", "project_root", "source_path"):
        contract.pop(key, None)
    return json.loads(json.dumps(contract, default=str))


def _yolo_contract(config: Any) -> dict[str, Any]:
    return config.model_dump(
        mode="json",
        exclude={"seed", "experiment_id", "outputs", "project_root", "source_path"},
    )


def load_and_validate_training_configs(
    config: Phase5Config,
) -> dict[tuple[str, int], Any]:
    """Validate per-detector parity and the common Phase 5 threshold contract."""

    from src.models.faster_rcnn_config import load_faster_rcnn_config
    from src.models.yolo_config import load_yolo_config

    loaded: dict[tuple[str, int], Any] = {}
    contracts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in config.runs:
        training_path = config.resolve(run.training_config)
        model_config = (
            load_faster_rcnn_config(training_path)
            if run.detector == "faster_rcnn"
            else load_yolo_config(training_path)
        )
        if model_config.seed != run.seed:
            raise ValueError(f"{training_path} seed does not match its evaluation run")
        loaded[(run.detector, run.seed)] = model_config
        contracts[run.detector].append(
            _faster_contract(model_config)
            if run.detector == "faster_rcnn"
            else _yolo_contract(model_config)
        )

        checkpoint = config.resolve(run.checkpoint).resolve()
        expected_checkpoint = (
            model_config.resolve(model_config.outputs.best_checkpoint_path).resolve()
            if run.detector == "faster_rcnn"
            else (
                model_config.resolve(model_config.outputs.checkpoint_dir) / "best_model.pt"
            ).resolve()
        )
        if checkpoint != expected_checkpoint:
            raise ValueError(f"configured checkpoint disagrees with {training_path}")
        expected_compute = (
            model_config.resolve(model_config.outputs.compute_table_path).resolve()
            if run.detector == "faster_rcnn"
            else model_config.resolve(model_config.outputs.compute_table).resolve()
        )
        if config.resolve(run.compute_table).resolve() != expected_compute:
            raise ValueError(f"configured compute table disagrees with {training_path}")

    for detector, detector_contracts in contracts.items():
        reference = detector_contracts[0]
        if any(candidate != reference for candidate in detector_contracts[1:]):
            raise ValueError(f"{detector} training configs differ beyond seed/output identity")

    settings = config.evaluation
    faster = loaded[("faster_rcnn", config.seeds[0])]
    yolo = loaded[("yolo11s", config.seeds[0])]
    if (
        faster.model.min_size != yolo.model.input_size
        or faster.model.max_size != yolo.model.input_size
    ):
        raise ValueError("detector input resolutions do not match")
    if faster.model.box_nms_threshold != settings.nms_iou_threshold:
        raise ValueError("Faster R-CNN NMS threshold differs from the unified contract")
    if yolo.evaluation.nms_iou_threshold != settings.nms_iou_threshold:
        raise ValueError("YOLO NMS threshold differs from the unified contract")
    if faster.model.box_score_threshold > settings.coco_minimum_score:
        raise ValueError("Faster R-CNN discards scores required by unified COCO evaluation")
    if yolo.evaluation.inference_minimum_score > settings.coco_minimum_score:
        raise ValueError("YOLO discards scores required by unified COCO evaluation")
    for detector_config in (faster, yolo):
        detector_eval = detector_config.evaluation
        if detector_eval.score_threshold != settings.score_threshold:
            raise ValueError("training and unified operating-point score thresholds differ")
        if detector_eval.match_iou_threshold != settings.match_iou_threshold:
            raise ValueError("training and unified match-IoU thresholds differ")
        if detector_eval.max_detections != settings.max_detections:
            raise ValueError("training and unified maximum detections differ")
    return loaded


def evaluate_prediction_records(
    predictions: list[ImagePrediction],
    targets: list[ImageTarget],
    *,
    category_names: dict[int, str],
    settings: EvaluationSettings,
) -> dict[str, Any]:
    """Apply the exact same operating-point and COCO evaluators to any detector."""

    class_ids = tuple(sorted(category_names))
    operating = evaluate_operating_point(
        predictions,
        targets,
        class_ids=class_ids,
        score_threshold=settings.score_threshold,
        iou_threshold=settings.match_iou_threshold,
        max_detections=settings.max_detections,
    )
    coco = evaluate_coco(
        predictions,
        targets,
        class_ids=class_ids,
        class_names=category_names,
        minimum_score=settings.coco_minimum_score,
        max_detections=settings.max_detections,
    )
    return {"operating_point": operating, "coco": coco}


def _targets_from_dataset(dataset: Any) -> list[ImageTarget]:
    targets: list[ImageTarget] = []
    for record in dataset.records:
        boxes = np.asarray(
            [annotation.bbox_xyxy for annotation in record.annotations], dtype=np.float64
        ).reshape(-1, 4)
        targets.append(
            ImageTarget(
                image_id=record.file_name,
                image_size=(record.height, record.width),
                boxes_xyxy=boxes,
                labels=np.asarray(
                    [annotation.category_id for annotation in record.annotations],
                    dtype=np.int64,
                ),
            )
        )
    return targets


def _load_test_dataset(config: Phase5Config, faster_config: Any) -> Any:
    from src.models.faster_rcnn_data import CocoDetectionDataset

    dataset = CocoDetectionDataset(
        faster_config.resolve(faster_config.data.dataset_config),
        config.split,
        mode="full",
        project_root=config.project_root,
    )
    configured_test = faster_config.resolve(faster_config.data.test_annotations).resolve()
    if dataset.annotation_file.resolve() != configured_test:
        raise ValueError("dataset and detector configs disagree on test annotations")
    return dataset


def _collect_faster_rcnn_predictions(
    run: RunSpec,
    phase_config: Phase5Config,
    model_config: Any,
    dataset: Any,
    *,
    expected_config_sha256: str | None = None,
) -> tuple[list[ImagePrediction], float]:
    import torch
    from torch.utils.data import DataLoader

    from src.models.faster_rcnn_config import config_fingerprint
    from src.models.faster_rcnn_data import detection_collate_fn
    from src.models.faster_rcnn_model import build_faster_rcnn

    checkpoint_path = phase_config.resolve(run.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_config_sha256 = expected_config_sha256 or config_fingerprint(model_config)
    if checkpoint.get("config_sha256") != required_config_sha256:
        raise ValueError(f"Faster R-CNN checkpoint/config mismatch for seed {run.seed}")
    model, _metadata = build_faster_rcnn(
        dataset.num_foreground_classes,
        model_config.model,
        use_pretrained_weights=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(phase_config.runtime.device)
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=phase_config.runtime.inference_batch_size,
        shuffle=False,
        num_workers=phase_config.runtime.num_workers,
        pin_memory=phase_config.runtime.pin_memory,
        persistent_workers=False,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )
    by_numeric_id = {record.id: record for record in dataset.records}
    predictions: list[ImagePrediction] = []
    elapsed_seconds = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            device_images = [image.to(device, non_blocking=True) for image in images]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=model_config.runtime.amp
            ):
                outputs = model(device_images)
            torch.cuda.synchronize(device)
            elapsed_seconds += time.perf_counter() - started
            for image, target, output in zip(images, targets, outputs, strict=True):
                numeric_id = int(target["image_id"].reshape(-1)[0].item())
                record = by_numeric_id[numeric_id]
                scores = output["scores"].detach().cpu().numpy().astype(np.float64)
                keep = scores >= phase_config.evaluation.coco_minimum_score
                labels = []
                for label in output["labels"].detach().cpu().numpy()[keep]:
                    numeric_label = int(label)
                    if numeric_label not in dataset.label_to_category_id:
                        raise ValueError(f"unknown Faster R-CNN label {numeric_label}")
                    labels.append(dataset.label_to_category_id[numeric_label])
                predictions.append(
                    ImagePrediction(
                        image_id=record.file_name,
                        image_size=(int(image.shape[-2]), int(image.shape[-1])),
                        boxes_xyxy=output["boxes"].detach().cpu().numpy()[keep],
                        labels=np.asarray(labels, dtype=np.int64),
                        scores=scores[keep],
                    )
                )
    del checkpoint, loader, model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, elapsed_seconds


def _recorded_training_config_sha256(run: RunSpec, config: Phase5Config) -> str:
    """Return the immutable config hash recorded by a completed training run."""

    summary_path = config.resolve(run.training_summary)
    summary = _json_payload(summary_path)
    if summary.get("status") != "complete" or summary.get("seed") != run.seed:
        raise ValueError(f"invalid Faster R-CNN training summary for seed {run.seed}")
    digest = summary.get("config_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"missing Faster R-CNN config hash for seed {run.seed}")
    return digest


def _collect_yolo_predictions(
    run: RunSpec,
    phase_config: Phase5Config,
    model_config: Any,
    dataset: Any,
) -> tuple[list[ImagePrediction], float]:
    import torch
    from ultralytics import YOLO

    checkpoint_path = phase_config.resolve(run.checkpoint)
    yolo = YOLO(checkpoint_path.as_posix())
    device = torch.device(f"cuda:{model_config.runtime.device}")
    category_ids = tuple(sorted(dataset.category_names))
    yolo_to_category = {index: category_id for index, category_id in enumerate(category_ids)}
    predictions: list[ImagePrediction] = []
    elapsed_seconds = 0.0
    for index, record in enumerate(dataset.records):
        image_path = dataset.image_path(index)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        results = yolo.predict(
            source=image_path.as_posix(),
            stream=False,
            batch=phase_config.runtime.inference_batch_size,
            imgsz=model_config.model.input_size,
            device=model_config.runtime.device,
            amp=model_config.runtime.amp,
            conf=phase_config.evaluation.coco_minimum_score,
            iou=phase_config.evaluation.nms_iou_threshold,
            max_det=phase_config.evaluation.max_detections,
            agnostic_nms=False,
            augment=False,
            verbose=False,
            save=False,
        )
        torch.cuda.synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        if len(results) != 1 or Path(results[0].path).name != record.file_name:
            raise ValueError(f"unexpected YOLO prediction source for {record.file_name}")
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64, copy=False)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float64, copy=False)
        yolo_labels = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
        try:
            labels = np.asarray(
                [yolo_to_category[int(label)] for label in yolo_labels], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(f"YOLO predicted an unknown class label: {error.args[0]}") from error
        predictions.append(
            ImagePrediction(
                image_id=record.file_name,
                image_size=(record.height, record.width),
                boxes_xyxy=xyxy,
                labels=labels,
                scores=scores,
            )
        )
    del boxes, results, yolo
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, elapsed_seconds


def _serialize_prediction(prediction: ImagePrediction) -> dict[str, Any]:
    return {
        "image_id": prediction.image_id,
        "image_size": list(prediction.image_size),
        "boxes_xyxy": prediction.boxes_xyxy.tolist(),
        "labels": prediction.labels.tolist(),
        "scores": prediction.scores.tolist(),
    }


def _write_prediction_bundle(
    path: Path,
    *,
    run: RunSpec,
    checkpoint_sha256: str,
    annotation_path: Path,
    predictions: list[ImagePrediction],
    metrics: dict[str, Any],
    settings: EvaluationSettings,
) -> Path:
    payload = {
        "schema_version": 1,
        "detector": run.detector,
        "seed": run.seed,
        "split": "test",
        "checkpoint_sha256": checkpoint_sha256,
        "annotation_path": annotation_path.as_posix(),
        "annotation_sha256": sha256_file(annotation_path),
        "evaluation": settings.model_dump(mode="json"),
        "predictions": [_serialize_prediction(item) for item in predictions],
        "operating_point": metrics["operating_point"],
        "coco": metrics["coco"],
    }
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return _atomic_bytes(path, gzip.compress(raw, compresslevel=9, mtime=0))


def _number(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _artifact_row(
    run: RunSpec,
    config: Phase5Config,
    metrics: dict[str, Any],
    inference_seconds: float,
    bundle_path: Path,
    maximum_prediction_score: float | None,
) -> dict[str, Any]:
    summary_path = config.resolve(run.training_summary)
    compute_path = config.resolve(run.compute_table)
    checkpoint_path = config.resolve(run.checkpoint)
    summary = _json_payload(summary_path)
    compute = _read_one_csv_row(compute_path)
    if summary.get("status") != "complete":
        raise ValueError(f"training summary is not complete: {summary_path}")
    summary_seed = (
        summary.get("seed")
        if run.detector == "faster_rcnn"
        else summary.get("train_args", {}).get("seed")
    )
    if summary_seed != run.seed or int(compute["seed"]) != run.seed:
        raise ValueError(f"seed provenance mismatch for {run.detector} seed {run.seed}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if compute["model_sha256"] != checkpoint_sha256:
        raise ValueError(f"compute/checkpoint hash mismatch for {checkpoint_path}")
    operating = metrics["operating_point"]["overall"]
    coco = metrics["coco"]
    conditional_defined = int(operating["tp"]) > 0
    conditional_reason = ""
    if not conditional_defined:
        conditional_reason = (
            "no_detections_at_frozen_score_threshold"
            if int(operating["prediction_count"]) == 0
            else "no_iou_qualified_true_positive"
        )
    training_seconds = (
        summary.get("total_training_seconds")
        if run.detector == "faster_rcnn"
        else summary.get("training_seconds")
    )
    peak_memory = (
        summary.get("runtime", {}).get("peak_train_gpu_memory_mib")
        if run.detector == "faster_rcnn"
        else summary.get("peak_train_gpu_memory_mib")
    )
    return {
        "detector": run.detector,
        "seed": run.seed,
        "run_id": summary["run_id"],
        "best_epoch": summary["best_epoch"],
        "completed_epochs": summary["completed_epochs"],
        "split": config.split,
        "image_count": coco["image_count"],
        "target_count": coco["annotation_count"],
        "true_positives": operating["tp"],
        "false_positives": operating["fp"],
        "false_negatives": operating["fn"],
        "operating_point_prediction_count": operating["prediction_count"],
        "coco_prediction_count": coco["prediction_count"],
        "maximum_prediction_score": maximum_prediction_score,
        "score_threshold": config.evaluation.score_threshold,
        "match_iou_threshold": config.evaluation.match_iou_threshold,
        "coco_minimum_score": config.evaluation.coco_minimum_score,
        "max_detections": config.evaluation.max_detections,
        "precision": operating["precision"],
        "recall": operating["recall"],
        "f1": operating["f1"],
        "iou": operating["matched_mean_iou"],
        "dice": operating["matched_mean_box_dice"],
        "conditional_localization_defined": conditional_defined,
        "conditional_localization_undefined_reason": conditional_reason,
        "map_50": coco["ap50"],
        "map_50_95": coco["ap50_95"],
        "evaluation_inference_seconds": inference_seconds,
        "evaluation_fps": coco["image_count"] / inference_seconds,
        "profile_fps": _number(compute["throughput_fps"], field="throughput_fps"),
        "inference_time_ms": _number(compute["mean_latency_ms"], field="mean_latency_ms"),
        "p50_inference_time_ms": _number(compute["p50_latency_ms"], field="p50_latency_ms"),
        "p95_inference_time_ms": _number(compute["p95_latency_ms"], field="p95_latency_ms"),
        "total_parameters": int(compute["total_parameters"]),
        "trainable_parameters": int(compute["trainable_parameters"]),
        "gflops": _number(compute["estimated_gflops"], field="estimated_gflops"),
        "peak_gpu_memory_mib": _number(peak_memory, field="peak_train_gpu_memory_mib"),
        "training_time_seconds": _number(training_seconds, field="training_seconds"),
        "checkpoint_size_mib": _number(compute["model_size_mib"], field="model_size_mib"),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_bundle": bundle_path.resolve().relative_to(config.project_root).as_posix(),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, ddof: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return long-form and detector-comparison mean ± sample-SD tables."""

    detectors = sorted({str(row["detector"]) for row in rows})
    if detectors != ["faster_rcnn", "yolo11s"]:
        raise ValueError("aggregation requires Faster R-CNN and YOLO11s rows")
    long_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for detector in detectors:
        detector_rows = sorted(
            (row for row in rows if row["detector"] == detector),
            key=lambda row: int(row["seed"]),
        )
        if len({int(row["seed"]) for row in detector_rows}) != len(detector_rows):
            raise ValueError(f"duplicate {detector} seed in aggregation rows")
        for row in detector_rows:
            true_positives = row.get("true_positives")
            if (
                isinstance(true_positives, bool)
                or not isinstance(true_positives, (int, np.integer))
                or int(true_positives) < 0
            ):
                raise ValueError(f"invalid true-positive count for {detector} seed {row['seed']}")
            undefined = [metric for metric in CONDITIONAL_METRICS if row.get(metric) is None]
            if undefined and set(undefined) != CONDITIONAL_METRICS:
                raise ValueError(
                    f"conditional IoU/Dice must be jointly defined for {detector} seed "
                    f"{row['seed']}"
                )
            if int(true_positives) == 0 and set(undefined) != CONDITIONAL_METRICS:
                raise ValueError(
                    f"conditional IoU/Dice must be undefined with zero true positives for "
                    f"{detector} seed {row['seed']}"
                )
            if int(true_positives) > 0 and undefined:
                raise ValueError(
                    f"conditional IoU/Dice cannot be undefined with true positives for "
                    f"{detector} seed {row['seed']}"
                )
            declared_reason = str(row.get("conditional_localization_undefined_reason") or "")
            if int(true_positives) == 0:
                prediction_count = row.get("operating_point_prediction_count")
                if (
                    isinstance(prediction_count, bool)
                    or not isinstance(prediction_count, (int, np.integer))
                    or int(prediction_count) < 0
                ):
                    raise ValueError(f"invalid prediction count for {detector} seed {row['seed']}")
                expected_reason = (
                    "no_detections_at_frozen_score_threshold"
                    if int(prediction_count) == 0
                    else "no_iou_qualified_true_positive"
                )
                if declared_reason != expected_reason:
                    raise ValueError(
                        f"conditional undefined reason differs from counts for {detector} "
                        f"seed {row['seed']}"
                    )
            elif declared_reason:
                raise ValueError(
                    f"defined conditional metrics cannot carry an undefined reason for "
                    f"{detector} seed {row['seed']}"
                )
        for metric, unit in METRICS:
            undefined_rows = [row for row in detector_rows if row.get(metric) is None]
            if undefined_rows and metric not in CONDITIONAL_METRICS:
                raise ValueError(f"nonconditional metric is undefined: {detector} {metric}")
            defined_rows = [row for row in detector_rows if row.get(metric) is not None]
            try:
                values = np.asarray([float(row[metric]) for row in defined_rows], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"cannot aggregate {detector} {metric}") from error
            if len(values) <= ddof or not np.isfinite(values).all():
                raise ValueError(f"cannot aggregate {detector} {metric}")
            mean = float(np.mean(values))
            std = 0.0 if np.all(values == values[0]) else float(np.std(values, ddof=ddof))
            undefined_seeds = ";".join(str(int(row["seed"])) for row in undefined_rows)
            undefined_reasons = sorted(
                {
                    str(row.get("conditional_localization_undefined_reason") or "")
                    for row in undefined_rows
                }
            )
            if undefined_rows and any(not reason for reason in undefined_reasons):
                raise ValueError(f"undefined conditional metric lacks reason: {detector} {metric}")
            entry = {
                "detector": detector,
                "metric": metric,
                "unit": unit,
                "n": len(values),
                "attempted_n": len(detector_rows),
                "undefined_n": len(undefined_rows),
                "undefined_seeds": undefined_seeds,
                "undefined_reason": ";".join(undefined_reasons),
                "mean": mean,
                "std": std,
                "mean_plus_minus_std": f"{mean:.6g} ± {std:.6g}",
            }
            long_rows.append(entry)
            by_key[(detector, metric)] = entry
    comparison: list[dict[str, Any]] = []
    for metric, unit in METRICS:
        faster = by_key[("faster_rcnn", metric)]
        yolo = by_key[("yolo11s", metric)]
        comparison.append(
            {
                "metric": metric,
                "unit": unit,
                "faster_rcnn_mean": faster["mean"],
                "faster_rcnn_std": faster["std"],
                "faster_rcnn_mean_plus_minus_std": faster["mean_plus_minus_std"],
                "faster_rcnn_n": faster["n"],
                "faster_rcnn_attempted_n": faster["attempted_n"],
                "faster_rcnn_undefined_seeds": faster["undefined_seeds"],
                "faster_rcnn_undefined_reason": faster["undefined_reason"],
                "yolo11s_mean": yolo["mean"],
                "yolo11s_std": yolo["std"],
                "yolo11s_mean_plus_minus_std": yolo["mean_plus_minus_std"],
                "yolo11s_n": yolo["n"],
                "yolo11s_attempted_n": yolo["attempted_n"],
                "yolo11s_undefined_seeds": yolo["undefined_seeds"],
                "yolo11s_undefined_reason": yolo["undefined_reason"],
                "sample_size_note": (
                    "Conditional matched-box metric; detector-specific n excludes only "
                    "seeds with no fixed-threshold true positive."
                    if metric in CONDITIONAL_METRICS
                    else "All predeclared attempted seeds are included."
                ),
            }
        )
    return long_rows, comparison


def preflight(config: Phase5Config) -> dict[str, Any]:
    """Validate contracts and report which trained artifacts are available."""

    loaded = load_and_validate_training_configs(config)
    missing: list[str] = []
    for run in config.runs:
        for path in (run.checkpoint, run.training_summary, run.compute_table):
            resolved = config.resolve(path)
            if not resolved.is_file():
                missing.append(resolved.as_posix())
    return {
        "experiment_id": config.experiment_id,
        "split": config.split,
        "seeds": list(config.seeds),
        "run_count": len(config.runs),
        "training_contracts_validated": len(loaded),
        "missing_artifacts": missing,
        "ready": not missing,
    }


def materialize_seed_timing_gates(config: Phase5Config) -> dict[str, Any]:
    """Reuse accepted timings for seed-only reruns with explicit source provenance."""

    from src.models.faster_rcnn_config import config_fingerprint
    from src.models.train_faster_rcnn import _implementation_identity as faster_identity
    from src.models.train_yolo import _implementation_identity as yolo_identity
    from src.models.yolo_config import yolo_config_sha256

    if config.timing_reuse_reviews is None:
        raise ValueError("seed timing reuse requires detector-specific reviewed source drift")
    training_configs = load_and_validate_training_configs(config)
    source_paths = {
        "faster_rcnn": config.resolve(config.timing_sources.faster_rcnn),
        "yolo11s": config.resolve(config.timing_sources.yolo11s),
    }
    written: list[dict[str, Any]] = []
    for detector, source_path in source_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"accepted timing source is missing: {source_path}")
        source = _json_payload(source_path)
        reference_config = training_configs[(detector, config.seeds[0])]
        reference_hash = (
            config_fingerprint(reference_config)
            if detector == "faster_rcnn"
            else yolo_config_sha256(reference_config)
        )
        if source.get("config_sha256") != reference_hash:
            raise ValueError(f"accepted {detector} timing source/config mismatch")
        if source.get("completed_epochs") != reference_config.benchmark.epochs:
            raise ValueError(f"accepted {detector} timing source is incomplete")
        source_sha256 = sha256_file(source_path)
        current_identity = (
            faster_identity(reference_config)
            if detector == "faster_rcnn"
            else yolo_identity(reference_config)
        )
        source_identity = source.get("implementation_identity")
        if not isinstance(source_identity, dict):
            raise ValueError(f"accepted {detector} timing source lacks source identity")
        if detector == "yolo11s":
            baseline_run = next(
                run
                for run in config.runs
                if run.detector == detector and run.seed == config.seeds[0]
            )
            baseline_summary = _json_payload(config.resolve(baseline_run.training_summary))
            if baseline_summary.get("implementation_identity") != source_identity:
                raise ValueError("YOLO timing and accepted training identities differ")
            reporting_identity = baseline_summary.get("reporting_implementation_identity")
            if not isinstance(reporting_identity, dict):
                raise ValueError("YOLO summary lacks its post-training reporting identity")
        source_files = {item["path"]: item["sha256"] for item in source_identity["source_files"]}
        current_files = {item["path"]: item["sha256"] for item in current_identity["source_files"]}
        changed_existing = {
            path for path, digest in source_files.items() if current_files.get(path) != digest
        }
        additions = set(current_files) - set(source_files)
        review = getattr(config.timing_reuse_reviews, detector)
        reviewed_changes = {path.as_posix() for path in review.reviewed_changes}
        reviewed_additions = {path.as_posix() for path in review.reviewed_additions}
        current_manifest = current_identity.get("source_manifest_sha256")
        if source_sha256 != review.timing_source_sha256:
            raise ValueError(f"{detector} timing source changed after its reuse review")
        if current_manifest != review.current_source_manifest_sha256:
            raise ValueError(f"{detector} current source changed after its reuse review")
        if changed_existing != reviewed_changes or additions != reviewed_additions:
            raise ValueError(
                f"{detector} timing source drift differs from its reviewed set: "
                f"changed={sorted(changed_existing)}, added={sorted(additions)}"
            )

        for seed in config.seeds[1:]:
            target_config = training_configs[(detector, seed)]
            target_hash = (
                config_fingerprint(target_config)
                if detector == "faster_rcnn"
                else yolo_config_sha256(target_config)
            )
            target_path = (
                target_config.run_artifact_path(
                    "benchmark", target_config.outputs.benchmark_estimate_path
                )
                if detector == "faster_rcnn"
                else target_config.run_dir("benchmark")
                / target_config.outputs.benchmark_estimate_name
            )
            derived = dict(source)
            derived.update(
                {
                    "config_sha256": target_hash,
                    "run_id": target_config.run_name("benchmark")
                    if detector == "yolo11s"
                    else target_config.outputs.benchmark_run_name,
                    "phase5_seed_only_timing_reuse": True,
                    "timing_source_path": source_path.as_posix(),
                    "timing_source_sha256": source_sha256,
                    "timing_source_run_id": source.get("run_id"),
                    "timing_source_seed": config.seeds[0],
                    "target_training_seed": seed,
                    "implementation_identity": current_identity,
                    "timing_source_implementation_identity": source_identity,
                    "timing_identity_compatible_additions": sorted(additions),
                    "timing_identity_compatible_changes": sorted(changed_existing),
                    "timing_reuse_review": review.model_dump(mode="json"),
                    "timing_reuse_reason": (
                        "Only the RNG seed and artifact identity change; model, data, "
                        "optimizer, AMP, batch, resolution, software, and GPU contracts "
                        "match the accepted timing run."
                    ),
                }
            )
            encoded = json.dumps(derived, indent=2, sort_keys=True, allow_nan=False) + "\n"
            gate_status = "created"
            if target_path.exists():
                existing = _json_payload(target_path)
                target_run = next(
                    run for run in config.runs if run.detector == detector and run.seed == seed
                )
                summary_path = config.resolve(target_run.training_summary)
                completed_summary = _json_payload(summary_path) if summary_path.is_file() else {}
                historical_config_match = completed_summary.get(
                    "status"
                ) == "complete" and existing.get("config_sha256") == completed_summary.get(
                    "config_sha256"
                )
                reusable_existing = (
                    existing.get("phase5_seed_only_timing_reuse") is True
                    and existing.get("timing_source_sha256") == source_sha256
                    and existing.get("target_training_seed") == seed
                    and (existing.get("config_sha256") == target_hash or historical_config_match)
                )
                if target_path.read_text(encoding="utf-8") == encoded:
                    gate_status = "unchanged"
                elif reusable_existing:
                    gate_status = "preserved_historical"
                else:
                    raise FileExistsError(f"refusing to overwrite timing gate: {target_path}")
            else:
                _atomic_bytes(target_path, encoded.encode("utf-8"))
            written.append(
                {
                    "detector": detector,
                    "seed": seed,
                    "path": target_path.as_posix(),
                    "sha256": sha256_file(target_path),
                    "status": gate_status,
                    "timing_source_sha256": source_sha256,
                }
            )
    return {"status": "complete", "timing_gates": written}


def run_evaluation(config: Phase5Config) -> dict[str, Any]:
    """Collect all raw predictions, evaluate identically, and write comparison tables."""

    from src.utils.seed import initialize_reproducibility, seed_everything

    training_configs = load_and_validate_training_configs(config)
    readiness = preflight(config)
    if not readiness["ready"]:
        raise FileNotFoundError(
            "Phase 5 evaluation is missing trained artifacts:\n"
            + "\n".join(readiness["missing_artifacts"])
        )
    log_dir = config.resolve(config.outputs.log_dir)
    initialize_reproducibility(config.seeds[0], log_dir)
    faster_reference = training_configs[("faster_rcnn", config.seeds[0])]
    dataset = _load_test_dataset(config, faster_reference)
    targets = _targets_from_dataset(dataset)
    annotation_path = dataset.annotation_file.resolve()
    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run in sorted(config.runs, key=lambda item: (item.detector, item.seed)):
        seed_everything(run.seed)
        print(f"evaluating {run.detector} seed {run.seed} on {config.split}", flush=True)
        model_config = training_configs[(run.detector, run.seed)]
        if run.detector == "faster_rcnn":
            predictions, inference_seconds = _collect_faster_rcnn_predictions(
                run,
                config,
                model_config,
                dataset,
                expected_config_sha256=_recorded_training_config_sha256(run, config),
            )
        else:
            predictions, inference_seconds = _collect_yolo_predictions(
                run, config, model_config, dataset
            )
        metrics = evaluate_prediction_records(
            predictions,
            targets,
            category_names=dataset.category_names,
            settings=config.evaluation,
        )
        checkpoint_sha256 = sha256_file(config.resolve(run.checkpoint))
        bundle_path = config.resolve(config.outputs.prediction_bundles_dir) / (
            f"{run.detector}_seed{run.seed}_test_predictions.json.gz"
        )
        _write_prediction_bundle(
            bundle_path,
            run=run,
            checkpoint_sha256=checkpoint_sha256,
            annotation_path=annotation_path,
            predictions=predictions,
            metrics=metrics,
            settings=config.evaluation,
        )
        maximum_prediction_score = max(
            (float(np.max(item.scores)) for item in predictions if len(item.scores)),
            default=None,
        )
        row = _artifact_row(
            run,
            config,
            metrics,
            inference_seconds,
            bundle_path,
            maximum_prediction_score,
        )
        rows.append(row)
        run_summaries.append(
            {
                "detector": run.detector,
                "seed": run.seed,
                "metrics": metrics,
                "comparison_row": row,
                "prediction_bundle_sha256": sha256_file(bundle_path),
            }
        )
        del predictions, metrics
        gc.collect()

    rows.sort(key=lambda row: (str(row["detector"]), int(row["seed"])))
    long_rows, comparison = aggregate_rows(rows, ddof=config.runtime.statistics_ddof)
    per_seed_path = _atomic_csv(
        config.resolve(config.outputs.per_seed_table), PER_SEED_FIELDS, rows
    )
    mean_std_fields = (
        "detector",
        "metric",
        "unit",
        "n",
        "attempted_n",
        "undefined_n",
        "undefined_seeds",
        "undefined_reason",
        "mean",
        "std",
        "mean_plus_minus_std",
    )
    mean_std_path = _atomic_csv(
        config.resolve(config.outputs.mean_std_table), mean_std_fields, long_rows
    )
    publication_fields = (
        "metric",
        "unit",
        "faster_rcnn_mean",
        "faster_rcnn_std",
        "faster_rcnn_mean_plus_minus_std",
        "faster_rcnn_n",
        "faster_rcnn_attempted_n",
        "faster_rcnn_undefined_seeds",
        "faster_rcnn_undefined_reason",
        "yolo11s_mean",
        "yolo11s_std",
        "yolo11s_mean_plus_minus_std",
        "yolo11s_n",
        "yolo11s_attempted_n",
        "yolo11s_undefined_seeds",
        "yolo11s_undefined_reason",
        "sample_size_note",
    )
    publication_path = _atomic_csv(
        config.resolve(config.outputs.publication_table), publication_fields, comparison
    )
    summary = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "config_path": config.source_path.as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "split": config.split,
        "test_annotation_path": annotation_path.as_posix(),
        "test_annotation_sha256": sha256_file(annotation_path),
        "image_count": len(dataset),
        "annotation_count": sum(len(record.annotations) for record in dataset.records),
        "category_names": {str(key): value for key, value in dataset.category_names.items()},
        "evaluation": config.evaluation.model_dump(mode="json"),
        "statistics": {
            "seeds": list(config.seeds),
            "n": len(config.seeds),
            "n_definition": "predeclared attempted training seeds",
            "standard_deviation": "sample",
            "ddof": config.runtime.statistics_ddof,
            "metric_sample_sizes": [
                {
                    "detector": row["detector"],
                    "metric": row["metric"],
                    "defined_n": row["n"],
                    "attempted_n": row["attempted_n"],
                    "undefined_seeds": row["undefined_seeds"],
                    "undefined_reason": row["undefined_reason"],
                }
                for row in long_rows
            ],
            "conditional_metric_policy": {
                "metrics": sorted(CONDITIONAL_METRICS),
                "definition": "mean localization over fixed-threshold true-positive matches",
                "undefined_when": "a seed has zero fixed-threshold true positives",
                "aggregation": "mean and sample SD over defined seed values only",
                "nulls_are_never_coerced_to_zero": True,
            },
        },
        "runs": run_summaries,
        "mean_std": long_rows,
        "comparison": comparison,
        "artifacts": {
            "per_seed_table": per_seed_path.as_posix(),
            "per_seed_table_sha256": sha256_file(per_seed_path),
            "mean_std_table": mean_std_path.as_posix(),
            "mean_std_table_sha256": sha256_file(mean_std_path),
            "publication_table": publication_path.as_posix(),
            "publication_table_sha256": sha256_file(publication_path),
        },
        "test_split_accessed": True,
        "status": "complete",
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--mode", choices=("preflight", "seed-gates", "evaluate"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_phase5_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return 0
    if args.mode == "seed-gates":
        print(json.dumps(materialize_seed_timing_gates(config), indent=2, sort_keys=True))
        return 0
    run_evaluation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
