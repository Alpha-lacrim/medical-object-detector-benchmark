"""Run the primary-seed Phase 6 corruption grid through the unified evaluator."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import (
    EvaluationSettings,
    evaluate_prediction_records,
    load_and_validate_training_configs,
    load_phase5_config,
    sha256_file,
)
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget
from src.robustness.corruptions import (
    CorruptionApplier,
    CorruptionCondition,
    CorruptionConfig,
    CorruptionDefinition,
    corruption_fingerprint,
    expand_conditions,
)

DetectorName = Literal["faster_rcnn", "yolo11s"]
METRIC_FIELDS = ("precision", "recall", "f1", "iou", "dice", "map_50", "map_50_95")


class StrictModel(BaseModel):
    """Reject undeclared config keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingSettings(StrictModel):
    """Fixed proportional stratified-sampling contract."""

    source_manifest: Path
    output_manifest: Path
    size: int = Field(ge=200, le=400)
    split_column: str = Field(min_length=1)
    split_value: str = Field(min_length=1)
    id_column: str = Field(min_length=1)
    stratum_column: str = Field(min_length=1)
    allocation: Literal["proportional_largest_remainder"]


class RuntimeSettings(StrictModel):
    """Matched evaluation runtime settings."""

    device: Literal["cuda"]
    inference_batch_size: Literal[1]
    num_workers: int = Field(ge=0)
    pin_memory: bool


class DetectorSettings(StrictModel):
    """One primary checkpoint and its frozen clean predictions."""

    detector: DetectorName
    training_config: Path
    checkpoint: Path
    clean_prediction_bundle: Path


class OutputSettings(StrictModel):
    """All Phase 6 generated artifacts."""

    log_dir: Path
    prediction_bundles_dir: Path
    summary_json: Path
    detailed_table: Path
    curve_table: Path
    family_mean_table: Path
    raw_curve_figure: Path
    relative_curve_figure: Path


class Phase6Config(StrictModel):
    """Strict complete Phase 6 experiment contract."""

    schema_version: Literal[2]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    phase5_evaluation_config: Path
    phase5_summary: Path
    sampling: SamplingSettings
    evaluation: EvaluationSettings
    runtime: RuntimeSettings
    detectors: tuple[DetectorSettings, DetectorSettings]
    corruptions: tuple[CorruptionDefinition, ...]
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_grid(self) -> Phase6Config:
        detector_names = [item.detector for item in self.detectors]
        if set(detector_names) != {"faster_rcnn", "yolo11s"} or len(set(detector_names)) != 2:
            raise ValueError("detectors must contain Faster R-CNN and YOLO11s exactly once")
        names = [item.name for item in self.corruptions]
        if len(set(names)) != len(names):
            raise ValueError("corruption names must be unique")
        families = {item.family for item in self.corruptions}
        if families != {"lighting", "noise", "blur", "compression"}:
            raise ValueError("grid must cover lighting, noise, blur, and compression")
        kind_counts = Counter(item.kind for item in self.corruptions)
        expected = {
            "brightness": 2,
            "gaussian_noise": 1,
            "salt_pepper": 1,
            "gaussian_blur": 1,
            "motion_blur": 1,
            "jpeg": 1,
        }
        if kind_counts != expected:
            raise ValueError("grid does not contain the required seven corruption types")
        brightness = [item for item in self.corruptions if item.kind == "brightness"]
        if not any(all(level.value < 1 for level in item.levels) for item in brightness):
            raise ValueError("lighting grid lacks a darker curve")
        if not any(all(level.value > 1 for level in item.levels) for item in brightness):
            raise ValueError("lighting grid lacks a brighter curve")
        jpeg_values = {
            int(level.value)
            for item in self.corruptions
            if item.kind == "jpeg"
            for level in item.levels
        }
        if not {20, 50} <= jpeg_values:
            raise ValueError("JPEG curve must explicitly include qualities 20 and 50")
        return self

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def corruption_config(self) -> CorruptionConfig:
        return CorruptionConfig(
            schema_version=self.schema_version,
            seed=self.seed,
            corruptions=self.corruptions,
        )


def load_robustness_config(path: str | Path) -> Phase6Config:
    """Load Phase 6 YAML without importing detector frameworks."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("robustness config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return Phase6Config.model_validate(payload)


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
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, encoded)


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
                writer.writerow({field: row.get(field) for field in fieldnames})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _largest_remainder_allocation(counts: Mapping[str, int], size: int) -> dict[str, int]:
    """Allocate an exact proportional sample with deterministic remainder ties."""

    total = sum(counts.values())
    if total < size or not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("strata must be non-empty and contain at least the sample size")
    quotas = {name: size * count / total for name, count in counts.items()}
    allocation = {name: math.floor(quota) for name, quota in quotas.items()}
    remaining = size - sum(allocation.values())
    order = sorted(counts, key=lambda name: (-(quotas[name] % 1), name))
    for name in order[:remaining]:
        allocation[name] += 1
    if sum(allocation.values()) != size:
        raise AssertionError("largest-remainder allocation did not preserve sample size")
    return allocation


def draw_stratified_subsample(
    settings: SamplingSettings,
    *,
    project_root: Path,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Draw and atomically materialize the fixed robustness image manifest."""

    source = (
        settings.source_manifest
        if settings.source_manifest.is_absolute()
        else (project_root / settings.source_manifest).resolve()
    )
    output = (
        settings.output_manifest
        if settings.output_manifest.is_absolute()
        else (project_root / settings.output_manifest).resolve()
    )
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("sampling manifest has no CSV header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    required = {
        settings.split_column,
        settings.id_column,
        settings.stratum_column,
        "image_id",
        "is_positive",
        "box_count",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"sampling manifest lacks columns: {sorted(missing)}")
    if any(row[settings.split_column] != settings.split_value for row in rows):
        raise ValueError("sampling source contains a row from the wrong split")
    identifiers = [row[settings.id_column] for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError("sampling identifiers must be non-empty and unique")

    by_stratum: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        stratum = row[settings.stratum_column]
        if not stratum:
            raise ValueError("sampling strata must be non-empty")
        by_stratum[stratum].append(row)
    full_counts = {name: len(items) for name, items in sorted(by_stratum.items())}
    allocation = _largest_remainder_allocation(full_counts, settings.size)
    generator = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    for name in sorted(by_stratum):
        candidates = sorted(by_stratum[name], key=lambda row: row[settings.id_column])
        chosen_indices = generator.choice(len(candidates), size=allocation[name], replace=False)
        selected.extend(candidates[int(index)] for index in sorted(chosen_indices))
    selected.sort(key=lambda row: row[settings.id_column])
    if len(selected) != settings.size:
        raise AssertionError("stratified sampler returned the wrong image count")

    output_rows = [
        {**row, "robustness_sample_index": str(index)} for index, row in enumerate(selected)
    ]
    output_fields = [*fieldnames, "robustness_sample_index"]
    _atomic_csv(output, output_fields, output_rows)
    sample_counts = Counter(row[settings.stratum_column] for row in selected)
    audit = {
        "method": settings.allocation,
        "rng": "numpy.random.Generator(PCG64)",
        "seed": seed,
        "source_manifest": source.as_posix(),
        "source_manifest_sha256": sha256_file(source),
        "output_manifest": output.as_posix(),
        "output_manifest_sha256": sha256_file(output),
        "full_image_count": len(rows),
        "sample_image_count": len(selected),
        "full_stratum_counts": full_counts,
        "sample_stratum_counts": dict(sorted(sample_counts.items())),
        "sample_positive_images": sum(int(row["is_positive"]) for row in selected),
        "sample_negative_images": sum(not int(row["is_positive"]) for row in selected),
        "sample_box_count": sum(int(row["box_count"]) for row in selected),
        "sample_patient_count": len({row["nih_patient_id"] for row in selected}),
        "ordering": (
            "Sort strata and identifiers lexicographically, draw without replacement "
            "sequentially from one seeded generator, then sort selected identifiers."
        ),
    }
    return selected, audit


class CorruptedSubsetDataset:
    """Selected canonical COCO records with an optional photometric corruption."""

    def __init__(
        self,
        base_dataset: Any,
        selected_names: set[str],
        *,
        condition: CorruptionCondition | None,
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.condition = condition
        self.applier = (
            CorruptionApplier(condition, base_seed=seed) if condition is not None else None
        )
        base_indices = {
            record.file_name: index for index, record in enumerate(base_dataset.records)
        }
        unknown = selected_names - set(base_indices)
        if unknown:
            raise ValueError(f"sample manifest references unknown COCO images: {sorted(unknown)}")
        self.records = tuple(
            record for record in base_dataset.records if record.file_name in selected_names
        )
        if len(self.records) != len(selected_names):
            raise ValueError("sample and COCO filenames are not one-to-one")
        self._base_indices = tuple(base_indices[record.file_name] for record in self.records)
        self.annotation_file = base_dataset.annotation_file
        self.category_names = base_dataset.category_names
        self.category_id_to_label = base_dataset.category_id_to_label
        self.label_to_category_id = base_dataset.label_to_category_id
        self.num_foreground_classes = base_dataset.num_foreground_classes

    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, index: int) -> Path:
        return self.base_dataset.image_path(self._base_indices[index])

    def load_pil(self, index: int) -> Image.Image:
        record = self.records[index]
        with Image.open(self.image_path(index)) as source:
            if source.size != (record.width, record.height):
                raise ValueError(f"image dimensions changed for {record.file_name}")
            image = source.convert("RGB")
            image.load()
        if self.applier is not None:
            image = self.applier(image, image_id=record.file_name)
        return image

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch

        record = self.records[index]
        pixels = np.asarray(self.load_pil(index), dtype=np.float32) / np.float32(255.0)
        image = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        if record.annotations:
            boxes = torch.tensor(
                [annotation.bbox_xyxy for annotation in record.annotations],
                dtype=torch.float32,
            )
            labels = torch.tensor(
                [
                    self.category_id_to_label[annotation.category_id]
                    for annotation in record.annotations
                ],
                dtype=torch.int64,
            )
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([record.id], dtype=torch.int64),
        }
        return image, target


def _targets_from_dataset(dataset: CorruptedSubsetDataset) -> list[ImageTarget]:
    return [
        ImageTarget(
            image_id=record.file_name,
            image_size=(record.height, record.width),
            boxes_xyxy=np.asarray(
                [annotation.bbox_xyxy for annotation in record.annotations],
                dtype=np.float64,
            ).reshape(-1, 4),
            labels=np.asarray(
                [annotation.category_id for annotation in record.annotations],
                dtype=np.int64,
            ),
        )
        for record in dataset.records
    ]


def _serialize_prediction(prediction: ImagePrediction) -> dict[str, Any]:
    return {
        "image_id": prediction.image_id,
        "image_size": list(prediction.image_size),
        "boxes_xyxy": prediction.boxes_xyxy.tolist(),
        "labels": prediction.labels.tolist(),
        "scores": prediction.scores.tolist(),
    }


def _deserialize_prediction(payload: Mapping[str, Any]) -> ImagePrediction:
    return ImagePrediction(
        image_id=str(payload["image_id"]),
        image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        boxes_xyxy=np.asarray(payload["boxes_xyxy"], dtype=np.float64).reshape(-1, 4),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        scores=np.asarray(payload["scores"], dtype=np.float64),
    )


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"prediction bundle must contain an object: {path}")
    return payload


def _write_prediction_bundle(
    path: Path,
    *,
    identity: Mapping[str, Any],
    predictions: Sequence[ImagePrediction],
    metrics: Mapping[str, Any],
    inference_seconds: float | None,
    clean_source: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "identity": dict(identity),
        "predictions": [_serialize_prediction(item) for item in predictions],
        "metrics": metrics,
        "inference_seconds": inference_seconds,
        "clean_source": clean_source,
        "status": "complete",
    }
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, gzip.compress(raw, compresslevel=9, mtime=0))


def _cached_result(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_gzip_json(path)
    if payload.get("status") != "complete" or payload.get("identity") != dict(identity):
        raise ValueError(f"refusing incompatible cached robustness bundle: {path}")
    if len(payload.get("predictions", [])) != identity["sample_image_count"]:
        raise ValueError(f"cached bundle has the wrong prediction count: {path}")
    return payload


def _condition_payload(condition: CorruptionCondition | None) -> dict[str, Any]:
    if condition is None:
        return {
            "condition_id": "clean",
            "family": "clean",
            "corruption": "clean",
            "kind": "clean",
            "severity": 0,
            "value": None,
            "unit": "none",
        }
    return {
        "condition_id": condition.condition_id,
        "family": condition.family,
        "corruption": condition.name,
        "kind": condition.kind,
        "severity": condition.severity,
        "value": condition.value,
        "unit": condition.unit,
    }


def _source_identity(config: Phase6Config) -> dict[str, str]:
    paths = (
        config.source_path,
        config.project_root / "src" / "robustness" / "run_robustness.py",
        config.project_root / "src" / "meddet_benchmark" / "corruptions.py",
        config.project_root / "src" / "evaluate.py",
    )
    return {path.relative_to(config.project_root).as_posix(): sha256_file(path) for path in paths}


def _bundle_identity(
    *,
    config: Phase6Config,
    detector: DetectorSettings,
    condition: CorruptionCondition | None,
    sample_audit: Mapping[str, Any],
    checkpoint_sha256: str,
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "detector": detector.detector,
        "seed": config.seed,
        "condition": _condition_payload(condition),
        "sample_manifest_sha256": sample_audit["output_manifest_sha256"],
        "sample_image_count": sample_audit["sample_image_count"],
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation": config.evaluation.model_dump(mode="json"),
        "corruption_fingerprint": corruption_fingerprint(config.corruption_config),
        "source_identity": dict(source_identity),
    }


def _bundle_path(
    config: Phase6Config,
    detector: DetectorSettings,
    condition: CorruptionCondition | None,
) -> Path:
    condition_id = "clean" if condition is None else condition.condition_id
    return config.resolve(config.outputs.prediction_bundles_dir) / (
        f"{detector.detector}__{condition_id}.json.gz"
    )


def _clean_predictions(
    detector: DetectorSettings,
    config: Phase6Config,
    selected_names: set[str],
) -> tuple[list[ImagePrediction], dict[str, Any]]:
    source_path = config.resolve(detector.clean_prediction_bundle)
    payload = _read_gzip_json(source_path)
    if payload.get("detector") != detector.detector or payload.get("seed") != config.seed:
        raise ValueError(f"clean bundle identity mismatch: {source_path}")
    checkpoint_sha256 = sha256_file(config.resolve(detector.checkpoint))
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"clean bundle checkpoint mismatch: {source_path}")
    predictions = [
        _deserialize_prediction(item)
        for item in payload.get("predictions", [])
        if item.get("image_id") in selected_names
    ]
    if {item.image_id for item in predictions} != selected_names:
        raise ValueError(f"clean bundle does not cover the robustness sample: {source_path}")
    predictions.sort(key=lambda item: item.image_id)
    provenance = {
        "path": source_path.as_posix(),
        "sha256": sha256_file(source_path),
        "reuse_reason": (
            "The Phase 5 seed-17 predictions use the same checkpoint, images, and "
            "unified thresholds; filtering them avoids an unnecessary clean re-run."
        ),
    }
    return predictions, provenance


def _collect_faster_predictions(
    model: Any,
    device: Any,
    model_config: Any,
    phase_config: Phase6Config,
    dataset: CorruptedSubsetDataset,
) -> tuple[list[ImagePrediction], float]:
    import torch
    from torch.utils.data import DataLoader

    from src.models.faster_rcnn_data import detection_collate_fn

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
                labels = np.asarray(
                    [
                        dataset.label_to_category_id[int(label)]
                        for label in output["labels"].detach().cpu().numpy()[keep]
                    ],
                    dtype=np.int64,
                )
                predictions.append(
                    ImagePrediction(
                        image_id=record.file_name,
                        image_size=(int(image.shape[-2]), int(image.shape[-1])),
                        boxes_xyxy=output["boxes"].detach().cpu().numpy()[keep],
                        labels=labels,
                        scores=scores[keep],
                    )
                )
    del loader
    return predictions, elapsed_seconds


def _load_faster_model(
    detector: DetectorSettings,
    config: Phase6Config,
    model_config: Any,
    dataset: CorruptedSubsetDataset,
) -> tuple[Any, Any]:
    import torch

    from src.models.faster_rcnn_config import config_fingerprint
    from src.models.faster_rcnn_model import build_faster_rcnn

    checkpoint = torch.load(
        config.resolve(detector.checkpoint), map_location="cpu", weights_only=False
    )
    if checkpoint.get("config_sha256") != config_fingerprint(model_config):
        raise ValueError("Faster R-CNN checkpoint/config mismatch")
    model, _metadata = build_faster_rcnn(
        dataset.num_foreground_classes,
        model_config.model,
        use_pretrained_weights=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(config.runtime.device)
    model.to(device).eval()
    del checkpoint
    return model, device


def _collect_yolo_predictions(
    yolo: Any,
    device: Any,
    model_config: Any,
    phase_config: Phase6Config,
    dataset: CorruptedSubsetDataset,
) -> tuple[list[ImagePrediction], float]:
    import torch

    category_ids = tuple(sorted(dataset.category_names))
    yolo_to_category = {index: category_id for index, category_id in enumerate(category_ids)}
    predictions: list[ImagePrediction] = []
    elapsed_seconds = 0.0
    for index, record in enumerate(dataset.records):
        image = dataset.load_pil(index)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        results = yolo.predict(
            source=image,
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
        if len(results) != 1 or tuple(results[0].orig_shape) != (record.height, record.width):
            raise ValueError(f"unexpected YOLO result geometry for {record.file_name}")
        boxes = results[0].boxes
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
                boxes_xyxy=boxes.xyxy.detach().cpu().numpy().astype(np.float64, copy=False),
                labels=labels,
                scores=boxes.conf.detach().cpu().numpy().astype(np.float64, copy=False),
            )
        )
    return predictions, elapsed_seconds


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    overall = metrics["operating_point"]["overall"]
    coco = metrics["coco"]
    return {
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "iou": overall["matched_mean_iou"],
        "dice": overall["matched_mean_box_dice"],
        "map_50": coco["ap50"],
        "map_50_95": coco["ap50_95"],
    }


def _result_row(
    record: Mapping[str, Any], clean_values: Mapping[str, float | None]
) -> dict[str, Any]:
    metrics = record["metrics"]
    values = _metric_values(metrics)
    overall = metrics["operating_point"]["overall"]
    condition = record["condition"]
    row: dict[str, Any] = {
        "detector": record["detector"],
        **condition,
        "image_count": metrics["coco"]["image_count"],
        "target_count": overall["target_count"],
        "true_positives": overall["tp"],
        "false_positives": overall["fp"],
        "false_negatives": overall["fn"],
        "inference_seconds": record["inference_seconds"],
        "prediction_bundle": record["prediction_bundle"],
        "prediction_bundle_sha256": record["prediction_bundle_sha256"],
        **values,
    }
    for metric in METRIC_FIELDS:
        raw = values[metric]
        clean = clean_values[metric]
        relative = None if raw is None or clean in {None, 0} else raw / clean
        row[f"{metric}_relative"] = relative
        row[f"{metric}_degradation"] = None if relative is None else 1 - relative
    return row


def _write_figure(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    corruption_order: Sequence[str],
    *,
    relative: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"faster_rcnn": "#1f77b4", "yolo11s": "#d62728"}
    labels = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    figure, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    for axis, corruption in zip(axes.flat, corruption_order, strict=False):
        for detector in ("faster_rcnn", "yolo11s"):
            clean = next(
                row for row in rows if row["detector"] == detector and row["severity"] == 0
            )
            corrupted = sorted(
                (
                    row
                    for row in rows
                    if row["detector"] == detector and row["corruption"] == corruption
                ),
                key=lambda row: int(row["severity"]),
            )
            field = "map_50_95_relative" if relative else "map_50_95"
            clean_value = 1.0 if relative else clean["map_50_95"]
            axis.plot(
                [0, *[int(row["severity"]) for row in corrupted]],
                [clean_value, *[row[field] for row in corrupted]],
                marker="o",
                linewidth=1.8,
                color=colors[detector],
                label=labels[detector],
            )
        axis.set_title(corruption.replace("_", " ").title())
        axis.set_xlabel("Severity (0 = clean)")
        axis.set_xticks(range(6))
        axis.grid(alpha=0.25)
        if relative:
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            axis.set_ylabel("mAP@0.5:0.95 / clean")
        else:
            axis.set_ylabel("mAP@0.5:0.95")
            axis.set_ylim(bottom=0)
    for axis in axes.flat[len(corruption_order) :]:
        axis.axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=2)
    title = (
        "Relative common-corruption performance"
        if relative
        else "Raw common-corruption performance"
    )
    figure.suptitle(
        title,
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=180)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return path


def _write_aggregate_artifacts(
    config: Phase6Config,
    records: Sequence[Mapping[str, Any]],
    sample_audit: Mapping[str, Any],
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    clean_values = {
        record["detector"]: _metric_values(record["metrics"])
        for record in records
        if record["condition"]["condition_id"] == "clean"
    }
    rows = [_result_row(record, clean_values[record["detector"]]) for record in records]
    condition_order = [
        "clean",
        *[item.condition_id for item in expand_conditions(config.corruption_config)],
    ]
    detector_order = {"faster_rcnn": 0, "yolo11s": 1}
    rows.sort(
        key=lambda row: (
            detector_order[str(row["detector"])],
            condition_order.index(str(row["condition_id"])),
        )
    )
    base_fields = (
        "detector",
        "condition_id",
        "family",
        "corruption",
        "kind",
        "severity",
        "value",
        "unit",
        "image_count",
        "target_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "inference_seconds",
        "prediction_bundle",
        "prediction_bundle_sha256",
    )
    metric_fields = tuple(
        field
        for metric in METRIC_FIELDS
        for field in (metric, f"{metric}_relative", f"{metric}_degradation")
    )
    detailed_path = _atomic_csv(
        config.resolve(config.outputs.detailed_table), (*base_fields, *metric_fields), rows
    )

    curve_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["severity"] == 0:
            continue
        for metric in METRIC_FIELDS:
            curve_rows.append(
                {
                    "detector": row["detector"],
                    "family": row["family"],
                    "corruption": row["corruption"],
                    "severity": row["severity"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "metric": metric,
                    "clean_performance": clean_values[row["detector"]][metric],
                    "corrupted_performance": row[metric],
                    "relative_performance": row[f"{metric}_relative"],
                    "relative_degradation": row[f"{metric}_degradation"],
                }
            )
    curve_fields = (
        "detector",
        "family",
        "corruption",
        "severity",
        "value",
        "unit",
        "metric",
        "clean_performance",
        "corrupted_performance",
        "relative_performance",
        "relative_degradation",
    )
    curve_path = _atomic_csv(config.resolve(config.outputs.curve_table), curve_fields, curve_rows)

    grouped: defaultdict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(row["detector"], row["family"], row["severity"], row["metric"])].append(row)
    family_rows: list[dict[str, Any]] = []
    for (detector, family, severity, metric), items in sorted(grouped.items()):
        raw = [
            float(item["corrupted_performance"])
            for item in items
            if item["corrupted_performance"] is not None
        ]
        relative = [
            float(item["relative_performance"])
            for item in items
            if item["relative_performance"] is not None
        ]
        degradation = [
            float(item["relative_degradation"])
            for item in items
            if item["relative_degradation"] is not None
        ]
        family_rows.append(
            {
                "detector": detector,
                "family": family,
                "severity": severity,
                "metric": metric,
                "corruption_type_count": len(items),
                "mean_corrupted_performance": float(np.mean(raw)) if raw else None,
                "mean_relative_performance": float(np.mean(relative)) if relative else None,
                "mean_relative_degradation": float(np.mean(degradation)) if degradation else None,
            }
        )
    family_fields = (
        "detector",
        "family",
        "severity",
        "metric",
        "corruption_type_count",
        "mean_corrupted_performance",
        "mean_relative_performance",
        "mean_relative_degradation",
    )
    family_path = _atomic_csv(
        config.resolve(config.outputs.family_mean_table), family_fields, family_rows
    )
    corruption_order = [item.name for item in config.corruptions]
    raw_figure = _write_figure(
        config.resolve(config.outputs.raw_curve_figure),
        rows,
        corruption_order,
        relative=False,
    )
    relative_figure = _write_figure(
        config.resolve(config.outputs.relative_curve_figure),
        rows,
        corruption_order,
        relative=True,
    )
    artifacts = {
        "detailed_table": detailed_path.as_posix(),
        "detailed_table_sha256": sha256_file(detailed_path),
        "curve_table": curve_path.as_posix(),
        "curve_table_sha256": sha256_file(curve_path),
        "family_mean_table": family_path.as_posix(),
        "family_mean_table_sha256": sha256_file(family_path),
        "raw_curve_figure": raw_figure.as_posix(),
        "raw_curve_figure_sha256": sha256_file(raw_figure),
        "relative_curve_figure": relative_figure.as_posix(),
        "relative_curve_figure_sha256": sha256_file(relative_figure),
    }
    summary = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "seed_scope": {
            "training_seed": config.seed,
            "scope": "primary checkpoint only for both detectors",
        },
        "config_path": config.source_path.as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "corruption_fingerprint": corruption_fingerprint(config.corruption_config),
        "source_identity": dict(source_identity),
        "sampling": dict(sample_audit),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "grid": {
            "corruption_type_count": len(config.corruptions),
            "severity_count_per_type": [len(item.levels) for item in config.corruptions],
            "corrupted_condition_count": len(expand_conditions(config.corruption_config)),
            "detector_condition_count_including_clean": len(records),
            "corruptions": [item.model_dump(mode="json") for item in config.corruptions],
        },
        "results": rows,
        "family_means": family_rows,
        "prediction_bundles": [
            {
                "detector": record["detector"],
                "condition_id": record["condition"]["condition_id"],
                "path": record["prediction_bundle"],
                "sha256": record["prediction_bundle_sha256"],
            }
            for record in records
        ],
        "artifacts": artifacts,
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    artifacts["summary_json"] = summary_path.as_posix()
    artifacts["summary_json_sha256"] = sha256_file(summary_path)
    return {"summary": summary, "artifacts": artifacts}


def _prepare_experiment(config: Phase6Config) -> dict[str, Any]:
    phase5_config = load_phase5_config(config.resolve(config.phase5_evaluation_config))
    if phase5_config.evaluation != config.evaluation:
        raise ValueError("Phase 6 evaluation thresholds differ from frozen Phase 5")
    phase5_summary_path = config.resolve(config.phase5_summary)
    phase5_summary = json.loads(phase5_summary_path.read_text(encoding="utf-8"))
    if phase5_summary.get("status") != "complete" or phase5_summary.get(
        "config_sha256"
    ) != sha256_file(phase5_config.source_path):
        raise ValueError("Phase 5 summary is incomplete or does not match its config")
    training_configs = load_and_validate_training_configs(phase5_config)
    primary_runs = {run.detector: run for run in phase5_config.runs if run.seed == config.seed}
    if set(primary_runs) != {"faster_rcnn", "yolo11s"}:
        raise ValueError("Phase 5 does not contain both configured primary-seed runs")
    for detector in config.detectors:
        primary = primary_runs[detector.detector]
        if config.resolve(detector.training_config) != phase5_config.resolve(
            primary.training_config
        ) or config.resolve(detector.checkpoint) != phase5_config.resolve(primary.checkpoint):
            raise ValueError(f"{detector.detector} does not match its frozen Phase 5 run")
        for path in (
            detector.training_config,
            detector.checkpoint,
            detector.clean_prediction_bundle,
        ):
            if not config.resolve(path).is_file():
                raise FileNotFoundError(config.resolve(path))

    from src.models.faster_rcnn_data import CocoDetectionDataset

    faster_config = training_configs[("faster_rcnn", config.seed)]
    base_dataset = CocoDetectionDataset(
        faster_config.resolve(faster_config.data.dataset_config),
        "test",
        mode="full",
        project_root=config.project_root,
    )
    selected_rows, sample_audit = draw_stratified_subsample(
        config.sampling, project_root=config.project_root, seed=config.seed
    )
    selected_names = {row[config.sampling.id_column] for row in selected_rows}
    subset = CorruptedSubsetDataset(base_dataset, selected_names, condition=None, seed=config.seed)
    targets = _targets_from_dataset(subset)
    if len(targets) != sample_audit["sample_image_count"]:
        raise ValueError("sample target count disagrees with the manifest")
    return {
        "phase5_config": phase5_config,
        "phase5_summary_path": phase5_summary_path,
        "training_configs": training_configs,
        "base_dataset": base_dataset,
        "selected_names": selected_names,
        "sample_audit": sample_audit,
        "targets": targets,
        "category_names": subset.category_names,
    }


def preflight(config: Phase6Config) -> dict[str, Any]:
    """Validate all frozen identities and materialize the deterministic sample."""

    prepared = _prepare_experiment(config)
    conditions = expand_conditions(config.corruption_config)
    return {
        "status": "ready",
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "sampling": prepared["sample_audit"],
        "corruption_type_count": len(config.corruptions),
        "severity_count_per_type": [len(item.levels) for item in config.corruptions],
        "corrupted_condition_count": len(conditions),
        "detector_count": len(config.detectors),
        "corrupted_inference_count": len(conditions) * len(config.detectors) * config.sampling.size,
        "clean_baseline": "filtered from frozen Phase 5 seed-17 prediction bundles",
    }


def run_robustness(config: Phase6Config) -> dict[str, Any]:
    """Run or resume the complete corruption grid and write report artifacts."""

    prepared = _prepare_experiment(config)
    from src.utils.seed import initialize_reproducibility, seed_everything

    log_dir = config.resolve(config.outputs.log_dir)
    initialize_reproducibility(config.seed, log_dir)
    source_identity = _source_identity(config)
    sample_audit = prepared["sample_audit"]
    selected_names = prepared["selected_names"]
    targets = prepared["targets"]
    category_names = prepared["category_names"]
    conditions = expand_conditions(config.corruption_config)
    records: list[dict[str, Any]] = []

    for detector in config.detectors:
        checkpoint_sha256 = sha256_file(config.resolve(detector.checkpoint))
        clean_identity = _bundle_identity(
            config=config,
            detector=detector,
            condition=None,
            sample_audit=sample_audit,
            checkpoint_sha256=checkpoint_sha256,
            source_identity=source_identity,
        )
        clean_path = _bundle_path(config, detector, None)
        clean_cached = _cached_result(clean_path, clean_identity)
        if clean_cached is None:
            predictions, provenance = _clean_predictions(detector, config, selected_names)
            metrics = evaluate_prediction_records(
                predictions,
                targets,
                category_names=category_names,
                settings=config.evaluation,
            )
            _write_prediction_bundle(
                clean_path,
                identity=clean_identity,
                predictions=predictions,
                metrics=metrics,
                inference_seconds=None,
                clean_source=provenance,
            )
            clean_cached = _read_gzip_json(clean_path)
        records.append(
            {
                "detector": detector.detector,
                "condition": _condition_payload(None),
                "metrics": clean_cached["metrics"],
                "inference_seconds": clean_cached["inference_seconds"],
                "prediction_bundle": clean_path.as_posix(),
                "prediction_bundle_sha256": sha256_file(clean_path),
            }
        )

    for detector in config.detectors:
        checkpoint_sha256 = sha256_file(config.resolve(detector.checkpoint))
        pending: list[tuple[CorruptionCondition, Path, dict[str, Any]]] = []
        cached_by_condition: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            identity = _bundle_identity(
                config=config,
                detector=detector,
                condition=condition,
                sample_audit=sample_audit,
                checkpoint_sha256=checkpoint_sha256,
                source_identity=source_identity,
            )
            path = _bundle_path(config, detector, condition)
            cached = _cached_result(path, identity)
            if cached is None:
                pending.append((condition, path, identity))
            else:
                cached_by_condition[condition.condition_id] = cached

        model_config = prepared["training_configs"][(detector.detector, config.seed)]
        model: Any | None = None
        device: Any | None = None
        if pending:
            seed_everything(config.seed)
            clean_subset = CorruptedSubsetDataset(
                prepared["base_dataset"], selected_names, condition=None, seed=config.seed
            )
            if detector.detector == "faster_rcnn":
                model, device = _load_faster_model(detector, config, model_config, clean_subset)
            else:
                import torch
                from ultralytics import YOLO

                model = YOLO(config.resolve(detector.checkpoint).as_posix())
                device = torch.device(f"cuda:{model_config.runtime.device}")
        for index, (condition, path, identity) in enumerate(pending, start=1):
            print(
                f"[{detector.detector} {index}/{len(pending)}] {condition.condition_id}",
                flush=True,
            )
            seed_everything(config.seed)
            dataset = CorruptedSubsetDataset(
                prepared["base_dataset"],
                selected_names,
                condition=condition,
                seed=config.seed,
            )
            if detector.detector == "faster_rcnn":
                predictions, inference_seconds = _collect_faster_predictions(
                    model, device, model_config, config, dataset
                )
            else:
                predictions, inference_seconds = _collect_yolo_predictions(
                    model, device, model_config, config, dataset
                )
            metrics = evaluate_prediction_records(
                predictions,
                targets,
                category_names=category_names,
                settings=config.evaluation,
            )
            _write_prediction_bundle(
                path,
                identity=identity,
                predictions=predictions,
                metrics=metrics,
                inference_seconds=inference_seconds,
            )
            cached_by_condition[condition.condition_id] = _read_gzip_json(path)
            del predictions, metrics, dataset
            gc.collect()
        if model is not None:
            del model
            gc.collect()
            import torch

            torch.cuda.empty_cache()

        for condition in conditions:
            path = _bundle_path(config, detector, condition)
            cached = cached_by_condition[condition.condition_id]
            records.append(
                {
                    "detector": detector.detector,
                    "condition": _condition_payload(condition),
                    "metrics": cached["metrics"],
                    "inference_seconds": cached["inference_seconds"],
                    "prediction_bundle": path.as_posix(),
                    "prediction_bundle_sha256": sha256_file(path),
                }
            )

    expected_records = len(config.detectors) * (1 + len(conditions))
    if len(records) != expected_records:
        raise AssertionError("robustness grid is incomplete")
    result = _write_aggregate_artifacts(config, records, sample_audit, source_identity)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": result["artifacts"]["summary_json"],
                "record_count": len(records),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corruptions.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_robustness_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return 0
    run_robustness(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
