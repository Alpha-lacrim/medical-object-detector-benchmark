"""Strict configuration contract for the Batch 3 YOLO11s baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class YoloDataConfig(StrictModel):
    """Canonical COCO inputs and generated Ultralytics dataset paths."""

    images_dir: Path
    train_annotations: Path
    val_annotations: Path
    test_annotations: Path
    yolo_root: Path
    dataset_yaml: Path
    smoke_dataset_yaml: Path
    link_mode: Literal["hardlink"]


class YoloModelConfig(StrictModel):
    """Pinned detector identity and image policy."""

    family: Literal["YOLO11"]
    scale: Literal["small"]
    weights: str
    weights_path: Path
    implementation: Literal["ultralytics"]
    ultralytics_version: str
    pretrained: bool
    input_size: int = Field(ge=32)
    batch_norm_policy: Literal["train_statistics"]
    rationale: str


class YoloRuntimeConfig(StrictModel):
    """Single-GPU execution settings."""

    device: int = Field(ge=0)
    amp: bool
    amp_dtype: Literal["bfloat16"]
    loss_dtype: Literal["float32"]
    batch_size: int = Field(ge=2, le=4)
    nominal_batch_size: int = Field(ge=2)
    workers: int = Field(ge=0)
    cache: bool
    deterministic: bool


class YoloOptimizerConfig(StrictModel):
    """Optimizer settings matched to the Faster R-CNN arm."""

    name: Literal["SGD"]
    learning_rate: float = Field(gt=0)
    final_learning_rate_fraction: float = Field(gt=0)
    momentum: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    nesterov: bool
    warmup_epochs: float = Field(ge=0)
    warmup_bias_learning_rate: float = Field(ge=0)
    warmup_momentum: float = Field(ge=0, lt=1)


class YoloLossConfig(StrictModel):
    """Ultralytics detection-loss weights."""

    box: float = Field(gt=0)
    cls: float = Field(gt=0)
    dfl: float = Field(gt=0)


class YoloEarlyStoppingConfig(StrictModel):
    """Validation-mAP early-stopping settings."""

    metric: Literal["val_map_50_95"]
    mode: Literal["max"]
    patience: int = Field(gt=0)
    min_delta: float = Field(ge=0)
    min_epochs: int = Field(gt=0)


class YoloAugmentationConfig(StrictModel):
    """Explicit identity augmentation policy for detector parity."""

    policy: Literal["match_faster_rcnn_none"]
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    flipud: float
    fliplr: float
    bgr: float
    mosaic: float
    mixup: float
    cutmix: float
    copy_paste: float
    auto_augment: str | None
    erasing: float
    close_mosaic: int
    multi_scale: float

    @model_validator(mode="after")
    def identity_policy_is_really_disabled(self) -> YoloAugmentationConfig:
        """Prevent a documented parity run from silently enabling augmentation."""

        values = self.model_dump(exclude={"policy", "auto_augment"})
        if any(float(value) != 0.0 for value in values.values()):
            raise ValueError(
                "match_faster_rcnn_none requires every numeric augmentation to be zero"
            )
        if self.auto_augment is not None:
            raise ValueError("match_faster_rcnn_none requires auto_augment: null")
        return self


class YoloTrainingConfig(StrictModel):
    """Full optimization and augmentation contract."""

    max_epochs: int = Field(gt=0)
    optimizer: YoloOptimizerConfig
    loss: YoloLossConfig
    early_stopping: YoloEarlyStoppingConfig
    augmentation: YoloAugmentationConfig

    @model_validator(mode="after")
    def stopping_fits_epoch_budget(self) -> YoloTrainingConfig:
        """Ensure the declared minimum can be reached."""

        if self.early_stopping.min_epochs > self.max_epochs:
            raise ValueError("early-stopping minimum exceeds maximum epochs")
        return self


class YoloEvaluationConfig(StrictModel):
    """Native-NMS and shared evaluation thresholds."""

    inference_minimum_score: float = Field(ge=0, le=1)
    score_threshold: float = Field(ge=0, le=1)
    match_iou_threshold: float = Field(gt=0, le=1)
    nms_iou_threshold: float = Field(gt=0, le=1)
    max_detections: int = Field(ge=10)


class YoloBenchmarkConfig(StrictModel):
    """Real-data timing benchmark size."""

    epochs: int = Field(ge=2, le=3)


class YoloSmokeConfig(StrictModel):
    """Bounded smoke dataset and epoch counts."""

    epochs: int = Field(gt=0)
    train_images: int = Field(ge=2)
    val_images: int = Field(ge=2)


class YoloProfilingConfig(StrictModel):
    """Synchronized inference and FLOP profiling settings."""

    batch_size: int = Field(gt=0)
    warmup_batches: int = Field(ge=0)
    timed_batches: int = Field(gt=0)
    profile_flops: bool


class YoloOutputsConfig(StrictModel):
    """All generated Batch 3 artifact locations."""

    logs_dir: Path
    checkpoints_dir: Path
    tables_dir: Path
    figures_dir: Path
    smoke_run_name: str
    benchmark_run_name: str
    train_run_name: str
    checkpoint_dir: Path
    validation_table: Path
    compute_table: Path
    training_curves: Path
    summary_name: str
    epoch_timing_name: str
    benchmark_estimate_name: str


class YoloConfig(StrictModel):
    """Top-level Batch 3 configuration."""

    schema_version: Literal[1]
    experiment_id: str
    seed: int = Field(ge=0)
    data: YoloDataConfig
    model: YoloModelConfig
    runtime: YoloRuntimeConfig
    training: YoloTrainingConfig
    evaluation: YoloEvaluationConfig
    benchmark: YoloBenchmarkConfig
    smoke: YoloSmokeConfig
    profiling: YoloProfilingConfig
    outputs: YoloOutputsConfig
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def enforce_matched_hardware_policy(self) -> YoloConfig:
        """Reject accidental drift from the approved hardware protocol."""

        if not self.runtime.amp:
            raise ValueError("YOLO AMP is mandatory")
        if self.runtime.nominal_batch_size != self.runtime.batch_size:
            raise ValueError("nominal_batch_size must equal batch_size for effective batch parity")
        if self.runtime.cache:
            raise ValueError("RAM/disk image caching is disabled for the scoped run")
        if self.model.input_size != 640:
            raise ValueError("the controlled comparison requires 640-pixel input")
        if self.training.optimizer.nesterov:
            raise ValueError("Nesterov must remain disabled to match Faster R-CNN")
        optimizer = self.training.optimizer
        if optimizer.warmup_epochs != 1.0:
            raise ValueError("the audited YOLO stability warmup must remain exactly one epoch")
        if optimizer.warmup_bias_learning_rate != 0.0:
            raise ValueError("YOLO warmup must ramp bias learning rates from zero")
        if optimizer.warmup_momentum != optimizer.momentum:
            raise ValueError("momentum must remain constant during YOLO LR warmup")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()

    def run_name(self, mode: Literal["smoke", "benchmark", "train"]) -> str:
        """Return the configured run name for one execution mode."""

        return {
            "smoke": self.outputs.smoke_run_name,
            "benchmark": self.outputs.benchmark_run_name,
            "train": self.outputs.train_run_name,
        }[mode]

    def run_dir(self, mode: Literal["smoke", "benchmark", "train"]) -> Path:
        """Return the configured Ultralytics run directory."""

        return self.resolve(self.outputs.logs_dir) / self.run_name(mode)


def load_yolo_config(path: str | Path) -> YoloConfig:
    """Load and strictly validate ``configs/yolo.yaml``."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YOLO config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return YoloConfig.model_validate(payload)


def yolo_config_sha256(config: YoloConfig) -> str:
    """Hash the exact YAML bytes used by an experiment."""

    return hashlib.sha256(config.source_path.read_bytes()).hexdigest()
