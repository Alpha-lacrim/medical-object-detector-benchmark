"""Strict, immutable experiment configuration and run gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DetectorName = Literal["faster_rcnn", "yolo"]
Operation = Literal["smoke", "train", "test"]


class StrictModel(BaseModel):
    """Base model that rejects silent configuration drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _portable_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or ":" in path.parts[0]
    ):
        raise ValueError("path must be a portable, relative POSIX path")
    return value


class Manifests(StrictModel):
    train: str
    validation: str
    test: str

    _paths = field_validator("train", "validation", "test")(_portable_relative_path)


class DataConfig(StrictModel):
    kind: Literal["synthetic", "kaggle", "manual"]
    source_id: str = Field(min_length=1)
    root: str
    manifests: Manifests
    class_names: tuple[str, ...] = Field(min_length=1)

    _root_path = field_validator("root")(_portable_relative_path)

    @field_validator("class_names")
    @classmethod
    def unique_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(name.strip() for name in value)
        if any(not name for name in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("class names must be non-empty and unique")
        return normalized


class ModelSpec(StrictModel):
    implementation: str = Field(min_length=1)
    weights: str = Field(min_length=1)
    head_mode: str = Field(min_length=1)


class RuntimeConfig(StrictModel):
    device: Literal["cpu", "cuda"]
    precision: Literal["fp32", "amp"]
    deterministic: bool
    allow_tf32: bool


class EvaluationConfig(StrictModel):
    fixed_score_threshold: float = Field(ge=0, le=1)
    match_iou_threshold: float = Field(gt=0, le=1)
    max_detections: int = Field(gt=0)


class OptimizerConfig(StrictModel):
    name: Literal["adamw"]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class SchedulerConfig(StrictModel):
    name: Literal["cosine"]
    warmup_epochs: int = Field(ge=0)


class TrainingConfig(StrictModel):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    accumulation_steps: int = Field(gt=0)
    input_size: int = Field(ge=32)
    augmentation_profile: str = Field(min_length=1)
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps

    @model_validator(mode="after")
    def warmup_fits_budget(self) -> TrainingConfig:
        if self.scheduler.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        return self


class TrackAConfig(StrictModel):
    """One shared object makes controlled-setting drift unrepresentable."""

    training: TrainingConfig


class TrackBConfig(StrictModel):
    trials_per_model: int = Field(gt=0)
    accelerator_hours_per_model: float = Field(gt=0)
    selection_metric: Literal["val_ap50_95"]
    training: dict[DetectorName, TrainingConfig]

    @model_validator(mode="after")
    def both_models_present(self) -> TrackBConfig:
        if set(self.training) != {"faster_rcnn", "yolo"}:
            raise ValueError("Track B must configure exactly Faster R-CNN and YOLO")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    status: Literal["smoke", "draft", "frozen"]
    test_access_allowed: bool
    decision_ids: tuple[str, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    data: DataConfig
    models: dict[DetectorName, ModelSpec]
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    track_a: TrackAConfig
    track_b: TrackBConfig

    @model_validator(mode="after")
    def enforce_protocol(self) -> ExperimentConfig:
        if set(self.models) != {"faster_rcnn", "yolo"}:
            raise ValueError("models must contain exactly Faster R-CNN and YOLO")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if len(set(self.decision_ids)) != len(self.decision_ids):
            raise ValueError("decision_ids must be unique")
        if self.test_access_allowed and self.status != "frozen":
            raise ValueError("test access requires a frozen configuration")
        if self.status == "frozen" and self.data.kind == "synthetic":
            raise ValueError("a synthetic dataset cannot be a frozen final experiment")
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load YAML with safe parsing and validate the complete protocol."""

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("experiment YAML must contain a mapping")
    return ExperimentConfig.model_validate(payload)


def canonical_json(config: ExperimentConfig) -> bytes:
    """Return deterministic bytes for provenance and cache keys."""

    payload = config.model_dump(mode="json")
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def config_fingerprint(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_json(config)).hexdigest()


def assert_run_allowed(config: ExperimentConfig, operation: Operation) -> None:
    """Prevent accidental training/test access from an inappropriate config."""

    if operation == "smoke":
        return
    if config.status == "smoke":
        raise RuntimeError("synthetic smoke configuration cannot train or test final models")
    if operation == "test" and not (config.status == "frozen" and config.test_access_allowed):
        raise RuntimeError("test evaluation requires explicit access in a frozen configuration")
