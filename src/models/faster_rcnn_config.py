"""Strict configuration loading for the Faster R-CNN baseline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

RunMode = Literal["smoke", "benchmark", "train", "finalize"]


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """Return ``value`` as a string-keyed mapping or raise a useful error."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys")
    return value


def _strict_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    """Reject missing and unknown keys so configuration drift is visible."""

    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{context} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {', '.join(unknown)}")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _probability(value: Any, context: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not lower_ok or result > 1:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{context} must be in {interval}")
    return result


def _positive_float(value: Any, context: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _portable_path(value: Any, context: str) -> Path:
    """Validate a project-relative POSIX path and return it as a ``Path``."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty path string")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"{context} must be a portable project-relative POSIX path")
    return Path(*path.parts)


@dataclass(frozen=True)
class DataPaths:
    """Canonical images and split annotations used by Faster R-CNN."""

    dataset_config: Path
    images_dir: Path
    train_annotations: Path
    val_annotations: Path
    test_annotations: Path
    grayscale_to_rgb: str
    train_augmentation: str


@dataclass(frozen=True)
class ModelSettings:
    """Torchvision model construction settings."""

    architecture: str
    weights: str
    trainable_backbone_layers: int
    min_size: int
    max_size: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    box_detections_per_image: int
    box_score_threshold: float
    box_nms_threshold: float
    batch_norm_policy: str


@dataclass(frozen=True)
class RuntimeSettings:
    """Device, precision, and DataLoader settings."""

    device: str
    amp: bool
    amp_dtype: str
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    validation_persistent_workers: bool
    deterministic: bool
    deterministic_warn_only: bool
    allow_tf32: bool

    @property
    def effective_batch_size(self) -> int:
        """Return the optimizer-level batch size."""

        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class OptimizerSettings:
    """SGD hyperparameters."""

    name: str
    learning_rate: float
    momentum: float
    weight_decay: float
    gradient_clip_norm: float
    nesterov: bool


@dataclass(frozen=True)
class SchedulerSettings:
    """Validation-metric scheduler settings."""

    name: str
    mode: str
    factor: float
    patience: int
    threshold: float
    min_learning_rate: float


@dataclass(frozen=True)
class EarlyStoppingSettings:
    """Patience-based validation-mAP early stopping settings."""

    metric: str
    mode: str
    patience: int
    min_delta: float
    min_epochs: int


@dataclass(frozen=True)
class TrainingSettings:
    """Full-run optimization and stopping settings."""

    max_epochs: int
    optimizer: OptimizerSettings
    scheduler: SchedulerSettings
    early_stopping: EarlyStoppingSettings
    log_every_batches: int


@dataclass(frozen=True)
class EvaluationSettings:
    """Unified validation operating point and COCO settings."""

    coco_minimum_score: float
    score_threshold: float
    match_iou_threshold: float
    max_detections: int


@dataclass(frozen=True)
class BenchmarkSettings:
    """Pre-full-run benchmark and extrapolation settings."""

    epochs: int
    require_complete_dataset: bool


@dataclass(frozen=True)
class SmokeSettings:
    """Bounded pipeline-smoke settings."""

    max_train_batches: int
    max_val_batches: int
    use_pretrained_weights: bool


@dataclass(frozen=True)
class ProfilingSettings:
    """Final inference-speed and FLOP profiling settings."""

    batch_size: int
    num_workers: int
    persistent_workers: bool
    warmup_batches: int
    timed_batches: int
    profile_flops: bool


@dataclass(frozen=True)
class OutputSettings:
    """Project-relative output roots and deterministic run names."""

    logs_dir: Path
    checkpoints_dir: Path
    tables_dir: Path
    figures_dir: Path
    smoke_run_name: str
    benchmark_run_name: str
    train_run_name: str
    resolved_config_path: Path
    epoch_csv_path: Path
    epoch_jsonl_path: Path
    summary_path: Path
    benchmark_estimate_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    benchmark_timing_checkpoints_dir: Path
    validation_table_path: Path
    compute_table_path: Path
    training_curves_path: Path

    def run_name(self, mode: RunMode) -> str:
        """Return the configured run name for ``mode``."""

        return {
            "smoke": self.smoke_run_name,
            "benchmark": self.benchmark_run_name,
            "train": self.train_run_name,
            "finalize": self.train_run_name,
        }[mode]


@dataclass(frozen=True)
class FasterRCNNConfig:
    """Complete immutable Batch 2 experiment configuration."""

    schema_version: int
    experiment_id: str
    seed: int
    data: DataPaths
    model: ModelSettings
    runtime: RuntimeSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    benchmark: BenchmarkSettings
    smoke: SmokeSettings
    profiling: ProfilingSettings
    outputs: OutputSettings
    project_root: Path
    source_path: Path

    def resolve(self, path: Path) -> Path:
        """Resolve a validated project-relative path."""

        return self.project_root / path

    def run_dir(self, mode: RunMode) -> Path:
        """Return the log directory for one configured run mode."""

        return self.resolve(self.outputs.logs_dir) / self.outputs.run_name(mode)

    def run_artifact_path(self, mode: RunMode, relative_path: Path) -> Path:
        """Resolve one configured run-scoped artifact below the mode's log directory."""

        return self.run_dir(mode) / relative_path


def _parse_data(payload: Any) -> DataPaths:
    value = _mapping(payload, "data")
    required = {
        "dataset_config",
        "images_dir",
        "train_annotations",
        "val_annotations",
        "test_annotations",
        "grayscale_to_rgb",
        "train_augmentation",
    }
    _strict_keys(value, required=required, context="data")
    path_keys = required - {"grayscale_to_rgb", "train_augmentation"}
    if value["grayscale_to_rgb"] != "replicate":
        raise ValueError("data.grayscale_to_rgb must be replicate")
    if value["train_augmentation"] != "none":
        raise ValueError("data.train_augmentation must be none for the controlled baseline")
    return DataPaths(
        **{key: _portable_path(value[key], f"data.{key}") for key in sorted(path_keys)},
        grayscale_to_rgb=value["grayscale_to_rgb"],
        train_augmentation=value["train_augmentation"],
    )


def _channel_values(value: Any, context: str) -> tuple[float, ...]:
    """Validate a three-channel mean or standard-deviation sequence."""

    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must contain exactly three values")
    result = tuple(_probability(item, f"{context}[{index}]") for index, item in enumerate(value))
    if context.endswith("std") and any(item == 0 for item in result):
        raise ValueError(f"{context} values must be positive")
    return result


def _parse_model(payload: Any) -> ModelSettings:
    value = _mapping(payload, "model")
    required = {
        "architecture",
        "weights",
        "trainable_backbone_layers",
        "min_size",
        "max_size",
        "image_mean",
        "image_std",
        "box_detections_per_image",
        "box_score_threshold",
        "box_nms_threshold",
        "batch_norm_policy",
    }
    _strict_keys(value, required=required, context="model")
    if value["architecture"] != "fasterrcnn_resnet50_fpn_v2":
        raise ValueError("model.architecture must be fasterrcnn_resnet50_fpn_v2")
    if value["weights"] != "DEFAULT":
        raise ValueError("model.weights must be DEFAULT for COCO transfer learning")
    if value["batch_norm_policy"] not in {"freeze_statistics", "train"}:
        raise ValueError("model.batch_norm_policy must be freeze_statistics or train")
    min_size = _positive_int(value["min_size"], "model.min_size")
    max_size = _positive_int(value["max_size"], "model.max_size")
    if min_size > max_size:
        raise ValueError("model.min_size cannot exceed model.max_size")
    backbone_layers = _non_negative_int(
        value["trainable_backbone_layers"], "model.trainable_backbone_layers"
    )
    if backbone_layers > 5:
        raise ValueError("model.trainable_backbone_layers cannot exceed 5")
    return ModelSettings(
        architecture=value["architecture"],
        weights=value["weights"],
        trainable_backbone_layers=backbone_layers,
        min_size=min_size,
        max_size=max_size,
        image_mean=_channel_values(value["image_mean"], "model.image_mean"),
        image_std=_channel_values(value["image_std"], "model.image_std"),
        box_detections_per_image=_positive_int(
            value["box_detections_per_image"], "model.box_detections_per_image"
        ),
        box_score_threshold=_probability(value["box_score_threshold"], "model.box_score_threshold"),
        box_nms_threshold=_probability(value["box_nms_threshold"], "model.box_nms_threshold"),
        batch_norm_policy=value["batch_norm_policy"],
    )


def _parse_runtime(payload: Any) -> RuntimeSettings:
    value = _mapping(payload, "runtime")
    required = {
        "device",
        "amp",
        "amp_dtype",
        "batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "validation_persistent_workers",
        "deterministic",
        "deterministic_warn_only",
        "allow_tf32",
    }
    _strict_keys(value, required=required, context="runtime")
    if value["device"] != "cuda":
        raise ValueError("runtime.device must be cuda for the scoped Batch 2 run")
    if value["amp"] is not True or value["amp_dtype"] != "float16":
        raise ValueError("runtime must enable float16 AMP")
    batch_size = _positive_int(value["batch_size"], "runtime.batch_size")
    if not 2 <= batch_size <= 4:
        raise ValueError("runtime.batch_size must be between 2 and 4")
    workers = _non_negative_int(value["num_workers"], "runtime.num_workers")
    for key in (
        "pin_memory",
        "persistent_workers",
        "validation_persistent_workers",
        "deterministic",
        "deterministic_warn_only",
        "allow_tf32",
    ):
        if not isinstance(value[key], bool):
            raise ValueError(f"runtime.{key} must be boolean")
    if (value["persistent_workers"] or value["validation_persistent_workers"]) and workers == 0:
        raise ValueError("persistent workers require runtime.num_workers > 0")
    return RuntimeSettings(
        device=value["device"],
        amp=value["amp"],
        amp_dtype=value["amp_dtype"],
        batch_size=batch_size,
        gradient_accumulation_steps=_positive_int(
            value["gradient_accumulation_steps"],
            "runtime.gradient_accumulation_steps",
        ),
        num_workers=workers,
        pin_memory=value["pin_memory"],
        persistent_workers=value["persistent_workers"],
        validation_persistent_workers=value["validation_persistent_workers"],
        deterministic=value["deterministic"],
        deterministic_warn_only=value["deterministic_warn_only"],
        allow_tf32=value["allow_tf32"],
    )


def _parse_optimizer(payload: Any) -> OptimizerSettings:
    value = _mapping(payload, "training.optimizer")
    required = {
        "name",
        "learning_rate",
        "momentum",
        "weight_decay",
        "gradient_clip_norm",
        "nesterov",
    }
    _strict_keys(value, required=required, context="training.optimizer")
    if value["name"] != "sgd":
        raise ValueError("training.optimizer.name must be sgd")
    if not isinstance(value["nesterov"], bool):
        raise ValueError("training.optimizer.nesterov must be boolean")
    return OptimizerSettings(
        name=value["name"],
        learning_rate=_positive_float(value["learning_rate"], "training.optimizer.learning_rate"),
        momentum=_probability(value["momentum"], "training.optimizer.momentum"),
        weight_decay=_positive_float(
            value["weight_decay"], "training.optimizer.weight_decay", allow_zero=True
        ),
        gradient_clip_norm=_positive_float(
            value["gradient_clip_norm"], "training.optimizer.gradient_clip_norm"
        ),
        nesterov=value["nesterov"],
    )


def _parse_scheduler(payload: Any) -> SchedulerSettings:
    value = _mapping(payload, "training.scheduler")
    required = {"name", "mode", "factor", "patience", "threshold", "min_learning_rate"}
    _strict_keys(value, required=required, context="training.scheduler")
    if value["name"] != "reduce_on_plateau" or value["mode"] != "max":
        raise ValueError("training.scheduler must be reduce_on_plateau in max mode")
    factor = _probability(value["factor"], "training.scheduler.factor", allow_zero=False)
    if factor == 1:
        raise ValueError("training.scheduler.factor must be smaller than 1")
    return SchedulerSettings(
        name=value["name"],
        mode=value["mode"],
        factor=factor,
        patience=_non_negative_int(value["patience"], "training.scheduler.patience"),
        threshold=_positive_float(
            value["threshold"], "training.scheduler.threshold", allow_zero=True
        ),
        min_learning_rate=_positive_float(
            value["min_learning_rate"], "training.scheduler.min_learning_rate"
        ),
    )


def _parse_early_stopping(payload: Any) -> EarlyStoppingSettings:
    value = _mapping(payload, "training.early_stopping")
    required = {"metric", "mode", "patience", "min_delta", "min_epochs"}
    _strict_keys(value, required=required, context="training.early_stopping")
    if value["metric"] != "val_map_50_95" or value["mode"] != "max":
        raise ValueError("early stopping must maximize val_map_50_95")
    return EarlyStoppingSettings(
        metric=value["metric"],
        mode=value["mode"],
        patience=_positive_int(value["patience"], "training.early_stopping.patience"),
        min_delta=_positive_float(
            value["min_delta"], "training.early_stopping.min_delta", allow_zero=True
        ),
        min_epochs=_positive_int(value["min_epochs"], "training.early_stopping.min_epochs"),
    )


def _parse_training(payload: Any) -> TrainingSettings:
    value = _mapping(payload, "training")
    required = {
        "max_epochs",
        "optimizer",
        "scheduler",
        "early_stopping",
        "log_every_batches",
    }
    _strict_keys(value, required=required, context="training")
    early_stopping = _parse_early_stopping(value["early_stopping"])
    max_epochs = _positive_int(value["max_epochs"], "training.max_epochs")
    if early_stopping.min_epochs > max_epochs:
        raise ValueError("early-stopping min_epochs cannot exceed max_epochs")
    return TrainingSettings(
        max_epochs=max_epochs,
        optimizer=_parse_optimizer(value["optimizer"]),
        scheduler=_parse_scheduler(value["scheduler"]),
        early_stopping=early_stopping,
        log_every_batches=_positive_int(value["log_every_batches"], "training.log_every_batches"),
    )


def _parse_evaluation(payload: Any) -> EvaluationSettings:
    value = _mapping(payload, "evaluation")
    required = {
        "coco_minimum_score",
        "score_threshold",
        "match_iou_threshold",
        "max_detections",
    }
    _strict_keys(value, required=required, context="evaluation")
    max_detections = _positive_int(value["max_detections"], "evaluation.max_detections")
    if max_detections < 10:
        raise ValueError("evaluation.max_detections must be at least 10 for COCO evaluation")
    return EvaluationSettings(
        coco_minimum_score=_probability(
            value["coco_minimum_score"], "evaluation.coco_minimum_score"
        ),
        score_threshold=_probability(value["score_threshold"], "evaluation.score_threshold"),
        match_iou_threshold=_probability(
            value["match_iou_threshold"],
            "evaluation.match_iou_threshold",
            allow_zero=False,
        ),
        max_detections=max_detections,
    )


def _parse_benchmark(payload: Any) -> BenchmarkSettings:
    value = _mapping(payload, "benchmark")
    required = {"epochs", "require_complete_dataset"}
    _strict_keys(value, required=required, context="benchmark")
    epochs = _positive_int(value["epochs"], "benchmark.epochs")
    if not 2 <= epochs <= 3:
        raise ValueError("benchmark.epochs must be 2 or 3")
    if value["require_complete_dataset"] is not True:
        raise ValueError("benchmark.require_complete_dataset must be true")
    return BenchmarkSettings(epochs=epochs, require_complete_dataset=True)


def _parse_smoke(payload: Any) -> SmokeSettings:
    value = _mapping(payload, "smoke")
    required = {"max_train_batches", "max_val_batches", "use_pretrained_weights"}
    _strict_keys(value, required=required, context="smoke")
    if not isinstance(value["use_pretrained_weights"], bool):
        raise ValueError("smoke.use_pretrained_weights must be boolean")
    return SmokeSettings(
        max_train_batches=_positive_int(value["max_train_batches"], "smoke.max_train_batches"),
        max_val_batches=_positive_int(value["max_val_batches"], "smoke.max_val_batches"),
        use_pretrained_weights=value["use_pretrained_weights"],
    )


def _parse_profiling(payload: Any) -> ProfilingSettings:
    value = _mapping(payload, "profiling")
    required = {
        "batch_size",
        "num_workers",
        "persistent_workers",
        "warmup_batches",
        "timed_batches",
        "profile_flops",
    }
    _strict_keys(value, required=required, context="profiling")
    if not isinstance(value["profile_flops"], bool):
        raise ValueError("profiling.profile_flops must be boolean")
    if not isinstance(value["persistent_workers"], bool):
        raise ValueError("profiling.persistent_workers must be boolean")
    workers = _non_negative_int(value["num_workers"], "profiling.num_workers")
    if value["persistent_workers"] and workers == 0:
        raise ValueError("profiling persistent workers require profiling.num_workers > 0")
    return ProfilingSettings(
        batch_size=_positive_int(value["batch_size"], "profiling.batch_size"),
        num_workers=workers,
        persistent_workers=value["persistent_workers"],
        warmup_batches=_non_negative_int(value["warmup_batches"], "profiling.warmup_batches"),
        timed_batches=_positive_int(value["timed_batches"], "profiling.timed_batches"),
        profile_flops=value["profile_flops"],
    )


def _run_name(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in value)
    ):
        raise ValueError(f"{context} must use lowercase letters, digits, hyphens, or underscores")
    return value


def _parse_outputs(payload: Any) -> OutputSettings:
    value = _mapping(payload, "outputs")
    path_keys = {
        "logs_dir",
        "checkpoints_dir",
        "tables_dir",
        "figures_dir",
        "resolved_config_path",
        "epoch_csv_path",
        "epoch_jsonl_path",
        "summary_path",
        "benchmark_estimate_path",
        "best_checkpoint_path",
        "last_checkpoint_path",
        "benchmark_timing_checkpoints_dir",
        "validation_table_path",
        "compute_table_path",
        "training_curves_path",
    }
    run_keys = {"smoke_run_name", "benchmark_run_name", "train_run_name"}
    _strict_keys(value, required=path_keys | run_keys, context="outputs")
    return OutputSettings(
        **{key: _portable_path(value[key], f"outputs.{key}") for key in sorted(path_keys)},
        **{key: _run_name(value[key], f"outputs.{key}") for key in sorted(run_keys)},
    )


def load_faster_rcnn_config(path: str | Path) -> FasterRCNNConfig:
    """Load and strictly validate a Batch 2 YAML configuration."""

    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    value = _mapping(payload, "configuration")
    required = {
        "schema_version",
        "experiment_id",
        "seed",
        "data",
        "model",
        "runtime",
        "training",
        "evaluation",
        "benchmark",
        "smoke",
        "profiling",
        "outputs",
    }
    _strict_keys(value, required=required, context="configuration")
    if value["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    experiment_id = _run_name(value["experiment_id"], "experiment_id")
    seed = _non_negative_int(value["seed"], "seed")
    config = FasterRCNNConfig(
        schema_version=1,
        experiment_id=experiment_id,
        seed=seed,
        data=_parse_data(value["data"]),
        model=_parse_model(value["model"]),
        runtime=_parse_runtime(value["runtime"]),
        training=_parse_training(value["training"]),
        evaluation=_parse_evaluation(value["evaluation"]),
        benchmark=_parse_benchmark(value["benchmark"]),
        smoke=_parse_smoke(value["smoke"]),
        profiling=_parse_profiling(value["profiling"]),
        outputs=_parse_outputs(value["outputs"]),
        project_root=source_path.parent.parent,
        source_path=source_path,
    )
    if config.evaluation.coco_minimum_score != config.model.box_score_threshold:
        raise ValueError("evaluation.coco_minimum_score must equal model.box_score_threshold")
    return config


def config_fingerprint(config: FasterRCNNConfig) -> str:
    """Return a SHA-256 fingerprint of the immutable source YAML bytes."""

    return hashlib.sha256(config.source_path.read_bytes()).hexdigest()


def serializable_config(config: FasterRCNNConfig) -> dict[str, Any]:
    """Return the parsed configuration using JSON-safe path strings."""

    payload = asdict(config)
    return json.loads(json.dumps(payload, default=str))
