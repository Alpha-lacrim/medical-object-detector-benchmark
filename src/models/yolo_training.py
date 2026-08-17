"""Matched Ultralytics trainer controls and timing callbacks."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Literal

import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model

from src.models.faster_rcnn_training import EarlyStopper
from src.models.yolo_config import YoloConfig

YoloRunMode = Literal["smoke", "benchmark", "train"]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def _floating_tensors_to_float32(value: Any) -> Any:
    """Recursively cast floating prediction tensors while preserving gradients."""

    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _floating_tensors_to_float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_floating_tensors_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_floating_tensors_to_float32(item) for item in value)
    return value


class Float32DetectionLoss(v8DetectionLoss):
    """Run task assignment and detector losses in float32 under AMP training."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> Any:
        """Avoid float16 sigmoid underflow in task-aligned target assignment."""

        with torch.autocast(device_type=self.device.type, enabled=False):
            return super().__call__(_floating_tensors_to_float32(preds), batch)


class MixedPrecisionDetectionModel(DetectionModel):
    """YOLO detection model with float32 assignment/loss and AMP forward pass."""

    def init_criterion(self) -> Float32DetectionLoss:
        """Construct the numerically stable detection criterion."""

        return Float32DetectionLoss(self)


class ComparableEarlyStopping:
    """Adapt the shared min-epoch/min-delta stopper to Ultralytics' callback API."""

    def __init__(self, *, patience: int, min_delta: float, min_epochs: int) -> None:
        self._stopper = EarlyStopper(
            patience=patience,
            min_delta=min_delta,
            min_epochs=min_epochs,
        )
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.possible_stop = False

    @property
    def epochs_without_improvement(self) -> int:
        """Return the shared patience counter."""

        return self._stopper.epochs_without_improvement

    def __call__(self, epoch: int, fitness: float | None) -> bool:
        """Observe one 1-based epoch and return whether training should stop."""

        if fitness is None:
            return False
        metric = float(fitness)
        if not math.isfinite(metric):
            raise ValueError("YOLO early-stopping fitness must be finite")
        observation = self._stopper.observe(epoch=epoch, metric=metric)
        self.best_fitness = float(self._stopper.best_metric or 0.0)
        self.best_epoch = int(self._stopper.best_epoch or 0)
        self.possible_stop = (
            epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience - 1
        )
        return observation.should_stop

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-safe state for the run summary."""

        return self._stopper.state_dict()


class MatchedDetectionTrainer(DetectionTrainer):
    """Trainer with no stochastic transforms and stable mixed precision."""

    @staticmethod
    def _bfloat16_amp_check(_model: Any) -> bool:
        """Validate the configured AMP mode without the float16-only equivalence test."""

        return bool(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        )

    def _setup_train(self) -> None:
        """Run Ultralytics setup with a bfloat16-aware AMP capability check."""

        from ultralytics.engine import trainer as trainer_module

        original_check = trainer_module.check_amp
        trainer_module.check_amp = self._bfloat16_amp_check
        try:
            super()._setup_train()
        finally:
            trainer_module.check_amp = original_check

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ) -> YOLODataset:
        """Build a resize-only dataset for both training and validation.

        Ultralytics' normal training dataset unconditionally adds a small-probability
        Albumentations block even when every public augmentation hyperparameter is
        zero. Constructing the canonical dataset with ``augment=False`` is the
        supported resize-only path and enforces the predeclared parity decision.
        """

        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        is_validation = mode == "val"
        fraction = 1.0 if is_validation else self.args.fraction
        return YOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect or is_validation,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.5 if is_validation else 0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=fraction,
        )

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> MixedPrecisionDetectionModel:
        """Build the pinned YOLO model with a float32 criterion."""

        model = self.set_model_names_for_load(
            MixedPrecisionDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        return model


class YoloRunTracker:
    """Enforce runtime parity and record complete epoch timings."""

    def __init__(self, config: YoloConfig, mode: YoloRunMode) -> None:
        self.config = config
        self.mode = mode
        self.run_dir = config.run_dir(mode)
        self.timing_path = self.run_dir / config.outputs.epoch_timing_name
        self.epoch_started: float | None = None
        self.run_started: float | None = None
        self.records: list[dict[str, Any]] = []
        self.stopper: ComparableEarlyStopping | None = None
        self.total_wall_seconds: float | None = None
        self.epoch_batch = 0
        self.consecutive_zero_classification_losses = 0

    def on_train_start(self, trainer: Any) -> None:
        """Install the matched stopper and audit the realized optimizer contract."""

        settings = self.config.training.early_stopping
        self.stopper = ComparableEarlyStopping(
            patience=settings.patience,
            min_delta=settings.min_delta,
            min_epochs=settings.min_epochs,
        )
        trainer.stopper = self.stopper
        for group in trainer.optimizer.param_groups:
            group["nesterov"] = self.config.training.optimizer.nesterov
        if type(trainer.optimizer).__name__ != self.config.training.optimizer.name:
            raise RuntimeError("Ultralytics did not construct the configured SGD optimizer")
        if trainer.batch_size != self.config.runtime.batch_size or trainer.accumulate != 1:
            raise RuntimeError("realized YOLO physical/effective batch does not match config")
        dataset_augment = bool(getattr(trainer.train_loader.dataset, "augment", True))
        if dataset_augment:
            raise RuntimeError("training dataset retained stochastic augmentation")
        if not bool(trainer.amp):
            raise RuntimeError("Ultralytics AMP verification disabled mandatory AMP")
        if any(bool(group.get("nesterov")) for group in trainer.optimizer.param_groups):
            raise RuntimeError("failed to disable Nesterov in realized optimizer")
        self.run_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(trainer.device)
        _atomic_json(
            self.run_dir / "matched_training_contract.json",
            {
                "amp": bool(trainer.amp),
                "amp_dtype": self.config.runtime.amp_dtype,
                "amp_validation": ("cuda_bfloat16_support_plus_per_batch_numerical_guards"),
                "augmentation_policy": self.config.training.augmentation.policy,
                "dataset_augment": dataset_augment,
                "batch_norm_policy": self.config.model.batch_norm_policy,
                "loss_dtype": self.config.runtime.loss_dtype,
                "effective_batch_size": trainer.batch_size * trainer.accumulate,
                "minimum_epochs": settings.min_epochs,
                "optimizer": type(trainer.optimizer).__name__,
                "nesterov": any(
                    bool(group.get("nesterov")) for group in trainer.optimizer.param_groups
                ),
                "patience": settings.patience,
                "min_delta": settings.min_delta,
                "physical_batch_size": trainer.batch_size,
                "test_split_accessed": False,
            },
        )

    def on_train_epoch_start(self, _trainer: Any) -> None:
        """Start full train-plus-validation epoch timing."""

        self.epoch_started = time.perf_counter()
        self.epoch_batch = 0
        self.consecutive_zero_classification_losses = 0

    def on_train_batch_end(self, trainer: Any) -> None:
        """Fail immediately if a component loss becomes non-finite."""

        self.epoch_batch += 1
        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is None:
            return
        if not isinstance(unwrap_model(trainer.model).criterion, Float32DetectionLoss):
            raise RuntimeError("YOLO did not retain the configured float32 loss criterion")
        invalid = {
            str(name): float(value.detach().cpu())
            for name, value in loss_items.items()
            if not bool(torch.isfinite(value).all())
        }
        if invalid:
            _atomic_json(
                self.run_dir / "numerical_failure.json",
                {
                    "epoch": int(trainer.epoch) + 1,
                    "epoch_batch": self.epoch_batch,
                    "loss_items": invalid,
                    "amp_scale": float(trainer.scaler.get_scale()),
                    "test_split_accessed": False,
                },
            )
            raise FloatingPointError(
                f"non-finite YOLO loss at epoch {int(trainer.epoch) + 1}, "
                f"batch {self.epoch_batch}: {invalid}"
            )
        classification_loss = float(loss_items["cls_loss"].detach().cpu())
        self.consecutive_zero_classification_losses = (
            self.consecutive_zero_classification_losses + 1 if classification_loss == 0.0 else 0
        )
        if self.consecutive_zero_classification_losses >= 5:
            _atomic_json(
                self.run_dir / "numerical_failure.json",
                {
                    "amp_scale": float(trainer.scaler.get_scale()),
                    "epoch": int(trainer.epoch) + 1,
                    "epoch_batch": self.epoch_batch,
                    "failure": "zero_classification_loss_collapse",
                    "test_split_accessed": False,
                },
            )
            raise FloatingPointError(
                "YOLO classification loss collapsed to exactly zero for five batches"
            )

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Record timing after validation, metric logging, and checkpoint I/O."""

        if self.epoch_started is None:
            # Ultralytics invokes this callback once more during final_eval after
            # the last timed epoch. That pass must not create a duplicate record.
            return
        if self.stopper is None:
            raise RuntimeError("YOLO matched stopper was not installed")
        record = {
            "epoch": int(trainer.epoch) + 1,
            "epoch_seconds": time.perf_counter() - self.epoch_started,
            "fitness": float(trainer.fitness),
            "best_fitness": float(trainer.best_fitness),
            "epochs_without_improvement": self.stopper.epochs_without_improvement,
            "learning_rates": {key: float(value) for key, value in trainer.lr.items()},
            "amp_scale": float(trainer.scaler.get_scale()),
            "peak_gpu_memory_mib": (
                float(torch.cuda.max_memory_allocated(trainer.device)) / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        self.records.append(record)
        with self.timing_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.epoch_started = None

    def on_train_end(self, _trainer: Any) -> None:
        """Capture total wall time before final reporting."""

        if self.run_started is not None:
            self.total_wall_seconds = time.perf_counter() - self.run_started


def build_ultralytics_train_args(config: YoloConfig, mode: YoloRunMode) -> dict[str, Any]:
    """Translate the strict project config into explicit Ultralytics arguments."""

    augmentation = config.training.augmentation
    optimizer = config.training.optimizer
    dataset_yaml = config.data.smoke_dataset_yaml if mode == "smoke" else config.data.dataset_yaml
    epochs = {
        "smoke": config.smoke.epochs,
        "benchmark": config.benchmark.epochs,
        "train": config.training.max_epochs,
    }[mode]
    return {
        "data": config.resolve(dataset_yaml).as_posix(),
        "epochs": epochs,
        "patience": config.training.early_stopping.patience,
        "batch": config.runtime.batch_size,
        "nbs": config.runtime.nominal_batch_size,
        "imgsz": config.model.input_size,
        "device": config.runtime.device,
        "workers": config.runtime.workers,
        "cache": config.runtime.cache,
        "optimizer": optimizer.name,
        "lr0": optimizer.learning_rate,
        "lrf": optimizer.final_learning_rate_fraction,
        "momentum": optimizer.momentum,
        "weight_decay": optimizer.weight_decay,
        "warmup_epochs": optimizer.warmup_epochs,
        "warmup_momentum": optimizer.warmup_momentum,
        "warmup_bias_lr": optimizer.warmup_bias_learning_rate,
        "box": config.training.loss.box,
        "cls": config.training.loss.cls,
        "dfl": config.training.loss.dfl,
        "hsv_h": augmentation.hsv_h,
        "hsv_s": augmentation.hsv_s,
        "hsv_v": augmentation.hsv_v,
        "degrees": augmentation.degrees,
        "translate": augmentation.translate,
        "scale": augmentation.scale,
        "shear": augmentation.shear,
        "perspective": augmentation.perspective,
        "flipud": augmentation.flipud,
        "fliplr": augmentation.fliplr,
        "bgr": augmentation.bgr,
        "mosaic": augmentation.mosaic,
        "mixup": augmentation.mixup,
        "cutmix": augmentation.cutmix,
        "copy_paste": augmentation.copy_paste,
        "auto_augment": augmentation.auto_augment,
        "erasing": augmentation.erasing,
        "close_mosaic": augmentation.close_mosaic,
        "multi_scale": augmentation.multi_scale,
        "augment": False,
        "rect": False,
        "cos_lr": False,
        "amp": config.runtime.amp,
        "deterministic": config.runtime.deterministic,
        "seed": config.seed,
        "single_cls": False,
        "pretrained": config.model.pretrained,
        "resume": False,
        "val": True,
        "conf": config.evaluation.inference_minimum_score,
        "iou": config.evaluation.nms_iou_threshold,
        "max_det": config.evaluation.max_detections,
        "save": True,
        "save_period": -1,
        "plots": False,
        "project": config.resolve(config.outputs.logs_dir).as_posix(),
        "name": config.run_name(mode),
        "exist_ok": True,
        "verbose": True,
    }
