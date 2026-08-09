"""Tests for matched YOLO training controls."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.yolo_config import load_yolo_config
from src.models.yolo_training import (
    ComparableEarlyStopping,
    YoloRunTracker,
    _floating_tensors_to_float32,
    build_ultralytics_train_args,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "yolo.yaml"


def test_comparable_early_stopping_enforces_minimum_and_delta() -> None:
    stopper = ComparableEarlyStopping(patience=3, min_delta=0.01, min_epochs=5)

    assert stopper(1, 0.10) is False
    assert stopper(2, 0.105) is False
    assert stopper(3, 0.106) is False
    assert stopper(4, 0.107) is False
    assert stopper(5, 0.108) is True
    assert stopper.best_epoch == 5
    assert stopper.epochs_without_improvement == 4


def test_tracker_ignores_ultralytics_final_eval_callback() -> None:
    tracker = YoloRunTracker(load_yolo_config(CONFIG_PATH), "smoke")

    tracker.on_fit_epoch_end(object())

    assert tracker.records == []


def test_loss_prediction_cast_is_float32_and_preserves_gradient() -> None:
    source = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)

    converted = _floating_tensors_to_float32({"scores": source})
    converted["scores"].sum().backward()

    assert converted["scores"].dtype == torch.float32
    assert source.grad is not None


def test_ultralytics_arguments_explicitly_disable_augmentation_and_match_batch() -> None:
    config = load_yolo_config(CONFIG_PATH)
    arguments = build_ultralytics_train_args(config, "train")

    assert arguments["batch"] == arguments["nbs"] == 4
    assert arguments["amp"] is True
    assert arguments["optimizer"] == "SGD"
    assert arguments["lr0"] == 0.001
    assert arguments["warmup_epochs"] == 1
    assert arguments["warmup_bias_lr"] == 0
    assert arguments["warmup_momentum"] == 0.9
    assert arguments["cache"] is False
    assert arguments["auto_augment"] is None
    for key in (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
        "close_mosaic",
        "multi_scale",
    ):
        assert arguments[key] == 0
