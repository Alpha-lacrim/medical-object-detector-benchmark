"""Tests for the strict Batch 3 YOLO configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.models.yolo_config import YoloConfig, load_yolo_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "yolo.yaml"


def test_real_yolo_config_pins_matched_hardware_and_augmentation() -> None:
    config = load_yolo_config(CONFIG_PATH)

    assert config.model.weights == "yolo11s.pt"
    assert config.model.ultralytics_version == "8.4.110"
    assert config.model.input_size == 640
    assert config.model.batch_norm_policy == "train_statistics"
    assert config.runtime.amp is True
    assert config.runtime.amp_dtype == "bfloat16"
    assert config.runtime.batch_size == config.runtime.nominal_batch_size == 4
    assert config.training.optimizer.name == "SGD"
    assert config.training.optimizer.nesterov is False
    assert config.training.optimizer.warmup_epochs == 1
    assert config.training.optimizer.warmup_bias_learning_rate == 0
    assert config.training.optimizer.warmup_momentum == 0.9
    assert config.runtime.loss_dtype == "float32"
    assert config.training.early_stopping.min_epochs == 8
    assert config.training.early_stopping.patience == 5
    assert config.training.augmentation.policy == "match_faster_rcnn_none"
    assert config.training.augmentation.auto_augment is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("runtime", "amp"), False, "AMP is mandatory"),
        (("runtime", "nominal_batch_size"), 64, "effective batch parity"),
        (("training", "optimizer", "nesterov"), True, "Nesterov"),
        (("training", "augmentation", "mosaic"), 1.0, "requires every numeric"),
    ],
)
def test_yolo_config_rejects_protocol_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    payload["project_root"] = CONFIG_PATH.parents[1]
    payload["source_path"] = CONFIG_PATH

    with pytest.raises(ValidationError, match=message):
        YoloConfig.model_validate(payload)


def test_yolo_config_rejects_unknown_keys() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["surprise"] = True
    payload["project_root"] = CONFIG_PATH.parents[1]
    payload["source_path"] = CONFIG_PATH

    with pytest.raises(ValidationError, match="surprise"):
        YoloConfig.model_validate(payload)
