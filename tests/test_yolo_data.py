"""Tests for canonical COCO to YOLO materialization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from PIL import Image

from src.models.yolo_config import load_yolo_config
from src.models.yolo_data import prepare_yolo_dataset

ROOT = Path(__file__).parents[1]


def _coco(images: list[dict[str, object]], annotations: list[dict[str, object]]) -> dict:
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 5, "name": "Finding"}],
    }


def _temporary_config(tmp_path: Path):
    images_dir = tmp_path / "pixels"
    images_dir.mkdir()
    names = ("train-positive.png", "train-negative.png", "val-positive.png", "val-negative.png")
    for name in names:
        Image.new("L", (100, 80), color=32).save(images_dir / name)
    train = _coco(
        [
            {"id": 1, "file_name": names[0], "width": 100, "height": 80},
            {"id": 2, "file_name": names[1], "width": 100, "height": 80},
        ],
        [{"id": 10, "image_id": 1, "category_id": 5, "bbox": [10, 20, 30, 20], "iscrowd": 0}],
    )
    val = _coco(
        [
            {"id": 3, "file_name": names[2], "width": 100, "height": 80},
            {"id": 4, "file_name": names[3], "width": 100, "height": 80},
        ],
        [{"id": 11, "image_id": 3, "category_id": 5, "bbox": [20, 10, 20, 40], "iscrowd": 0}],
    )
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (annotations / "val.json").write_text(json.dumps(val), encoding="utf-8")

    payload = yaml.safe_load((ROOT / "configs" / "yolo.yaml").read_text(encoding="utf-8"))
    payload["data"].update(
        {
            "images_dir": images_dir.as_posix(),
            "train_annotations": (annotations / "train.json").as_posix(),
            "val_annotations": (annotations / "val.json").as_posix(),
            "test_annotations": (annotations / "must-not-open.json").as_posix(),
            "yolo_root": (tmp_path / "yolo").as_posix(),
            "dataset_yaml": (tmp_path / "yolo" / "dataset.yaml").as_posix(),
            "smoke_dataset_yaml": (tmp_path / "yolo" / "dataset_smoke.yaml").as_posix(),
        }
    )
    payload["smoke"].update({"train_images": 2, "val_images": 2})
    config_path = tmp_path / "configs" / "yolo.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_yolo_config(config_path), images_dir


def test_prepare_yolo_dataset_preserves_negatives_and_box_geometry(tmp_path: Path) -> None:
    config, images_dir = _temporary_config(tmp_path)

    summary = prepare_yolo_dataset(config)
    repeated = prepare_yolo_dataset(config)

    assert repeated == summary
    assert summary["train"]["images"] == 2
    assert summary["train"]["annotations"] == 1
    assert summary["train"]["negative_images"] == 1
    assert summary["validation"]["images"] == 2
    assert summary["test_split_accessed"] is False
    root = config.resolve(config.data.yolo_root)
    assert os.path.samefile(
        images_dir / "train-positive.png", root / "images" / "train" / "train-positive.png"
    )
    assert (root / "labels" / "train" / "train-negative.txt").read_text() == ""
    values = (root / "labels" / "train" / "train-positive.txt").read_text().split()
    assert values[0] == "0"
    assert [float(value) for value in values[1:]] == [0.25, 0.375, 0.3, 0.25]
    dataset = yaml.safe_load(config.resolve(config.data.dataset_yaml).read_text(encoding="utf-8"))
    assert dataset["names"] == {0: "Finding"}
    assert not config.resolve(config.data.test_annotations).exists()
