"""Tests for YOLO curve reporting."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.models.yolo_reporting import plot_yolo_training_curves


def test_plot_yolo_training_curves_writes_valid_png(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    fieldnames = (
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "lr/pg0",
    )
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                dict.fromkeys(fieldnames, 0.1) | {"epoch": 1},
                dict.fromkeys(fieldnames, 0.2) | {"epoch": 2},
            ]
        )
    destination = tmp_path / "curve.png"

    result = plot_yolo_training_curves(source, destination, best_epoch=2)

    assert result == destination
    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.width > 1000
        assert image.height > 700
