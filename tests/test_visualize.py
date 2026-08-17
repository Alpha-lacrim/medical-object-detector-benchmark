from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from src.data.visualize import run_eda


def _write_manifest(path: Path, split: str, stratum: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["exam_id", "nih_patient_id", "study_stratum", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "exam_id": f"{split}-exam",
                "nih_patient_id": f"{split}-patient",
                "study_stratum": stratum,
                "split": split,
            }
        )


def test_run_eda_creates_distribution_samples_and_summary(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    annotations_dir = tmp_path / "annotations"
    processed_dir = tmp_path / "processed"
    images_dir = processed_dir / "images"
    figures_dir = tmp_path / "figures"
    for directory in (splits_dir, annotations_dir, images_dir):
        directory.mkdir(parents=True)

    strata = ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"]
    for split, stratum in zip(("train", "val", "test"), strata, strict=True):
        _write_manifest(splits_dir / f"{split}.csv", split, stratum)

    Image.new("L", (32, 32), color=96).save(images_dir / "example.png")
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "example.png",
                "width": 32,
                "height": 32,
                "study_stratum": "Lung Opacity",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 5, 12, 10],
                "area": 120,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "Lung Opacity"}],
    }
    (annotations_dir / "train.json").write_text(json.dumps(coco), encoding="utf-8")

    config = {
        "dataset": {
            "paths": {
                "processed_dir": str(processed_dir),
                "processed_images_dir": str(images_dir),
                "annotations_dir": str(annotations_dir),
                "splits_dir": str(splits_dir),
                "figures_dir": str(figures_dir),
            },
            "classes": {"study_strata": strata},
            "split": {"ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
            "eda": {
                "sample_seed": 17,
                "sample_images": 1,
                "grid_columns": 1,
                "thumbnail_size": 64,
                "distribution_width": 600,
                "distribution_height": 400,
            },
        }
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_eda(config_path)

    assert summary["sample_count"] == 1
    assert summary["study_counts"]["train"]["Lung Opacity"] == 1
    assert summary["study_counts"]["val"]["No Lung Opacity / Not Normal"] == 1
    assert summary["study_counts"]["test"]["Normal"] == 1
    for name in (
        "rsna_class_distribution.png",
        "rsna_annotation_samples.png",
        "rsna_eda_summary.json",
    ):
        artifact = figures_dir / name
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
