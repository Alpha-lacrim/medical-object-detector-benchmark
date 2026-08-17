from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.models.faster_rcnn_config import config_fingerprint, load_faster_rcnn_config
from src.models.train_faster_rcnn import (
    _approved_benchmark,
    _completed_artifacts_intact,
    _dataset_summary,
    _load_datasets,
    build_parser,
    main,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "faster_rcnn.yaml"


def _coco_payload(*, image_id: int, file_name: str) -> dict[str, object]:
    """Return one valid negative-image COCO split for loader isolation tests."""

    return {
        "images": [
            {
                "id": image_id,
                "file_name": file_name,
                "width": 8,
                "height": 6,
            }
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "opacity"}],
    }


def test_train_and_validation_loading_never_reads_test_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the held-out test COCO file unopened throughout Batch 2 loading."""

    images_dir = tmp_path / "processed" / "images"
    annotations_dir = tmp_path / "processed" / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    Image.new("L", (8, 6), color=64).save(images_dir / "train.png")
    Image.new("L", (8, 6), color=128).save(images_dir / "val.png")

    train_annotations = annotations_dir / "instances_train.json"
    val_annotations = annotations_dir / "instances_val.json"
    test_annotations = annotations_dir / "instances_test.json"
    train_annotations.write_text(
        json.dumps(_coco_payload(image_id=1, file_name="train.png")),
        encoding="utf-8",
    )
    val_annotations.write_text(
        json.dumps(_coco_payload(image_id=2, file_name="val.png")),
        encoding="utf-8",
    )
    test_annotations.write_text("this sentinel must never be parsed", encoding="utf-8")

    dataset_config = tmp_path / "dataset.json"
    dataset_config.write_text(
        json.dumps(
            {
                "dataset": {
                    "paths": {
                        "processed_images_dir": "processed/images",
                        "annotations_dir": "processed/annotations",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    base = load_faster_rcnn_config(CONFIG_PATH)
    data = replace(
        base.data,
        dataset_config=Path("dataset.json"),
        images_dir=Path("processed/images"),
        train_annotations=Path("processed/annotations/instances_train.json"),
        val_annotations=Path("processed/annotations/instances_val.json"),
        test_annotations=Path("processed/annotations/instances_test.json"),
    )
    config = replace(base, data=data, project_root=tmp_path.resolve())

    original_read_text = Path.read_text
    read_paths: list[Path] = []

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        resolved = path.resolve()
        read_paths.append(resolved)
        if resolved == test_annotations.resolve():
            raise AssertionError("Batch 2 attempted to read the held-out test annotations")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    train_dataset, val_dataset = _load_datasets(config, mode="smoke")

    assert train_dataset.annotation_file == train_annotations
    assert val_dataset.annotation_file == val_annotations
    assert test_annotations.resolve() not in read_paths
    assert train_annotations.resolve() in read_paths
    assert val_annotations.resolve() in read_paths
    dataset_summary = _dataset_summary(train_dataset, val_dataset)
    assert json.loads(json.dumps(dataset_summary)) == dataset_summary
    assert dataset_summary["test_split_accessed"] is False


def test_timed_loading_aggregates_exact_train_and_validation_missing_paths(
    tmp_path: Path,
) -> None:
    annotations_dir = tmp_path / "processed" / "annotations"
    annotations_dir.mkdir(parents=True)
    for split, image_id in (("train", 1), ("val", 2)):
        (annotations_dir / f"instances_{split}.json").write_text(
            json.dumps(_coco_payload(image_id=image_id, file_name=f"{split}.png")),
            encoding="utf-8",
        )
    dataset_config = tmp_path / "dataset.json"
    dataset_config.write_text(
        json.dumps(
            {
                "dataset": {
                    "paths": {
                        "processed_images_dir": "processed/images",
                        "annotations_dir": "processed/annotations",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    base = load_faster_rcnn_config(CONFIG_PATH)
    config = replace(
        base,
        data=replace(
            base.data,
            dataset_config=Path("dataset.json"),
            images_dir=Path("processed/images"),
            train_annotations=Path("processed/annotations/instances_train.json"),
            val_annotations=Path("processed/annotations/instances_val.json"),
            test_annotations=Path("processed/annotations/instances_test.json"),
        ),
        project_root=tmp_path.resolve(),
    )

    with pytest.raises(FileNotFoundError) as caught:
        _load_datasets(config, mode="benchmark")

    message = str(caught.value)
    assert "train: 1 missing" in message
    assert "validation: 1 missing" in message
    assert str(tmp_path / "processed" / "images" / "train.png") in message
    assert str(tmp_path / "processed" / "images" / "val.png") in message


def test_approved_benchmark_binds_data_execution_and_implementation(
    tmp_path: Path,
) -> None:
    base = load_faster_rcnn_config(CONFIG_PATH)
    config = replace(base, project_root=tmp_path.resolve())
    path = config.run_artifact_path("benchmark", config.outputs.benchmark_estimate_path)
    path.parent.mkdir(parents=True)
    dataset_identity = {"categories": {"1": "opacity"}, "train": {"images": 1}}
    execution_identity = {"gpu_name": "test-gpu", "amp": True}
    implementation_identity = {"source_manifest_sha256": "a" * 64}
    path.write_text(
        json.dumps(
            {
                "config_sha256": config_fingerprint(config),
                "completed_epochs": config.benchmark.epochs,
                "dataset_identity": dataset_identity,
                "execution_identity": execution_identity,
                "implementation_identity": implementation_identity,
            }
        ),
        encoding="utf-8",
    )

    assert (
        _approved_benchmark(
            config,
            path,
            dataset_identity=dataset_identity,
            execution_identity=execution_identity,
            implementation_identity=implementation_identity,
        )["completed_epochs"]
        == config.benchmark.epochs
    )
    with pytest.raises(ValueError, match="dataset_identity"):
        _approved_benchmark(
            config,
            path,
            dataset_identity={"changed": True},
            execution_identity=execution_identity,
            implementation_identity=implementation_identity,
        )


def test_parser_exposes_recoverable_finalize_mode() -> None:
    assert build_parser().parse_args(["--mode", "finalize"]).mode == "finalize"


def test_train_mode_requires_signoff_artifact_before_data_or_cuda() -> None:
    with pytest.raises(ValueError, match="requires --approved-benchmark"):
        main(["--config", str(CONFIG_PATH), "--mode", "train"])


def test_complete_finalization_is_a_noop_only_while_artifacts_are_intact(
    tmp_path: Path,
) -> None:
    base = load_faster_rcnn_config(CONFIG_PATH)
    config = replace(base, project_root=tmp_path.resolve())
    validation = config.resolve(config.outputs.validation_table_path)
    compute = config.resolve(config.outputs.compute_table_path)
    curves = config.resolve(config.outputs.training_curves_path)
    checkpoint = config.resolve(config.outputs.best_checkpoint_path)
    for path, payload in (
        (validation, b"validation"),
        (compute, b"compute"),
        (curves, b"png"),
        (checkpoint, b"checkpoint"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    summary = {
        "artifacts": {
            "validation_table": validation.as_posix(),
            "compute_table": compute.as_posix(),
            "training_curves": curves.as_posix(),
            "model_artifact": {
                "path": checkpoint.as_posix(),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            },
            "file_integrity": {
                key: {
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for key, path in (
                    ("validation_table", validation),
                    ("compute_table", compute),
                    ("training_curves", curves),
                )
            },
        }
    }

    assert _completed_artifacts_intact(summary, config) is True
    compute.write_bytes(b"tampered")
    assert _completed_artifacts_intact(summary, config) is False
