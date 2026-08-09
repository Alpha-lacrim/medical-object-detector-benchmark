import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.models.faster_rcnn_data import (
    CocoDetectionDataset,
    CocoValidationError,
    MissingImageFilesError,
    detection_collate_fn,
)


def _canonical_coco() -> dict[str, Any]:
    return {
        "images": [
            {"id": 11, "file_name": "positive.png", "width": 8, "height": 6},
            {"id": 12, "file_name": "negative.png", "width": 8, "height": 6},
        ],
        "annotations": [
            {
                "id": 21,
                "image_id": 11,
                "category_id": 1,
                "bbox": [1.0, 2.0, 3.0, 2.0],
                "area": 6.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "opacity"}],
    }


def _write_fixture(
    tmp_path: Path,
    *,
    coco: dict[str, Any] | None = None,
    files: tuple[str, ...] = ("positive.png", "negative.png"),
    file_size: tuple[int, int] = (8, 6),
) -> Path:
    images_dir = tmp_path / "processed" / "images"
    annotations_dir = tmp_path / "processed" / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    for index, name in enumerate(files):
        Image.new("L", file_size, color=64 + index * 64).save(images_dir / name)
    (annotations_dir / "instances_train.json").write_text(
        json.dumps(_canonical_coco() if coco is None else coco),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "paths": {
                "processed_images_dir": str(images_dir),
                "annotations_dir": str(annotations_dir),
            }
        }
    }
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_metadata_derives_categories_and_preserves_negative_images(tmp_path: Path) -> None:
    dataset = CocoDetectionDataset(_write_fixture(tmp_path), "train", mode="full")

    assert len(dataset) == 2
    assert dataset.category_names == {1: "opacity"}
    assert dataset.num_foreground_classes == 1
    assert dataset.num_classes == 2
    assert [len(record.annotations) for record in dataset.records] == [1, 0]
    assert dataset.records[0].annotations[0].bbox_xyxy == (1.0, 2.0, 4.0, 4.0)
    assert dataset.preflight.expected_images == 2
    assert dataset.preflight.available_images == 2
    assert dataset.preflight.complete


@pytest.mark.parametrize("mode", ["benchmark", "full"])
def test_complete_modes_list_every_missing_file(tmp_path: Path, mode: str) -> None:
    config_path = _write_fixture(tmp_path, files=())
    expected = tuple(
        (tmp_path / "processed" / "images" / name).resolve()
        for name in ("positive.png", "negative.png")
    )

    with pytest.raises(MissingImageFilesError) as raised:
        CocoDetectionDataset(config_path, "train", mode=mode)  # type: ignore[arg-type]

    assert raised.value.expected_count == 2
    assert raised.value.missing_files == expected
    assert all(str(path) in str(raised.value) for path in expected)


def test_partial_mode_reports_missing_without_dropping_records(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path, files=("positive.png",))

    dataset = CocoDetectionDataset(config_path, "train", mode="partial")

    assert len(dataset) == 2
    assert dataset.preflight.available_images == 1
    assert dataset.preflight.missing_files == (
        (tmp_path / "processed" / "images" / "negative.png").resolve(),
    )


@pytest.mark.parametrize(
    ("section", "duplicate", "message"),
    [
        ("categories", {"id": 1, "name": "other"}, "Duplicate COCO category id"),
        (
            "images",
            {"id": 11, "file_name": "third.png", "width": 8, "height": 6},
            "Duplicate COCO image id",
        ),
        (
            "annotations",
            {
                "id": 21,
                "image_id": 12,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "iscrowd": 0,
            },
            "Duplicate COCO annotation id",
        ),
    ],
)
def test_duplicate_ids_are_rejected(
    tmp_path: Path,
    section: str,
    duplicate: dict[str, Any],
    message: str,
) -> None:
    coco = _canonical_coco()
    coco[section].append(duplicate)
    config_path = _write_fixture(tmp_path, coco=coco)

    with pytest.raises(CocoValidationError, match=message):
        CocoDetectionDataset(config_path, "train")


def test_unknown_category_reference_is_rejected(tmp_path: Path) -> None:
    coco = _canonical_coco()
    coco["annotations"][0]["category_id"] = 99
    config_path = _write_fixture(tmp_path, coco=coco)

    with pytest.raises(CocoValidationError, match="unknown category id 99"):
        CocoDetectionDataset(config_path, "train")


@pytest.mark.parametrize(
    "escaped_name",
    [
        "../outside.png",
        "D:outside.png",
        "D:/outside.png",
        "\\outside.png",
        "nested/bad:name.png",
    ],
)
def test_coco_filename_cannot_escape_processed_images_dir(
    tmp_path: Path,
    escaped_name: str,
) -> None:
    coco = _canonical_coco()
    coco["images"][0]["file_name"] = escaped_name
    config_path = _write_fixture(tmp_path, coco=coco)

    with pytest.raises(CocoValidationError, match="safe path relative"):
        CocoDetectionDataset(config_path, "train")


@pytest.mark.parametrize(
    ("bbox", "area", "message"),
    [
        ([1, 1, 0, 2], 1, "positive size"),
        ([-1, 1, 2, 2], 4, "non-negative origin"),
        ([7, 1, 2, 2], 4, "exceeds image"),
        ([1, 1, 2, 2], 5, "does not match bbox area"),
    ],
)
def test_invalid_boxes_are_rejected(
    tmp_path: Path,
    bbox: list[int],
    area: int,
    message: str,
) -> None:
    coco = _canonical_coco()
    coco["annotations"][0]["bbox"] = bbox
    coco["annotations"][0]["area"] = area
    config_path = _write_fixture(tmp_path, coco=coco)

    with pytest.raises(CocoValidationError, match=message):
        CocoDetectionDataset(config_path, "train")


def test_invalid_declared_dimensions_are_rejected(tmp_path: Path) -> None:
    coco = _canonical_coco()
    coco["images"][0]["width"] = 0
    config_path = _write_fixture(tmp_path, coco=coco)

    with pytest.raises(CocoValidationError, match=r"images\[0\]\.width"):
        CocoDetectionDataset(config_path, "train")


def test_actual_image_dimensions_are_checked_during_preflight(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path, file_size=(7, 6))
    positive_path = (tmp_path / "processed" / "images" / "positive.png").resolve()

    with pytest.raises(CocoValidationError) as raised:
        CocoDetectionDataset(config_path, "train")

    assert str(positive_path) in str(raised.value)
    assert "expected 8x6, found 7x6" in str(raised.value)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="This check applies only to metadata-only environments",
)
def test_getitem_reports_clear_missing_torch_dependency(tmp_path: Path) -> None:
    dataset = CocoDetectionDataset(_write_fixture(tmp_path), "train")

    with pytest.raises(RuntimeError, match="requires PyTorch"):
        dataset[0]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is not installed in the lightweight audit environment",
)
def test_tensor_contract_and_empty_negative_target(tmp_path: Path) -> None:
    import torch

    dataset = CocoDetectionDataset(_write_fixture(tmp_path), "train")
    positive_image, positive_target = dataset[0]
    negative_image, negative_target = dataset[1]

    assert positive_image.dtype == torch.float32
    assert positive_image.shape == (3, 6, 8)
    assert float(positive_image.min()) >= 0.0
    assert float(positive_image.max()) <= 1.0
    assert torch.equal(
        positive_target["boxes"], torch.tensor([[1.0, 2.0, 4.0, 4.0]])
    )
    assert torch.equal(positive_target["labels"], torch.tensor([1]))
    assert torch.equal(positive_target["area"], torch.tensor([6.0]))
    assert torch.equal(positive_target["iscrowd"], torch.tensor([0]))
    assert torch.equal(positive_target["image_id"], torch.tensor([11]))
    assert negative_image.shape == (3, 6, 8)
    assert negative_target["boxes"].shape == (0, 4)
    assert negative_target["labels"].shape == (0,)
    images, targets = detection_collate_fn(
        [(positive_image, positive_target), (negative_image, negative_target)]
    )
    assert len(images) == len(targets) == 2


def test_sparse_category_ids_determine_detector_output_space(tmp_path: Path) -> None:
    coco = deepcopy(_canonical_coco())
    coco["categories"] = [{"id": 3, "name": "opacity"}]
    coco["annotations"][0]["category_id"] = 3
    dataset = CocoDetectionDataset(_write_fixture(tmp_path, coco=coco), "train")

    assert dataset.category_names == {3: "opacity"}
    assert dataset.num_foreground_classes == 1
    assert dataset.num_classes == 2
    assert dataset.category_id_to_label == {3: 1}
    assert dataset.label_to_category_id == {1: 3}
    assert dataset.records[0].annotations[0].category_id == 3
