"""Canonical COCO to Ultralytics YOLO dataset adapter for Batch 3."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.models.yolo_config import YoloConfig


@dataclass(frozen=True)
class CocoImageRecord:
    """One canonical image with validated pixel-space annotations."""

    image_id: str
    file_name: str
    path: Path
    width: int
    height: int
    boxes_xywh: tuple[tuple[float, float, float, float], ...]
    category_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically inside an existing or new parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""

    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_coco(path: Path, images_dir: Path) -> tuple[list[CocoImageRecord], dict[int, str]]:
    """Load and validate the canonical subset needed by Ultralytics."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO file must contain an object: {path}")
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if (
        not isinstance(images, list)
        or not isinstance(annotations, list)
        or not isinstance(categories, list)
    ):
        raise ValueError(f"COCO file lacks images/annotations/categories lists: {path}")

    category_names: dict[int, str] = {}
    for category in categories:
        category_id = int(category["id"])
        name = str(category["name"])
        if category_id in category_names or not name:
            raise ValueError("COCO categories must have unique IDs and non-empty names")
        category_names[category_id] = name
    if not category_names:
        raise ValueError("COCO dataset must contain at least one category")

    by_image: dict[int, list[dict[str, Any]]] = {}
    annotation_ids: set[int] = set()
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if annotation_id in annotation_ids:
            raise ValueError("duplicate COCO annotation ID")
        annotation_ids.add(annotation_id)
        if category_id not in category_names:
            raise ValueError("annotation references an unknown COCO category")
        if int(annotation.get("iscrowd", 0)) != 0:
            raise ValueError("crowd annotations are unsupported")
        by_image.setdefault(image_id, []).append(annotation)

    records: list[CocoImageRecord] = []
    seen_image_ids: set[int] = set()
    seen_names: set[str] = set()
    for image in sorted(images, key=lambda item: str(item["file_name"])):
        numeric_id = int(image["id"])
        file_name = str(image["file_name"])
        width, height = int(image["width"]), int(image["height"])
        if numeric_id in seen_image_ids or file_name in seen_names:
            raise ValueError("COCO image IDs and file names must be unique")
        if Path(file_name).name != file_name:
            raise ValueError("COCO file_name must be a basename")
        if width <= 0 or height <= 0:
            raise ValueError("COCO image dimensions must be positive")
        source = (images_dir / file_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"missing prepared image: {source}")
        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        for annotation in sorted(by_image.get(numeric_id, []), key=lambda item: int(item["id"])):
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            if (
                box_width <= 0
                or box_height <= 0
                or x < 0
                or y < 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ValueError(f"invalid or off-image box in {path}: {annotation['id']}")
            boxes.append((x, y, box_width, box_height))
            labels.append(int(annotation["category_id"]))
        records.append(
            CocoImageRecord(
                image_id=str(numeric_id),
                file_name=file_name,
                path=source,
                width=width,
                height=height,
                boxes_xywh=tuple(boxes),
                category_ids=tuple(labels),
            )
        )
        seen_image_ids.add(numeric_id)
        seen_names.add(file_name)
    if set(by_image) - seen_image_ids:
        raise ValueError("COCO annotation references an unknown image")
    return records, category_names


def load_coco_records(
    config: YoloConfig, split: str
) -> tuple[list[CocoImageRecord], dict[int, str]]:
    """Load train or validation records without allowing test access."""

    paths = {
        "train": config.data.train_annotations,
        "validation": config.data.val_annotations,
    }
    if split not in paths:
        raise ValueError("Batch 3 may load only train or validation annotations")
    return _load_coco(
        config.resolve(paths[split]),
        config.resolve(config.data.images_dir),
    )


def _label_text(record: CocoImageRecord, category_to_yolo: dict[int, int]) -> str:
    """Convert one image's boxes to normalized YOLO center format."""

    lines: list[str] = []
    for (x, y, width, height), category_id in zip(
        record.boxes_xywh, record.category_ids, strict=True
    ):
        x_center = (x + width / 2.0) / record.width
        y_center = (y + height / 2.0) / record.height
        normalized_width = width / record.width
        normalized_height = height / record.height
        values = (x_center, y_center, normalized_width, normalized_height)
        if any(value <= 0 or value > 1 for value in values):
            raise ValueError(f"invalid normalized YOLO box for {record.file_name}")
        lines.append(
            f"{category_to_yolo[category_id]} "
            + " ".join(f"{value:.10f}" for value in values)
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _ensure_hardlink(source: Path, destination: Path) -> None:
    """Create or verify one same-volume hardlink."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise FileExistsError(
                f"existing YOLO image is not the expected hardlink: {destination}"
            )
        return
    os.link(source, destination)


def _balanced_smoke(records: list[CocoImageRecord], count: int) -> list[CocoImageRecord]:
    """Choose a deterministic positive/negative smoke subset."""

    positives = [record for record in records if record.boxes_xywh]
    negatives = [record for record in records if not record.boxes_xywh]
    positive_count = min(len(positives), count // 2)
    selected = positives[:positive_count] + negatives[: count - positive_count]
    if len(selected) != count:
        raise ValueError("not enough positive/negative records for configured smoke subset")
    return sorted(selected, key=lambda record: record.file_name)


def _image_manifest_sha256(records: list[CocoImageRecord]) -> str:
    """Hash selected image names and exact file sizes without decoding pixels."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.file_name):
        digest.update(record.file_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_yolo_dataset(config: YoloConfig) -> dict[str, Any]:
    """Materialize hardlinked train/validation images and exact YOLO labels."""

    root = config.resolve(config.data.yolo_root)
    split_records: dict[str, list[CocoImageRecord]] = {}
    common_categories: dict[int, str] | None = None
    category_to_yolo: dict[int, int] = {}
    summaries: dict[str, Any] = {}

    for canonical_split, yolo_split in (("train", "train"), ("validation", "val")):
        records, categories = load_coco_records(config, canonical_split)
        if common_categories is None:
            common_categories = categories
            category_to_yolo = {
                category_id: index for index, category_id in enumerate(sorted(categories))
            }
        elif categories != common_categories:
            raise ValueError("train and validation COCO categories differ")
        split_records[canonical_split] = records
        label_digest = hashlib.sha256()
        annotation_count = 0
        for record in records:
            image_destination = root / "images" / yolo_split / record.file_name
            label_destination = root / "labels" / yolo_split / f"{Path(record.file_name).stem}.txt"
            _ensure_hardlink(record.path, image_destination)
            label_text = _label_text(record, category_to_yolo)
            _atomic_text(label_destination, label_text)
            annotation_count += len(record.boxes_xywh)
            label_digest.update(label_destination.name.encode("utf-8"))
            label_digest.update(b"\0")
            label_digest.update(label_text.encode("utf-8"))
        summaries[canonical_split] = {
            "images": len(records),
            "annotations": annotation_count,
            "negative_images": sum(not record.boxes_xywh for record in records),
            "annotation_sha256": sha256_file(
                config.resolve(
                    config.data.train_annotations
                    if canonical_split == "train"
                    else config.data.val_annotations
                )
            ),
            "image_manifest_sha256": _image_manifest_sha256(records),
            "label_manifest_sha256": label_digest.hexdigest(),
        }

    assert common_categories is not None
    names = {category_to_yolo[key]: value for key, value in sorted(common_categories.items())}
    dataset_payload = {
        "path": root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    _atomic_text(
        config.resolve(config.data.dataset_yaml),
        yaml.safe_dump(dataset_payload, sort_keys=False, allow_unicode=True),
    )

    smoke_lists: dict[str, str] = {}
    for canonical_split, yolo_split, count in (
        ("train", "train", config.smoke.train_images),
        ("validation", "val", config.smoke.val_images),
    ):
        selected = _balanced_smoke(split_records[canonical_split], count)
        list_path = root / f"smoke_{yolo_split}.txt"
        _atomic_text(
            list_path,
            "".join(
                f"{(root / 'images' / yolo_split / record.file_name).as_posix()}\n"
                for record in selected
            ),
        )
        smoke_lists[yolo_split] = list_path.name
    smoke_payload = {
        "path": root.as_posix(),
        "train": smoke_lists["train"],
        "val": smoke_lists["val"],
        "names": names,
    }
    _atomic_text(
        config.resolve(config.data.smoke_dataset_yaml),
        yaml.safe_dump(smoke_payload, sort_keys=False, allow_unicode=True),
    )

    summary = {
        "categories": {str(key): value for key, value in sorted(common_categories.items())},
        "category_id_to_yolo_label": {
            str(key): value for key, value in sorted(category_to_yolo.items())
        },
        "train": summaries["train"],
        "validation": summaries["validation"],
        "dataset_yaml": config.resolve(config.data.dataset_yaml).as_posix(),
        "smoke_dataset_yaml": config.resolve(config.data.smoke_dataset_yaml).as_posix(),
        "link_mode": config.data.link_mode,
        "test_split_accessed": False,
    }
    _atomic_json(root / "manifest.json", summary)
    return summary
