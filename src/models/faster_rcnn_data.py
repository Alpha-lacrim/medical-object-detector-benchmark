"""Validated canonical COCO input pipeline for Faster R-CNN.

The module intentionally imports PyTorch only when an item is materialized.  This
keeps metadata validation and image-availability preflight usable in lightweight
preparation environments while still returning the tensor contract expected by
``torchvision`` detection models during training.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image

from src.data.prepare import load_dataset_config

if TYPE_CHECKING:
    import torch

DatasetMode = Literal["benchmark", "full", "partial"]


class CocoValidationError(ValueError):
    """Raised when a canonical COCO file violates the loader contract."""


class MissingImageFilesError(FileNotFoundError):
    """Raised when a complete run references image files that are not present."""

    def __init__(
        self,
        *,
        split: str,
        expected_count: int,
        missing_files: Sequence[Path],
    ) -> None:
        self.split = split
        self.expected_count = expected_count
        self.missing_files = tuple(missing_files)
        listing = "\n".join(str(path) for path in self.missing_files)
        super().__init__(
            f"Missing {len(self.missing_files)} of {expected_count} configured image "
            f"files for split {split!r}:\n{listing}"
        )


@dataclass(frozen=True, slots=True)
class CocoCategory:
    """One foreground category read directly from canonical COCO metadata."""

    id: int
    name: str


@dataclass(frozen=True, slots=True)
class CocoAnnotation:
    """One validated COCO bounding-box annotation."""

    id: int
    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    iscrowd: int

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Return the bounding box as absolute ``(x1, y1, x2, y2)`` coordinates."""

        x, y, width, height = self.bbox_xywh
        return (x, y, x + width, y + height)


@dataclass(frozen=True, slots=True)
class CocoImageRecord:
    """One image and all annotations attached to it, including empty negatives."""

    id: int
    file_name: str
    width: int
    height: int
    annotations: tuple[CocoAnnotation, ...]


@dataclass(frozen=True, slots=True)
class DatasetPreflight:
    """Image-availability result computed before a dataset can be iterated."""

    expected_images: int
    available_images: int
    missing_files: tuple[Path, ...]

    @property
    def complete(self) -> bool:
        """Whether every image referenced by the COCO file is available."""

        return not self.missing_files


@dataclass(frozen=True, slots=True)
class _ImageMetadata:
    """Validated image metadata before annotations are attached."""

    id: int
    file_name: str
    width: int
    height: int


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Return a mapping or raise a location-aware validation error."""

    if not isinstance(value, Mapping):
        raise CocoValidationError(f"{location} must be a mapping")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    """Return a JSON list or raise a location-aware validation error."""

    if not isinstance(value, list):
        raise CocoValidationError(f"{location} must be a list")
    return value


def _require_int(
    value: Any,
    location: str,
    *,
    minimum: int = 0,
) -> int:
    """Return an integer that satisfies the configured lower bound."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = "positive" if minimum == 1 else f">= {minimum}"
        raise CocoValidationError(f"{location} must be an integer that is {comparator}")
    return value


def _require_number(value: Any, location: str) -> float:
    """Return a finite JSON number without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CocoValidationError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CocoValidationError(f"{location} must be a finite number")
    return result


def _require_string(value: Any, location: str) -> str:
    """Return a non-empty stripped string."""

    if not isinstance(value, str) or not value.strip():
        raise CocoValidationError(f"{location} must be a non-empty string")
    return value.strip()


def _validate_relative_file_name(value: Any, location: str) -> str:
    """Validate that an image filename stays inside the configured image root."""

    file_name = _require_string(value, location)
    path = Path(file_name)
    if (
        path.is_absolute()
        or path.drive
        or path.root
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise CocoValidationError(
            f"{location} must be a safe path relative to processed_images_dir"
        )
    return str(path)


def _load_coco_payload(annotation_file: Path) -> Mapping[str, Any]:
    """Read one COCO JSON object with clear syntax and top-level errors."""

    if not annotation_file.is_file():
        raise FileNotFoundError(f"COCO annotation file does not exist: {annotation_file}")
    try:
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CocoValidationError(
            f"COCO annotation file is not valid JSON: {annotation_file}: {exc}"
        ) from exc
    return _require_mapping(payload, "COCO root")


def _parse_categories(payload: Mapping[str, Any]) -> tuple[CocoCategory, ...]:
    """Validate foreground categories while preserving their COCO IDs."""

    raw_categories = _require_list(payload.get("categories"), "COCO categories")
    if not raw_categories:
        raise CocoValidationError("COCO categories must contain a foreground category")
    categories: list[CocoCategory] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_categories):
        location = f"COCO categories[{index}]"
        item = _require_mapping(raw, location)
        category_id = _require_int(item.get("id"), f"{location}.id", minimum=1)
        name = _require_string(item.get("name"), f"{location}.name")
        if category_id in seen_ids:
            raise CocoValidationError(f"Duplicate COCO category id: {category_id}")
        if name in seen_names:
            raise CocoValidationError(f"Duplicate COCO category name: {name!r}")
        seen_ids.add(category_id)
        seen_names.add(name)
        categories.append(CocoCategory(id=category_id, name=name))
    return tuple(categories)


def _parse_images(payload: Mapping[str, Any]) -> tuple[_ImageMetadata, ...]:
    """Validate unique image IDs, filenames, and declared dimensions."""

    raw_images = _require_list(payload.get("images"), "COCO images")
    if not raw_images:
        raise CocoValidationError("COCO images cannot be empty")
    images: list[_ImageMetadata] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_images):
        location = f"COCO images[{index}]"
        item = _require_mapping(raw, location)
        image_id = _require_int(item.get("id"), f"{location}.id")
        file_name = _validate_relative_file_name(item.get("file_name"), f"{location}.file_name")
        width = _require_int(item.get("width"), f"{location}.width", minimum=1)
        height = _require_int(item.get("height"), f"{location}.height", minimum=1)
        normalized_name = file_name.casefold()
        if image_id in seen_ids:
            raise CocoValidationError(f"Duplicate COCO image id: {image_id}")
        if normalized_name in seen_names:
            raise CocoValidationError(f"Duplicate COCO image filename: {file_name!r}")
        seen_ids.add(image_id)
        seen_names.add(normalized_name)
        images.append(
            _ImageMetadata(
                id=image_id,
                file_name=file_name,
                width=width,
                height=height,
            )
        )
    return tuple(images)


def _parse_annotations(
    payload: Mapping[str, Any],
    images: Sequence[_ImageMetadata],
    categories: Sequence[CocoCategory],
) -> dict[int, tuple[CocoAnnotation, ...]]:
    """Validate annotations and group them by image, retaining empty images."""

    raw_annotations = _require_list(payload.get("annotations"), "COCO annotations")
    image_by_id = {image.id: image for image in images}
    category_ids = {category.id for category in categories}
    annotations_by_image: defaultdict[int, list[CocoAnnotation]] = defaultdict(list)
    seen_annotation_ids: set[int] = set()
    for index, raw in enumerate(raw_annotations):
        location = f"COCO annotations[{index}]"
        item = _require_mapping(raw, location)
        annotation_id = _require_int(item.get("id"), f"{location}.id")
        image_id = _require_int(item.get("image_id"), f"{location}.image_id")
        category_id = _require_int(item.get("category_id"), f"{location}.category_id", minimum=1)
        if annotation_id in seen_annotation_ids:
            raise CocoValidationError(f"Duplicate COCO annotation id: {annotation_id}")
        if image_id not in image_by_id:
            raise CocoValidationError(f"{location}.image_id references unknown image id {image_id}")
        if category_id not in category_ids:
            raise CocoValidationError(
                f"{location}.category_id references unknown category id {category_id}"
            )

        raw_bbox = _require_list(item.get("bbox"), f"{location}.bbox")
        if len(raw_bbox) != 4:
            raise CocoValidationError(f"{location}.bbox must contain exactly four numbers")
        x, y, width, height = (
            _require_number(value, f"{location}.bbox[{coordinate}]")
            for coordinate, value in enumerate(raw_bbox)
        )
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise CocoValidationError(
                f"{location}.bbox must have non-negative origin and positive size"
            )
        image = image_by_id[image_id]
        if x + width > image.width or y + height > image.height:
            raise CocoValidationError(
                f"{location}.bbox {list(map(float, raw_bbox))} exceeds image "
                f"{image_id} dimensions {image.width}x{image.height}"
            )

        area = _require_number(item.get("area"), f"{location}.area")
        if area <= 0:
            raise CocoValidationError(f"{location}.area must be positive")
        expected_area = width * height
        if not math.isclose(area, expected_area, rel_tol=1e-6, abs_tol=1e-6):
            raise CocoValidationError(
                f"{location}.area {area} does not match bbox area {expected_area}"
            )
        iscrowd = _require_int(item.get("iscrowd"), f"{location}.iscrowd")
        if iscrowd not in {0, 1}:
            raise CocoValidationError(f"{location}.iscrowd must be 0 or 1")
        if iscrowd:
            raise CocoValidationError(
                f"{location}.iscrowd=1 is unsupported by the unified evaluator"
            )

        seen_annotation_ids.add(annotation_id)
        annotations_by_image[image_id].append(
            CocoAnnotation(
                id=annotation_id,
                image_id=image_id,
                category_id=category_id,
                bbox_xywh=(x, y, width, height),
                area=area,
                iscrowd=iscrowd,
            )
        )
    return {image.id: tuple(annotations_by_image[image.id]) for image in images}


def _configured_path(value: Any, location: str, project_root: Path) -> Path:
    """Resolve one configured path against the explicit project root."""

    configured = Path(_require_string(value, location)).expanduser()
    path = configured if configured.is_absolute() else project_root / configured
    return path.resolve(strict=False)


def _validate_split(split: str) -> str:
    """Reject path traversal while permitting conventional split names."""

    invalid_character = any(
        not (character.isalnum() or character in {"_", "-"}) for character in split
    )
    if not split or invalid_character:
        raise ValueError("split must contain only letters, digits, underscores, or hyphens")
    return split


def _require_torch() -> Any:
    """Import PyTorch lazily with an actionable dependency error."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Materializing Faster R-CNN samples requires PyTorch; install the exact "
            "torch and torchvision versions pinned in requirements.txt"
        ) from exc
    return torch


class CocoDetectionDataset:
    """Config-driven canonical COCO dataset for ``torchvision`` Faster R-CNN.

    ``benchmark`` and ``full`` modes require every configured image during
    construction.  ``partial`` mode is intended only for metadata inspection and
    reports missing images without silently removing them from the dataset.
    """

    def __init__(
        self,
        config_path: str | Path,
        split: str,
        *,
        mode: DatasetMode = "full",
        project_root: str | Path | None = None,
    ) -> None:
        if mode not in {"benchmark", "full", "partial"}:
            raise ValueError("mode must be 'benchmark', 'full', or 'partial'")
        self.mode: DatasetMode = mode
        self.split = _validate_split(split)
        self.project_root = (
            Path.cwd() if project_root is None else Path(project_root).expanduser()
        ).resolve(strict=False)
        config_file = Path(config_path).expanduser()
        if not config_file.is_absolute():
            config_file = self.project_root / config_file
        self.config_path = config_file.resolve(strict=False)
        config = load_dataset_config(self.config_path)
        dataset = _require_mapping(config.get("dataset"), "dataset")
        paths = _require_mapping(dataset.get("paths"), "dataset.paths")
        self.images_dir = _configured_path(
            paths.get("processed_images_dir"),
            "dataset.paths.processed_images_dir",
            self.project_root,
        )
        annotations_dir = _configured_path(
            paths.get("annotations_dir"),
            "dataset.paths.annotations_dir",
            self.project_root,
        )
        self.annotation_file = annotations_dir / f"instances_{self.split}.json"

        payload = _load_coco_payload(self.annotation_file)
        self.categories = _parse_categories(payload)
        image_metadata = _parse_images(payload)
        annotations_by_image = _parse_annotations(payload, image_metadata, self.categories)
        self.records = tuple(
            CocoImageRecord(
                id=image.id,
                file_name=image.file_name,
                width=image.width,
                height=image.height,
                annotations=annotations_by_image[image.id],
            )
            for image in image_metadata
        )
        self.category_names = {category.id: category.name for category in self.categories}
        self.category_id_to_label = {
            category.id: label for label, category in enumerate(self.categories, start=1)
        }
        self.label_to_category_id = {
            label: category_id for category_id, label in self.category_id_to_label.items()
        }
        self.num_foreground_classes = len(self.categories)
        self.num_classes = self.num_foreground_classes + 1
        self.preflight = self._run_preflight(require_complete=mode != "partial")

    def __len__(self) -> int:
        """Return all configured images, including negative images."""

        return len(self.records)

    def image_path(self, index: int) -> Path:
        """Return the resolved image path for one dataset index."""

        images_root = self.images_dir.resolve(strict=False)
        candidate = (images_root / self.records[index].file_name).resolve(strict=False)
        try:
            candidate.relative_to(images_root)
        except ValueError as error:
            raise CocoValidationError(
                f"COCO image path escapes processed_images_dir: {candidate}"
            ) from error
        return candidate

    def _run_preflight(self, *, require_complete: bool) -> DatasetPreflight:
        """Check exact availability and dimensions before a full run starts."""

        paths = tuple(self.image_path(index) for index in range(len(self.records)))
        missing = tuple(path for path in paths if not path.is_file())
        if missing and require_complete:
            raise MissingImageFilesError(
                split=self.split,
                expected_count=len(paths),
                missing_files=missing,
            )

        dimension_errors: list[str] = []
        for record, path in zip(self.records, paths, strict=True):
            if not path.is_file():
                continue
            try:
                with Image.open(path) as image:
                    actual_size = image.size
                    image.load()
            except OSError as exc:
                dimension_errors.append(f"{path}: unreadable image ({exc})")
                continue
            expected_size = (record.width, record.height)
            if actual_size != expected_size:
                dimension_errors.append(
                    f"{path}: expected {record.width}x{record.height}, "
                    f"found {actual_size[0]}x{actual_size[1]}"
                )
        if dimension_errors:
            raise CocoValidationError("Image preflight failed:\n" + "\n".join(dimension_errors))
        return DatasetPreflight(
            expected_images=len(paths),
            available_images=len(paths) - len(missing),
            missing_files=missing,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load one RGB float tensor and its torchvision detection target."""

        torch = _require_torch()
        record = self.records[index]
        path = self.image_path(index)
        if not path.is_file():
            raise FileNotFoundError(f"Configured image file is missing: {path}")
        with Image.open(path) as source:
            if source.size != (record.width, record.height):
                raise CocoValidationError(
                    f"Image dimensions changed after preflight for {path}: expected "
                    f"{record.width}x{record.height}, found "
                    f"{source.size[0]}x{source.size[1]}"
                )
            rgb = source.convert("RGB")
            pixels = np.array(rgb, dtype=np.float32, copy=True) / np.float32(255.0)
        chw = np.ascontiguousarray(pixels.transpose(2, 0, 1))
        image_tensor = torch.from_numpy(chw)

        if record.annotations:
            boxes = torch.tensor(
                [annotation.bbox_xyxy for annotation in record.annotations],
                dtype=torch.float32,
            )
            labels = torch.tensor(
                [
                    self.category_id_to_label[annotation.category_id]
                    for annotation in record.annotations
                ],
                dtype=torch.int64,
            )
            areas = torch.tensor(
                [annotation.area for annotation in record.annotations],
                dtype=torch.float32,
            )
            crowds = torch.tensor(
                [annotation.iscrowd for annotation in record.annotations],
                dtype=torch.int64,
            )
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            areas = torch.empty((0,), dtype=torch.float32)
            crowds = torch.empty((0,), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": crowds,
            "image_id": torch.tensor([record.id], dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(
    batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate variable-length detection targets without stacking their boxes."""

    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
