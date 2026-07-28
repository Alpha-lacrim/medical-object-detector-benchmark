"""Deterministic audit of a YOLO-format image-detection dataset."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_DEFAULT_SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    path: Path,
    message: str,
    line: int | None = None,
) -> None:
    issue = {
        "severity": severity,
        "code": code,
        "path": path.as_posix(),
        "message": message,
    }
    if line is not None:
        issue["line"] = line
    issues.append(issue)


def _index_files(
    directory: Path,
    suffixes: set[str],
    issues: list[dict[str, Any]],
    *,
    code: str,
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    if not directory.is_dir():
        _issue(issues, "error", code, directory, "required directory is missing")
        return indexed
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.suffix.lower() not in suffixes:
            continue
        key = path.relative_to(directory).with_suffix("").as_posix().casefold()
        if key in indexed:
            _issue(
                issues,
                "error",
                "duplicate_stem",
                path,
                f"collides with {indexed[key].as_posix()} after suffix removal",
            )
        else:
            indexed[key] = path
    return indexed


def _parse_label(
    path: Path,
    class_count: int,
    issues: list[dict[str, Any]],
) -> tuple[list[tuple[int, float, float, float, float]], str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        _issue(issues, "error", "unreadable_label", path, str(error))
        return [], "invalid"
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        return [], "empty"

    boxes: list[tuple[int, float, float, float, float]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    invalid = False
    for line_number, row in enumerate(rows, start=1):
        fields = row.split()
        if len(fields) != 5:
            _issue(
                issues,
                "error",
                "label_field_count",
                path,
                "detection rows require class, x_center, y_center, width, height",
                line_number,
            )
            invalid = True
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(item) for item in fields[1:])
        except ValueError:
            _issue(issues, "error", "label_parse", path, "row is not numeric", line_number)
            invalid = True
            continue
        coordinates = (x_center, y_center, width, height)
        if not 0 <= class_id < class_count:
            _issue(
                issues,
                "error",
                "unknown_class",
                path,
                f"class {class_id} is outside [0, {class_count - 1}]",
                line_number,
            )
            invalid = True
            continue
        if not all(math.isfinite(value) for value in coordinates):
            _issue(issues, "error", "nonfinite_box", path, "box must be finite", line_number)
            invalid = True
            continue
        left, right = x_center - width / 2, x_center + width / 2
        top, bottom = y_center - height / 2, y_center + height / 2
        if width <= 0 or height <= 0 or min(left, top) < 0 or max(right, bottom) > 1:
            _issue(
                issues,
                "error",
                "invalid_box",
                path,
                "normalized box must have positive area and lie inside [0, 1]",
                line_number,
            )
            invalid = True
            continue
        box = (class_id, x_center, y_center, width, height)
        if box in seen:
            _issue(issues, "error", "duplicate_box", path, "duplicate annotation", line_number)
            invalid = True
            continue
        seen.add(box)
        boxes.append(box)
    return boxes, "invalid" if invalid else "valid"


def audit_yolo_dataset(
    root: str | Path,
    *,
    class_names: tuple[str, ...],
    split_directories: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Audit images, labels, counts, hashes, and exact cross-split leakage."""

    root = Path(root)
    if not class_names or any(not name.strip() for name in class_names):
        raise ValueError("class_names must be non-empty")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")
    splits = split_directories or _DEFAULT_SPLITS
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("split_directories must define train, validation, and test")

    issues: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    split_counts: dict[str, Counter[str]] = {}
    hashes: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

    for split, split_directory in splits.items():
        counts: Counter[str] = Counter()
        split_counts[split] = counts
        split_root = root / split_directory
        images_root, labels_root = split_root / "images", split_root / "labels"
        images = _index_files(images_root, _IMAGE_SUFFIXES, issues, code="missing_images_directory")
        labels = _index_files(labels_root, {".txt"}, issues, code="missing_labels_directory")

        for key, label_path in labels.items():
            if key not in images:
                _issue(issues, "error", "orphan_label", label_path, "no matching image")

        for key, image_path in images.items():
            counts["images"] += 1
            relative_image = image_path.relative_to(root).as_posix()
            image_hash = _sha256(image_path)
            hashes[image_hash].append((split, relative_image))
            width = height = None
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
            except (OSError, SyntaxError, ValueError) as error:
                _issue(issues, "error", "unreadable_image", image_path, str(error))
                counts["unreadable_images"] += 1

            label_path = labels.get(key)
            label_hash = None
            if label_path is None:
                status, boxes = "missing", []
                counts["missing_labels"] += 1
                _issue(
                    issues,
                    "warning",
                    "missing_label",
                    image_path,
                    "may be a negative image; confirm rather than infer",
                )
            else:
                label_hash = _sha256(label_path)
                boxes, status = _parse_label(label_path, len(class_names), issues)
                counts[f"{status}_labels"] += 1
            counts["boxes"] += len(boxes)
            class_counts.update(box[0] for box in boxes)
            files.append(
                {
                    "split": split,
                    "image_path": relative_image,
                    "image_sha256": image_hash,
                    "width": width,
                    "height": height,
                    "label_path": label_path.relative_to(root).as_posix() if label_path else None,
                    "label_sha256": label_hash,
                    "label_status": status,
                    "box_count": len(boxes),
                }
            )

    duplicate_groups: list[list[str]] = []
    for _digest, members in sorted(hashes.items()):
        if len(members) < 2:
            continue
        paths = [path for _, path in members]
        duplicate_groups.append(paths)
        member_splits = {split for split, _ in members}
        severity = "error" if len(member_splits) > 1 else "warning"
        code = "cross_split_duplicate" if len(member_splits) > 1 else "duplicate_image"
        _issue(issues, severity, code, Path(paths[0]), f"same SHA-256: {', '.join(paths)}")

    manifest_payload = {"class_names": class_names, "files": files}
    manifest_bytes = json.dumps(
        manifest_payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "class_names": list(class_names),
        "passed": not any(issue["severity"] == "error" for issue in issues),
        "image_count": len(files),
        "box_count": sum(record["box_count"] for record in files),
        "class_box_counts": {
            str(class_id): class_counts[class_id] for class_id in range(len(class_names))
        },
        "split_counts": {
            split: dict(sorted(counts.items())) for split, counts in split_counts.items()
        },
        "exact_duplicate_groups": duplicate_groups,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": files,
        "issues": issues,
    }
