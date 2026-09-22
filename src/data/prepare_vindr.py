"""Data-only, fail-closed adapter for the frozen official VinDr test release.

No models, predictions or performance measures are loaded by this module.
Per-image artifacts are written only to the frozen, Git-ignored private tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import yaml
from PIL import Image, features
from pydicom.pixels import get_decoder

from src.data.prepare import scale_radiograph_to_uint8
from src.meddet_benchmark.evaluation import ImageTarget


class PreflightError(ValueError):
    """A data/protocol prerequisite needs review before proceeding."""


def sha256(path: Path) -> str:
    """Hash exact bytes with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON, replacing a complete file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_freeze(config_path: Path, config: Mapping[str, Any]) -> dict[str, str]:
    """Verify all frozen definitions, including the supplied config itself."""
    freeze_path = Path(config["freeze_manifest"])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["protocol_id"] != config["protocol_id"]:
        raise PreflightError("Protocol identity differs from freeze")
    verified = {}
    for entry in freeze["artifacts"]:
        path = Path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise PreflightError(f"Frozen input changed or missing: {path}")
        verified[path.as_posix()] = entry["sha256"]
    if config_path.resolve() not in {Path(p).resolve() for p in verified}:
        raise PreflightError("Supplied config is not bound by the freeze")
    return verified


def resolve_root(config: Mapping[str, Any]) -> Path:
    """Require the recorded authorization and explicit environment root."""
    dataset = config["dataset"]
    if not dataset["access"]["approved_access_dua_and_official_download_confirmed"]:
        raise PreflightError("Approved access/DUA/official-download attestation missing")
    value = os.environ.get(dataset["root_environment_variable"])
    if not value:
        raise PreflightError(f"Set {dataset['root_environment_variable']} to the authorized root")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise PreflightError("Authorized dataset root is unavailable")
    return root


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV without normalizing identifiers or ontology labels."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise PreflightError("Malformed CSV row")
        return list(reader.fieldnames or []), rows


def verify_inventory(root: Path, config: Mapping[str, Any]) -> list[tuple[str, Path, str]]:
    """Verify release hashes, exact test inventory, schemas and label IDs."""
    dataset = config["dataset"]
    paths = dataset["paths"]
    for key, expected in dataset["source_sha256"].items():
        path = root / paths[key]
        if not path.is_file() or sha256(path) != expected:
            raise PreflightError(f"Release file missing/wrong version/checksum: {key}")
    checksums = {}
    for line in (root / paths["checksums"]).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name in checksums:
            raise PreflightError("Duplicate checksum-manifest path")
        checksums[name] = digest
    prefix = paths["images"] + "/"
    expected = {name: digest for name, digest in checksums.items() if name.startswith(prefix)}
    actual = {
        p.relative_to(root).as_posix(): p
        for p in (root / paths["images"]).rglob("*")
        if p.is_file() and p.suffix == dataset["extension"]
    }
    if len(expected) != dataset["expected_images"] or set(actual) != set(expected):
        raise PreflightError("Official test inventory incomplete or contains unexpected DICOMs")
    ids = {p.stem for p in actual.values()}
    if len(ids) != len(actual):
        raise PreflightError("Duplicate image IDs")
    columns, rows = read_csv(root / paths["image_labels"])
    id_key = dataset["image_id_column"]
    if id_key not in columns or config["ontology"]["source_exact_class_name"] not in columns:
        raise PreflightError("Image-label schema differs")
    label_ids = [r[id_key] for r in rows]
    if len(label_ids) != len(ids) or set(label_ids) != ids:
        raise PreflightError("Image-label IDs do not match official test images uniquely")
    columns, rows = read_csv(root / paths["boxes"])
    if columns != dataset["annotation_columns"]:
        raise PreflightError("Box annotation schema differs from frozen protocol")
    if any(row[id_key] not in ids for row in rows):
        raise PreflightError("Unknown annotation image ID")
    return [(actual[name].stem, actual[name], expected[name]) for name in sorted(actual)]


def validate_header(dataset: pydicom.Dataset) -> tuple[int, int]:
    """Reject unsupported image semantics; never infer orientation or rescale."""
    height, width = int(dataset.Rows), int(dataset.Columns)
    if min(height, width) <= 1:
        raise PreflightError("Invalid radiograph dimensions")
    if (
        int(getattr(dataset, "NumberOfFrames", 1)) != 1
        or int(dataset.SamplesPerPixel) != 1
        or dataset.PhotometricInterpretation not in {"MONOCHROME1", "MONOCHROME2"}
    ):
        raise PreflightError("Expected one frame of monochrome pixels")
    if (
        int(dataset.PixelRepresentation) not in {0, 1}
        or not 0 < int(dataset.BitsStored) <= int(dataset.BitsAllocated)
        or int(dataset.HighBit) != int(dataset.BitsStored) - 1
    ):
        raise PreflightError("Unsupported pixel representation")
    if (
        float(getattr(dataset, "RescaleSlope", 1)) != 1
        or float(getattr(dataset, "RescaleIntercept", 0)) != 0
        or getattr(dataset, "ModalityLUTSequence", None)
        or getattr(dataset, "PresentationLUTShape", "IDENTITY") != "IDENTITY"
    ):
        raise PreflightError("Unexpected modality/presentation semantics need pre-inference review")
    decoder = get_decoder(dataset.file_meta.TransferSyntaxUID)
    if not decoder.is_available:
        raise PreflightError("Lossless pixel decoder unavailable")
    return height, width


def decode_image(path: Path, config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode stored pixels and apply precisely the shared frozen transform."""
    dataset = pydicom.dcmread(path)
    size = validate_header(dataset)
    pixels = dataset.pixel_array
    if pixels.shape != size:
        raise PreflightError("Decoded array dimensions differ from DICOM Rows/Columns")
    output = scale_radiograph_to_uint8(
        pixels,
        photometric_interpretation=str(dataset.PhotometricInterpretation),
        invert_monochrome1=config["preprocessing"]["invert_monochrome1"],
    )
    return output, {
        "constant": bool(pixels.min() == pixels.max()),
        "nonfinite_pixels": int((~np.isfinite(pixels)).sum()),
    }


def build_targets(
    rows: Sequence[Mapping[str, str]],
    sizes: Mapping[str, tuple[int, int]],
    config: Mapping[str, Any],
) -> list[ImageTarget]:
    """Exact ontology filtering; retain every zero-target image as a negative."""
    ontology, dataset = config["ontology"], config["dataset"]
    category = next(
        k
        for k, v in ontology["canonical_classes"].items()
        if v == ontology["source_exact_class_name"]
    )
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    seen = set()
    for row in rows:
        image_id = row[dataset["image_id_column"]]
        if image_id not in sizes:
            raise PreflightError("Unknown annotation image ID")
        if row[dataset["class_name_column"]] != ontology["source_exact_class_name"]:
            continue
        box = [float(row[key]) for key in dataset["annotation_columns"][2:]]
        key = (image_id, *box)
        if key in seen:
            raise PreflightError("Duplicate strict-target box; stop for annotation review")
        seen.add(key)
        grouped[image_id].append(box)
    return [
        ImageTarget(
            image_id, sizes[image_id], grouped[image_id], [category] * len(grouped[image_id])
        )
        for image_id in sorted(sizes)
    ]


def targets_to_coco(targets: Sequence[ImageTarget], config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert source xyxy to canonical native-size COCO xywh without a +1."""
    images, annotations = [], []
    for index, target in enumerate(targets, 1):
        height, width = target.image_size
        images.append(
            {"id": index, "file_name": target.image_id + ".png", "width": width, "height": height}
        )
        for box, label in zip(target.boxes_xyxy, target.labels, strict=True):
            x, y, x2, y2 = map(float, box)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": int(label),
                    "bbox": [x, y, x2 - x, y2 - y],
                    "area": (x2 - x) * (y2 - y),
                    "iscrowd": 0,
                }
            )
    return {
        "info": {"protocol_id": config["protocol_id"]},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": k, "name": v} for k, v in config["ontology"]["canonical_classes"].items()
        ],
    }


def geometry_check(target: ImageTarget, input_size: int, tolerance: float) -> float:
    """Check native resize/restore utilities on source boxes and corner probes.

    Torchvision uses separate realized width/height ratios after floor rounding.
    Ultralytics uses a common gain and rounded centered letterbox padding.
    This is coordinate validation only, with no detector construction/forward.
    """
    import torch
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.instance import Instances
    from ultralytics.utils.ops import scale_boxes

    height, width = target.image_size
    boxes = np.vstack(
        [
            target.boxes_xyxy,
            [0, 0, width, height],
            [width / 7, height / 5, width * 0.8, height * 0.9],
        ]
    )
    original = torch.tensor(boxes, dtype=torch.float64)
    transform = GeneralizedRCNNTransform(input_size, input_size, [], []).eval()
    resized_image, mapped_target = transform.resize(
        torch.zeros((1, height, width)), {"boxes": original.clone()}
    )
    resized = tuple(resized_image.shape[-2:])
    restored = transform.postprocess([mapped_target], [resized], [(height, width)])[0]["boxes"]
    error = float(torch.max(torch.abs(restored - original)))
    for auto in (False, True):
        letterbox = LetterBox(new_shape=(input_size, input_size), auto=auto)
        labels = {
            "img": np.broadcast_to(np.uint8(0), (height, width, 1)),
            "instances": Instances(
                boxes.copy(), segments=np.zeros((0, 0, 2)), bbox_format="xyxy", normalized=False
            ),
        }
        params = letterbox.get_params(labels)
        mapped = letterbox.apply_instances(labels, params)["instances"].bboxes
        shape = (
            params["new_unpad"][1] + params["top"] + params["bottom"],
            params["new_unpad"][0] + params["left"] + params["right"],
        )
        # Exercise the same automatic gain/padding inference as native prediction restoration.
        restored = scale_boxes(shape, torch.tensor(mapped), (height, width))
        error = max(error, float(torch.max(torch.abs(restored - original))))
    if error > tolerance:
        raise PreflightError("Native coordinate restoration exceeds tolerance")
    return error


def ensure_private(path: Path, private_root: Path) -> None:
    """Require restricted artifacts to stay inside the ignored private root."""
    if not path.resolve().is_relative_to(private_root.resolve()):
        raise PreflightError("Restricted output escapes private root")
    result = subprocess.run(["git", "check-ignore", "-q", "--", str(path)], check=False)
    if result.returncode != 0:
        raise PreflightError("Restricted output is not Git-ignored")


def run(config_path: Path, adapter_path: Path, mode: str) -> dict[str, Any]:
    """Audit the full release; optionally emit native PNGs and canonical targets."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapter = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    import torch

    torch.set_num_threads(config["runtime"]["torch_threads"])
    frozen = verify_freeze(config_path, config)
    root = resolve_root(config)
    outputs = config["outputs"]
    private_root = Path(outputs["private_root"])
    images_root = private_root / adapter["images_directory_name"]
    manifest_path, annotations_path = (
        Path(outputs["local_manifest"]),
        Path(outputs["local_annotations"]),
    )
    private_audit = private_root / adapter["private_audit_name"]
    loader_config = private_root / adapter["loader_config_name"]
    summary_path = Path(outputs["aggregate_results_root"]) / adapter["aggregate_summary_name"]
    for path in (images_root, manifest_path, annotations_path, private_audit, loader_config):
        ensure_private(path, private_root)
    prediction_root = Path(outputs["local_predictions"])
    if prediction_root.exists() and any(p.is_file() for p in prediction_root.rglob("*")):
        raise PreflightError(
            "External predictions already exist; Batch-48 pre-performance gate fails"
        )
    if summary_path.parent.exists() and any(
        p.is_file() and p != summary_path for p in summary_path.parent.rglob("*")
    ):
        raise PreflightError("Unexpected external results exist; review before adapter work")
    write_json(
        summary_path,
        {
            "status": "in_progress",
            "mode": mode,
            "protocol_id": config["protocol_id"],
            "inference_performed": False,
        },
    )
    inventory = verify_inventory(root, config)
    sizes, headers, records = {}, {}, []
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for image_id, path, expected in inventory:
        if sha256(path) != expected:
            raise PreflightError("DICOM checksum mismatch; no silent exclusions")
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        sizes[image_id] = validate_header(dataset)
        header = {key: str(getattr(dataset, key, "<absent>")) for key in adapter["header_fields"]}
        header["TransferSyntaxUID"] = str(dataset.file_meta.TransferSyntaxUID)
        headers[image_id] = header
        strata[tuple(header[key] for key in adapter["technical_sample_key"])].append(image_id)
    _, rows = read_csv(root / config["dataset"]["paths"]["boxes"])
    targets = build_targets(rows, sizes, config)
    sample = {
        image_id
        for ids in strata.values()
        for image_id in ids[: adapter["technical_sample_per_stratum"]]
    }
    sample.update(t.image_id for t in targets if len(t.boxes_xyxy))
    target_map = {target.image_id: target for target in targets}
    counters: Counter[str] = Counter()
    max_error = 0.0
    for index, (image_id, path, digest) in enumerate(inventory, 1):
        pixels, pixel_audit = decode_image(path, config)
        counters.update(
            {
                "decoded": 1,
                "constant": int(pixel_audit["constant"]),
                "nonfinite_pixels": pixel_audit["nonfinite_pixels"],
            }
        )
        if pixels.shape != sizes[image_id]:
            raise PreflightError("Preprocessing changed native dimensions")
        if image_id in sample:
            repeated, _ = decode_image(path, config)
            if not np.array_equal(pixels, repeated):
                raise PreflightError("Preprocessing is not deterministic")
            max_error = max(
                max_error,
                geometry_check(
                    target_map[image_id],
                    config["preprocessing"]["input_size"],
                    adapter["geometry_absolute_tolerance"],
                ),
            )
        png_buffer = io.BytesIO()
        Image.fromarray(pixels).save(
            png_buffer, format="PNG", compress_level=adapter["png_compress_level"]
        )
        encoded = png_buffer.getvalue()
        with Image.open(io.BytesIO(encoded)) as restored:
            if restored.mode != "L" or not np.array_equal(np.asarray(restored), pixels):
                raise PreflightError("PNG round-trip mismatch")
        destination = images_root / (image_id + ".png")
        if mode == "prepare":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(encoded)
        height, width = sizes[image_id]
        records.append(
            {
                "image_id": image_id,
                "file_name": destination.name,
                "height": height,
                "width": width,
                "box_count": len(target_map[image_id].boxes_xyxy),
                "source_sha256": digest,
                "png_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        if index % adapter["progress_every"] == 0:
            print(f"Validated {index}/{len(inventory)} images (no inference)", flush=True)
    coco = targets_to_coco(targets, config)
    positives = sum(bool(len(target.boxes_xyxy)) for target in targets)
    summary = {
        "status": "passed",
        "mode": mode,
        "protocol_id": config["protocol_id"],
        "dataset_version": config["dataset"]["version"],
        "test_count": len(targets),
        "strict_positive_count": positives,
        "strict_box_count": len(coco["annotations"]),
        "strict_negative_count": len(targets) - positives,
        "ontology": config["ontology"],
        "preprocessing": config["preprocessing"],
        "evaluation": config["evaluation"],
        "pixel_audit": dict(counters),
        "technical_repeat_and_geometry_count": len(sample),
        "geometry_max_absolute_error": max_error,
        "freeze_files_verified": len(frozen),
        "config_sha256": sha256(config_path),
        "protocol_sha256": sha256(Path(config["protocol_document"])),
        "adapter_config_sha256": sha256(adapter_path),
        "adapter_sha256": sha256(Path(__file__)),
        "source_sha256": config["dataset"]["source_sha256"],
        "inference_performed": False,
        "performance_analysis_performed": False,
        "environment": {
            key: version(key)
            for key in ("numpy", "pydicom", "Pillow", "torch", "torchvision", "ultralytics")
        },
    }
    summary["environment"].update(
        {"python": platform.python_version(), "openjpeg": features.version_codec("jpg_2000")}
    )
    # Publish only compact aggregate metadata; never raw header distributions/IDs.
    summary["header_audit"] = {
        key: {
            "present": sum(h[key] != "<absent>" for h in headers.values()),
            "distinct_nonmissing": len({h[key] for h in headers.values()} - {"<absent>"}),
        }
        for key in next(iter(headers.values()))
    }
    for key in (
        "PhotometricInterpretation",
        "PixelRepresentation",
        "BitsAllocated",
        "BitsStored",
        "TransferSyntaxUID",
    ):
        summary["header_audit"][key]["counts"] = dict(Counter(h[key] for h in headers.values()))
    summary["dimension_range"] = {
        "height": [min(h for h, _ in sizes.values()), max(h for h, _ in sizes.values())],
        "width": [min(w for _, w in sizes.values()), max(w for _, w in sizes.values())],
    }
    if mode == "prepare":
        write_json(annotations_path, coco)
        loader_config.write_text(
            yaml.safe_dump(
                {
                    "dataset": {
                        "paths": {
                            "processed_images_dir": images_root.as_posix(),
                            "annotations_dir": annotations_path.parent.as_posix(),
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        from src.models.faster_rcnn_data import CocoDetectionDataset

        loaded = CocoDetectionDataset(loader_config, "test", mode="full")
        if len(loaded.records) != len(targets):
            raise PreflightError("Common-loader image count mismatch")
        for record, target in zip(loaded.records, targets, strict=True):
            if (
                record.file_name != target.image_id + ".png"
                or (record.height, record.width) != target.image_size
                or not np.array_equal(
                    np.asarray([a.bbox_xyxy for a in record.annotations]).reshape(-1, 4),
                    target.boxes_xyxy,
                )
            ):
                raise PreflightError("Canonical loader coordinate/identity round-trip mismatch")
        summary["common_loader_images_verified"] = len(loaded.records)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        summary["manifest_sha256"] = sha256(manifest_path)
        summary["annotations_sha256"] = sha256(annotations_path)
    write_json(
        private_audit,
        {
            "headers": headers,
            "records": records,
            "technical_sample": sorted(sample),
            "summary": summary,
        },
    )
    write_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data-only CLI and fail visibly on any invalid prerequisite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "prepare"), required=True)
    args = parser.parse_args(argv)
    summary = run(args.config, args.adapter_config, args.mode)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "mode",
                    "test_count",
                    "strict_positive_count",
                    "strict_box_count",
                    "strict_negative_count",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
