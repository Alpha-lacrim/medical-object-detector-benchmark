import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.prepare import (
    build_coco,
    load_and_audit_records,
    load_dataset_config,
    prepare_dataset,
    split_records,
    subsample_records,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_config(tmp_path: Path, *, max_images: int, enabled: bool = True) -> dict[str, Any]:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    return {
        "schema_version": 1,
        "dataset": {
            "id": "synthetic-rsna",
            "display_name": "Synthetic RSNA",
            "paths": {
                "source_images_dir": str(raw / "dicoms"),
                "labels_csv": str(raw / "labels.csv"),
                "class_info_csv": str(raw / "classes.csv"),
                "mapping_json": str(raw / "mapping.json"),
                "processed_images_dir": str(processed / "images"),
                "annotations_dir": str(processed / "annotations"),
                "audit_json": str(processed / "audit_summary.json"),
                "splits_dir": str(tmp_path / "splits"),
            },
            "image": {
                "source_extension": ".dicom",
                "processed_extension": ".png",
                "width": 100,
                "height": 80,
                "conversion": {
                    "normalization": "per_image_minmax",
                    "output_bit_depth": 8,
                    "invert_monochrome1": True,
                },
            },
            "mapping": {
                "exam_id_field": "subset_key",
                "original_image_field": "original_name",
                "patient_id_pattern": r"^(?P<patient_id>[0-9]+)_[0-9]+\.png$",
            },
            "annotation": {
                "exam_id_field": "exam",
                "x_field": "left",
                "y_field": "top",
                "width_field": "box_width",
                "height_field": "box_height",
                "target_field": "flag",
                "positive_target": "yes",
                "negative_target": "no",
                "class_info_exam_id_field": "exam_ref",
                "class_info_class_field": "label",
            },
            "classes": {
                "foreground": ["opacity"],
                "study_strata": ["opacity", "other", "normal"],
                "positive_study_stratum": "opacity",
            },
            "subsample": {
                "enabled": enabled,
                "max_images": max_images,
                "seed": 23,
                "stratify_by": "study_stratum",
                "group_by": "nih_patient_id",
            },
            "split": {
                "seed": 29,
                "group_by": "nih_patient_id",
                "stratify_by": "study_stratum",
                "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            },
        },
    }


def _write_clean_fixture(tmp_path: Path, *, image_count: int = 30) -> Path:
    config = _base_config(tmp_path, max_images=20)
    dataset = config["dataset"]
    raw = Path(dataset["paths"]["labels_csv"]).parent
    raw.mkdir(parents=True)

    mapping: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []
    class_rows: list[dict[str, str]] = []
    for index in range(image_count):
        exam_id = f"exam-{index:03d}"
        if index < 8:
            patient_number = index // 2
            instance_number = index % 2
        else:
            patient_number = index - 4
            instance_number = 0
        mapping.append(
            {
                "subset_key": exam_id,
                "original_name": f"{patient_number:08d}_{instance_number:03d}.png",
            }
        )
        if index % 4 == 0:
            study_class = "opacity"
            label_rows.append(
                {
                    "exam": exam_id,
                    "left": "10",
                    "top": "5",
                    "box_width": "20",
                    "box_height": "10",
                    "flag": "yes",
                }
            )
            if index == 0:
                label_rows.append(
                    {
                        "exam": exam_id,
                        "left": "40",
                        "top": "20",
                        "box_width": "15",
                        "box_height": "12",
                        "flag": "yes",
                    }
                )
        elif index % 4 == 1:
            study_class = "other"
            label_rows.append(
                {
                    "exam": exam_id,
                    "left": "",
                    "top": "",
                    "box_width": "",
                    "box_height": "",
                    "flag": "no",
                }
            )
        else:
            study_class = "normal"
            label_rows.append(
                {
                    "exam": exam_id,
                    "left": "",
                    "top": "",
                    "box_width": "",
                    "box_height": "",
                    "flag": "no",
                }
            )
        class_rows.append({"exam_ref": exam_id, "label": study_class})

    Path(dataset["paths"]["mapping_json"]).write_text(json.dumps(mapping), encoding="utf-8")
    _write_csv(
        Path(dataset["paths"]["labels_csv"]),
        ["exam", "left", "top", "box_width", "box_height", "flag"],
        label_rows,
    )
    _write_csv(
        Path(dataset["paths"]["class_info_csv"]),
        ["exam_ref", "label"],
        class_rows,
    )
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_coco_conversion_preserves_boxes_and_negative_images(tmp_path: Path) -> None:
    config_path = _write_clean_fixture(tmp_path)
    config = load_dataset_config(config_path)
    audit = load_and_audit_records(config)

    assert not audit.issues
    coco = build_coco(audit.valid_records, config)
    assert coco["categories"] == [{"id": 1, "name": "opacity"}]
    assert len(coco["images"]) == 30

    image_by_exam = {image["rsna_exam_id"]: image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    positive_id = image_by_exam["exam-000"]["id"]
    negative_id = image_by_exam["exam-001"]["id"]
    assert [annotation["bbox"] for annotation in annotations_by_image[positive_id]] == [
        [10.0, 5.0, 20.0, 10.0],
        [40.0, 20.0, 15.0, 12.0],
    ]
    assert [annotation["area"] for annotation in annotations_by_image[positive_id]] == [
        200.0,
        180.0,
    ]
    assert negative_id not in annotations_by_image
    assert all(image["file_name"].endswith(".png") for image in coco["images"])


def test_audit_reports_invalid_and_duplicate_boxes(tmp_path: Path) -> None:
    config = _base_config(tmp_path, max_images=20, enabled=False)
    dataset = config["dataset"]
    raw = Path(dataset["paths"]["labels_csv"]).parent
    raw.mkdir(parents=True)
    exam_ids = [
        "good",
        "malformed",
        "nonpositive",
        "off-image",
        "duplicate",
        "mismatch",
        "negative-box",
    ]
    mapping = [
        {
            "subset_key": exam_id,
            "original_name": f"{index + 1:08d}_000.png",
        }
        for index, exam_id in enumerate(exam_ids)
    ]
    Path(dataset["paths"]["mapping_json"]).write_text(json.dumps(mapping), encoding="utf-8")
    rows = [
        ["good", "1", "2", "10", "12", "yes"],
        ["malformed", "bad", "2", "10", "12", "yes"],
        ["nonpositive", "1", "2", "0", "12", "yes"],
        ["off-image", "95", "2", "10", "12", "yes"],
        ["duplicate", "1", "2", "10", "12", "yes"],
        ["duplicate", "1", "2", "10", "12", "yes"],
        ["mismatch", "1", "2", "10", "12", "yes"],
        ["negative-box", "1", "2", "10", "12", "no"],
    ]
    label_fields = ["exam", "left", "top", "box_width", "box_height", "flag"]
    _write_csv(
        Path(dataset["paths"]["labels_csv"]),
        label_fields,
        [dict(zip(label_fields, row, strict=True)) for row in rows],
    )
    _write_csv(
        Path(dataset["paths"]["class_info_csv"]),
        ["exam_ref", "label"],
        [
            {
                "exam_ref": exam_id,
                "label": (
                    "normal"
                    if exam_id == "mismatch"
                    else "other"
                    if exam_id == "negative-box"
                    else "opacity"
                ),
            }
            for exam_id in exam_ids
        ],
    )

    result = load_and_audit_records(config)
    codes = {issue.code for issue in result.issues}
    assert {
        "malformed_box",
        "nonpositive_box",
        "off_image_box",
        "duplicate_box",
        "target_class_mismatch",
        "negative_with_box",
    } <= codes
    duplicate = next(record for record in result.records if record.exam_id == "duplicate")
    assert duplicate.valid
    assert len(duplicate.boxes) == 1
    assert {record.exam_id for record in result.valid_records} == {"good", "duplicate"}


def test_preparation_is_deterministic_exact_and_patient_disjoint(tmp_path: Path) -> None:
    config_path = _write_clean_fixture(tmp_path)
    config = load_dataset_config(config_path)
    audit = load_and_audit_records(config)

    first_sample = subsample_records(audit.valid_records, config)
    second_sample = subsample_records(tuple(reversed(audit.valid_records)), config)
    assert [record.exam_id for record in first_sample] == [
        record.exam_id for record in second_sample
    ]
    assert len(first_sample) == 20

    first_splits = split_records(first_sample, config)
    second_splits = split_records(second_sample, config)
    assert {name: len(records) for name, records in first_splits.items()} == {
        "train": 14,
        "val": 3,
        "test": 3,
    }
    assert {
        name: [record.exam_id for record in records] for name, records in first_splits.items()
    } == {name: [record.exam_id for record in records] for name, records in second_splits.items()}

    patient_sets = {
        name: {record.nih_patient_id for record in records}
        for name, records in first_splits.items()
    }
    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    summary = prepare_dataset(config_path, metadata_only=True)
    assert summary["subsample"]["selected_images"] == 20
    assert summary["image_processing"]["mode"] == "metadata_only"
    for split_name, expected in {"train": 14, "val": 3, "test": 3}.items():
        manifest_path = Path(summary["outputs"][f"manifest_{split_name}"])
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        assert len(manifest) == expected
        coco = json.loads(
            Path(summary["outputs"][f"coco_{split_name}"]).read_text(encoding="utf-8")
        )
        assert len(coco["images"]) == expected
    audit_summary = json.loads(
        Path(summary["outputs"]["audit_summary"]).read_text(encoding="utf-8")
    )
    assert audit_summary["audit"]["issue_count"] == 0
    assert all(overlap == 0 for overlap in audit_summary["split_group_overlap"].values())
    label_path = Path(config["dataset"]["paths"]["labels_csv"])
    assert (
        audit_summary["input_files"]["labels_csv"]["sha256"]
        == hashlib.sha256(label_path.read_bytes()).hexdigest()
    )


def test_preparation_rejects_mismatched_official_mapping_hash(tmp_path: Path) -> None:
    config_path = _write_clean_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["official_supplements"] = {"mapping_sha256": "0" * 64}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        prepare_dataset(config_path, metadata_only=True)
    except ValueError as error:
        assert "mapping SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("mismatched official mapping hash was accepted")
