"""Synthetic release tests: no restricted annotations or images are fixtures."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from scripts.verify_frozen_external import historical_editorial_view
from src.data.prepare import scale_radiograph_to_uint8
from src.data.prepare_vindr import (
    PreflightError,
    build_targets,
    decode_image,
    geometry_check,
    resolve_root,
    run,
    sha256,
    targets_to_coco,
    validate_header,
    verify_freeze,
    verify_inventory,
)
from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget, evaluate_operating_point


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/vindr_external_v1.yaml").read_text())


def row(image_id="positive", name="Lung Opacity", box=(3, 7, 23, 17)):
    return dict(
        zip(
            ["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"],
            [image_id, name, *map(str, box)],
            strict=True,
        )
    )


def dicom(path, polarity="MONOCHROME2", signed=False):
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.Rows, dataset.Columns = 2, 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = polarity
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = int(signed)
    pixels = np.array([[0, 10, 20], [30, 40, 50]], dtype=np.int16 if signed else np.uint16)
    if signed:
        pixels -= 20
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path)
    return dataset


def test_exact_ontology_and_all_negative_behaviors(config):
    rows = [
        row(),
        row("other", "Consolidation"),
        row("case", "Lung opacity"),
        row("synonym", "Infiltration"),
        row("global", "Pneumonia"),
    ]
    sizes = dict.fromkeys(["positive", "other", "case", "synonym", "global", "empty"], (40, 70))
    targets = build_targets(rows, sizes, config)
    assert [t.image_id for t in targets] == sorted(sizes)
    assert {t.image_id: len(t.boxes_xyxy) for t in targets} == {
        "positive": 1,
        "other": 0,
        "case": 0,
        "synonym": 0,
        "global": 0,
        "empty": 0,
    }
    assert sum(len(t.boxes_xyxy) for t in targets) == 1


@pytest.mark.parametrize(
    "box",
    [(-1, 0, 2, 3), (5, 3, 5, 7), (8, 0, 2, 3), (0, 0, 71, 4), (0, 0, 4, 41), (0, 0, "nan", 3)],
)
def test_bad_coordinates_stop(config, box):
    with pytest.raises(ValueError):
        build_targets([row(box=box)], {"positive": (40, 70)}, config)


def test_duplicates_and_unknown_ids_stop(config):
    with pytest.raises(PreflightError, match="Duplicate"):
        build_targets([row(), row()], {"positive": (40, 70)}, config)
    with pytest.raises(PreflightError, match="Unknown"):
        build_targets([row(name="other finding")], {"unrelated": (40, 70)}, config)


def test_xyxy_xywh_area_source_order_and_common_evaluator(config):
    targets = build_targets(
        [row(box=(3, 7, 23, 17)), row(box=(30, 2, 40, 9))],
        {"positive": (40, 70), "negative": (40, 70)},
        config,
    )
    coco = targets_to_coco(targets, config)
    assert coco["annotations"][0]["bbox"] == [3, 7, 20, 10]
    assert coco["annotations"][0]["area"] == 200
    assert coco["annotations"][1]["bbox"] == [30, 2, 10, 7]
    assert coco["images"][0]["file_name"] == "negative.png"
    predictions = [
        ImagePrediction(t.image_id, t.image_size, t.boxes_xyxy, t.labels, [0.9] * len(t.labels))
        for t in targets
    ]
    classes = tuple(config["ontology"]["canonical_classes"])
    assert evaluate_coco(predictions, targets, class_ids=classes)["ap50"] == pytest.approx(1)
    # Another finding remains a strict-target negative: one detection is one false positive.
    predictions[0] = ImagePrediction("negative", (40, 70), [[1, 2, 3, 4]], [classes[0]], [0.8])
    metrics = evaluate_operating_point(
        predictions,
        targets,
        class_ids=classes,
        score_threshold=0.1,
        iou_threshold=0.5,
        max_detections=100,
    )
    assert metrics["overall"]["fp"] == 1


@pytest.mark.parametrize("polarity", ["MONOCHROME1", "MONOCHROME2"])
@pytest.mark.parametrize("signed", [False, True])
def test_dicom_polarity_and_determinism(tmp_path, config, polarity, signed):
    path = tmp_path / "synthetic.dicom"
    dataset = dicom(path, polarity, signed)
    # Window metadata is deliberately ignored by the frozen stored-pixel pipeline.
    dataset.WindowCenter, dataset.WindowWidth = 1, 2
    dataset.RescaleSlope, dataset.RescaleIntercept = 1, 0
    dataset.save_as(path)
    first, audit = decode_image(path, config)
    second, _ = decode_image(path, config)
    expected = np.array([[0, 51, 102], [153, 204, 255]], dtype=np.uint8)
    if polarity == "MONOCHROME1":
        expected = 255 - expected
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(first, second)
    assert audit == {"constant": False, "nonfinite_pixels": 0}


def test_constant_and_nonfinite_reference_contract():
    kwargs = {"photometric_interpretation": "MONOCHROME1", "invert_monochrome1": True}
    np.testing.assert_array_equal(
        scale_radiograph_to_uint8(np.ones((2, 3)), **kwargs), np.zeros((2, 3), dtype=np.uint8)
    )
    result = scale_radiograph_to_uint8([[np.nan, -np.inf, 0], [np.inf, 5, 10]], **kwargs)
    np.testing.assert_array_equal(result, [[255, 255, 255], [0, 128, 0]])
    with pytest.raises(ValueError, match="no finite"):
        scale_radiograph_to_uint8(np.full((2, 2), np.nan), **kwargs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("PhotometricInterpretation", "RGB"),
        ("NumberOfFrames", 2),
        ("SamplesPerPixel", 3),
        ("RescaleSlope", -1),
        ("HighBit", 4),
    ],
)
def test_unexpected_dicom_semantics_stop(tmp_path, key, value):
    dataset = dicom(tmp_path / "synthetic.dicom")
    setattr(dataset, key, value)
    with pytest.raises(PreflightError):
        validate_header(dataset)


@pytest.mark.parametrize("size", [(379, 811), (811, 379), (640, 640), (3408, 3320)])
def test_native_resize_and_inverse_restoration(config, size):
    target = ImageTarget("synthetic", size, [[3, 7, 23, 17]], [1])
    assert geometry_check(target, config["preprocessing"]["input_size"], 0.001) < 0.001


def test_root_has_no_fallback_and_requires_authorization(config, monkeypatch, tmp_path):
    variable = config["dataset"]["root_environment_variable"]
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(PreflightError, match="Set"):
        resolve_root(config)
    monkeypatch.setenv(variable, str(tmp_path))
    assert resolve_root(config) == tmp_path.resolve()
    config["dataset"]["access"]["approved_access_dua_and_official_download_confirmed"] = False
    with pytest.raises(PreflightError, match="attestation"):
        resolve_root(config)


def test_frozen_config_and_dependencies(config):
    with historical_editorial_view(Path.cwd()):
        assert len(verify_freeze(Path("configs/vindr_external_v1.yaml"), config)) == 38


def test_synthetic_release_inventory_and_failure_gates(tmp_path, config, monkeypatch):
    config = copy.deepcopy(config)
    dataset = config["dataset"]
    dataset["expected_images"] = 2
    (tmp_path / "test").mkdir()
    for image_id in ("a", "b"):
        dicom(tmp_path / "test" / (image_id + ".dicom"))
    (tmp_path / dataset["paths"]["image_labels"]).write_text("image_id,Lung Opacity\na,1\nb,0\n")
    with (tmp_path / dataset["paths"]["boxes"]).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=dataset["annotation_columns"])
        writer.writeheader()
        writer.writerow(row("a", box=(0, 0, 2, 1)))
    for key in ("license", "dicom_tag_documentation"):
        (tmp_path / dataset["paths"][key]).write_text("synthetic fixture only")
    checksum_path = tmp_path / dataset["paths"]["checksums"]
    checksum_path.write_text(
        "".join(f"{sha256(p)} test/{p.name}\n" for p in sorted((tmp_path / "test").iterdir()))
    )
    dataset["source_sha256"] = {
        key: sha256(tmp_path / name) for key, name in dataset["paths"].items() if key != "images"
    }
    assert [item[0] for item in verify_inventory(tmp_path, config)] == ["a", "b"]
    # Full CLI core, private artifacts and the existing loader run end to end.
    private = tmp_path / "processed"
    config["outputs"].update(
        {
            "private_root": str(private),
            "local_manifest": str(private / "manifest.csv"),
            "local_annotations": str(private / "instances_test.json"),
            "local_predictions": str(private / "predictions"),
            "aggregate_results_root": str(tmp_path / "aggregates"),
        }
    )
    config["freeze_manifest"] = str(tmp_path / "freeze.json")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    Path(config["freeze_manifest"]).write_text(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "artifacts": [{"path": str(config_file), "sha256": sha256(config_file)}],
            }
        )
    )
    monkeypatch.setenv(dataset["root_environment_variable"], str(tmp_path))
    operational = Path("configs/vindr_adapter_v1.yaml")
    result = run(config_file, operational, "prepare")
    assert (
        result["test_count"],
        result["strict_positive_count"],
        result["strict_box_count"],
        result["strict_negative_count"],
    ) == (2, 1, 1, 1)
    assert result["common_loader_images_verified"] == 2
    original_manifest = (private / "manifest.csv").read_bytes()
    audit_name = yaml.safe_load(operational.read_text())["private_audit_name"]
    first_records = json.loads((private / audit_name).read_text())["records"]
    second = run(config_file, operational, "preflight")
    assert second["pixel_audit"]["decoded"] == 2
    assert (private / "manifest.csv").read_bytes() == original_manifest
    assert json.loads((private / audit_name).read_text())["records"] == first_records
    assert not list((tmp_path / "aggregates").glob("*.csv"))
    predictions = private / "predictions"
    predictions.mkdir()
    (predictions / "synthetic.json").write_text("{}")
    with pytest.raises(PreflightError, match="predictions already exist"):
        run(config_file, operational, "preflight")
    (tmp_path / "test" / "b.dicom").unlink()
    with pytest.raises(PreflightError, match="inventory"):
        verify_inventory(tmp_path, config)
    (tmp_path / dataset["paths"]["boxes"]).write_text("wrong,release\n")
    with pytest.raises(PreflightError, match="checksum"):
        verify_inventory(tmp_path, config)


def test_freeze_tampering_stops(tmp_path):
    source = tmp_path / "frozen.yaml"
    source.write_text("original")
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "synthetic",
                "artifacts": [{"path": str(source), "sha256": sha256(source)}],
            }
        )
    )
    config = {"protocol_id": "synthetic", "freeze_manifest": str(manifest)}
    source.write_text("edited")
    with pytest.raises(PreflightError, match="Frozen input"):
        verify_freeze(source, config)
