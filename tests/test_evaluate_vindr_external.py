"""Synthetic scientific-contract checks, without restricted data or weights."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluate import _serialize_prediction
from src.evaluate_vindr_external import (
    aggregate_metrics,
    compute_metrics,
    deserialize,
    dump_gzip,
    native_floor_audit,
    read_gzip,
    restore_native_bounds,
    score_summary,
    select_budgets,
    verify_boundary_repairs,
)
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/vindr_external_v1.yaml").read_text(encoding="utf-8"))


def records(scores=(0.8, 0.05)):
    target = ImageTarget("positive", (20, 20), [[0, 0, 10, 10]], [1])
    negative = ImageTarget("other-finding-negative", (20, 20), [], [])
    prediction = ImagePrediction("positive", (20, 20), [[0, 0, 10, 10]], [1], [scores[0]])
    false_positive = ImagePrediction(
        "other-finding-negative", (20, 20), [[0, 0, 10, 10]], [1], [scores[1]]
    )
    return [prediction, false_positive], [target, negative]


def test_transport_and_all_image_denominators(config):
    predictions, targets = records()
    metrics, curve = compute_metrics(predictions, targets, config, "yolo11s", 271)
    primary = metrics["threshold_transport"]["primary"]
    assert primary["threshold"] == 0.05
    assert primary["selection_seeds"] == [17, 42, 137]
    assert primary["tp"] == primary["fp"] == 1
    assert primary["precision"] == 0.5
    assert primary["recall"] == 1
    assert primary["f1"] == pytest.approx(2 / 3)
    assert primary["fp_per_image"] == 0.5
    assert primary["detections_per_image"] == 1
    assert primary["percent_images_with_detections"] == 100
    assert metrics["seed"] == 271
    secondary = metrics["threshold_transport"]["secondary"]
    assert secondary["threshold"] == 0.01
    assert secondary["selection_seeds"] == [17, 42, 137, 271, 314]
    assert "posthoc_n5" in secondary["label"]
    assert metrics["ap"]["ap50"] == pytest.approx(1)
    assert curve[-1]["fp_per_image"] == 0.5
    assert metrics["froc_budgets"][-1]["candidate_floor_limited"]
    assert all("selected_threshold" not in row for row in metrics["froc_budgets"])


def test_empty_seed_is_retained_with_defined_zeros(config):
    _, targets = records()
    predictions = [ImagePrediction(t.image_id, t.image_size, [], [], []) for t in targets]
    metrics, _ = compute_metrics(predictions, targets, config, "yolo11s", 271)
    assert metrics["ap"]["ap50"] == 0
    assert metrics["ap"]["ap50_95"] == 0
    for policy in metrics["threshold_transport"].values():
        assert [policy[k] for k in ("precision", "recall", "f1", "fp_per_image")] == [0] * 4
        assert policy["zero_detection_images"] == 2
    for population in metrics["score_summaries"].values():
        assert population["detection_count"] == 0
        assert population["median"] is None


def test_ap_floor_differs_from_froc_floor(config):
    predictions, targets = records((0.0005, 0.0001))
    metrics, curve = compute_metrics(predictions, targets, config, "faster_rcnn", 17)
    assert metrics["ap"]["prediction_count"] == 0
    assert metrics["ap"]["ap50"] == 0
    assert curve[-1]["true_positives"] == 1
    assert metrics["score_summaries"]["all_retained_candidates"]["detection_count"] == 2
    assert metrics["score_summaries"]["ap_candidates"]["detection_count"] == 0


def test_linear_score_quantiles(config):
    summary = score_summary(np.array([0.0, 0.2, 0.8, 1.0]), config)
    assert summary["mean"] == summary["median"] == 0.5
    assert summary["quantiles"]["0.25"] == pytest.approx(0.15)
    assert score_summary(np.array([]), config)["mean"] is None


def test_equal_run_aggregation_keeps_empty_seed_and_defined_counts(config):
    runs = []
    for detector in config["models"]["detectors"]:
        for seed in config["models"]["seeds"]:
            value = None if seed == 271 else 0.5
            runs.append(
                {
                    "metrics": {
                        "detector": detector,
                        "seed": seed,
                        "ap": {"ap50": 0.0 if seed == 271 else 1.0},
                        "score_summaries": {"median": value},
                    }
                }
            )
    summary = aggregate_metrics(runs, config)
    for group in summary.values():
        assert group["ap.ap50"]["defined_run_count"] == 5
        assert group["ap.ap50"]["mean"] == 0.8
        assert group["ap.ap50"]["sample_sd"] == pytest.approx(np.std([1, 1, 1, 0, 1], ddof=1))
        assert group["score_summaries.median"]["defined_run_count"] == 4
    with pytest.raises(ValueError, match="lost a run"):
        aggregate_metrics(runs[:-1], config)


def test_per_run_budget_rule_matches_frozen_multi_run_helper(config):
    from src.analyze_exact_score_froc import select_exact_operating_points

    predictions, targets = records()
    _, curve = compute_metrics(predictions, targets, config, "faster_rcnn", 17)
    expanded = [dict(row, seed=seed) for seed in config["models"]["seeds"] for row in curve]
    settings = config["evaluation"]
    _, reference = select_exact_operating_points(
        expanded,
        settings["froc"]["fp_per_image_budgets"],
        candidate_score_floor=settings["candidate_score_floor"],
        tolerance=settings["froc"]["numeric_tolerance"],
    )
    selected = select_budgets(curve, config)
    for actual, expected in zip(selected, reference[: len(selected)], strict=True):
        for key, value in actual.items():
            assert expected[key] == value


def test_native_candidate_floor_equality(config):
    audit = native_floor_audit(config["evaluation"]["candidate_score_floor"])
    assert audit["equality_candidate_not_emitted_by_native_filters"]
    assert audit["common_evaluator_score_comparison"] == "greater_than_or_equal"


def test_torchvision_one_ulp_restore_repair_preserves_interior_coordinates():
    import torch
    from torchvision.models.detection.transform import resize_boxes

    native = resize_boxes(torch.tensor([[2.0, 3.0, 578.0, 640.0]]), (640, 578), (2680, 2424))
    raw = native.numpy()
    assert float(raw[0, 2]) > 2424
    fixed, changes = restore_native_bounds(raw, (2680, 2424), "faster_rcnn")
    assert fixed[0, 2] == 2424
    assert np.array_equal(fixed[0, [0, 1, 3]], raw[0, [0, 1, 3]])
    assert changes == [
        {
            "box_index": 0,
            "coordinate_index": 2,
            "raw_value": 2424.000244140625,
            "canonical_value": 2424.0,
        }
    ]
    assert raw[0, 2] > 2424  # original native evidence was not mutated


@pytest.mark.parametrize("fault", ["two_ulps", "negative", "wrong_detector", "wrong_dtype"])
def test_restoration_repair_rejects_other_errors(fault):
    raw = np.array(
        [[0, 0, np.nextafter(np.float32(2424), np.float32(np.inf)), 2680]], dtype=np.float32
    )
    detector = "faster_rcnn"
    if fault == "two_ulps":
        raw[0, 2] = np.nextafter(raw[0, 2], np.float32(np.inf))
    elif fault == "negative":
        raw[0, 0] = -np.finfo(np.float32).eps
    elif fault == "wrong_detector":
        detector = "yolo11s"
    else:
        raw = raw.astype(np.float64)
    with pytest.raises(ValueError):
        restore_native_bounds(raw, (2680, 2424), detector)


def test_saved_raw_boundary_repair_replays_and_rejects_tampering():
    raw = np.array([[0, 0, 2424.000244140625, 2680]], dtype=np.float32)
    fixed, changes = restore_native_bounds(raw, (2680, 2424), "faster_rcnn")
    prediction = ImagePrediction("synthetic", (2680, 2424), fixed, [1], [0.5])
    bundle = {
        "predictions": [_serialize_prediction(prediction)],
        "native_boundary_repairs": [{"image_id": "synthetic", "changes": changes}],
        "metadata": {
            "detector": "faster_rcnn",
            "native_boundary_repair": {
                "images": 1,
                "coordinates": 1,
                "maximum_excess_pixels": 0.000244140625,
            },
        },
    }
    verify_boundary_repairs(bundle)
    bundle["native_boundary_repairs"][0]["changes"][0]["raw_value"] = 2425
    with pytest.raises(ValueError):
        verify_boundary_repairs(bundle)


def test_private_bundle_roundtrip_and_refuse_overwrite(tmp_path):
    predictions, _ = records()
    payload = [_serialize_prediction(p) for p in predictions]
    path = tmp_path / "bundle.json.gz"
    dump_gzip(path, payload)
    restored = deserialize(read_gzip(path))
    assert json.dumps([_serialize_prediction(p) for p in restored]) == json.dumps(payload)
    with pytest.raises(FileExistsError):
        dump_gzip(path, [])


@pytest.mark.parametrize("fault", ["floor", "dimension", "order", "cap"])
def test_reject_invalid_prediction_contract(config, fault):
    predictions, targets = records()
    if fault == "floor":
        predictions, targets = records((1e-6, 0.1))
    elif fault == "dimension":
        predictions[0] = ImagePrediction("positive", (30, 30), [[0, 0, 10, 10]], [1], [0.8])
    elif fault == "order":
        predictions.reverse()
    else:
        count = config["evaluation"]["max_detections_per_image"] + 1
        predictions[0] = ImagePrediction(
            "positive", (20, 20), [[0, 0, 10, 10]] * count, [1] * count, [0.8] * count
        )
    with pytest.raises(ValueError):
        compute_metrics(predictions, targets, config, "faster_rcnn", 17)
