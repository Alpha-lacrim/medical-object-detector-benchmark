"""Focused guards and analytic oracles for the secondary numerical-path study."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analyze_yolo_numerical_path import (
    candidate_agreement,
    canonical_bytes,
    inference_only,
    load_config,
    make_tables,
    preflight,
    validate_contract,
    write_new,
)
from src.meddet_benchmark.evaluation import ImagePrediction

CONFIG = Path("configs/yolo_numerical_path_sensitivity_v1.yaml")


@pytest.fixture
def protocol() -> tuple[dict, dict]:
    config = load_config(CONFIG)
    external = yaml.safe_load(Path(config["inputs"]["external_config"]).read_text())
    return config, external


def test_complete_run_scope(protocol: tuple[dict, dict]) -> None:
    config, _ = protocol
    assert config["seeds"] == [17, 42, 137, 271, 314]
    assert [r["seed"] for r in config["runs"]] == config["seeds"]
    assert len({r["sha256"] for r in config["runs"]}) == 5


@pytest.mark.parametrize("seed", [17, 42, 137, 271, 314])
def test_no_run_can_be_excluded(protocol: tuple[dict, dict], seed: int) -> None:
    config, external = protocol
    config["runs"] = [r for r in config["runs"] if r["seed"] != seed]
    with pytest.raises(ValueError, match="scope"):
        validate_contract(config, external)


@pytest.mark.parametrize("field", ["training", "threshold_selection", "external_inference"])
def test_forbidden_actions(protocol: tuple[dict, dict], field: str) -> None:
    config, external = protocol
    config["scope"][field] = True
    with pytest.raises(ValueError, match="Forbidden"):
        validate_contract(config, external)


def test_training_api_guard() -> None:
    from ultralytics.engine.model import Model

    with inference_only(), pytest.raises(RuntimeError, match="Training is forbidden"):
        Model.train(object())


@pytest.mark.parametrize("field,value", [("activation_dtype", "torch.float16"), ("amp", False)])
def test_dtype_explicit(protocol: tuple[dict, dict], field: str, value: object) -> None:
    config, external = protocol
    config["runtime"][field] = value
    with pytest.raises(ValueError):
        validate_contract(config, external)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_score_floor",
        "ap_minimum_score",
        "matching_iou_threshold",
        "nms_iou_threshold",
        "max_detections_per_image",
    ],
)
def test_frozen_evaluator_constants(protocol: tuple[dict, dict], field: str) -> None:
    config, external = protocol
    config["evaluation"][field] *= 2
    with pytest.raises(ValueError, match="evaluation"):
        validate_contract(config, external)


def test_threshold_reselection_forbidden(protocol: tuple[dict, dict]) -> None:
    config, external = protocol
    assert config["threshold_transport"]["primary"]["thresholds"]["yolo11s"] == 0.05
    assert config["threshold_transport"]["secondary"]["thresholds"]["yolo11s"] == 0.01
    config["threshold_transport"]["primary"]["thresholds"]["yolo11s"] = 0.04
    with pytest.raises(ValueError, match="threshold_transport"):
        validate_contract(config, external)
    # No selector module or training module is imported by this experiment.
    source = Path("src/analyze_yolo_numerical_path.py").read_text()
    assert "import src.evaluate_threshold_selection" not in source
    assert "from src.evaluate_threshold_selection" not in source
    assert "train_yolo" not in source


@pytest.mark.parametrize(
    "root",
    [
        "results/logs/phase5_evaluation",
        "results/logs/phase42_froc_lower_floor_v4",
        "results/vindr_external_v1",
        "results/logs/phase53_../bad_v1",
    ],
)
def test_historical_output_paths_refused(protocol: tuple[dict, dict], root: str) -> None:
    config, external = protocol
    config["outputs"]["root"] = root
    with pytest.raises(ValueError, match="output root"):
        validate_contract(config, external)


def test_existing_version_refused_before_inputs_or_gpu(
    tmp_path: Path, protocol: tuple[dict, dict]
) -> None:
    config, _ = protocol
    config["outputs"]["root"] = str(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        preflight(config)


def test_exclusive_deterministic_outputs(tmp_path: Path) -> None:
    payload = {"schema_version": 1, "b": [1, 2], "a": None}
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    write_new(first, payload, compressed=True)
    write_new(second, payload, compressed=True)
    assert first.read_bytes() == second.read_bytes()
    assert gzip.decompress(first.read_bytes()) == canonical_bytes(payload)
    with pytest.raises(FileExistsError):
        write_new(first, {"changed": True})
    assert json.loads(gzip.decompress(first.read_bytes())) == payload


def prediction(boxes: list, scores: list) -> ImagePrediction:
    return ImagePrediction(
        "image.png", (100, 100), boxes, np.ones(len(scores), dtype=np.int64), scores
    )


def test_matching_crossings_and_unmatched_candidates(protocol: tuple[dict, dict]) -> None:
    config, _ = protocol
    a = prediction([[0, 0, 20, 20], [50, 50, 60, 60]], [0.049, 0.8])
    b = prediction([[0.1, 0, 20, 20], [75, 75, 85, 85]], [0.051, 0.7])
    result = candidate_agreement([a], [b], config["agreement"])
    assert result["matched_candidates"] == 1
    assert result["appeared_candidates"] == result["disappeared_candidates"] == 1
    assert result["fraction_images_equal_count"] == 1
    assert result["threshold_crossings"]["0.05"] == {
        "matched_up": 1,
        "matched_down": 0,
        "unmatched_old_above": 1,
        "unmatched_new_above": 1,
        "matched_up_fraction": 1.0,
        "matched_down_fraction": 0.0,
    }
    assert result["score_difference"]["mean"] == pytest.approx(0.002)
    assert result["box_absolute_coordinate_difference_pixels"]["max"] == pytest.approx(0.1)
    assert result["spearman_matched_scores"] is None


def test_empty_and_identical_agreement(protocol: tuple[dict, dict]) -> None:
    config, _ = protocol
    empty = prediction([], [])
    p = prediction([[0, 0, 20, 20], [50, 50, 60, 60]], [0.1, 0.2])
    result = candidate_agreement([empty, p], [empty, p], config["agreement"])
    assert result["spearman_matched_scores"] == pytest.approx(1)
    assert result["appeared_fraction"] == result["disappeared_fraction"] == 0
    assert result["absolute_score_difference"]["max"] == 0


def test_equal_run_aggregation_schema() -> None:
    pairs = [
        {"seed": 17, "fp32": {"ap": 0.1}, "bfloat16": {"ap": 0.2}, "agreement": {}},
        {"seed": 271, "fp32": {"ap": 0.3}, "bfloat16": {"ap": 0.1}, "agreement": {}},
    ]
    one = make_tables(pairs, 1)
    assert canonical_bytes(one) == canonical_bytes(make_tables(copy.deepcopy(pairs), 1))
    assert one[1][0]["fp32_mean"] == pytest.approx(0.2)
    assert one[1][0]["delta_mean"] == pytest.approx(-0.05)
    assert one[1][0]["runs"] == 2
