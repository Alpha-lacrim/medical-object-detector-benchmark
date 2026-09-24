from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.analyze_exact_score_froc import (
    audit_exact_score_monotonicity,
    conservative_frontier_bounds,
    exact_score_froc_rows,
    load_exact_froc_config,
    load_exact_inputs,
    load_prior_exact_operating_points,
    select_exact_operating_points,
)
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget


def _small_records() -> tuple[list[ImagePrediction], list[ImageTarget]]:
    targets = [
        ImageTarget(
            image_id="case.png",
            image_size=(10, 10),
            boxes_xyxy=[[0, 0, 4, 4]],
            labels=[1],
        )
    ]
    predictions = [
        ImagePrediction(
            image_id="case.png",
            image_size=(10, 10),
            boxes_xyxy=[[6, 6, 9, 9], [5, 5, 9, 9], [0, 0, 4, 4]],
            labels=[1, 1, 1],
            scores=[0.2, 0.9, 0.8],
        )
    ]
    return predictions, targets


@pytest.fixture(scope="module")
def frozen_exact_inputs() -> tuple[Any, dict[str, Any], list[dict[str, Any]], Any, Any]:
    config = load_exact_froc_config("configs/froc_exact_score_v2.yaml")
    return load_exact_inputs(config)


@pytest.fixture(scope="module")
def frozen_exact_rows(
    frozen_exact_inputs: tuple[Any, dict[str, Any], list[dict[str, Any]], Any, Any],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    phase5, _summary, bundles, targets, category_names = frozen_exact_inputs
    config = load_exact_froc_config("configs/froc_exact_score_v2.yaml")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        run_rows, run_summary = exact_score_froc_rows(
            bundle["predictions"],
            targets,
            detector=bundle["detector"],
            seed=bundle["seed"],
            class_ids=tuple(sorted(category_names)),
            candidate_score_floor=config.analysis.candidate_score_floor,
            iou_threshold=config.analysis.match_iou_threshold,
            max_detections=config.analysis.max_detections_per_image,
        )
        rows.extend(run_rows)
        summaries.append(run_summary)
    assert phase5.seeds == config.analysis.expected_seeds
    return config, rows, summaries


@pytest.fixture(scope="module")
def lower_floor_exact_inputs() -> tuple[Any, dict[str, Any], list[dict[str, Any]], Any, Any]:
    config = load_exact_froc_config("configs/froc_exact_score_v3.yaml")
    return load_exact_inputs(config)


@pytest.fixture(scope="module")
def lower_floor_exact_rows(
    lower_floor_exact_inputs: tuple[Any, dict[str, Any], list[dict[str, Any]], Any, Any],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    phase5, _summary, bundles, targets, category_names = lower_floor_exact_inputs
    config = load_exact_froc_config("configs/froc_exact_score_v3.yaml")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        run_rows, run_summary = exact_score_froc_rows(
            bundle["predictions"],
            targets,
            detector=bundle["detector"],
            seed=bundle["seed"],
            class_ids=tuple(sorted(category_names)),
            candidate_score_floor=config.analysis.candidate_score_floor,
            iou_threshold=config.analysis.match_iou_threshold,
            max_detections=config.analysis.max_detections_per_image,
        )
        rows.extend(run_rows)
        summaries.append(run_summary)
    assert phase5.seeds == config.analysis.expected_seeds
    return config, rows, summaries


def test_exact_threshold_order_is_deterministic() -> None:
    predictions, targets = _small_records()
    kwargs = {
        "detector": "detector",
        "seed": 17,
        "class_ids": (1,),
        "candidate_score_floor": 0.1,
        "iou_threshold": 0.5,
        "max_detections": 100,
    }
    first, _summary = exact_score_froc_rows(predictions, targets, **kwargs)
    second, _summary = exact_score_froc_rows(predictions, targets, **kwargs)

    assert first == second
    assert [row["threshold"] for row in first] == [
        np.nextafter(0.9, np.inf),
        0.9,
        0.8,
        0.2,
        0.1,
    ]
    assert [row["threshold_relaxation_rank"] for row in first] == list(range(5))


def test_exact_frontier_endpoint_behavior() -> None:
    predictions, targets = _small_records()
    rows, summary = exact_score_froc_rows(
        predictions,
        targets,
        detector="detector",
        seed=17,
        class_ids=(1,),
        candidate_score_floor=0.1,
        iou_threshold=0.5,
        max_detections=100,
    )

    assert rows[0]["threshold_kind"] == "upper_empty_sentinel"
    assert rows[0]["prediction_count"] == 0
    assert rows[0]["sensitivity"] == 0
    assert rows[0]["fp_per_image"] == 0
    assert rows[-1]["threshold_kind"] == "candidate_floor_sentinel"
    assert rows[-1]["is_candidate_floor_boundary"] is True
    assert rows[-1]["threshold"] == 0.1
    assert rows[-1]["prediction_count"] == 3
    assert rows[-1]["sensitivity"] == 1
    assert rows[-1]["fp_per_image"] == 2
    assert summary["candidate_floor_endpoint_sensitivity"] == 1


def test_sensitivity_and_fp_are_monotonic_as_threshold_relaxes() -> None:
    predictions, targets = _small_records()
    rows, _summary = exact_score_froc_rows(
        predictions,
        targets,
        detector="detector",
        seed=17,
        class_ids=(1,),
        candidate_score_floor=0.1,
        iou_threshold=0.5,
        max_detections=100,
    )

    audit_exact_score_monotonicity(rows, tolerance=0)
    assert [row["sensitivity"] for row in rows] == [0, 0, 1, 1, 1]
    assert [row["fp_per_image"] for row in rows] == [0, 1, 1, 2, 2]


@pytest.mark.scientific_data
def test_exact_config_and_inputs_include_all_five_runs_and_seed_271(
    frozen_exact_inputs: tuple[Any, dict[str, Any], list[dict[str, Any]], Any, Any],
) -> None:
    phase5, summary, bundles, targets, category_names = frozen_exact_inputs
    config = load_exact_froc_config("configs/froc_exact_score_v2.yaml")

    assert config.analysis.expected_seeds == (17, 42, 137, 271, 314)
    assert config.analysis.fp_per_image_budgets == (0.125, 0.25, 0.5, 1.0, 2.0)
    assert config.analysis.candidate_score_floor == 0.001
    assert config.analysis.match_iou_threshold == 0.5
    assert config.analysis.nms_iou_threshold == 0.5
    assert config.analysis.max_detections_per_image == 100
    assert phase5.seeds == config.analysis.expected_seeds
    assert summary["status"] == "complete"
    assert len(targets) == 750
    assert category_names == {1: "Lung Opacity"}
    assert {(bundle["detector"], bundle["seed"]) for bundle in bundles} == {
        (detector, seed)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42, 137, 271, 314)
    }


@pytest.mark.scientific_data
def test_all_frozen_runs_pass_exact_monotonicity_and_seed_271_is_retained(
    frozen_exact_rows: tuple[Any, list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    _config, rows, summaries = frozen_exact_rows
    identities = {(row["detector"], row["seed"]) for row in summaries}
    assert identities == {
        (detector, seed)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42, 137, 271, 314)
    }
    for detector, seed in identities:
        selected = [row for row in rows if row["detector"] == detector and row["seed"] == seed]
        audit_exact_score_monotonicity(selected, tolerance=0)
    yolo_271 = next(row for row in summaries if row["detector"] == "yolo11s" and row["seed"] == 271)
    assert yolo_271["unique_evaluated_scores"] == 962
    assert yolo_271["candidate_floor_endpoint_sensitivity"] > 0
    assert yolo_271["maximum_emitted_score"] < 0.05


@pytest.mark.scientific_data
def test_exact_budget_aggregation_retains_five_runs_and_marks_floor_limits(
    frozen_exact_rows: tuple[Any, list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    config, rows, _summaries = frozen_exact_rows
    aggregate, per_seed = select_exact_operating_points(
        rows,
        config.analysis.fp_per_image_budgets,
        candidate_score_floor=config.analysis.candidate_score_floor,
        tolerance=config.analysis.numeric_tolerance,
    )

    assert len(per_seed) == 50
    assert len(aggregate) == 10
    assert {row["seed_count"] for row in aggregate} == {5}
    assert any(row["detector"] == "yolo11s" and row["seed"] == 271 for row in per_seed)
    yolo_two = next(
        row
        for row in aggregate
        if row["detector"] == "yolo11s" and row["fp_per_image_budget"] == 2.0
    )
    assert yolo_two["candidate_floor_limited_run_count"] == 5
    assert yolo_two["candidate_floor_limited_seeds"] == "17;42;137;271;314"


@pytest.mark.scientific_data
def test_lower_floor_inputs_retain_all_runs_and_approved_contract(
    lower_floor_exact_inputs: tuple[Any, dict[str, Any], list[dict[str, Any]], Any, Any],
) -> None:
    phase5, summary, bundles, targets, category_names = lower_floor_exact_inputs
    config = load_exact_froc_config("configs/froc_exact_score_v3.yaml")

    assert config.schema_version == 3
    assert config.analysis.expected_seeds == (17, 42, 137, 271, 314)
    assert config.analysis.candidate_score_floor == 0.0001
    assert config.analysis.historical_candidate_score_floor == 0.001
    assert config.analysis.fp_per_image_budgets == (0.125, 0.25, 0.5, 1.0, 2.0)
    assert config.inference_sensitivity_proposal is None
    assert phase5.seeds == config.analysis.expected_seeds
    assert phase5.evaluation.match_iou_threshold == 0.5
    assert phase5.evaluation.nms_iou_threshold == 0.5
    assert phase5.evaluation.max_detections == 100
    assert summary["approval"]["status"] == "user_approved"
    assert summary["protocol"]["performs_training"] is False
    assert len(targets) == 750
    assert category_names == {1: "Lung Opacity"}
    assert {(bundle["detector"], bundle["seed"]) for bundle in bundles} == {
        (detector, seed)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42, 137, 271, 314)
    }


@pytest.mark.scientific_data
def test_lower_floor_runs_are_monotonic_and_seed_271_is_retained(
    lower_floor_exact_rows: tuple[Any, list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    _config, rows, summaries = lower_floor_exact_rows
    identities = {(row["detector"], row["seed"]) for row in summaries}
    assert identities == {
        (detector, seed)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42, 137, 271, 314)
    }
    for detector, seed in identities:
        selected = [row for row in rows if row["detector"] == detector and row["seed"] == seed]
        audit_exact_score_monotonicity(selected, tolerance=0)
        assert selected[0]["threshold_kind"] == "upper_empty_sentinel"
        assert selected[-1]["threshold_kind"] == "candidate_floor_sentinel"
    yolo_271 = next(row for row in summaries if row["detector"] == "yolo11s" and row["seed"] == 271)
    assert yolo_271["unique_evaluated_scores"] == 4406
    assert yolo_271["maximum_emitted_score"] < 0.05


@pytest.mark.scientific_data
def test_lower_floor_budget_aggregation_retains_five_runs_and_residual_limit(
    lower_floor_exact_rows: tuple[Any, list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    config, rows, _summaries = lower_floor_exact_rows
    aggregate, per_seed = select_exact_operating_points(
        rows,
        config.analysis.fp_per_image_budgets,
        candidate_score_floor=config.analysis.candidate_score_floor,
        tolerance=config.analysis.numeric_tolerance,
    )

    assert len(per_seed) == 50
    assert len(aggregate) == 10
    assert {row["seed_count"] for row in aggregate} == {5}
    assert any(row["detector"] == "yolo11s" and row["seed"] == 271 for row in per_seed)
    assert all(
        row["candidate_floor_limited_run_count"] == 0
        for row in aggregate
        if row["detector"] == "faster_rcnn"
    )
    yolo_half = next(
        row
        for row in aggregate
        if row["detector"] == "yolo11s" and row["fp_per_image_budget"] == 0.5
    )
    yolo_one = next(
        row
        for row in aggregate
        if row["detector"] == "yolo11s" and row["fp_per_image_budget"] == 1.0
    )
    yolo_two = next(
        row
        for row in aggregate
        if row["detector"] == "yolo11s" and row["fp_per_image_budget"] == 2.0
    )
    assert yolo_half["candidate_floor_limited_run_count"] == 0
    assert yolo_one["candidate_floor_limited_seeds"] == "137"
    assert yolo_two["candidate_floor_limited_seeds"] == "137"


@pytest.mark.scientific_data
def test_v4_inputs_retain_ten_runs_seed_271_and_prior_hash_binding() -> None:
    config = load_exact_froc_config("configs/froc_exact_score_v4.yaml")
    phase5, summary, bundles, targets, category_names = load_exact_inputs(config)
    prior_per_seed, prior_aggregate, prior_summary = load_prior_exact_operating_points(config)

    assert config.schema_version == 4
    assert config.analysis.candidate_score_floor == 0.00001
    assert phase5.seeds == (17, 42, 137, 271, 314)
    assert len(bundles) == 10
    assert any(bundle["seed"] == 271 for bundle in bundles)
    assert len(targets) == 750
    assert category_names == {1: "Lung Opacity"}
    assert summary["protocol"]["performs_training"] is False
    assert len(prior_per_seed) == 50
    assert len(prior_aggregate) == 10
    assert prior_summary["analysis_id"] == "rsna-phase42-exact-score-froc-n5-lower-floor-v3"


def test_v4_monotonicity_endpoint_and_residual_floor_assessment() -> None:
    summary = json.loads(
        Path("results/logs/phase42_froc_exact_score_v4/summary.json").read_text(encoding="utf-8")
    )

    assert summary["monotonicity_audit"] == {
        "fp_per_image_failures": 0,
        "runs_checked": 10,
        "sensitivity_failures": 0,
        "status": "pass",
    }
    assert len(summary["run_reachability"]) == 10
    yolo_137 = next(
        row
        for row in summary["run_reachability"]
        if row["detector"] == "yolo11s" and row["seed"] == 137
    )
    assert yolo_137["maximum_reachable_fp_per_image"] == pytest.approx(1.9906666666666666)
    assert [row["status"] for row in yolo_137["budgets"]] == [
        "observed_within_reachable_frontier",
        "observed_within_reachable_frontier",
        "observed_within_reachable_frontier",
        "observed_within_reachable_frontier",
        "floor_limited_lower_bound",
    ]


def test_conservative_bound_proves_two_fp_order_cannot_reverse() -> None:
    summary = json.loads(
        Path("results/logs/phase42_froc_exact_score_v4/summary.json").read_text(encoding="utf-8")
    )
    rows, ordering = conservative_frontier_bounds(
        summary["operating_points"],
        summary["operating_points_per_seed"],
        (0.125, 0.25, 0.5, 1.0, 2.0),
        ("faster_rcnn", "yolo11s"),
        tolerance=1.0e-12,
    )

    yolo_two = next(
        row for row in rows if row["detector"] == "yolo11s" and row["fp_per_image_budget"] == 2.0
    )
    two_order = next(row for row in ordering if row["fp_per_image_budget"] == 2.0)
    assert yolo_two["conservative_upper_aggregate_sensitivity"] == pytest.approx(0.6962686567164178)
    assert two_order["ordering_can_theoretically_reverse"] is False
    assert two_order["minimum_possible_gap_preserving_observed_order"] == pytest.approx(
        0.0014925373134330178
    )
