import pytest

from src.evaluate import (
    METRICS,
    EvaluationSettings,
    aggregate_rows,
    evaluate_prediction_records,
    load_and_validate_training_configs,
    load_phase5_config,
)
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget


def test_phase5_config_declares_complete_three_seed_grid() -> None:
    config = load_phase5_config("configs/evaluation.yaml")

    assert config.seeds == (17, 42, 137)
    assert config.split == "test"
    assert {(run.detector, run.seed) for run in config.runs} == {
        (detector, seed) for detector in ("faster_rcnn", "yolo11s") for seed in config.seeds
    }
    assert len(load_and_validate_training_configs(config)) == 6


def test_unified_metrics_include_conditional_iou_and_dice() -> None:
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
            boxes_xyxy=[[0, 0, 4, 2]],
            labels=[1],
            scores=[0.9],
        )
    ]
    settings = EvaluationSettings(
        score_threshold=0.25,
        match_iou_threshold=0.5,
        coco_minimum_score=0.001,
        nms_iou_threshold=0.5,
        max_detections=100,
    )

    first = evaluate_prediction_records(
        predictions, targets, category_names={1: "opacity"}, settings=settings
    )
    second = evaluate_prediction_records(
        predictions, targets, category_names={1: "opacity"}, settings=settings
    )

    assert first == second
    assert first["operating_point"]["overall"]["matched_mean_iou"] == 0.5
    assert first["operating_point"]["overall"]["matched_mean_box_dice"] == pytest.approx(2 / 3)
    assert first["coco"]["ap50"] == pytest.approx(1)


def test_aggregation_uses_sample_standard_deviation() -> None:
    rows = []
    for detector, values in (
        ("faster_rcnn", (1.0, 2.0, 3.0)),
        ("yolo11s", (2.0, 4.0, 6.0)),
    ):
        for seed, value in zip((17, 42, 137), values, strict=True):
            row = {"detector": detector, "seed": seed}
            row.update({metric: value for metric, _unit in METRICS})
            row["gflops"] = 5.0
            rows.append(row)

    long_rows, comparison = aggregate_rows(rows, ddof=1)

    faster_precision = next(
        row
        for row in long_rows
        if row["detector"] == "faster_rcnn" and row["metric"] == "precision"
    )
    assert faster_precision["mean"] == 2.0
    assert faster_precision["std"] == 1.0
    assert faster_precision["n"] == 3
    assert comparison[0]["yolo11s_std"] == 2.0
    faster_gflops = next(
        row for row in long_rows if row["detector"] == "faster_rcnn" and row["metric"] == "gflops"
    )
    assert faster_gflops["std"] == 0.0
