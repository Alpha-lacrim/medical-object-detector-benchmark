import pytest

from src.evaluate_threshold_sweep import (
    aggregate_threshold_rows,
    load_threshold_sweep_config,
    select_operating_targets,
    sweep_prediction_records,
)
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget


def test_threshold_config_declares_requested_grid_and_outputs() -> None:
    config = load_threshold_sweep_config("configs/threshold_sweep.yaml")

    thresholds = config.sweep.thresholds()
    assert len(thresholds) == 99
    assert thresholds[0] == 0.01
    assert thresholds[-1] == 0.99
    assert 0.25 in thresholds
    assert config.outputs.threshold_table.as_posix() == "results/tables/threshold_sweep.csv"


def test_sweep_uses_canonical_score_ordered_matching() -> None:
    target = ImageTarget("case", (20, 20), [[0, 0, 10, 10]], [1])
    prediction = ImagePrediction(
        "case",
        (20, 20),
        [[0, 0, 4, 4], [0, 0, 10, 10]],
        [1, 1],
        [0.9, 0.4],
    )

    rows = sweep_prediction_records(
        [prediction],
        [target],
        detector="faster_rcnn",
        seed=17,
        thresholds=(0.25, 0.5),
        class_ids=(1,),
        iou_threshold=0.5,
        max_detections=100,
    )

    assert rows[0]["true_positives"] == 1
    assert rows[0]["false_positives"] == 1
    assert rows[0]["precision"] == 0.5
    assert rows[0]["recall"] == 1
    assert rows[1]["true_positives"] == 0
    assert rows[1]["precision"] == 0
    assert rows[1]["recall"] == 0


def test_threshold_aggregation_uses_sample_standard_deviation() -> None:
    rows = [
        {
            "detector": "faster_rcnn",
            "seed": seed,
            "threshold": 0.25,
            "precision": value,
            "recall": value / 2,
            "f1": value / 3,
        }
        for seed, value in zip((17, 42, 137), (0.2, 0.4, 0.6), strict=True)
    ]

    aggregate = aggregate_threshold_rows(rows, ddof=1)[0]

    assert aggregate["precision"] == pytest.approx(0.4)
    assert aggregate["precision_std"] == pytest.approx(0.2)
    assert aggregate["seed_count"] == 3


def test_fixed_targets_do_not_interpolate_or_extrapolate() -> None:
    aggregate = [
        {
            "detector": "faster_rcnn",
            "threshold": threshold,
            "seed_count": 3,
            "precision": precision,
            "precision_std": 0.01,
            "recall": recall,
            "recall_std": 0.02,
            "f1": 0.2,
            "f1_std": 0.01,
        }
        for threshold, precision, recall in (
            (0.1, 0.4, 0.8),
            (0.2, 0.6, 0.5),
            (0.3, 0.8, 0.2),
        )
    ]
    per_seed = [
        {**row, "seed": seed}
        for seed in (17, 42, 137)
        for row in aggregate
    ]

    rows = select_operating_targets(
        aggregate,
        per_seed,
        precision_targets=(0.5, 0.9),
        recall_targets=(0.5,),
    )

    precision_05 = next(
        row
        for row in rows
        if row["target_metric"] == "precision" and row["target_value"] == 0.5
    )
    precision_09 = next(
        row
        for row in rows
        if row["target_metric"] == "precision" and row["target_value"] == 0.9
    )
    fixed_recall = next(row for row in rows if row["target_metric"] == "recall")
    assert precision_05["threshold"] == 0.2
    assert precision_05["response"] == 0.5
    assert precision_09["status"] == "not_reachable_at_any_swept_threshold"
    assert precision_09["response"] is None
    assert fixed_recall["threshold"] == 0.2
    assert fixed_recall["response"] == 0.6
