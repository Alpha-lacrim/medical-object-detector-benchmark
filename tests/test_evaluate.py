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


def test_phase5_config_declares_complete_five_seed_grid() -> None:
    config = load_phase5_config("configs/evaluation.yaml")

    assert config.seeds == (17, 42, 137, 271, 314)
    assert config.split == "test"
    assert {(run.detector, run.seed) for run in config.runs} == {
        (detector, seed) for detector in ("faster_rcnn", "yolo11s") for seed in config.seeds
    }
    assert len(load_and_validate_training_configs(config)) == 10


def test_archived_n3_config_remains_loadable_for_frozen_downstream_analyses() -> None:
    config = load_phase5_config("configs/evaluation_n3_archive.yaml")

    assert config.seeds == (17, 42, 137)
    assert config.timing_reuse_reviews is None


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


def test_coco_ap_uses_retained_predictions_below_the_operating_threshold() -> None:
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
            boxes_xyxy=[[0, 0, 4, 4]],
            labels=[1],
            scores=[0.01],
        )
    ]
    settings = EvaluationSettings(
        score_threshold=0.25,
        match_iou_threshold=0.5,
        coco_minimum_score=0.001,
        nms_iou_threshold=0.5,
        max_detections=100,
    )

    metrics = evaluate_prediction_records(
        predictions, targets, category_names={1: "opacity"}, settings=settings
    )

    assert metrics["operating_point"]["overall"]["prediction_count"] == 0
    assert metrics["coco"]["prediction_count"] == 1
    assert metrics["coco"]["ap50"] == pytest.approx(1)


def test_aggregation_uses_sample_standard_deviation() -> None:
    rows = []
    for detector, values in (
        ("faster_rcnn", (1.0, 2.0, 3.0)),
        ("yolo11s", (2.0, 4.0, 6.0)),
    ):
        for seed, value in zip((17, 42, 137), values, strict=True):
            row = {"detector": detector, "seed": seed, "true_positives": 1}
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
    assert faster_precision["attempted_n"] == 3
    assert faster_precision["undefined_n"] == 0
    assert comparison[0]["yolo11s_std"] == 2.0
    faster_gflops = next(
        row for row in long_rows if row["detector"] == "faster_rcnn" and row["metric"] == "gflops"
    )
    assert faster_gflops["std"] == 0.0


def _aggregation_row(detector: str, seed: int, value: float) -> dict:
    row = {
        "detector": detector,
        "seed": seed,
        "true_positives": 1,
        "operating_point_prediction_count": 1,
        "conditional_localization_undefined_reason": "",
    }
    row.update({metric: value for metric, _unit in METRICS})
    return row


def test_aggregation_retains_all_attempts_but_uses_defined_conditional_n() -> None:
    seeds = (17, 42, 137, 271, 314)
    rows = [
        _aggregation_row(detector, seed, float(index + 1))
        for detector in ("faster_rcnn", "yolo11s")
        for index, seed in enumerate(seeds)
    ]
    collapsed = next(row for row in rows if row["detector"] == "yolo11s" and row["seed"] == 271)
    collapsed.update(
        {
            "true_positives": 0,
            "operating_point_prediction_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": None,
            "dice": None,
            "conditional_localization_undefined_reason": (
                "no_detections_at_frozen_score_threshold"
            ),
        }
    )

    long_rows, comparison = aggregate_rows(rows)

    yolo = {row["metric"]: row for row in long_rows if row["detector"] == "yolo11s"}
    assert yolo["precision"]["n"] == 5
    assert yolo["precision"]["mean"] == pytest.approx((1 + 2 + 3 + 0 + 5) / 5)
    assert yolo["map_50_95"]["n"] == 5
    assert yolo["iou"]["n"] == 4
    assert yolo["iou"]["attempted_n"] == 5
    assert yolo["iou"]["undefined_n"] == 1
    assert yolo["iou"]["undefined_seeds"] == "271"
    assert yolo["iou"]["undefined_reason"] == "no_detections_at_frozen_score_threshold"
    iou_comparison = next(row for row in comparison if row["metric"] == "iou")
    assert iou_comparison["faster_rcnn_n"] == 5
    assert iou_comparison["yolo11s_n"] == 4
    assert iou_comparison["yolo11s_undefined_seeds"] == "271"
    assert "detector-specific n" in iou_comparison["sample_size_note"]


@pytest.mark.parametrize(
    ("field", "value", "true_positives", "error"),
    (
        ("precision", None, 1, "nonconditional metric is undefined"),
        ("iou", None, 1, "cannot be undefined with true positives"),
        ("iou", 0.5, 0, "must be undefined with zero true positives"),
    ),
)
def test_aggregation_rejects_invalid_missingness_contract(
    field: str, value: float | None, true_positives: int, error: str
) -> None:
    rows = [
        _aggregation_row(detector, seed, 0.5)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42)
    ]
    target = rows[-1]
    target["true_positives"] = true_positives
    target[field] = value
    if field == "iou" and value is None:
        target["dice"] = None
        target["conditional_localization_undefined_reason"] = (
            "no_detections_at_frozen_score_threshold"
        )
    if field == "iou" and true_positives == 0:
        target["dice"] = 0.6

    with pytest.raises(ValueError, match=error):
        aggregate_rows(rows)


def test_aggregation_rejects_partially_defined_conditional_metrics() -> None:
    rows = [
        _aggregation_row(detector, seed, 0.5)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42)
    ]
    rows[-1]["iou"] = None

    with pytest.raises(ValueError, match="jointly defined"):
        aggregate_rows(rows)


def test_aggregation_rejects_undefined_reason_that_disagrees_with_counts() -> None:
    rows = [
        _aggregation_row(detector, seed, 0.5)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42)
    ]
    rows[-1].update(
        {
            "true_positives": 0,
            "operating_point_prediction_count": 0,
            "iou": None,
            "dice": None,
            "conditional_localization_undefined_reason": "no_iou_qualified_true_positive",
        }
    )

    with pytest.raises(ValueError, match="reason differs from counts"):
        aggregate_rows(rows)
