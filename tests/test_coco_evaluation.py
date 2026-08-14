import json

import pytest

from meddet_benchmark.coco_evaluation import evaluate_coco
from meddet_benchmark.evaluation import ImagePrediction, ImageTarget

SIZE = (20, 20)


def target(image_id: str, boxes=(), labels=()) -> ImageTarget:
    return ImageTarget(image_id, SIZE, boxes, labels)


def prediction(image_id: str, boxes=(), labels=(), scores=()) -> ImagePrediction:
    return ImagePrediction(image_id, SIZE, boxes, labels, scores)


def test_perfect_detection_has_perfect_coco_ap_and_silent_output(capsys) -> None:
    metrics = evaluate_coco(
        [
            prediction("negative"),
            prediction("positive", [[2, 2, 8, 8]], [0], [0.9]),
        ],
        [target("negative"), target("positive", [[2, 2, 8, 8]], [0])],
        class_ids=(0,),
        class_names={0: "lesion"},
    )

    assert metrics["ap50_95"] == pytest.approx(1)
    assert metrics["ap50"] == pytest.approx(1)
    assert metrics["per_class"]["0"]["ap50_95"] == pytest.approx(1)
    assert metrics["annotation_count"] == 1
    assert capsys.readouterr().out == ""
    json.dumps(metrics, allow_nan=False)


def test_official_precision_recall_curves_are_exposed_on_request() -> None:
    metrics = evaluate_coco(
        [prediction("case", [[2, 2, 8, 8]], [0], [0.9])],
        [target("case", [[2, 2, 8, 8]], [0])],
        class_ids=(0,),
        include_precision_recall=True,
    )

    curve = metrics["precision_recall"]
    assert len(curve["recall"]) == 101
    assert curve["recall"][0] == 0
    assert curve["recall"][-1] == 1
    assert curve["precision_iou_50"] == pytest.approx([1] * 101)
    assert curve["precision_iou_50_95"] == pytest.approx([1] * 101)
    assert curve["per_class"]["0"]["precision_iou_50"] == pytest.approx([1] * 101)
    json.dumps(metrics, allow_nan=False)


def test_missed_target_has_zero_ap() -> None:
    metrics = evaluate_coco(
        [prediction("case")],
        [target("case", [[2, 2, 8, 8]], [0])],
        class_ids=(0,),
    )

    assert metrics["ap50_95"] == 0
    assert metrics["ap50"] == 0
    assert metrics["prediction_count"] == 0


def test_class_without_ground_truth_has_null_ap() -> None:
    metrics = evaluate_coco(
        [prediction("case", [[2, 2, 8, 8]], [1], [0.9])],
        [target("case", [[2, 2, 8, 8]], [0])],
        class_ids=(0, 1),
    )

    assert metrics["ap50_95"] == 0
    assert metrics["per_class"]["0"]["ap50_95"] == 0
    assert metrics["per_class"]["1"]["ap50_95"] is None


def test_all_negative_dataset_returns_null_ap() -> None:
    metrics = evaluate_coco(
        [prediction("negative")],
        [target("negative")],
        class_ids=(0,),
    )

    assert metrics["ap50_95"] is None
    assert metrics["ap50"] is None


def test_identity_and_class_contracts_are_enforced() -> None:
    with pytest.raises(ValueError, match="image_id sets"):
        evaluate_coco([prediction("a")], [target("b")], class_ids=(0,))
    with pytest.raises(ValueError, match="outside class_ids"):
        evaluate_coco(
            [prediction("a")],
            [target("a", [[0, 0, 1, 1]], [2])],
            class_ids=(0,),
        )
