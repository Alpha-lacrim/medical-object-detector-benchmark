import json

import numpy as np
import pytest

from meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    box_iou,
    evaluate_operating_point,
    match_image,
)

SIZE = (10, 10)


def target(image_id: str, boxes=(), labels=()) -> ImageTarget:
    return ImageTarget(image_id, SIZE, boxes, labels)


def prediction(image_id: str, boxes=(), labels=(), scores=()) -> ImagePrediction:
    return ImagePrediction(image_id, SIZE, boxes, labels, scores)


def test_box_iou_uses_continuous_coordinates() -> None:
    actual = box_iou([[0, 0, 2, 2], [2, 2, 4, 4]], [[1, 1, 3, 3], [0, 0, 2, 2]])

    np.testing.assert_allclose(actual, [[1 / 7, 1], [1 / 7, 0]])


def test_duplicate_and_wrong_class_predictions_are_errors() -> None:
    truth = target(
        "case",
        boxes=[[0, 0, 2, 2], [3, 3, 5, 5], [6, 6, 8, 8]],
        labels=[0, 0, 1],
    )
    predicted = prediction(
        "case",
        boxes=[[0, 0, 2, 2], [0, 0, 2, 2], [6, 6, 8, 8], [3, 3, 5, 5]],
        labels=[0, 0, 1, 1],
        scores=[0.9, 0.8, 0.7, 0.6],
    )

    metrics = evaluate_operating_point([predicted], [truth], class_ids=(0, 1), score_threshold=0.25)

    assert metrics["overall"] == {
        "tp": 2,
        "fp": 2,
        "fn": 1,
        "prediction_count": 4,
        "target_count": 3,
        "precision": 0.5,
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(4 / 7),
        "matched_mean_iou": 1.0,
        "matched_mean_box_dice": 1.0,
    }
    json.dumps(metrics, allow_nan=False)


def test_matching_is_score_first_and_stable() -> None:
    truth = target("case", [[0, 0, 2, 2]], [0])
    predicted = prediction(
        "case",
        [[0, 0, 2, 1], [0, 0, 2, 2]],
        [0, 0],
        [0.9, 0.8],
    )

    result = match_image(predicted, truth, score_threshold=0.0)

    assert len(result.matches) == 1
    assert result.matches[0].prediction_index == 0
    assert result.matches[0].iou == 0.5
    assert result.unmatched_prediction_indices == (1,)


def test_per_class_localization_means_do_not_mix_classes() -> None:
    metrics = evaluate_operating_point(
        [
            prediction(
                "case",
                [[0, 0, 2, 1], [4, 4, 6, 6]],
                [0, 1],
                [0.9, 0.8],
            )
        ],
        [target("case", [[0, 0, 2, 2], [4, 4, 6, 6]], [0, 1])],
        class_ids=(0, 1),
        score_threshold=0.25,
    )

    assert metrics["per_class"]["0"]["matched_mean_iou"] == 0.5
    assert metrics["per_class"]["1"]["matched_mean_iou"] == 1.0
    assert metrics["overall"]["matched_mean_iou"] == 0.75


@pytest.mark.parametrize(
    ("predicted", "truth", "expected"),
    [
        (prediction("case"), target("case"), (None, None, None)),
        (prediction("case", [[0, 0, 1, 1]], [0], [1]), target("case"), (0, 0, 0)),
        (prediction("case"), target("case", [[0, 0, 1, 1]], [0]), (0, 0, 0)),
    ],
)
def test_empty_image_policy(predicted, truth, expected) -> None:
    result = evaluate_operating_point([predicted], [truth], class_ids=(0,), score_threshold=0.25)[
        "overall"
    ]

    assert (result["precision"], result["recall"], result["f1"]) == expected
    assert result["matched_mean_iou"] is None


def test_silent_negative_does_not_create_artificial_perfect_score() -> None:
    metrics = evaluate_operating_point(
        [
            prediction("negative"),
            prediction("positive", [[0, 0, 2, 2]], [0], [0.9]),
        ],
        [target("negative"), target("positive", [[0, 0, 2, 2]], [0])],
        class_ids=(0,),
        score_threshold=0.25,
    )

    assert metrics["overall"]["f1"] == 1
    assert metrics["per_image"][0]["f1"] is None
    assert metrics["per_image"][1]["f1"] == 1


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: target("x", [[0, 0, 0, 1]], [0]), "positive area"),
        (lambda: target("x", [[0, 0, 11, 1]], [0]), "inside image"),
        (lambda: target("x", [[0, 0, 1, 1]], [0.5]), "integers"),
        (lambda: prediction("x", [[0, 0, 1, 1]], [0], [1.1]), r"\[0, 1\]"),
    ],
)
def test_malformed_records_are_rejected(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_id_sets_and_known_classes_are_enforced() -> None:
    with pytest.raises(ValueError, match="image_id sets"):
        evaluate_operating_point(
            [prediction("a")], [target("b")], class_ids=(0,), score_threshold=0.25
        )
    with pytest.raises(ValueError, match="outside class_ids"):
        evaluate_operating_point(
            [prediction("a")],
            [target("a", [[0, 0, 1, 1]], [2])],
            class_ids=(0,),
            score_threshold=0.25,
        )
