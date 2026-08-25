from __future__ import annotations

import numpy as np
import pytest

from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget
from src.stats.calibration import (
    build_detection_calibration_samples,
    detection_expected_calibration_error,
    reliability_bins,
)


def test_detection_ece_matches_hand_computed_multivariate_example() -> None:
    features = np.asarray(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.7, 0.8],
            [0.9, 0.8],
        ],
        dtype=np.float64,
    )
    matched = np.asarray([False, True, True, True])

    result = detection_expected_calibration_error(
        features, matched, bins=(2, 2), minimum_samples_per_bin=1
    )

    assert result["detection_ece"] == pytest.approx(0.275)
    assert result["prediction_count"] == 4
    assert result["included_prediction_count"] == 4
    assert result["nonempty_multivariate_bins"] == 2
    assert all(isinstance(index, int) for record in result["bins"] for index in record["indices"])


def test_detection_ece_reports_samples_below_paper_threshold() -> None:
    features = np.asarray(
        [[0.1, 0.1], [0.2, 0.1], [0.7, 0.8], [0.9, 0.8], [0.95, 0.8]],
        dtype=np.float64,
    )
    matched = np.asarray([False, True, True, True, True])

    result = detection_expected_calibration_error(
        features, matched, bins=(2, 2), minimum_samples_per_bin=3
    )

    assert result["detection_ece"] == pytest.approx(0.09)
    assert result["included_prediction_count"] == 3
    assert result["included_prediction_fraction"] == pytest.approx(0.6)
    assert result["included_multivariate_bins"] == 1


def test_detection_samples_use_variable_predictions_and_canonical_matching() -> None:
    target = ImageTarget(
        image_id="image.png",
        image_size=(100, 200),
        boxes_xyxy=np.asarray([[20, 10, 60, 50]], dtype=np.float64),
        labels=np.asarray([1]),
    )
    prediction = ImagePrediction(
        image_id="image.png",
        image_size=(100, 200),
        boxes_xyxy=np.asarray(
            [[20, 10, 60, 50], [100, 50, 140, 90], [21, 11, 61, 51]], dtype=np.float64
        ),
        labels=np.asarray([1, 1, 1]),
        scores=np.asarray([0.9, 0.8, 0.7]),
    )

    samples = build_detection_calibration_samples(
        [prediction], [target], score_floor=0.0, iou_threshold=0.5, max_detections=2
    )

    np.testing.assert_allclose(
        samples.features,
        np.asarray(
            [
                [0.9, 0.2, 0.3, 0.2, 0.4],
                [0.8, 0.6, 0.7, 0.2, 0.4],
            ]
        ),
    )
    np.testing.assert_array_equal(samples.matched, np.asarray([True, False]))


def test_reliability_bins_return_mean_score_and_true_positive_fraction() -> None:
    rows = reliability_bins(
        np.asarray([0.05, 0.15, 0.55, 1.0]),
        np.asarray([False, True, True, True]),
        bins=2,
    )

    assert rows == [
        {
            "bin_index": 0,
            "lower_bound": 0.0,
            "upper_bound": 0.5,
            "sample_count": 2,
            "mean_confidence": pytest.approx(0.1),
            "fraction_true_positives": pytest.approx(0.5),
        },
        {
            "bin_index": 1,
            "lower_bound": 0.5,
            "upper_bound": 1.0,
            "sample_count": 2,
            "mean_confidence": pytest.approx(0.775),
            "fraction_true_positives": pytest.approx(1.0),
        },
    ]
