from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget
from src.stats import calibration
from src.stats.calibration import (
    build_detection_calibration_samples,
    detection_expected_calibration_error,
    load_calibration_config,
    reliability_bins,
)


def test_detection_ece_matches_hand_computed_multivariate_example() -> None:
    features = np.asarray([[0.1, 0.1], [0.2, 0.1], [0.7, 0.8], [0.9, 0.8]], dtype=np.float64)
    matched = np.asarray([False, True, True, True])

    result = detection_expected_calibration_error(
        features, matched, bins=(2, 2), minimum_samples_per_cell=1
    )

    assert result["detection_ece"] == pytest.approx(0.275)
    assert result["prediction_count"] == 4
    assert result["detections_in_supported_cells"] == 4
    assert result["occupied_cells"] == 2
    assert result["supported_cells"] == 2
    assert result["total_possible_multidimensional_cells"] == 4
    assert all(isinstance(index, int) for record in result["bins"] for index in record["indices"])


def test_detection_ece_weights_supported_cells_by_all_detections() -> None:
    features = np.asarray(
        [[0.1, 0.1], [0.2, 0.1], [0.7, 0.8], [0.9, 0.8], [0.95, 0.8]],
        dtype=np.float64,
    )
    matched = np.asarray([False, True, True, True, True])

    result = detection_expected_calibration_error(
        features, matched, bins=(2, 2), minimum_samples_per_cell=3
    )

    assert result["detection_ece"] == pytest.approx(0.09)
    assert result["detections_in_supported_cells"] == 3
    assert result["fraction_detections_in_supported_cells"] == pytest.approx(0.6)
    assert result["supported_cells"] == 1
    assert result["supported_cell_size_median"] == 3
    assert result["supported_cell_size_min"] == 3
    assert result["supported_cell_size_max"] == 3


def test_equal_width_binning_edges_use_upper_cell_and_include_one() -> None:
    below_half = np.nextafter(0.5, 0.0)
    features = np.asarray([[0.0], [below_half], [0.5], [1.0]], dtype=np.float64)
    matched = np.asarray([False, False, True, True])

    result = detection_expected_calibration_error(
        features, matched, bins=(2,), minimum_samples_per_cell=1
    )

    assert [record["indices"] for record in result["bins"]] == [[0], [1]]
    assert [record["sample_count"] for record in result["bins"]] == [2, 2]


def test_class_is_a_categorical_cell_stratum() -> None:
    features = np.asarray([[0.2], [0.2]], dtype=np.float64)
    matched = np.asarray([True, False])
    class_ids = np.asarray([4, 9])

    result = detection_expected_calibration_error(
        features,
        matched,
        class_ids=class_ids,
        possible_class_ids=(4, 9),
        bins=(2,),
        minimum_samples_per_cell=1,
    )

    assert result["total_possible_multidimensional_cells"] == 4
    assert result["occupied_cells"] == 2
    assert {record["class_id"] for record in result["bins"]} == {4, 9}
    assert result["detection_ece"] == pytest.approx(0.5)


def test_empty_detection_population_is_reported_as_undefined() -> None:
    result = detection_expected_calibration_error(
        np.empty((0, 2)),
        np.empty((0,), dtype=np.bool_),
        class_ids=np.empty((0,), dtype=np.int64),
        possible_class_ids=(1,),
        bins=(3, 3),
        minimum_samples_per_cell=8,
    )

    assert result["detection_ece"] is None
    assert result["detection_ece_defined"] is False
    assert result["undefined_reason"] == "no_detections_at_confidence_floor"
    assert result["total_possible_multidimensional_cells"] == 9
    assert result["occupied_cells"] == 0
    assert result["supported_cells"] == 0


def test_occupied_but_unsupported_cells_are_undefined_not_zero() -> None:
    result = detection_expected_calibration_error(
        np.asarray([[0.2], [0.8]]),
        np.asarray([True, False]),
        bins=(2,),
        minimum_samples_per_cell=2,
    )

    assert result["detection_ece"] is None
    assert result["detection_ece_defined"] is False
    assert result["undefined_reason"] == "no_occupied_cell_meets_minimum_support"
    assert result["occupied_cells"] == 2
    assert result["supported_cells"] == 0


def _single_image_example(scores: list[float]) -> tuple[ImagePrediction, ImageTarget]:
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
            [[20, 10, 60, 50], [100, 50, 140, 90], [21, 11, 61, 51]],
            dtype=np.float64,
        ),
        labels=np.asarray([1, 1, 1]),
        scores=np.asarray(scores),
    )
    return prediction, target


def test_detection_samples_use_variable_predictions_and_canonical_matching() -> None:
    prediction, target = _single_image_example([0.9, 0.8, 0.7])

    samples = build_detection_calibration_samples(
        [prediction], [target], score_floor=0.0, iou_threshold=0.5, max_detections=2
    )

    np.testing.assert_allclose(
        samples.features,
        np.asarray([[0.9, 0.2, 0.3, 0.2, 0.4], [0.8, 0.6, 0.7, 0.2, 0.4]]),
    )
    np.testing.assert_array_equal(samples.matched, np.asarray([True, False]))
    np.testing.assert_array_equal(samples.class_ids, np.asarray([1, 1]))


def test_score_floor_is_inclusive_and_can_produce_empty_population() -> None:
    prediction, target = _single_image_example([0.001, 0.0009, 0.0008])

    at_floor = build_detection_calibration_samples(
        [prediction], [target], score_floor=0.001, iou_threshold=0.5, max_detections=100
    )
    above_all = build_detection_calibration_samples(
        [prediction], [target], score_floor=0.01, iou_threshold=0.5, max_detections=100
    )

    assert at_floor.features.shape == (1, 5)
    assert at_floor.confidence[0] == pytest.approx(0.001)
    assert at_floor.matched.tolist() == [True]
    assert above_all.features.shape == (0, 5)
    assert above_all.matched.shape == (0,)
    assert above_all.class_ids.shape == (0,)


def test_reliability_bins_return_confidence_only_marginal_points() -> None:
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
    assert reliability_bins(np.asarray([]), np.asarray([], dtype=np.bool_), bins=10) == []


def test_config_predeclares_original_setting_inside_sensitivity_grid() -> None:
    config = load_calibration_config(Path("configs/calibration.yaml"))
    settings = config.calibration

    assert settings.bins_per_dimension[0] in settings.sensitivity.equal_bins_per_dimension
    assert settings.minimum_samples_per_cell in settings.sensitivity.minimum_samples_per_cell
    assert settings.prediction_score_floor in settings.sensitivity.confidence_floors
    assert min(settings.sensitivity.confidence_floors) == settings.prediction_score_floor


def test_calibration_source_has_no_seed_conditionals() -> None:
    tree = ast.parse(inspect.getsource(calibration))
    offending: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        compares = [child for child in ast.walk(node.test) if isinstance(child, ast.Compare)]
        seed_to_numeric_constant = any(
            any(isinstance(child, ast.Name) and child.id == "seed" for child in ast.walk(compare))
            and any(
                isinstance(child, ast.Constant)
                and isinstance(child.value, int)
                and not isinstance(child.value, bool)
                for child in ast.walk(compare)
            )
            for compare in compares
        )
        if seed_to_numeric_constant:
            offending.append(node.lineno)

    assert offending == []
