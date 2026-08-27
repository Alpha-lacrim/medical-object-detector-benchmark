from __future__ import annotations

import numpy as np
import pytest

from src.clinical.decision_curve import (
    contiguous_ranges,
    load_decision_curve_config,
    maximum_scores_by_image,
    net_benefit,
    summarize_net_benefit,
    treat_all_net_benefit,
)
from src.stats.threshold_calibration import BootstrapPlan


def test_net_benefit_matches_hand_computed_binary_example() -> None:
    assert net_benefit(2, 1, 4, 0.2) == pytest.approx(0.4375)
    assert treat_all_net_benefit(0.5, 0.2) == pytest.approx(0.375)
    np.testing.assert_allclose(
        net_benefit(
            np.asarray([1, 2]),
            np.asarray([1, 1]),
            np.asarray([4, 4]),
            0.2,
        ),
        [0.1875, 0.4375],
    )

    with pytest.raises(ValueError, match="strictly inside"):
        net_benefit(1, 0, 1, 1.0)
    with pytest.raises(ValueError, match="disjoint"):
        net_benefit(1, 1, 1, 0.5)


def test_exam_score_is_maximum_emitted_box_confidence() -> None:
    records = [
        {"image_id": "a.png", "scores": [0.2, 0.8, 0.4]},
        {"image_id": "b.png", "scores": []},
    ]

    np.testing.assert_allclose(
        maximum_scores_by_image(records, ("b.png", "a.png")),
        [0.0, 0.8],
    )

    with pytest.raises(ValueError, match="grid mismatch"):
        maximum_scores_by_image(records, ("a.png", "c.png"))


def test_hierarchical_summary_averages_seed_net_benefit() -> None:
    predicted = np.asarray(
        [
            [True, True, False, False],
            [True, False, True, True],
        ],
        dtype=np.bool_,
    )
    outcome = np.asarray([True, False, True, False], dtype=np.bool_)
    bootstrap_plan = BootstrapPlan(
        image_multiplicities=np.ones((4, 4), dtype=np.int64),
        seed_multiplicities=np.ones((4, 2), dtype=np.int64),
    )

    result = summarize_net_benefit(
        predicted,
        outcome,
        threshold_probability=0.2,
        bootstrap_plan=bootstrap_plan,
        confidence_level=0.95,
    )

    assert result.net_benefit == pytest.approx(0.3125)
    assert result.ci_lower == pytest.approx(0.3125)
    assert result.ci_upper == pytest.approx(0.3125)
    assert result.mean_true_positives == pytest.approx(1.5)
    assert result.mean_false_positives == pytest.approx(1.0)
    assert result.valid_resamples == 4


def test_contiguous_ranges_collapse_adjacent_threshold_preferences() -> None:
    rows = [
        {"threshold_probability": 0.01, "preferred": "a"},
        {"threshold_probability": 0.02, "preferred": "a"},
        {"threshold_probability": 0.03, "preferred": "b"},
        {"threshold_probability": 0.04, "preferred": "a"},
    ]

    assert contiguous_ranges(rows, key="preferred") == [
        {"start": 0.01, "stop": 0.02, "value": "a"},
        {"start": 0.03, "stop": 0.03, "value": "b"},
        {"start": 0.04, "stop": 0.04, "value": "a"},
    ]


def test_project_config_declares_full_test_dca_contract() -> None:
    config = load_decision_curve_config("configs/decision_curve.yaml")

    assert config.analysis.thresholds() == tuple(
        float(value) for value in np.round(np.linspace(0.01, 0.99, 99), 12)
    )
    assert config.analysis.expected_image_count == 750
    assert config.analysis.detectors == ("faster_rcnn", "yolo11s")
    assert config.analysis.bootstrap_resamples == 2000
    assert config.analysis.decision_unit == "exam_image"
    assert config.analysis.bootstrap_method == ("paired_hierarchical_patient_cluster_percentile")
    assert config.plot.y_min < 0 < config.plot.y_max
