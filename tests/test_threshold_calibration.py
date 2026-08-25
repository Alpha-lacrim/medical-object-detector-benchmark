from __future__ import annotations

import numpy as np
import pytest

from src.stats.threshold_calibration import (
    BootstrapPlan,
    f_beta,
    load_threshold_calibration_config,
    select_calibrated_thresholds,
    summarize_threshold_counts,
)


def test_f_beta_matches_cost_weighted_formula() -> None:
    assert f_beta(0.5, 0.25, 2.0) == pytest.approx(0.625 / 2.25)
    assert f_beta(0.0, 0.0, 1.0) == 0.0
    np.testing.assert_allclose(
        f_beta(
            np.asarray([0.5, 0.0]),
            np.asarray([0.25, 0.0]),
            2.0,
        ),
        [0.625 / 2.25, 0.0],
    )

    with pytest.raises(ValueError, match="positive"):
        f_beta(0.5, 0.5, 0.0)


def test_selection_optimizes_lower_bound_and_breaks_ties_high() -> None:
    rows = [
        {
            "detector": "model",
            "beta": 1.0,
            "threshold": 0.1,
            "f_beta": 0.9,
            "f_beta_ci_lower": 0.2,
        },
        {
            "detector": "model",
            "beta": 1.0,
            "threshold": 0.2,
            "f_beta": 0.5,
            "f_beta_ci_lower": 0.3,
        },
        {
            "detector": "model",
            "beta": 1.0,
            "threshold": 0.3,
            "f_beta": 0.4,
            "f_beta_ci_lower": 0.3,
        },
    ]

    selected = select_calibrated_thresholds(rows)

    assert len(selected) == 1
    assert selected[0]["threshold"] == 0.3


def test_threshold_summary_averages_seed_metrics_and_reuses_common_draws() -> None:
    tp = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    fp = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    fn = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    plan = BootstrapPlan(
        image_multiplicities=np.asarray([[1, 1], [2, 0], [0, 2], [1, 1]], dtype=np.int64),
        seed_multiplicities=np.ones((4, 2), dtype=np.int64),
    )

    result = summarize_threshold_counts(
        tp,
        fp,
        fn,
        beta=1.0,
        bootstrap_plan=plan,
        confidence_level=0.95,
    )

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f_beta"] == pytest.approx(0.5)
    assert result["f_beta_ci_lower"] == pytest.approx(0.5)
    assert result["f_beta_ci_upper"] == pytest.approx(0.5)
    assert result["bootstrap_valid_resamples"] == 4


def test_project_config_declares_complete_cost_sensitivity_sweep() -> None:
    config = load_threshold_calibration_config("configs/threshold_calibration.yaml")

    assert config.analysis.thresholds() == tuple(
        float(value) for value in np.round(np.linspace(0.01, 0.99, 99), 12)
    )
    assert config.analysis.beta_values == (1.0, 3.0, 5.0, 10.0)
    assert config.analysis.bootstrap_method == ("paired_hierarchical_patient_cluster_percentile")
    assert config.analysis.bootstrap_resamples == 2000
