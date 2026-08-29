from __future__ import annotations

import inspect

import numpy as np
import pytest

import src.stats.threshold_calibration as threshold_calibration
from src.stats.threshold_calibration import (
    SUMMARY_FIELDS,
    BootstrapPlan,
    f_beta,
    load_threshold_calibration_config,
    select_calibrated_thresholds,
    select_hypothetical_loss_thresholds,
    summarize_hypothetical_detection_error_loss,
    summarize_threshold_counts,
    threshold_stability_diagnostics,
    validate_validation_only_selection_contract,
)


def test_f_beta_matches_recall_weighted_harmonic_mean() -> None:
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


def test_f_beta_is_not_silently_labeled_as_clinical_harm() -> None:
    source = inspect.getsource(f_beta).lower()
    module_source = inspect.getsource(threshold_calibration).lower()

    assert "c_fn" not in source
    assert "c_fp" not in source
    assert "cost" not in source
    assert "recall_to_precision_weight" in SUMMARY_FIELDS
    assert "false_negative_to_false_positive_cost_ratio" not in SUMMARY_FIELDS
    assert "cost_ratio_definition" not in module_source
    assert "c_fn" not in module_source
    assert "c_fp" not in module_source
    assert "false_negative_to_false_positive_cost_ratio" not in module_source
    assert '"beta_is_measured_clinical_harm_ratio": false' in module_source

    # Equal linear loss at r=4 need not give equal F_2, so beta^2 is not a
    # drop-in empirical loss ratio. Both examples have 10 positive targets.
    threshold_a = f_beta(10 / 18, 1.0, 2.0)  # FN=0, FP=8
    threshold_b = f_beta(1.0, 0.8, 2.0)  # FN=2, FP=0
    assert 4 * 0 + 8 == 4 * 2 + 0
    assert threshold_a > threshold_b


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


def test_threshold_stability_reports_plateau_and_selection_frequencies() -> None:
    curve_rows = [
        {"detector": "model", "beta": 1.0, "threshold": 0.1, "f_beta_ci_lower": 0.49},
        {"detector": "model", "beta": 1.0, "threshold": 0.2, "f_beta_ci_lower": 0.50},
        {"detector": "model", "beta": 1.0, "threshold": 0.3, "f_beta_ci_lower": 0.495},
    ]
    distributions = {
        ("model", 1.0, 0.1): np.asarray([0.9, 0.2, 0.3, 0.5]),
        ("model", 1.0, 0.2): np.asarray([0.8, 0.7, 0.3, 0.5]),
        ("model", 1.0, 0.3): np.asarray([0.1, 0.6, 0.4, 0.5]),
    }

    rows, summaries = threshold_stability_diagnostics(
        curve_rows,
        distributions,
        near_optimal_absolute_tolerance=0.01,
        confidence_level=0.95,
        bootstrap_selection_rule="maximum_bootstrap_mean_f_beta",
        tie_breaker="highest_threshold",
    )

    summary = summaries[("model", 1.0)]
    assert summary["near_optimal_plateau_start"] == pytest.approx(0.1)
    assert summary["near_optimal_plateau_end"] == pytest.approx(0.3)
    assert summary["near_optimal_plateau_width"] == pytest.approx(0.2)
    assert summary["near_optimal_plateau_candidate_count"] == 3
    assert summary["bootstrap_modal_selected_tau"] == pytest.approx(0.3)
    assert summary["bootstrap_modal_selection_frequency"] == pytest.approx(0.5)
    assert summary["canonical_tau_bootstrap_selection_frequency"] == pytest.approx(0.25)
    assert sum(row["bootstrap_selection_count"] for row in rows) == 4
    assert sum(row["bootstrap_selection_frequency"] for row in rows) == pytest.approx(1.0)


def test_hypothetical_detection_error_loss_is_separate_and_linear() -> None:
    tp = np.asarray([[1, 0]], dtype=np.int64)
    fp = np.asarray([[0, 2]], dtype=np.int64)
    fn = np.asarray([[0, 1]], dtype=np.int64)
    plan = BootstrapPlan(
        image_multiplicities=np.asarray([[1, 1], [1, 1]], dtype=np.int64),
        seed_multiplicities=np.ones((2, 1), dtype=np.int64),
    )

    result = summarize_hypothetical_detection_error_loss(
        tp,
        fp,
        fn,
        fn_to_fp_loss_ratio=3.0,
        bootstrap_plan=plan,
        confidence_level=0.95,
    )

    assert result["false_negatives_per_image"] == pytest.approx(0.5)
    assert result["false_positives_per_image"] == pytest.approx(1.0)
    assert result["hypothetical_loss_per_image"] == pytest.approx(2.5)
    assert result["loss_ci_lower"] == pytest.approx(2.5)
    assert result["loss_ci_upper"] == pytest.approx(2.5)

    selected = select_hypothetical_loss_thresholds(
        [
            {
                "detector": "model",
                "hypothetical_fn_to_fp_loss_ratio": 3.0,
                "threshold": 0.1,
                "hypothetical_loss_per_image": 1.0,
            },
            {
                "detector": "model",
                "hypothetical_fn_to_fp_loss_ratio": 3.0,
                "threshold": 0.2,
                "hypothetical_loss_per_image": 1.0,
            },
        ]
    )
    assert selected[0]["threshold"] == pytest.approx(0.2)


def test_validation_only_contract_rejects_test_labels_and_partition_mismatch() -> None:
    common = {
        "selection_data_role": "model_development_validation",
        "upstream_split_name": "val",
        "expected_validation_split_name": "val",
        "test_split_accessed": False,
        "annotation_image_ids": ("a", "b"),
        "split_image_ids": ("b", "a"),
    }
    validate_validation_only_selection_contract(**common)

    with pytest.raises(ValueError, match="test isolation"):
        validate_validation_only_selection_contract(**{**common, "test_split_accessed": True})
    with pytest.raises(ValueError, match="different images"):
        validate_validation_only_selection_contract(
            **{**common, "split_image_ids": ("test-a", "test-b")}
        )
    with pytest.raises(ValueError, match="declared validation split"):
        validate_validation_only_selection_contract(**{**common, "upstream_split_name": "test"})


def test_project_config_declares_complete_preference_and_loss_sweeps() -> None:
    config = load_threshold_calibration_config("configs/threshold_calibration.yaml")

    assert config.analysis.thresholds() == tuple(
        float(value) for value in np.round(np.linspace(0.01, 0.99, 99), 12)
    )
    assert config.analysis.beta_values == (1.0, 3.0, 5.0, 10.0)
    assert config.analysis.bootstrap_method == ("paired_hierarchical_patient_cluster_percentile")
    assert config.analysis.bootstrap_resamples == 2000
    assert config.analysis.bootstrap_stream_label == (
        "rsna-phase19-patient-cluster-threshold-calibration"
    )
    assert config.analysis.near_optimal_absolute_tolerance == pytest.approx(0.01)
    assert config.inputs.selection_data_role == "model_development_validation"
    assert config.hypothetical_loss.fn_to_fp_loss_ratios == (1.0, 9.0, 25.0, 100.0)
    assert config.hypothetical_loss.normalization_unit == "validation_image"
