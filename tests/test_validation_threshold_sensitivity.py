from pathlib import Path

import pytest

from src.analyze_validation_threshold_sensitivity import (
    classify_margin_change,
    load_sensitivity_config,
    require_validation_selection,
    validate_output_isolation,
    validate_run_grid,
)

CONFIG = Path("configs/threshold_selection_n5_validation_sensitivity.yaml")


def test_config_preserves_historical_rule_and_declares_posthoc_scope() -> None:
    config = load_sensitivity_config(CONFIG)

    assert config.scope.label == "post-hoc n=5 validation threshold-selection sensitivity"
    assert config.scope.selection_split == "validation"
    assert config.scope.historical_seeds == (17, 42, 137)
    assert config.scope.sensitivity_seeds == (17, 42, 137, 271, 314)
    assert config.scope.inference_seeds == (271, 314)
    assert config.scope.influence_seed == 271
    assert config.selection.thresholds() == tuple(value / 100 for value in range(1, 100))
    assert config.selection.rule == "maximum_mean_f1"
    assert config.selection.averaging == "equal_run_arithmetic_mean"
    assert config.selection.tie_breaker == "highest_threshold"


def test_test_set_threshold_selection_is_rejected() -> None:
    require_validation_selection("validation")
    with pytest.raises(ValueError, match="validation evidence only"):
        require_validation_selection("test")


def test_seed_271_exclusion_is_rejected() -> None:
    records = [
        {"detector": detector, "seed": seed}
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42, 137, 314)
    ]

    with pytest.raises(ValueError, match="271"):
        validate_run_grid(
            records,
            detectors=("faster_rcnn", "yolo11s"),
            seeds=(17, 42, 137, 271, 314),
            label="synthetic sensitivity",
        )


def test_historical_artifact_overwrite_is_rejected() -> None:
    config = load_sensitivity_config(CONFIG)
    unsafe_outputs = config.outputs.model_copy(
        update={"summary_json": Path("results/logs/phase14_threshold_selection/summary.json")}
    )
    unsafe = config.model_copy(update={"outputs": unsafe_outputs})

    with pytest.raises(ValueError, match="protected historical path"):
        validate_output_isolation(unsafe)


def test_margin_change_classification_covers_all_outcomes() -> None:
    tolerance = 1e-12
    assert classify_margin_change(0.2, 0.2, tolerance=tolerance) == "unchanged"
    assert classify_margin_change(0.2, 0.3, tolerance=tolerance) == "strengthened"
    assert classify_margin_change(0.3, 0.2, tolerance=tolerance) == "weakened"
    assert classify_margin_change(0.2, -0.1, tolerance=tolerance) == "reversed"
