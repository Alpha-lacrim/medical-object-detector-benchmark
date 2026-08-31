from src.analyze_operating_regime_sensitivity import (
    build_fixed_threshold_tables,
    classify_margin_change,
    load_sensitivity_config,
)


def test_config_declares_historical_and_all_attempt_seed_grids() -> None:
    config = load_sensitivity_config("configs/operating_regime_n5_sensitivity.yaml")

    assert config.analysis.expected_historical_seeds == (17, 42, 137)
    assert config.analysis.expected_sensitivity_seeds == (17, 42, 137, 271, 314)
    assert config.analysis.influence_seed == 271


def test_fixed_threshold_builder_does_not_filter_zero_detection_run() -> None:
    historical = [
        {
            "detector": "yolo11s",
            "selection_split": "validation",
            "selection_rule": "maximum arithmetic mean F1 across three validation seeds",
            "tie_breaker": "highest threshold among exact mean-F1 ties",
            "selected_threshold": "0.05",
            "seed_count": "3",
            "validation_precision": "0.4",
            "validation_precision_std": "0.01",
            "validation_recall": "0.3",
            "validation_recall_std": "0.02",
            "validation_f1": "0.34",
            "validation_f1_std": "0.01",
        }
    ]
    per_seed = [
        {
            "detector": "yolo11s",
            "seed": str(seed),
            "threshold": "0.05",
            "precision": "0" if seed == 271 else "0.3",
            "recall": "0" if seed == 271 else "0.2",
            "f1": "0" if seed == 271 else "0.24",
            "true_positives": "0" if seed == 271 else "2",
            "false_positives": "0" if seed == 271 else "4",
            "false_negatives": "10" if seed == 271 else "8",
            "prediction_count": "0" if seed == 271 else "6",
            "target_count": "10",
        }
        for seed in (17, 42, 137, 271, 314)
    ]

    aggregate, rows = build_fixed_threshold_tables(
        historical,
        per_seed,
        expected_test_seeds=(17, 42, 137, 271, 314),
        tolerance=1e-12,
    )

    assert len(rows) == 5
    seed_271 = next(row for row in rows if row["seed"] == 271)
    assert seed_271["test_prediction_count"] == 0
    assert seed_271["test_precision"] == 0
    assert aggregate[0]["test_run_count"] == 5
    assert aggregate[0]["test_precision"] == 0.24


def test_conclusion_change_classification_reports_adverse_changes() -> None:
    assert classify_margin_change(0.2, 0.3, tolerance=1e-12) == "strengthened"
    assert classify_margin_change(0.3, 0.2, tolerance=1e-12) == "weakened"
    assert classify_margin_change(0.2, -0.1, tolerance=1e-12) == "reversed"
    assert classify_margin_change(0.2, 0.2, tolerance=1e-12) == "unchanged"
