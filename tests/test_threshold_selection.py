from src.evaluate_threshold_selection import (
    load_threshold_selection_config,
    select_validation_thresholds,
)


def test_threshold_selection_config_matches_batch10_grid() -> None:
    config = load_threshold_selection_config("configs/threshold_selection.yaml")

    thresholds = config.selection.thresholds()
    assert len(thresholds) == 99
    assert thresholds[0] == 0.01
    assert thresholds[-1] == 0.99
    assert 0.25 in thresholds
    assert config.selection.rule == "maximum_mean_f1"
    assert config.selection.tie_breaker == "highest_threshold"


def test_validation_selection_uses_mean_f1_and_declared_tie_breaker() -> None:
    rows = [
        {"detector": "faster_rcnn", "threshold": 0.2, "f1": 0.5},
        {"detector": "faster_rcnn", "threshold": 0.3, "f1": 0.5},
        {"detector": "yolo11s", "threshold": 0.1, "f1": 0.4},
        {"detector": "yolo11s", "threshold": 0.2, "f1": 0.3},
    ]

    selected = select_validation_thresholds(rows)

    assert selected["faster_rcnn"]["threshold"] == 0.3
    assert selected["yolo11s"]["threshold"] == 0.1
