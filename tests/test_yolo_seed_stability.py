from __future__ import annotations

from src.analyze_yolo_seed_stability import (
    classify_convergence,
    classify_operational,
    load_config,
    summarize_training_curve,
)


def _curve_row(*, loss: float, map_50: float, map_50_95: float) -> dict[str, str]:
    return {
        "train/box_loss": str(loss),
        "train/cls_loss": str(loss),
        "train/dfl_loss": str(loss),
        "metrics/mAP50(B)": str(map_50),
        "metrics/mAP50-95(B)": str(map_50_95),
        "val/box_loss": str(loss),
        "val/cls_loss": str(loss),
        "val/dfl_loss": str(loss),
    }


def test_curve_summary_distinguishes_learning_from_all_zero_loss_collapse() -> None:
    normal = summarize_training_curve(
        [
            _curve_row(loss=1.0, map_50=0.0, map_50_95=0.0),
            _curve_row(loss=0.5, map_50=0.2, map_50_95=0.08),
        ]
    )
    collapsed = summarize_training_curve(
        [
            _curve_row(loss=1.0, map_50=0.0, map_50_95=0.0),
            _curve_row(loss=0.0, map_50=0.0, map_50_95=0.0),
        ]
    )

    assert classify_convergence("complete", normal) == "normal_converged"
    assert normal["positive_val_map_50_95_epoch_count"] == 1
    assert classify_convergence("complete", collapsed) == "collapse_or_invalid"
    assert collapsed["curve_all_zero_loss_epoch_count"] == 1


def test_operational_classification_requires_floor_localization() -> None:
    assert (
        classify_operational(
            convergence="normal_converged",
            frozen_prediction_count=0,
            frozen_true_positives=0,
            floor_tp=12,
        )
        == "confidence_score_degeneracy_at_frozen_threshold"
    )
    assert (
        classify_operational(
            convergence="normal_converged",
            frozen_prediction_count=4,
            frozen_true_positives=0,
            floor_tp=12,
        )
        == "detections_without_iou_qualified_match"
    )


def test_stability_config_freezes_five_seed_contract() -> None:
    config = load_config("configs/yolo_seed_stability.yaml")

    assert config.expected_seeds == (17, 42, 137, 271, 314)
    assert config.expected_confidence_degeneracy_seeds == (271,)
