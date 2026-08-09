from __future__ import annotations

import math

import pytest

from src.models.faster_rcnn_training import (
    EarlyStopObservation,
    EarlyStopper,
    estimate_training_seconds,
)


def test_early_stopper_records_initial_and_later_improvements() -> None:
    stopper = EarlyStopper(patience=3, min_delta=0.01, min_epochs=2)

    assert stopper.observe(epoch=1, metric=0.40) == EarlyStopObservation(
        is_new_best=True,
        is_significant_improvement=True,
        should_stop=False,
    )
    assert stopper.observe(epoch=2, metric=0.42) == EarlyStopObservation(
        is_new_best=True,
        is_significant_improvement=True,
        should_stop=False,
    )
    assert stopper.best_metric == pytest.approx(0.42)
    assert stopper.best_epoch == 2
    assert stopper.epochs_without_improvement == 0


def test_early_stopper_tie_is_not_an_improvement() -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.0, min_epochs=1)

    assert stopper.observe(epoch=1, metric=0.5).is_new_best
    assert stopper.observe(epoch=2, metric=0.5) == EarlyStopObservation(
        is_new_best=False,
        is_significant_improvement=False,
        should_stop=False,
    )
    assert stopper.best_metric == 0.5
    assert stopper.best_epoch == 1
    assert stopper.epochs_without_improvement == 1


def test_early_stopper_requires_improvement_strictly_above_min_delta() -> None:
    stopper = EarlyStopper(patience=3, min_delta=0.125, min_epochs=1)

    assert stopper.observe(epoch=1, metric=0.5).is_significant_improvement
    assert stopper.observe(epoch=2, metric=0.625) == EarlyStopObservation(
        is_new_best=True,
        is_significant_improvement=False,
        should_stop=False,
    )
    assert stopper.observe(epoch=3, metric=0.626) == EarlyStopObservation(
        is_new_best=True,
        is_significant_improvement=True,
        should_stop=False,
    )
    assert stopper.best_metric == pytest.approx(0.626)
    assert stopper.best_epoch == 3
    assert stopper.epochs_without_improvement == 0


def test_sub_delta_new_best_updates_exact_best_without_resetting_patience() -> None:
    stopper = EarlyStopper(patience=3, min_delta=0.1, min_epochs=1)

    stopper.observe(epoch=1, metric=0.5)
    observation = stopper.observe(epoch=2, metric=0.55)

    assert observation == EarlyStopObservation(
        is_new_best=True,
        is_significant_improvement=False,
        should_stop=False,
    )
    assert stopper.best_metric == pytest.approx(0.55)
    assert stopper.best_epoch == 2
    assert stopper.patience_reference_metric == pytest.approx(0.5)
    assert stopper.patience_reference_epoch == 1
    assert stopper.epochs_without_improvement == 1


def test_early_stopper_stops_at_patience_boundary() -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.0, min_epochs=1)

    assert not stopper.observe(epoch=1, metric=0.5).should_stop
    assert not stopper.observe(epoch=2, metric=0.4).should_stop
    assert stopper.observe(epoch=3, metric=0.4).should_stop
    assert stopper.epochs_without_improvement == 2


@pytest.mark.parametrize("metric", [math.nan, math.inf, -math.inf])
def test_early_stopper_rejects_nonfinite_metrics(metric: float) -> None:
    stopper = EarlyStopper(patience=2, min_delta=0.0, min_epochs=1)

    with pytest.raises(ValueError, match="must be finite"):
        stopper.observe(epoch=1, metric=metric)


def test_training_projection_matches_hand_calculation_for_three_epochs() -> None:
    projection = estimate_training_seconds(
        [12.0, 8.0, 10.0],
        minimum_epochs=3,
        maximum_epochs=5,
    )

    assert projection["method"] == "first_epoch_plus_median_remaining_epochs"
    assert projection["observed_epoch_seconds"] == [12.0, 8.0, 10.0]
    assert projection["steady_state_seconds_per_epoch"] == 9.0
    assert projection["steady_state_range_seconds_per_epoch"] == [8.0, 10.0]
    assert projection["estimated_minimum_seconds"] == 30.0
    assert projection["estimated_maximum_seconds"] == 48.0
    assert projection["estimated_maximum_range_seconds"] == [44.0, 52.0]


def test_training_projection_preserves_first_epoch_startup_cost() -> None:
    projection = estimate_training_seconds(
        [20.0, 10.0],
        minimum_epochs=1,
        maximum_epochs=4,
    )

    assert projection["steady_state_seconds_per_epoch"] == 10.0
    assert projection["estimated_minimum_seconds"] == 20.0
    assert projection["estimated_maximum_seconds"] == 50.0
    assert projection["estimated_maximum_range_seconds"] == [50.0, 50.0]
