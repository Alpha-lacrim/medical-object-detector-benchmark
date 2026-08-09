"""Pure training-control and timing helpers for the Faster R-CNN baseline."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EarlyStopObservation:
    """Describe checkpoint and patience decisions for one validation metric."""

    is_new_best: bool
    is_significant_improvement: bool
    should_stop: bool


@dataclass
class EarlyStopper:
    """Track the exact best mAP separately from patience improvements."""

    patience: int
    min_delta: float
    min_epochs: int
    best_metric: float | None = None
    best_epoch: int | None = None
    patience_reference_metric: float | None = None
    patience_reference_epoch: int | None = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.min_epochs <= 0:
            raise ValueError("min_epochs must be positive")

    def observe(self, *, epoch: int, metric: float) -> EarlyStopObservation:
        """Update exact-best and patience state after observing one epoch.

        A strictly higher metric is always an exact new best and should be
        checkpointed. Patience resets only when the metric is strictly more
        than ``min_delta`` above the last significant-improvement reference.
        """

        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if not math.isfinite(metric):
            raise ValueError("early-stopping metric must be finite")

        is_new_best = self.best_metric is None or metric > self.best_metric
        if is_new_best:
            self.best_metric = metric
            self.best_epoch = epoch

        is_significant_improvement = (
            self.patience_reference_metric is None
            or metric > self.patience_reference_metric + self.min_delta
        )
        if is_significant_improvement:
            self.patience_reference_metric = metric
            self.patience_reference_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = (
            epoch >= self.min_epochs
            and self.epochs_without_improvement >= self.patience
        )
        return EarlyStopObservation(
            is_new_best=is_new_best,
            is_significant_improvement=is_significant_improvement,
            should_stop=should_stop,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return JSON/checkpoint-safe state."""

        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "patience_reference_metric": self.patience_reference_metric,
            "patience_reference_epoch": self.patience_reference_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }


def estimate_training_seconds(
    epoch_seconds: list[float],
    *,
    minimum_epochs: int,
    maximum_epochs: int,
) -> dict[str, Any]:
    """Project total time using epoch one plus median steady-state duration.

    The first epoch retains startup/cache costs. Epochs two onward define the
    steady-state estimate and conservative lower/upper bounds.
    """

    if len(epoch_seconds) < 2 or any(
        not math.isfinite(value) or value <= 0 for value in epoch_seconds
    ):
        raise ValueError("at least two positive finite epoch durations are required")
    if minimum_epochs <= 0 or maximum_epochs < minimum_epochs:
        raise ValueError("epoch projection bounds are invalid")

    first = epoch_seconds[0]
    steady_samples = epoch_seconds[1:]
    steady = statistics.median(steady_samples)
    lower_steady = min(steady_samples)
    upper_steady = max(steady_samples)

    def project(epochs: int, per_epoch: float) -> float:
        return first + max(epochs - 1, 0) * per_epoch

    return {
        "method": "first_epoch_plus_median_remaining_epochs",
        "observed_epoch_seconds": epoch_seconds,
        "steady_state_seconds_per_epoch": steady,
        "steady_state_range_seconds_per_epoch": [lower_steady, upper_steady],
        "minimum_epochs": minimum_epochs,
        "maximum_epochs": maximum_epochs,
        "estimated_minimum_seconds": project(minimum_epochs, steady),
        "estimated_maximum_seconds": project(maximum_epochs, steady),
        "estimated_maximum_range_seconds": [
            project(maximum_epochs, lower_steady),
            project(maximum_epochs, upper_steady),
        ],
        "early_stopping_caveat": (
            "The stopping epoch is unknown after the timing benchmark; "
            "the maximum-epoch estimate is the sign-off upper bound."
        ),
    }
