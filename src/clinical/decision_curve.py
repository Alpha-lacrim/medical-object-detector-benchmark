"""Conventional decision-curve primitives with an explicit probability-scale gate.

This module intentionally does not provide an end-to-end project analysis. The
repository has no validation-frozen probability mapping for every retained test
run. Its purpose is to make the conventional DCA contract executable: model
actions must be derived from calibrated predicted outcome probabilities on the
same scale as the threshold probability used in the harm weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _validated_unit_interval(values: object, *, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite one-dimensional vector")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{label} must lie in [0, 1]")
    return array


@dataclass(frozen=True)
class CalibratedOutcomeProbabilities:
    """Predicted exam-level outcome probabilities from a frozen calibrator."""

    values: FloatArray
    validation_frozen_mapping_id: str

    def __post_init__(self) -> None:
        validated = _validated_unit_interval(self.values, label="calibrated outcome probabilities")
        if not self.validation_frozen_mapping_id.strip():
            raise ValueError("validation_frozen_mapping_id must be non-empty")
        object.__setattr__(self, "values", validated)


@dataclass(frozen=True)
class RawDetectorConfidenceScores:
    """Uncalibrated detector confidences, which are ineligible for standard DCA."""

    values: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _validated_unit_interval(self.values, label="raw detector confidence scores"),
        )


@dataclass(frozen=True)
class DecisionThresholdProbability:
    """Elicited decision threshold on the predicted-outcome probability scale."""

    value: float
    decision_context_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validated_threshold_probability(self.value))
        if not self.decision_context_id.strip():
            raise ValueError("decision_context_id must be non-empty")


@dataclass(frozen=True)
class RawDetectorConfidenceThreshold:
    """Raw score cutoff, which cannot serve directly as a standard-DCA p_t."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validated_threshold_probability(self.value))


def _validated_threshold_probability(threshold_probability: float) -> float:
    value = float(threshold_probability)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("threshold_probability must lie strictly inside (0, 1)")
    return value


def _require_decision_threshold_probability(
    threshold_probability: DecisionThresholdProbability,
) -> float:
    if not isinstance(threshold_probability, DecisionThresholdProbability):
        raise TypeError(
            "standard DCA requires an elicited DecisionThresholdProbability; "
            "raw detector confidence cannot be passed directly as p_t"
        )
    return threshold_probability.value


def standard_dca_actions(
    predicted_probabilities: CalibratedOutcomeProbabilities,
    *,
    threshold_probability: DecisionThresholdProbability,
) -> BoolArray:
    """Threshold calibrated outcome probabilities for conventional DCA.

    A raw detector-confidence vector or a plain numeric array is rejected even
    when its values happen to lie in [0, 1]. The threshold must separately be
    typed as an elicited decision threshold; a raw confidence cutoff or plain
    scalar cannot be passed directly as p_t. Numeric range alone establishes
    neither probability semantics nor a harm/benefit trade-off.
    """

    if not isinstance(predicted_probabilities, CalibratedOutcomeProbabilities):
        raise TypeError(
            "standard DCA requires CalibratedOutcomeProbabilities from a "
            "validation-frozen mapping; raw detector confidence is not eligible"
        )
    threshold = _require_decision_threshold_probability(threshold_probability)
    return predicted_probabilities.values >= threshold


def standard_dca_net_benefit(
    predicted_probabilities: CalibratedOutcomeProbabilities,
    outcome_positive: BoolArray,
    *,
    threshold_probability: DecisionThresholdProbability,
) -> float:
    """Compute conventional net benefit after enforcing probability semantics."""

    actions = standard_dca_actions(
        predicted_probabilities,
        threshold_probability=threshold_probability,
    )
    outcomes = np.asarray(outcome_positive, dtype=np.bool_)
    if outcomes.ndim != 1 or len(outcomes) != len(actions):
        raise ValueError("outcomes must be one-dimensional and aligned with predictions")
    if not len(outcomes):
        raise ValueError("standard DCA requires at least one observation")
    true_positives = int(np.sum(actions & outcomes))
    false_positives = int(np.sum(actions & ~outcomes))
    threshold = _require_decision_threshold_probability(threshold_probability)
    return true_positives / len(outcomes) - (false_positives / len(outcomes)) * threshold / (
        1 - threshold
    )
