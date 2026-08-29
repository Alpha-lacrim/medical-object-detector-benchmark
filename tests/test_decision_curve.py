from __future__ import annotations

import numpy as np
import pytest

from src.clinical.decision_curve import (
    CalibratedOutcomeProbabilities,
    DecisionThresholdProbability,
    RawDetectorConfidenceScores,
    RawDetectorConfidenceThreshold,
    standard_dca_actions,
    standard_dca_net_benefit,
)


def test_standard_dca_accepts_validation_frozen_outcome_probabilities() -> None:
    probabilities = CalibratedOutcomeProbabilities(
        values=np.asarray([0.8, 0.7, 0.1, 0.2]),
        validation_frozen_mapping_id="faster_rcnn-seed17-logistic-v1",
    )
    threshold = DecisionThresholdProbability(
        value=0.2,
        decision_context_id="hypothetical-reviewed-action-v1",
    )

    np.testing.assert_array_equal(
        standard_dca_actions(probabilities, threshold_probability=threshold),
        [True, True, False, True],
    )
    assert standard_dca_net_benefit(
        probabilities,
        np.asarray([True, False, False, True]),
        threshold_probability=threshold,
    ) == pytest.approx(0.4375)


@pytest.mark.parametrize(
    "ineligible",
    [
        RawDetectorConfidenceScores(values=np.asarray([0.8, 0.1])),
        np.asarray([0.8, 0.1]),
    ],
)
def test_standard_dca_rejects_raw_or_semantically_untyped_scores(ineligible: object) -> None:
    threshold = DecisionThresholdProbability(value=0.2, decision_context_id="context-v1")
    with pytest.raises(TypeError, match="raw detector confidence is not eligible"):
        standard_dca_actions(  # type: ignore[arg-type]
            ineligible,
            threshold_probability=threshold,
        )


@pytest.mark.parametrize(
    "ineligible_threshold",
    [RawDetectorConfidenceThreshold(value=0.2), 0.2],
)
def test_standard_dca_rejects_raw_confidence_passed_directly_as_pt(
    ineligible_threshold: object,
) -> None:
    probabilities = CalibratedOutcomeProbabilities(
        values=np.asarray([0.8, 0.1]),
        validation_frozen_mapping_id="mapping-v1",
    )

    with pytest.raises(TypeError, match="raw detector confidence cannot be passed directly"):
        standard_dca_actions(  # type: ignore[arg-type]
            probabilities,
            threshold_probability=ineligible_threshold,
        )
    with pytest.raises(TypeError, match="raw detector confidence cannot be passed directly"):
        standard_dca_net_benefit(  # type: ignore[arg-type]
            probabilities,
            np.asarray([True, False]),
            threshold_probability=ineligible_threshold,
        )


def test_standard_dca_validates_probability_contract() -> None:
    with pytest.raises(ValueError, match="mapping_id"):
        CalibratedOutcomeProbabilities(
            values=np.asarray([0.5]),
            validation_frozen_mapping_id=" ",
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CalibratedOutcomeProbabilities(
            values=np.asarray([1.1]),
            validation_frozen_mapping_id="mapping-v1",
        )

    with pytest.raises(ValueError, match="strictly inside"):
        DecisionThresholdProbability(value=0.0, decision_context_id="context-v1")
    with pytest.raises(ValueError, match="decision_context_id"):
        DecisionThresholdProbability(value=0.5, decision_context_id=" ")

    probabilities = CalibratedOutcomeProbabilities(
        values=np.asarray([0.5]),
        validation_frozen_mapping_id="mapping-v1",
    )
    threshold = DecisionThresholdProbability(value=0.5, decision_context_id="context-v1")
    with pytest.raises(ValueError, match="aligned"):
        standard_dca_net_benefit(
            probabilities,
            np.asarray([True, False]),
            threshold_probability=threshold,
        )
