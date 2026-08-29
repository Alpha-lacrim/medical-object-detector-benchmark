from __future__ import annotations

import numpy as np
import pytest

from src.clinical.raw_score_utility import (
    load_raw_score_utility_config,
    maximum_scores_by_image,
    raw_score_utility_index,
)


def test_raw_score_utility_matches_historical_hand_calculation() -> None:
    assert raw_score_utility_index(2, 1, 4, 0.2) == pytest.approx(0.4375)
    np.testing.assert_allclose(
        raw_score_utility_index(
            np.asarray([1, 2]),
            np.asarray([1, 1]),
            np.asarray([4, 4]),
            0.2,
        ),
        [0.1875, 0.4375],
    )

    with pytest.raises(ValueError, match="strictly inside"):
        raw_score_utility_index(1, 0, 1, 1.0)
    with pytest.raises(ValueError, match="disjoint"):
        raw_score_utility_index(1, 1, 1, 0.5)


def test_exam_marker_is_maximum_emitted_box_confidence() -> None:
    records = [
        {"image_id": "a.png", "scores": [0.2, 0.8, 0.4]},
        {"image_id": "b.png", "scores": []},
    ]

    np.testing.assert_allclose(
        maximum_scores_by_image(records, ("b.png", "a.png")),
        [0.0, 0.8],
    )

    with pytest.raises(ValueError, match="grid mismatch"):
        maximum_scores_by_image(records, ("a.png", "c.png"))


def test_project_config_declares_nonstandard_interpretation_and_archives() -> None:
    config = load_raw_score_utility_config("configs/raw_score_utility.yaml")

    assert config.analysis.classification == "NON_STANDARD_RAW_SCORE_THRESHOLD_UTILITY"
    assert config.analysis.conventional_dca_interpretation_valid is False
    assert config.analysis.expected_salvage_outcome == (
        "incomplete_run_specific_validation_predictions"
    )
    assert len(config.historical_archives) == 5
    assert config.outputs.summary_table.name == "raw_score_threshold_utility_summary.csv"
