from __future__ import annotations

import pytest

from src.collect_froc_lower_floor_predictions import validate_protocol_contract
from src.evaluate import load_phase5_config


def test_approved_lower_floor_config_changes_only_candidate_collection_floor() -> None:
    source = load_phase5_config("configs/evaluation.yaml")
    lower = load_phase5_config("configs/evaluation_froc_lower_floor_v3.yaml")

    validate_protocol_contract(lower, source)
    assert source.evaluation.coco_minimum_score == 0.001
    assert lower.evaluation.coco_minimum_score == 0.0001
    assert lower.seeds == (17, 42, 137, 271, 314)
    assert len(lower.runs) == 10
    assert any(run.seed == 271 for run in lower.runs)
    assert lower.evaluation.score_threshold == source.evaluation.score_threshold
    assert lower.evaluation.match_iou_threshold == source.evaluation.match_iou_threshold
    assert lower.evaluation.nms_iou_threshold == source.evaluation.nms_iou_threshold
    assert lower.evaluation.max_detections == source.evaluation.max_detections


def test_lower_floor_contract_rejects_a_matching_change() -> None:
    source = load_phase5_config("configs/evaluation.yaml")
    lower = load_phase5_config("configs/evaluation_froc_lower_floor_v3.yaml")
    changed_evaluation = lower.evaluation.model_copy(update={"match_iou_threshold": 0.6})
    changed = lower.model_copy(update={"evaluation": changed_evaluation})

    with pytest.raises(ValueError, match="match_iou_threshold"):
        validate_protocol_contract(changed, source)


def test_next_proposed_floor_is_detector_neutral_and_approval_gated() -> None:
    source = load_phase5_config("configs/evaluation.yaml")
    proposal = load_phase5_config("configs/evaluation_froc_lower_floor_v4_proposal.yaml")

    validate_protocol_contract(proposal, source)
    assert proposal.evaluation.coco_minimum_score == 0.00001
    assert proposal.evaluation.score_threshold == source.evaluation.score_threshold
    assert proposal.evaluation.match_iou_threshold == source.evaluation.match_iou_threshold
    assert proposal.evaluation.nms_iou_threshold == source.evaluation.nms_iou_threshold
    assert proposal.evaluation.max_detections == source.evaluation.max_detections
    assert "proposal" in proposal.experiment_id


def test_approved_v4_floor_changes_only_candidate_collection_floor_from_v3() -> None:
    source = load_phase5_config("configs/evaluation.yaml")
    prior = load_phase5_config("configs/evaluation_froc_lower_floor_v3.yaml")
    lower = load_phase5_config("configs/evaluation_froc_lower_floor_v4.yaml")

    validate_protocol_contract(lower, source)
    validate_protocol_contract(lower, prior)
    assert prior.evaluation.coco_minimum_score == 0.0001
    assert lower.evaluation.coco_minimum_score == 0.00001
    assert lower.seeds == (17, 42, 137, 271, 314)
    assert len(lower.runs) == 10
    assert lower.evaluation.model_dump(exclude={"coco_minimum_score"}) == (
        prior.evaluation.model_dump(exclude={"coco_minimum_score"})
    )
