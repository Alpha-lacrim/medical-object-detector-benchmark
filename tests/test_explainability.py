from pathlib import Path

import pytest

from src.explainability.config import load_explainability_config
from src.explainability.run_explainability import aggregate_localization
from src.explainability.selection import select_qualitative_cases
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget


def _target(image_id: str, boxes=()) -> ImageTarget:
    return ImageTarget(image_id, (10, 10), boxes, [1] * len(boxes))


def _prediction(image_id: str, boxes=(), scores=()) -> ImagePrediction:
    return ImagePrediction(image_id, (10, 10), boxes, [1] * len(boxes), scores)


def test_explainability_config_matches_stride_and_sample_contract() -> None:
    config = load_explainability_config(Path("configs/explainability.yaml"))

    assert config.seed == 17
    assert {item.detector for item in config.detectors} == {"faster_rcnn", "yolo11s"}
    assert {item.expected_stride for item in config.detectors} == {16}
    assert {item.expected_spatial_size for item in config.detectors} == {40}


def test_case_selection_returns_paired_objective_categories() -> None:
    targets = {
        "good": _target("good", [[1, 1, 5, 5]]),
        "bad": _target("bad"),
        "failure": _target("failure", [[5, 5, 9, 9]]),
    }
    shared = {
        "good": _prediction("good", [[1, 1, 5, 5]], [0.9]),
        "bad": _prediction("bad", [[0, 0, 2, 2]], [0.8]),
        "failure": _prediction("failure", [[0, 0, 2, 2]], [0.1]),
    }

    cases = select_qualitative_cases(
        {"faster_rcnn": shared, "yolo11s": shared},
        targets,
        score_threshold=0.25,
        iou_threshold=0.5,
        max_detections=100,
        cases_per_category=1,
        failure_quantiles=(0.5,),
    )

    assert [item["category"] for item in cases] == [
        "good_prediction",
        "bad_prediction",
        "failure_case",
    ]


def test_aggregation_excludes_zero_maps_and_reports_status() -> None:
    records = []
    for detector, energy, hit in (
        ("faster_rcnn", 0.5, True),
        ("yolo11s", 0.25, False),
    ):
        records.extend(
            [
                {
                    "detector": detector,
                    "image_id": "a",
                    "operating_status": "true_positive",
                    "valid_cam": True,
                    "energy_in_box": energy,
                    "pointing_hit": hit,
                    "box_pixel_fraction": 0.2,
                    "energy_lift_over_area": energy - 0.2,
                },
                {
                    "detector": detector,
                    "image_id": "b",
                    "operating_status": "false_negative",
                    "valid_cam": False,
                    "energy_in_box": None,
                    "pointing_hit": None,
                    "box_pixel_fraction": 0.1,
                    "energy_lift_over_area": None,
                },
            ]
        )

    rows = aggregate_localization(records)
    faster_all = next(
        item
        for item in rows
        if item["detector"] == "faster_rcnn" and item["operating_status"] == "all"
    )

    assert faster_all["target_count"] == 2
    assert faster_all["valid_cam_count"] == 1
    assert faster_all["zero_energy_cam_count"] == 1
    assert faster_all["mean_energy_in_box"] == pytest.approx(0.5)
