from __future__ import annotations

import csv

import numpy as np
import pytest

from src.evaluate import EvaluationSettings, evaluate_prediction_records
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget
from src.stats.paired import (
    METRICS,
    EvidencePair,
    aggregate_hybrid,
    analyze_pair,
    build_evidence,
    build_patient_clusters,
    draw_hierarchical_bootstrap_multiplicities,
    expand_patient_group_choices,
    expand_patient_group_multiplicities,
    holm_adjust,
)
from src.stats.run_statistics import (
    _merge_clean_endpoint_rows,
    _validate_clean_seed_eligibility,
    apply_clean_holm,
    apply_grid_holm,
    build_parser,
    compare_holm_significance,
    load_statistics_config,
)

SETTINGS = EvaluationSettings(
    score_threshold=0.25,
    match_iou_threshold=0.5,
    coco_minimum_score=0.001,
    nms_iou_threshold=0.5,
    max_detections=100,
)


def _records() -> tuple[list[ImageTarget], list[ImagePrediction], list[ImagePrediction]]:
    targets = [
        ImageTarget("a.png", (20, 20), [[0, 0, 10, 10]], [1]),
        ImageTarget("b.png", (20, 20), [[5, 5, 15, 15]], [1]),
        ImageTarget("c.png", (20, 20), [], []),
        ImageTarget("d.png", (20, 20), [[0, 0, 8, 8]], [1]),
    ]
    detector_a = [
        ImagePrediction("a.png", (20, 20), [[0, 0, 10, 10]], [1], [0.9]),
        ImagePrediction("b.png", (20, 20), [[0, 0, 4, 4]], [1], [0.8]),
        ImagePrediction("c.png", (20, 20), [[1, 1, 5, 5]], [1], [0.7]),
        ImagePrediction("d.png", (20, 20), [[0, 0, 8, 5]], [1], [0.6]),
    ]
    detector_b = [
        ImagePrediction("a.png", (20, 20), [[0, 0, 7, 10]], [1], [0.85]),
        ImagePrediction("b.png", (20, 20), [[5, 5, 15, 15]], [1], [0.75]),
        ImagePrediction("c.png", (20, 20), [], [], []),
        ImagePrediction("d.png", (20, 20), [[10, 10, 18, 18]], [1], [0.55]),
    ]
    return targets, detector_a, detector_b


def _evidence(predictions: list[ImagePrediction], targets: list[ImageTarget]):
    return build_evidence(
        predictions,
        targets,
        class_ids=(1,),
        score_threshold=SETTINGS.score_threshold,
        match_iou_threshold=SETTINGS.match_iou_threshold,
        coco_minimum_score=SETTINGS.coco_minimum_score,
        max_detections=SETTINGS.max_detections,
    )


def _metric_vector(result: dict) -> np.ndarray:
    operating, coco = result["operating_point"]["overall"], result["coco"]
    return np.asarray(
        [
            operating["precision"],
            operating["recall"],
            operating["f1"],
            operating["matched_mean_iou"],
            operating["matched_mean_box_dice"],
            coco["ap50"],
            coco["ap50_95"],
        ],
        dtype=np.float64,
    )


def test_fast_metric_reconstruction_matches_unified_evaluator() -> None:
    targets, detector_a, detector_b = _records()
    evidence_a, evidence_b = _evidence(detector_a, targets), _evidence(detector_b, targets)
    counts = np.ones(len(targets), dtype=np.int64)
    choose_a = np.zeros(len(targets), dtype=np.bool_)

    reconstructed_a = aggregate_hybrid(
        evidence_a, evidence_b, multiplicities=counts, choose_b=choose_a
    )
    reconstructed_b = aggregate_hybrid(
        evidence_a, evidence_b, multiplicities=counts, choose_b=~choose_a
    )
    official_a = evaluate_prediction_records(
        detector_a, targets, category_names={1: "opacity"}, settings=SETTINGS
    )
    official_b = evaluate_prediction_records(
        detector_b, targets, category_names={1: "opacity"}, settings=SETTINGS
    )

    assert reconstructed_a == pytest.approx(_metric_vector(official_a), abs=1e-12)
    assert reconstructed_b == pytest.approx(_metric_vector(official_b), abs=1e-12)


def test_integer_image_weights_match_explicit_resampling() -> None:
    targets, detector_a, _detector_b = _records()
    evidence = _evidence(detector_a, targets)
    counts = np.asarray([2, 0, 1, 2], dtype=np.int64)
    weighted = aggregate_hybrid(
        evidence,
        evidence,
        multiplicities=counts,
        choose_b=np.zeros(len(targets), dtype=np.bool_),
    )

    expanded_targets: list[ImageTarget] = []
    expanded_predictions: list[ImagePrediction] = []
    for index, count in enumerate(counts):
        for copy_index in range(int(count)):
            image_id = f"{index}-{copy_index}.png"
            target, prediction = targets[index], detector_a[index]
            expanded_targets.append(
                ImageTarget(
                    image_id,
                    target.image_size,
                    target.boxes_xyxy,
                    target.labels,
                )
            )
            expanded_predictions.append(
                ImagePrediction(
                    image_id,
                    prediction.image_size,
                    prediction.boxes_xyxy,
                    prediction.labels,
                    prediction.scores,
                )
            )
    official = evaluate_prediction_records(
        expanded_predictions,
        expanded_targets,
        category_names={1: "opacity"},
        settings=SETTINGS,
    )
    assert weighted == pytest.approx(_metric_vector(official), abs=1e-12)


def test_zero_prediction_positive_target_matches_operating_point_contract() -> None:
    targets = [ImageTarget("positive.png", (20, 20), [[0, 0, 10, 10]], [1])]
    predictions = [ImagePrediction("positive.png", (20, 20), [], [], [])]
    evidence = _evidence(predictions, targets)

    reconstructed = aggregate_hybrid(
        evidence,
        evidence,
        multiplicities=np.ones(1, dtype=np.int64),
        choose_b=np.zeros(1, dtype=np.bool_),
    )
    official = evaluate_prediction_records(
        predictions, targets, category_names={1: "opacity"}, settings=SETTINGS
    )

    assert reconstructed[:3] == pytest.approx([0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        reconstructed, _metric_vector(official), rtol=0, atol=1e-12, equal_nan=True
    )


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_patient_group_draws_and_swaps_move_complete_clusters() -> None:
    clusters = build_patient_clusters(
        ("a.png", "b.png", "c.png", "d.png"),
        {"a.png": "p1", "b.png": "p1", "c.png": "p2", "d.png": "p3"},
    )

    multiplicities = expand_patient_group_multiplicities(
        np.asarray([2, 0, 1], dtype=np.int64), clusters
    )
    choices = expand_patient_group_choices(
        np.asarray([True, False, True], dtype=np.bool_), clusters
    )

    assert clusters.patient_group_ids == ("p1", "p2", "p3")
    assert multiplicities.tolist() == [2, 2, 0, 1]
    assert choices.tolist() == [True, True, False, True]

    image_draw, seed_draw = draw_hierarchical_bootstrap_multiplicities(
        np.random.default_rng(123), clusters, seed_count=3
    )
    assert image_draw[0] == image_draw[1]
    assert image_draw.shape == (4,)
    assert seed_draw.shape == (3,)
    assert int(np.sum(seed_draw)) == 3


def test_grid_holm_excludes_explicitly_non_estimable_hypotheses() -> None:
    rows = [
        {
            "condition_id": condition,
            "metric": "iou",
            "estimand": "raw",
            "status": status,
            "p_value_raw": p_value,
            "p_value_holm": p_value,
            "holm_family_size": 1,
        }
        for condition, status, p_value in (
            ("corruption_s1", "complete", 0.01),
            ("corruption_s2", "not_estimable", None),
            ("corruption_s3", "complete", 0.04),
        )
    ]

    apply_grid_holm(rows)

    assert [row["holm_family_size"] for row in rows] == [2, 2, 2]
    assert rows[0]["p_value_holm"] == pytest.approx(0.02)
    assert rows[1]["p_value_holm"] is None
    assert rows[2]["p_value_holm"] == pytest.approx(0.04)


def test_clean_holm_uses_exact_seven_endpoint_family() -> None:
    raw_values = (0.01, 0.04, 0.03, 0.2, 0.5, 0.001, 0.9)
    rows = [
        {
            "metric": metric,
            "status": "complete",
            "p_value_raw": value,
            "p_value_holm": value,
            "holm_family_size": 1,
        }
        for metric, value in zip(METRICS, raw_values, strict=True)
    ]

    apply_clean_holm(rows)

    assert [row["p_value_holm"] for row in rows] == pytest.approx(holm_adjust(raw_values))
    assert [row["holm_family_size"] for row in rows] == [7] * 7


def test_clean_holm_fails_closed_if_an_endpoint_is_missing() -> None:
    rows = [{"metric": metric, "status": "complete", "p_value_raw": 0.5} for metric in METRICS[:-1]]

    with pytest.raises(ValueError, match="seven metrics"):
        apply_clean_holm(rows)


def test_paired_analysis_is_seeded_and_reports_every_metric() -> None:
    targets, detector_a, detector_b = _records()
    pair = EvidencePair(
        detector_a=(_evidence(detector_a, targets),),
        detector_b=(_evidence(detector_b, targets),),
    )
    patient_clusters = build_patient_clusters(
        pair.detector_a[0].image_ids,
        {"a.png": "p1", "b.png": "p1", "c.png": "p2", "d.png": "p3"},
    )
    first = analyze_pair(
        pair,
        patient_clusters=patient_clusters,
        base_seed=123,
        comparison_label="unit-test",
        bootstrap_resamples=100,
        permutation_resamples=100,
        confidence_level=0.95,
    )
    second = analyze_pair(
        pair,
        patient_clusters=patient_clusters,
        base_seed=123,
        comparison_label="unit-test",
        bootstrap_resamples=100,
        permutation_resamples=100,
        confidence_level=0.95,
    )

    assert first == second
    assert tuple(row["metric"] for row in first["raw"]) == METRICS
    assert all(0 < row["p_value_raw"] <= 1 for row in first["raw"])
    assert all(row["effect_size_name"] == "paired_raw_difference" for row in first["raw"])
    assert all(row["effect_size_n"] == 3 for row in first["raw"])
    assert all(
        row["effect_size"] == pytest.approx(row["difference_a_minus_b"]) for row in first["raw"]
    )


def test_statistics_config_covers_five_seed_and_corruption_inputs() -> None:
    config = load_statistics_config("configs/statistics.yaml")

    assert config.analysis.metrics == METRICS
    assert config.analysis.bootstrap_resamples == 2000
    assert config.analysis.permutation_resamples == 5000
    assert config.analysis.bootstrap_method == "paired_hierarchical_patient_cluster_percentile"
    assert config.analysis.permutation_method == "paired_patient_cluster_label_swap"
    assert config.analysis.effect_size == "paired_raw_difference_with_cluster_bootstrap_ci"
    assert config.analysis.multiple_comparison_correction == "holm"
    assert config.analysis.clean_correction_scope == "across_7_predictive_metrics"
    assert config.resolve(config.inputs.phase5_summary).is_file()
    assert config.resolve(config.inputs.phase6_summary).is_file()
    assert config.resolve(config.inputs.test_split_manifest).is_file()
    assert config.resolve(config.outputs.patient_cluster_clean_n3_archive).is_file()
    eligibility = config.analysis.clean_seed_eligibility
    assert eligibility.all_attempt_seeds == (17, 42, 137, 271, 314)
    assert eligibility.paired_complete_case_seeds == (17, 42, 137, 314)
    assert eligibility.paired_complete_case_metrics == ("iou", "dice")


def test_endpoint_groups_keep_zero_attempts_and_use_complete_conditional_pairs() -> None:
    config = load_statistics_config("configs/statistics.yaml")
    targets, detector_a, detector_b = _records()
    evidence_a = _evidence(detector_a, targets)
    evidence_b = _evidence(detector_b, targets)
    below_threshold_b = [
        ImagePrediction(
            item.image_id,
            item.image_size,
            item.boxes_xyxy,
            item.labels,
            np.full_like(item.scores, 0.1),
        )
        for item in detector_b
    ]
    zero_operating_point_b = _evidence(below_threshold_b, targets)
    pair = EvidencePair(
        detector_a=(evidence_a,) * 5,
        detector_b=(evidence_b, evidence_b, evidence_b, zero_operating_point_b, evidence_b),
    )

    conditional_pair, eligibility = _validate_clean_seed_eligibility(
        pair,
        seeds=(17, 42, 137, 271, 314),
        analysis=config.analysis,
    )

    assert conditional_pair.seed_count == 4
    assert eligibility["paired_complete_case_seed_ids"] == [17, 42, 137, 314]
    assert eligibility["excluded_pair_seed_ids"] == [271]
    assert eligibility["expected_undefined"][0]["prediction_count"] == 0

    template = [
        {
            "metric": metric,
            "status": "complete",
            "p_value_raw": 0.5,
            "detector_a_estimate": float(index),
        }
        for index, metric in enumerate(METRICS)
    ]
    conditional = [
        dict(row, detector_a_estimate=float(index + 100)) for index, row in enumerate(template)
    ]
    merged = _merge_clean_endpoint_rows(template, conditional, eligibility=eligibility)

    assert tuple(row["metric"] for row in merged) == METRICS
    assert [row["seed_count"] for row in merged] == [5, 5, 5, 4, 4, 5, 5]
    assert merged[0]["detector_a_estimate"] == 0.0
    assert merged[3]["detector_a_estimate"] == 103.0
    assert merged[3]["paired_seed_ids"] == "17;42;137;314"
    assert merged[3]["excluded_pair_seed_ids"] == "271"
    assert "yolo11s seed 271" in merged[3]["seed_eligibility_note"]


def test_clean_eligibility_rejects_an_unexpected_zero_tp_seed() -> None:
    config = load_statistics_config("configs/statistics.yaml")
    targets, detector_a, _detector_b = _records()
    evidence = _evidence(detector_a, targets)
    pair = EvidencePair(detector_a=(evidence,) * 5, detector_b=(evidence,) * 5)

    with pytest.raises(ValueError, match="differ from the declared contract"):
        _validate_clean_seed_eligibility(
            pair,
            seeds=(17, 42, 137, 271, 314),
            analysis=config.analysis,
        )


def test_statistics_cli_exposes_clean_only_scope() -> None:
    args = build_parser().parse_args(["--mode", "run", "--scope", "clean"])

    assert args.mode == "run"
    assert args.scope == "clean"


def test_holm_comparison_supports_n3_to_n5_labels(tmp_path) -> None:
    archived = {
        "scope": "phase5_clean_multi_seed",
        "condition_id": "clean",
        "estimand": "raw",
        "metric": "precision",
        "status": "complete",
        "p_value_holm": 0.01,
        "difference_a_minus_b": -0.2,
    }
    archive_path = tmp_path / "n3.csv"
    with archive_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(archived))
        writer.writeheader()
        writer.writerow(archived)
    current = {
        **archived,
        "scope": "phase5_clean_multi_seed",
        "p_value_holm": 0.2,
        "difference_a_minus_b": -0.1,
    }

    result = compare_holm_significance(
        [current],
        archive_path,
        archived_label="n3_patient_cluster",
        current_label="n5_patient_cluster",
        include_scope_in_key=False,
    )

    assert result["pattern_changed"] is True
    assert result["n3_patient_cluster_significant_count"] == 1
    assert result["n5_patient_cluster_significant_count"] == 0
    assert result["became_non_significant"][0]["n3_patient_cluster_p_holm"] == 0.01
    assert result["became_non_significant"][0]["n5_patient_cluster_p_holm"] == 0.2
