"""Synthetic clone-oracle tests for frozen external resampling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.analyze_exact_score_froc import exact_score_froc_rows
from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget, evaluate_operating_point
from src.stats.paired import build_patient_clusters
from src.stats.transportability import (
    aligned_records,
    bootstrap,
    build_ap_cache,
    build_froc_cache,
    classify_change,
    draw_observations,
    interval_rows,
    operating_counts,
)


def records():
    targets = [
        ImageTarget("a", (40, 40), [[0, 0, 10, 10], [1, 0, 11, 10]], [1, 1]),
        ImageTarget("b", (40, 40), [], []),
        ImageTarget("c", (40, 40), [[0, 0, 10, 10]], [1]),
    ]
    predictions = [
        ImagePrediction(
            "a",
            (40, 40),
            [[0, 0, 10, 10], [20, 20, 30, 30], [1, 0, 11, 10]],
            [1, 1, 1],
            [0.8, 0.8, 0.8],
        ),
        ImagePrediction("b", (40, 40), [[20, 20, 30, 30]], [1], [0.8]),
        ImagePrediction("c", (40, 40), [[0, 0, 9, 10]], [1], [0.3]),
    ]
    return predictions, targets


def clones(predictions, targets, weights):
    ps, ts = [], []
    for i, (p, t, w) in enumerate(zip(predictions, targets, weights, strict=True)):
        for j in range(w):
            key = f"{i:06d}-{j:06d}"
            ps.append(ImagePrediction(key, p.image_size, p.boxes_xyxy, p.labels, p.scores))
            ts.append(ImageTarget(key, t.image_size, t.boxes_xyxy, t.labels))
    return ps, ts


def ap_cache(predictions, targets, cap=100, floor=0.001):
    return build_ap_cache(
        predictions,
        targets,
        class_ids=(1,),
        minimum_score=floor,
        max_dets=[1, 10, cap],
        iou_thresholds=np.linspace(0.5, 0.95, 10),
        recall_points=101,
    )


def froc_cache(predictions, targets, cap=100):
    return build_froc_cache(predictions, targets, class_ids=(1,), floor=0.001, iou=0.5, cap=cap)


@pytest.mark.parametrize("weights", [[1, 1, 1], [3, 2, 1], [0, 2, 4], [5, 0, 0]])
def test_cached_ap_equals_coco_with_distinct_image_copies_and_score_ties(weights):
    ps, ts = records()
    cache = ap_cache(ps, ts)
    aps = cache.evaluate(np.array(weights))
    cp, ct = clones(ps, ts, weights)
    oracle = evaluate_coco(cp, ct, class_ids=(1,), minimum_score=0.001)
    assert [aps[0], aps.mean()] == pytest.approx([oracle["ap50"], oracle["ap50_95"]], abs=1e-12)


def test_cached_ap_matches_coco_on_equal_iou_target_ties():
    # COCO's target-tie rule is preserved by caching official matches, not a surrogate matcher.
    ts = [ImageTarget("a", (30, 30), [[0, 0, 10, 10], [2, 0, 12, 10]], [1, 1])]
    ps = [ImagePrediction("a", (30, 30), [[1, 0, 11, 10], [0, 0, 9, 10]], [1, 1], [0.9, 0.8])]
    aps = ap_cache(ps, ts).evaluate(np.array([3]))
    cp, ct = clones(ps, ts, [3])
    oracle = evaluate_coco(cp, ct, class_ids=(1,))
    assert [aps[0], aps.mean()] == pytest.approx([oracle["ap50"], oracle["ap50_95"]], abs=1e-12)


def test_randomized_clone_oracle_ap_and_froc():
    rng = np.random.default_rng(700)
    ps, ts = records()
    ap, fc = ap_cache(ps, ts), froc_cache(ps, ts)
    budgets = [0.0, 0.125, 0.5, 1.0, 2.0]
    for _ in range(25):
        w = rng.multinomial(7, [1 / 3] * 3)
        if not w[0] + w[2]:
            continue
        cp, ct = clones(ps, ts, w)
        oracle = evaluate_coco(cp, ct, class_ids=(1,), minimum_score=0.001)
        aps = ap.evaluate(w)
        assert [aps[0], aps.mean()] == pytest.approx([oracle["ap50"], oracle["ap50_95"]])
        rows, _ = exact_score_froc_rows(
            cp,
            ct,
            detector="synthetic",
            seed=17,
            class_ids=(1,),
            candidate_score_floor=0.001,
            iou_threshold=0.5,
            max_detections=100,
        )
        expected = [
            max(r["sensitivity"] for r in rows if r["fp_per_image"] <= b + 1e-12) for b in budgets
        ]
        actual, limited = fc.budgets(w, budgets, 1e-12)
        assert actual == pytest.approx(expected)
        assert limited.tolist() == [rows[-1]["fp_per_image"] < b - 1e-12 for b in budgets]
        counts = w @ operating_counts(fc, 0.8)
        op = evaluate_operating_point(
            cp, ct, class_ids=(1,), score_threshold=0.8, iou_threshold=0.5, max_detections=100
        )["overall"]
        assert counts.tolist() == [op["tp"], op["fp"]]


def test_tied_froc_events_enter_together_and_negative_images_count():
    ps, ts = records()
    fc = froc_cache(ps, ts)
    # The first tied-score group has 2 TP and 2 FP. It cannot partially enter a 0.5 budget.
    values, limited = fc.budgets(np.ones(3, dtype=int), [0.5, 2 / 3, 1.0], 1e-12)
    assert values == pytest.approx([0, 1, 1])
    assert limited.tolist() == [False, False, True]


def test_floor_equality_caps_and_empty_predictions():
    ps, ts = records()
    # Native filtering has already happened. Common evaluation is >= at equality.
    cache = ap_cache(ps, ts, cap=100, floor=0.8)
    ref = evaluate_coco(ps, ts, class_ids=(1,), minimum_score=0.8, max_detections=100)
    vals = cache.evaluate(np.ones(3, dtype=int))
    assert [vals[0], vals.mean()] == pytest.approx([ref["ap50"], ref["ap50_95"]])
    empty = [ImagePrediction(t.image_id, t.image_size, [], [], []) for t in ts]
    assert np.all(ap_cache(empty, ts).evaluate(np.ones(3, dtype=int)) == 0)
    fc = froc_cache(empty, ts)
    vals, limited = fc.budgets(np.ones(3, dtype=int), [0.125, 2], 1e-12)
    assert vals.tolist() == [0, 0] and limited.all()


def test_detection_cap_is_preserved_before_resampling():
    ts = [ImageTarget("a", (40, 40), [[0, 0, 10, 10]], [1])]
    ps = [
        ImagePrediction(
            "a",
            (40, 40),
            [[20, 20, 30, 30]] * 100 + [[0, 0, 10, 10]],
            [1] * 101,
            np.linspace(0.9, 0.1, 101),
        )
    ]
    assert np.all(ap_cache(ps, ts).evaluate(np.array([3])) == 0)
    fc = froc_cache(ps, ts)
    assert fc.curve(np.array([3]))[0][-1] == 100
    assert fc.curve(np.array([3]))[1][-1] == 0


def test_no_target_draw_is_undefined_not_retried_or_imputed():
    ps, ts = records()
    w = np.array([0, 3, 0])
    assert np.isnan(ap_cache(ps, ts).evaluate(w)).all()
    assert np.isnan(froc_cache(ps, ts).budgets(w, [0.5], 1e-12)[0]).all()
    samples = np.array([[[np.nan], [np.nan]], [[0.1], [0.0]], [[0.2], [0.1]]])
    row = interval_rows(np.array([[0.1], [0]]), samples, ["ap50"], 0.95)[0]
    assert row["difference_valid"] == 2 and row["difference_undefined"] == 1
    assert "no retry" in row["reason"]
    assert row["difference_ci_low"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "w",
    [np.array([1.0, 1, 1]), np.array([-1, 1, 3]), np.zeros(3, dtype=int), np.ones(2, dtype=int)],
)
def test_invalid_observation_multiplicity_rejected(w):
    ps, ts = records()
    with pytest.raises(ValueError, match="weights"):
        ap_cache(ps, ts).evaluate(w)


def test_id_and_category_guards():
    ps, ts = records()
    with pytest.raises(ValueError, match="unique"):
        aligned_records([*ps, ps[0]], ts)
    with pytest.raises(ValueError, match="category"):
        build_froc_cache(ps, ts, class_ids=(2,), floor=0.001, iou=0.5, cap=100)
    with pytest.raises(ValueError, match="floor"):
        build_froc_cache(ps, ts, class_ids=(1,), floor=0.5, iou=0.5, cap=100)


def test_known_patient_clusters_move_together_and_external_images_do_not_invent_groups():
    clusters = build_patient_clusters(("a", "b", "c"), {"a": "p1", "b": "p1", "c": "p2"})
    rng = np.random.default_rng(9)
    for _ in range(20):
        w = draw_observations(rng, 3, clusters)
        assert w[0] == w[1]
    draws = [draw_observations(rng, 3, None) for _ in range(20)]
    assert all(w.sum() == 3 for w in draws)
    assert any(w[0] != w[1] for w in draws)


def test_bootstrap_shares_observations_but_not_same_number_runs(monkeypatch):
    observations = np.array([2, 0, 1])
    calls = []

    class FakeRun:
        def __init__(self, seed, value):
            self.seed, self.value = seed, value
            self.ap = SimpleNamespace(image_ids=("a", "b", "c"), target_counts=np.ones(3))
            self.froc = self.ap

        def evaluate(self, weights, budgets, tolerance):
            calls.append(weights)
            return np.full(7, self.value)

    monkeypatch.setattr("src.stats.transportability.draw_observations", lambda *args: observations)
    monkeypatch.setattr(
        "src.stats.transportability.draw_independent_run_bootstrap_multiplicities",
        lambda *args, **kwargs: (np.array([2, 0]), np.array([0, 2])),
    )
    result = bootstrap(
        ((FakeRun(17, 1), FakeRun(42, 9)), (FakeRun(17, 3), FakeRun(42, 7))),
        clusters=None,
        settings={"bootstrap_resamples": 2, "bootstrap_seed": 12},
        label="test",
        budgets=[0.5],
        tolerance=1e-12,
        conditional_seed=17,
        progress_every=0,
    )
    assert all(w is observations for w in calls)
    assert result["training_procedure"][0, :, 0].tolist() == [1, 7]
    assert result["checkpoint_conditional_seed17"][0, :, 0].tolist() == [1, 3]


@pytest.mark.parametrize(
    "before,after,ordering,gap",
    [
        (1, 1, "unchanged", "unchanged"),
        (1, 2, "unchanged", "strengthened"),
        (1, 0.1, "unchanged", "weakened"),
        (1, -0.1, "reversed", "reversed"),
        (-1, -0.1, "unchanged", "weakened"),
        (0, 1, "tie_changed", "strengthened"),
    ],
)
def test_change_classification_is_descriptive(before, after, ordering, gap):
    assert classify_change(before, after, 1e-12) == {"ordering_change": ordering, "gap_change": gap}


def test_all_undefined_intervals_remain_na_with_zero_valid_draws():
    rows = interval_rows(np.array([[np.nan], [np.nan]]), np.full((5, 2, 1), np.nan), ["ap50"], 0.95)
    assert rows[0]["difference_valid"] == 0
    assert rows[0]["difference_undefined"] == 5
    assert np.isnan(rows[0]["difference_ci_low"])


def test_provenance_rejects_same_size_tampering(tmp_path, monkeypatch):
    from src.stats.run_vindr_statistics import artifact, verify_artifact

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source.json"
    path.write_bytes(b'{"value": 1}\n')
    receipt = artifact(path)
    verify_artifact(receipt)
    path.write_bytes(b'{"value": 2}\n')
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_artifact(receipt)


def test_descriptive_runs_are_equal_weighted_and_empty_scores_are_explicit():
    from src.stats.run_vindr_statistics import summarize_values

    result = summarize_values([0.1, 0.9, 0.0])
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["sd"] == pytest.approx(np.std([0.1, 0.9, 0.0], ddof=1))
    empty = summarize_values([None, None])
    assert empty == {"mean": None, "sd": None, "defined_runs": 0, "attempted_runs": 2}
    mixed = summarize_values([0.5, None])
    assert mixed == {"mean": 0.5, "sd": None, "defined_runs": 1, "attempted_runs": 2}


def test_inferential_metric_inventory_excludes_posthoc_policy():
    from src.stats.run_vindr_statistics import metric_names

    names = metric_names(
        {"evaluation": {"froc": {"fp_per_image_budgets": [0.125, 0.25, 0.5, 1, 2]}}}
    )
    assert len(names) == 11 and len(set(names)) == 11
    assert not any("secondary" in name for name in names)
    assert names[-4:] == [
        "primary_precision",
        "primary_recall",
        "primary_f1",
        "primary_fp_per_image",
    ]


def test_bootstrap_is_deterministic_and_propagates_undefined_selected_runs():
    class FakeRun:
        def __init__(self, seed, value):
            self.seed, self.value = seed, value
            self.ap = SimpleNamespace(image_ids=("a", "b", "c"), target_counts=np.ones(3))
            self.froc = self.ap

        def evaluate(self, weights, budgets, tolerance):
            return np.full(7, self.value * weights[0])

    runs = ((FakeRun(17, 1), FakeRun(42, np.nan)), (FakeRun(17, 2), FakeRun(42, 5)))
    kwargs = {
        "clusters": None,
        "settings": {"bootstrap_resamples": 20, "bootstrap_seed": 12},
        "label": "determinism",
        "budgets": [0.5],
        "tolerance": 1e-12,
        "conditional_seed": 17,
        "progress_every": 0,
    }
    first, second = bootstrap(runs, **kwargs), bootstrap(runs, **kwargs)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert np.isnan(first["training_procedure"][:, 0]).any()
    assert np.isfinite(first["training_procedure"][:, 0]).any()
    assert np.isfinite(first["checkpoint_conditional_seed17"]).all()


def test_report_replay_ignores_json_object_key_order(tmp_path):
    from pathlib import Path

    from src.stats.report_transportability import report
    from src.stats.run_vindr_statistics import read_json, read_yaml

    config = read_yaml("configs/vindr_statistics_v1.yaml")
    summary = read_json(Path(config["outputs"]["public_root"]) / config["outputs"]["summary"])
    protocol = read_yaml(config["protocol_config"])
    config["outputs"]["report"] = str(tmp_path / "report.md")
    path = Path(config["outputs"]["report"])
    summary["populations"] = {k: summary["populations"][k] for k in ("internal", "external")}
    report(summary, config, protocol)
    first = path.read_bytes()
    summary["populations"] = dict(reversed(list(summary["populations"].items())))
    report(summary, config, protocol)
    assert path.read_bytes() == first
