import pytest

from src.plot_froc_curves import (
    aggregate_froc_curve,
    load_froc_config,
    load_froc_rows,
    select_froc_operating_points,
)


def test_froc_archive_config_and_frozen_source_grid() -> None:
    config = load_froc_config("configs/froc_n3_archive.yaml")
    rows, image_count, _summary = load_froc_rows(config)

    assert config.analysis.fp_per_image_budgets == (0.125, 0.25, 0.5, 1.0, 2.0)
    assert image_count == 750
    assert len(rows) == 594
    assert {(row["detector"], row["seed"]) for row in rows} == {
        (detector, seed) for detector in ("faster_rcnn", "yolo11s") for seed in (17, 42, 137)
    }
    assert len(aggregate_froc_curve(rows)) == 198


def test_froc_n5_path_contains_all_runs_including_seed_271() -> None:
    config = load_froc_config("configs/froc_n5_sensitivity.yaml")
    rows, image_count, summary = load_froc_rows(config)

    assert image_count == 750
    assert summary["counts"]["seeds_per_detector"] == 5
    assert len(rows) == 990
    assert {(row["detector"], row["seed"]) for row in rows} == {
        (detector, seed)
        for detector in ("faster_rcnn", "yolo11s")
        for seed in (17, 42, 137, 271, 314)
    }
    yolo_271 = [row for row in rows if row["detector"] == "yolo11s" and row["seed"] == 271]
    assert len(yolo_271) == 99
    assert any(row["sensitivity"] > 0 for row in yolo_271 if row["threshold"] < 0.05)
    assert all(row["sensitivity"] == 0 for row in yolo_271 if row["threshold"] >= 0.05)


def test_yolo_high_budget_plateau_is_the_observed_lower_grid_boundary() -> None:
    config = load_froc_config("configs/froc_n5_sensitivity.yaml")
    rows, _image_count, _summary = load_froc_rows(config)

    aggregate, per_seed = select_froc_operating_points(
        rows,
        config.analysis.fp_per_image_budgets,
        tolerance=config.analysis.numeric_tolerance,
    )

    high_budgets = {0.5, 1.0, 2.0}
    yolo_seed_rows = [
        row
        for row in per_seed
        if row["detector"] == "yolo11s" and row["fp_per_image_budget"] in high_budgets
    ]
    yolo_aggregate_rows = [
        row
        for row in aggregate
        if row["detector"] == "yolo11s" and row["fp_per_image_budget"] in high_budgets
    ]

    assert len(yolo_seed_rows) == 15
    assert {row["selected_threshold"] for row in yolo_seed_rows} == {0.01}
    assert len(yolo_aggregate_rows) == 3
    assert {row["selected_threshold_min"] for row in yolo_aggregate_rows} == {0.01}
    assert {row["selected_threshold_max"] for row in yolo_aggregate_rows} == {0.01}
    assert len({row["sensitivity"] for row in yolo_aggregate_rows}) == 1


def test_froc_budget_selection_is_non_interpolated_and_seed_specific() -> None:
    rows = [
        {
            "detector": "detector",
            "seed": seed,
            "threshold": threshold,
            "sensitivity": sensitivity,
            "fp_per_image": fp_per_image,
        }
        for seed in (17, 42, 137)
        for threshold, sensitivity, fp_per_image in (
            (0.3, 0.2, 0.1),
            (0.2, 0.5, 0.25),
            (0.1, 0.8, 0.6),
        )
    ]

    aggregate, per_seed = select_froc_operating_points(rows, (0.2, 0.5), tolerance=1e-12)

    low = next(row for row in aggregate if row["fp_per_image_budget"] == 0.2)
    high = next(row for row in aggregate if row["fp_per_image_budget"] == 0.5)
    assert low["sensitivity"] == pytest.approx(0.2)
    assert low["achieved_fp_per_image"] == pytest.approx(0.1)
    assert high["sensitivity"] == pytest.approx(0.5)
    assert high["achieved_fp_per_image"] == pytest.approx(0.25)
    assert len(per_seed) == 6
