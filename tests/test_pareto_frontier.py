from dataclasses import replace

import pytest

from src.plot_pareto_frontier import (
    ParetoPoint,
    dominance_by_panel,
    load_pareto_config,
    load_pareto_points,
    strict_detector_dominance,
)


def _point(detector: str, seed: int, *, fps: float, map_value: float) -> ParetoPoint:
    return ParetoPoint(
        detector=detector,
        seed=seed,
        run_id=f"{detector}-{seed}",
        map_50_95=map_value,
        recall=map_value,
        recall_threshold=0.25,
        throughput_fps=fps,
        mean_latency_ms=1000 / fps,
        total_parameters=1,
        estimated_gflops=1,
    )


def test_config_and_frozen_inputs_join_all_three_seeds() -> None:
    config = load_pareto_config("configs/pareto.yaml")
    points = load_pareto_points(config)

    assert {(point.detector, point.seed) for point in points} == {
        (detector, seed) for detector in ("faster_rcnn", "yolo11s") for seed in (17, 42, 137)
    }
    assert {point.recall_threshold for point in points if point.detector == "faster_rcnn"} == {0.63}
    assert {point.recall_threshold for point in points if point.detector == "yolo11s"} == {0.01}


def test_strict_dominance_requires_both_seed_clouds_to_be_ordered() -> None:
    points = [
        _point("a", 1, fps=20, map_value=0.8),
        _point("a", 2, fps=22, map_value=0.9),
        _point("b", 1, fps=10, map_value=0.4),
        _point("b", 2, fps=12, map_value=0.5),
    ]

    assert (
        strict_detector_dominance(
            points,
            x_field="throughput_fps",
            y_field="map_50_95",
            x_direction="higher",
            y_direction="higher",
        )
        == "a"
    )

    trade_off = [
        replace(point, throughput_fps=5) if point.detector == "a" else point for point in points
    ]
    assert (
        strict_detector_dominance(
            trade_off,
            x_field="throughput_fps",
            y_field="map_50_95",
            x_direction="higher",
            y_direction="higher",
        )
        is None
    )


def test_observed_panels_have_no_strict_detector_dominance() -> None:
    points = load_pareto_points(load_pareto_config("configs/pareto.yaml"))

    assert dominance_by_panel(points) == {"a": None, "b": None, "c": None, "d": None}
    faster_recall = [point.recall for point in points if point.detector == "faster_rcnn"]
    yolo_recall = [point.recall for point in points if point.detector == "yolo11s"]
    assert min(faster_recall) > max(yolo_recall)


def test_config_rejects_missing_compute_table(tmp_path) -> None:
    config = load_pareto_config("configs/pareto.yaml")
    broken = config.model_copy(
        update={
            "inputs": config.inputs.model_copy(
                update={"compute_tables": (*config.inputs.compute_tables, tmp_path / "missing.csv")}
            )
        }
    )

    with pytest.raises(FileNotFoundError, match="required Pareto input is missing"):
        load_pareto_points(broken)
