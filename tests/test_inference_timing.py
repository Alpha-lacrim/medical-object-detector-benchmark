"""Protocol boundary, deterministic subset, parity and publication-gate tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.benchmark_inference import (
    Agreement,
    Subset,
    TimingConfig,
    compare_predictions,
    hardware_snapshot,
    load_config,
    select_indices,
    summarize,
    time_call,
    verify_history,
)

ROOT = Path(__file__).resolve().parents[1]


def test_timer_brackets_the_whole_prediction_with_cuda_completion() -> None:
    events = []
    clocks = iter([10.0, 10.04])

    def sync() -> None:
        events.append("sync")

    def clock() -> float:
        events.append("clock")
        return next(clocks)

    def prediction() -> dict:
        events.extend(["resize", "tensor", "h2d", "forward", "nms", "restore", "d2h"])
        return {"sentinel": np.array([1])}

    result, elapsed = time_call(prediction, sync, clock)
    assert result["sentinel"].item() == 1
    assert elapsed == pytest.approx(0.04)
    assert events == [
        "sync",
        "clock",
        "resize",
        "tensor",
        "h2d",
        "forward",
        "nms",
        "restore",
        "d2h",
        "sync",
        "clock",
    ]


def test_summary_uses_total_time_and_linear_iqr_not_mean_fps() -> None:
    result = summarize([0.01, 0.02, 0.03, 0.1])
    assert result["total_elapsed_seconds"] == pytest.approx(0.16)
    assert result["fps"] == 25
    assert result["median_latency_ms"] == 25
    assert result["q1_latency_ms"] == 17.5
    assert result["q3_latency_ms"] == 47.5
    assert result["iqr_latency_ms"] == 30


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timings_fail(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize(values)


def test_subset_is_order_invariant_and_changes_with_seed() -> None:
    names = [f"image_{i}.png" for i in range(30)]
    settings = Subset(size=10, seed=17, selection="sha256-ranked-file-name")
    selected = [names[i] for i in select_indices(names, settings)]
    shuffled = names[::-1]
    assert selected == [shuffled[i] for i in select_indices(shuffled, settings)]
    assert selected != [
        names[i] for i in select_indices(names, settings.model_copy(update={"seed": 42}))
    ]
    with pytest.raises(ValueError, match="unique"):
        select_indices(["same", "same"], settings)
    with pytest.raises(ValueError, match="sufficient"):
        select_indices(names[:2], settings)


def prediction() -> dict[str, np.ndarray]:
    """Provide two source-coordinate boxes with different scores/categories."""
    return {
        "boxes": np.array([[10.0, 20.0, 110.0, 220.0], [2.0, 3.0, 4.0, 5.0]]),
        "scores": np.array([0.9, 0.4]),
        "labels": np.array([1, 2]),
    }


def test_prediction_parity_allows_order_and_bounded_numerical_error() -> None:
    expected = prediction()
    actual = {k: v[::-1].copy() for k, v in expected.items()}
    actual["boxes"] += 0.001
    actual["scores"] += 0.000001
    result = compare_predictions(
        actual, expected, Agreement(box_atol_pixels=0.01, score_atol=0.00001, rtol=0.00001)
    )
    assert result["detections"] == 2
    assert result["max_boxes_absolute_error"] == pytest.approx(0.001)


@pytest.mark.parametrize("field", ["count", "boxes", "scores", "labels", "nan"])
def test_prediction_parity_fails_on_scientifically_different_output(field: str) -> None:
    expected, actual = prediction(), prediction()
    if field == "count":
        actual = {k: v[:1] for k, v in actual.items()}
    elif field == "nan":
        actual["scores"][0] = np.nan
    else:
        actual[field][0] += 1
    with pytest.raises(ValueError):
        compare_predictions(
            actual, expected, Agreement(box_atol_pixels=0.01, score_atol=0.00001, rtol=0.00001)
        )


def test_empty_prediction_parity_is_valid() -> None:
    empty = {k: v[:0] for k, v in prediction().items()}
    result = compare_predictions(
        empty, empty, Agreement(box_atol_pixels=0.01, score_atol=0.00001, rtol=0)
    )
    assert result["detections"] == 0


@pytest.mark.parametrize(
    "update",
    [
        {"batch_size": 2},
        {"repetitions": 2},
        {"warmup_images": 0},
        {"device": "cpu"},
        {"allow_tf32": True},
    ],
)
def test_protocol_rejects_weakened_publication_contract(update: dict) -> None:
    payload = yaml.safe_load((ROOT / "configs/inference_timing_v1.yaml").read_text())
    payload.update(update, project_root=ROOT, source_path=ROOT / "configs/inference_timing_v1.yaml")
    with pytest.raises(ValueError):
        TimingConfig.model_validate(payload)


def test_historical_overwrite_namespace_is_rejected() -> None:
    payload = yaml.safe_load((ROOT / "configs/inference_timing_v1.yaml").read_text())
    payload["outputs"]["table"] = "results/tables/faster_rcnn_compute.csv"
    payload.update(project_root=ROOT, source_path=ROOT / "configs/inference_timing_v1.yaml")
    with pytest.raises(ValueError, match="namespace"):
        TimingConfig.model_validate(payload)


def test_history_tampering_is_detected(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/inference_timing_v1.yaml")
    config = config.model_copy(
        update={"project_root": tmp_path, "historical_manifest": Path("history.json")}
    )
    (tmp_path / "history.json").write_text(json.dumps({"files": {"original.csv": "0" * 64}}))
    (tmp_path / "original.csv").write_text("modified historical values")
    with pytest.raises(ValueError, match="historical artifact drift"):
        verify_history(config)


def test_hardware_gate_refuses_cpu_before_any_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(ROOT / "configs/inference_timing_v1.yaml")
    with pytest.raises(RuntimeError, match="no publication timing"):
        hardware_snapshot(config)


def test_tolerance_configuration_cannot_hide_large_prediction_changes() -> None:
    config = load_config(ROOT / "configs/inference_timing_v1.yaml")
    values = copy.deepcopy(config.agreement.model_dump())
    values["score_atol"] = 0.1
    with pytest.raises(ValueError):
        Agreement.model_validate(values)
