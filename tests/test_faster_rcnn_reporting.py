import builtins
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.models.faster_rcnn_reporting import (
    COMPUTE_FIELDNAMES,
    EPOCH_FIELDNAMES,
    VALIDATION_FIELDNAMES,
    append_epoch_metrics,
    build_benchmark_projection,
    load_epoch_metrics,
    plot_training_curves,
    sha256_file,
    summarize_inference_timings,
    summarize_model_artifact,
    summarize_parameter_counts,
    write_atomic_json,
    write_benchmark_projection,
    write_compute_metrics_csv,
    write_validation_metrics_csv,
)


def epoch_record(epoch: int = 1) -> dict:
    return {
        "run_id": "baseline-seed17",
        "seed": 17,
        "epoch": epoch,
        "optimizer_steps": epoch * 10,
        "learning_rate": 0.005,
        "train_loss_total": 1.0 / epoch,
        "train_loss_classifier": 0.4,
        "train_loss_box_reg": 0.3,
        "train_loss_objectness": 0.2,
        "train_loss_rpn_box_reg": 0.1,
        "val_precision": 0.75,
        "val_recall": 0.6,
        "val_f1": 2 / 3,
        "val_map_50": 0.7,
        "val_map_50_95": 0.4,
        "val_true_positives": 12,
        "val_false_positives": 4,
        "val_false_negatives": 8,
        "train_seconds": 100.0,
        "validation_seconds": 20.0,
        "epoch_seconds": 120.0,
        "peak_gpu_memory_mib": 7000.0,
        "is_best": True,
        "epochs_without_improvement": 0,
    }


def validation_row() -> dict:
    return dict(
        zip(
            VALIDATION_FIELDNAMES,
            (
                "baseline-seed17",
                17,
                4,
                "validation",
                0.25,
                0.5,
                0.0,
                100,
                750,
                277,
                140,
                200,
                120,
                20,
                157,
                120 / 140,
                120 / 277,
                0.57,
                0.70,
                0.40,
            ),
            strict=True,
        )
    )


def compute_row() -> dict:
    return dict(
        zip(
            COMPUTE_FIELDNAMES,
            (
                "baseline-seed17",
                17,
                640,
                640,
                "float16",
                1,
                200,
                200,
                18.0,
                55.0,
                52.0,
                65.0,
                43_000_000,
                30_000_000,
                180.0,
                "torch.utils.flop_counter; 2 FLOPs/MAC",
                170_000_000,
                162.12,
                "a" * 64,
                3600.0,
                7200.0,
            ),
            strict=True,
        )
    )


def test_atomic_json_is_stable_replaces_existing_and_rejects_nan(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "summary.json"
    write_atomic_json(destination, {"z": 1, "a": {"value": None}})

    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "value": null\n  },\n  "z": 1\n}\n'
    )
    write_atomic_json(destination, {"replaced": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"replaced": True}
    assert list(destination.parent.glob("*.tmp")) == []

    with pytest.raises(ValueError, match="Out of range float"):
        write_atomic_json(destination, {"metric": float("nan")})


def test_epoch_csv_and_jsonl_have_stable_schema_and_idempotent_append(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "epochs.csv"
    jsonl_path = tmp_path / "epochs.jsonl"
    first = epoch_record()
    first["val_precision"] = None

    append_epoch_metrics(csv_path, jsonl_path, first)
    append_epoch_metrics(csv_path, jsonl_path, first)
    second = epoch_record(2)
    second["is_best"] = False
    second["epochs_without_improvement"] = 1
    append_epoch_metrics(csv_path, jsonl_path, second)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == EPOCH_FIELDNAMES
    assert len(rows) == 2
    assert rows[0]["val_precision"] == ""
    assert rows[0]["is_best"] == "true"
    assert rows[1]["is_best"] == "false"

    json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(json_rows) == 2
    assert json_rows[0]["val_precision"] is None
    assert load_epoch_metrics(csv_path) == json_rows

    conflict = deepcopy(first)
    conflict["train_loss_total"] = 99.0
    with pytest.raises(ValueError, match="conflicting epoch"):
        append_epoch_metrics(csv_path, jsonl_path, conflict)


def test_epoch_logger_rejects_nonfinite_values_before_writing(tmp_path: Path) -> None:
    record = epoch_record()
    record["train_loss_total"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        append_epoch_metrics(tmp_path / "epochs.csv", tmp_path / "epochs.jsonl", record)

    assert not (tmp_path / "epochs.csv").exists()
    assert not (tmp_path / "epochs.jsonl").exists()


def test_artifact_digest_and_size_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "best_model.pt"
    content = b"deterministic-model-state"
    artifact.write_bytes(content)

    assert sha256_file(artifact, chunk_bytes=3) == hashlib.sha256(content).hexdigest()
    summary = summarize_model_artifact(artifact)
    assert summary == {
        "path": artifact.as_posix(),
        "size_bytes": len(content),
        "size_mib": len(content) / (1024**2),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_benchmark_projection_uses_first_epoch_and_steady_median(tmp_path: Path) -> None:
    projection = build_benchmark_projection(
        [120.0, 100.0, 110.0],
        minimum_epochs=5,
        maximum_epochs=30,
        scenario_epochs=(10, 20),
    )

    assert projection["steady_state_seconds_per_epoch"] == 105.0
    assert projection["estimated_minimum_seconds"] == 540.0
    assert projection["estimated_maximum_seconds"] == 3165.0
    assert projection["scenario_estimated_seconds"] == {"10": 1065.0, "20": 2115.0}

    destination = tmp_path / "benchmark.json"
    persisted = write_benchmark_projection(
        destination,
        [120.0, 100.0, 110.0],
        minimum_epochs=5,
        maximum_epochs=30,
        scenario_epochs=(10,),
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == persisted


def test_validation_and_compute_tables_enforce_exact_finite_schema(tmp_path: Path) -> None:
    validation_path = write_validation_metrics_csv(tmp_path / "validation.csv", [validation_row()])
    compute_path = write_compute_metrics_csv(tmp_path / "compute.csv", [compute_row()])

    assert validation_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        VALIDATION_FIELDNAMES
    )
    assert compute_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(COMPUTE_FIELDNAMES)

    malformed = validation_row()
    malformed.pop("f1")
    with pytest.raises(ValueError, match="does not match schema"):
        write_validation_metrics_csv(tmp_path / "bad.csv", [malformed])

    nonfinite = compute_row()
    nonfinite["throughput_fps"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_compute_metrics_csv(tmp_path / "bad-compute.csv", [nonfinite])


def test_parameter_and_raw_inference_timing_summaries() -> None:
    class FakeParameter:
        def __init__(self, count: int, requires_grad: bool) -> None:
            self.count = count
            self.requires_grad = requires_grad

        def numel(self) -> int:
            return self.count

    parameters = [FakeParameter(7, True), FakeParameter(3, False)]
    assert summarize_parameter_counts(parameters) == {
        "total_parameters": 10,
        "trainable_parameters": 7,
        "frozen_parameters": 3,
    }

    timings = summarize_inference_timings(
        [0.1, 0.2],
        images_per_batch=[2, 4],
    )
    assert timings["timed_batches"] == 2
    assert timings["timed_images"] == 6
    assert timings["total_seconds"] == pytest.approx(0.3)
    assert timings["throughput_fps"] == pytest.approx(20.0)
    assert timings["mean_latency_ms"] == pytest.approx(50.0)
    assert timings["p50_latency_ms"] == pytest.approx(50.0)
    assert timings["p95_latency_ms"] == pytest.approx(50.0)


def test_plotting_dependency_is_lazy_and_has_a_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "epochs.csv"
    append_epoch_metrics(csv_path, tmp_path / "epochs.jsonl", epoch_record())
    real_import = builtins.__import__

    def block_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            raise ModuleNotFoundError("blocked for test", name="matplotlib")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_matplotlib)
    with pytest.raises(RuntimeError, match="matplotlib is required"):
        plot_training_curves(csv_path, tmp_path / "curves.png")
