"""Batch 3 YOLO11s preparation, smoke, benchmark, training, and finalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import torch

from src.models.faster_rcnn_reporting import (
    build_benchmark_projection,
    summarize_model_artifact,
    write_atomic_json,
)
from src.models.yolo_config import YoloConfig, load_yolo_config, yolo_config_sha256
from src.models.yolo_data import prepare_yolo_dataset, sha256_file
from src.models.yolo_reporting import (
    evaluate_yolo_checkpoint,
    plot_yolo_training_curves,
    profile_yolo_checkpoint,
    write_yolo_tables,
)
from src.models.yolo_training import (
    MatchedDetectionTrainer,
    YoloRunMode,
    YoloRunTracker,
    build_ultralytics_train_args,
)
from src.utils.seed import initialize_reproducibility, seed_everything

ExecutionMode = Literal["prepare", "preflight", "smoke", "benchmark", "train", "finalize"]


def build_parser() -> argparse.ArgumentParser:
    """Build the Batch 3 command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/yolo.yaml"))
    parser.add_argument(
        "--mode",
        required=True,
        choices=("prepare", "preflight", "smoke", "benchmark", "train", "finalize"),
    )
    return parser


def _git_commit(project_root: Path) -> str | None:
    """Return the current Git commit without requiring Git metadata in artifacts."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _implementation_identity(config: YoloConfig) -> dict[str, Any]:
    """Hash implementation and dependency-manifest bytes that define the run."""

    roots = (
        config.project_root / "src" / "models",
        config.project_root / "src" / "meddet_benchmark",
        config.project_root / "src" / "utils",
    )
    paths = {
        path.resolve()
        for root in roots
        for path in root.rglob("*.py")
        if path.is_file()
    }
    paths |= {
        path.resolve()
        for path in (
            config.project_root / "src" / "data" / "prepare.py",
            config.project_root / "requirements.txt",
            config.project_root / "pyproject.toml",
            config.project_root / "uv.lock",
        )
        if path.is_file()
    }
    entries: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(config.project_root).as_posix()
        file_hash = sha256_file(path)
        entries.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "git_commit": _git_commit(config.project_root),
        "source_manifest_sha256": digest.hexdigest(),
        "source_files": entries,
    }


def _driver_version() -> str | None:
    """Return the active NVIDIA driver version."""

    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if completed.returncode == 0 and lines else None


def _execution_identity(config: YoloConfig) -> dict[str, Any]:
    """Capture the exact software, GPU, precision, batch, and augmentation identity."""

    if not torch.cuda.is_available():
        raise RuntimeError("Batch 3 requires CUDA")
    if config.runtime.amp_dtype == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured bfloat16 AMP is unsupported on this CUDA device")
        torch.set_autocast_dtype("cuda", torch.bfloat16)
    if torch.get_autocast_dtype("cuda") != torch.bfloat16:
        raise RuntimeError("failed to configure the requested CUDA autocast dtype")
    properties = torch.cuda.get_device_properties(config.runtime.device)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": version("torchvision"),
        "ultralytics_version": version("ultralytics"),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": _driver_version(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "loss_dtype": config.runtime.loss_dtype,
        "physical_batch_size": config.runtime.batch_size,
        "effective_batch_size": config.runtime.nominal_batch_size,
        "input_size": config.model.input_size,
        "batch_norm_policy": config.model.batch_norm_policy,
        "augmentation_policy": config.training.augmentation.policy,
        "workers": config.runtime.workers,
        "cache": config.runtime.cache,
    }


def _preflight(config: YoloConfig) -> dict[str, Any]:
    """Validate data, dependency, weight, CUDA, and augmentation readiness."""

    dataset = prepare_yolo_dataset(config)
    weights = config.resolve(config.model.weights_path)
    if version("ultralytics") != config.model.ultralytics_version:
        raise RuntimeError("installed Ultralytics version does not match configs/yolo.yaml")
    if not weights.is_file():
        raise FileNotFoundError(f"missing pinned YOLO checkpoint: {weights}")
    execution = _execution_identity(config)
    return {
        "status": "ready",
        "config_sha256": yolo_config_sha256(config),
        "dataset": dataset,
        "execution_identity": execution,
        "implementation_identity": _implementation_identity(config),
        "pretrained_weights": summarize_model_artifact(weights),
        "test_split_accessed": False,
    }


def _guard_new_run(config: YoloConfig, mode: YoloRunMode) -> None:
    """Refuse to merge an experiment with previous metrics or weights."""

    run_dir = config.run_dir(mode)
    guarded = (
        run_dir / "results.csv",
        run_dir / config.outputs.summary_name,
        run_dir / config.outputs.epoch_timing_name,
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
    )
    existing = [path for path in guarded if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing YOLO run artifacts: "
            + ", ".join(path.as_posix() for path in existing)
        )


def _register_tracker(model: Any, tracker: YoloRunTracker) -> None:
    """Attach all matched-policy callbacks to one Ultralytics model."""

    model.add_callback("on_train_start", tracker.on_train_start)
    model.add_callback("on_train_epoch_start", tracker.on_train_epoch_start)
    model.add_callback("on_train_batch_end", tracker.on_train_batch_end)
    model.add_callback("on_fit_epoch_end", tracker.on_fit_epoch_end)
    model.add_callback("on_train_end", tracker.on_train_end)


def _load_result_rows(path: Path) -> list[dict[str, str]]:
    """Read Ultralytics' per-epoch CSV with normalized headers."""

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _run_ultralytics(
    config: YoloConfig,
    mode: YoloRunMode,
    dataset_identity: dict[str, Any],
) -> tuple[Any, YoloRunTracker, dict[str, Any]]:
    """Execute one bounded Ultralytics run from pinned pretrained weights."""

    from ultralytics import YOLO

    _guard_new_run(config, mode)
    run_dir = config.run_dir(mode)
    initialize_reproducibility(
        config.seed,
        run_dir,
        deterministic=config.runtime.deterministic,
        warn_only=True,
    )
    train_args = build_ultralytics_train_args(config, mode)
    identity = {
        "config_sha256": yolo_config_sha256(config),
        "dataset_identity": dataset_identity,
        "execution_identity": _execution_identity(config),
        "implementation_identity": _implementation_identity(config),
        "pretrained_weights": summarize_model_artifact(config.resolve(config.model.weights_path)),
        "train_args": train_args,
        "test_split_accessed": False,
    }
    write_atomic_json(run_dir / "resolved_experiment.json", identity)
    model = YOLO(config.resolve(config.model.weights_path).as_posix())
    tracker = YoloRunTracker(config, mode)
    _register_tracker(model, tracker)
    started = time.perf_counter()
    metrics = model.train(trainer=MatchedDetectionTrainer, **train_args)
    process_seconds = time.perf_counter() - started
    rows = _load_result_rows(run_dir / "results.csv")
    expected_epochs = {
        "smoke": config.smoke.epochs,
        "benchmark": config.benchmark.epochs,
        "train": None,
    }[mode]
    if expected_epochs is not None and len(rows) != expected_epochs:
        raise RuntimeError(f"{mode} did not complete its configured epoch count")
    if len(rows) != len(tracker.records):
        raise RuntimeError("Ultralytics CSV and project timing log have different epoch counts")
    result = {
        **identity,
        "mode": mode,
        "run_id": config.run_name(mode),
        "completed_epochs": len(rows),
        "epoch_seconds": [record["epoch_seconds"] for record in tracker.records],
        "process_seconds": process_seconds,
        "training_wall_seconds": tracker.total_wall_seconds,
        "peak_train_gpu_memory_mib": max(
            record["peak_gpu_memory_mib"] for record in tracker.records
        ),
        "native_metrics": metrics.results_dict if hasattr(metrics, "results_dict") else None,
        "test_split_accessed": False,
    }
    return model, tracker, result


def _write_benchmark(config: YoloConfig, result: dict[str, Any]) -> dict[str, Any]:
    """Write the real-data three-epoch projection for the full Batch 3 run."""

    projection = build_benchmark_projection(
        result["epoch_seconds"],
        minimum_epochs=config.training.early_stopping.min_epochs,
        maximum_epochs=config.training.max_epochs,
        scenario_epochs=(config.training.max_epochs,),
    )
    payload = {
        **projection,
        "config_sha256": result["config_sha256"],
        "dataset_identity": result["dataset_identity"],
        "execution_identity": result["execution_identity"],
        "implementation_identity": result["implementation_identity"],
        "pretrained_weights": result["pretrained_weights"],
        "completed_epochs": result["completed_epochs"],
        "run_id": result["run_id"],
        "restart_full_run_from_pretrained_weights": True,
        "test_split_accessed": False,
    }
    path = config.run_dir("benchmark") / config.outputs.benchmark_estimate_name
    write_atomic_json(path, payload)
    result["benchmark_estimate"] = payload
    write_atomic_json(config.run_dir("benchmark") / config.outputs.summary_name, result)
    return result


def _require_benchmark(
    config: YoloConfig,
    dataset_identity: dict[str, Any],
    execution_identity: dict[str, Any],
    implementation_identity: dict[str, Any],
) -> dict[str, Any]:
    """Bind the full run to the current completed timing benchmark."""

    path = config.run_dir("benchmark") / config.outputs.benchmark_estimate_name
    if not path.is_file():
        raise FileNotFoundError("full YOLO training requires the configured benchmark estimate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": yolo_config_sha256(config),
        "dataset_identity": dataset_identity,
        "execution_identity": execution_identity,
        "implementation_identity": implementation_identity,
        "completed_epochs": config.benchmark.epochs,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"YOLO benchmark {key} no longer matches the full-run identity")
    return payload


def _copy_checkpoint(source: Path, destination: Path) -> Path:
    """Copy a checkpoint once and refuse conflicting output."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"conflicting YOLO checkpoint exists: {destination}")
        return destination
    shutil.copy2(source, destination)
    return destination


def _finalize(config: YoloConfig, base_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create final shared metrics, compute profile, tables, curves, and summary."""

    run_dir = config.run_dir("train")
    results_csv = run_dir / "results.csv"
    source_best = run_dir / "weights" / "best.pt"
    source_last = run_dir / "weights" / "last.pt"
    if not results_csv.is_file() or not source_best.is_file() or not source_last.is_file():
        raise FileNotFoundError("YOLO finalization requires results.csv plus best.pt and last.pt")
    rows = _load_result_rows(results_csv)
    native_best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    best_epoch = int(native_best["epoch"])
    checkpoint_dir = config.resolve(config.outputs.checkpoint_dir)
    best_checkpoint = _copy_checkpoint(source_best, checkpoint_dir / "best_model.pt")
    last_checkpoint = _copy_checkpoint(source_last, checkpoint_dir / "last_state.pt")

    validation = evaluate_yolo_checkpoint(config, best_checkpoint)
    profile = profile_yolo_checkpoint(config, best_checkpoint)
    timing_records = [
        json.loads(line)
        for line in (run_dir / config.outputs.epoch_timing_name)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    training_seconds = sum(float(record["epoch_seconds"]) for record in timing_records)
    peak_memory = max(float(record["peak_gpu_memory_mib"]) for record in timing_records)
    model_artifact = summarize_model_artifact(best_checkpoint)
    tables = write_yolo_tables(
        config,
        run_id=config.outputs.train_run_name,
        best_epoch=best_epoch,
        validation=validation,
        profile=profile,
        model_artifact=model_artifact,
        training_seconds=training_seconds,
        peak_gpu_memory_mib=peak_memory,
    )
    curve_path = plot_yolo_training_curves(
        results_csv,
        config.resolve(config.outputs.training_curves),
        best_epoch=best_epoch,
    )
    artifacts = {
        "best_checkpoint": model_artifact,
        "last_checkpoint": summarize_model_artifact(last_checkpoint),
        "validation_table": summarize_model_artifact(tables["validation_table"]),
        "compute_table": summarize_model_artifact(tables["compute_table"]),
        "training_curves": summarize_model_artifact(curve_path),
    }
    summary = dict(base_summary or {})
    if not summary:
        summary_path = run_dir / config.outputs.summary_name
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resolved_path = run_dir / "resolved_experiment.json"
        if not resolved_path.is_file():
            raise FileNotFoundError("YOLO finalization requires resolved_experiment.json")
        training_identity = json.loads(resolved_path.read_text(encoding="utf-8"))
        summary.update(training_identity)
        benchmark_path = config.run_dir("benchmark") / config.outputs.benchmark_estimate_name
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        for key in (
            "config_sha256",
            "dataset_identity",
            "execution_identity",
            "implementation_identity",
            "pretrained_weights",
        ):
            if benchmark.get(key) != training_identity.get(key):
                raise ValueError(f"recorded YOLO training {key} does not match its benchmark")
        summary["approved_benchmark"] = {
            "path": benchmark_path.as_posix(),
            "sha256": sha256_file(benchmark_path),
            "validated_payload": benchmark,
        }
        summary.update(
            {
                "mode": "train",
                "run_id": config.outputs.train_run_name,
                "epoch_seconds": [
                    float(record["epoch_seconds"]) for record in timing_records
                ],
                "training_wall_seconds": training_seconds,
            }
        )
    summary.update(
        {
            "status": "complete",
            "stop_reason": (
                "early_stopping_validation_map"
                if len(rows) < config.training.max_epochs
                else "maximum_epochs"
            ),
            "completed_epochs": len(rows),
            "best_epoch": best_epoch,
            "native_best_epoch": best_epoch,
            "native_best_map_50_95": float(native_best["metrics/mAP50-95(B)"]),
            "reporting_implementation_identity": _implementation_identity(config),
            "shared_validation": validation,
            "profile": profile,
            "training_seconds": training_seconds,
            "peak_train_gpu_memory_mib": peak_memory,
            "artifacts": artifacts,
            "test_split_accessed": False,
        }
    )
    write_atomic_json(run_dir / config.outputs.summary_name, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Execute exactly one Batch 3 workflow mode."""

    args = build_parser().parse_args(argv)
    config = load_yolo_config(args.config)
    mode: ExecutionMode = args.mode
    if mode == "prepare":
        print(json.dumps(prepare_yolo_dataset(config), indent=2, sort_keys=True))
        return 0
    if mode == "preflight":
        print(json.dumps(_preflight(config), indent=2, sort_keys=True))
        return 0
    if mode == "finalize":
        print(json.dumps(_finalize(config), indent=2, sort_keys=True))
        return 0

    dataset_identity = prepare_yolo_dataset(config)
    if mode == "train":
        # Establish deterministic process state before any CUDA-backed identity probe.
        seed_everything(config.seed)
        execution = _execution_identity(config)
        implementation = _implementation_identity(config)
        benchmark = _require_benchmark(config, dataset_identity, execution, implementation)
        _model, _tracker, result = _run_ultralytics(config, mode, dataset_identity)
        result["approved_benchmark"] = {
            "path": (
                config.run_dir("benchmark") / config.outputs.benchmark_estimate_name
            ).as_posix(),
            "sha256": sha256_file(
                config.run_dir("benchmark") / config.outputs.benchmark_estimate_name
            ),
            "validated_payload": benchmark,
        }
        result["status"] = "trained_pending_finalization"
        write_atomic_json(config.run_dir("train") / config.outputs.summary_name, result)
        final = _finalize(config, result)
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0

    _model, _tracker, result = _run_ultralytics(config, mode, dataset_identity)
    if mode == "benchmark":
        result = _write_benchmark(config, result)
    else:
        result["status"] = "complete"
        write_atomic_json(config.run_dir("smoke") / config.outputs.summary_name, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
