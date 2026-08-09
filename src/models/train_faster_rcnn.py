"""Batch 2 Faster R-CNN preflight, smoke, benchmark, and full-run entry point."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.models.faster_rcnn_config import (
    FasterRCNNConfig,
    RunMode,
    config_fingerprint,
    load_faster_rcnn_config,
    serializable_config,
)
from src.models.faster_rcnn_data import CocoDetectionDataset
from src.models.faster_rcnn_reporting import (
    append_epoch_metrics,
    build_benchmark_projection,
    plot_training_curves,
    sha256_file,
    summarize_inference_timings,
    summarize_model_artifact,
    summarize_parameter_counts,
    write_atomic_json,
    write_compute_metrics_csv,
    write_validation_metrics_csv,
)
from src.models.faster_rcnn_training import EarlyStopper


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract for the safe run modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/faster_rcnn.yaml"))
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "benchmark", "train", "finalize"),
        required=True,
    )
    parser.add_argument(
        "--approved-benchmark",
        type=Path,
        help=(
            "Benchmark-estimate JSON explicitly approved by the user; required for train mode"
        ),
    )
    return parser


def _validate_dataset_config_paths(
    dataset: CocoDetectionDataset,
    config: FasterRCNNConfig,
    *,
    split: str,
) -> None:
    """Ensure the loader-resolved canonical paths equal the Batch 2 config paths."""

    annotation_key = {
        "train": "train_annotations",
        "val": "val_annotations",
    }[split]
    configured_annotation = config.resolve(getattr(config.data, annotation_key)).resolve()
    configured_images = config.resolve(config.data.images_dir).resolve()
    if dataset.annotation_file.resolve() != configured_annotation:
        raise ValueError(
            f"Dataset and Faster R-CNN configs disagree on {split} annotations: "
            f"{dataset.annotation_file} != {configured_annotation}"
        )
    if dataset.images_dir.resolve() != configured_images:
        raise ValueError(
            "Dataset and Faster R-CNN configs disagree on processed images: "
            f"{dataset.images_dir} != {configured_images}"
        )


def _load_datasets(
    config: FasterRCNNConfig,
    *,
    mode: RunMode,
) -> tuple[CocoDetectionDataset, CocoDetectionDataset]:
    """Load train/validation only and aggregate completeness failures."""

    dataset_config = config.resolve(config.data.dataset_config)
    train_dataset = CocoDetectionDataset(
        dataset_config,
        "train",
        mode="partial",
        project_root=config.project_root,
    )
    val_dataset = CocoDetectionDataset(
        dataset_config,
        "val",
        mode="partial",
        project_root=config.project_root,
    )
    _validate_dataset_config_paths(train_dataset, config, split="train")
    _validate_dataset_config_paths(val_dataset, config, split="val")
    if (
        train_dataset.category_names != val_dataset.category_names
        or train_dataset.category_id_to_label != val_dataset.category_id_to_label
    ):
        raise ValueError("train and validation category mappings do not match")

    if mode in {"benchmark", "train", "finalize"}:
        missing_by_split = {
            "train": train_dataset.preflight.missing_files,
            "validation": val_dataset.preflight.missing_files,
        }
        if any(missing_by_split.values()):
            details = []
            for split, missing_files in missing_by_split.items():
                details.append(f"{split}: {len(missing_files)} missing")
                details.extend(f"  {path}" for path in missing_files)
            raise FileNotFoundError(
                "Timed/final Faster R-CNN modes require every configured train and "
                "validation image. Exact missing paths:\n" + "\n".join(details)
            )
    return train_dataset, val_dataset


def _dataset_summary(
    train_dataset: CocoDetectionDataset,
    val_dataset: CocoDetectionDataset,
) -> dict[str, Any]:
    """Return JSON-safe preflight counts without inspecting the test split."""

    def split_summary(dataset: CocoDetectionDataset) -> dict[str, Any]:
        manifest = hashlib.sha256()
        available_bytes = 0
        for index, record in enumerate(dataset.records):
            image_path = dataset.image_path(index)
            if not image_path.is_file():
                continue
            size_bytes = image_path.stat().st_size
            available_bytes += size_bytes
            manifest.update(record.file_name.encode("utf-8"))
            manifest.update(b"\0")
            manifest.update(str(size_bytes).encode("ascii"))
            manifest.update(b"\0")
            manifest.update(sha256_file(image_path).encode("ascii"))
            manifest.update(b"\n")
        return {
            "expected_images": dataset.preflight.expected_images,
            "available_images": dataset.preflight.available_images,
            "missing_images": len(dataset.preflight.missing_files),
            "missing_file_paths": [
                path.as_posix() for path in dataset.preflight.missing_files
            ],
            "available_image_bytes": available_bytes,
            "available_image_manifest_sha256": manifest.hexdigest(),
            "annotations": sum(len(record.annotations) for record in dataset.records),
            "annotation_file": dataset.annotation_file.as_posix(),
            "annotation_sha256": sha256_file(dataset.annotation_file),
        }

    return {
        "categories": {
            str(category_id): name
            for category_id, name in train_dataset.category_names.items()
        },
        "category_id_to_model_label": {
            str(category_id): label
            for category_id, label in train_dataset.category_id_to_label.items()
        },
        "train": split_summary(train_dataset),
        "validation": split_summary(val_dataset),
        "test_split_accessed": False,
    }


def _available_subset(dataset: CocoDetectionDataset) -> Any:
    """Return a Torch subset containing only locally available smoke images."""

    from torch.utils.data import Subset

    indices = [
        index for index in range(len(dataset)) if dataset.image_path(index).is_file()
    ]
    if not indices:
        raise RuntimeError(f"No images are locally available for smoke split {dataset.split}")
    return Subset(dataset, indices)


def _make_data_loader(
    dataset: Any,
    config: FasterRCNNConfig,
    *,
    shuffle: bool,
    batch_size: int | None = None,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
) -> Any:
    """Create a deterministic Windows-safe detection DataLoader."""

    from torch.utils.data import DataLoader

    from src.models.faster_rcnn_data import detection_collate_fn
    from src.utils.seed import make_torch_generator, seed_worker

    worker_count = config.runtime.num_workers if num_workers is None else num_workers
    keep_workers = (
        config.runtime.persistent_workers
        if persistent_workers is None
        else persistent_workers
    )
    if worker_count == 0:
        keep_workers = False
    return DataLoader(
        dataset,
        batch_size=(config.runtime.batch_size if batch_size is None else batch_size),
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=config.runtime.pin_memory,
        persistent_workers=keep_workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed),
        drop_last=False,
    )


def _shutdown_data_loader(data_loader: Any) -> None:
    """Release persistent worker processes before the profiling phase."""

    iterator = getattr(data_loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _require_cuda(config: FasterRCNNConfig) -> tuple[Any, Any]:
    """Resolve the mandatory CUDA/float16 AMP device or fail explicitly."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Batch 2 requires CUDA, but torch.cuda.is_available() is false")
    device = torch.device(config.runtime.device)
    amp_dtype = torch.float16
    return device, amp_dtype


def _make_optimizer_and_scheduler(model: Any, config: FasterRCNNConfig) -> tuple[Any, Any]:
    """Create the configured SGD optimizer and validation-mAP scheduler."""

    import torch

    optimizer_settings = config.training.optimizer
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=optimizer_settings.learning_rate,
        momentum=optimizer_settings.momentum,
        weight_decay=optimizer_settings.weight_decay,
        nesterov=optimizer_settings.nesterov,
    )
    scheduler_settings = config.training.scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_settings.mode,
        factor=scheduler_settings.factor,
        patience=scheduler_settings.patience,
        threshold=scheduler_settings.threshold,
        min_lr=scheduler_settings.min_learning_rate,
    )
    return optimizer, scheduler


def _train_one_epoch(
    model: Any,
    data_loader: Any,
    optimizer: Any,
    scaler: Any,
    *,
    device: Any,
    amp_dtype: Any,
    config: FasterRCNNConfig,
    max_batches: int | None,
) -> tuple[dict[str, float], int, float, float | None]:
    """Train one epoch with AMP and correctly normalized gradient accumulation."""

    import torch

    from src.models.faster_rcnn_model import apply_training_mode

    apply_training_mode(model, config.model)
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(data_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if total_batches <= 0:
        raise RuntimeError("training DataLoader contains no batches")

    accumulation = config.runtime.gradient_accumulation_steps
    loss_sums = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    total_loss_sum = 0.0
    image_count = 0
    optimizer_steps = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (images, targets) in enumerate(data_loader):
        if batch_index >= total_batches:
            break
        device_images = [image.to(device, non_blocking=True) for image in images]
        device_targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]
        window_start = (batch_index // accumulation) * accumulation
        window_size = min(accumulation, total_batches - window_start)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=config.runtime.amp,
        ):
            loss_dict = model(device_images, device_targets)
            total_loss = sum(loss_dict.values())
            normalized_loss = total_loss / window_size
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_index + 1}: {total_loss.item()}"
            )
        scaler.scale(normalized_loss).backward()

        batch_images = len(images)
        image_count += batch_images
        total_loss_sum += float(total_loss.detach().item()) * batch_images
        for key in loss_sums:
            if key not in loss_dict:
                raise RuntimeError(f"Faster R-CNN loss dictionary is missing {key}")
            loss_sums[key] += float(loss_dict[key].detach().item()) * batch_images

        end_of_window = (batch_index + 1) % accumulation == 0
        final_batch = batch_index + 1 == total_batches
        if end_of_window or final_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.optimizer.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if (batch_index + 1) % config.training.log_every_batches == 0:
            print(
                f"train batch {batch_index + 1}/{total_batches} "
                f"loss={float(total_loss.detach().item()):.5f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mib: float | None = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory_mib = None
    elapsed_seconds = time.perf_counter() - started
    means = {key: value / image_count for key, value in loss_sums.items()}
    means["loss_total"] = total_loss_sum / image_count
    return means, optimizer_steps, elapsed_seconds, peak_memory_mib


def _git_commit(project_root: Path) -> str | None:
    """Return the current commit without making Git a runtime dependency."""

    try:
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
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _implementation_identity(config: FasterRCNNConfig) -> dict[str, Any]:
    """Fingerprint the source and dependency manifests that define the run."""

    source_roots = (
        config.project_root / "src" / "models",
        config.project_root / "src" / "meddet_benchmark",
        config.project_root / "src" / "utils",
    )
    files = sorted(
        {
            path.resolve()
            for root in source_roots
            for path in root.rglob("*.py")
            if path.is_file()
        }
        | {
            path.resolve()
            for path in (
                config.project_root / "src" / "data" / "prepare.py",
                config.project_root / "requirements.txt",
                config.project_root / "pyproject.toml",
                config.project_root / "uv.lock",
            )
            if path.is_file()
        },
        key=lambda path: path.as_posix(),
    )
    manifest = hashlib.sha256()
    entries: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(config.project_root).as_posix()
        digest = sha256_file(path)
        entries.append({"path": relative, "sha256": digest})
        manifest.update(relative.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(digest.encode("ascii"))
        manifest.update(b"\n")
    return {
        "git_commit": _git_commit(config.project_root),
        "source_manifest_sha256": manifest.hexdigest(),
        "source_files": entries,
    }


def _nvidia_driver_version() -> str | None:
    """Return the active NVIDIA driver version when nvidia-smi is available."""

    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return versions[0] if completed.returncode == 0 and versions else None


def _execution_identity(config: FasterRCNNConfig) -> dict[str, Any]:
    """Describe the exact software, GPU, and shape settings used for timing."""

    import platform
    from importlib.metadata import PackageNotFoundError, version

    import torch
    import torchvision

    def installed_version(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    required_distributions = (
        "matplotlib",
        "numpy",
        "Pillow",
        "pycocotools",
        "pydantic",
        "PyYAML",
    )
    package_versions = {
        distribution: installed_version(distribution)
        for distribution in required_distributions
    }
    missing_distributions = [
        name for name, installed in package_versions.items() if installed is None
    ]
    if missing_distributions:
        raise RuntimeError(
            "Batch 2 environment is missing required distributions: "
            + ", ".join(missing_distributions)
        )
    device, _amp_dtype = _require_cuda(config)
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions,
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": _nvidia_driver_version(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "amp": config.runtime.amp,
        "amp_dtype": config.runtime.amp_dtype,
        "microbatch_size": config.runtime.batch_size,
        "gradient_accumulation_steps": config.runtime.gradient_accumulation_steps,
        "effective_batch_size": config.runtime.effective_batch_size,
        "input_min_size": config.model.min_size,
        "input_max_size": config.model.max_size,
        "batch_norm_policy": config.model.batch_norm_policy,
    }


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write a Torch checkpoint by atomic same-directory replacement."""

    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _rng_state() -> dict[str, Any]:
    """Capture process RNG state for an auditable last-state checkpoint."""

    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _checkpoint_payload(
    *,
    model: Any,
    config: FasterRCNNConfig,
    train_dataset: CocoDetectionDataset,
    epoch: int,
    validation_metrics: dict[str, Any],
    dataset_identity: dict[str, Any],
) -> dict[str, Any]:
    """Build the model-only best-checkpoint payload."""

    return {
        "architecture": config.model.architecture,
        "config_sha256": config_fingerprint(config),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "category_id_to_model_label": train_dataset.category_id_to_label,
        "category_names": train_dataset.category_names,
        "validation_metrics": validation_metrics,
        "dataset_identity": dataset_identity,
    }


def _last_state_payload(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    stopper: EarlyStopper,
    config: FasterRCNNConfig,
    epoch: int,
    global_optimizer_steps: int,
    dataset_identity: dict[str, Any],
) -> dict[str, Any]:
    """Build an auditable last-state payload for failure analysis."""

    return {
        "config_sha256": config_fingerprint(config),
        "epoch": epoch,
        "global_optimizer_steps": global_optimizer_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "early_stopper": stopper.state_dict(),
        "rng_state": _rng_state(),
        "dataset_identity": dataset_identity,
    }


def _ensure_new_run(config: FasterRCNNConfig, *, mode: RunMode) -> None:
    """Refuse to merge a new experiment with an existing metric history."""

    guarded = (
        config.outputs.epoch_csv_path,
        config.outputs.epoch_jsonl_path,
        config.outputs.summary_path,
        config.outputs.benchmark_estimate_path,
    )
    existing = [
        config.run_artifact_path(mode, path)
        for path in guarded
        if config.run_artifact_path(mode, path).exists()
    ]
    if existing:
        listing = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing {mode} run artifacts: {listing}")


def _approved_benchmark(
    config: FasterRCNNConfig,
    approved_path: Path | None,
    *,
    dataset_identity: dict[str, Any],
    execution_identity: dict[str, Any],
    implementation_identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate sign-off against exact config, data, code, software, and GPU."""

    if approved_path is None:
        raise ValueError("train mode requires --approved-benchmark after user sign-off")
    resolved = approved_path.resolve()
    expected = config.run_artifact_path(
        "benchmark", config.outputs.benchmark_estimate_path
    ).resolve()
    if resolved != expected:
        raise ValueError(f"approved benchmark must be the configured artifact: {expected}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != config_fingerprint(config):
        raise ValueError("approved benchmark was produced by a different configuration")
    if payload.get("completed_epochs") != config.benchmark.epochs:
        raise ValueError("approved benchmark did not complete the configured epoch count")
    comparisons = {
        "dataset_identity": dataset_identity,
        "execution_identity": execution_identity,
        "implementation_identity": implementation_identity,
    }
    for key, current in comparisons.items():
        if payload.get(key) != current:
            raise ValueError(
                f"approved benchmark {key} no longer matches the current run environment"
            )
    return payload


def _profile_inference(
    model: Any,
    data_loader: Any,
    *,
    device: Any,
    amp_dtype: Any,
    config: FasterRCNNConfig,
) -> tuple[dict[str, Any], float, str]:
    """Measure synchronized model-only FPS and supported-operation FLOPs."""

    import torch

    model.eval()
    iterator = iter(data_loader)
    timings: list[float] = []
    images_per_batch: list[int] = []
    if not config.profiling.profile_flops:
        raise RuntimeError("GFLOP profiling is mandatory for the Batch 2 baseline")
    flop_gflops: float | None = None
    flop_method = ""
    with torch.inference_mode():
        for index in range(config.profiling.warmup_batches + config.profiling.timed_batches):
            try:
                images, _targets = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                images, _targets = next(iterator)
            device_images = [image.to(device, non_blocking=True) for image in images]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=config.runtime.amp,
            ):
                model(device_images)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if index >= config.profiling.warmup_batches:
                timings.append(elapsed)
                images_per_batch.append(len(images))

            if index == 0:
                try:
                    from torch.utils.flop_counter import FlopCounterMode

                    with (
                        FlopCounterMode(display=False) as counter,
                        torch.amp.autocast(
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=config.runtime.amp,
                        ),
                    ):
                        model(device_images)
                    flop_gflops = float(counter.get_total_flops()) / 1e9
                    flop_method = (
                        "torch.utils.flop_counter registered-operation FLOPs; multiply-add "
                        "formulas count 2 FLOPs; proposal-dependent single validation image; "
                        "unsupported operations such as ROIAlign/NMS excluded"
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Mandatory GFLOP profiling failed; rerun finalization after fixing "
                        f"{type(error).__name__}: {error}"
                    ) from error
    timing_summary = summarize_inference_timings(
        timings,
        images_per_batch=images_per_batch,
    )
    if flop_gflops is None or not math.isfinite(flop_gflops) or flop_gflops <= 0:
        raise RuntimeError("Mandatory GFLOP profiling did not produce a positive finite value")
    return timing_summary, flop_gflops, flop_method


def _final_artifacts(
    *,
    model: Any,
    train_dataset: CocoDetectionDataset,
    val_dataset: CocoDetectionDataset,
    profile_loader: Any,
    device: Any,
    amp_dtype: Any,
    config: FasterRCNNConfig,
    best_checkpoint: Path,
    best_epoch: int,
    best_metrics: dict[str, Any],
    training_seconds: float,
    peak_train_gpu_memory_mib: float,
) -> dict[str, Any]:
    """Generate the final validation, compute, and curve artifacts."""

    import torch

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    timing, estimated_gflops, flop_method = _profile_inference(
        model,
        profile_loader,
        device=device,
        amp_dtype=amp_dtype,
        config=config,
    )
    parameter_summary = summarize_parameter_counts(model.parameters())
    artifact_summary = summarize_model_artifact(best_checkpoint)

    validation_table = config.resolve(config.outputs.validation_table_path)
    compute_table = config.resolve(config.outputs.compute_table_path)
    curves_path = config.resolve(config.outputs.training_curves_path)
    write_validation_metrics_csv(
        validation_table,
        [
            {
                "run_id": config.outputs.train_run_name,
                "seed": config.seed,
                "best_epoch": best_epoch,
                "split": "validation",
                "operating_point_score_threshold": config.evaluation.score_threshold,
                "operating_point_match_iou_threshold": (
                    config.evaluation.match_iou_threshold
                ),
                "coco_minimum_score": config.evaluation.coco_minimum_score,
                "max_detections": config.evaluation.max_detections,
                "image_count": best_metrics["val_image_count"],
                "target_count": best_metrics["val_annotation_count"],
                "operating_point_prediction_count": best_metrics[
                    "val_prediction_count"
                ],
                "coco_prediction_count": best_metrics[
                    "val_coco_prediction_count"
                ],
                "true_positives": best_metrics["val_true_positives"],
                "false_positives": best_metrics["val_false_positives"],
                "false_negatives": best_metrics["val_false_negatives"],
                "precision": best_metrics["val_precision"],
                "recall": best_metrics["val_recall"],
                "f1": best_metrics["val_f1"],
                "map_50": best_metrics["val_map_50"],
                "map_50_95": best_metrics["val_map_50_95"],
            }
        ],
    )
    write_compute_metrics_csv(
        compute_table,
        [
            {
                "run_id": config.outputs.train_run_name,
                "seed": config.seed,
                "input_min_size": config.model.min_size,
                "input_max_size": config.model.max_size,
                "amp_dtype": config.runtime.amp_dtype,
                "inference_batch_size": config.profiling.batch_size,
                "timed_batches": timing["timed_batches"],
                "timed_images": timing["timed_images"],
                "throughput_fps": timing["throughput_fps"],
                "mean_latency_ms": timing["mean_latency_ms"],
                "p50_latency_ms": timing["p50_latency_ms"],
                "p95_latency_ms": timing["p95_latency_ms"],
                "total_parameters": parameter_summary["total_parameters"],
                "trainable_parameters": parameter_summary["trainable_parameters"],
                "estimated_gflops": estimated_gflops,
                "flop_count_method": flop_method,
                "model_size_bytes": artifact_summary["size_bytes"],
                "model_size_mib": artifact_summary["size_mib"],
                "model_sha256": artifact_summary["sha256"],
                "training_seconds": training_seconds,
                "peak_train_gpu_memory_mib": peak_train_gpu_memory_mib,
            }
        ],
    )
    plot_training_curves(
        config.run_artifact_path("train", config.outputs.epoch_csv_path), curves_path
    )
    file_integrity = {
        "validation_table": summarize_model_artifact(validation_table),
        "compute_table": summarize_model_artifact(compute_table),
        "training_curves": summarize_model_artifact(curves_path),
    }
    return {
        "validation_table": validation_table.as_posix(),
        "compute_table": compute_table.as_posix(),
        "training_curves": curves_path.as_posix(),
        "inference": timing,
        "estimated_gflops": estimated_gflops,
        "flop_count_method": flop_method,
        "parameters": parameter_summary,
        "model_artifact": artifact_summary,
        "file_integrity": file_integrity,
    }


def _run_training(
    *,
    mode: RunMode,
    config: FasterRCNNConfig,
    train_dataset: CocoDetectionDataset,
    val_dataset: CocoDetectionDataset,
    dataset_summary: dict[str, Any],
    approved_benchmark: Path | None = None,
) -> dict[str, Any]:
    """Execute one bounded smoke/benchmark/full run."""

    if mode not in {"smoke", "benchmark", "train"}:
        raise ValueError(f"training mode is not executable: {mode}")

    import torch

    from src.models.faster_rcnn_evaluation import evaluate_model
    from src.models.faster_rcnn_model import build_faster_rcnn
    from src.utils.seed import initialize_reproducibility, log_run_environment

    run_dir = config.run_dir(mode)
    _ensure_new_run(config, mode=mode)
    seed_report = initialize_reproducibility(
        config.seed,
        run_dir,
        deterministic=config.runtime.deterministic,
        warn_only=config.runtime.deterministic_warn_only,
        log_environment=False,
    )

    device, amp_dtype = _require_cuda(config)
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    implementation_identity = _implementation_identity(config)
    execution_identity = _execution_identity(config)
    approved_benchmark_payload: dict[str, Any] | None = None
    approved_benchmark_provenance: dict[str, Any] | None = None
    if mode == "train":
        approved_benchmark_payload = _approved_benchmark(
            config,
            approved_benchmark,
            dataset_identity=dataset_summary,
            execution_identity=execution_identity,
            implementation_identity=implementation_identity,
        )
        if approved_benchmark is None:
            raise AssertionError("validated train approval path unexpectedly missing")
        approved_path = approved_benchmark.resolve()
        approved_benchmark_provenance = {
            "path": approved_path.as_posix(),
            "sha256": sha256_file(approved_path),
            "validated_payload": approved_benchmark_payload,
        }
    elif approved_benchmark is not None:
        raise ValueError("approved_benchmark is only valid in train mode")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_run_environment(run_dir, seed_report)
    write_atomic_json(
        config.run_artifact_path(mode, config.outputs.resolved_config_path),
        serializable_config(config),
    )
    if mode == "smoke":
        train_input = _available_subset(train_dataset)
        val_input = _available_subset(val_dataset)
        use_pretrained_weights = config.smoke.use_pretrained_weights
        epochs = 1
        max_train_batches = config.smoke.max_train_batches
        max_val_batches = config.smoke.max_val_batches
    else:
        train_input = train_dataset
        val_input = val_dataset
        use_pretrained_weights = True
        epochs = config.benchmark.epochs if mode == "benchmark" else config.training.max_epochs
        max_train_batches = None
        max_val_batches = None

    train_loader = _make_data_loader(
        train_input,
        config,
        shuffle=True,
        persistent_workers=config.runtime.persistent_workers,
    )
    val_loader = _make_data_loader(
        val_input,
        config,
        shuffle=False,
        persistent_workers=config.runtime.validation_persistent_workers,
    )
    model, weight_metadata = build_faster_rcnn(
        train_dataset.num_foreground_classes,
        config.model,
        use_pretrained_weights=use_pretrained_weights,
    )
    model.to(device)
    optimizer, scheduler = _make_optimizer_and_scheduler(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=config.runtime.amp)
    stopper_settings = config.training.early_stopping
    stopper = EarlyStopper(
        patience=stopper_settings.patience,
        min_delta=stopper_settings.min_delta,
        min_epochs=stopper_settings.min_epochs,
    )

    if mode == "train":
        best_checkpoint = config.resolve(config.outputs.best_checkpoint_path)
        last_checkpoint = config.resolve(config.outputs.last_checkpoint_path)
    else:
        checkpoint_dir = config.resolve(
            config.outputs.benchmark_timing_checkpoints_dir
        )
        best_checkpoint = checkpoint_dir / config.outputs.best_checkpoint_path.name
        last_checkpoint = checkpoint_dir / config.outputs.last_checkpoint_path.name
    checkpoint_dir = best_checkpoint.parent
    if mode in {"benchmark", "train"} and (
        best_checkpoint.exists() or last_checkpoint.exists()
    ):
        raise FileExistsError(f"Refusing to overwrite checkpoints in {checkpoint_dir}")

    global_optimizer_steps = 0
    epoch_durations: list[float] = []
    best_metrics: dict[str, Any] | None = None
    peak_train_memory = 0.0
    run_started = time.perf_counter()
    stop_reason = "maximum_epochs"
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        losses, optimizer_steps, train_seconds, peak_memory = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_dtype=amp_dtype,
            config=config,
            max_batches=max_train_batches,
        )
        global_optimizer_steps += optimizer_steps
        if peak_memory is not None:
            peak_train_memory = max(peak_train_memory, peak_memory)
        validation, validation_seconds = evaluate_model(
            model,
            val_loader,
            device=device,
            amp_enabled=config.runtime.amp,
            amp_dtype=amp_dtype,
            label_to_category_id=val_dataset.label_to_category_id,
            category_names=val_dataset.category_names,
            settings=config.evaluation,
            max_batches=max_val_batches,
        )
        map_50_95 = validation["val_map_50_95"]
        if map_50_95 is None:
            if mode != "smoke":
                raise RuntimeError("validation mAP is undefined on the complete validation split")
            is_new_best = False
            should_stop = False
        else:
            scheduler.step(map_50_95)
            observation = stopper.observe(epoch=epoch, metric=map_50_95)
            is_new_best = observation.is_new_best
            should_stop = observation.should_stop

        if mode in {"benchmark", "train"}:
            if is_new_best:
                best_metrics = validation
                _atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        config=config,
                        train_dataset=train_dataset,
                        epoch=epoch,
                        validation_metrics=validation,
                        dataset_identity=dataset_summary,
                    ),
                    best_checkpoint,
                )
            _atomic_torch_save(
                _last_state_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    stopper=stopper,
                    config=config,
                    epoch=epoch,
                    global_optimizer_steps=global_optimizer_steps,
                    dataset_identity=dataset_summary,
                ),
                last_checkpoint,
            )

        epoch_seconds = time.perf_counter() - epoch_started
        epoch_durations.append(epoch_seconds)
        metrics_record = {
            "run_id": config.outputs.run_name(mode),
            "seed": config.seed,
            "epoch": epoch,
            "optimizer_steps": global_optimizer_steps,
            "learning_rate": learning_rate_used,
            "train_loss_total": losses["loss_total"],
            "train_loss_classifier": losses["loss_classifier"],
            "train_loss_box_reg": losses["loss_box_reg"],
            "train_loss_objectness": losses["loss_objectness"],
            "train_loss_rpn_box_reg": losses["loss_rpn_box_reg"],
            **{key: validation[key] for key in (
                "val_precision",
                "val_recall",
                "val_f1",
                "val_map_50",
                "val_map_50_95",
                "val_true_positives",
                "val_false_positives",
                "val_false_negatives",
            )},
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_mib": peak_memory,
            "is_best": is_new_best,
            "epochs_without_improvement": stopper.epochs_without_improvement,
        }
        append_epoch_metrics(
            config.run_artifact_path(mode, config.outputs.epoch_csv_path),
            config.run_artifact_path(mode, config.outputs.epoch_jsonl_path),
            metrics_record,
        )
        print(
            f"epoch {epoch}/{epochs} loss={losses['loss_total']:.5f} "
            f"val_mAP50:95={map_50_95} seconds={epoch_seconds:.1f}",
            flush=True,
        )

        if mode == "train" and should_stop:
            stop_reason = "early_stopping_validation_map"
            break

    _shutdown_data_loader(train_loader)
    _shutdown_data_loader(val_loader)
    del train_loader, val_loader
    del optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    total_training_seconds = time.perf_counter() - run_started
    base_summary: dict[str, Any] = {
        "status": "trained_pending_finalization" if mode == "train" else "complete",
        "mode": mode,
        "run_id": config.outputs.run_name(mode),
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "config_sha256": config_fingerprint(config),
        "git_commit": implementation_identity["git_commit"],
        "implementation_identity": implementation_identity,
        "execution_identity": execution_identity,
        "completed_epochs": len(epoch_durations),
        "epoch_seconds": epoch_durations,
        "total_training_seconds": total_training_seconds,
        "stop_reason": stop_reason,
        "best_epoch": stopper.best_epoch,
        "best_val_map_50_95": stopper.best_metric,
        "best_validation_metrics": best_metrics,
        "dataset": dataset_summary,
        "weights": weight_metadata,
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "amp": config.runtime.amp,
            "amp_dtype": config.runtime.amp_dtype,
            "microbatch_size": config.runtime.batch_size,
            "gradient_accumulation_steps": config.runtime.gradient_accumulation_steps,
            "effective_batch_size": config.runtime.effective_batch_size,
            "batch_norm_policy": config.model.batch_norm_policy,
            "peak_train_gpu_memory_mib": peak_train_memory,
        },
        "test_split_accessed": False,
    }
    if approved_benchmark_provenance is not None:
        base_summary["approved_benchmark"] = approved_benchmark_provenance

    if mode == "benchmark":
        projection = build_benchmark_projection(
            epoch_durations,
            minimum_epochs=config.training.early_stopping.min_epochs,
            maximum_epochs=config.training.max_epochs,
            scenario_epochs=(config.training.max_epochs,),
        )
        benchmark_payload = {
            **projection,
            "config_sha256": config_fingerprint(config),
            "completed_epochs": len(epoch_durations),
            "run_id": config.outputs.benchmark_run_name,
            "includes_full_training_and_validation_epochs": True,
            "timing_includes_equivalent_best_and_last_checkpoint_io": True,
            "timing_excludes_epoch_metric_log_write": True,
            "restart_full_run_from_coco_weights": True,
            "dataset_identity": dataset_summary,
            "execution_identity": execution_identity,
            "implementation_identity": implementation_identity,
        }
        write_atomic_json(
            config.run_artifact_path(mode, config.outputs.benchmark_estimate_path),
            benchmark_payload,
        )
        base_summary["benchmark_estimate"] = benchmark_payload
    elif mode == "train":
        if best_metrics is None or stopper.best_epoch is None or not best_checkpoint.is_file():
            raise RuntimeError("full training completed without a best validation checkpoint")
        base_summary["checkpoint_path"] = best_checkpoint.as_posix()
        base_summary["last_state_path"] = last_checkpoint.as_posix()
        write_atomic_json(
            config.run_artifact_path(mode, config.outputs.summary_path), base_summary
        )
        profile_loader = _make_data_loader(
            val_dataset,
            config,
            shuffle=False,
            batch_size=config.profiling.batch_size,
            num_workers=config.profiling.num_workers,
            persistent_workers=config.profiling.persistent_workers,
        )
        try:
            base_summary["artifacts"] = _final_artifacts(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                profile_loader=profile_loader,
                device=device,
                amp_dtype=amp_dtype,
                config=config,
                best_checkpoint=best_checkpoint,
                best_epoch=stopper.best_epoch,
                best_metrics=best_metrics,
                training_seconds=total_training_seconds,
                peak_train_gpu_memory_mib=peak_train_memory,
            )
        finally:
            _shutdown_data_loader(profile_loader)
        base_summary["status"] = "complete"
    write_atomic_json(
        config.run_artifact_path(mode, config.outputs.summary_path), base_summary
    )
    return base_summary


def _completed_artifacts_intact(
    summary: dict[str, Any], config: FasterRCNNConfig
) -> bool:
    """Return whether a complete summary's configured final artifacts still exist."""

    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    expected_paths = {
        "validation_table": config.resolve(config.outputs.validation_table_path),
        "compute_table": config.resolve(config.outputs.compute_table_path),
        "training_curves": config.resolve(config.outputs.training_curves_path),
    }
    file_integrity = artifacts.get("file_integrity")
    if not isinstance(file_integrity, dict):
        return False
    for key, expected in expected_paths.items():
        recorded = artifacts.get(key)
        integrity = file_integrity.get(key)
        if not isinstance(recorded, str) or not isinstance(integrity, dict):
            return False
        path = Path(recorded)
        if (
            path.resolve() != expected.resolve()
            or not path.is_file()
            or Path(integrity.get("path", "")).resolve() != expected.resolve()
            or integrity.get("size_bytes") != path.stat().st_size
            or integrity.get("sha256") != sha256_file(path)
        ):
            return False
    model_artifact = artifacts.get("model_artifact")
    if not isinstance(model_artifact, dict):
        return False
    checkpoint = config.resolve(config.outputs.best_checkpoint_path)
    return (
        Path(model_artifact.get("path", "")).resolve() == checkpoint.resolve()
        and checkpoint.is_file()
        and model_artifact.get("size_bytes") == checkpoint.stat().st_size
        and model_artifact.get("sha256") == sha256_file(checkpoint)
    )


def _run_finalization(
    *,
    config: FasterRCNNConfig,
    train_dataset: CocoDetectionDataset,
    val_dataset: CocoDetectionDataset,
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    """Idempotently regenerate final artifacts from a completed best checkpoint."""

    summary_path = config.run_artifact_path("train", config.outputs.summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(
            "finalize mode requires the full-run summary written after training: "
            f"{summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {"trained_pending_finalization", "complete"}:
        raise ValueError(f"full-run summary has non-finalizable status: {summary.get('status')}")
    if summary.get("config_sha256") != config_fingerprint(config):
        raise ValueError("full-run summary was produced by a different configuration")
    if summary.get("dataset") != dataset_summary:
        raise ValueError("full-run summary dataset identity no longer matches local data")
    if summary.get("status") == "complete" and _completed_artifacts_intact(summary, config):
        return summary

    import torch

    from src.models.faster_rcnn_model import build_faster_rcnn
    from src.utils.seed import seed_everything

    seed_everything(
        config.seed,
        deterministic=config.runtime.deterministic,
        warn_only=config.runtime.deterministic_warn_only,
    )
    execution_identity = _execution_identity(config)
    if summary.get("execution_identity") != execution_identity:
        raise ValueError(
            "finalization must use the same software and GPU identity as full training"
        )
    expected_checkpoint = config.resolve(config.outputs.best_checkpoint_path).resolve()
    recorded_checkpoint = Path(summary.get("checkpoint_path", "")).resolve()
    if recorded_checkpoint != expected_checkpoint or not expected_checkpoint.is_file():
        raise FileNotFoundError(
            f"finalize mode requires the configured best checkpoint: {expected_checkpoint}"
        )
    checkpoint = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("config_sha256") != config_fingerprint(config):
        raise ValueError("best checkpoint was produced by a different configuration")
    if checkpoint.get("dataset_identity") != dataset_summary:
        raise ValueError("best checkpoint dataset identity no longer matches local data")
    best_metrics = checkpoint.get("validation_metrics")
    best_epoch = checkpoint.get("epoch")
    if not isinstance(best_metrics, dict) or not isinstance(best_epoch, int):
        raise ValueError("best checkpoint lacks validation metrics or epoch metadata")
    del checkpoint

    device, amp_dtype = _require_cuda(config)
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    model, _weight_metadata = build_faster_rcnn(
        train_dataset.num_foreground_classes,
        config.model,
        use_pretrained_weights=False,
    )
    model.to(device)
    profile_loader = _make_data_loader(
        val_dataset,
        config,
        shuffle=False,
        batch_size=config.profiling.batch_size,
        num_workers=config.profiling.num_workers,
        persistent_workers=config.profiling.persistent_workers,
    )
    started = time.perf_counter()
    summary["status"] = "trained_pending_finalization"
    summary["finalization"] = {
        "status": "running",
        "implementation_identity": _implementation_identity(config),
        "execution_identity": execution_identity,
    }
    write_atomic_json(summary_path, summary)
    try:
        summary["artifacts"] = _final_artifacts(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            profile_loader=profile_loader,
            device=device,
            amp_dtype=amp_dtype,
            config=config,
            best_checkpoint=expected_checkpoint,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            training_seconds=float(summary["total_training_seconds"]),
            peak_train_gpu_memory_mib=float(
                summary["runtime"]["peak_train_gpu_memory_mib"]
            ),
        )
    finally:
        _shutdown_data_loader(profile_loader)
    summary["status"] = "complete"
    summary["best_epoch"] = best_epoch
    summary["best_val_map_50_95"] = best_metrics["val_map_50_95"]
    summary["best_validation_metrics"] = best_metrics
    summary["finalization"].update(
        {
            "status": "complete",
            "seconds": time.perf_counter() - started,
        }
    )
    write_atomic_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight or one explicitly bounded experiment mode."""

    args = build_parser().parse_args(argv)
    config = load_faster_rcnn_config(args.config)
    if args.mode == "preflight":
        train_dataset, val_dataset = _load_datasets(config, mode="smoke")
        print(json.dumps(_dataset_summary(train_dataset, val_dataset), indent=2, sort_keys=True))
        return 0 if train_dataset.preflight.complete and val_dataset.preflight.complete else 2

    mode: RunMode = args.mode
    if mode == "train" and args.approved_benchmark is None:
        raise ValueError("train mode requires --approved-benchmark after user sign-off")
    if mode != "train" and args.approved_benchmark is not None:
        raise ValueError("--approved-benchmark is only valid with --mode train")

    train_dataset, val_dataset = _load_datasets(config, mode=mode)
    summary = _dataset_summary(train_dataset, val_dataset)
    if mode == "finalize":
        result = _run_finalization(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataset_summary=summary,
        )
    else:
        result = _run_training(
            mode=mode,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataset_summary=summary,
            approved_benchmark=args.approved_benchmark,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
