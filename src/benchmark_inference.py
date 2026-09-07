"""Versioned matched-boundary inference timing; no training or historical writes.

Run only on the configured reporting machine. Ordinary inference is independently
decoded from disk outside timing, under the same explicit AMP precision. Each
timed call starts with the same decoded uint8 RGB host array and ends with CPU
source-coordinate boxes, scores and labels. CUDA completion brackets every call.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import Field, model_validator

from src.evaluate import (
    StrictModel,
    _load_test_dataset,
    _recorded_training_config_sha256,
    load_and_validate_training_configs,
    load_phase5_config,
    sha256_file,
)
from src.utils.seed import log_run_environment, seed_everything


class Subset(StrictModel):
    """Detector-independent subset selection without label or score access."""

    size: int = Field(gt=0)
    seed: int = Field(ge=0)
    selection: Literal["sha256-ranked-file-name"]


class Agreement(StrictModel):
    """Numerical comparison bounds; counts and category labels must be exact."""

    box_atol_pixels: float = Field(ge=0, le=0.1)
    score_atol: float = Field(ge=0, le=0.0001)
    rtol: float = Field(ge=0, le=0.0001)


class Hardware(StrictModel):
    """Fail-closed publication hardware identity."""

    gpu_name: str
    cpu_contains: str
    system_contains: str
    ram_gib_min: float
    ram_gib_max: float
    vram_gib_min: float
    vram_gib_max: float


class Versions(StrictModel):
    """Pinned framework and CUDA build identities."""

    torch: str
    torchvision: str
    ultralytics: str
    cuda: str


class Outputs(StrictModel):
    """New version-specific output files, never historical result paths."""

    log_dir: Path
    summary: Path
    latencies: Path
    repetitions: Path
    table: Path
    figure: Path


class TimingConfig(StrictModel):
    """All timing settings and input/output paths are explicit in YAML."""

    schema_version: Literal[1]
    protocol: Literal["decoded-host-to-source-detections-v1"]
    evaluation_config: Path
    checkpoint_manifest: Path
    historical_manifest: Path
    checkpoint_seed: int = Field(ge=0)
    subset: Subset
    batch_size: Literal[1]
    warmup_images: int = Field(gt=0)
    repetitions: int = Field(ge=3)
    device: str
    deterministic: Literal[True]
    allow_tf32: Literal[False]
    torch_threads: int = Field(gt=0)
    agreement: Agreement
    hardware: Hardware
    versions: Versions
    outputs: Outputs
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve only paths within this project."""
        result = (self.project_root / path).resolve()
        if path.is_absolute() or not result.is_relative_to(self.project_root):
            raise ValueError(f"path must stay within the project: {path}")
        return result

    @model_validator(mode="after")
    def isolated_outputs(self) -> TimingConfig:
        """Reject output collisions, including nested historical log locations."""
        if not self.device.startswith("cuda:"):
            raise ValueError("publication timing requires a CUDA device")
        paths = [self.resolve(p) for p in self.outputs.model_dump().values()]
        if len(set(paths)) != len(paths):
            raise ValueError("output paths must be distinct")
        for path in paths:
            if "inference_timing_v1" not in path.as_posix():
                raise ValueError("outputs require the inference_timing_v1 namespace")
        return self


def load_config(path: str | Path) -> TimingConfig:
    """Load strict, versioned YAML configuration."""
    source = Path(path).resolve()
    return TimingConfig.model_validate(
        dict(
            yaml.safe_load(source.read_text(encoding="utf-8")),
            project_root=source.parent.parent,
            source_path=source,
        )
    )


def select_indices(names: list[str], settings: Subset) -> list[int]:
    """Rank seed/name SHA256 digests; preserve the resulting order for both arms."""
    if len(set(names)) != len(names) or settings.size > len(names):
        raise ValueError("subset requires unique names and sufficient images")
    return sorted(
        range(len(names)),
        key=lambda i: (
            hashlib.sha256(f"{settings.seed}:{names[i]}".encode()).hexdigest(),
            names[i],
        ),
    )[: settings.size]


def compare_predictions(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], tolerance: Agreement
) -> dict[str, float | int]:
    """Compare score-sorted detections without dropping unmatched boxes."""
    if len(actual["scores"]) != len(expected["scores"]):
        raise ValueError("detection count mismatch")
    for prediction in (actual, expected):
        n = len(prediction["scores"])
        if prediction["boxes"].shape != (n, 4) or prediction["labels"].shape != (n,):
            raise ValueError("invalid prediction shape")
        if not all(np.isfinite(v).all() for v in prediction.values()):
            raise ValueError("nonfinite prediction")
    orders = [np.argsort(-p["scores"], kind="stable") for p in (actual, expected)]
    if not np.array_equal(actual["labels"][orders[0]], expected["labels"][orders[1]]):
        raise ValueError("detection labels mismatch")
    errors: dict[str, float | int] = {"detections": len(actual["scores"])}
    for field, atol in (("boxes", tolerance.box_atol_pixels), ("scores", tolerance.score_atol)):
        a, b = actual[field][orders[0]], expected[field][orders[1]]
        if not np.allclose(a, b, atol=atol, rtol=tolerance.rtol):
            raise ValueError(f"detection {field} mismatch")
        errors[f"max_{field}_absolute_error"] = float(np.max(np.abs(a - b), initial=0))
    return errors


def time_call(
    predict: Callable[[], dict[str, np.ndarray]],
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, np.ndarray], float]:
    """Synchronize immediately before starting and before stopping the clock."""
    synchronize()
    start = clock()
    prediction = predict()
    synchronize()
    elapsed = clock() - start
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid synchronized elapsed time")
    return prediction, elapsed


def summarize(seconds: list[float]) -> dict[str, float | int]:
    """Sum timed intervals, derive throughput, and report robust image dispersion."""
    values = np.asarray(seconds, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("latencies must be nonempty, finite and positive")
    q1, median, q3 = np.quantile(values * 1000, [0.25, 0.5, 0.75], method="linear")
    total = float(values.sum())
    return {
        "timed_images": len(values),
        "total_elapsed_seconds": total,
        "fps": len(values) / total,
        "median_latency_ms": float(median),
        "q1_latency_ms": float(q1),
        "q3_latency_ms": float(q3),
        "iqr_latency_ms": float(q3 - q1),
    }


def verify_history(config: TimingConfig) -> dict[str, Any]:
    """Rehash every frozen historical result and timing implementation."""
    history = json.loads(config.resolve(config.historical_manifest).read_text(encoding="utf-8"))
    if not history.get("files"):
        raise ValueError("historical preservation manifest is empty")
    for relative, expected in history["files"].items():
        if sha256_file(config.resolve(Path(relative))) != expected:
            raise ValueError(f"historical artifact drift: {relative}")
    for row in history.get("historical_rows", []):
        total = row["total_elapsed_seconds_reconstructed"]
        if not np.isclose(row["historical_fps"], row["timed_images"] / total):
            raise ValueError("historical FPS/elapsed reconstruction mismatch")
    return history


def hardware_snapshot(config: TimingConfig) -> dict[str, Any]:
    """Verify actual GPU, CPU, machine, RAM and pinned software before timing."""
    import psutil
    import torch
    import torchvision
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError("reporting hardware gate: CUDA unavailable; no publication timing")
    if platform.system() != "Windows":
        raise RuntimeError("reporting hardware gate: intended Windows laptop required")
    import winreg

    def registry(key: str, field: str) -> str:
        """Read machine identity without requiring WMI privileges."""
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            return str(winreg.QueryValueEx(handle, field)[0])

    props = torch.cuda.get_device_properties(torch.device(config.device))
    observed = {
        "gpu_name": props.name,
        "cpu": registry(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString"),
        "system": registry(r"HARDWARE\DESCRIPTION\System\BIOS", "SystemProductName"),
        "ram_gib": psutil.virtual_memory().total / 1024**3,
        "vram_gib": props.total_memory / 1024**3,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
        },
        "cudnn": torch.backends.cudnn.version(),
    }
    expected = config.hardware
    valid = (
        observed["gpu_name"] == expected.gpu_name
        and expected.cpu_contains in observed["cpu"]
        and expected.system_contains in observed["system"]
        and expected.ram_gib_min <= observed["ram_gib"] <= expected.ram_gib_max
        and expected.vram_gib_min <= observed["vram_gib"] <= expected.vram_gib_max
        and observed["versions"] == config.versions.model_dump()
    )
    if not valid:
        raise RuntimeError(f"reporting hardware/software gate failed: {observed}")
    observed["gate_passed"] = True
    return observed


def telemetry() -> dict[str, Any]:
    """Capture power/temperature/driver context without altering device settings."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,"
            "utilization.gpu,memory.used,clocks.sm,clocks.mem",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class Predictor:
    """Use native detector inference, changing only the source input boundary."""

    def __init__(self, run: Any, phase: Any, model_config: Any, dataset: Any) -> None:
        """Load the exact checkpoint; model load/fusion are outside timing."""
        import torch

        self.torch = torch
        self.run, self.phase, self.config, self.dataset = run, phase, model_config, dataset
        self.device = torch.device(phase.runtime.device)
        self.amp_dtype = getattr(torch, model_config.runtime.amp_dtype)
        if not model_config.runtime.amp:
            raise ValueError("AMP is mandatory")
        if run.detector == "faster_rcnn":
            from src.models.faster_rcnn_model import build_faster_rcnn

            checkpoint = torch.load(
                phase.resolve(run.checkpoint), map_location="cpu", weights_only=False
            )
            if checkpoint["config_sha256"] != _recorded_training_config_sha256(run, phase):
                raise ValueError("checkpoint training config hash mismatch")
            self.model, _ = build_faster_rcnn(
                dataset.num_foreground_classes, model_config.model, use_pretrained_weights=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device).eval()
        else:
            from ultralytics import YOLO

            self.model = YOLO(phase.resolve(run.checkpoint).as_posix())

    def predict(self, index: int, rgb: np.ndarray | None) -> dict[str, np.ndarray]:
        """None uses ordinary disk inference; RGB uses decoded-host timing input."""
        torch = self.torch
        with (
            torch.inference_mode(),
            torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True),
        ):
            if self.run.detector == "faster_rcnn":
                if rgb is None:
                    tensor, _ = self.dataset[index]
                else:
                    pixels = np.array(rgb, dtype=np.float32, copy=True) / np.float32(255.0)
                    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
                output = self.model([tensor.to(self.device, non_blocking=True)])[0]
                scores = output["scores"].detach().cpu().numpy().astype(np.float64)
                keep = scores >= self.phase.evaluation.coco_minimum_score
                labels = output["labels"].detach().cpu().numpy()[keep]
                return {
                    "boxes": output["boxes"].detach().cpu().numpy()[keep],
                    "scores": scores[keep],
                    "labels": np.asarray(
                        [self.dataset.label_to_category_id[int(v)] for v in labels]
                    ),
                }
            # Ultralytics ndarray sources are BGR; conversion is INSIDE timing.
            source = (
                self.dataset.image_path(index).as_posix() if rgb is None else rgb[:, :, ::-1].copy()
            )
            result = self.model.predict(
                source=source,
                stream=False,
                batch=self.phase.runtime.inference_batch_size,
                imgsz=self.config.model.input_size,
                device=self.config.runtime.device,
                amp=True,
                quantize=None,
                conf=self.phase.evaluation.coco_minimum_score,
                iou=self.phase.evaluation.nms_iou_threshold,
                max_det=self.phase.evaluation.max_detections,
                agnostic_nms=False,
                augment=False,
                verbose=False,
                save=False,
            )[0]
            category_ids = tuple(sorted(self.dataset.category_names))
            return {
                "boxes": result.boxes.xyxy.detach().cpu().numpy().astype(np.float64),
                "scores": result.boxes.conf.detach().cpu().numpy().astype(np.float64),
                "labels": np.asarray(
                    [category_ids[int(v)] for v in result.boxes.cls.cpu().numpy()]
                ),
            }

    def verify_amp(self, index: int) -> str:
        """Confirm a native convolution actually emits the configured AMP dtype."""
        module = self.model if self.run.detector == "faster_rcnn" else self.model.model
        convolution = next(m for m in module.modules() if isinstance(m, self.torch.nn.Conv2d))
        observed = []

        def record_dtype(_module: Any, _inputs: Any, output: Any) -> None:
            observed.append(output.dtype)

        handle = convolution.register_forward_hook(record_dtype)
        try:
            self.predict(index, None)
        finally:
            handle.remove()
        if not observed or any(dtype != self.amp_dtype for dtype in observed):
            raise ValueError(f"AMP execution dtype mismatch: {observed}")
        return str(observed[0])


def preflight(config: TimingConfig) -> tuple[Any, Any, Any, list[int], dict[str, Any]]:
    """Check history, frozen checkpoints, identical test source and target machine."""
    verify_history(config)
    phase = load_phase5_config(config.resolve(config.evaluation_config))
    model_configs = load_and_validate_training_configs(phase)
    yolo_config = model_configs[("yolo11s", config.checkpoint_seed)]
    if config.device != f"cuda:{yolo_config.runtime.device}":
        raise ValueError("timing and ordinary inference CUDA devices disagree")
    runs = [run for run in phase.runs if run.seed == config.checkpoint_seed]
    if len(runs) != 2:
        raise ValueError("exactly two frozen checkpoints required")
    release = json.loads(config.resolve(config.checkpoint_manifest).read_text(encoding="utf-8"))
    expected = {item["source_path"]: item["sha256"] for item in release["checkpoints"]}
    for run in runs:
        digest = sha256_file(phase.resolve(run.checkpoint))
        with phase.resolve(run.compute_table).open(encoding="utf-8", newline="") as handle:
            row = list(csv.DictReader(handle))
        if (
            digest != expected[run.checkpoint.as_posix()]
            or len(row) != 1
            or digest != row[0]["model_sha256"]
        ):
            raise ValueError("checkpoint hash mismatch")
    dataset = _load_test_dataset(phase, model_configs[("faster_rcnn", config.checkpoint_seed)])
    indices = select_indices([r.file_name for r in dataset.records], config.subset)
    for index in indices:
        if not dataset.image_path(index).is_file():
            raise FileNotFoundError(dataset.image_path(index))
    return phase, model_configs, dataset, indices, hardware_snapshot(config)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an auditable UTF-8 CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    """Write finite JSON, preserving all raw timings and provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render(config: TimingConfig, payload: dict[str, Any]) -> None:
    """Create summary/repetition CSVs and a technical-repetition latency figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_csv(config.resolve(config.outputs.table), payload["aggregate"])
    write_csv(config.resolve(config.outputs.repetitions), payload["repetitions"])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for position, row in enumerate(payload["aggregate"]):
        ax.errorbar(
            position,
            row["median_latency_ms"],
            yerr=[
                [row["median_latency_ms"] - row["q1_latency_ms"]],
                [row["q3_latency_ms"] - row["median_latency_ms"]],
            ],
            fmt="o",
            capsize=7,
            color=("#24618c", "#b45124")[position],
            markersize=9,
        )
        reps = [r for r in payload["repetitions"] if r["detector"] == row["detector"]]
        ax.scatter(
            np.linspace(position - 0.08, position + 0.08, len(reps)),
            [r["median_latency_ms"] for r in reps],
            marker="x",
            color="black",
            zorder=3,
        )
    ax.set_xticks([0, 1], ["Faster R-CNN", "YOLO11s"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Decoded host image to CPU detections (ms/image)")
    ax.set_title("Matched inference timing v1 · RTX 4060 Laptop GPU")
    ax.grid(axis="y", alpha=0.2)
    fig.supxlabel(
        f"Batch 1 · {config.subset.size} identical test images · "
        f"{config.repetitions} technical repeats\n"
        "Circles / bars: pooled median / IQR · crosses: repetition medians\n"
        f"Frozen seed-{config.checkpoint_seed} checkpoints · disk I/O excluded · "
        "no inferential intervals",
        fontsize=9,
    )
    path = config.resolve(config.outputs.figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_timing(config: TimingConfig) -> dict[str, Any]:
    """Run the complete protocol once; refuse to replace accepted measurements."""
    import torch

    summary_path = config.resolve(config.outputs.summary)
    if any(
        config.resolve(p).exists() for k, p in config.outputs.model_dump().items() if k != "log_dir"
    ):
        raise FileExistsError("timing outputs already exist; use verify/report, not overwrite")
    report = seed_everything(config.subset.seed, deterministic=True, warn_only=False)
    torch.set_num_threads(config.torch_threads)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    phase, model_configs, dataset, indices, hardware = preflight(config)
    torch.cuda.set_device(torch.device(config.device))
    runs = [r for r in phase.runs if r.seed == config.checkpoint_seed]
    # Only the selected subset is decoded. No disk read belongs to time_call.
    decoded, subset = {}, []
    for index in indices:
        path = dataset.image_path(index)
        with Image.open(path) as source:
            rgb = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
        record = dataset.records[index]
        if rgb.shape != (record.height, record.width, 3):
            raise ValueError("decoded image dimensions disagree with immutable metadata")
        rgb.flags.writeable = False
        decoded[index] = rgb
        subset.append(
            {
                "image_id": record.file_name,
                "source_path": path.relative_to(config.project_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    environment_files = log_run_environment(config.resolve(config.outputs.log_dir), report)
    predictors = {
        r.detector: Predictor(r, phase, model_configs[(r.detector, r.seed)], dataset) for r in runs
    }
    reference = {}
    # Native loader/transform/forward/postprocess reference; never timed as evidence.
    for name, predictor in predictors.items():
        reference[name] = {i: predictor.predict(i, None) for i in indices}
    observed_amp = {name: p.verify_amp(indices[0]) for name, p in predictors.items()}
    runtime = {
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if runtime != {
        "torch_threads": config.torch_threads,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }:
        raise ValueError(f"native inference changed the configured runtime: {runtime}")
    before = telemetry()
    latencies, repetitions, agreements = [], [], []

    def synchronize() -> None:
        """Complete all CUDA streams on the reporting device."""
        torch.cuda.synchronize(torch.device(config.device))

    for repetition in range(config.repetitions):
        order = list(predictors) if repetition % 2 == 0 else list(reversed(predictors))
        for name in order:
            predictor = predictors[name]
            for warm in range(config.warmup_images):
                i = indices[warm % len(indices)]
                predictor.predict(i, decoded[i])
            seconds, errors = [], []
            for index in indices:
                output, elapsed = time_call(
                    partial(predictor.predict, index, decoded[index]), synchronize
                )
                # Validation, recording and hashing are outside the timed boundary.
                errors.append(compare_predictions(output, reference[name][index], config.agreement))
                seconds.append(elapsed)
                latencies.append(
                    {
                        "detector": name,
                        "repetition": repetition + 1,
                        "image_id": dataset.records[index].file_name,
                        "seconds": elapsed,
                        **errors[-1],
                    }
                )
            repetitions.append(
                {"detector": name, "repetition": repetition + 1, **summarize(seconds)}
            )
            agreements.append(
                {
                    "detector": name,
                    "repetition": repetition + 1,
                    "images_checked": len(errors),
                    "detections_checked": sum(e["detections"] for e in errors),
                    "max_boxes_absolute_error": max(e["max_boxes_absolute_error"] for e in errors),
                    "max_scores_absolute_error": max(
                        e["max_scores_absolute_error"] for e in errors
                    ),
                    "passed": True,
                }
            )
            print(json.dumps(repetitions[-1]), flush=True)
    aggregate = []
    for run in runs:
        with phase.resolve(run.compute_table).open(encoding="utf-8", newline="") as handle:
            historical = next(csv.DictReader(handle))
        aggregate.append(
            {
                "detector": run.detector,
                "checkpoint_seed": run.seed,
                "protocol": config.protocol,
                "subset_size": len(indices),
                "repetitions": config.repetitions,
                "warmup_images_per_repetition": config.warmup_images,
                "batch_size": config.batch_size,
                "amp_dtype": model_configs[(run.detector, run.seed)].runtime.amp_dtype,
                **summarize([r["seconds"] for r in latencies if r["detector"] == run.detector]),
                "total_parameters": int(historical["total_parameters"]),
                "training_trainable_parameters": int(historical["trainable_parameters"]),
                "historical_incomplete_registered_gflops": float(historical["estimated_gflops"]),
            }
        )
    sources = [
        config.source_path,
        config.resolve(config.evaluation_config),
        config.resolve(config.checkpoint_manifest),
        config.resolve(config.historical_manifest),
        dataset.annotation_file,
        Path(__file__).resolve(),
        Path(__file__).parent / "evaluate.py",
        Path(__file__).parent / "utils/seed.py",
        Path(__file__).parent / "models/faster_rcnn_model.py",
        Path(__file__).parent / "models/faster_rcnn_data.py",
    ]
    for run in runs:
        sources.extend(
            phase.resolve(p) for p in (run.training_config, run.checkpoint, run.training_summary)
        )
    sources.extend(environment_files)
    payload = {
        "schema_version": 1,
        "protocol": config.protocol,
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "boundary": {
            "start": "decoded uint8 RGB source image in host memory",
            "end": "source-coordinate boxes, scores and category labels in CPU arrays",
            "includes": [
                "resize/letterbox",
                "tensor conversion",
                "host-to-device transfer",
                "AMP model forward",
                "native postprocessing/NMS",
                "coordinate restoration",
                "device-to-host results",
                "Python prediction API overhead",
            ],
            "excludes": [
                "disk I/O",
                "decoding",
                "checkpoint loading/fusion",
                "warmup",
                "agreement checks",
                "metric evaluation",
                "artifact writes",
            ],
        },
        "hardware": hardware,
        "observed_runtime": runtime,
        "telemetry_before": before,
        "telemetry_after": telemetry(),
        "subset": subset,
        "subset_sha256": hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest(),
        "observed_convolution_amp_dtype": observed_amp,
        "reference": (
            "ordinary disk-based native inference under identical explicit AMP and thresholds"
        ),
        "precision_note": (
            "YOLO predict amp=True alone is not inference autocast; v1 explicitly uses "
            "configured bfloat16, including the ordinary reference. Historical Phase 5 YOLO "
            "predictions are not the AMP parity reference."
        ),
        "repetition_scope": (
            "technical timing repetitions of one checkpoint per detector; not biological or "
            "training replicates; no hypothesis tests or inferential intervals"
        ),
        "model_order": "alternate Faster/YOLO and YOLO/Faster per repetition; both models resident",
        "source_sha256": {
            p.relative_to(config.project_root).as_posix(): sha256_file(p) for p in sources
        },
        "agreement": agreements,
        "aggregate": aggregate,
        "repetitions": repetitions,
    }
    verify_history(config)
    write_csv(config.resolve(config.outputs.latencies), latencies)
    render(config, payload)
    payload["output_sha256"] = {
        str(p).replace("\\", "/"): sha256_file(config.resolve(p))
        for k, p in config.outputs.model_dump().items()
        if k not in {"summary", "log_dir"}
    }
    write_json(summary_path, payload)
    del predictors, reference, decoded
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def verify_results(config: TimingConfig) -> dict[str, Any]:
    """Offline verification from per-image times, without a new GPU measurement."""
    verify_history(config)
    payload = json.loads(config.resolve(config.outputs.summary).read_text(encoding="utf-8"))
    if payload["status"] != "complete" or payload["config"] != config.model_dump(mode="json"):
        raise ValueError("timing result/config mismatch")
    for relative, digest in (payload["source_sha256"] | payload["output_sha256"]).items():
        if sha256_file(config.resolve(Path(relative))) != digest:
            raise ValueError(f"timing provenance drift: {relative}")
    with config.resolve(config.outputs.latencies).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [r["image_id"] for r in payload["subset"]]
    if not payload["hardware"]["gate_passed"]:
        raise ValueError("reporting hardware gate was not passed")
    if (
        payload["subset_sha256"]
        != hashlib.sha256(json.dumps(payload["subset"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("subset hash mismatch")
    for image in payload["subset"]:
        path = config.resolve(Path(image["source_path"]))
        if path.exists() and sha256_file(path) != image["sha256"]:
            raise ValueError("source image hash mismatch")
    expected_pairs = {
        (name, repetition)
        for name in ("faster_rcnn", "yolo11s")
        for repetition in range(1, config.repetitions + 1)
    }
    for key in ("repetitions", "agreement"):
        pairs = [(row["detector"], row["repetition"]) for row in payload[key]]
        if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
            raise ValueError("missing or duplicate detector/repetition pair")
    if [r["detector"] for r in payload["aggregate"]] != ["faster_rcnn", "yolo11s"]:
        raise ValueError("aggregate detector mismatch")
    if len(rows) != 2 * config.repetitions * config.subset.size or len(names) != config.subset.size:
        raise ValueError("timing row count mismatch")
    for row in payload["repetitions"]:
        selected = [
            r
            for r in rows
            if r["detector"] == row["detector"] and int(r["repetition"]) == row["repetition"]
        ]
        if [r["image_id"] for r in selected] != names:
            raise ValueError("repetition subset/order mismatch")
        recomputed = summarize([float(r["seconds"]) for r in selected])
        if any(row[k] != v for k, v in recomputed.items()):
            raise ValueError("repetition statistics mismatch")
    for row in payload["aggregate"]:
        recomputed = summarize(
            [float(r["seconds"]) for r in rows if r["detector"] == row["detector"]]
        )
        if any(row[k] != v for k, v in recomputed.items()):
            raise ValueError("aggregate statistics mismatch")
    if len(payload["agreement"]) != 2 * config.repetitions or not all(
        r["passed"] and r["images_checked"] == config.subset.size for r in payload["agreement"]
    ):
        raise ValueError("incomplete prediction agreement")
    return payload


def main() -> None:
    """Provide read-only preflight/verification, GPU timing and offline rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run", "verify", "report"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "preflight":
        *_, hardware = preflight(config)
        print(json.dumps(hardware, indent=2))
    elif args.mode == "run":
        run_timing(config)
    else:
        payload = verify_results(config)
        if args.mode == "report":
            render(config, payload)
        print("Timing protocol, historical preservation, raw statistics and provenance verified.")


if __name__ == "__main__":
    main()
