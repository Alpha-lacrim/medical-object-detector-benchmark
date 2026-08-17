"""Strict reporting and profiling summaries for the Faster R-CNN baseline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .faster_rcnn_training import estimate_training_seconds

EPOCH_FIELDNAMES = (
    "run_id",
    "seed",
    "epoch",
    "optimizer_steps",
    "learning_rate",
    "train_loss_total",
    "train_loss_classifier",
    "train_loss_box_reg",
    "train_loss_objectness",
    "train_loss_rpn_box_reg",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_map_50",
    "val_map_50_95",
    "val_true_positives",
    "val_false_positives",
    "val_false_negatives",
    "train_seconds",
    "validation_seconds",
    "epoch_seconds",
    "peak_gpu_memory_mib",
    "is_best",
    "epochs_without_improvement",
)

VALIDATION_FIELDNAMES = (
    "run_id",
    "seed",
    "best_epoch",
    "split",
    "operating_point_score_threshold",
    "operating_point_match_iou_threshold",
    "coco_minimum_score",
    "max_detections",
    "image_count",
    "target_count",
    "operating_point_prediction_count",
    "coco_prediction_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "map_50",
    "map_50_95",
)

COMPUTE_FIELDNAMES = (
    "run_id",
    "seed",
    "input_min_size",
    "input_max_size",
    "amp_dtype",
    "inference_batch_size",
    "timed_batches",
    "timed_images",
    "throughput_fps",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "total_parameters",
    "trainable_parameters",
    "estimated_gflops",
    "flop_count_method",
    "model_size_bytes",
    "model_size_mib",
    "model_sha256",
    "training_seconds",
    "peak_train_gpu_memory_mib",
)

_EPOCH_INTEGER_FIELDS = frozenset(
    {
        "seed",
        "epoch",
        "optimizer_steps",
        "val_true_positives",
        "val_false_positives",
        "val_false_negatives",
        "epochs_without_improvement",
    }
)
_EPOCH_FLOAT_FIELDS = frozenset(
    {
        "learning_rate",
        "train_loss_total",
        "train_loss_classifier",
        "train_loss_box_reg",
        "train_loss_objectness",
        "train_loss_rpn_box_reg",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_map_50",
        "val_map_50_95",
        "train_seconds",
        "validation_seconds",
        "epoch_seconds",
        "peak_gpu_memory_mib",
    }
)
_OPTIONAL_EPOCH_FLOAT_FIELDS = frozenset(
    {
        "val_precision",
        "val_recall",
        "val_f1",
        "val_map_50",
        "val_map_50_95",
        "peak_gpu_memory_mib",
    }
)
_PROBABILITY_EPOCH_FIELDS = frozenset(
    {"val_precision", "val_recall", "val_f1", "val_map_50", "val_map_50_95"}
)


class ParameterLike(Protocol):
    """Minimal interface needed to summarize model parameter counts."""

    requires_grad: bool

    def numel(self) -> int:
        """Return the number of scalar values in this parameter."""


def _json_text(payload: Any, *, indent: int | None = 2) -> str:
    """Serialize JSON deterministically while rejecting NaN and infinity."""

    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> Path:
    """Replace ``path`` atomically with UTF-8 ``text`` from the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def write_atomic_json(path: str | Path, payload: Any) -> Path:
    """Write deterministic, strict JSON through an atomic same-directory replace."""

    destination = Path(path)
    return _atomic_write_text(destination, _json_text(payload))


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file without loading it at once."""

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_model_artifact(path: str | Path) -> dict[str, Any]:
    """Return a JSON-safe size and digest summary for a model-only artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    size_bytes = source.stat().st_size
    return {
        "path": source.as_posix(),
        "size_bytes": size_bytes,
        "size_mib": size_bytes / (1024**2),
        "sha256": sha256_file(source),
    }


def _require_non_negative_integer(value: Any, field: str) -> int:
    """Validate and return one non-negative integer record value."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_finite_non_negative_float(
    value: Any,
    field: str,
    *,
    optional: bool = False,
) -> float | None:
    """Validate a non-negative finite float, optionally accepting ``None``."""

    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _normalize_epoch_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one epoch mapping and return values in the stable field order."""

    missing = sorted(set(EPOCH_FIELDNAMES) - set(record))
    unknown = sorted(set(record) - set(EPOCH_FIELDNAMES))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError("invalid epoch record (" + "; ".join(details) + ")")

    run_id = record["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    normalized: dict[str, Any] = {"run_id": run_id}
    for field in EPOCH_FIELDNAMES[1:]:
        value = record[field]
        if field in _EPOCH_INTEGER_FIELDS:
            normalized[field] = _require_non_negative_integer(value, field)
        elif field in _EPOCH_FLOAT_FIELDS:
            normalized[field] = _require_finite_non_negative_float(
                value,
                field,
                optional=field in _OPTIONAL_EPOCH_FLOAT_FIELDS,
            )
        elif field == "is_best":
            if not isinstance(value, bool):
                raise ValueError("is_best must be boolean")
            normalized[field] = value
        else:  # pragma: no cover - the constant schema makes this unreachable
            raise RuntimeError(f"epoch field has no validator: {field}")

    if normalized["epoch"] == 0:
        raise ValueError("epoch must be positive")
    if normalized["epoch_seconds"] == 0:
        raise ValueError("epoch_seconds must be positive")
    for field in _PROBABILITY_EPOCH_FIELDS:
        value = normalized[field]
        if value is not None and value > 1:
            raise ValueError(f"{field} must be in [0, 1]")
    _json_text(normalized, indent=None)
    return {field: normalized[field] for field in EPOCH_FIELDNAMES}


def _csv_cell(value: Any) -> Any:
    """Return a deterministic scalar representation for one CSV cell."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """Render validated flat rows as deterministic CSV text."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_cell(row[field]) for field in fieldnames})
    return stream.getvalue()


def _load_epoch_csv(path: Path) -> list[dict[str, Any]]:
    """Load and validate epoch records from the stable CSV representation."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EPOCH_FIELDNAMES:
            raise ValueError(f"{path} has an incompatible epoch CSV schema")
        records: list[dict[str, Any]] = []
        for raw in reader:
            parsed: dict[str, Any] = {"run_id": raw["run_id"]}
            for field in EPOCH_FIELDNAMES[1:]:
                text = raw[field]
                if field in _EPOCH_INTEGER_FIELDS:
                    parsed[field] = int(text)
                elif field in _EPOCH_FLOAT_FIELDS:
                    parsed[field] = None if text == "" else float(text)
                elif field == "is_best":
                    if text not in {"true", "false"}:
                        raise ValueError(f"{path} contains an invalid boolean value")
                    parsed[field] = text == "true"
            records.append(_normalize_epoch_record(parsed))
    return records


def _load_epoch_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load and validate epoch records from the strict JSONL representation."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path} contains a blank JSONL line at {line_number}")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} JSONL line {line_number} is not an object")
            records.append(_normalize_epoch_record(payload))
    return records


def load_epoch_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Load validated epoch metrics from a stable epoch CSV file."""

    return _load_epoch_csv(Path(path))


def _write_epoch_files(
    csv_path: Path,
    jsonl_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Atomically rewrite both tabular and structured epoch-log views."""

    csv_text = _csv_text(EPOCH_FIELDNAMES, records)
    jsonl_text = "".join(_json_text(record, indent=None) for record in records)
    _atomic_write_text(jsonl_path, jsonl_text)
    _atomic_write_text(csv_path, csv_text)
    return csv_path, jsonl_path


def append_epoch_metrics(
    csv_path: str | Path,
    jsonl_path: str | Path,
    record: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Append one validated epoch to synchronized CSV and JSONL artifacts.

    Repeating an identical ``(run_id, epoch)`` record is idempotent. A
    conflicting duplicate raises rather than silently rewriting history.
    """

    destination_csv = Path(csv_path)
    destination_jsonl = Path(jsonl_path)
    csv_records = _load_epoch_csv(destination_csv)
    jsonl_records = _load_epoch_jsonl(destination_jsonl)
    if csv_records and jsonl_records and csv_records != jsonl_records:
        raise ValueError("existing epoch CSV and JSONL artifacts disagree")
    existing = csv_records or jsonl_records
    normalized = _normalize_epoch_record(record)
    key = (normalized["run_id"], normalized["epoch"])
    for previous in existing:
        if (previous["run_id"], previous["epoch"]) == key:
            if previous != normalized:
                raise ValueError(f"conflicting epoch record for run {key[0]!r}, epoch {key[1]}")
            return _write_epoch_files(destination_csv, destination_jsonl, existing)

    run_epochs = [
        previous["epoch"] for previous in existing if previous["run_id"] == normalized["run_id"]
    ]
    if run_epochs and normalized["epoch"] <= max(run_epochs):
        raise ValueError("epoch records for a run must be appended in increasing order")
    return _write_epoch_files(
        destination_csv,
        destination_jsonl,
        [*existing, normalized],
    )


def _normalize_flat_rows(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate a sequence of flat table rows against one exact schema."""

    expected = set(fieldnames)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        missing = sorted(expected - set(row))
        unknown = sorted(set(row) - expected)
        if missing or unknown:
            raise ValueError(
                f"table row {index} does not match schema; missing={missing}, unknown={unknown}"
            )
        ordered: dict[str, Any] = {}
        for field in fieldnames:
            value = row[field]
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError(f"table field {field} must be a scalar or null")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"table field {field} must be finite")
            ordered[field] = value
        _json_text(ordered, indent=None)
        normalized.append(ordered)
    return normalized


def write_validation_metrics_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write the stable baseline-validation performance table atomically."""

    normalized = _normalize_flat_rows(rows, VALIDATION_FIELDNAMES)
    return _atomic_write_text(
        Path(path),
        _csv_text(VALIDATION_FIELDNAMES, normalized),
    )


def write_compute_metrics_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write the stable baseline computational-profile table atomically."""

    normalized = _normalize_flat_rows(rows, COMPUTE_FIELDNAMES)
    return _atomic_write_text(Path(path), _csv_text(COMPUTE_FIELDNAMES, normalized))


def build_benchmark_projection(
    epoch_seconds: Sequence[float],
    *,
    minimum_epochs: int,
    maximum_epochs: int,
    scenario_epochs: Sequence[int] = (),
) -> dict[str, Any]:
    """Build the benchmark estimate and optional named epoch-count scenarios."""

    durations = list(epoch_seconds)
    projection = estimate_training_seconds(
        durations,
        minimum_epochs=minimum_epochs,
        maximum_epochs=maximum_epochs,
    )
    scenarios: dict[str, float] = {}
    seen: set[int] = set()
    for epoch_count in scenario_epochs:
        if isinstance(epoch_count, bool) or not isinstance(epoch_count, int) or epoch_count <= 0:
            raise ValueError("scenario epochs must be positive integers")
        if epoch_count in seen:
            raise ValueError("scenario epochs must be unique")
        seen.add(epoch_count)
        scenario = estimate_training_seconds(
            durations,
            minimum_epochs=epoch_count,
            maximum_epochs=epoch_count,
        )
        scenarios[str(epoch_count)] = scenario["estimated_maximum_seconds"]
    return {**projection, "scenario_estimated_seconds": scenarios}


def write_benchmark_projection(
    path: str | Path,
    epoch_seconds: Sequence[float],
    *,
    minimum_epochs: int,
    maximum_epochs: int,
    scenario_epochs: Sequence[int] = (),
) -> dict[str, Any]:
    """Build and atomically persist a strict benchmark timing projection."""

    projection = build_benchmark_projection(
        epoch_seconds,
        minimum_epochs=minimum_epochs,
        maximum_epochs=maximum_epochs,
        scenario_epochs=scenario_epochs,
    )
    write_atomic_json(path, projection)
    return projection


def summarize_parameter_counts(parameters: Iterable[ParameterLike]) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts without importing Torch."""

    total = 0
    trainable = 0
    for parameter in parameters:
        count = parameter.numel()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("parameter numel() must return a non-negative integer")
        if not isinstance(parameter.requires_grad, bool):
            raise TypeError("parameter requires_grad must be boolean")
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_inference_timings(
    batch_seconds: Sequence[float],
    *,
    images_per_batch: int | Sequence[int],
) -> dict[str, int | float]:
    """Summarize synchronized raw batch timings without running model or CUDA code."""

    durations = list(batch_seconds)
    if not durations:
        raise ValueError("at least one timed batch is required")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in durations
    ):
        raise ValueError("batch timings must be positive finite numbers")
    normalized_durations = [float(value) for value in durations]

    if isinstance(images_per_batch, int) and not isinstance(images_per_batch, bool):
        if images_per_batch <= 0:
            raise ValueError("images_per_batch must be positive")
        image_counts = [images_per_batch] * len(normalized_durations)
    else:
        image_counts = list(images_per_batch)
        if len(image_counts) != len(normalized_durations) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in image_counts
        ):
            raise ValueError("per-batch image counts must be positive and match timings")

    total_seconds = sum(normalized_durations)
    total_images = sum(image_counts)
    per_image_seconds = [
        duration / count for duration, count in zip(normalized_durations, image_counts, strict=True)
    ]
    return {
        "timed_batches": len(normalized_durations),
        "timed_images": total_images,
        "total_seconds": total_seconds,
        "throughput_fps": total_images / total_seconds,
        "mean_latency_ms": total_seconds / total_images * 1000,
        "p50_latency_ms": _percentile(per_image_seconds, 0.50) * 1000,
        "p95_latency_ms": _percentile(per_image_seconds, 0.95) * 1000,
    }


def _plot_available_series(
    axes: Any,
    epochs: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[tuple[str, str]],
) -> None:
    """Plot every requested series that contains at least one non-null value."""

    for field, label in fields:
        points = [
            (epoch, record[field])
            for epoch, record in zip(epochs, records, strict=True)
            if record[field] is not None
        ]
        if points:
            axes.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                markersize=3,
                label=label,
            )
    if axes.lines:
        axes.legend(fontsize="small")


def plot_training_curves(
    epoch_csv_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    """Render the four-panel baseline training curve figure from epoch CSV.

    Matplotlib is imported only when this function is called so reporting and
    unit tests that do not plot remain lightweight.
    """

    records = load_epoch_metrics(epoch_csv_path)
    if not records:
        raise ValueError("cannot plot an empty epoch log")
    run_ids = {record["run_id"] for record in records}
    if len(run_ids) != 1:
        raise ValueError("one training-curve figure must contain exactly one run")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("matplotlib is required to render training curves") from error

    epochs = [record["epoch"] for record in records]
    figure, axes_grid = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes_grid.flatten()
    try:
        _plot_available_series(
            axes[0],
            epochs,
            records,
            (
                ("train_loss_total", "total"),
                ("train_loss_classifier", "classifier"),
                ("train_loss_box_reg", "box regression"),
                ("train_loss_objectness", "objectness"),
                ("train_loss_rpn_box_reg", "RPN box regression"),
            ),
        )
        axes[0].set_title("Training loss")
        axes[0].set_ylabel("Loss")

        _plot_available_series(
            axes[1],
            epochs,
            records,
            (("val_precision", "precision"), ("val_recall", "recall")),
        )
        axes[1].set_title("Validation operating point")
        axes[1].set_ylabel("Metric")
        axes[1].set_ylim(0, 1.02)

        _plot_available_series(
            axes[2],
            epochs,
            records,
            (("val_map_50", "mAP@0.5"), ("val_map_50_95", "mAP@0.5:0.95")),
        )
        axes[2].set_title("Validation average precision")
        axes[2].set_ylabel("Average precision")
        axes[2].set_ylim(0, 1.02)

        _plot_available_series(
            axes[3],
            epochs,
            records,
            (("learning_rate", "learning rate"),),
        )
        axes[3].set_title("Learning rate")
        axes[3].set_ylabel("Learning rate")

        best_epochs = [record["epoch"] for record in records if record["is_best"]]
        if best_epochs:
            best_epoch = best_epochs[-1]
            for axis in axes:
                axis.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.6)
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
        figure.suptitle(f"Faster R-CNN training curves — {next(iter(run_ids))}")
        figure.tight_layout()

        destination = Path(output_path)
        if not destination.suffix:
            raise ValueError("training-curve output path must have a file extension")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            figure.savefig(
                temporary_path,
                dpi=dpi,
                bbox_inches="tight",
                format=destination.suffix.lstrip("."),
            )
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        plt.close(figure)
    return Path(output_path)
