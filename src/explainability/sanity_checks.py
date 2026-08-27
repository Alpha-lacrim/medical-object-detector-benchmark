"""Run model- and data-randomization sanity checks for detector Grad-CAM maps."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import sha256_file
from src.explainability.config import DetectorName, ExplainabilityConfig, load_explainability_config
from src.explainability.gradcam import ActivationCapture, resolve_module
from src.explainability.run_explainability import (
    DETECTORS,
    _cam_numpy,
    _faster_forward,
    _load_faster_model,
    _load_yolo_model,
    _prepare,
    _validate_feature,
    _yolo_forward,
)
from src.robustness.run_robustness import CorruptedSubsetDataset, _largest_remainder_allocation

SanityTest = Literal["parameter_randomization", "data_randomization"]

DETAIL_FIELDS = (
    "detector",
    "test",
    "image_id",
    "nih_patient_id",
    "study_stratum",
    "trained_valid",
    "trained_failure_reason",
    "randomized_valid",
    "randomized_failure_reason",
    "pair_valid",
    "pair_failure_reason",
    "correlation",
    "reference_candidate_score",
    "trained_target_score",
    "randomized_target_score",
    "target_region_x1",
    "target_region_y1",
    "target_region_x2",
    "target_region_y2",
)

SUMMARY_FIELDS = (
    "detector",
    "test",
    "subset_image_count",
    "subset_patient_count",
    "subset_stratum_counts",
    "trained_valid_count",
    "trained_failure_count",
    "trained_failure_rate",
    "randomized_valid_count",
    "randomized_failure_count",
    "randomized_failure_rate",
    "k_valid_pairs",
    "map_pair_failure_count",
    "map_pair_failure_rate",
    "c_sanity",
    "correlation_std",
    "correlation_median",
    "correlation_min",
    "correlation_max",
    "sanity_failure_threshold",
    "sanity_failure_count",
    "sanity_failure_rate",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingSettings(StrictModel):
    """Nested sampling contract bound to the frozen robustness pool."""

    source_pool_manifest: Path
    output_manifest: Path
    size: int = Field(gt=0)
    id_column: str = Field(min_length=1)
    patient_column: str = Field(min_length=1)
    stratum_column: str = Field(min_length=1)
    allocation: Literal["proportional_largest_remainder"]
    selection: Literal["seeded_without_replacement_within_frozen_pool"]
    selection_seed: int = Field(ge=0, le=2**32 - 1)


class TargetSettings(StrictModel):
    """One image-level detector-score target per condition."""

    reference_selection: Literal["trained_highest_score_retained_candidate"]
    map_target: Literal["pre_activation_foreground_score_at_trained_reference_region"]
    faster_rcnn_region_scoring: Literal["fixed_roi_classifier"]
    yolo11s_region_scoring: Literal["closest_raw_anchor_center"]


class ParameterRandomizationSettings(StrictModel):
    """Model-copy and parameter-initialization policy."""

    copy_method: Literal["deep_copy_trained_model"]
    weight_initialization: Literal["xavier_normal"]
    one_dimensional_weight_view: Literal["row_vector"]
    bias_initialization: Literal["zeros"]
    preserve_non_weight_non_bias_buffers: Literal[True]
    autocast: Literal["disabled_for_randomized_weight_numerical_validity"]
    gain: float = Field(gt=0)
    seed: int = Field(ge=0, le=2**32 - 1)


class DataRandomizationSettings(StrictModel):
    """Deterministic within-image pixel permutation policy."""

    method: Literal["spatial_pixel_vector_permutation_without_replacement"]
    seed_derivation: Literal["sha256_global_seed_and_image_id"]
    seed: int = Field(ge=0, le=2**32 - 1)


class CorrelationSettings(StrictModel):
    """Map-pair validity and correlation estimand."""

    method: Literal["pearson"]
    population: Literal["valid_nonconstant_paired_maps"]
    zero_energy_policy: Literal["exclude_and_report"]
    sanity_failure_threshold: float = Field(ge=-1, le=1)
    epsilon: float = Field(gt=0)


class RuntimeSettings(StrictModel):
    """Bounded single-GPU execution settings."""

    device: Literal["cuda"]
    progress_every_images: int = Field(gt=0)


class PanelSettings(StrictModel):
    """Result-independent qualitative panel selection and styling."""

    cases_per_stratum: Literal[1]
    selection: Literal["lexicographically_first_nested_subset_image"]
    overlay_alpha: float = Field(gt=0, le=1)
    colormap: str = Field(min_length=1)
    candidate_color: str = Field(min_length=1)
    line_width: float = Field(gt=0)
    width_inches: float = Field(gt=0)
    row_height_inches: float = Field(gt=0)
    dpi: int = Field(gt=0)


class OutputSettings(StrictModel):
    """Batch 21 generated artifacts."""

    log_dir: Path
    summary_json: Path
    detail_table: Path
    summary_table: Path
    panel_figure: Path


class XaiSanityConfig(StrictModel):
    """Complete immutable Batch 21 experiment contract."""

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    phase7_config: Path
    phase7_summary: Path
    sampling: SamplingSettings
    target: TargetSettings
    parameter_randomization: ParameterRandomizationSettings
    data_randomization: DataRandomizationSettings
    correlation: CorrelationSettings
    runtime: RuntimeSettings
    panel: PanelSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_seeds(self) -> XaiSanityConfig:
        if self.sampling.selection_seed != self.seed:
            raise ValueError("nested sample must reuse the primary experiment seed")
        if len({self.parameter_randomization.seed, self.data_randomization.seed}) != 2:
            raise ValueError("parameter and data randomization require distinct seeds")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_xai_sanity_config(path: str | Path) -> XaiSanityConfig:
    """Load the strict Batch 21 YAML contract."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("XAI sanity config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return XaiSanityConfig.model_validate(payload)


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, raw)


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return _atomic_bytes(path, buffer.getvalue().encode())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def select_nested_stratified_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    size: int,
    id_column: str,
    stratum_column: str,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Select a deterministic proportional subset from an already-frozen pool."""

    if len(rows) < size:
        raise ValueError("frozen robustness pool is smaller than the requested nested subset")
    identifiers = [str(row.get(id_column, "")) for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("frozen-pool identifiers must be non-empty and unique")
    by_stratum: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        stratum = str(row.get(stratum_column, ""))
        if not stratum:
            raise ValueError("frozen-pool strata must be non-empty")
        by_stratum[stratum].append(row)
    counts = {name: len(values) for name, values in sorted(by_stratum.items())}
    allocation = _largest_remainder_allocation(counts, size)
    generator = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    for stratum in sorted(by_stratum):
        candidates = sorted(by_stratum[stratum], key=lambda row: row[id_column])
        indices = generator.choice(len(candidates), size=allocation[stratum], replace=False)
        selected.extend(candidates[int(index)] for index in sorted(indices))
    selected.sort(key=lambda row: row[id_column])
    if len(selected) != size:
        raise AssertionError("nested stratified sampler returned the wrong image count")
    return selected, allocation


def _materialize_nested_subset(
    config: XaiSanityConfig,
    phase7: ExplainabilityConfig,
    phase7_prepared: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    phase6 = phase7_prepared["phase6"]
    source = config.resolve(config.sampling.source_pool_manifest)
    expected_source = phase6.resolve(phase6.sampling.output_manifest)
    if source != expected_source:
        raise ValueError("Batch 21 source must be the configured frozen Phase 6 pool manifest")
    if sha256_file(source) != phase7_prepared["sample_sha256"]:
        raise ValueError("Batch 21 source-pool hash disagrees with validated Phase 7 provenance")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("frozen robustness pool manifest has no CSV header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    required = {
        config.sampling.id_column,
        config.sampling.patient_column,
        config.sampling.stratum_column,
        "is_positive",
        "box_count",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"frozen robustness pool lacks columns: {sorted(missing)}")
    if len(rows) != phase6.sampling.size:
        raise ValueError("frozen robustness pool size changed")
    selected, allocation = select_nested_stratified_rows(
        rows,
        size=config.sampling.size,
        id_column=config.sampling.id_column,
        stratum_column=config.sampling.stratum_column,
        seed=config.sampling.selection_seed,
    )
    output_rows = [
        {**row, "xai_sanity_subset_index": str(index)} for index, row in enumerate(selected)
    ]
    output = config.resolve(config.sampling.output_manifest)
    _atomic_csv(output, [*fieldnames, "xai_sanity_subset_index"], output_rows)
    audit = {
        "method": config.sampling.allocation,
        "selection": config.sampling.selection,
        "rng": "numpy.random.Generator(PCG64)",
        "seed": config.sampling.selection_seed,
        "source_pool_manifest": source.as_posix(),
        "source_pool_manifest_sha256": sha256_file(source),
        "source_pool_image_count": len(rows),
        "source_pool_stratum_counts": dict(
            sorted(Counter(row[config.sampling.stratum_column] for row in rows).items())
        ),
        "output_manifest": output.as_posix(),
        "output_manifest_sha256": sha256_file(output),
        "subset_image_count": len(selected),
        "subset_patient_count": len({row[config.sampling.patient_column] for row in selected}),
        "subset_positive_image_count": sum(int(row["is_positive"]) for row in selected),
        "subset_box_count": sum(int(row["box_count"]) for row in selected),
        "subset_stratum_counts": dict(sorted(allocation.items())),
        "ordering": (
            "Sort strata and identifiers lexicographically, draw without replacement "
            "sequentially from one seeded generator, then sort selected identifiers."
        ),
    }
    return selected, audit


def _validate_phase7_summary(
    config: XaiSanityConfig,
    phase7: ExplainabilityConfig,
    prepared: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = config.resolve(config.phase7_summary)
    summary = _read_json(path)
    if (
        summary.get("status") != "complete"
        or summary.get("config_sha256") != sha256_file(phase7.source_path)
        or summary.get("seed_scope", {}).get("training_seed") != config.seed
    ):
        raise ValueError("completed Phase 7 summary is incompatible with Batch 21")
    if config.seed != phase7.seed or config.runtime.device != phase7.runtime.device:
        raise ValueError("Batch 21 must reuse the Phase 7 seed and CUDA runtime")
    phase6 = prepared["phase6"]
    expected_checkpoints = {
        item.detector: sha256_file(phase6.resolve(item.checkpoint)) for item in phase6.detectors
    }
    if summary.get("checkpoints") != expected_checkpoints:
        raise ValueError("Phase 7 summary checkpoint identities changed")
    expected_layers = [item.model_dump(mode="json") for item in phase7.detectors]
    if summary.get("target_layers") != expected_layers:
        raise ValueError("Batch 21 target layers differ from the completed Phase 7 protocol")
    return path, summary


def _prepare_sanity(config: XaiSanityConfig) -> dict[str, Any]:
    phase7 = load_explainability_config(config.resolve(config.phase7_config))
    prepared = _prepare(phase7)
    summary_path, phase7_summary = _validate_phase7_summary(config, phase7, prepared)
    selected_rows, sampling_audit = _materialize_nested_subset(config, phase7, prepared)
    selected_names = {row[config.sampling.id_column] for row in selected_rows}
    subset = CorruptedSubsetDataset(
        prepared["subset"].base_dataset,
        selected_names,
        condition=None,
        seed=config.seed,
    )
    if len(subset) != config.sampling.size:
        raise AssertionError("canonical nested subset has the wrong size")
    return {
        "phase7": phase7,
        "phase7_prepared": prepared,
        "phase7_summary": phase7_summary,
        "phase7_summary_path": summary_path,
        "selected_rows": selected_rows,
        "sampling_audit": sampling_audit,
        "subset": subset,
    }


def preflight(config: XaiSanityConfig) -> dict[str, Any]:
    """Validate all frozen identities and materialize the nested 50-image manifest."""

    prepared = _prepare_sanity(config)
    panel_cases = select_panel_cases(
        prepared["selected_rows"],
        id_column=config.sampling.id_column,
        stratum_column=config.sampling.stratum_column,
    )
    return {
        "status": "ready",
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "sampling": prepared["sampling_audit"],
        "target_layers": [item.model_dump(mode="json") for item in prepared["phase7"].detectors],
        "panel_cases": panel_cases,
    }


def shuffle_pixel_vectors(image: Image.Image, *, seed: int, image_id: str) -> Image.Image:
    """Permute spatial RGB pixel vectors while preserving their exact multiset."""

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, channels = pixels.shape
    digest = hashlib.sha256(f"{seed}\0{image_id}".encode()).digest()
    image_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = np.random.default_rng(image_seed)
    permutation = generator.permutation(height * width)
    shuffled = pixels.reshape(-1, channels)[permutation].reshape(height, width, channels)
    return Image.fromarray(np.ascontiguousarray(shuffled))


def pearson_map_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float,
) -> float | None:
    """Return full-resolution Pearson correlation or null for an invalid pair."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("sanity-check heatmaps must have matching two-dimensional shapes")
    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        return None
    first_centered = first_values - first_values.mean()
    second_centered = second_values - second_values.mean()
    first_norm = float(np.linalg.norm(first_centered))
    second_norm = float(np.linalg.norm(second_centered))
    if first_norm <= epsilon or second_norm <= epsilon:
        return None
    value = float(np.dot(first_centered, second_centered) / (first_norm * second_norm))
    if not np.isfinite(value):
        return None
    return float(np.clip(value, -1.0, 1.0))


def xavier_reinitialize_model(
    model: Any,
    *,
    seed: int,
    gain: float,
) -> dict[str, Any]:
    """Xavier-randomize every module weight tensor and zero every module bias tensor."""

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    xavier_tensors = 0
    xavier_values = 0
    one_dimensional_tensors = 0
    weight_buffer_tensors = 0
    bias_tensors = 0
    bias_values = 0
    bias_buffer_tensors = 0
    seen_weights: set[int] = set()
    seen_biases: set[int] = set()
    with torch.no_grad():
        for module_name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and id(weight) not in seen_weights:
                if not weight.is_floating_point():
                    raise TypeError(f"cannot randomize non-floating weight: {module_name}")
                shaped = weight if weight.ndim >= 2 else weight.reshape(1, -1)
                torch.nn.init.xavier_normal_(shaped, gain=gain, generator=generator)
                seen_weights.add(id(weight))
                xavier_tensors += 1
                xavier_values += weight.numel()
                one_dimensional_tensors += int(weight.ndim < 2)
                weight_buffer_tensors += int("weight" in module._buffers)
            bias = getattr(module, "bias", None)
            if isinstance(bias, torch.Tensor) and id(bias) not in seen_biases:
                if not bias.is_floating_point():
                    raise TypeError(f"cannot zero non-floating bias: {module_name}")
                torch.nn.init.zeros_(bias)
                seen_biases.add(id(bias))
                bias_tensors += 1
                bias_values += bias.numel()
                bias_buffer_tensors += int("bias" in module._buffers)
    if xavier_tensors == 0:
        raise ValueError("model contains no non-bias parameter to Xavier-randomize")
    return {
        "seed": seed,
        "gain": gain,
        "generator": "torch.Generator(device=cpu)",
        "xavier_weight_tensor_count": xavier_tensors,
        "xavier_weight_value_count": xavier_values,
        "one_dimensional_row_view_tensor_count": one_dimensional_tensors,
        "xavier_weight_buffer_tensor_count": weight_buffer_tensors,
        "zeroed_bias_tensor_count": bias_tensors,
        "zeroed_bias_value_count": bias_values,
        "zeroed_bias_buffer_tensor_count": bias_buffer_tensors,
        "non_weight_non_bias_buffers_preserved": True,
    }


def select_panel_cases(
    rows: Sequence[Mapping[str, str]],
    *,
    id_column: str,
    stratum_column: str,
) -> list[dict[str, str]]:
    """Select one result-independent, lexicographically first case per stratum."""

    by_stratum: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_stratum[row[stratum_column]].append(row)
    return [
        {
            "study_stratum": stratum,
            "image_id": min(items, key=lambda item: item[id_column])[id_column],
        }
        for stratum, items in sorted(by_stratum.items())
    ]


def _pil_to_tensor(image: Image.Image) -> Any:
    import torch

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))


def _select_trained_reference(
    *,
    detector: DetectorName,
    model: Any,
    device: Any,
    model_config: Any,
    capture: ActivationCapture,
    image: Image.Image,
    phase7: ExplainabilityConfig,
    class_count: int,
) -> dict[str, Any]:
    """Select the trained model's highest-score retained candidate as a fixed region."""

    try:
        if detector == "faster_rcnn":
            _activation, candidate_boxes, candidate_scores, _output_size = _faster_forward(
                model,
                device,
                model_config,
                capture,
                _pil_to_tensor(image),
                phase7,
            )
        else:
            _activation, candidate_boxes, candidate_scores, _output_size = _yolo_forward(
                model,
                device,
                model_config,
                capture,
                image,
                phase7,
                class_count,
            )
        score_values = candidate_scores.detach().float().cpu().numpy().astype(np.float64)
        candidate_index = int(np.argmax(score_values))
        result = {
            "valid": True,
            "failure_reason": None,
            "candidate_box": candidate_boxes[candidate_index].copy(),
            "candidate_score": float(score_values[candidate_index]),
        }
        capture.clear()
        return result
    except (RuntimeError, ValueError) as error:
        capture.clear()
        return {
            "valid": False,
            "failure_reason": f"{type(error).__name__}: {str(error).replace(chr(10), ' ')}",
            "candidate_box": None,
            "candidate_score": None,
        }


def _faster_reference_score(
    *,
    model: Any,
    device: Any,
    model_config: Any,
    capture: ActivationCapture,
    image: Image.Image,
    reference_box: np.ndarray,
    phase7: ExplainabilityConfig,
) -> tuple[Any, Any, tuple[int, int]]:
    """Score one fixed original-space ROI without requiring randomized RPN output."""

    import torch

    device_image = _pil_to_tensor(image).to(device).requires_grad_(True)
    target = {
        "boxes": torch.as_tensor(reference_box, dtype=torch.float32, device=device).reshape(1, 4),
        "labels": torch.ones((1,), dtype=torch.int64, device=device),
    }
    capture.clear()
    images, transformed_targets = model.transform([device_image], [target])
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals = [transformed_targets[0]["boxes"]]
    pooled = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
    representation = model.roi_heads.box_head(pooled)
    class_logits, _box_regression = model.roi_heads.box_predictor(representation)
    foreground_score = class_logits[0, 1:].max()
    activation = capture.activation
    _validate_feature(activation, phase7, "faster_rcnn")
    return activation, foreground_score, tuple(int(value) for value in device_image.shape[-2:])


def _yolo_reference_score(
    *,
    model: Any,
    device: Any,
    model_config: Any,
    capture: ActivationCapture,
    image: Image.Image,
    reference_box: np.ndarray,
    phase7: ExplainabilityConfig,
    class_count: int,
) -> tuple[Any, Any, tuple[int, int]]:
    """Score the randomized raw anchor closest to the trained candidate center."""

    import torch
    from ultralytics.data.augment import LetterBox

    original = np.asarray(image.convert("RGB"))
    input_size = int(model_config.model.input_size)
    letterboxed = LetterBox(
        new_shape=(input_size, input_size),
        auto=False,
        scale_fill=False,
        scaleup=True,
        stride=phase7.layer("yolo11s").expected_stride * 2,
    )(image=original)
    tensor = torch.from_numpy(np.ascontiguousarray(letterboxed.transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(device).float().div_(255).requires_grad_(True)
    capture.clear()
    output = model(tensor)
    prediction = output[0] if isinstance(output, tuple) else output
    if prediction.ndim != 3 or prediction.shape[1] < 4 + class_count:
        raise ValueError("YOLO11s emitted an unexpected raw prediction tensor")
    activation = capture.activation
    _validate_feature(activation, phase7, "yolo11s")
    original_height, original_width = original.shape[:2]
    gain = min(input_size / original_height, input_size / original_width)
    resized_height = min(input_size, round(original_height * gain))
    resized_width = min(input_size, round(original_width * gain))
    top = round((input_size - resized_height) / 2 - 0.1)
    left = round((input_size - resized_width) / 2 - 0.1)
    reference_center = torch.tensor(
        [
            (float(reference_box[0]) + float(reference_box[2])) * 0.5 * gain + left,
            (float(reference_box[1]) + float(reference_box[3])) * 0.5 * gain + top,
        ],
        device=device,
        dtype=prediction.dtype,
    )
    centers = prediction[0, :2].transpose(0, 1)
    distances = ((centers - reference_center) ** 2).sum(dim=1)
    finite = torch.isfinite(distances)
    if not bool(finite.any()):
        raise ValueError("YOLO11s randomized raw anchors have no finite center")
    distances = torch.where(finite, distances, torch.full_like(distances, torch.inf))
    anchor_index = int(torch.argmin(distances).detach().cpu())
    raw_output = output[1] if isinstance(output, tuple) else None
    raw_scores = raw_output.get("scores") if isinstance(raw_output, dict) else None
    if raw_scores is None or raw_scores.shape[1] < class_count:
        raise ValueError("YOLO11s raw foreground logits are unavailable")
    foreground_score = raw_scores[0, :class_count, anchor_index].max()
    if not bool(torch.isfinite(foreground_score)):
        raise ValueError("YOLO11s randomized foreground score is non-finite")
    return activation, foreground_score, tuple(int(value) for value in tensor.shape[-2:])


def _extract_reference_map(
    *,
    detector: DetectorName,
    model: Any,
    device: Any,
    model_config: Any,
    capture: ActivationCapture,
    image: Image.Image,
    image_size: tuple[int, int],
    phase7: ExplainabilityConfig,
    class_count: int,
    reference_box: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    """Extract a pre-activation foreground CAM at the fixed trained reference region."""

    try:
        if detector == "faster_rcnn":
            activation, score, feature_output_size = _faster_reference_score(
                model=model,
                device=device,
                model_config=model_config,
                capture=capture,
                image=image,
                reference_box=reference_box,
                phase7=phase7,
            )
        else:
            activation, score, feature_output_size = _yolo_reference_score(
                model=model,
                device=device,
                model_config=model_config,
                capture=capture,
                image=image,
                reference_box=reference_box,
                phase7=phase7,
                class_count=class_count,
            )
        if not bool(score.detach().isfinite().all()):
            raise ValueError("foreground target score is non-finite")
        heatmap = _cam_numpy(
            score,
            activation,
            feature_output_size=feature_output_size,
            original_size=image_size,
            detector=detector,
            config=phase7,
            retain_graph=False,
        )
        candidate_score = float(score.detach().float().cpu())
        capture.clear()
        valid = bool(
            np.isfinite(heatmap).all()
            and float(heatmap.sum(dtype=np.float64)) > epsilon
            and float(np.std(heatmap, dtype=np.float64)) > epsilon
        )
        return {
            "valid": valid,
            "failure_reason": None if valid else "zero_or_constant_heatmap",
            "heatmap": heatmap,
            "candidate_box": reference_box.copy(),
            "candidate_score": candidate_score,
            "candidate_iou_to_reference": None,
        }
    except (RuntimeError, ValueError) as error:
        capture.clear()
        return {
            "valid": False,
            "failure_reason": f"{type(error).__name__}: {str(error).replace(chr(10), ' ')}",
            "heatmap": None,
            "candidate_box": reference_box.copy(),
            "candidate_score": None,
            "candidate_iou_to_reference": None,
        }


def _detail_row(
    *,
    detector: DetectorName,
    test: SanityTest,
    source_row: Mapping[str, str],
    config: XaiSanityConfig,
    reference: Mapping[str, Any],
    trained: Mapping[str, Any],
    randomized: Mapping[str, Any],
) -> dict[str, Any]:
    correlation = None
    pair_failure = None
    if trained["valid"] and randomized["valid"]:
        correlation = pearson_map_correlation(
            trained["heatmap"],
            randomized["heatmap"],
            epsilon=config.correlation.epsilon,
        )
        if correlation is None:
            pair_failure = "undefined_pearson_correlation"
    else:
        failures = []
        if not trained["valid"]:
            failures.append("trained_cam_invalid")
        if not randomized["valid"]:
            failures.append("randomized_cam_invalid")
        pair_failure = "+".join(failures)
    box = reference.get("candidate_box")
    coordinates = [None, None, None, None] if box is None else [float(value) for value in box]
    return {
        "detector": detector,
        "test": test,
        "image_id": source_row[config.sampling.id_column],
        "nih_patient_id": source_row[config.sampling.patient_column],
        "study_stratum": source_row[config.sampling.stratum_column],
        "trained_valid": bool(trained["valid"]),
        "trained_failure_reason": trained["failure_reason"],
        "randomized_valid": bool(randomized["valid"]),
        "randomized_failure_reason": randomized["failure_reason"],
        "pair_valid": correlation is not None,
        "pair_failure_reason": pair_failure,
        "correlation": correlation,
        "reference_candidate_score": reference.get("candidate_score"),
        "trained_target_score": trained["candidate_score"],
        "randomized_target_score": randomized["candidate_score"],
        "target_region_x1": coordinates[0],
        "target_region_y1": coordinates[1],
        "target_region_x2": coordinates[2],
        "target_region_y2": coordinates[3],
    }


def _missing_reference_result() -> dict[str, Any]:
    return {
        "valid": False,
        "failure_reason": "trained_candidate_reference_unavailable",
        "heatmap": None,
        "candidate_box": None,
        "candidate_score": None,
        "candidate_iou_to_reference": None,
    }


def _run_detector(
    config: XaiSanityConfig,
    prepared: Mapping[str, Any],
    detector: DetectorName,
    panel_names: set[str],
) -> tuple[list[dict[str, Any]], dict[tuple[DetectorName, str], dict[str, Any]], dict[str, Any]]:
    import torch

    phase7: ExplainabilityConfig = prepared["phase7"]
    phase7_prepared = prepared["phase7_prepared"]
    subset = prepared["subset"]
    indices = {record.file_name: index for index, record in enumerate(subset.records)}
    source_rows = {row[config.sampling.id_column]: row for row in prepared["selected_rows"]}
    image_names = sorted(source_rows)
    if detector == "faster_rcnn":
        model, device, model_config = _load_faster_model(phase7, phase7_prepared)
    else:
        model, device, model_config = _load_yolo_model(phase7, phase7_prepared)

    clean_results: dict[str, dict[str, Any]] = {}
    reference_results: dict[str, dict[str, Any]] = {}
    detail: list[dict[str, Any]] = []
    panel_maps: dict[tuple[DetectorName, str], dict[str, Any]] = {}
    module = resolve_module(model, phase7.layer(detector).module_path)
    with ActivationCapture(module) as capture:
        for number, image_id in enumerate(image_names, start=1):
            index = indices[image_id]
            record = subset.records[index]
            image = subset.load_pil(index)
            image_size = (record.height, record.width)
            reference = _select_trained_reference(
                detector=detector,
                model=model,
                device=device,
                model_config=model_config,
                capture=capture,
                image=image,
                phase7=phase7,
                class_count=subset.num_foreground_classes,
            )
            reference_results[image_id] = reference
            reference_box = reference["candidate_box"]
            trained = (
                _extract_reference_map(
                    detector=detector,
                    model=model,
                    device=device,
                    model_config=model_config,
                    capture=capture,
                    image=image,
                    image_size=image_size,
                    phase7=phase7,
                    class_count=subset.num_foreground_classes,
                    reference_box=reference_box,
                    epsilon=config.correlation.epsilon,
                )
                if reference_box is not None
                else _missing_reference_result()
            )
            clean_results[image_id] = trained
            shuffled = shuffle_pixel_vectors(
                image,
                seed=config.data_randomization.seed,
                image_id=image_id,
            )
            data_randomized = (
                _extract_reference_map(
                    detector=detector,
                    model=model,
                    device=device,
                    model_config=model_config,
                    capture=capture,
                    image=shuffled,
                    image_size=image_size,
                    phase7=phase7,
                    class_count=subset.num_foreground_classes,
                    reference_box=reference_box,
                    epsilon=config.correlation.epsilon,
                )
                if reference_box is not None
                else _missing_reference_result()
            )
            detail.append(
                _detail_row(
                    detector=detector,
                    test="data_randomization",
                    source_row=source_rows[image_id],
                    config=config,
                    reference=reference,
                    trained=trained,
                    randomized=data_randomized,
                )
            )
            if image_id in panel_names:
                panel_maps[(detector, image_id)] = {
                    "image": np.asarray(image).copy(),
                    "shuffled_image": np.asarray(shuffled).copy(),
                    "trained": trained,
                    "data_randomized": data_randomized,
                }
            if number % config.runtime.progress_every_images == 0:
                print(
                    f"[{detector}] clean/data-randomization {number}/{len(image_names)}",
                    flush=True,
                )

    model.to("cpu")
    torch.cuda.empty_cache()
    randomized_model = copy.deepcopy(model)
    del model
    initialization_audit = xavier_reinitialize_model(
        randomized_model,
        seed=config.parameter_randomization.seed,
        gain=config.parameter_randomization.gain,
    )
    randomized_model.to(device).eval()
    for parameter in randomized_model.parameters():
        parameter.requires_grad_(False)
    module = resolve_module(randomized_model, phase7.layer(detector).module_path)
    with ActivationCapture(module) as capture:
        for number, image_id in enumerate(image_names, start=1):
            index = indices[image_id]
            record = subset.records[index]
            image = subset.load_pil(index)
            trained = clean_results[image_id]
            reference = reference_results[image_id]
            reference_box = reference["candidate_box"]
            parameter_randomized = (
                _extract_reference_map(
                    detector=detector,
                    model=randomized_model,
                    device=device,
                    model_config=model_config,
                    capture=capture,
                    image=image,
                    image_size=(record.height, record.width),
                    phase7=phase7,
                    class_count=subset.num_foreground_classes,
                    reference_box=reference_box,
                    epsilon=config.correlation.epsilon,
                )
                if reference_box is not None
                else _missing_reference_result()
            )
            detail.append(
                _detail_row(
                    detector=detector,
                    test="parameter_randomization",
                    source_row=source_rows[image_id],
                    config=config,
                    reference=reference,
                    trained=trained,
                    randomized=parameter_randomized,
                )
            )
            if image_id in panel_names:
                panel_maps[(detector, image_id)]["parameter_randomized"] = parameter_randomized
            if number % config.runtime.progress_every_images == 0:
                print(
                    f"[{detector}] parameter-randomization {number}/{len(image_names)}",
                    flush=True,
                )

    del randomized_model, clean_results, reference_results
    gc.collect()
    torch.cuda.empty_cache()
    return detail, panel_maps, initialization_audit


def aggregate_sanity(
    records: Sequence[Mapping[str, Any]],
    *,
    sampling_audit: Mapping[str, Any],
    failure_threshold: float,
) -> list[dict[str, Any]]:
    """Compute C_sanity, extraction failures, and high-correlation sanity failures."""

    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for test in ("parameter_randomization", "data_randomization"):
            group = [
                item for item in records if item["detector"] == detector and item["test"] == test
            ]
            if len(group) != sampling_audit["subset_image_count"]:
                raise AssertionError(f"incomplete sanity grid for {detector}/{test}")
            correlations = np.asarray(
                [float(item["correlation"]) for item in group if item["correlation"] is not None],
                dtype=np.float64,
            )
            trained_valid = sum(bool(item["trained_valid"]) for item in group)
            randomized_valid = sum(bool(item["randomized_valid"]) for item in group)
            total = len(group)
            valid_pairs = len(correlations)
            sanity_failures = int(np.sum(correlations >= failure_threshold))
            rows.append(
                {
                    "detector": detector,
                    "test": test,
                    "subset_image_count": total,
                    "subset_patient_count": sampling_audit["subset_patient_count"],
                    "subset_stratum_counts": json.dumps(
                        sampling_audit["subset_stratum_counts"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "trained_valid_count": trained_valid,
                    "trained_failure_count": total - trained_valid,
                    "trained_failure_rate": (total - trained_valid) / total,
                    "randomized_valid_count": randomized_valid,
                    "randomized_failure_count": total - randomized_valid,
                    "randomized_failure_rate": (total - randomized_valid) / total,
                    "k_valid_pairs": valid_pairs,
                    "map_pair_failure_count": total - valid_pairs,
                    "map_pair_failure_rate": (total - valid_pairs) / total,
                    "c_sanity": float(correlations.mean()) if valid_pairs else None,
                    "correlation_std": (
                        float(correlations.std(ddof=1)) if valid_pairs > 1 else None
                    ),
                    "correlation_median": (float(np.median(correlations)) if valid_pairs else None),
                    "correlation_min": float(correlations.min()) if valid_pairs else None,
                    "correlation_max": float(correlations.max()) if valid_pairs else None,
                    "sanity_failure_threshold": failure_threshold,
                    "sanity_failure_count": sanity_failures,
                    "sanity_failure_rate": (sanity_failures / valid_pairs if valid_pairs else None),
                }
            )
    return rows


def _draw_candidate(axis: Any, result: Mapping[str, Any], config: XaiSanityConfig) -> None:
    if result.get("candidate_box") is None:
        return
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = (float(value) for value in result["candidate_box"])
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=config.panel.candidate_color,
            linewidth=config.panel.line_width,
        )
    )


def _overlay(
    axis: Any, image: np.ndarray, result: Mapping[str, Any], config: XaiSanityConfig
) -> None:
    axis.imshow(image, cmap="gray")
    if result.get("heatmap") is not None:
        axis.imshow(
            result["heatmap"],
            cmap=config.panel.colormap,
            alpha=config.panel.overlay_alpha,
            vmin=0,
            vmax=1,
        )
        _draw_candidate(axis, result, config)
    else:
        axis.text(
            0.5,
            0.5,
            "CAM failed",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 4},
        )


def _write_panel(
    path: Path,
    *,
    config: XaiSanityConfig,
    panel_cases: Sequence[Mapping[str, str]],
    panel_maps: Mapping[tuple[DetectorName, str], Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correlation_lookup = {
        (item["detector"], item["test"], item["image_id"]): item["correlation"] for item in records
    }
    rows = [(detector, case) for detector in DETECTORS for case in panel_cases]
    figure, axes = plt.subplots(
        len(rows),
        5,
        figsize=(config.panel.width_inches, config.panel.row_height_inches * len(rows)),
        squeeze=False,
    )
    column_titles = (
        "Original radiograph",
        "Trained model CAM",
        "Xavier-randomized CAM",
        "Pixel-shuffled input",
        "Trained model CAM\non shuffled pixels",
    )
    detector_titles = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    for row_number, (detector, case) in enumerate(rows):
        image_id = case["image_id"]
        rendered = panel_maps[(detector, image_id)]
        original = rendered["image"]
        shuffled = rendered["shuffled_image"]
        axes[row_number, 0].imshow(original, cmap="gray")
        _overlay(axes[row_number, 1], original, rendered["trained"], config)
        _overlay(axes[row_number, 2], original, rendered["parameter_randomized"], config)
        axes[row_number, 3].imshow(shuffled, cmap="gray")
        _overlay(axes[row_number, 4], shuffled, rendered["data_randomized"], config)
        parameter_correlation = correlation_lookup[(detector, "parameter_randomization", image_id)]
        data_correlation = correlation_lookup[(detector, "data_randomization", image_id)]
        parameter_text = "NA" if parameter_correlation is None else f"{parameter_correlation:.3f}"
        data_text = "NA" if data_correlation is None else f"{data_correlation:.3f}"
        axes[row_number, 2].set_title(f"r = {parameter_text}", fontsize=9)
        axes[row_number, 4].set_title(f"r = {data_text}", fontsize=9)
        axes[row_number, 0].set_ylabel(
            f"{detector_titles[detector]}\n{case['study_stratum']}",
            fontsize=9,
        )
        for column_number, axis in enumerate(axes[row_number]):
            axis.set_xticks([])
            axis.set_yticks([])
            if row_number == 0 and column_number not in {2, 4}:
                axis.set_title(column_titles[column_number], fontsize=10)
    axes[0, 2].text(
        0.5,
        1.12,
        column_titles[2],
        transform=axes[0, 2].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    axes[0, 4].text(
        0.5,
        1.12,
        column_titles[4],
        transform=axes[0, 4].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.suptitle(
        "Grad-CAM sanity checks (cyan: differentiated candidate box)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=config.panel.dpi,
        bbox_inches="tight",
        metadata={"Software": "meddet-benchmark"},
    )
    plt.close(figure)
    return _atomic_bytes(path, buffer.getvalue())


def _source_identity(config: XaiSanityConfig) -> dict[str, str]:
    paths = {
        "sanity_checks": Path(__file__).resolve(),
        "phase7_runner": config.project_root / "src/explainability/run_explainability.py",
        "gradcam": config.project_root / "src/explainability/gradcam.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run_sanity_checks(config: XaiSanityConfig) -> dict[str, Any]:
    """Execute both sanity tests for both seed-17 detector checkpoints."""

    from src.utils.seed import initialize_reproducibility, seed_everything

    initialize_reproducibility(config.seed, config.resolve(config.outputs.log_dir))
    prepared = _prepare_sanity(config)
    panel_cases = select_panel_cases(
        prepared["selected_rows"],
        id_column=config.sampling.id_column,
        stratum_column=config.sampling.stratum_column,
    )
    panel_names = {item["image_id"] for item in panel_cases}
    records: list[dict[str, Any]] = []
    panel_maps: dict[tuple[DetectorName, str], dict[str, Any]] = {}
    initialization: dict[str, Any] = {}
    for detector in DETECTORS:
        seed_everything(config.seed)
        detector_rows, detector_panel, audit = _run_detector(
            config,
            prepared,
            detector,
            panel_names,
        )
        records.extend(detector_rows)
        panel_maps.update(detector_panel)
        initialization[detector] = audit
    records.sort(key=lambda item: (item["detector"], item["test"], item["image_id"]))
    if len(records) != config.sampling.size * len(DETECTORS) * 2:
        raise AssertionError("Batch 21 detector/test/image grid is incomplete")
    aggregate = aggregate_sanity(
        records,
        sampling_audit=prepared["sampling_audit"],
        failure_threshold=config.correlation.sanity_failure_threshold,
    )
    detail_path = _atomic_csv(
        config.resolve(config.outputs.detail_table),
        DETAIL_FIELDS,
        records,
    )
    summary_path = _atomic_csv(
        config.resolve(config.outputs.summary_table),
        SUMMARY_FIELDS,
        aggregate,
    )
    figure_path = _write_panel(
        config.resolve(config.outputs.panel_figure),
        config=config,
        panel_cases=panel_cases,
        panel_maps=panel_maps,
        records=records,
    )
    phase7: ExplainabilityConfig = prepared["phase7"]
    phase7_prepared = prepared["phase7_prepared"]
    phase6 = phase7_prepared["phase6"]
    checkpoints = {
        item.detector: sha256_file(phase6.resolve(item.checkpoint)) for item in phase6.detectors
    }
    artifacts = {
        "subset_manifest": config.resolve(config.sampling.output_manifest).as_posix(),
        "subset_manifest_sha256": sha256_file(config.resolve(config.sampling.output_manifest)),
        "detail_table": detail_path.as_posix(),
        "detail_table_sha256": sha256_file(detail_path),
        "summary_table": summary_path.as_posix(),
        "summary_table_sha256": sha256_file(summary_path),
        "panel_figure": figure_path.as_posix(),
        "panel_figure_sha256": sha256_file(figure_path),
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "seed_scope": {
            "training_seed": config.seed,
            "scope": "primary checkpoint only for both detectors",
        },
        "config_path": config.source_path.as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": _source_identity(config),
        "phase7_provenance": {
            "config_path": phase7.source_path.as_posix(),
            "config_sha256": sha256_file(phase7.source_path),
            "summary_path": prepared["phase7_summary_path"].as_posix(),
            "summary_sha256": sha256_file(prepared["phase7_summary_path"]),
            "target_layers": [item.model_dump(mode="json") for item in phase7.detectors],
        },
        "checkpoints": checkpoints,
        "sampling": prepared["sampling_audit"],
        "target": config.target.model_dump(mode="json"),
        "parameter_randomization": {
            **config.parameter_randomization.model_dump(mode="json"),
            "per_detector_initialization": initialization,
        },
        "data_randomization": config.data_randomization.model_dump(mode="json"),
        "correlation": {
            **config.correlation.model_dump(mode="json"),
            "formula": "C_sanity = (1/K) * sum_k Corr(M_trained[k], M_randomized[k])",
            "k_definition": "image pairs with two valid, nonconstant heatmaps",
        },
        "aggregate": aggregate,
        "per_image": records,
        "qualitative_selection": {
            "settings": config.panel.model_dump(mode="json"),
            "cases": panel_cases,
        },
        "artifacts": artifacts,
    }
    machine_summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    artifacts["summary_json"] = machine_summary_path.as_posix()
    artifacts["summary_json_sha256"] = sha256_file(machine_summary_path)
    return {"summary": summary, "artifacts": artifacts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai_sanity.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_xai_sanity_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True), flush=True)
        return 0
    result = run_sanity_checks(config)
    print(
        json.dumps(
            {"status": result["summary"]["status"], "artifacts": result["artifacts"]},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
