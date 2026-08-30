"""Run parameter-sensitivity and input-perturbation controls for detector Grad-CAM."""

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

ControlName = Literal[
    "cascading_model_parameter_randomization",
    "input_pixel_randomization_control",
]

DETAIL_FIELDS = (
    "detector",
    "control",
    "cascade_stage_index",
    "cascade_stage_name",
    "randomized_group_names",
    "randomized_module_prefixes",
    "full_model_parameter_randomization",
    "training_data_randomization_performed",
    "image_id",
    "nih_patient_id",
    "study_stratum",
    "trained_valid",
    "trained_failure_reason",
    "randomized_valid",
    "randomized_failure_reason",
    "similarity_valid",
    "similarity_failure_reason",
    "pearson_correlation",
    "spearman_correlation",
    "ssim",
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
    "control",
    "cascade_stage_index",
    "cascade_stage_name",
    "randomized_group_count",
    "randomized_group_names",
    "randomized_module_prefixes",
    "full_model_parameter_randomization",
    "training_data_randomization_performed",
    "subset_image_count",
    "subset_patient_count",
    "subset_stratum_counts",
    "trained_valid_count",
    "trained_failure_count",
    "trained_failure_rate",
    "randomized_valid_count",
    "randomized_failure_count",
    "randomized_failure_rate",
    "k_valid_similarity_pairs",
    "map_pair_failure_count",
    "map_pair_failure_rate",
    "pearson_mean",
    "pearson_std",
    "pearson_median",
    "pearson_min",
    "pearson_max",
    "spearman_mean",
    "spearman_std",
    "spearman_median",
    "spearman_min",
    "spearman_max",
    "ssim_mean",
    "ssim_std",
    "ssim_median",
    "ssim_min",
    "ssim_max",
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


class ParameterInitializationSettings(StrictModel):
    """Shared model-copy and parameter-initialization policy."""

    copy_method: Literal["deep_copy_trained_model"]
    weight_initialization: Literal["xavier_normal"]
    one_dimensional_weight_view: Literal["row_vector"]
    bias_initialization: Literal["zeros"]
    preserve_non_weight_non_bias_buffers: Literal[True]
    autocast: Literal["disabled_for_randomized_weight_numerical_validity"]
    gain: float = Field(gt=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    rng_reset_per_cumulative_stage: Literal[True]


class LayerGroup(StrictModel):
    """One transparent, non-overlapping detector parameter group."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    description: str = Field(min_length=1)
    module_prefixes: tuple[str, ...] = Field(min_length=1)


class DetectorLayerGroups(StrictModel):
    """Output-to-input group order for one detector architecture."""

    detector: DetectorName
    groups: tuple[LayerGroup, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_groups(self) -> DetectorLayerGroups:
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate layer-group name for {self.detector}")
        prefixes = [prefix for group in self.groups for prefix in group.module_prefixes]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"duplicate module prefix for {self.detector}")
        return self


class CascadingParameterRandomizationSettings(StrictModel):
    """Cumulative head-to-input detector-parameter randomization policy."""

    order: Literal["detector_output_to_input"]
    cumulative: Literal[True]
    initialization: ParameterInitializationSettings
    detectors: tuple[DetectorLayerGroups, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_detectors(self) -> CascadingParameterRandomizationSettings:
        names = [item.detector for item in self.detectors]
        if set(names) != set(DETECTORS) or len(set(names)) != len(names):
            raise ValueError("cascading groups must define Faster R-CNN and YOLO11s exactly once")
        return self

    def layer_groups(self, detector: DetectorName) -> DetectorLayerGroups:
        """Return the declared groups for one detector."""

        return next(item for item in self.detectors if item.detector == detector)


class InputPixelRandomizationSettings(StrictModel):
    """Deterministic inference-time input-pixel perturbation policy."""

    method: Literal["spatial_pixel_vector_permutation_without_replacement"]
    seed_derivation: Literal["sha256_global_seed_and_image_id"]
    seed: int = Field(ge=0, le=2**32 - 1)
    interpretation: Literal["input_perturbation_stress_control_only"]


class SimilaritySettings(StrictModel):
    """Map normalization, degeneracy, and multi-metric similarity contract."""

    methods: tuple[Literal["pearson", "spearman", "ssim"], ...]
    population: Literal["valid_nonconstant_paired_maps"]
    preprocessing: Literal["bilinear_resize_then_independent_minmax_0_1"]
    evaluation_size: int = Field(gt=6)
    zero_energy_policy: Literal["exclude_and_report"]
    epsilon: float = Field(gt=0)
    ssim_gaussian_sigma: float = Field(gt=0)
    ssim_k1: float = Field(gt=0)
    ssim_k2: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_methods(self) -> SimilaritySettings:
        if tuple(self.methods) != ("pearson", "spearman", "ssim"):
            raise ValueError("v2 requires Pearson, Spearman, and SSIM in that order")
        return self


class ClaimBoundaries(StrictModel):
    """Machine-readable limits on what the controls establish."""

    adebayo_training_data_randomization: Literal["not_performed"]
    reason: Literal["requires_retraining_on_randomized_training_annotations_and_fit_verification"]
    parameter_control_interpretation: Literal["parameter_sensitivity_only"]


class HistoricalArtifacts(StrictModel):
    """Immutable Batch 21 outputs retained with their legacy labels."""

    summary_table: Path
    detail_table: Path
    panel_figure: Path
    summary_json: Path


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
    """Versioned Batch 31 generated artifacts."""

    log_dir: Path
    summary_json: Path
    detail_table: Path
    summary_table: Path
    panel_figure: Path


class XaiSanityConfig(StrictModel):
    """Complete immutable Batch 31 experiment contract."""

    schema_version: Literal[2]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    phase7_config: Path
    phase7_summary: Path
    sampling: SamplingSettings
    target: TargetSettings
    cascading_model_parameter_randomization: CascadingParameterRandomizationSettings
    input_pixel_randomization_control: InputPixelRandomizationSettings
    similarity: SimilaritySettings
    claim_boundaries: ClaimBoundaries
    historical_artifacts: HistoricalArtifacts
    runtime: RuntimeSettings
    panel: PanelSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_seeds(self) -> XaiSanityConfig:
        if self.sampling.selection_seed != self.seed:
            raise ValueError("nested sample must reuse the primary experiment seed")
        parameter_seed = self.cascading_model_parameter_randomization.initialization.seed
        pixel_seed = self.input_pixel_randomization_control.seed
        if len({parameter_seed, pixel_seed}) != 2:
            raise ValueError("parameter and input-pixel controls require distinct seeds")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_xai_sanity_config(path: str | Path) -> XaiSanityConfig:
    """Load the strict Batch 31 YAML contract."""

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


def _portable_path(config: XaiSanityConfig, path: Path) -> str:
    """Return a clone-stable repository-relative artifact path."""

    return path.resolve().relative_to(config.project_root).as_posix()


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
        "source_pool_manifest": _portable_path(config, source),
        "source_pool_manifest_sha256": sha256_file(source),
        "source_pool_image_count": len(rows),
        "source_pool_stratum_counts": dict(
            sorted(Counter(row[config.sampling.stratum_column] for row in rows).items())
        ),
        "output_manifest": _portable_path(config, output),
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
        "cascading_model_parameter_randomization": (
            config.cascading_model_parameter_randomization.model_dump(mode="json")
        ),
        "input_pixel_randomization_control": config.input_pixel_randomization_control.model_dump(
            mode="json"
        ),
        "claim_boundaries": config.claim_boundaries.model_dump(mode="json"),
        "historical_artifacts": _historical_artifact_identity(config),
        "panel_cases": panel_cases,
    }


def _historical_artifact_identity(config: XaiSanityConfig) -> dict[str, dict[str, str]]:
    """Resolve and hash the immutable Batch 21 artifacts retained for historical continuity."""

    return {
        name: {
            "path": _portable_path(config, config.resolve(path)),
            "sha256": sha256_file(config.resolve(path)),
        }
        for name, path in config.historical_artifacts.model_dump().items()
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


def _pearson_values(first: np.ndarray, second: np.ndarray, *, epsilon: float) -> float | None:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    first_norm = float(np.linalg.norm(first_centered))
    second_norm = float(np.linalg.norm(second_centered))
    if first_norm <= epsilon or second_norm <= epsilon:
        return None
    value = float(np.dot(first_centered, second_centered) / (first_norm * second_norm))
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else None


def pearson_map_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float,
) -> float | None:
    """Return Pearson correlation or null for a non-finite or constant map pair."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("heatmaps must have matching two-dimensional shapes")
    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        return None
    return _pearson_values(first_values, second_values, epsilon=epsilon)


def normalize_map_for_similarity(
    heatmap: np.ndarray,
    *,
    evaluation_size: int,
    epsilon: float,
) -> np.ndarray | None:
    """Bilinearly resize one finite map and independently min-max normalize it."""

    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        return None
    resized = np.asarray(
        Image.fromarray(values.astype(np.float32), mode="F").resize(
            (evaluation_size, evaluation_size),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.float64,
    )
    minimum = float(resized.min())
    value_range = float(resized.max() - minimum)
    if not np.isfinite(value_range) or value_range <= epsilon:
        return None
    return np.ascontiguousarray((resized - minimum) / value_range)


def spearman_map_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float,
) -> float | None:
    """Return tie-aware Spearman correlation for two normalized maps."""

    from scipy.stats import rankdata

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("normalized heatmaps must have matching two-dimensional shapes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    first_ranks = np.asarray(rankdata(first.reshape(-1), method="average"), dtype=np.float64)
    second_ranks = np.asarray(rankdata(second.reshape(-1), method="average"), dtype=np.float64)
    return _pearson_values(first_ranks, second_ranks, epsilon=epsilon)


def ssim_map_similarity(
    first: np.ndarray,
    second: np.ndarray,
    *,
    sigma: float,
    k1: float,
    k2: float,
) -> float | None:
    """Return Gaussian-window SSIM for two maps normalized to the unit interval."""

    from scipy.ndimage import gaussian_filter

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("normalized heatmaps must have matching two-dimensional shapes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    mean_first = gaussian_filter(first_values, sigma=sigma, mode="reflect", truncate=3.5)
    mean_second = gaussian_filter(second_values, sigma=sigma, mode="reflect", truncate=3.5)
    variance_first = np.maximum(
        gaussian_filter(first_values * first_values, sigma=sigma, mode="reflect", truncate=3.5)
        - mean_first * mean_first,
        0.0,
    )
    variance_second = np.maximum(
        gaussian_filter(second_values * second_values, sigma=sigma, mode="reflect", truncate=3.5)
        - mean_second * mean_second,
        0.0,
    )
    covariance = (
        gaussian_filter(first_values * second_values, sigma=sigma, mode="reflect", truncate=3.5)
        - mean_first * mean_second
    )
    c1 = k1**2
    c2 = k2**2
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first * mean_first + mean_second * mean_second + c1) * (
        variance_first + variance_second + c2
    )
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        return None
    value = float(np.mean(numerator / denominator, dtype=np.float64))
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else None


def map_similarity_metrics(
    first: np.ndarray,
    second: np.ndarray,
    *,
    settings: SimilaritySettings,
) -> dict[str, Any]:
    """Normalize a map pair once and compute Pearson, Spearman, and SSIM."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("control heatmaps must have matching two-dimensional shapes")
    first_normalized = normalize_map_for_similarity(
        first,
        evaluation_size=settings.evaluation_size,
        epsilon=settings.epsilon,
    )
    second_normalized = normalize_map_for_similarity(
        second,
        evaluation_size=settings.evaluation_size,
        epsilon=settings.epsilon,
    )
    if first_normalized is None or second_normalized is None:
        return {
            "valid": False,
            "failure_reason": "nonfinite_or_degenerate_map_after_similarity_preprocessing",
            "pearson_correlation": None,
            "spearman_correlation": None,
            "ssim": None,
        }
    pearson = pearson_map_correlation(
        first_normalized,
        second_normalized,
        epsilon=settings.epsilon,
    )
    spearman = spearman_map_correlation(
        first_normalized,
        second_normalized,
        epsilon=settings.epsilon,
    )
    ssim = ssim_map_similarity(
        first_normalized,
        second_normalized,
        sigma=settings.ssim_gaussian_sigma,
        k1=settings.ssim_k1,
        k2=settings.ssim_k2,
    )
    if pearson is None or spearman is None or ssim is None:
        return {
            "valid": False,
            "failure_reason": "undefined_similarity_metric",
            "pearson_correlation": None,
            "spearman_correlation": None,
            "ssim": None,
        }
    return {
        "valid": True,
        "failure_reason": None,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "ssim": ssim,
    }


def xavier_reinitialize_model(
    model: Any,
    *,
    seed: int,
    gain: float,
    module_prefixes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Xavier-randomize selected module weights and zero their biases."""

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
    randomized_modules: list[str] = []
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if module_prefixes is not None and not any(
                _module_matches_prefix(module_name, prefix) for prefix in module_prefixes
            ):
                continue
            weight = getattr(module, "weight", None)
            module_changed = False
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
                module_changed = True
            bias = getattr(module, "bias", None)
            if isinstance(bias, torch.Tensor) and id(bias) not in seen_biases:
                if not bias.is_floating_point():
                    raise TypeError(f"cannot zero non-floating bias: {module_name}")
                torch.nn.init.zeros_(bias)
                seen_biases.add(id(bias))
                bias_tensors += 1
                bias_values += bias.numel()
                bias_buffer_tensors += int("bias" in module._buffers)
                module_changed = True
            if module_changed:
                randomized_modules.append(module_name)
    if xavier_tensors == 0:
        raise ValueError("model contains no non-bias parameter to Xavier-randomize")
    return {
        "seed": seed,
        "gain": gain,
        "generator": "torch.Generator(device=cpu)",
        "module_prefixes": None if module_prefixes is None else list(module_prefixes),
        "randomized_modules": randomized_modules,
        "xavier_weight_tensor_count": xavier_tensors,
        "xavier_weight_value_count": xavier_values,
        "one_dimensional_row_view_tensor_count": one_dimensional_tensors,
        "xavier_weight_buffer_tensor_count": weight_buffer_tensors,
        "zeroed_bias_tensor_count": bias_tensors,
        "zeroed_bias_value_count": bias_values,
        "zeroed_bias_buffer_tensor_count": bias_buffer_tensors,
        "non_weight_non_bias_buffers_preserved": True,
    }


def _module_matches_prefix(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def audit_layer_group_partition(model: Any, groups: DetectorLayerGroups) -> dict[str, Any]:
    """Require every weight/bias-bearing module to belong to exactly one declared group."""

    import torch

    grouped_modules: dict[str, list[str]] = {group.name: [] for group in groups.groups}
    unassigned: list[str] = []
    multiply_assigned: dict[str, list[str]] = {}
    for module_name, module in model.named_modules():
        if not any(
            isinstance(getattr(module, name, None), torch.Tensor) for name in ("weight", "bias")
        ):
            continue
        matched = [
            group.name
            for group in groups.groups
            if any(_module_matches_prefix(module_name, prefix) for prefix in group.module_prefixes)
        ]
        if not matched:
            unassigned.append(module_name)
        elif len(matched) > 1:
            multiply_assigned[module_name] = matched
        else:
            grouped_modules[matched[0]].append(module_name)
    empty = [name for name, modules in grouped_modules.items() if not modules]
    if unassigned or multiply_assigned or empty:
        raise ValueError(
            "invalid detector layer-group partition: "
            f"unassigned={unassigned}, multiply_assigned={multiply_assigned}, empty={empty}"
        )
    return {
        "detector": groups.detector,
        "group_order": [group.name for group in groups.groups],
        "group_module_counts": {name: len(modules) for name, modules in grouped_modules.items()},
        "group_modules": grouped_modules,
        "weight_bias_module_count": sum(len(modules) for modules in grouped_modules.values()),
        "partition_complete_and_nonoverlapping": True,
    }


def randomize_cumulative_model(
    model: Any,
    *,
    groups: DetectorLayerGroups,
    stage_index: int,
    settings: ParameterInitializationSettings,
) -> tuple[Any, dict[str, Any]]:
    """Deep-copy a trained model and randomize groups through one cumulative stage."""

    if not 1 <= stage_index <= len(groups.groups):
        raise ValueError("cascade stage index is outside the declared layer-group order")
    partition = audit_layer_group_partition(model, groups)
    selected_groups = groups.groups[:stage_index]
    prefixes = [prefix for group in selected_groups for prefix in group.module_prefixes]
    randomized_model = copy.deepcopy(model)
    initialization = xavier_reinitialize_model(
        randomized_model,
        seed=settings.seed,
        gain=settings.gain,
        module_prefixes=prefixes,
    )
    audit = {
        "stage_index": stage_index,
        "stage_name": selected_groups[-1].name,
        "randomized_group_names": [group.name for group in selected_groups],
        "randomized_group_descriptions": [group.description for group in selected_groups],
        "randomized_module_prefixes": prefixes,
        "full_model_parameter_randomization": stage_index == len(groups.groups),
        "partition": partition,
        "initialization": initialization,
    }
    return randomized_model, audit


def checkpoint_hashes(paths: Mapping[DetectorName, Path]) -> dict[str, str]:
    """Hash the on-disk detector checkpoints without loading or modifying them."""

    return {detector: sha256_file(path) for detector, path in paths.items()}


def assert_checkpoint_immutability(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    """Fail if any checkpoint identity changed during an in-memory control run."""

    if dict(before) != dict(after):
        raise RuntimeError(f"checkpoint immutability violated: before={before}, after={after}")


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
    control: ControlName,
    stage_audit: Mapping[str, Any] | None,
    source_row: Mapping[str, str],
    config: XaiSanityConfig,
    reference: Mapping[str, Any],
    trained: Mapping[str, Any],
    randomized: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {
        "valid": False,
        "failure_reason": None,
        "pearson_correlation": None,
        "spearman_correlation": None,
        "ssim": None,
    }
    if trained["valid"] and randomized["valid"]:
        metrics = map_similarity_metrics(
            trained["heatmap"],
            randomized["heatmap"],
            settings=config.similarity,
        )
    else:
        failures = []
        if not trained["valid"]:
            failures.append("trained_cam_invalid")
        if not randomized["valid"]:
            failures.append("randomized_cam_invalid")
        metrics["failure_reason"] = "+".join(failures)
    group_names = [] if stage_audit is None else stage_audit["randomized_group_names"]
    prefixes = [] if stage_audit is None else stage_audit["randomized_module_prefixes"]
    stage_index = 0 if stage_audit is None else int(stage_audit["stage_index"])
    stage_name = "input_pixel_permutation" if stage_audit is None else stage_audit["stage_name"]
    full_randomization = bool(
        stage_audit is not None and stage_audit["full_model_parameter_randomization"]
    )
    box = reference.get("candidate_box")
    coordinates = [None, None, None, None] if box is None else [float(value) for value in box]
    return {
        "detector": detector,
        "control": control,
        "cascade_stage_index": stage_index,
        "cascade_stage_name": stage_name,
        "randomized_group_names": json.dumps(group_names, separators=(",", ":")),
        "randomized_module_prefixes": json.dumps(prefixes, separators=(",", ":")),
        "full_model_parameter_randomization": full_randomization,
        "training_data_randomization_performed": False,
        "image_id": source_row[config.sampling.id_column],
        "nih_patient_id": source_row[config.sampling.patient_column],
        "study_stratum": source_row[config.sampling.stratum_column],
        "trained_valid": bool(trained["valid"]),
        "trained_failure_reason": trained["failure_reason"],
        "randomized_valid": bool(randomized["valid"]),
        "randomized_failure_reason": randomized["failure_reason"],
        "similarity_valid": bool(metrics["valid"]),
        "similarity_failure_reason": metrics["failure_reason"],
        "pearson_correlation": metrics["pearson_correlation"],
        "spearman_correlation": metrics["spearman_correlation"],
        "ssim": metrics["ssim"],
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
                    epsilon=config.similarity.epsilon,
                )
                if reference_box is not None
                else _missing_reference_result()
            )
            clean_results[image_id] = trained
            shuffled = shuffle_pixel_vectors(
                image,
                seed=config.input_pixel_randomization_control.seed,
                image_id=image_id,
            )
            input_randomized = (
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
                    epsilon=config.similarity.epsilon,
                )
                if reference_box is not None
                else _missing_reference_result()
            )
            detail.append(
                _detail_row(
                    detector=detector,
                    control="input_pixel_randomization_control",
                    stage_audit=None,
                    source_row=source_rows[image_id],
                    config=config,
                    reference=reference,
                    trained=trained,
                    randomized=input_randomized,
                )
            )
            if image_id in panel_names:
                panel_maps[(detector, image_id)] = {
                    "image": np.asarray(image).copy(),
                    "shuffled_image": np.asarray(shuffled).copy(),
                    "trained": trained,
                    "input_pixel_randomized": input_randomized,
                    "cascading": {},
                }
            if number % config.runtime.progress_every_images == 0:
                print(
                    f"[{detector}] clean/input-pixel control {number}/{len(image_names)}",
                    flush=True,
                )

    model.to("cpu")
    torch.cuda.empty_cache()
    cascade_settings = config.cascading_model_parameter_randomization
    detector_groups = cascade_settings.layer_groups(detector)
    partition_audit = audit_layer_group_partition(model, detector_groups)
    stage_audits: list[dict[str, Any]] = []
    for stage_index in range(1, len(detector_groups.groups) + 1):
        randomized_model, stage_audit = randomize_cumulative_model(
            model,
            groups=detector_groups,
            stage_index=stage_index,
            settings=cascade_settings.initialization,
        )
        stage_audits.append(stage_audit)
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
                        epsilon=config.similarity.epsilon,
                    )
                    if reference_box is not None
                    else _missing_reference_result()
                )
                detail.append(
                    _detail_row(
                        detector=detector,
                        control="cascading_model_parameter_randomization",
                        stage_audit=stage_audit,
                        source_row=source_rows[image_id],
                        config=config,
                        reference=reference,
                        trained=trained,
                        randomized=parameter_randomized,
                    )
                )
                if image_id in panel_names:
                    panel_maps[(detector, image_id)]["cascading"][stage_audit["stage_name"]] = (
                        parameter_randomized
                    )
                if number % config.runtime.progress_every_images == 0:
                    print(
                        f"[{detector}] cascade {stage_index}/{len(detector_groups.groups)} "
                        f"{number}/{len(image_names)}",
                        flush=True,
                    )
        randomized_model.to("cpu")
        del randomized_model
        gc.collect()
        torch.cuda.empty_cache()

    del model, clean_results, reference_results
    gc.collect()
    torch.cuda.empty_cache()
    return detail, panel_maps, {"partition": partition_audit, "stages": stage_audits}


def aggregate_sanity(
    records: Sequence[Mapping[str, Any]],
    *,
    sampling_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Summarize valid-pair Pearson, Spearman, and SSIM by control stage."""

    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        conditions = sorted(
            {
                (item["control"], int(item["cascade_stage_index"]), item["cascade_stage_name"])
                for item in records
                if item["detector"] == detector
            },
            key=lambda item: (0 if item[0] == "input_pixel_randomization_control" else 1, item[1]),
        )
        for control, stage_index, stage_name in conditions:
            group = [
                item
                for item in records
                if item["detector"] == detector
                and item["control"] == control
                and int(item["cascade_stage_index"]) == stage_index
            ]
            if len(group) != sampling_audit["subset_image_count"]:
                raise AssertionError(
                    f"incomplete control grid for {detector}/{control}/{stage_name}"
                )
            valid = [item for item in group if item["similarity_valid"]]
            metric_values = {
                metric: np.asarray([float(item[metric]) for item in valid], dtype=np.float64)
                for metric in ("pearson_correlation", "spearman_correlation", "ssim")
            }
            trained_valid = sum(bool(item["trained_valid"]) for item in group)
            randomized_valid = sum(bool(item["randomized_valid"]) for item in group)
            total = len(group)
            valid_pairs = len(valid)
            row = {
                "detector": detector,
                "control": control,
                "cascade_stage_index": stage_index,
                "cascade_stage_name": stage_name,
                "randomized_group_count": len(json.loads(group[0]["randomized_group_names"])),
                "randomized_group_names": group[0]["randomized_group_names"],
                "randomized_module_prefixes": group[0]["randomized_module_prefixes"],
                "full_model_parameter_randomization": group[0][
                    "full_model_parameter_randomization"
                ],
                "training_data_randomization_performed": False,
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
                "k_valid_similarity_pairs": valid_pairs,
                "map_pair_failure_count": total - valid_pairs,
                "map_pair_failure_rate": (total - valid_pairs) / total,
            }
            for metric, values in metric_values.items():
                prefix = metric.removesuffix("_correlation")
                row[f"{prefix}_mean"] = float(values.mean()) if valid_pairs else None
                row[f"{prefix}_std"] = float(values.std(ddof=1)) if valid_pairs > 1 else None
                row[f"{prefix}_median"] = float(np.median(values)) if valid_pairs else None
                row[f"{prefix}_min"] = float(values.min()) if valid_pairs else None
                row[f"{prefix}_max"] = float(values.max()) if valid_pairs else None
            rows.append(row)
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

    similarity_lookup = {
        (
            item["detector"],
            item["control"],
            int(item["cascade_stage_index"]),
            item["image_id"],
        ): item
        for item in records
    }
    detector_groups = {
        detector: config.cascading_model_parameter_randomization.layer_groups(detector)
        for detector in DETECTORS
    }
    stage_count = len(detector_groups[DETECTORS[0]].groups)
    if any(len(groups.groups) != stage_count for groups in detector_groups.values()):
        raise ValueError("the qualitative panel requires the same cascade-stage count per detector")
    rows = [(detector, case) for detector in DETECTORS for case in panel_cases]
    figure, axes = plt.subplots(
        len(rows),
        4 + stage_count,
        figsize=(config.panel.width_inches, config.panel.row_height_inches * len(rows)),
        squeeze=False,
    )
    fixed_titles = (
        "Original radiograph",
        "Trained model CAM",
        "Pixel-shuffled input",
        "CAM on shuffled pixels",
    )
    detector_titles = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    for row_number, (detector, case) in enumerate(rows):
        image_id = case["image_id"]
        rendered = panel_maps[(detector, image_id)]
        original = rendered["image"]
        shuffled = rendered["shuffled_image"]
        axes[row_number, 0].imshow(original, cmap="gray")
        _overlay(axes[row_number, 1], original, rendered["trained"], config)
        axes[row_number, 2].imshow(shuffled, cmap="gray")
        _overlay(axes[row_number, 3], shuffled, rendered["input_pixel_randomized"], config)
        input_record = similarity_lookup[
            (detector, "input_pixel_randomization_control", 0, image_id)
        ]
        input_text = (
            "NA"
            if not input_record["similarity_valid"]
            else f"rho={input_record['spearman_correlation']:.2f}\nSSIM={input_record['ssim']:.2f}"
        )
        axes[row_number, 3].text(
            0.02,
            0.02,
            input_text,
            transform=axes[row_number, 3].transAxes,
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
        for stage_index, group in enumerate(detector_groups[detector].groups, start=1):
            axis = axes[row_number, 3 + stage_index]
            _overlay(axis, original, rendered["cascading"][group.name], config)
            stage_record = similarity_lookup[
                (detector, "cascading_model_parameter_randomization", stage_index, image_id)
            ]
            metric_text = (
                f"{group.name}\nNA"
                if not stage_record["similarity_valid"]
                else (
                    f"{group.name}\nrho={stage_record['spearman_correlation']:.2f} "
                    f"SSIM={stage_record['ssim']:.2f}"
                )
            )
            axis.text(
                0.02,
                0.02,
                metric_text,
                transform=axis.transAxes,
                fontsize=6,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
            )
        axes[row_number, 0].set_ylabel(
            f"{detector_titles[detector]}\n{case['study_stratum']}",
            fontsize=9,
        )
        for column_number, axis in enumerate(axes[row_number]):
            axis.set_xticks([])
            axis.set_yticks([])
            if row_number == 0:
                if column_number < len(fixed_titles):
                    title = fixed_titles[column_number]
                else:
                    title = f"Cascade {column_number - len(fixed_titles) + 1}"
                axis.set_title(title.replace("_", " "), fontsize=8)
    figure.suptitle(
        "Grad-CAM v2 controls: input-pixel perturbation and cumulative parameter randomization",
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
    """Execute v2 controls for both seed-17 detector checkpoints."""

    from src.utils.seed import initialize_reproducibility, seed_everything

    initialize_reproducibility(config.seed, config.resolve(config.outputs.log_dir))
    prepared = _prepare_sanity(config)
    phase7_prepared = prepared["phase7_prepared"]
    phase6 = phase7_prepared["phase6"]
    checkpoint_paths = {item.detector: phase6.resolve(item.checkpoint) for item in phase6.detectors}
    checkpoints_before = checkpoint_hashes(checkpoint_paths)
    panel_cases = select_panel_cases(
        prepared["selected_rows"],
        id_column=config.sampling.id_column,
        stratum_column=config.sampling.stratum_column,
    )
    panel_names = {item["image_id"] for item in panel_cases}
    records: list[dict[str, Any]] = []
    panel_maps: dict[tuple[DetectorName, str], dict[str, Any]] = {}
    cascade_audits: dict[str, Any] = {}
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
        cascade_audits[detector] = audit
    records.sort(
        key=lambda item: (
            item["detector"],
            item["control"],
            int(item["cascade_stage_index"]),
            item["image_id"],
        )
    )
    condition_count = sum(
        1 + len(config.cascading_model_parameter_randomization.layer_groups(detector).groups)
        for detector in DETECTORS
    )
    if len(records) != config.sampling.size * condition_count:
        raise AssertionError("Batch 31 detector/control/stage/image grid is incomplete")
    aggregate = aggregate_sanity(
        records,
        sampling_audit=prepared["sampling_audit"],
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
    checkpoints_after = checkpoint_hashes(checkpoint_paths)
    assert_checkpoint_immutability(checkpoints_before, checkpoints_after)
    phase7: ExplainabilityConfig = prepared["phase7"]
    artifacts = {
        "subset_manifest": _portable_path(config, config.resolve(config.sampling.output_manifest)),
        "subset_manifest_sha256": sha256_file(config.resolve(config.sampling.output_manifest)),
        "detail_table": _portable_path(config, detail_path),
        "detail_table_sha256": sha256_file(detail_path),
        "summary_table": _portable_path(config, summary_path),
        "summary_table_sha256": sha256_file(summary_path),
        "panel_figure": _portable_path(config, figure_path),
        "panel_figure_sha256": sha256_file(figure_path),
    }
    summary = {
        "schema_version": 2,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "seed_scope": {
            "training_seed": config.seed,
            "scope": "primary checkpoint only for both detectors",
        },
        "config_path": _portable_path(config, config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": _source_identity(config),
        "phase7_provenance": {
            "config_path": _portable_path(config, phase7.source_path),
            "config_sha256": sha256_file(phase7.source_path),
            "summary_path": _portable_path(config, prepared["phase7_summary_path"]),
            "summary_sha256": sha256_file(prepared["phase7_summary_path"]),
            "target_layers": [item.model_dump(mode="json") for item in phase7.detectors],
        },
        "checkpoints": checkpoints_before,
        "checkpoint_immutability": {
            "before": checkpoints_before,
            "after": checkpoints_after,
            "verified": True,
            "mutation_scope": "in_memory_deep_copies_only",
        },
        "sampling": prepared["sampling_audit"],
        "target": config.target.model_dump(mode="json"),
        "cascading_model_parameter_randomization": {
            **config.cascading_model_parameter_randomization.model_dump(mode="json"),
            "per_detector_audit": cascade_audits,
        },
        "input_pixel_randomization_control": config.input_pixel_randomization_control.model_dump(
            mode="json"
        ),
        "claim_boundaries": config.claim_boundaries.model_dump(mode="json"),
        "similarity": {
            **config.similarity.model_dump(mode="json"),
            "pearson": "linear correlation of normalized evaluation-grid values",
            "spearman": "Pearson correlation of average ranks on normalized evaluation-grid values",
            "ssim": "mean Gaussian-window structural similarity with data_range=1",
            "k_definition": "pairs with two finite, nonconstant maps after preprocessing",
            "degenerate_map_handling": (
                "exclude from every metric and report the shared failure count"
            ),
        },
        "historical_batch21_artifacts": _historical_artifact_identity(config),
        "aggregate": aggregate,
        "per_image": records,
        "qualitative_selection": {
            "settings": config.panel.model_dump(mode="json"),
            "cases": panel_cases,
        },
        "artifacts": artifacts,
    }
    machine_summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    artifacts["summary_json"] = _portable_path(config, machine_summary_path)
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
