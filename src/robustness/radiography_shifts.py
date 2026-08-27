"""Evaluate acquisition-physics-motivated shifts on raw planar radiographs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.prepare import load_dataset_config, scale_radiograph_to_uint8
from src.evaluate import evaluate_prediction_records, sha256_file
from src.robustness.run_robustness import (
    METRIC_FIELDS,
    CorruptedSubsetDataset,
    DetectorSettings,
    Phase6Config,
    _atomic_csv,
    _atomic_json,
    _cached_result,
    _clean_predictions,
    _collect_faster_predictions,
    _collect_yolo_predictions,
    _load_faster_model,
    _metric_values,
    _prepare_experiment,
    _read_gzip_json,
    _targets_from_dataset,
    _write_prediction_bundle,
    load_robustness_config,
)

ShiftFamily = Literal["voi_window", "dose_noise", "detector_blur"]
ShiftKind = Literal["dicom_linear_window", "poisson_dose", "gaussian_blur"]


class StrictModel(BaseModel):
    """Reject undeclared config keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Frozen upstream and raw-data inputs."""

    phase6_config: Path
    dataset_config: Path
    subset_manifest: Path
    expected_subset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_id_column: str = Field(min_length=1)
    raw_file_column: str = Field(min_length=1)


class MetadataContract(StrictModel):
    """Expected metadata for the frozen raw-radiograph sample."""

    allowed_modalities: tuple[str, ...] = Field(min_length=1)
    allowed_photometric_interpretations: tuple[str, ...] = Field(min_length=1)
    bits_allocated: int = Field(ge=1, le=64)
    bits_stored: int = Field(ge=1, le=64)
    pixel_representation: Literal[0, 1]
    require_native_voi_absent: bool
    require_modality_transform_absent: bool


class RuntimeSettings(StrictModel):
    """Bounded smoke-test settings; inference settings come from Phase 6."""

    smoke_images: int = Field(ge=1, le=10)


class RadiographyShift(StrictModel):
    """One raw-array radiography shift and its declared parameters."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    family: ShiftFamily
    kind: ShiftKind
    center_offset_fraction: float | None = None
    width_multiplier: float | None = None
    reference_full_scale_counts: float | None = None
    dose_fraction: float | None = None
    gaussian_kernel_size: int | None = None
    gaussian_sigma_pixels: float | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> RadiographyShift:
        """Require exactly the parameters used by the selected shift kind."""

        values = {
            "center_offset_fraction": self.center_offset_fraction,
            "width_multiplier": self.width_multiplier,
            "reference_full_scale_counts": self.reference_full_scale_counts,
            "dose_fraction": self.dose_fraction,
            "gaussian_kernel_size": self.gaussian_kernel_size,
            "gaussian_sigma_pixels": self.gaussian_sigma_pixels,
        }
        required = {
            "dicom_linear_window": {"center_offset_fraction", "width_multiplier"},
            "poisson_dose": {"reference_full_scale_counts", "dose_fraction"},
            "gaussian_blur": {"gaussian_kernel_size", "gaussian_sigma_pixels"},
        }[self.kind]
        present = {name for name, value in values.items() if value is not None}
        if present != required:
            raise ValueError(
                f"{self.kind} requires exactly {sorted(required)}, found {sorted(present)}"
            )
        expected_family = {
            "dicom_linear_window": "voi_window",
            "poisson_dose": "dose_noise",
            "gaussian_blur": "detector_blur",
        }[self.kind]
        if self.family != expected_family:
            raise ValueError(f"{self.kind} must use family {expected_family}")
        for name in required:
            value = float(values[name])
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.width_multiplier is not None and self.width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive")
        if self.reference_full_scale_counts is not None and self.reference_full_scale_counts <= 0:
            raise ValueError("reference_full_scale_counts must be positive")
        if self.dose_fraction is not None and not 0 < self.dose_fraction <= 1:
            raise ValueError("dose_fraction must be in (0, 1]")
        if self.gaussian_kernel_size is not None and (
            self.gaussian_kernel_size < 3 or self.gaussian_kernel_size % 2 == 0
        ):
            raise ValueError("gaussian_kernel_size must be an odd integer of at least 3")
        if self.gaussian_sigma_pixels is not None and self.gaussian_sigma_pixels <= 0:
            raise ValueError("gaussian_sigma_pixels must be positive")
        return self


class OutputSettings(StrictModel):
    """Full-run and isolated smoke-test artifacts."""

    log_dir: Path
    prediction_bundles_dir: Path
    smoke_log_dir: Path
    smoke_prediction_bundles_dir: Path
    smoke_summary_json: Path
    smoke_results_table: Path
    summary_json: Path
    results_table: Path


class AcquisitionShiftConfig(StrictModel):
    """Strict complete acquisition-shift experiment contract."""

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    inputs: InputSettings
    metadata_contract: MetadataContract
    runtime: RuntimeSettings
    shifts: tuple[RadiographyShift, ...] = Field(min_length=3)
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_grid(self) -> AcquisitionShiftConfig:
        """Require unique conditions and all three acquisition-shift families."""

        identifiers = [shift.id for shift in self.shifts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("radiography shift IDs must be unique")
        if {shift.family for shift in self.shifts} != {
            "voi_window",
            "dose_noise",
            "detector_blur",
        }:
            raise ValueError("shift grid must cover VOI windows, dose noise, and detector blur")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one config path against the project root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True, slots=True)
class PreparedAnalysis:
    """Validated frozen inputs and aggregated raw-file audit."""

    phase6: Phase6Config
    phase6_prepared: Mapping[str, Any]
    raw_paths: Mapping[str, Path]
    selected_rows: tuple[Mapping[str, str], ...]
    raw_audit: Mapping[str, Any]
    invert_monochrome1: bool


def load_acquisition_shift_config(path: str | Path) -> AcquisitionShiftConfig:
    """Load the acquisition-shift YAML without importing detector frameworks."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("acquisition-shift config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return AcquisitionShiftConfig.model_validate(payload)


def dicom_linear_window(
    pixels: np.ndarray,
    *,
    center: float,
    width: float,
    output_min: float,
    output_max: float,
) -> np.ndarray:
    """Apply the DICOM PS3.3 C.11.2.1.2.1 default ``LINEAR`` VOI function.

    The half-unit center offset and ``width - 1`` denominator are normative
    parts of the DICOM definition, rather than an approximate clipping rule.
    """

    if not all(math.isfinite(value) for value in (center, width, output_min, output_max)):
        raise ValueError("DICOM window parameters must be finite")
    if width < 1:
        raise ValueError("DICOM LINEAR Window Width must be at least 1")
    if output_max <= output_min:
        raise ValueError("DICOM window output_max must exceed output_min")
    array = np.asarray(pixels, dtype=np.float64)
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    if width == 1:
        return np.where(array <= lower, output_min, output_max).astype(np.float64)
    scaled = ((array - (center - 0.5)) / (width - 1.0) + 0.5) * (
        output_max - output_min
    ) + output_min
    return np.where(
        array <= lower,
        output_min,
        np.where(array > upper, output_max, scaled),
    ).astype(np.float64)


def _stored_value_range(dataset: Any) -> tuple[float, float]:
    bits_stored = int(dataset.BitsStored)
    representation = int(dataset.PixelRepresentation)
    if representation == 0:
        return 0.0, float(2**bits_stored - 1)
    return float(-(2 ** (bits_stored - 1))), float(2 ** (bits_stored - 1) - 1)


def _derived_seed(base_seed: int, image_id: str, shift_id: str) -> int:
    payload = f"{base_seed}|{image_id}|{shift_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def apply_radiography_shift(
    pixels: np.ndarray,
    dataset: Any,
    shift: RadiographyShift,
    *,
    seed: int,
    image_id: str,
) -> np.ndarray:
    """Apply one deterministic shift to a decoded raw radiograph array."""

    array = np.asarray(pixels, dtype=np.float64).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D raw radiograph, found shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Raw radiograph contains non-finite values")
    possible_low, possible_high = _stored_value_range(dataset)

    if shift.kind == "dicom_linear_window":
        input_low = float(array.min())
        input_high = float(array.max())
        if input_high <= input_low:
            raise ValueError("DICOM windowing requires a non-constant raw image")
        baseline_width = input_high - input_low + 1.0
        baseline_center = (input_low + input_high + 1.0) / 2.0
        center = baseline_center + float(shift.center_offset_fraction) * baseline_width
        width = max(1.0, baseline_width * float(shift.width_multiplier))
        return dicom_linear_window(
            array,
            center=center,
            width=width,
            output_min=possible_low,
            output_max=possible_high,
        )

    if shift.kind == "poisson_dose":
        dynamic_range = possible_high - possible_low
        normalized = np.clip((array - possible_low) / dynamic_range, 0.0, 1.0)
        expected_full_scale = float(shift.reference_full_scale_counts) * float(shift.dose_fraction)
        generator = np.random.default_rng(_derived_seed(seed, image_id, shift.id))
        sampled_counts = generator.poisson(normalized * expected_full_scale)
        estimate = possible_low + sampled_counts * (dynamic_range / expected_full_scale)
        return np.clip(estimate, possible_low, possible_high).astype(np.float64)

    if shift.kind == "gaussian_blur":
        from scipy.ndimage import gaussian_filter

        radius = (int(shift.gaussian_kernel_size) - 1) // 2
        return gaussian_filter(
            array,
            sigma=float(shift.gaussian_sigma_pixels),
            radius=radius,
            mode="reflect",
        ).astype(np.float64)

    raise AssertionError(f"Unsupported radiography shift: {shift.kind}")


class RadiographyShiftSubsetDataset(CorruptedSubsetDataset):
    """Frozen COCO subset whose pixels are regenerated from shifted raw DICOM arrays."""

    def __init__(
        self,
        base_dataset: Any,
        selected_names: set[str],
        *,
        raw_paths: Mapping[str, Path],
        shift: RadiographyShift,
        seed: int,
        invert_monochrome1: bool,
    ) -> None:
        super().__init__(base_dataset, selected_names, condition=None, seed=seed)
        if set(raw_paths) != selected_names:
            raise ValueError("raw DICOM mapping and selected processed-image IDs differ")
        self.raw_paths = dict(raw_paths)
        self.shift = shift
        self.seed = seed
        self.invert_monochrome1 = invert_monochrome1

    def load_pil(self, index: int) -> Image.Image:
        """Decode, shift, canonically scale, and return one RGB model input."""

        import pydicom

        record = self.records[index]
        dataset = pydicom.dcmread(self.raw_paths[record.file_name])
        shifted = apply_radiography_shift(
            dataset.pixel_array,
            dataset,
            self.shift,
            seed=self.seed,
            image_id=record.file_name,
        )
        output = scale_radiograph_to_uint8(
            shifted,
            photometric_interpretation=str(dataset.PhotometricInterpretation),
            invert_monochrome1=self.invert_monochrome1,
        )
        if output.shape != (record.height, record.width):
            raise ValueError(f"shifted DICOM dimensions changed for {record.file_name}")
        return Image.fromarray(output).convert("RGB")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _dataset_contract(config: AcquisitionShiftConfig) -> dict[str, Any]:
    dataset_path = config.resolve(config.inputs.dataset_config)
    payload = load_dataset_config(dataset_path)
    dataset = _require_mapping(payload.get("dataset"), "dataset")
    paths = _require_mapping(dataset.get("paths"), "dataset.paths")
    image = _require_mapping(dataset.get("image"), "dataset.image")
    conversion = _require_mapping(image.get("conversion"), "dataset.image.conversion")
    if (
        conversion.get("normalization") != "per_image_minmax"
        or int(conversion.get("output_bit_depth", 0)) != 8
    ):
        raise ValueError("acquisition shifts require the frozen 8-bit per-image min-max contract")
    raw_dir_value = paths.get("source_images_dir")
    if not isinstance(raw_dir_value, str) or not raw_dir_value:
        raise ValueError("dataset.paths.source_images_dir must be a non-empty path")
    raw_dir = Path(raw_dir_value)
    raw_dir = raw_dir if raw_dir.is_absolute() else config.project_root / raw_dir
    source_extension = image.get("source_extension")
    if not isinstance(source_extension, str) or not source_extension:
        raise ValueError("dataset.image.source_extension must be a non-empty string")
    return {
        "path": dataset_path,
        "raw_dir": raw_dir.resolve(),
        "source_extension": source_extension.lower(),
        "invert_monochrome1": bool(conversion.get("invert_monochrome1")),
    }


def _read_selected_rows(config: AcquisitionShiftConfig) -> tuple[Mapping[str, str], ...]:
    manifest = config.resolve(config.inputs.subset_manifest)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("frozen robustness manifest has no header")
        required = {config.inputs.processed_id_column, config.inputs.raw_file_column}
        if not required <= set(reader.fieldnames):
            raise ValueError(f"frozen robustness manifest lacks columns: {sorted(required)}")
        rows = tuple(dict(row) for row in reader)
    processed = [row[config.inputs.processed_id_column] for row in rows]
    raw = [row[config.inputs.raw_file_column] for row in rows]
    if len(set(processed)) != len(rows) or len(set(raw)) != len(rows):
        raise ValueError("frozen robustness manifest IDs must be one-to-one")
    return rows


def _native_voi_present(dataset: Any) -> bool:
    return bool(
        hasattr(dataset, "VOILUTSequence")
        or hasattr(dataset, "WindowCenter")
        or hasattr(dataset, "WindowWidth")
    )


def _modality_transform_present(dataset: Any) -> bool:
    return bool(
        hasattr(dataset, "ModalityLUTSequence")
        or hasattr(dataset, "RescaleSlope")
        or hasattr(dataset, "RescaleIntercept")
    )


def _audit_raw_files(
    config: AcquisitionShiftConfig,
    rows: Sequence[Mapping[str, str]],
    *,
    dataset_contract: Mapping[str, Any],
    base_dataset: Any,
) -> tuple[dict[str, Path], dict[str, Any]]:
    import pydicom

    raw_dir = Path(dataset_contract["raw_dir"])
    raw_paths: dict[str, Path] = {}
    missing: list[str] = []
    modality_counts: Counter[str] = Counter()
    photometric_counts: Counter[str] = Counter()
    native_voi_count = 0
    modality_transform_count = 0
    clean_pixel_mismatch_count = 0
    raw_minima: list[float] = []
    raw_maxima: list[float] = []
    raw_manifest = hashlib.sha256()
    base_paths = {
        record.file_name: base_dataset.image_path(index)
        for index, record in enumerate(base_dataset.records)
    }

    for row in rows:
        processed_name = row[config.inputs.processed_id_column]
        raw_name = row[config.inputs.raw_file_column]
        if Path(raw_name).name != raw_name or Path(raw_name).suffix.lower() != str(
            dataset_contract["source_extension"]
        ):
            raise ValueError(f"unsafe or unexpected raw DICOM filename: {raw_name}")
        path = (raw_dir / raw_name).resolve()
        try:
            path.relative_to(raw_dir)
        except ValueError as error:
            raise ValueError(f"raw DICOM path escapes configured directory: {path}") from error
        if not path.is_file():
            missing.append(raw_name)
            continue
        if processed_name not in base_paths:
            raise ValueError(f"manifest references an unknown processed image: {processed_name}")
        dataset = pydicom.dcmread(path)
        pixels = np.asarray(dataset.pixel_array)
        modality = str(getattr(dataset, "Modality", "")).upper()
        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
        modality_counts[modality] += 1
        photometric_counts[photometric] += 1
        native_voi_count += int(_native_voi_present(dataset))
        modality_transform_count += int(_modality_transform_present(dataset))
        if modality not in config.metadata_contract.allowed_modalities:
            raise ValueError(f"unexpected DICOM modality {modality!r} in {raw_name}")
        if photometric not in config.metadata_contract.allowed_photometric_interpretations:
            raise ValueError(f"unexpected Photometric Interpretation {photometric!r}")
        expected_metadata = {
            "BitsAllocated": config.metadata_contract.bits_allocated,
            "BitsStored": config.metadata_contract.bits_stored,
            "PixelRepresentation": config.metadata_contract.pixel_representation,
        }
        for attribute, expected in expected_metadata.items():
            if int(getattr(dataset, attribute)) != expected:
                raise ValueError(f"unexpected {attribute} in {raw_name}")
        if pixels.ndim != 2:
            raise ValueError(f"expected a 2-D raw radiograph in {raw_name}")
        raw_minima.append(float(pixels.min()))
        raw_maxima.append(float(pixels.max()))
        clean = scale_radiograph_to_uint8(
            pixels,
            photometric_interpretation=photometric,
            invert_monochrome1=bool(dataset_contract["invert_monochrome1"]),
        )
        with Image.open(base_paths[processed_name]) as processed:
            processed_pixels = np.asarray(processed.convert("L"))
        clean_pixel_mismatch_count += int(not np.array_equal(clean, processed_pixels))
        digest = sha256_file(path)
        raw_manifest.update(raw_name.encode())
        raw_manifest.update(b"\0")
        raw_manifest.update(digest.encode("ascii"))
        raw_manifest.update(b"\n")
        raw_paths[processed_name] = path

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} raw DICOM files; first entries: {preview}")
    if config.metadata_contract.require_native_voi_absent and native_voi_count:
        raise ValueError("native VOI metadata is present despite the frozen absence contract")
    if config.metadata_contract.require_modality_transform_absent and modality_transform_count:
        raise ValueError("modality transforms are present despite the frozen absence contract")
    if clean_pixel_mismatch_count:
        raise ValueError(
            f"raw clean conversion differs from {clean_pixel_mismatch_count} canonical PNGs"
        )
    return raw_paths, {
        "raw_dicom_count": len(raw_paths),
        "missing_raw_dicom_count": 0,
        "raw_files_sha256": raw_manifest.hexdigest(),
        "modality_counts": dict(sorted(modality_counts.items())),
        "photometric_interpretation_counts": dict(sorted(photometric_counts.items())),
        "native_voi_metadata_count": native_voi_count,
        "modality_transform_count": modality_transform_count,
        "bits_allocated": config.metadata_contract.bits_allocated,
        "bits_stored": config.metadata_contract.bits_stored,
        "pixel_representation": config.metadata_contract.pixel_representation,
        "raw_value_minimum": min(raw_minima),
        "raw_value_maximum": max(raw_maxima),
        "clean_png_pixel_mismatch_count": clean_pixel_mismatch_count,
        "clean_png_pixel_identity": "all raw reconversions are byte-for-byte pixel identical",
    }


def _prepare_analysis(config: AcquisitionShiftConfig) -> PreparedAnalysis:
    phase6 = load_robustness_config(config.resolve(config.inputs.phase6_config))
    if phase6.seed != config.seed:
        raise ValueError("acquisition-shift seed differs from the frozen Phase 6 seed")
    expected_manifest = phase6.resolve(phase6.sampling.output_manifest)
    configured_manifest = config.resolve(config.inputs.subset_manifest)
    if configured_manifest != expected_manifest:
        raise ValueError("acquisition shifts must use the configured Phase 6 robustness manifest")
    actual_manifest_hash = sha256_file(configured_manifest)
    if actual_manifest_hash != config.inputs.expected_subset_manifest_sha256:
        raise ValueError("frozen robustness manifest SHA-256 changed")
    phase6_summary = json.loads(
        phase6.resolve(phase6.outputs.summary_json).read_text(encoding="utf-8")
    )
    if phase6_summary.get("status") != "complete" or (
        phase6_summary.get("sampling", {}).get("output_manifest_sha256") != actual_manifest_hash
    ):
        raise ValueError("completed Phase 6 summary does not bind the frozen sample manifest")

    phase6_prepared = _prepare_experiment(phase6)
    dataset_contract = _dataset_contract(config)
    faster_config = phase6_prepared["training_configs"][("faster_rcnn", config.seed)]
    if config.resolve(config.inputs.dataset_config) != faster_config.resolve(
        faster_config.data.dataset_config
    ):
        raise ValueError("acquisition dataset config differs from the primary training config")
    rows = _read_selected_rows(config)
    if len(rows) != phase6.sampling.size:
        raise ValueError("raw acquisition manifest does not contain the frozen sample size")
    processed_names = {row[config.inputs.processed_id_column] for row in rows}
    if processed_names != phase6_prepared["selected_names"]:
        raise ValueError("raw acquisition rows differ from the frozen Phase 6 selection")
    raw_paths, raw_audit = _audit_raw_files(
        config,
        rows,
        dataset_contract=dataset_contract,
        base_dataset=phase6_prepared["base_dataset"],
    )
    return PreparedAnalysis(
        phase6=phase6,
        phase6_prepared=phase6_prepared,
        raw_paths=raw_paths,
        selected_rows=tuple(rows),
        raw_audit=raw_audit,
        invert_monochrome1=bool(dataset_contract["invert_monochrome1"]),
    )


def _condition_payload(shift: RadiographyShift) -> dict[str, Any]:
    return {
        "shift_id": shift.id,
        "shift_family": shift.family,
        "shift_kind": shift.kind,
        "center_offset_fraction": shift.center_offset_fraction,
        "width_multiplier": shift.width_multiplier,
        "reference_full_scale_counts": shift.reference_full_scale_counts,
        "dose_fraction": shift.dose_fraction,
        "expected_full_scale_counts": (
            None
            if shift.reference_full_scale_counts is None
            else shift.reference_full_scale_counts * float(shift.dose_fraction)
        ),
        "gaussian_kernel_size": shift.gaussian_kernel_size,
        "gaussian_sigma_pixels": shift.gaussian_sigma_pixels,
    }


def _source_identity(config: AcquisitionShiftConfig) -> dict[str, str]:
    paths = (
        config.source_path,
        config.project_root / "src" / "robustness" / "radiography_shifts.py",
        config.project_root / "src" / "robustness" / "run_robustness.py",
        config.project_root / "src" / "data" / "prepare.py",
        config.project_root / "src" / "evaluate.py",
    )
    return {path.relative_to(config.project_root).as_posix(): sha256_file(path) for path in paths}


def _bundle_identity(
    config: AcquisitionShiftConfig,
    detector: DetectorSettings,
    shift: RadiographyShift,
    *,
    prepared: PreparedAnalysis,
    sample_image_count: int,
    scope: Literal["smoke", "full"],
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "scope": scope,
        "detector": detector.detector,
        "seed": config.seed,
        "condition": _condition_payload(shift),
        "sample_manifest_sha256": config.inputs.expected_subset_manifest_sha256,
        "sample_image_count": sample_image_count,
        "raw_files_sha256": prepared.raw_audit["raw_files_sha256"],
        "checkpoint_sha256": sha256_file(prepared.phase6.resolve(detector.checkpoint)),
        "evaluation": prepared.phase6.evaluation.model_dump(mode="json"),
        "source_identity": dict(source_identity),
    }


def _bundle_path(
    config: AcquisitionShiftConfig,
    detector: DetectorSettings,
    shift: RadiographyShift,
    *,
    scope: Literal["smoke", "full"],
) -> Path:
    directory = (
        config.outputs.smoke_prediction_bundles_dir
        if scope == "smoke"
        else config.outputs.prediction_bundles_dir
    )
    return config.resolve(directory) / f"{detector.detector}__{shift.id}.json.gz"


def _repository_path(path: Path, project_root: Path) -> str:
    """Return a clone-stable repository-relative artifact path."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact path is outside the project root: {path}") from error


def _result_row(
    detector: DetectorSettings,
    shift: RadiographyShift,
    *,
    shifted_metrics: Mapping[str, Any],
    clean_metrics: Mapping[str, Any],
    inference_seconds: float,
    prediction_bundle: Path,
    project_root: Path,
) -> dict[str, Any]:
    condition = _condition_payload(shift)
    clean_values = _metric_values(clean_metrics)
    shifted_values = _metric_values(shifted_metrics)
    overall = shifted_metrics["operating_point"]["overall"]
    row: dict[str, Any] = {
        "detector": detector.detector,
        **condition,
        "image_count": shifted_metrics["coco"]["image_count"],
        "target_count": overall["target_count"],
        "true_positives": overall["tp"],
        "false_positives": overall["fp"],
        "false_negatives": overall["fn"],
        "inference_seconds": inference_seconds,
        "prediction_bundle": _repository_path(prediction_bundle, project_root),
        "prediction_bundle_sha256": sha256_file(prediction_bundle),
    }
    for metric in METRIC_FIELDS:
        clean = clean_values[metric]
        shifted = shifted_values[metric]
        ratio = None if clean in {None, 0} or shifted is None else shifted / clean
        row[f"clean_{metric}"] = clean
        row[f"shifted_{metric}"] = shifted
        row[f"{metric}_ratio"] = ratio
        row[f"{metric}_dsi"] = None if ratio is None else 1.0 - ratio
    return row


def _write_results(
    config: AcquisitionShiftConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    prepared: PreparedAnalysis,
    scope: Literal["smoke", "full"],
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    parameter_fields = (
        "center_offset_fraction",
        "width_multiplier",
        "reference_full_scale_counts",
        "dose_fraction",
        "expected_full_scale_counts",
        "gaussian_kernel_size",
        "gaussian_sigma_pixels",
    )
    base_fields = (
        "detector",
        "shift_id",
        "shift_family",
        "shift_kind",
        *parameter_fields,
        "image_count",
        "target_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "inference_seconds",
        "prediction_bundle",
        "prediction_bundle_sha256",
    )
    metric_fields = tuple(
        field
        for metric in METRIC_FIELDS
        for field in (
            f"clean_{metric}",
            f"shifted_{metric}",
            f"{metric}_ratio",
            f"{metric}_dsi",
        )
    )
    table_path = config.resolve(
        config.outputs.smoke_results_table if scope == "smoke" else config.outputs.results_table
    )
    _atomic_csv(table_path, (*base_fields, *metric_fields), rows)
    artifacts = {
        "results_table": _repository_path(table_path, config.project_root),
        "results_table_sha256": sha256_file(table_path),
    }
    summary = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "complete",
        "scope": scope,
        "performs_training": False,
        "seed_scope": "primary seed-17 checkpoints only",
        "sample_manifest": _repository_path(
            config.resolve(config.inputs.subset_manifest), config.project_root
        ),
        "sample_manifest_sha256": config.inputs.expected_subset_manifest_sha256,
        "raw_input_audit": dict(prepared.raw_audit),
        "clean_reference": (
            "Filtered frozen Phase 5 seed-17 predictions; the raw-array clean "
            "reconversion is pixel-identical to every canonical sample PNG."
        ),
        "dsi_definition": "1 - (shifted performance / clean performance); not clipped",
        "primary_performance_metric": "map_50_95",
        "evaluation": prepared.phase6.evaluation.model_dump(mode="json"),
        "config_path": _repository_path(config.source_path, config.project_root),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": dict(source_identity),
        "conditions": [_condition_payload(shift) for shift in config.shifts],
        "results": list(rows),
        "artifacts": artifacts,
    }
    summary_path = config.resolve(
        config.outputs.smoke_summary_json if scope == "smoke" else config.outputs.summary_json
    )
    _atomic_json(summary_path, summary)
    artifacts["summary_json"] = _repository_path(summary_path, config.project_root)
    artifacts["summary_json_sha256"] = sha256_file(summary_path)
    return {"summary": summary, "artifacts": artifacts}


def preflight(config: AcquisitionShiftConfig) -> dict[str, Any]:
    """Validate raw files, upstream identities, and clean pixel equivalence without CUDA."""

    prepared = _prepare_analysis(config)
    return {
        "status": "ready",
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "sample_image_count": len(prepared.selected_rows),
        "detector_count": len(prepared.phase6.detectors),
        "shift_condition_count": len(config.shifts),
        "shifted_inference_count": (
            len(prepared.selected_rows) * len(prepared.phase6.detectors) * len(config.shifts)
        ),
        "raw_input_audit": dict(prepared.raw_audit),
        "clean_reference": "frozen Phase 5 seed-17 sample predictions",
        "performs_training": False,
    }


def run_acquisition_shifts(
    config: AcquisitionShiftConfig, *, scope: Literal["smoke", "full"]
) -> dict[str, Any]:
    """Run or resume the smoke/full raw-radiograph shift grid on both detectors."""

    prepared = _prepare_analysis(config)
    from src.utils.seed import initialize_reproducibility, seed_everything

    log_dir = config.resolve(
        config.outputs.smoke_log_dir if scope == "smoke" else config.outputs.log_dir
    )
    initialize_reproducibility(config.seed, log_dir)
    source_identity = _source_identity(config)
    ordered_names = [row[config.inputs.processed_id_column] for row in prepared.selected_rows]
    if scope == "smoke":
        ordered_names = ordered_names[: config.runtime.smoke_images]
    selected_names = set(ordered_names)
    raw_paths = {name: prepared.raw_paths[name] for name in selected_names}
    template_dataset = RadiographyShiftSubsetDataset(
        prepared.phase6_prepared["base_dataset"],
        selected_names,
        raw_paths=raw_paths,
        shift=config.shifts[0],
        seed=config.seed,
        invert_monochrome1=prepared.invert_monochrome1,
    )
    targets = _targets_from_dataset(template_dataset)
    category_names = template_dataset.category_names
    rows: list[dict[str, Any]] = []

    for detector in prepared.phase6.detectors:
        clean_predictions, _clean_provenance = _clean_predictions(
            detector, prepared.phase6, selected_names
        )
        clean_metrics = evaluate_prediction_records(
            clean_predictions,
            targets,
            category_names=category_names,
            settings=prepared.phase6.evaluation,
        )
        model_config = prepared.phase6_prepared["training_configs"][
            (detector.detector, config.seed)
        ]
        seed_everything(config.seed)
        if detector.detector == "faster_rcnn":
            model, device = _load_faster_model(
                detector, prepared.phase6, model_config, template_dataset
            )
        else:
            import torch
            from ultralytics import YOLO

            model = YOLO(prepared.phase6.resolve(detector.checkpoint).as_posix())
            device = torch.device(f"cuda:{model_config.runtime.device}")

        for index, shift in enumerate(config.shifts, start=1):
            print(
                f"[{scope} {detector.detector} {index}/{len(config.shifts)}] {shift.id}",
                flush=True,
            )
            identity = _bundle_identity(
                config,
                detector,
                shift,
                prepared=prepared,
                sample_image_count=len(selected_names),
                scope=scope,
                source_identity=source_identity,
            )
            bundle_path = _bundle_path(config, detector, shift, scope=scope)
            cached = _cached_result(bundle_path, identity)
            if cached is None:
                seed_everything(config.seed)
                dataset = RadiographyShiftSubsetDataset(
                    prepared.phase6_prepared["base_dataset"],
                    selected_names,
                    raw_paths=raw_paths,
                    shift=shift,
                    seed=config.seed,
                    invert_monochrome1=prepared.invert_monochrome1,
                )
                if detector.detector == "faster_rcnn":
                    predictions, inference_seconds = _collect_faster_predictions(
                        model, device, model_config, prepared.phase6, dataset
                    )
                else:
                    predictions, inference_seconds = _collect_yolo_predictions(
                        model, device, model_config, prepared.phase6, dataset
                    )
                shifted_metrics = evaluate_prediction_records(
                    predictions,
                    targets,
                    category_names=category_names,
                    settings=prepared.phase6.evaluation,
                )
                _write_prediction_bundle(
                    bundle_path,
                    identity=identity,
                    predictions=predictions,
                    metrics=shifted_metrics,
                    inference_seconds=inference_seconds,
                )
                cached = _read_gzip_json(bundle_path)
                del predictions, shifted_metrics, dataset
                gc.collect()
            rows.append(
                _result_row(
                    detector,
                    shift,
                    shifted_metrics=cached["metrics"],
                    clean_metrics=clean_metrics,
                    inference_seconds=float(cached["inference_seconds"]),
                    prediction_bundle=bundle_path,
                    project_root=config.project_root,
                )
            )

        del model
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    detector_order = {"faster_rcnn": 0, "yolo11s": 1}
    shift_order = {shift.id: index for index, shift in enumerate(config.shifts)}
    rows.sort(
        key=lambda row: (
            detector_order[str(row["detector"])],
            shift_order[str(row["shift_id"])],
        )
    )
    result = _write_results(
        config,
        rows,
        prepared=prepared,
        scope=scope,
        source_identity=source_identity,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "scope": scope,
                "row_count": len(rows),
                "results_table": result["artifacts"]["results_table"],
                "summary": result["artifacts"]["summary_json"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/acquisition_shifts.yaml"))
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight, smoke inference, or the complete acquisition-shift grid."""

    args = build_parser().parse_args(argv)
    config = load_acquisition_shift_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return 0
    run_acquisition_shifts(config, scope="smoke" if args.mode == "smoke" else "full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
