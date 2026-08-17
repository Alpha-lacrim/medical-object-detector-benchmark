"""Strict Phase 7 explainability experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DetectorName = Literal["faster_rcnn", "yolo11s"]


class StrictModel(BaseModel):
    """Reject undeclared keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GradCamSettings(StrictModel):
    """Differentiable target, candidate filtering, and CAM settings."""

    method: Literal["grad_cam"]
    target_score: Literal["post_activation_foreground_probability"]
    target_selection: Literal["highest_iou_then_score_low_threshold_retained_candidate"]
    candidate_score_floor: float = Field(ge=0, le=1)
    nms_iou_threshold: float = Field(gt=0, le=1)
    max_candidates: int = Field(gt=0)
    yolo_max_nms_candidates: int = Field(gt=0)
    yolo_nms_time_limit_seconds: float = Field(gt=0)
    interpolation_mode: Literal["bilinear"]
    align_corners: Literal[False]
    epsilon: float = Field(gt=0)
    zero_energy_epsilon: float = Field(gt=0)
    zero_energy_policy: Literal["exclude_and_report"]


class MetricSettings(StrictModel):
    """Quantitative explainability population and estimands."""

    population: Literal["every_ground_truth_box_in_phase6_sample"]
    unit: Literal["ground_truth_box"]
    box_rasterization: Literal["pixel_center"]
    primary: Literal["energy_in_box"]
    secondary: Literal["pointing_game"]
    random_baseline: Literal["rasterized_box_area_fraction"]


class RuntimeSettings(StrictModel):
    """Single-GPU execution and bounded smoke settings."""

    device: Literal["cuda"]
    smoke_positive_images: int = Field(gt=0)
    progress_every_images: int = Field(gt=0)


class DetectorLayerSettings(StrictModel):
    """One semantically documented stride-matched backbone hook."""

    detector: DetectorName
    module_path: str = Field(min_length=1)
    expected_stride: int = Field(gt=0)
    expected_spatial_size: int = Field(gt=0)
    stage_description: str = Field(min_length=1)


class QualitativeSettings(StrictModel):
    """Deterministic case-selection and plotting contract."""

    cases_per_category: int = Field(ge=1, le=5)
    good_selection: Literal["highest_minimum_iou_shared_true_positive_unique_image"]
    bad_selection: Literal["highest_minimum_score_shared_false_positive_negative_image"]
    failure_selection: Literal["shared_false_negative_proxy_iou_quantiles_unique_image"]
    failure_quantiles: tuple[float, ...]
    overlay_alpha: float = Field(gt=0, le=1)
    colormap: str = Field(min_length=1)
    ground_truth_color: str = Field(min_length=1)
    candidate_color: str = Field(min_length=1)
    line_width: float = Field(gt=0)
    panel_width_inches: float = Field(gt=0)
    row_height_inches: float = Field(gt=0)
    dpi: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_failure_quantiles(self) -> QualitativeSettings:
        if len(self.failure_quantiles) != self.cases_per_category:
            raise ValueError("failure_quantiles must match cases_per_category")
        if any(not 0 <= value <= 1 for value in self.failure_quantiles):
            raise ValueError("failure_quantiles must be within [0, 1]")
        if tuple(sorted(set(self.failure_quantiles))) != self.failure_quantiles:
            raise ValueError("failure_quantiles must be unique and increasing")
        return self


class OutputSettings(StrictModel):
    """All Phase 7 generated artifacts."""

    log_dir: Path
    smoke_summary: Path
    summary_json: Path
    target_table: Path
    aggregate_table: Path
    qualitative_manifest: Path
    good_figure: Path
    bad_figure: Path
    failure_figure: Path


class ExplainabilityConfig(StrictModel):
    """Complete immutable Phase 7 contract."""

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    phase6_config: Path
    phase6_summary: Path
    gradcam: GradCamSettings
    metric: MetricSettings
    runtime: RuntimeSettings
    detectors: tuple[DetectorLayerSettings, DetectorLayerSettings]
    qualitative: QualitativeSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_detector_pair(self) -> ExplainabilityConfig:
        names = [item.detector for item in self.detectors]
        if set(names) != {"faster_rcnn", "yolo11s"} or len(set(names)) != 2:
            raise ValueError("detectors must contain Faster R-CNN and YOLO11s exactly once")
        strides = {item.expected_stride for item in self.detectors}
        spatial_sizes = {item.expected_spatial_size for item in self.detectors}
        if len(strides) != 1 or len(spatial_sizes) != 1:
            raise ValueError("detector target layers must have matching stride and resolution")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()

    def layer(self, detector: DetectorName) -> DetectorLayerSettings:
        """Return the configured layer for one detector."""

        return next(item for item in self.detectors if item.detector == detector)


def load_explainability_config(path: str | Path) -> ExplainabilityConfig:
    """Load strict Phase 7 YAML without importing either detector framework."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("explainability config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return ExplainabilityConfig.model_validate(payload)
