"""Build the reviewed scientific artifact manifest from a fixed inventory.

This is a maintenance command, not a CI auto-update. Run it only after reviewing
intentional scientific artifact changes, then inspect the manifest diff before
commit. CI executes ``verify_scientific_artifacts.py`` and never calls this file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/scientific_artifact_manifest.json"


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InputSpec:
    path: str
    availability: str = "committed"
    sha256: str | None = None


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    generator: str
    config: str
    inputs: tuple[InputSpec, ...]
    phase: str
    reproduction_tier: str
    gpu_required: bool
    training_required: bool
    referenced_results: tuple[str, ...] = ()


def committed(path: str) -> InputSpec:
    return InputSpec(path)


def external(path: str, sha256: str) -> InputSpec:
    return InputSpec(path, "external_or_ignored", sha256)


PHASE5_BUNDLES = tuple(
    f"results/logs/phase5_evaluation/predictions/{detector}_seed{seed}_test_predictions.json.gz"
    for detector in ("faster_rcnn", "yolo11s")
    for seed in (17, 42, 137, 271, 314)
)
VALIDATION_BUNDLES = tuple(
    "results/logs/phase14_threshold_selection/validation_predictions/"
    f"{detector}_seed{seed}_validation_predictions.json.gz"
    for detector in ("faster_rcnn", "yolo11s")
    for seed in (17, 42, 137)
)
PHASE6_BUNDLES = tuple(
    path.relative_to(ROOT).as_posix()
    for path in sorted((ROOT / "results/logs/phase6_robustness/predictions").glob("*.json.gz"))
)
PHASE22_BUNDLES = tuple(
    path.relative_to(ROOT).as_posix()
    for path in sorted(
        (ROOT / "results/logs/phase22_acquisition_shifts/predictions").glob("*.json.gz")
    )
)

RAW_DATA_INPUTS = (
    external(
        "data/raw/rsna-pneumonia/stage_2_train_labels.csv",
        "bb40b7e956e9922a6b275ed4a158197568cf9ab618017d53db6159b1b624bb65",
    ),
    external(
        "data/raw/rsna-pneumonia/stage_2_detailed_class_info.csv",
        "c004c12dea2042cc23e3b848f65e8cb2e725799afaa90f13ee81f854bcc9614d",
    ),
    external(
        "data/raw/rsna-pneumonia/mappings.json",
        "803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27",
    ),
)

PHASE5_INPUTS = (
    committed("results/checkpoint_release_manifest.json"),
    committed("data/splits/rsna-pneumonia-5000/test.csv"),
    external(
        "data/processed/rsna-pneumonia-5000/annotations/instances_test.json",
        "86bbe6c238d651bfc0b017a447a67548aa631f95855d66b9b8b71a13807502cc",
    ),
)


def spec(
    path: str,
    generator: str,
    config: str,
    inputs: tuple[InputSpec, ...],
    phase: str,
    reproduction_tier: str,
    *,
    gpu: bool = False,
    training: bool = False,
    references: tuple[str, ...] = (),
) -> ArtifactSpec:
    return ArtifactSpec(
        path,
        generator,
        config,
        inputs,
        phase,
        reproduction_tier,
        gpu,
        training,
        references,
    )


SPECS = (
    spec(
        "data/manifests/rsna-pneumonia-5000-audit.json",
        "src/data/prepare.py",
        "configs/dataset.yaml",
        RAW_DATA_INPUTS,
        "Phase 2 - dataset preparation",
        "external_data_preparation",
    ),
    spec(
        "results/logs/phase5_evaluation/summary.json",
        "src/evaluate.py",
        "configs/evaluation.yaml",
        PHASE5_INPUTS,
        "Phase 5 - unified clean evaluation",
        "exact_inference",
        gpu=True,
        references=(
            *PHASE5_BUNDLES,
            "results/tables/detector_comparison.csv",
            "results/tables/detector_comparison_mean_std.csv",
            "results/tables/detector_comparison_per_seed.csv",
        ),
    ),
    spec(
        "results/tables/detector_comparison.csv",
        "src/evaluate.py",
        "configs/evaluation.yaml",
        (committed("results/logs/phase5_evaluation/summary.json"),),
        "Phase 5 - unified clean evaluation",
        "exact_inference",
        gpu=True,
    ),
    spec(
        "results/tables/detector_comparison_per_seed.csv",
        "src/evaluate.py",
        "configs/evaluation.yaml",
        (committed("results/logs/phase5_evaluation/summary.json"),),
        "Phase 5 - unified clean evaluation",
        "exact_inference",
        gpu=True,
    ),
    spec(
        "results/logs/phase35_operating_regime_n5/threshold_summary.json",
        "src/evaluate_threshold_sweep.py",
        "configs/threshold_sweep_n5_sensitivity.yaml",
        (
            committed("results/logs/phase5_evaluation/summary.json"),
            *(committed(path) for path in PHASE5_BUNDLES),
        ),
        "Phase 35 - five-run operating-regime sensitivity",
        "committed_analysis",
        references=(
            "results/tables/threshold_sweep_n5_sensitivity.csv",
            "results/tables/threshold_sweep_per_seed_n5_sensitivity.csv",
            "results/tables/precision_recall_curves_n5_sensitivity.csv",
            "results/tables/precision_recall_curves_per_seed_n5_sensitivity.csv",
            "results/tables/threshold_operating_targets_n5_sensitivity.csv",
            "results/figures/precision_recall_curves_n5_sensitivity.png",
            "results/figures/f1_vs_threshold_n5_sensitivity.png",
        ),
    ),
    spec(
        "results/tables/threshold_sweep_n5_sensitivity.csv",
        "src/evaluate_threshold_sweep.py",
        "configs/threshold_sweep_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/threshold_summary.json"),),
        "Phase 35 - five-run operating-regime sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/tables/precision_recall_curves_n5_sensitivity.csv",
        "src/evaluate_threshold_sweep.py",
        "configs/threshold_sweep_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/threshold_summary.json"),),
        "Phase 35 - five-run operating-regime sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/figures/precision_recall_curves_n5_sensitivity.png",
        "src/evaluate_threshold_sweep.py",
        "configs/threshold_sweep_n5_sensitivity.yaml",
        (committed("results/tables/precision_recall_curves_n5_sensitivity.csv"),),
        "Phase 35 - five-run operating-regime sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/figures/f1_vs_threshold_n5_sensitivity.png",
        "src/evaluate_threshold_sweep.py",
        "configs/threshold_sweep_n5_sensitivity.yaml",
        (committed("results/tables/threshold_sweep_n5_sensitivity.csv"),),
        "Phase 35 - five-run operating-regime sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase35_operating_regime_n5/froc_summary.json",
        "src/plot_froc_curves.py",
        "configs/froc_n5_sensitivity.yaml",
        (
            committed("results/logs/phase35_operating_regime_n5/threshold_summary.json"),
            committed("results/tables/threshold_sweep_per_seed_n5_sensitivity.csv"),
        ),
        "Phase 35 - five-run FROC sensitivity",
        "committed_analysis",
        references=(
            "results/tables/froc_curves_n5_sensitivity.csv",
            "results/tables/froc_curves_per_seed_n5_sensitivity.csv",
            "results/tables/froc_operating_points_n5_sensitivity.csv",
            "results/figures/froc_curves_n5_sensitivity.png",
        ),
    ),
    spec(
        "results/tables/froc_operating_points_n5_sensitivity.csv",
        "src/plot_froc_curves.py",
        "configs/froc_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/froc_summary.json"),),
        "Phase 35 - five-run FROC sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/figures/froc_curves_n5_sensitivity.png",
        "src/plot_froc_curves.py",
        "configs/froc_n5_sensitivity.yaml",
        (committed("results/tables/froc_curves_n5_sensitivity.csv"),),
        "Phase 35 - five-run FROC sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase35_operating_regime_n5/pareto_summary.json",
        "src/plot_pareto_frontier.py",
        "configs/pareto_n5_sensitivity.yaml",
        (
            committed("results/tables/detector_comparison_per_seed.csv"),
            committed("results/tables/selected_operating_points_per_seed_n5_sensitivity.csv"),
        ),
        "Phase 35 - five-run Pareto sensitivity",
        "committed_analysis",
        references=(
            "results/tables/pareto_points_n5_sensitivity.csv",
            "results/tables/pareto_summary_n5_sensitivity.csv",
            "results/figures/pareto_frontier_n5_sensitivity.png",
        ),
    ),
    spec(
        "results/tables/pareto_summary_n5_sensitivity.csv",
        "src/plot_pareto_frontier.py",
        "configs/pareto_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/pareto_summary.json"),),
        "Phase 35 - five-run Pareto sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/figures/pareto_frontier_n5_sensitivity.png",
        "src/plot_pareto_frontier.py",
        "configs/pareto_n5_sensitivity.yaml",
        (committed("results/tables/pareto_points_n5_sensitivity.csv"),),
        "Phase 35 - five-run Pareto sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase35_operating_regime_n5/summary.json",
        "src/analyze_operating_regime_sensitivity.py",
        "configs/operating_regime_n5_sensitivity.yaml",
        (
            committed("results/logs/phase35_operating_regime_n5/threshold_summary.json"),
            committed("results/logs/phase35_operating_regime_n5/froc_summary.json"),
            committed("results/logs/phase35_operating_regime_n5/pareto_summary.json"),
            committed("results/logs/phase14_threshold_selection/summary.json"),
        ),
        "Phase 35 - operating-regime conclusion audit",
        "committed_analysis",
        references=(
            "results/tables/selected_operating_points_n5_sensitivity.csv",
            "results/tables/selected_operating_points_per_seed_n5_sensitivity.csv",
            "results/tables/operating_regime_n5_inventory.csv",
            "results/tables/operating_regime_n3_vs_n5_conclusions.csv",
        ),
    ),
    spec(
        "results/tables/selected_operating_points_n5_sensitivity.csv",
        "src/analyze_operating_regime_sensitivity.py",
        "configs/operating_regime_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/summary.json"),),
        "Phase 35 - unchanged validation-threshold application",
        "committed_analysis",
    ),
    spec(
        "results/tables/operating_regime_n5_inventory.csv",
        "src/analyze_operating_regime_sensitivity.py",
        "configs/operating_regime_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/summary.json"),),
        "Phase 35 - operating-regime conclusion audit",
        "committed_analysis",
    ),
    spec(
        "results/tables/operating_regime_n3_vs_n5_conclusions.csv",
        "src/analyze_operating_regime_sensitivity.py",
        "configs/operating_regime_n5_sensitivity.yaml",
        (committed("results/logs/phase35_operating_regime_n5/summary.json"),),
        "Phase 35 - operating-regime conclusion audit",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase29_threshold_sensitivity/summary.json",
        "src/stats/threshold_calibration.py",
        "configs/threshold_calibration.yaml",
        (
            committed(
                "results/logs/phase14_threshold_selection/validation_prediction_manifest.json"
            ),
            *(committed(path) for path in VALIDATION_BUNDLES),
        ),
        "Phase 29 - validation threshold sensitivity",
        "committed_analysis",
        references=(
            "results/tables/recall_weighted_fbeta_threshold_summary.csv",
            "results/tables/recall_weighted_fbeta_threshold_stability.csv",
            "results/tables/hypothetical_detection_error_loss_summary.csv",
            "results/figures/recall_weighted_fbeta_threshold_sensitivity.png",
        ),
    ),
    spec(
        "results/tables/recall_weighted_fbeta_threshold_summary.csv",
        "src/stats/threshold_calibration.py",
        "configs/threshold_calibration.yaml",
        (committed("results/logs/phase29_threshold_sensitivity/summary.json"),),
        "Phase 29 - validation threshold sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/tables/recall_weighted_fbeta_threshold_stability.csv",
        "src/stats/threshold_calibration.py",
        "configs/threshold_calibration.yaml",
        (committed("results/logs/phase29_threshold_sensitivity/summary.json"),),
        "Phase 29 - validation threshold sensitivity",
        "committed_analysis",
    ),
    spec(
        "results/tables/hypothetical_detection_error_loss_summary.csv",
        "src/stats/threshold_calibration.py",
        "configs/threshold_calibration.yaml",
        (committed("results/logs/phase29_threshold_sensitivity/summary.json"),),
        "Phase 29 - hypothetical detection-error loss",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase33_calibration_support_v2/summary.json",
        "src/stats/calibration.py",
        "configs/calibration.yaml",
        (
            committed("results/logs/phase5_evaluation/summary.json"),
            *(committed(path) for path in PHASE5_BUNDLES),
        ),
        "Phase 33 - detection calibration support audit",
        "committed_analysis",
        references=(
            "results/tables/calibration_summary_v2.csv",
            "results/tables/calibration_support_v2.csv",
            "results/tables/calibration_sensitivity_v2.csv",
            "results/figures/reliability_diagrams_confidence_marginal_v2.png",
            "results/figures/calibration_support_occupancy_v2.png",
            "results/figures/calibration_binning_sensitivity_v2.png",
            "results/figures/calibration_confidence_floor_sensitivity_v2.png",
        ),
    ),
    spec(
        "results/tables/calibration_summary_v2.csv",
        "src/stats/calibration.py",
        "configs/calibration.yaml",
        (committed("results/logs/phase33_calibration_support_v2/summary.json"),),
        "Phase 33 - detection calibration support audit",
        "committed_analysis",
    ),
    spec(
        "results/tables/calibration_support_v2.csv",
        "src/stats/calibration.py",
        "configs/calibration.yaml",
        (committed("results/logs/phase33_calibration_support_v2/summary.json"),),
        "Phase 33 - detection calibration support audit",
        "committed_analysis",
    ),
    spec(
        "results/tables/calibration_sensitivity_v2.csv",
        "src/stats/calibration.py",
        "configs/calibration.yaml",
        (committed("results/logs/phase33_calibration_support_v2/summary.json"),),
        "Phase 33 - detection calibration support audit",
        "committed_analysis",
    ),
    spec(
        "results/figures/reliability_diagrams_confidence_marginal_v2.png",
        "src/stats/calibration.py",
        "configs/calibration.yaml",
        (committed("results/tables/calibration_summary_v2.csv"),),
        "Phase 33 - detection calibration support audit",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase23_reporting/raincloud_metrics_summary.json",
        "src/plot_raincloud_metrics.py",
        "configs/raincloud_metrics.yaml",
        (
            committed("results/tables/detector_comparison.csv"),
            committed("results/tables/detector_comparison_per_seed.csv"),
        ),
        "Phase 23 - seed-distribution visualization",
        "committed_analysis",
        references=("results/figures/raincloud_metrics.png",),
    ),
    spec(
        "results/figures/raincloud_metrics.png",
        "src/plot_raincloud_metrics.py",
        "configs/raincloud_metrics.yaml",
        (committed("results/logs/phase23_reporting/raincloud_metrics_summary.json"),),
        "Phase 23 - seed-distribution visualization",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase6_robustness/summary.json",
        "src/robustness/run_robustness.py",
        "configs/corruptions.yaml",
        (
            committed("results/checkpoint_release_manifest.json"),
            committed("data/splits/rsna-pneumonia-5000/test_robustness_seed17_n300.csv"),
            committed("results/logs/phase5_evaluation/summary.json"),
        ),
        "Phase 6 - common-corruption robustness",
        "checkpoint_inference",
        gpu=True,
        references=(
            *PHASE6_BUNDLES,
            "results/tables/robustness_results.csv",
            "results/tables/robustness_curves.csv",
            "results/tables/robustness_family_mean_curves.csv",
            "results/figures/robustness_map_50_95_raw.png",
            "results/figures/robustness_map_50_95_relative.png",
        ),
    ),
    spec(
        "results/tables/robustness_results.csv",
        "src/robustness/run_robustness.py",
        "configs/corruptions.yaml",
        (committed("results/logs/phase6_robustness/summary.json"),),
        "Phase 6 - common-corruption robustness",
        "checkpoint_inference",
        gpu=True,
    ),
    spec(
        "results/figures/robustness_map_50_95_relative.png",
        "src/robustness/run_robustness.py",
        "configs/corruptions.yaml",
        (committed("results/tables/robustness_curves.csv"),),
        "Phase 6 - common-corruption robustness",
        "checkpoint_inference",
        gpu=True,
    ),
    spec(
        "results/logs/phase32_acquisition_shift_audit/summary.json",
        "src/robustness/radiography_shifts.py",
        "configs/acquisition_shifts.yaml",
        (
            committed("results/logs/phase22_acquisition_shifts/summary.json"),
            committed("data/splits/rsna-pneumonia-5000/test_robustness_seed17_n300.csv"),
            *(committed(path) for path in PHASE22_BUNDLES),
        ),
        "Phase 32 - acquisition/display shift audit",
        "committed_analysis",
        references=(
            *PHASE22_BUNDLES,
            "results/tables/acquisition_shift_dicom_metadata_audit.csv",
            "results/tables/acquisition_shift_preprocessing_per_image.csv",
            "results/tables/acquisition_shift_preprocessing_summary.csv",
            "results/tables/radiography_synthetic_shift_results.csv",
        ),
    ),
    spec(
        "results/tables/acquisition_shift_dicom_metadata_audit.csv",
        "src/robustness/radiography_shifts.py",
        "configs/acquisition_shifts.yaml",
        (committed("results/logs/phase32_acquisition_shift_audit/summary.json"),),
        "Phase 32 - acquisition/display shift audit",
        "committed_analysis",
    ),
    spec(
        "results/tables/acquisition_shift_preprocessing_summary.csv",
        "src/robustness/radiography_shifts.py",
        "configs/acquisition_shifts.yaml",
        (committed("results/logs/phase32_acquisition_shift_audit/summary.json"),),
        "Phase 32 - acquisition/display shift audit",
        "committed_analysis",
    ),
    spec(
        "results/tables/radiography_synthetic_shift_results.csv",
        "src/robustness/radiography_shifts.py",
        "configs/acquisition_shifts.yaml",
        (committed("results/logs/phase32_acquisition_shift_audit/summary.json"),),
        "Phase 32 - acquisition/display shift audit",
        "committed_analysis",
    ),
    spec(
        "results/logs/phase7_explainability/summary.json",
        "src/explainability/run_explainability.py",
        "configs/explainability.yaml",
        (
            committed("results/logs/phase6_robustness/summary.json"),
            committed("results/checkpoint_release_manifest.json"),
        ),
        "Phase 7 - Grad-CAM localization",
        "checkpoint_inference",
        gpu=True,
        references=(
            "results/tables/gradcam_localization_summary.csv",
            "results/tables/gradcam_localization_per_target.csv",
            "results/tables/gradcam_qualitative_cases.csv",
            "results/figures/gradcam_good_predictions.png",
            "results/figures/gradcam_bad_predictions.png",
            "results/figures/gradcam_failure_cases.png",
        ),
    ),
    spec(
        "results/tables/gradcam_localization_summary.csv",
        "src/explainability/run_explainability.py",
        "configs/explainability.yaml",
        (committed("results/logs/phase7_explainability/summary.json"),),
        "Phase 7 - Grad-CAM localization",
        "checkpoint_inference",
        gpu=True,
    ),
    spec(
        "results/logs/phase31_xai_sanity_v2/summary.json",
        "src/explainability/sanity_checks.py",
        "configs/xai_sanity.yaml",
        (
            committed("results/logs/phase7_explainability/summary.json"),
            committed("results/checkpoint_release_manifest.json"),
            committed("data/splits/rsna-pneumonia-5000/test_robustness_seed17_n300.csv"),
        ),
        "Phase 31 - Grad-CAM sensitivity controls",
        "checkpoint_inference",
        gpu=True,
        references=(
            "results/tables/gradcam_sanity_v2_summary.csv",
            "results/tables/gradcam_sanity_v2_per_image.csv",
            "results/figures/gradcam_sanity_v2_panel.png",
            "results/logs/phase31_xai_sanity_v2/subset_manifest.csv",
        ),
    ),
    spec(
        "results/tables/gradcam_sanity_v2_summary.csv",
        "src/explainability/sanity_checks.py",
        "configs/xai_sanity.yaml",
        (committed("results/logs/phase31_xai_sanity_v2/summary.json"),),
        "Phase 31 - Grad-CAM sensitivity controls",
        "checkpoint_inference",
        gpu=True,
    ),
    spec(
        "results/figures/gradcam_sanity_v2_panel.png",
        "src/explainability/sanity_checks.py",
        "configs/xai_sanity.yaml",
        (committed("results/tables/gradcam_sanity_v2_summary.csv"),),
        "Phase 31 - Grad-CAM sensitivity controls",
        "checkpoint_inference",
        gpu=True,
    ),
    spec(
        "results/logs/phase8_statistics/summary.json",
        "src/stats/run_statistics.py",
        "configs/statistics.yaml",
        (
            committed("results/logs/phase5_evaluation/summary.json"),
            committed("results/logs/phase6_robustness/summary.json"),
            committed("data/splits/rsna-pneumonia-5000/test.csv"),
        ),
        "Phase 8/28 - estimand-separated statistics",
        "committed_analysis",
        references=(
            "results/tables/statistical_clean_comparison.csv",
            "results/tables/statistical_clean_per_run_metrics.csv",
            "results/tables/statistical_clean_leave_one_run_out.csv",
            "results/tables/statistical_clean_leave_one_seed_label_out.csv",
            "results/tables/statistical_robustness_comparison.csv",
        ),
    ),
    spec(
        "results/tables/statistical_clean_comparison.csv",
        "src/stats/run_statistics.py",
        "configs/statistics.yaml",
        (committed("results/logs/phase8_statistics/summary.json"),),
        "Phase 8/28 - estimand-separated statistics",
        "committed_analysis",
    ),
    spec(
        "results/tables/statistical_robustness_comparison.csv",
        "src/stats/run_statistics.py",
        "configs/statistics.yaml",
        (committed("results/logs/phase8_statistics/summary.json"),),
        "Phase 8/28 - estimand-separated statistics",
        "committed_analysis",
    ),
    spec(
        "results/tables/statistical_clean_leave_one_seed_label_out.csv",
        "src/stats/run_statistics.py",
        "configs/statistics.yaml",
        (committed("results/logs/phase8_statistics/summary.json"),),
        "Phase 28 - seed-influence diagnostic",
        "committed_analysis",
    ),
)


def _schema(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise ValueError(f"empty CSV cannot enter manifest: {path}")
        return {
            "type": "csv",
            "required_columns": rows[0],
            "exact_columns": rows[0],
            "row_count": len(rows) - 1,
        }
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {
                "type": "json",
                "top_level_type": "object",
                "required_top_level_keys": sorted(payload),
            }
        if isinstance(payload, list):
            return {
                "type": "json",
                "top_level_type": "array",
                "required_top_level_keys": [],
            }
        raise ValueError(f"unsupported JSON top level: {path}")
    if suffix == ".png":
        with path.open("rb") as handle:
            prefix = handle.read(24)
        if len(prefix) != 24 or prefix[12:16] != b"IHDR":
            raise ValueError(f"invalid PNG: {path}")
        width, height = struct.unpack(">II", prefix[16:24])
        return {"type": "png", "width": width, "height": height}
    raise ValueError(f"unsupported artifact type: {path}")


def _input_payload(input_spec: InputSpec) -> dict[str, str]:
    path = ROOT / input_spec.path
    if path.is_file():
        digest = sha256_file(path)
        if input_spec.sha256 is not None and digest != input_spec.sha256:
            raise ValueError(
                f"known input hash mismatch for {input_spec.path}: "
                f"expected {input_spec.sha256}, got {digest}"
            )
    elif input_spec.sha256 is not None:
        digest = input_spec.sha256
    else:
        raise FileNotFoundError(f"manifest input is missing: {input_spec.path}")
    return {
        "path": input_spec.path,
        "sha256": digest,
        "availability": input_spec.availability,
    }


def build_manifest() -> dict[str, Any]:
    """Return the manifest payload after hashing the fixed reviewed inventory."""

    artifacts: list[dict[str, Any]] = []
    for item in SPECS:
        path = ROOT / item.path
        generator = ROOT / item.generator
        config = ROOT / item.config
        if not path.is_file() or not generator.is_file() or not config.is_file():
            raise FileNotFoundError(f"inventory path is missing for {item.path}")
        artifacts.append(
            {
                "path": item.path,
                "sha256": sha256_file(path),
                "generating_script": item.generator,
                "generating_script_sha256": sha256_file(generator),
                "config": {"path": item.config, "sha256": sha256_file(config)},
                "input_artifacts": [_input_payload(value) for value in item.inputs],
                "expected_schema": _schema(path),
                "study_phase": item.phase,
                "reproduction_tier": item.reproduction_tier,
                "gpu_required": item.gpu_required,
                "training_required": item.training_required,
                "referenced_results": list(item.referenced_results),
            }
        )
    paths = [artifact["path"] for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("fixed scientific artifact inventory contains duplicate paths")
    return {
        "schema_version": 1,
        "manifest_purpose": (
            "Hash and schema gate for the committed artifacts that directly support the current "
            "manuscript; this does not claim that CI reruns inference or training."
        ),
        "canonical_manuscript": "report/paper_draft.md",
        "generated_by": "scripts/build_scientific_artifact_manifest.py",
        "verification_command": "python scripts/verify_scientific_artifacts.py",
        "manuscript_critical_paths": paths,
        "artifacts": artifacts,
    }


def main() -> int:
    """Write the reviewed inventory with deterministic LF JSON formatting."""

    payload = build_manifest()
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(payload['artifacts'])} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
