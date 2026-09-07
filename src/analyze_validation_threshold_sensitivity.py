"""Run the post-hoc five-run validation threshold-selection sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.collect_froc_lower_floor_predictions import (
    _validate_release_manifest,
    _validate_source_summary,
)
from src.evaluate import (
    _collect_faster_rcnn_predictions,
    _collect_yolo_predictions,
    _targets_from_dataset,
    evaluate_prediction_records,
    load_and_validate_training_configs,
    load_phase5_config,
    sha256_file,
)
from src.evaluate_threshold_selection import (
    VALIDATION_FIELDS,
    VALIDATION_PER_SEED_FIELDS,
    _atomic_csv,
    _atomic_json,
    _frozen_faster_config_sha256,
    _load_validation_dataset,
    _validation_reference_table,
    _write_validation_bundle,
    select_validation_thresholds,
)
from src.evaluate_threshold_sweep import (
    _deserialize_predictions,
    _read_bundle,
    _validate_upstream,
    aggregate_threshold_rows,
    load_threshold_sweep_config,
    sweep_prediction_records,
)

TEST_PER_SEED_FIELDS = (
    "selection_scope",
    "detector",
    "seed",
    "selection_split",
    "selection_rule",
    "tie_breaker",
    "selected_threshold",
    "threshold_selection_run_count",
    "test_run_count",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_true_positives",
    "test_false_positives",
    "test_false_negatives",
    "test_fp_per_image",
    "test_prediction_count",
    "test_target_count",
)
TEST_AGGREGATE_FIELDS = (
    "selection_scope",
    "sensitivity_label",
    "detector",
    "selection_split",
    "selection_rule",
    "tie_breaker",
    "selected_threshold",
    "threshold_selection_run_count",
    "test_run_count",
    "validation_precision",
    "validation_precision_std",
    "validation_recall",
    "validation_recall_std",
    "validation_f1",
    "validation_f1_std",
    "test_precision",
    "test_precision_std",
    "test_recall",
    "test_recall_std",
    "test_f1",
    "test_f1_std",
    "test_fp_per_image",
    "test_fp_per_image_std",
    "test_prediction_count",
    "test_prediction_count_std",
)
CONCLUSION_FIELDS = (
    "analysis",
    "conclusion",
    "historical_value",
    "posthoc_value",
    "historical_directional_margin",
    "posthoc_directional_margin",
    "classification",
    "interpretation",
)


class StrictModel(BaseModel):
    """Reject undeclared configuration keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScopeSettings(StrictModel):
    """Declare the post-hoc validation-only run scope."""

    label: Literal["post-hoc n=5 validation threshold-selection sensitivity"]
    selection_split: Literal["validation"]
    test_role: Literal["evaluation_only"]
    expected_detectors: tuple[str, ...]
    historical_seeds: tuple[int, ...]
    sensitivity_seeds: tuple[int, ...]
    inference_seeds: tuple[int, ...]
    influence_seed: int
    validation_images: int = Field(gt=0)
    validation_annotations: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_scope(self) -> ScopeSettings:
        if len(set(self.expected_detectors)) != len(self.expected_detectors):
            raise ValueError("detector scope contains duplicates")
        if len(set(self.sensitivity_seeds)) != len(self.sensitivity_seeds):
            raise ValueError("sensitivity seed scope contains duplicates")
        if not set(self.historical_seeds) < set(self.sensitivity_seeds):
            raise ValueError("historical seeds must be a strict subset of sensitivity seeds")
        if set(self.sensitivity_seeds) - set(self.historical_seeds) != set(self.inference_seeds):
            raise ValueError("inference seeds must be exactly the new sensitivity seeds")
        if self.influence_seed not in self.sensitivity_seeds:
            raise ValueError("influence seed is absent from the sensitivity scope")
        return self


class InputSettings(StrictModel):
    """Bind frozen model, validation, historical, and test-evaluation evidence."""

    phase5_config: Path
    phase5_summary: Path
    checkpoint_manifest: Path
    validation_split: str = Field(pattern=r"^[a-z0-9_-]+$")
    validation_annotations: Path
    historical_config: Path
    historical_manifest: Path
    historical_summary: Path
    historical_validation_table: Path
    historical_validation_per_seed_table: Path
    historical_selected_table: Path
    historical_selected_per_seed_table: Path
    n5_test_threshold_config: Path
    historical_n5_test_summary: Path
    historical_n5_test_table: Path
    historical_n5_test_per_seed_table: Path


class FrozenEvaluationSettings(StrictModel):
    """Record the exact inference/evaluation settings inherited from Phase 5."""

    score_threshold: float
    match_iou_threshold: float
    coco_minimum_score: float
    nms_iou_threshold: float
    max_detections: int


class ReferenceCheckSettings(StrictModel):
    """Bound inference-to-training validation count drift without changing selection."""

    max_true_positive_delta: int = Field(ge=0)
    max_false_positive_delta: int = Field(ge=0)
    max_false_negative_delta: int = Field(ge=0)
    max_prediction_count_delta: int = Field(ge=0)


class SelectionSettings(StrictModel):
    """Reproduce the historical grid, objective, aggregation, and tie rule."""

    start: float = Field(ge=0, le=1)
    stop: float = Field(ge=0, le=1)
    steps: int = Field(ge=2)
    rule: Literal["maximum_mean_f1"]
    averaging: Literal["equal_run_arithmetic_mean"]
    standard_deviation: Literal["sample"]
    tie_breaker: Literal["highest_threshold"]

    @model_validator(mode="after")
    def validate_grid(self) -> SelectionSettings:
        if self.stop <= self.start:
            raise ValueError("selection stop must exceed selection start")
        return self

    def thresholds(self) -> tuple[float, ...]:
        """Return the inclusive historical threshold grid."""

        return tuple(
            float(value) for value in np.round(np.linspace(self.start, self.stop, self.steps), 12)
        )


class ComparisonSettings(StrictModel):
    """Declare the operating endpoints and deterministic classification tolerance."""

    mean_metrics: tuple[str, ...]
    variability_metrics: tuple[str, ...]
    lower_is_preferred: tuple[str, ...]
    numeric_tolerance: float = Field(gt=0)


class ArtifactBinding(StrictModel):
    """Bind one protected historical artifact to immutable bytes."""

    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProtectionSettings(StrictModel):
    """Declare paths that Batch 43 must never overwrite."""

    protected_roots: tuple[Path, ...]
    historical_artifacts: tuple[ArtifactBinding, ...]


class OutputSettings(StrictModel):
    """Route every Batch 43 artifact to a versioned namespace."""

    log_dir: Path
    new_validation_bundles_dir: Path
    new_validation_manifest: Path
    validation_threshold_table: Path
    validation_threshold_per_seed_table: Path
    test_operating_points_table: Path
    test_operating_points_per_seed_table: Path
    conclusion_table: Path
    summary_json: Path


class SensitivityConfig(StrictModel):
    """Strict Batch 43 configuration."""

    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    scope: ScopeSettings
    inputs: InputSettings
    frozen_evaluation: FrozenEvaluationSettings
    reference_check: ReferenceCheckSettings
    selection: SelectionSettings
    comparison: ComparisonSettings
    protection: ProtectionSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        """Resolve one configured path against the repository root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


def load_sensitivity_config(path: str | Path) -> SensitivityConfig:
    """Load and strictly validate the Batch 43 YAML configuration."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("threshold sensitivity config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return SensitivityConfig.model_validate(payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _compare_reference_operating_point(
    metrics: Mapping[str, Any],
    reference_path: Path,
    *,
    detector: str,
    seed: int,
    limits: ReferenceCheckSettings,
) -> dict[str, Any]:
    """Compare re-inference with the archived training-time validation counts."""

    rows = _read_csv(reference_path)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one validation reference row: {reference_path}")
    reference = rows[0]
    observed = metrics["operating_point"]["overall"]
    fields = {
        "true_positives": ("tp", limits.max_true_positive_delta),
        "false_positives": ("fp", limits.max_false_positive_delta),
        "false_negatives": ("fn", limits.max_false_negative_delta),
        "operating_point_prediction_count": (
            "prediction_count",
            limits.max_prediction_count_delta,
        ),
    }
    comparisons: dict[str, Any] = {}
    exact = True
    for reference_field, (observed_field, maximum_delta) in fields.items():
        expected = int(reference[reference_field])
        actual = int(observed[observed_field])
        delta = actual - expected
        if abs(delta) > maximum_delta:
            raise ValueError(
                f"validation {observed_field} drift exceeds the configured bound for "
                f"{detector} seed {seed}: {actual} != {expected} "
                f"(allowed absolute delta {maximum_delta})"
            )
        comparisons[observed_field] = {
            "training_reference": expected,
            "reinference": actual,
            "delta": delta,
            "maximum_absolute_delta": maximum_delta,
        }
        exact = exact and delta == 0
    comparisons["metrics"] = {
        field: {
            "training_reference": float(reference[field]),
            "reinference": float(observed[field]),
            "delta": float(observed[field]) - float(reference[field]),
        }
        for field in ("precision", "recall", "f1")
    }
    return {
        "status": "exact" if exact else "within_configured_count_bounds",
        "exact": exact,
        "comparisons": comparisons,
    }


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def require_validation_selection(split: str) -> None:
    """Reject any attempt to select an operating threshold from test evidence."""

    if split != "validation":
        raise ValueError("threshold selection requires validation evidence only")


def validate_run_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    detectors: Sequence[str],
    seeds: Sequence[int],
    label: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Require the exact detector-by-seed factorial grid, including adverse runs."""

    expected = {(detector, seed) for detector in detectors for seed in seeds}
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        key = str(record.get("detector")), int(record.get("seed", -1))
        if key in observed:
            raise ValueError(f"duplicate {label} run: {key}")
        observed[key] = record
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"{label} run grid mismatch; missing={missing}, extra={extra}")
    return observed


def validate_output_isolation(config: SensitivityConfig) -> None:
    """Prevent any Batch 43 output from overlapping historical Phase 14 evidence."""

    outputs = [config.resolve(path) for path in config.outputs.model_dump().values()]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Batch 43 output paths must be unique")
    protected_files = {
        config.resolve(binding.path) for binding in config.protection.historical_artifacts
    }
    protected_roots = [config.resolve(path) for path in config.protection.protected_roots]
    for output in outputs:
        if output in protected_files or any(
            output.is_relative_to(root) for root in protected_roots
        ):
            raise ValueError(f"Batch 43 output overlaps a protected historical path: {output}")


def validate_historical_artifacts(config: SensitivityConfig) -> list[dict[str, Any]]:
    """Hash-check every protected historical artifact and output boundary."""

    validate_output_isolation(config)
    verified: list[dict[str, Any]] = []
    for binding in config.protection.historical_artifacts:
        path = config.resolve(binding.path)
        if not path.is_file():
            raise FileNotFoundError(f"protected historical artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != binding.sha256:
            raise ValueError(f"protected historical artifact hash mismatch: {path}")
        verified.append(_artifact(path, config.project_root))
    return verified


def _validate_historical_rule(config: SensitivityConfig) -> dict[str, Any]:
    historical_path = config.resolve(config.inputs.historical_config)
    payload = yaml.safe_load(historical_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("selection"), dict):
        raise ValueError("historical selection config is invalid")
    historical = payload["selection"]
    current = {
        "start": config.selection.start,
        "stop": config.selection.stop,
        "steps": config.selection.steps,
        "rule": config.selection.rule,
        "tie_breaker": config.selection.tie_breaker,
    }
    if historical != current:
        raise ValueError("Batch 43 changed the historical selection rule or grid")
    require_validation_selection(config.scope.selection_split)
    return {
        **current,
        "averaging": config.selection.averaging,
        "standard_deviation": config.selection.standard_deviation,
        "thresholds": list(config.selection.thresholds()),
    }


def _load_context(
    config: SensitivityConfig, *, require_cuda: bool
) -> tuple[Any, dict[str, Any], dict[tuple[str, int], Any], Any, dict[str, Any]]:
    protected = validate_historical_artifacts(config)
    rule = _validate_historical_rule(config)
    phase5 = load_phase5_config(config.resolve(config.inputs.phase5_config))
    summary_path, phase5_summary = _validate_source_summary(phase5)
    if summary_path != config.resolve(config.inputs.phase5_summary):
        raise ValueError("configured Phase 5 summary path differs from the evaluator contract")
    if tuple(phase5.seeds) != config.scope.sensitivity_seeds:
        raise ValueError("Phase 5 seed scope differs from the Batch 43 sensitivity scope")
    phase5_runs = validate_run_grid(
        [run.model_dump(mode="python") for run in phase5.runs],
        detectors=config.scope.expected_detectors,
        seeds=config.scope.sensitivity_seeds,
        label="Phase 5",
    )
    if phase5.evaluation.model_dump(mode="python") != config.frozen_evaluation.model_dump(
        mode="python"
    ):
        raise ValueError("Batch 43 evaluation settings differ from frozen Phase 5")
    training_configs = load_and_validate_training_configs(phase5)
    dataset = _load_validation_dataset(config, phase5, training_configs)
    if len(dataset) != config.scope.validation_images:
        raise ValueError("validation image count differs from the declared scope")
    annotation_count = sum(len(record.annotations) for record in dataset.records)
    if annotation_count != config.scope.validation_annotations:
        raise ValueError("validation annotation count differs from the declared scope")
    historical_manifest = _read_json(config.resolve(config.inputs.historical_manifest))
    if historical_manifest.get("status") != "complete":
        raise ValueError("historical validation manifest is incomplete")
    if historical_manifest.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError("historical validation evaluator differs from Phase 5")
    if historical_manifest.get("upstream", {}).get("validation_annotations", {}).get(
        "sha256"
    ) != sha256_file(dataset.annotation_file):
        raise ValueError("historical validation annotation hash mismatch")
    validate_run_grid(
        historical_manifest.get("runs", []),
        detectors=config.scope.expected_detectors,
        seeds=config.scope.historical_seeds,
        label="historical validation manifest",
    )
    release = _validate_release_manifest(
        config.resolve(config.inputs.checkpoint_manifest), phase5, summary_path
    )
    if set(release) != set(phase5_runs):
        raise ValueError("checkpoint manifest and Phase 5 run grids differ")
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("validation checkpoint inference requires CUDA")
    return (
        phase5,
        phase5_summary,
        training_configs,
        dataset,
        {
            "protected_historical_artifacts": protected,
            "historical_rule": rule,
            "checkpoint_count": len(release),
        },
    )


def _validate_bundle(
    path: Path,
    *,
    detector: str,
    seed: int,
    phase5: Any,
    dataset: Any,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = _read_bundle(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("detector") != detector
        or int(payload.get("seed", -1)) != seed
        or payload.get("split") != "validation"
    ):
        raise ValueError(f"validation bundle identity mismatch: {path}")
    if payload.get("evaluation") != phase5.evaluation.model_dump(mode="json"):
        raise ValueError(f"validation bundle evaluator mismatch: {path}")
    if payload.get("annotation_sha256") != sha256_file(dataset.annotation_file):
        raise ValueError(f"validation bundle annotation mismatch: {path}")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"validation bundle checkpoint mismatch: {path}")
    predictions = _deserialize_predictions(payload)
    expected_ids = {record.file_name for record in dataset.records}
    if len(predictions) != len(dataset) or {item.image_id for item in predictions} != expected_ids:
        raise ValueError(f"validation bundle image identity mismatch: {path}")
    scores = np.asarray(
        [score for item in predictions for score in item.scores.tolist()], dtype=np.float64
    )
    if scores.size and (
        not np.isfinite(scores).all()
        or np.any(scores + 1e-12 < phase5.evaluation.coco_minimum_score)
    ):
        raise ValueError(f"validation bundle contains invalid candidate scores: {path}")
    return {
        "payload": payload,
        "predictions": predictions,
        "prediction_count": int(scores.size),
        "artifact": _artifact(path, phase5.project_root),
    }


def preflight(config: SensitivityConfig) -> dict[str, Any]:
    """Validate all checkpoint, source-data, historical, and output gates."""

    import torch

    phase5, _summary, training_configs, dataset, audits = _load_context(config, require_cuda=True)
    reference_tables: list[dict[str, Any]] = []
    for run in phase5.runs:
        if run.seed not in config.scope.inference_seeds:
            continue
        reference = _validation_reference_table(
            run.detector, training_configs[(run.detector, run.seed)]
        )
        if not reference.is_file():
            raise FileNotFoundError(f"validation reference table is missing: {reference}")
        reference_tables.append(_artifact(reference, config.project_root))
    bundle_dir = config.resolve(config.outputs.new_validation_bundles_dir)
    expected_paths = [
        bundle_dir / f"{detector}_seed{seed}_validation_predictions.json.gz"
        for detector in config.scope.expected_detectors
        for seed in config.scope.inference_seeds
    ]
    return {
        "status": "ready",
        "analysis_id": config.analysis_id,
        "sensitivity_label": config.scope.label,
        "historical_thresholds_verified": True,
        "historical_rule": audits["historical_rule"],
        "checkpoints_verified": audits["checkpoint_count"],
        "checkpoint_bytes_verified": sum(
            phase5.resolve(run.checkpoint).stat().st_size for run in phase5.runs
        ),
        "validation_images": len(dataset),
        "validation_annotations": sum(len(record.annotations) for record in dataset.records),
        "new_bundle_count": len(expected_paths),
        "new_bundles_ready": all(path.is_file() for path in expected_paths),
        "reference_tables": reference_tables,
        "cuda_device": torch.cuda.get_device_name(0),
        "free_disk_bytes": shutil.disk_usage(config.project_root).free,
        "selection_uses_test_labels": False,
        "performs_training": False,
        "preflight_performs_checkpoint_loading": False,
        "preflight_performs_model_inference": False,
    }


def collect_validation_predictions(config: SensitivityConfig) -> dict[str, Any]:
    """Infer only the four missing validation bundles from frozen checkpoints."""

    from src.utils.seed import initialize_reproducibility, seed_everything

    readiness = preflight(config)
    phase5, phase5_summary, training_configs, dataset, audits = _load_context(
        config, require_cuda=True
    )
    targets = _targets_from_dataset(dataset)
    annotation_path = dataset.annotation_file.resolve()
    log_dir = config.resolve(config.outputs.log_dir)
    initialize_reproducibility(config.scope.inference_seeds[0], log_dir)
    records: list[dict[str, Any]] = []
    progress_path = log_dir / "progress.json"
    bundle_dir = config.resolve(config.outputs.new_validation_bundles_dir)
    for run in sorted(phase5.runs, key=lambda item: (item.detector, item.seed)):
        if run.seed not in config.scope.inference_seeds:
            continue
        checkpoint_path = phase5.resolve(run.checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        bundle_path = bundle_dir / f"{run.detector}_seed{run.seed}_validation_predictions.json.gz"
        model_config = training_configs[(run.detector, run.seed)]
        if bundle_path.is_file():
            print(
                f"reusing verified validation bundle for {run.detector} seed {run.seed}", flush=True
            )
            validated = _validate_bundle(
                bundle_path,
                detector=run.detector,
                seed=run.seed,
                phase5=phase5,
                dataset=dataset,
                checkpoint_sha256=checkpoint_sha256,
            )
            predictions = validated["predictions"]
            inference_seconds = float(validated["payload"]["inference_seconds"])
            action = "reused_verified"
        else:
            print(f"inferring validation for {run.detector} seed {run.seed}", flush=True)
            seed_everything(run.seed)
            if run.detector == "faster_rcnn":
                import torch

                torch.backends.cuda.matmul.allow_tf32 = model_config.runtime.allow_tf32
                torch.backends.cudnn.allow_tf32 = model_config.runtime.allow_tf32
                predictions, inference_seconds = _collect_faster_rcnn_predictions(
                    run,
                    phase5,
                    model_config,
                    dataset,
                    expected_config_sha256=_frozen_faster_config_sha256(run, phase5, model_config),
                )
            else:
                predictions, inference_seconds = _collect_yolo_predictions(
                    run, phase5, model_config, dataset
                )
            metrics = evaluate_prediction_records(
                predictions,
                targets,
                category_names=dataset.category_names,
                settings=phase5.evaluation,
            )
            _write_validation_bundle(
                bundle_path,
                detector=run.detector,
                seed=run.seed,
                checkpoint_sha256=checkpoint_sha256,
                annotation_path=annotation_path,
                predictions=predictions,
                metrics=metrics,
                evaluation=phase5.evaluation,
                inference_seconds=inference_seconds,
            )
            action = "generated_inference_only"
        metrics = evaluate_prediction_records(
            predictions,
            targets,
            category_names=dataset.category_names,
            settings=phase5.evaluation,
        )
        reference_path = _validation_reference_table(run.detector, model_config)
        reference_comparison = _compare_reference_operating_point(
            metrics,
            reference_path,
            detector=run.detector,
            seed=run.seed,
            limits=config.reference_check,
        )
        validated = _validate_bundle(
            bundle_path,
            detector=run.detector,
            seed=run.seed,
            phase5=phase5,
            dataset=dataset,
            checkpoint_sha256=checkpoint_sha256,
        )
        record = {
            "detector": run.detector,
            "seed": run.seed,
            "action": action,
            "checkpoint": _artifact(checkpoint_path, config.project_root),
            "training_config": _artifact(phase5.resolve(run.training_config), config.project_root),
            "validation_reference_table": _artifact(reference_path, config.project_root),
            "validation_bundle": validated["artifact"],
            "image_count": len(predictions),
            "prediction_count": validated["prediction_count"],
            "inference_seconds": inference_seconds,
            "reference_operating_point_comparison": reference_comparison,
        }
        records.append(record)
        _atomic_json(
            progress_path,
            {
                "status": "in_progress",
                "analysis_id": config.analysis_id,
                "config_sha256": sha256_file(config.source_path),
                "completed_runs": records,
            },
        )
        del predictions, metrics
    validate_run_grid(
        records,
        detectors=config.scope.expected_detectors,
        seeds=config.scope.inference_seeds,
        label="new validation bundle",
    )
    protected_after = validate_historical_artifacts(config)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "sensitivity_label": config.scope.label,
        "config": _artifact(config.source_path, config.project_root),
        "source_identity": {
            _relative(Path(__file__), config.project_root): sha256_file(Path(__file__)),
            "src/evaluate.py": sha256_file(config.project_root / "src/evaluate.py"),
            "src/evaluate_threshold_selection.py": sha256_file(
                config.project_root / "src/evaluate_threshold_selection.py"
            ),
        },
        "source_phase5": {
            "config": _artifact(config.resolve(config.inputs.phase5_config), config.project_root),
            "summary": _artifact(config.resolve(config.inputs.phase5_summary), config.project_root),
            "status": phase5_summary["status"],
        },
        "checkpoint_manifest": _artifact(
            config.resolve(config.inputs.checkpoint_manifest), config.project_root
        ),
        "historical_validation_manifest": _artifact(
            config.resolve(config.inputs.historical_manifest), config.project_root
        ),
        "validation_annotations": _artifact(annotation_path, config.project_root),
        "environment": {
            "run_environment": _artifact(log_dir / "run_environment.json", config.project_root),
            "pip_freeze": _artifact(log_dir / "pip_freeze.txt", config.project_root),
        },
        "protocol": {
            "selection_split": config.scope.selection_split,
            "inference_seeds": list(config.scope.inference_seeds),
            "historical_seeds_reused": list(config.scope.historical_seeds),
            "combined_sensitivity_seeds": list(config.scope.sensitivity_seeds),
            "evaluation": phase5.evaluation.model_dump(mode="json"),
            "reference_check": config.reference_check.model_dump(mode="json"),
            "preprocessing_and_adapter": "unchanged run-specific frozen training contracts",
            "faster_rcnn_tf32": (
                "training-config runtime.allow_tf32 applied before the frozen adapter"
            ),
            "coordinate_logic": (
                "canonical detector adapters and xyxy serialization from src.evaluate"
            ),
            "performs_training": False,
            "performs_checkpoint_loading": True,
            "performs_model_inference": True,
            "test_annotations_accessed": False,
            "test_labels_used_for_selection": False,
        },
        "counts": {
            "historical_bundles": len(config.scope.expected_detectors)
            * len(config.scope.historical_seeds),
            "new_bundles": len(records),
            "combined_bundles": len(config.scope.expected_detectors)
            * len(config.scope.sensitivity_seeds),
            "images_per_bundle": len(dataset),
            "annotations": sum(len(record.annotations) for record in dataset.records),
        },
        "runs": records,
        "historical_protection": {
            "verified_before": audits["protected_historical_artifacts"],
            "verified_after": protected_after,
            "overwritten": False,
        },
        "preflight": {key: value for key, value in readiness.items() if key != "free_disk_bytes"},
    }
    manifest_path = _atomic_json(config.resolve(config.outputs.new_validation_manifest), manifest)
    _atomic_json(
        progress_path,
        {
            "status": "complete",
            "analysis_id": config.analysis_id,
            "config_sha256": sha256_file(config.source_path),
            "completed_runs": records,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(
        json.dumps(
            {"status": "complete", "manifest": _relative(manifest_path, config.project_root)},
            indent=2,
        )
    )
    return manifest


def _bundle_record(
    record: Mapping[str, Any], *, phase5: Any, dataset: Any, root: Path
) -> dict[str, Any]:
    detector = str(record["detector"])
    seed = int(record["seed"])
    bundle_info = record["validation_bundle"]
    bundle_path = root / str(bundle_info["path"])
    checkpoint_info = record["checkpoint"]
    if sha256_file(bundle_path) != bundle_info["sha256"]:
        raise ValueError(f"validation bundle hash mismatch: {bundle_path}")
    validated = _validate_bundle(
        bundle_path,
        detector=detector,
        seed=seed,
        phase5=phase5,
        dataset=dataset,
        checkpoint_sha256=str(checkpoint_info["sha256"]),
    )
    return {
        "detector": detector,
        "seed": seed,
        "path": bundle_path,
        "sha256": bundle_info["sha256"],
        "predictions": validated["predictions"],
    }


def _load_validation_bundles(
    config: SensitivityConfig, phase5: Any, dataset: Any
) -> list[dict[str, Any]]:
    historical_manifest = _read_json(config.resolve(config.inputs.historical_manifest))
    historical = [
        _bundle_record(record, phase5=phase5, dataset=dataset, root=config.project_root)
        for record in historical_manifest["runs"]
    ]
    validate_run_grid(
        historical,
        detectors=config.scope.expected_detectors,
        seeds=config.scope.historical_seeds,
        label="historical validation bundle",
    )
    new_manifest_path = config.resolve(config.outputs.new_validation_manifest)
    if not new_manifest_path.is_file():
        raise FileNotFoundError("new validation manifest is missing; run collect-validation")
    new_manifest = _read_json(new_manifest_path)
    if (
        new_manifest.get("status") != "complete"
        or new_manifest.get("analysis_id") != config.analysis_id
        or new_manifest.get("config", {}).get("sha256") != sha256_file(config.source_path)
    ):
        raise ValueError("new validation manifest identity is invalid")
    protocol = new_manifest.get("protocol", {})
    if (
        protocol.get("test_annotations_accessed") is not False
        or protocol.get("test_labels_used_for_selection") is not False
    ):
        raise ValueError("new validation manifest does not prove test isolation")
    new = [
        _bundle_record(record, phase5=phase5, dataset=dataset, root=config.project_root)
        for record in new_manifest.get("runs", [])
    ]
    validate_run_grid(
        new,
        detectors=config.scope.expected_detectors,
        seeds=config.scope.inference_seeds,
        label="new validation bundle",
    )
    combined = sorted(historical + new, key=lambda item: (item["detector"], item["seed"]))
    validate_run_grid(
        combined,
        detectors=config.scope.expected_detectors,
        seeds=config.scope.sensitivity_seeds,
        label="combined validation bundle",
    )
    return combined


def _sweep_bundles(
    bundles: Sequence[Mapping[str, Any]],
    *,
    targets: Any,
    category_names: Mapping[int, str],
    phase5: Any,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_ids = tuple(sorted(category_names))
    for bundle in bundles:
        rows.extend(
            sweep_prediction_records(
                bundle["predictions"],
                targets,
                detector=str(bundle["detector"]),
                seed=int(bundle["seed"]),
                thresholds=thresholds,
                class_ids=class_ids,
                iou_threshold=phase5.evaluation.match_iou_threshold,
                max_detections=phase5.evaluation.max_detections,
            )
        )
    return rows


def _validate_numeric_table(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    numeric_fields: Sequence[str],
    tolerance: float,
    label: str,
) -> None:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], Mapping[str, Any]]:
        result: dict[tuple[str, ...], Mapping[str, Any]] = {}
        for row in rows:
            key = tuple(str(row[field]) for field in keys)
            if key in result:
                raise ValueError(f"duplicate {label} row: {key}")
            result[key] = row
        return result

    actual_by_key = keyed(actual)
    expected_by_key = keyed(expected)
    if set(actual_by_key) != set(expected_by_key):
        raise ValueError(f"{label} key grid does not reproduce")
    for key, row in actual_by_key.items():
        reference = expected_by_key[key]
        for field in numeric_fields:
            if not np.isclose(float(row[field]), float(reference[field]), rtol=0, atol=tolerance):
                raise ValueError(f"{label} does not reproduce at {key}, field {field}")


def _reproduce_historical_selection(
    config: SensitivityConfig,
    *,
    bundles: Sequence[Mapping[str, Any]],
    targets: Any,
    category_names: Mapping[int, str],
    phase5: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    historical_bundles = [
        bundle for bundle in bundles if int(bundle["seed"]) in config.scope.historical_seeds
    ]
    per_seed = _sweep_bundles(
        historical_bundles,
        targets=targets,
        category_names=category_names,
        phase5=phase5,
        thresholds=config.selection.thresholds(),
    )
    aggregate = aggregate_threshold_rows(per_seed, ddof=phase5.runtime.statistics_ddof)
    selected = select_validation_thresholds(aggregate)
    tolerance = config.comparison.numeric_tolerance
    _validate_numeric_table(
        per_seed,
        _read_csv(config.resolve(config.inputs.historical_validation_per_seed_table)),
        keys=("detector", "seed", "threshold"),
        numeric_fields=(
            "precision",
            "recall",
            "f1",
            "true_positives",
            "false_positives",
            "false_negatives",
            "prediction_count",
            "target_count",
        ),
        tolerance=tolerance,
        label="historical validation per-seed sweep",
    )
    _validate_numeric_table(
        aggregate,
        _read_csv(config.resolve(config.inputs.historical_validation_table)),
        keys=("detector", "threshold"),
        numeric_fields=(
            "seed_count",
            "precision",
            "precision_std",
            "recall",
            "recall_std",
            "f1",
            "f1_std",
        ),
        tolerance=tolerance,
        label="historical validation aggregate sweep",
    )
    summary = _read_json(config.resolve(config.inputs.historical_summary))
    selected_summary = summary.get("selection", {}).get("selected", {})
    for detector, row in selected.items():
        if not np.isclose(
            float(row["threshold"]),
            float(selected_summary[detector]["threshold"]),
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(f"historical selected threshold does not reproduce: {detector}")
    return per_seed, aggregate, selected


def _sample_std(values: Sequence[float], *, ddof: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= ddof or not np.isfinite(array).all():
        raise ValueError("cannot compute sample standard deviation")
    return 0.0 if np.all(array == array[0]) else float(np.std(array, ddof=ddof))


def _evaluate_test_scenarios(
    config: SensitivityConfig,
    *,
    phase5: Any,
    historical_selected: Mapping[str, Mapping[str, Any]],
    posthoc_selected: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Any]:
    test_config = load_threshold_sweep_config(
        config.resolve(config.inputs.n5_test_threshold_config)
    )
    test_phase5, _summary, test_bundles, test_targets, categories = _validate_upstream(test_config)
    if test_phase5.split != "test" or config.scope.test_role != "evaluation_only":
        raise ValueError("held-out evidence must be evaluation-only test data")
    if test_phase5.evaluation != phase5.evaluation:
        raise ValueError("validation and test evaluator settings differ")
    validate_run_grid(
        test_bundles,
        detectors=config.scope.expected_detectors,
        seeds=config.scope.sensitivity_seeds,
        label="test prediction bundle",
    )
    scenarios = {
        "historical_n3_selection": historical_selected,
        "posthoc_n5_validation_sensitivity": posthoc_selected,
    }
    rows: list[dict[str, Any]] = []
    class_ids = tuple(sorted(categories))
    for scope, selected in scenarios.items():
        selection_n = (
            len(config.scope.historical_seeds)
            if scope == "historical_n3_selection"
            else len(config.scope.sensitivity_seeds)
        )
        for bundle in test_bundles:
            detector = str(bundle["detector"])
            threshold = float(selected[detector]["threshold"])
            evaluated = sweep_prediction_records(
                bundle["predictions"],
                test_targets,
                detector=detector,
                seed=int(bundle["seed"]),
                thresholds=(threshold,),
                class_ids=class_ids,
                iou_threshold=phase5.evaluation.match_iou_threshold,
                max_detections=phase5.evaluation.max_detections,
            )[0]
            rows.append(
                {
                    "selection_scope": scope,
                    "detector": detector,
                    "seed": int(bundle["seed"]),
                    "selection_split": "validation",
                    "selection_rule": "maximum equal-run arithmetic mean validation F1",
                    "tie_breaker": "highest threshold among exact mean-F1 ties",
                    "selected_threshold": threshold,
                    "threshold_selection_run_count": selection_n,
                    "test_run_count": len(config.scope.sensitivity_seeds),
                    "test_precision": evaluated["precision"],
                    "test_recall": evaluated["recall"],
                    "test_f1": evaluated["f1"],
                    "test_true_positives": evaluated["true_positives"],
                    "test_false_positives": evaluated["false_positives"],
                    "test_false_negatives": evaluated["false_negatives"],
                    "test_fp_per_image": evaluated["false_positives"] / len(test_targets),
                    "test_prediction_count": evaluated["prediction_count"],
                    "test_target_count": evaluated["target_count"],
                }
            )
    validate_run_grid(
        [row for row in rows if row["selection_scope"] == "posthoc_n5_validation_sensitivity"],
        detectors=config.scope.expected_detectors,
        seeds=config.scope.sensitivity_seeds,
        label="post-hoc test operating point",
    )
    return rows, test_bundles


def _aggregate_test_rows(
    config: SensitivityConfig,
    *,
    test_rows: Sequence[Mapping[str, Any]],
    historical_selected: Mapping[str, Mapping[str, Any]],
    posthoc_selected: Mapping[str, Mapping[str, Any]],
    ddof: int,
) -> list[dict[str, Any]]:
    selected_by_scope = {
        "historical_n3_selection": historical_selected,
        "posthoc_n5_validation_sensitivity": posthoc_selected,
    }
    result: list[dict[str, Any]] = []
    for scope, selected in selected_by_scope.items():
        for detector in config.scope.expected_detectors:
            group = [
                row
                for row in test_rows
                if row["selection_scope"] == scope and row["detector"] == detector
            ]
            if {int(row["seed"]) for row in group} != set(config.scope.sensitivity_seeds):
                raise ValueError(f"test aggregation excludes a required run: {scope}, {detector}")
            validation = selected[detector]
            entry: dict[str, Any] = {
                "selection_scope": scope,
                "sensitivity_label": (
                    "historical n=3 selection applied to n=5 test"
                    if scope == "historical_n3_selection"
                    else config.scope.label
                ),
                "detector": detector,
                "selection_split": "validation",
                "selection_rule": "maximum equal-run arithmetic mean validation F1",
                "tie_breaker": "highest threshold among exact mean-F1 ties",
                "selected_threshold": validation["threshold"],
                "threshold_selection_run_count": (
                    len(config.scope.historical_seeds)
                    if scope == "historical_n3_selection"
                    else len(config.scope.sensitivity_seeds)
                ),
                "test_run_count": len(group),
                "validation_precision": validation["precision"],
                "validation_precision_std": validation["precision_std"],
                "validation_recall": validation["recall"],
                "validation_recall_std": validation["recall_std"],
                "validation_f1": validation["f1"],
                "validation_f1_std": validation["f1_std"],
            }
            for metric in config.comparison.mean_metrics:
                values = [float(row[f"test_{metric}"]) for row in group]
                entry[f"test_{metric}"] = float(np.mean(values))
                entry[f"test_{metric}_std"] = _sample_std(values, ddof=ddof)
            result.append(entry)
    return result


def _validate_batch35_reproduction(
    config: SensitivityConfig,
    *,
    aggregate: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
) -> None:
    tolerance = config.comparison.numeric_tolerance
    historical_rows = [
        row for row in per_seed if row["selection_scope"] == "historical_n3_selection"
    ]
    _validate_numeric_table(
        historical_rows,
        _read_csv(config.resolve(config.inputs.historical_n5_test_per_seed_table)),
        keys=("detector", "seed"),
        numeric_fields=(
            "selected_threshold",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_true_positives",
            "test_false_positives",
            "test_false_negatives",
            "test_prediction_count",
            "test_target_count",
        ),
        tolerance=tolerance,
        label="Batch 35 fixed-threshold per-seed result",
    )
    historical_aggregate = [
        row for row in aggregate if row["selection_scope"] == "historical_n3_selection"
    ]
    _validate_numeric_table(
        historical_aggregate,
        _read_csv(config.resolve(config.inputs.historical_n5_test_table)),
        keys=("detector",),
        numeric_fields=(
            "selected_threshold",
            "test_precision",
            "test_precision_std",
            "test_recall",
            "test_recall_std",
            "test_f1",
            "test_f1_std",
        ),
        tolerance=tolerance,
        label="Batch 35 fixed-threshold aggregate result",
    )


def classify_margin_change(old: float, new: float, *, tolerance: float) -> str:
    """Classify preservation or change of an originally positive directional margin."""

    if old < -tolerance:
        raise ValueError("historical directional margin must be nonnegative")
    if new < -tolerance:
        return "reversed"
    if np.isclose(new, old, rtol=0, atol=tolerance):
        return "unchanged"
    return "strengthened" if new > old else "weakened"


def _directional_conclusion(
    historical: Mapping[str, Mapping[str, Any]],
    posthoc: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    lower_is_preferred: bool,
    tolerance: float,
    noun: str,
) -> dict[str, Any]:
    detectors = sorted(historical)
    if len(detectors) != 2:
        raise ValueError("directional conclusions require exactly two detectors")
    first, second = detectors
    first_value = float(historical[first][field])
    second_value = float(historical[second][field])
    if np.isclose(first_value, second_value, rtol=0, atol=tolerance):
        leader, other = first, second
        direction = "equal"
        old_margin = 0.0
        new_margin = abs(float(posthoc[first][field]) - float(posthoc[second][field]))
        classification = "unchanged" if new_margin <= tolerance else "weakened"
        conclusion = f"The pipelines have equal {noun}"
    else:
        choose_first = (
            first_value < second_value if lower_is_preferred else first_value > second_value
        )
        leader, other = (first, second) if choose_first else (second, first)
        sign = -1.0 if lower_is_preferred else 1.0
        old_margin = sign * (float(historical[leader][field]) - float(historical[other][field]))
        new_margin = sign * (float(posthoc[leader][field]) - float(posthoc[other][field]))
        classification = classify_margin_change(old_margin, new_margin, tolerance=tolerance)
        direction = "lower" if lower_is_preferred else "higher"
        conclusion = f"{leader} has {direction} {noun} than {other}"
    return {
        "conclusion": conclusion,
        "historical_value": (
            f"{first}={historical[first][field]:.12g}; {second}={historical[second][field]:.12g}"
        ),
        "posthoc_value": (
            f"{first}={posthoc[first][field]:.12g}; {second}={posthoc[second][field]:.12g}"
        ),
        "historical_directional_margin": old_margin,
        "posthoc_directional_margin": new_margin,
        "classification": classification,
        "direction": direction,
    }


def build_conclusion_rows(
    config: SensitivityConfig, aggregate: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Classify threshold, mean-endpoint, and run-variability conclusions."""

    by_scope = {
        scope: {str(row["detector"]): row for row in aggregate if row["selection_scope"] == scope}
        for scope in ("historical_n3_selection", "posthoc_n5_validation_sensitivity")
    }
    historical = by_scope["historical_n3_selection"]
    posthoc = by_scope["posthoc_n5_validation_sensitivity"]
    rows: list[dict[str, Any]] = []
    threshold = _directional_conclusion(
        historical,
        posthoc,
        field="selected_threshold",
        lower_is_preferred=False,
        tolerance=config.comparison.numeric_tolerance,
        noun="selected validation threshold",
    )
    rows.append(
        {
            "analysis": "detector-specific threshold separation",
            **{key: threshold[key] for key in CONCLUSION_FIELDS if key in threshold},
            "interpretation": (
                "Higher means more selective; this is score-scale separation, not clinical utility."
            ),
        }
    )
    for metric in config.comparison.mean_metrics:
        comparison = _directional_conclusion(
            historical,
            posthoc,
            field=f"test_{metric}",
            lower_is_preferred=metric in config.comparison.lower_is_preferred,
            tolerance=config.comparison.numeric_tolerance,
            noun=f"mean test {metric.replace('_', ' ')}",
        )
        rows.append(
            {
                "analysis": f"test mean {metric}",
                **{key: comparison[key] for key in CONCLUSION_FIELDS if key in comparison},
                "interpretation": (
                    "Both scenarios use the same five frozen test runs; only validation "
                    "run coverage changes the selected threshold."
                ),
            }
        )
    for metric in config.comparison.variability_metrics:
        comparison = _directional_conclusion(
            historical,
            posthoc,
            field=f"test_{metric}_std",
            lower_is_preferred=True,
            tolerance=config.comparison.numeric_tolerance,
            noun=f"run-level variability in test {metric.replace('_', ' ')}",
        )
        rows.append(
            {
                "analysis": f"test run-level variability {metric}",
                **{key: comparison[key] for key in CONCLUSION_FIELDS if key in comparison},
                "interpretation": (
                    "Variability is the sample SD across five retained runs; lower is more stable."
                ),
            }
        )
    return rows


def run_analysis(config: SensitivityConfig) -> dict[str, Any]:
    """Select on five validation runs, apply once to test, and compare with history."""

    phase5, phase5_summary, _training_configs, dataset, audits = _load_context(
        config, require_cuda=False
    )
    validation_bundles = _load_validation_bundles(config, phase5, dataset)
    targets = _targets_from_dataset(dataset)
    historical_per_seed, historical_aggregate, historical_selected = (
        _reproduce_historical_selection(
            config,
            bundles=validation_bundles,
            targets=targets,
            category_names=dataset.category_names,
            phase5=phase5,
        )
    )
    require_validation_selection(config.scope.selection_split)
    validation_per_seed = _sweep_bundles(
        validation_bundles,
        targets=targets,
        category_names=dataset.category_names,
        phase5=phase5,
        thresholds=config.selection.thresholds(),
    )
    validation_aggregate = aggregate_threshold_rows(
        validation_per_seed, ddof=phase5.runtime.statistics_ddof
    )
    posthoc_selected = select_validation_thresholds(validation_aggregate)
    test_rows, test_bundles = _evaluate_test_scenarios(
        config,
        phase5=phase5,
        historical_selected=historical_selected,
        posthoc_selected=posthoc_selected,
    )
    test_aggregate = _aggregate_test_rows(
        config,
        test_rows=test_rows,
        historical_selected=historical_selected,
        posthoc_selected=posthoc_selected,
        ddof=phase5.runtime.statistics_ddof,
    )
    _validate_batch35_reproduction(config, aggregate=test_aggregate, per_seed=test_rows)
    conclusion_rows = build_conclusion_rows(config, test_aggregate)
    validation_path = _atomic_csv(
        config.resolve(config.outputs.validation_threshold_table),
        VALIDATION_FIELDS,
        validation_aggregate,
    )
    validation_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.validation_threshold_per_seed_table),
        VALIDATION_PER_SEED_FIELDS,
        validation_per_seed,
    )
    test_path = _atomic_csv(
        config.resolve(config.outputs.test_operating_points_table),
        TEST_AGGREGATE_FIELDS,
        test_aggregate,
    )
    test_per_seed_path = _atomic_csv(
        config.resolve(config.outputs.test_operating_points_per_seed_table),
        TEST_PER_SEED_FIELDS,
        test_rows,
    )
    conclusion_path = _atomic_csv(
        config.resolve(config.outputs.conclusion_table),
        CONCLUSION_FIELDS,
        conclusion_rows,
    )
    protected_after = validate_historical_artifacts(config)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": config.analysis_id,
        "sensitivity_label": config.scope.label,
        "config": _artifact(config.source_path, config.project_root),
        "source_identity": {
            _relative(Path(__file__), config.project_root): sha256_file(Path(__file__)),
            "src/evaluate.py": sha256_file(config.project_root / "src/evaluate.py"),
            "src/evaluate_threshold_selection.py": sha256_file(
                config.project_root / "src/evaluate_threshold_selection.py"
            ),
            "src/evaluate_threshold_sweep.py": sha256_file(
                config.project_root / "src/evaluate_threshold_sweep.py"
            ),
        },
        "upstream": {
            "phase5_config": _artifact(
                config.resolve(config.inputs.phase5_config), config.project_root
            ),
            "phase5_summary": _artifact(
                config.resolve(config.inputs.phase5_summary), config.project_root
            ),
            "new_validation_manifest": _artifact(
                config.resolve(config.outputs.new_validation_manifest), config.project_root
            ),
            "historical_selection_summary": _artifact(
                config.resolve(config.inputs.historical_summary), config.project_root
            ),
            "batch35_summary": _artifact(
                config.resolve(config.inputs.historical_n5_test_summary), config.project_root
            ),
            "test_prediction_bundles": [
                {
                    "detector": bundle["detector"],
                    "seed": bundle["seed"],
                    "path": _relative(bundle["path"], config.project_root),
                    "sha256": bundle["sha256"],
                }
                for bundle in test_bundles
            ],
        },
        "historical_reproduction": {
            "status": "exact_within_configured_tolerance",
            "selection_split": "validation",
            "seeds": list(config.scope.historical_seeds),
            "grid": audits["historical_rule"],
            "per_seed_rows": len(historical_per_seed),
            "aggregate_rows": len(historical_aggregate),
            "selected_thresholds": {
                detector: float(row["threshold"]) for detector, row in historical_selected.items()
            },
        },
        "posthoc_selection": {
            "label": config.scope.label,
            "selection_split": "validation",
            "seeds": list(config.scope.sensitivity_seeds),
            "rule": "maximum equal-run arithmetic mean validation F1",
            "tie_breaker": "highest threshold among exact mean-F1 ties",
            "selected": {
                detector: {
                    "threshold": float(row["threshold"]),
                    "validation_precision": float(row["precision"]),
                    "validation_recall": float(row["recall"]),
                    "validation_f1": float(row["f1"]),
                }
                for detector, row in posthoc_selected.items()
            },
            "prospectively_frozen": False,
        },
        "test_evaluation": {
            "role": "evaluation_only",
            "policy": (
                "apply each validation-selected threshold unchanged to all five frozen "
                "internal-test bundles"
            ),
            "threshold_evaluations_per_bundle_per_scenario": 1,
            "test_labels_used_for_threshold_selection": False,
            "performs_training": False,
            "performs_checkpoint_loading": False,
            "performs_model_inference": False,
            "operating_points": test_aggregate,
        },
        "conclusions": conclusion_rows,
        "counts": {
            "validation_bundles": len(validation_bundles),
            "validation_images_per_bundle": len(dataset),
            "validation_threshold_rows_per_seed": len(validation_per_seed),
            "validation_threshold_rows_aggregate": len(validation_aggregate),
            "test_bundles": len(test_bundles),
            "test_rows": len(test_rows),
            "conclusion_rows": len(conclusion_rows),
        },
        "historical_protection": {
            "verified_before": audits["protected_historical_artifacts"],
            "verified_after": protected_after,
            "overwritten": False,
        },
        "phase5_status": phase5_summary["status"],
        "artifacts": {
            "validation_threshold_table": _artifact(validation_path, config.project_root),
            "validation_threshold_per_seed_table": _artifact(
                validation_per_seed_path, config.project_root
            ),
            "test_operating_points_table": _artifact(test_path, config.project_root),
            "test_operating_points_per_seed_table": _artifact(
                test_per_seed_path, config.project_root
            ),
            "conclusion_table": _artifact(conclusion_path, config.project_root),
        },
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    print(
        json.dumps(
            {"status": "complete", "summary": _relative(summary_path, config.project_root)},
            indent=2,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the Batch 43 command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/threshold_selection_n5_validation_sensitivity.yaml"),
    )
    parser.add_argument(
        "--mode", choices=("preflight", "collect-validation", "run"), default="preflight"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight, collect only missing validation bundles, or run offline analysis."""

    args = build_parser().parse_args(argv)
    config = load_sensitivity_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
    elif args.mode == "collect-validation":
        collect_validation_predictions(config)
    else:
        run_analysis(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
