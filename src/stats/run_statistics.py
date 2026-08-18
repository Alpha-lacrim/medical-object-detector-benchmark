"""Run Phase 8 paired inference from the frozen Phase 5/6 prediction bundles."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evaluate import load_phase5_config, sha256_file
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget
from src.robustness.corruptions import expand_conditions
from src.robustness.run_robustness import load_robustness_config
from src.stats.paired import (
    METRICS,
    DetectionEvidence,
    EvidencePair,
    analyze_pair,
    build_evidence,
    build_patient_clusters,
    estimate_pair,
    holm_adjust,
    json_number,
)
from src.utils.seed import initialize_reproducibility

DetectorName = Literal["faster_rcnn", "yolo11s"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    phase5_config: Path
    phase5_summary: Path
    phase6_config: Path
    phase6_summary: Path
    test_annotations: Path
    test_split_manifest: Path


class AnalysisSettings(StrictModel):
    detector_a: DetectorName
    detector_b: DetectorName
    metrics: tuple[str, ...]
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    permutation_resamples: int = Field(ge=100)
    bootstrap_method: Literal["paired_hierarchical_patient_cluster_percentile"]
    permutation_method: Literal["paired_patient_cluster_label_swap"]
    effect_size: Literal["paired_raw_difference_with_cluster_bootstrap_ci"]
    manifest_image_column: str = Field(min_length=1)
    patient_group_column: str = Field(min_length=1)
    multiple_comparison_correction: Literal["holm"]
    clean_correction_scope: Literal["across_7_predictive_metrics"]
    corruption_correction_scope: Literal["per_metric_and_estimand_across_35_corrupted_conditions"]
    include_raw_corruption_comparisons: Literal[True]
    include_clean_relative_retention_comparisons: Literal[True]

    @model_validator(mode="after")
    def validate_contract(self) -> AnalysisSettings:
        if self.detector_a == self.detector_b:
            raise ValueError("detectors must differ")
        if self.metrics != METRICS:
            raise ValueError(f"metrics must be exactly {METRICS}")
        return self


class OutputSettings(StrictModel):
    log_dir: Path
    summary_json: Path
    clean_table: Path
    robustness_table: Path
    image_level_summary_archive: Path
    image_level_clean_table_archive: Path
    image_level_robustness_table_archive: Path


class StatisticsConfig(StrictModel):
    schema_version: Literal[2]
    experiment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    inputs: InputSettings
    analysis: AnalysisSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.project_root / path).resolve()


TABLE_FIELDS = (
    "scope",
    "condition_id",
    "family",
    "corruption",
    "severity",
    "estimand",
    "metric",
    "unit",
    "detector_a",
    "detector_b",
    "image_count",
    "patient_group_count",
    "seed_count",
    "confidence_level",
    "detector_a_estimate",
    "detector_a_ci_low",
    "detector_a_ci_high",
    "detector_b_estimate",
    "detector_b_ci_low",
    "detector_b_ci_high",
    "difference_a_minus_b",
    "difference_ci_low",
    "difference_ci_high",
    "bootstrap_resamples",
    "bootstrap_valid_a",
    "bootstrap_valid_b",
    "bootstrap_valid_difference",
    "test_name",
    "permutation_resamples",
    "permutation_valid",
    "p_value_raw",
    "p_value_holm",
    "holm_family_size",
    "effect_size_name",
    "effect_size",
    "effect_size_n",
    "status",
    "reason",
)


def load_statistics_config(path: str | Path) -> StatisticsConfig:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("statistics config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return StatisticsConfig.model_validate(payload)


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    safe = json_number(payload)
    raw = (json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, raw)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in json_number(list(rows)):
                writer.writerow({field: row.get(field) for field in TABLE_FIELDS})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected gzip JSON object: {path}")
    return payload


def load_patient_group_map(
    path: Path, *, image_column: str, patient_group_column: str
) -> dict[str, str]:
    """Load the committed Batch 1 image-to-patient mapping without reconstructing it."""

    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"patient manifest has no header: {path}")
        missing_columns = {image_column, patient_group_column} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"patient manifest lacks configured columns: {sorted(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            image_id = str(row[image_column]).strip()
            patient_group = str(row[patient_group_column]).strip()
            if not image_id or not patient_group:
                raise ValueError(f"empty image or patient group at {path}:{line_number}")
            if image_id in mapping:
                raise ValueError(f"duplicate image ID in patient manifest: {image_id}")
            mapping[image_id] = patient_group
    if not mapping:
        raise ValueError(f"patient manifest contains no rows: {path}")
    return mapping


def _ensure_image_level_archives(config: StatisticsConfig) -> tuple[Path, Path, Path]:
    """Preserve the superseded image-level tables and summary exactly once."""

    current_summary = config.resolve(config.outputs.summary_json)
    current_clean = config.resolve(config.outputs.clean_table)
    current_robustness = config.resolve(config.outputs.robustness_table)
    archive_summary = config.resolve(config.outputs.image_level_summary_archive)
    archive_clean = config.resolve(config.outputs.image_level_clean_table_archive)
    archive_robustness = config.resolve(config.outputs.image_level_robustness_table_archive)
    archives = (archive_summary, archive_clean, archive_robustness)

    if all(path.is_file() for path in archives):
        return archives
    if any(path.exists() for path in archives):
        missing = [str(path) for path in archives if not path.is_file()]
        raise ValueError(f"image-level archive is incomplete; missing {missing}")
    sources = (current_summary, current_clean, current_robustness)
    missing_sources = [str(path) for path in sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            f"cannot archive superseded image-level results; missing {missing_sources}"
        )
    previous = _read_json(current_summary)
    previous_analysis = previous.get("analysis", {})
    if previous_analysis.get("permutation_method") != "paired_image_label_swap":
        raise ValueError("current statistical outputs are not the expected image-level results")
    expected_hashes = {
        current_clean: previous["artifacts"]["clean_table"]["sha256"],
        current_robustness: previous["artifacts"]["robustness_table"]["sha256"],
    }
    for source, expected_hash in expected_hashes.items():
        if sha256_file(source) != expected_hash:
            raise ValueError(f"image-level artifact differs from its frozen summary: {source}")
    for source, destination in zip(sources, archives, strict=True):
        _atomic_bytes(destination, source.read_bytes())
    return archives


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_holm_significance(
    current_rows: Sequence[Mapping[str, Any]],
    archived_path: Path,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare Holm decisions between archived image- and patient-level results."""

    key_fields = ("scope", "condition_id", "estimand", "metric")

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in key_fields)

    def significant(row: Mapping[str, Any]) -> bool:
        value = row.get("p_value_holm")
        return row.get("status") == "complete" and value not in (None, "") and float(value) < alpha

    archived_rows = _read_csv_rows(archived_path)
    archived_by_key = {key(row): row for row in archived_rows}
    current_by_key = {key(row): row for row in current_rows}
    if len(archived_by_key) != len(archived_rows) or len(current_by_key) != len(current_rows):
        raise ValueError("statistical comparison rows do not have unique identities")
    if set(archived_by_key) != set(current_by_key):
        raise ValueError("image- and patient-level statistical tables cover different comparisons")

    became_non_significant: list[dict[str, Any]] = []
    became_significant: list[dict[str, Any]] = []
    for row_key in sorted(current_by_key):
        archived = archived_by_key[row_key]
        current = current_by_key[row_key]
        old_significant = significant(archived)
        new_significant = significant(current)
        if old_significant == new_significant:
            continue
        change = {
            **dict(zip(key_fields, row_key, strict=True)),
            "image_level_p_holm": float(archived["p_value_holm"]),
            "patient_cluster_p_holm": float(current["p_value_holm"]),
            "difference_a_minus_b": float(current["difference_a_minus_b"]),
        }
        (became_non_significant if old_significant else became_significant).append(change)

    return {
        "alpha": alpha,
        "comparison_count": len(current_rows),
        "image_level_significant_count": sum(significant(row) for row in archived_rows),
        "patient_cluster_significant_count": sum(significant(row) for row in current_rows),
        "became_non_significant": became_non_significant,
        "became_significant": became_significant,
        "pattern_changed": bool(became_non_significant or became_significant),
    }


def _deserialize_predictions(payload: Mapping[str, Any]) -> list[ImagePrediction]:
    records = payload.get("predictions")
    if not isinstance(records, list):
        raise ValueError("prediction bundle lacks a predictions list")
    return [
        ImagePrediction(
            image_id=str(item["image_id"]),
            image_size=(int(item["image_size"][0]), int(item["image_size"][1])),
            boxes_xyxy=np.asarray(item["boxes_xyxy"], dtype=np.float64).reshape(-1, 4),
            labels=np.asarray(item["labels"], dtype=np.int64),
            scores=np.asarray(item["scores"], dtype=np.float64),
        )
        for item in records
    ]


def load_coco_targets(
    path: Path, *, selected_image_ids: set[str] | None = None
) -> tuple[list[ImageTarget], dict[int, str]]:
    payload = _read_json(path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not all(isinstance(item, list) for item in (images, annotations, categories)):
        raise ValueError("COCO annotations lack images, annotations, or categories")
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    if not category_names:
        raise ValueError("COCO annotations contain no categories")
    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    targets: list[ImageTarget] = []
    for image in images:
        file_name = str(image["file_name"])
        if selected_image_ids is not None and file_name not in selected_image_ids:
            continue
        image_annotations = annotations_by_image[int(image["id"])]
        boxes = [
            [
                float(item["bbox"][0]),
                float(item["bbox"][1]),
                float(item["bbox"][0] + item["bbox"][2]),
                float(item["bbox"][1] + item["bbox"][3]),
            ]
            for item in image_annotations
        ]
        targets.append(
            ImageTarget(
                image_id=file_name,
                image_size=(int(image["height"]), int(image["width"])),
                boxes_xyxy=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                labels=np.asarray(
                    [int(item["category_id"]) for item in image_annotations], dtype=np.int64
                ),
            )
        )
    if selected_image_ids is not None and {item.image_id for item in targets} != selected_image_ids:
        raise ValueError("selected image IDs are not fully covered by COCO annotations")
    targets.sort(key=lambda item: item.image_id)
    return targets, category_names


def _expected_metric_vector(payload: Mapping[str, Any], *, phase: Literal[5, 6]) -> np.ndarray:
    metrics = payload if phase == 5 else payload["metrics"]
    operating = metrics["operating_point"]["overall"]
    coco = metrics["coco"]
    return np.asarray(
        [
            operating["precision"],
            operating["recall"],
            operating["f1"],
            operating["matched_mean_iou"],
            operating["matched_mean_box_dice"],
            coco["ap50"],
            coco["ap50_95"],
        ],
        dtype=np.float64,
    )


def _build_and_validate_evidence(
    bundle: Mapping[str, Any],
    targets: list[ImageTarget],
    *,
    category_names: dict[int, str],
    evaluation: Any,
    phase: Literal[5, 6],
) -> DetectionEvidence:
    evidence = build_evidence(
        _deserialize_predictions(bundle),
        targets,
        class_ids=tuple(sorted(category_names)),
        score_threshold=evaluation.score_threshold,
        match_iou_threshold=evaluation.match_iou_threshold,
        coco_minimum_score=evaluation.coco_minimum_score,
        max_detections=evaluation.max_detections,
    )
    reconstructed, _ = estimate_pair(
        EvidencePair((evidence,), (evidence,)),
        multiplicities=np.ones(evidence.image_count, dtype=np.int64),
    )
    expected = _expected_metric_vector(bundle, phase=phase)
    if not np.allclose(reconstructed, expected, rtol=0, atol=5e-12, equal_nan=True):
        differences = dict(zip(METRICS, (reconstructed - expected).tolist(), strict=True))
        raise ValueError(f"bundle metrics do not reconstruct exactly: {differences}")
    return evidence


def _validate_upstream(config: StatisticsConfig) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    paths = [config.resolve(path) for path in config.inputs.model_dump().values()]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing statistical input files: {[str(path) for path in missing]}"
        )
    phase5_config_path = config.resolve(config.inputs.phase5_config)
    phase6_config_path = config.resolve(config.inputs.phase6_config)
    phase5 = load_phase5_config(phase5_config_path)
    phase6 = load_robustness_config(phase6_config_path)
    phase5_summary = _read_json(config.resolve(config.inputs.phase5_summary))
    phase6_summary = _read_json(config.resolve(config.inputs.phase6_summary))
    for summary, source, name in (
        (phase5_summary, phase5_config_path, "Phase 5"),
        (phase6_summary, phase6_config_path, "Phase 6"),
    ):
        if summary.get("status") != "complete":
            raise ValueError(f"{name} summary is not complete")
        if summary.get("config_sha256") != sha256_file(source):
            raise ValueError(f"{name} config hash differs from its frozen summary")
    annotation_path = config.resolve(config.inputs.test_annotations)
    if phase5_summary.get("test_annotation_sha256") != sha256_file(annotation_path):
        raise ValueError("test annotations differ from the Phase 5 evaluation source")
    if phase5.evaluation != phase6.evaluation:
        raise ValueError("Phase 5 and Phase 6 evaluation thresholds differ")
    return phase5, phase6, phase5_summary, phase6_summary


def _phase5_pairs(
    config: StatisticsConfig,
    phase5: Any,
    summary: Mapping[str, Any],
    targets: list[ImageTarget],
    category_names: dict[int, str],
) -> EvidencePair:
    by_key: dict[tuple[str, int], DetectionEvidence] = {}
    for run in summary["runs"]:
        detector, seed = str(run["detector"]), int(run["seed"])
        path = config.resolve(Path(run["comparison_row"]["prediction_bundle"]))
        if sha256_file(path) != run["prediction_bundle_sha256"]:
            raise ValueError(f"Phase 5 bundle hash mismatch: {path}")
        bundle = _read_bundle(path)
        if bundle.get("detector") != detector or int(bundle.get("seed")) != seed:
            raise ValueError(f"Phase 5 bundle identity mismatch: {path}")
        by_key[(detector, seed)] = _build_and_validate_evidence(
            bundle,
            targets,
            category_names=category_names,
            evaluation=phase5.evaluation,
            phase=5,
        )
    seeds = tuple(phase5.seeds)
    detector_a = config.analysis.detector_a
    detector_b = config.analysis.detector_b
    expected = {(detector, seed) for detector in (detector_a, detector_b) for seed in seeds}
    if set(by_key) != expected:
        raise ValueError("Phase 5 summary does not contain the configured paired seed grid")
    return EvidencePair(
        detector_a=tuple(by_key[(detector_a, seed)] for seed in seeds),
        detector_b=tuple(by_key[(detector_b, seed)] for seed in seeds),
    )


def _bundle_hash_map(summary: Mapping[str, Any]) -> dict[tuple[str, str], tuple[Path, str]]:
    result: dict[tuple[str, str], tuple[Path, str]] = {}
    for item in summary["prediction_bundles"]:
        key = str(item["detector"]), str(item["condition_id"])
        if key in result:
            raise ValueError(f"duplicate Phase 6 bundle identity: {key}")
        result[key] = Path(item["path"]), str(item["sha256"])
    return result


def _phase6_evidence(
    config: StatisticsConfig,
    phase6: Any,
    bundle_map: Mapping[tuple[str, str], tuple[Path, str]],
    condition_id: str,
    targets: list[ImageTarget],
    category_names: dict[int, str],
) -> EvidencePair:
    evidence: dict[str, DetectionEvidence] = {}
    for detector in (config.analysis.detector_a, config.analysis.detector_b):
        source, expected_hash = bundle_map[(detector, condition_id)]
        path = source if source.is_absolute() else config.resolve(source)
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Phase 6 bundle hash mismatch: {path}")
        payload = _read_bundle(path)
        identity = payload.get("identity", {})
        if (
            payload.get("status") != "complete"
            or identity.get("detector") != detector
            or identity.get("condition", {}).get("condition_id") != condition_id
        ):
            raise ValueError(f"Phase 6 bundle identity mismatch: {path}")
        evidence[detector] = _build_and_validate_evidence(
            payload,
            targets,
            category_names=category_names,
            evaluation=phase6.evaluation,
            phase=6,
        )
    return EvidencePair(
        detector_a=(evidence[config.analysis.detector_a],),
        detector_b=(evidence[config.analysis.detector_b],),
    )


def _decorate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: StatisticsConfig,
    scope: str,
    condition: Mapping[str, Any],
    image_count: int,
    patient_group_count: int,
    seed_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "scope": scope,
            "condition_id": condition["condition_id"],
            "family": condition["family"],
            "corruption": condition["corruption"],
            "severity": condition["severity"],
            **row,
            "detector_a": config.analysis.detector_a,
            "detector_b": config.analysis.detector_b,
            "image_count": image_count,
            "patient_group_count": patient_group_count,
            "seed_count": seed_count,
            "confidence_level": config.analysis.confidence_level,
            "bootstrap_resamples": config.analysis.bootstrap_resamples,
            "test_name": config.analysis.permutation_method,
            "permutation_resamples": config.analysis.permutation_resamples,
            "p_value_holm": row["p_value_raw"],
            "holm_family_size": 1,
        }
        for row in rows
    ]


def apply_grid_holm(rows: list[dict[str, Any]]) -> None:
    """Correct each metric/estimand family over the 35 corrupted conditions."""

    families: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["condition_id"] != "clean" and row["status"] == "complete":
            families[(str(row["metric"]), str(row["estimand"]))].append(index)
    for indices in families.values():
        adjusted = holm_adjust([float(rows[index]["p_value_raw"]) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            rows[index]["p_value_holm"] = value
            rows[index]["holm_family_size"] = len(indices)
    for row in rows:
        if row["condition_id"] != "clean" and row["status"] != "complete":
            row["p_value_holm"] = None
            row["holm_family_size"] = len(families[(str(row["metric"]), str(row["estimand"]))])


def apply_clean_holm(rows: list[dict[str, Any]]) -> None:
    """Correct the seven predeclared clean predictive endpoints as one family."""

    indices = [index for index, row in enumerate(rows) if row["status"] == "complete"]
    adjusted = holm_adjust([float(rows[index]["p_value_raw"]) for index in indices])
    for index, value in zip(indices, adjusted, strict=True):
        rows[index]["p_value_holm"] = value
        rows[index]["holm_family_size"] = len(indices)


def _artifact_hash(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def run_statistics(config: StatisticsConfig) -> dict[str, Any]:
    initialize_reproducibility(config.seed, config.resolve(config.outputs.log_dir))
    phase5, phase6, phase5_summary, phase6_summary = _validate_upstream(config)
    archive_summary, archive_clean, archive_robustness = _ensure_image_level_archives(config)
    annotation_path = config.resolve(config.inputs.test_annotations)
    patient_manifest_path = config.resolve(config.inputs.test_split_manifest)
    patient_group_map = load_patient_group_map(
        patient_manifest_path,
        image_column=config.analysis.manifest_image_column,
        patient_group_column=config.analysis.patient_group_column,
    )
    targets, category_names = load_coco_targets(annotation_path)

    clean_pair = _phase5_pairs(config, phase5, phase5_summary, targets, category_names)
    clean_patient_clusters = build_patient_clusters(
        clean_pair.detector_a[0].image_ids, patient_group_map
    )
    clean_results = analyze_pair(
        clean_pair,
        patient_clusters=clean_patient_clusters,
        base_seed=config.seed,
        comparison_label="phase5-clean-three-seed",
        bootstrap_resamples=config.analysis.bootstrap_resamples,
        permutation_resamples=config.analysis.permutation_resamples,
        confidence_level=config.analysis.confidence_level,
    )
    clean_condition = {
        "condition_id": "clean",
        "family": "clean",
        "corruption": "clean",
        "severity": 0,
    }
    clean_rows = _decorate_rows(
        clean_results["raw"],
        config=config,
        scope="phase5_clean_three_seed",
        condition=clean_condition,
        image_count=clean_pair.image_count,
        patient_group_count=clean_patient_clusters.patient_group_count,
        seed_count=clean_pair.seed_count,
    )
    apply_clean_holm(clean_rows)

    bundle_map = _bundle_hash_map(phase6_summary)
    conditions = expand_conditions(phase6.corruption_config)
    expected_bundle_keys = {
        (detector, condition_id)
        for detector in (config.analysis.detector_a, config.analysis.detector_b)
        for condition_id in ("clean", *(item.condition_id for item in conditions))
    }
    if set(bundle_map) != expected_bundle_keys:
        raise ValueError("Phase 6 summary does not contain the complete paired grid")
    clean_bundle_path = bundle_map[(config.analysis.detector_a, "clean")][0]
    clean_bundle_path = (
        clean_bundle_path if clean_bundle_path.is_absolute() else config.resolve(clean_bundle_path)
    )
    selected_names = {
        item.image_id for item in _deserialize_predictions(_read_bundle(clean_bundle_path))
    }
    subset_targets, subset_category_names = load_coco_targets(
        annotation_path, selected_image_ids=selected_names
    )
    reference_pair = _phase6_evidence(
        config,
        phase6,
        bundle_map,
        "clean",
        subset_targets,
        subset_category_names,
    )
    robustness_patient_clusters = build_patient_clusters(
        reference_pair.detector_a[0].image_ids, patient_group_map
    )

    robustness_rows: list[dict[str, Any]] = []
    raw_clean = analyze_pair(
        reference_pair,
        patient_clusters=robustness_patient_clusters,
        base_seed=config.seed,
        comparison_label="phase6-clean",
        bootstrap_resamples=config.analysis.bootstrap_resamples,
        permutation_resamples=config.analysis.permutation_resamples,
        confidence_level=config.analysis.confidence_level,
    )
    robustness_clean_rows = _decorate_rows(
        raw_clean["raw"],
        config=config,
        scope="phase6_primary_seed_robustness",
        condition=clean_condition,
        image_count=reference_pair.image_count,
        patient_group_count=robustness_patient_clusters.patient_group_count,
        seed_count=1,
    )
    apply_clean_holm(robustness_clean_rows)
    robustness_rows.extend(robustness_clean_rows)
    for condition in conditions:
        condition_payload = {
            "condition_id": condition.condition_id,
            "family": condition.family,
            "corruption": condition.name,
            "severity": condition.severity,
        }
        pair = _phase6_evidence(
            config,
            phase6,
            bundle_map,
            condition.condition_id,
            subset_targets,
            subset_category_names,
        )
        results = analyze_pair(
            pair,
            patient_clusters=robustness_patient_clusters,
            reference_pair=reference_pair,
            base_seed=config.seed,
            comparison_label=f"phase6-{condition.condition_id}",
            bootstrap_resamples=config.analysis.bootstrap_resamples,
            permutation_resamples=config.analysis.permutation_resamples,
            confidence_level=config.analysis.confidence_level,
        )
        for estimand in ("raw", "retention"):
            robustness_rows.extend(
                _decorate_rows(
                    results[estimand],
                    config=config,
                    scope="phase6_primary_seed_robustness",
                    condition=condition_payload,
                    image_count=pair.image_count,
                    patient_group_count=robustness_patient_clusters.patient_group_count,
                    seed_count=1,
                )
            )
    apply_grid_holm(robustness_rows)

    clean_table = config.resolve(config.outputs.clean_table)
    robustness_table = config.resolve(config.outputs.robustness_table)
    _atomic_csv(clean_table, clean_rows)
    _atomic_csv(robustness_table, robustness_rows)
    significance_comparison = {
        "clean": compare_holm_significance(clean_rows, archive_clean),
        "robustness": compare_holm_significance(robustness_rows, archive_robustness),
    }
    source_paths = (
        config.source_path,
        config.project_root / "src" / "stats" / "paired.py",
        config.project_root / "src" / "stats" / "run_statistics.py",
    )
    summary = {
        "schema_version": config.schema_version,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "config_path": config.source_path.relative_to(config.project_root).as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": {
            path.relative_to(config.project_root).as_posix(): sha256_file(path)
            for path in source_paths
        },
        "upstream": {
            "phase5_summary": _artifact_hash(
                config.resolve(config.inputs.phase5_summary), config.project_root
            ),
            "phase6_summary": _artifact_hash(
                config.resolve(config.inputs.phase6_summary), config.project_root
            ),
            "test_annotations": _artifact_hash(annotation_path, config.project_root),
            "test_split_manifest": _artifact_hash(patient_manifest_path, config.project_root),
        },
        "analysis": config.analysis.model_dump(mode="json"),
        "inference_unit": {
            "bootstrap": (
                "Matched NIH patient groups are resampled with replacement; every sampled "
                "group contributes all of its observed images with one shared multiplicity. "
                "The clean analysis also resamples the three paired training seeds."
            ),
            "permutation": (
                "Detector labels are swapped independently by NIH patient group, so every "
                "image from a patient moves together. Swaps are shared across paired seeds "
                "and, for retention, across clean and corrupted evidence."
            ),
            "effect_size": (
                "The unstandardized paired aggregate difference (A minus B) is the effect "
                "size, accompanied by its patient-cluster bootstrap interval. The former "
                "image-jackknife Cohen's d is archived but not carried forward because its "
                "standardized interpretation is not established for unequal patient clusters."
            ),
        },
        "mcnemar": {
            "applied": False,
            "reason": (
                "The benchmark endpoint is object detection, not one binary decision per "
                "independent image. Collapsing multiple targets and negative-image false "
                "positives into correct/incorrect would discard outcome structure; target-"
                "level decisions are also nested within images."
            ),
        },
        "multiple_comparisons": {
            "method": config.analysis.multiple_comparison_correction,
            "clean_scope": config.analysis.clean_correction_scope,
            "corruption_scope": config.analysis.corruption_correction_scope,
            "pointwise_confidence_intervals_are_multiplicity_adjusted": False,
        },
        "clean": {
            "image_count": clean_pair.image_count,
            "patient_group_count": clean_patient_clusters.patient_group_count,
            "seed_count": clean_pair.seed_count,
            "comparison_count": len(clean_rows),
            "results": clean_rows,
        },
        "robustness": {
            "image_count": reference_pair.image_count,
            "patient_group_count": robustness_patient_clusters.patient_group_count,
            "seed_count": 1,
            "condition_count_including_clean": 36,
            "corrupted_condition_count": 35,
            "comparison_count": len(robustness_rows),
            "holm_scope": config.analysis.corruption_correction_scope,
            "results": robustness_rows,
        },
        "superseded_image_level_results": {
            "reason": (
                "Image-level resampling treated repeated exams from the same NIH patient as "
                "independent. The archived outputs are retained for audit only; patient-"
                "cluster inference is primary."
            ),
            "summary": _artifact_hash(archive_summary, config.project_root),
            "clean_table": _artifact_hash(archive_clean, config.project_root),
            "robustness_table": _artifact_hash(archive_robustness, config.project_root),
        },
        "holm_significance_comparison": significance_comparison,
        "artifacts": {
            "clean_table": _artifact_hash(clean_table, config.project_root),
            "robustness_table": _artifact_hash(robustness_table, config.project_root),
        },
    }
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_json(summary_path, summary)
    return summary


def preflight(config: StatisticsConfig) -> dict[str, Any]:
    phase5, phase6, phase5_summary, phase6_summary = _validate_upstream(config)
    bundle_map = _bundle_hash_map(phase6_summary)
    conditions = expand_conditions(phase6.corruption_config)
    expected = 2 * (1 + len(conditions))
    if len(bundle_map) != expected:
        raise ValueError(f"expected {expected} Phase 6 bundles, found {len(bundle_map)}")
    patient_group_map = load_patient_group_map(
        config.resolve(config.inputs.test_split_manifest),
        image_column=config.analysis.manifest_image_column,
        patient_group_column=config.analysis.patient_group_column,
    )
    targets, _ = load_coco_targets(config.resolve(config.inputs.test_annotations))
    clean_clusters = build_patient_clusters(
        tuple(sorted(target.image_id for target in targets)), patient_group_map
    )
    clean_bundle_path = bundle_map[(config.analysis.detector_a, "clean")][0]
    clean_bundle_path = (
        clean_bundle_path if clean_bundle_path.is_absolute() else config.resolve(clean_bundle_path)
    )
    selected_names = {
        item.image_id for item in _deserialize_predictions(_read_bundle(clean_bundle_path))
    }
    robustness_clusters = build_patient_clusters(tuple(sorted(selected_names)), patient_group_map)
    return {
        "status": "ready",
        "phase5_seed_count": len(phase5.seeds),
        "phase5_bundle_count": len(phase5_summary["runs"]),
        "clean_image_count": len(targets),
        "clean_patient_group_count": clean_clusters.patient_group_count,
        "phase6_bundle_count": len(bundle_map),
        "robustness_image_count": len(selected_names),
        "robustness_patient_group_count": robustness_clusters.patient_group_count,
        "corrupted_condition_count": len(conditions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/statistics.yaml"))
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_statistics_config(args.config)
    result = preflight(config) if args.mode == "preflight" else run_statistics(config)
    if args.mode == "preflight":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": result["status"], "artifacts": result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
