"""Generate publication-safe aggregate RSNA cohort characteristics.

Only selected DICOM header fields are read. The command never writes study-,
patient-, or identifier-level rows; its two outputs are aggregate-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.prepare import load_and_audit_records, load_dataset_config


class StrictModel(BaseModel):
    """Reject undeclared config keys and runtime mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InputSettings(StrictModel):
    """Immutable dataset, split, and DICOM-root inputs."""

    dataset_config: Path
    expected_dataset_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_json: Path
    expected_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dicom_root_environment_variable: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    split_manifests: dict[str, Path] = Field(min_length=1)
    expected_split_sha256: dict[str, str] = Field(min_length=1)
    output_split_names: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_split_keys(self) -> InputSettings:
        """Require one hash and output label for every split manifest."""

        manifest_keys = set(self.split_manifests)
        if set(self.expected_split_sha256) != manifest_keys:
            raise ValueError("expected_split_sha256 keys must match split_manifests")
        if set(self.output_split_names) != manifest_keys:
            raise ValueError("output_split_names keys must match split_manifests")
        if len(set(self.output_split_names.values())) != len(manifest_keys):
            raise ValueError("output split names must be unique")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in self.expected_split_sha256.values()
        ):
            raise ValueError(
                "every expected split SHA-256 must have 64 lowercase hexadecimal digits"
            )
        return self


class ManifestColumns(StrictModel):
    """Column names in the frozen patient-safe split manifests."""

    split: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    patient_group: str = Field(min_length=1)
    study_stratum: str = Field(min_length=1)
    positive: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    box_count: str = Field(min_length=1)


class ExpectedCounts(StrictModel):
    """Expected immutable cohort counts for one scope."""

    studies: int = Field(ge=1)
    patient_groups: int = Field(ge=1)
    opacity_positive_studies: int = Field(ge=0)
    bounding_boxes: int = Field(ge=0)


class ExpectedCohort(StrictModel):
    """Expected split and selected-total cohort counts."""

    total: ExpectedCounts
    splits: dict[str, ExpectedCounts] = Field(min_length=1)


class AgeSettings(StrictModel):
    """PatientAge syntax, unit conversion, and plausibility contract."""

    tag: str = Field(min_length=1)
    expected_vr: str = Field(min_length=2, max_length=2)
    conformant_pattern: str = Field(min_length=1)
    numeric_without_unit_pattern: str = Field(min_length=1)
    numeric_without_unit_policy: Literal["assume_years_with_caveat"]
    minimum_years: float = Field(ge=0)
    maximum_years: float = Field(gt=0)
    unit_denominators_per_year: dict[str, float] = Field(min_length=1)
    percentile_method: Literal[
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]

    @model_validator(mode="after")
    def validate_age_contract(self) -> AgeSettings:
        """Validate regex groups, range, and unit conversion factors."""

        if self.maximum_years <= self.minimum_years:
            raise ValueError("maximum_years must exceed minimum_years")
        pattern = re.compile(self.conformant_pattern)
        if not {"value", "unit"} <= set(pattern.groupindex):
            raise ValueError("conformant_pattern must define value and unit groups")
        re.compile(self.numeric_without_unit_pattern)
        if set(self.unit_denominators_per_year) != {"D", "W", "M", "Y"}:
            raise ValueError("age unit denominators must define D, W, M, and Y")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.unit_denominators_per_year.values()
        ):
            raise ValueError("age unit denominators must be finite and positive")
        return self


class CategoricalTagSettings(StrictModel):
    """One categorical DICOM tag and its expected categories."""

    tag: str = Field(min_length=1)
    expected_vr: str = Field(min_length=2, max_length=2)
    categories: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_categories(self) -> CategoricalTagSettings:
        """Require uppercase unique non-empty category labels."""

        if any(not value or value != value.upper() for value in self.categories):
            raise ValueError("categorical tag values must be non-empty uppercase strings")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categorical tag values must be unique")
        return self


class MetadataSettings(StrictModel):
    """Selected publication-safe DICOM header fields."""

    age: AgeSettings
    sex: CategoricalTagSettings
    projection: CategoricalTagSettings


class OutputSettings(StrictModel):
    """Aggregate-only output paths."""

    aggregate_table: Path
    summary_json: Path


class CohortCharacteristicsConfig(StrictModel):
    """Strict Batch 44 aggregate-generation contract."""

    schema_version: Literal[1]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    inputs: InputSettings
    manifest_columns: ManifestColumns
    expected_cohort: ExpectedCohort
    metadata: MetadataSettings
    outputs: OutputSettings
    project_root: Path = Field(exclude=True)
    source_path: Path = Field(exclude=True)

    @model_validator(mode="after")
    def validate_cohort_keys(self) -> CohortCharacteristicsConfig:
        """Require expected counts for exactly the configured manifests."""

        if set(self.expected_cohort.splits) != set(self.inputs.split_manifests):
            raise ValueError("expected cohort split keys must match split manifests")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve one repository path against the project root."""

        return path if path.is_absolute() else (self.project_root / path).resolve()


@dataclass(frozen=True, slots=True)
class AgeObservation:
    """Validated representation of one PatientAge value."""

    years: float | None
    syntax: str
    encoded_unit: str | None
    tag_missing: bool
    unit_missing: bool
    out_of_range: bool


@dataclass(slots=True)
class AggregateAccumulator:
    """In-memory aggregate state; identifiers are never serialized."""

    studies: int = 0
    patient_groups: set[str] = field(default_factory=set)
    opacity_positive_studies: int = 0
    bounding_boxes: int = 0
    age_years: list[float] = field(default_factory=list)
    age_syntax: Counter[str] = field(default_factory=Counter)
    age_units: Counter[str] = field(default_factory=Counter)
    age_tag_missing: int = 0
    age_unit_missing: int = 0
    age_out_of_range: int = 0
    age_unusable: int = 0
    sex: Counter[str] = field(default_factory=Counter)
    projection: Counter[str] = field(default_factory=Counter)
    tag_vrs: dict[str, Counter[str]] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    """Return a lowercase streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cohort_characteristics_config(path: str | Path) -> CohortCharacteristicsConfig:
    """Load the strict cohort-characteristics YAML."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cohort-characteristics config must contain a mapping")
    payload["source_path"] = source
    payload["project_root"] = source.parent.parent.resolve()
    return CohortCharacteristicsConfig.model_validate(payload)


def resolve_dicom_root(
    config: CohortCharacteristicsConfig,
    dataset_config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, str]:
    """Resolve the original DICOM root from an environment override or dataset config."""

    environment = os.environ if environ is None else environ
    variable = config.inputs.dicom_root_environment_variable
    override = environment.get(variable, "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        source = f"environment:{variable}"
    else:
        dataset = dataset_config.get("dataset")
        paths = dataset.get("paths") if isinstance(dataset, Mapping) else None
        configured = paths.get("source_images_dir") if isinstance(paths, Mapping) else None
        if not isinstance(configured, str) or not configured.strip():
            raise FileNotFoundError(f"Original RSNA DICOM root is not configured. Set {variable}.")
        value = Path(configured).expanduser()
        root = value.resolve() if value.is_absolute() else (config.project_root / value).resolve()
        source = "dataset_config:dataset.paths.source_images_dir"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Original RSNA DICOM root is unavailable via {source}. Set {variable} "
            "to the directory containing the Stage 2 training DICOMs."
        )
    return root, source


def parse_patient_age(raw_value: Any, settings: AgeSettings) -> AgeObservation:
    """Parse DICOM PatientAge, retaining syntax/unit and plausibility diagnostics."""

    text = "" if raw_value is None else str(raw_value).strip().upper()
    if not text:
        return AgeObservation(None, "missing", None, True, True, False)

    conformant = re.fullmatch(settings.conformant_pattern, text)
    if conformant is not None:
        value = int(conformant.group("value"))
        unit = conformant.group("unit")
        years = value / settings.unit_denominators_per_year[unit]
        out_of_range = not settings.minimum_years <= years <= settings.maximum_years
        return AgeObservation(
            None if out_of_range else years,
            "dicom_as_conformant",
            unit,
            False,
            False,
            out_of_range,
        )

    if re.fullmatch(settings.numeric_without_unit_pattern, text) is not None:
        years = float(int(text))
        out_of_range = not settings.minimum_years <= years <= settings.maximum_years
        return AgeObservation(
            None if out_of_range else years,
            "numeric_without_unit_assumed_years",
            None,
            False,
            True,
            out_of_range,
        )

    return AgeObservation(None, "invalid_other", None, False, True, False)


def _safe_relative_name(value: str, *, expected_suffix: str) -> str:
    """Reject absolute, nested, or extension-mismatched DICOM manifest paths."""

    candidate = Path(value)
    if not value or candidate.is_absolute() or candidate.name != value:
        raise ValueError("source_file entries must be safe leaf names")
    if candidate.suffix.lower() != expected_suffix.lower():
        raise ValueError("source_file extension does not match dataset.image.source_extension")
    return value


def _read_manifest(
    path: Path,
    *,
    split_name: str,
    columns: ManifestColumns,
) -> tuple[dict[str, str], ...]:
    """Read one immutable split manifest and validate its row-level shape."""

    required = {
        columns.split,
        columns.image_id,
        columns.patient_group,
        columns.study_stratum,
        columns.positive,
        columns.source_file,
        columns.box_count,
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"split manifest lacks configured columns: {sorted(missing)}")
        rows = tuple(dict(row) for row in reader)
    if any(row[columns.split] != split_name for row in rows):
        raise ValueError(f"manifest split column does not match configured split {split_name}")
    return rows


def _resolved_dataset_config(
    config: CohortCharacteristicsConfig,
    dataset_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Make metadata input paths absolute for the existing dataset auditor."""

    resolved = copy.deepcopy(dict(dataset_config))
    dataset = resolved.get("dataset")
    paths = dataset.get("paths") if isinstance(dataset, MutableMapping) else None
    if not isinstance(paths, MutableMapping):
        raise ValueError("dataset.paths must be a mapping")
    for key in ("labels_csv", "class_info_csv", "mapping_json"):
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"dataset.paths.{key} must be configured")
        path = Path(value)
        paths[key] = str(path if path.is_absolute() else config.project_root / path)
    return resolved


def _immutable_gate(
    config: CohortCharacteristicsConfig,
    dataset_config: Mapping[str, Any],
) -> tuple[dict[str, tuple[dict[str, str], ...]], dict[str, Any]]:
    """Verify split bytes, counts, disjointness, and official patient mapping."""

    dataset_path = config.resolve(config.inputs.dataset_config)
    audit_path = config.resolve(config.inputs.audit_json)
    observed_dataset_hash = sha256_file(dataset_path)
    observed_audit_hash = sha256_file(audit_path)
    if observed_dataset_hash != config.inputs.expected_dataset_config_sha256:
        raise ValueError("dataset config SHA-256 differs from the immutable cohort contract")
    if observed_audit_hash != config.inputs.expected_audit_sha256:
        raise ValueError("dataset audit SHA-256 differs from the immutable cohort contract")

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audited_splits = audit_payload.get("splits")
    if not isinstance(audited_splits, Mapping):
        raise ValueError("dataset audit lacks split counts")

    resolved_dataset = _resolved_dataset_config(config, dataset_config)
    resolved_dataset_section = resolved_dataset.get("dataset")
    resolved_paths = (
        resolved_dataset_section.get("paths")
        if isinstance(resolved_dataset_section, Mapping)
        else None
    )
    audited_inputs = audit_payload.get("input_files")
    if not isinstance(resolved_paths, Mapping) or not isinstance(audited_inputs, Mapping):
        raise ValueError("dataset config/audit lacks source metadata paths or hashes")
    source_metadata_hashes: dict[str, str] = {}
    for key in ("labels_csv", "class_info_csv", "mapping_json"):
        path_value = resolved_paths.get(key)
        audited_value = audited_inputs.get(key)
        if not isinstance(path_value, str) or not isinstance(audited_value, Mapping):
            raise ValueError(f"source metadata contract is missing {key}")
        actual = sha256_file(Path(path_value))
        expected = audited_value.get("sha256")
        if actual != expected:
            raise ValueError(f"source metadata SHA-256 differs from the audit for {key}")
        source_metadata_hashes[key] = actual
    supplements = (
        resolved_dataset_section.get("official_supplements")
        if isinstance(resolved_dataset_section, Mapping)
        else None
    )
    expected_mapping_hash = (
        supplements.get("mapping_sha256") if isinstance(supplements, Mapping) else None
    )
    if source_metadata_hashes["mapping_json"] != expected_mapping_hash:
        raise ValueError("official RSNA patient mapping SHA-256 is not the configured digest")

    columns = config.manifest_columns
    manifests: dict[str, tuple[dict[str, str], ...]] = {}
    split_hashes: dict[str, str] = {}
    all_image_ids: set[str] = set()
    all_source_files: set[str] = set()
    patient_sets: dict[str, set[str]] = {}
    dataset = dataset_config.get("dataset")
    image = dataset.get("image") if isinstance(dataset, Mapping) else None
    source_extension = image.get("source_extension") if isinstance(image, Mapping) else None
    if not isinstance(source_extension, str) or not source_extension:
        raise ValueError("dataset.image.source_extension must be configured")

    audited_key_map = {"train": "train", "val": "val", "test": "test"}
    for split_name, relative_path in config.inputs.split_manifests.items():
        path = config.resolve(relative_path)
        digest = sha256_file(path)
        if digest != config.inputs.expected_split_sha256[split_name]:
            raise ValueError(f"immutable split SHA-256 mismatch for {split_name}")
        rows = _read_manifest(path, split_name=split_name, columns=columns)
        image_ids = [row[columns.image_id] for row in rows]
        source_files = [
            _safe_relative_name(row[columns.source_file], expected_suffix=source_extension)
            for row in rows
        ]
        if len(set(image_ids)) != len(rows) or len(set(source_files)) != len(rows):
            raise ValueError(f"split {split_name} contains duplicate study/source identifiers")
        if all_image_ids.intersection(image_ids) or all_source_files.intersection(source_files):
            raise ValueError("study/source membership overlaps between immutable splits")
        all_image_ids.update(image_ids)
        all_source_files.update(source_files)
        patients = {row[columns.patient_group] for row in rows}
        if "" in patients:
            raise ValueError(f"split {split_name} contains a missing patient group")
        patient_sets[split_name] = patients
        manifests[split_name] = rows
        split_hashes[split_name] = digest

        observed = ExpectedCounts(
            studies=len(rows),
            patient_groups=len(patients),
            opacity_positive_studies=sum(int(row[columns.positive]) for row in rows),
            bounding_boxes=sum(int(row[columns.box_count]) for row in rows),
        )
        expected = config.expected_cohort.splits[split_name]
        if observed != expected:
            raise ValueError(f"split {split_name} counts differ from the immutable contract")
        audited = audited_splits.get(audited_key_map.get(split_name, split_name))
        if not isinstance(audited, Mapping) or (
            int(audited.get("images", -1)) != expected.studies
            or int(audited.get("groups", -1)) != expected.patient_groups
            or int(audited.get("positive_images", -1)) != expected.opacity_positive_studies
            or int(audited.get("boxes", -1)) != expected.bounding_boxes
        ):
            raise ValueError(f"dataset audit counts disagree for split {split_name}")

    overlap_counts: dict[str, int] = {}
    split_names = list(patient_sets)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = len(patient_sets[first] & patient_sets[second])
            overlap_counts[f"{first}__{second}"] = overlap
            if overlap:
                raise ValueError("patient groups overlap between immutable splits")

    audit_result = load_and_audit_records(resolved_dataset)
    errors = [issue for issue in audit_result.issues if issue.severity == "error"]
    if errors:
        raise ValueError("official source metadata audit contains errors")
    records = {record.exam_id: record for record in audit_result.valid_records}
    mapping_matches = 0
    for split_name, rows in manifests.items():
        for row in rows:
            image_id = row[columns.image_id]
            record = records.get(image_id)
            if record is None:
                raise ValueError("selected study is absent from the official source metadata")
            expected_source = f"{image_id}{source_extension}"
            if (
                record.nih_patient_id != row[columns.patient_group]
                or record.study_stratum != row[columns.study_stratum]
                or int(record.is_positive) != int(row[columns.positive])
                or len(record.boxes) != int(row[columns.box_count])
                or row[columns.source_file] != expected_source
            ):
                raise ValueError(
                    f"selected manifest metadata disagrees with official mapping in {split_name}"
                )
            mapping_matches += 1

    total = ExpectedCounts(
        studies=sum(len(rows) for rows in manifests.values()),
        patient_groups=len(set().union(*patient_sets.values())),
        opacity_positive_studies=sum(
            int(row[columns.positive]) for rows in manifests.values() for row in rows
        ),
        bounding_boxes=sum(
            int(row[columns.box_count]) for rows in manifests.values() for row in rows
        ),
    )
    if total != config.expected_cohort.total:
        raise ValueError("selected total differs from the immutable cohort contract")
    return manifests, {
        "passed": True,
        "dataset_config_sha256": observed_dataset_hash,
        "audit_sha256": observed_audit_hash,
        "source_metadata_sha256": source_metadata_hashes,
        "split_sha256": split_hashes,
        "patient_group_overlap_counts": overlap_counts,
        "selected_studies_matching_official_patient_mapping": mapping_matches,
        "source_metadata_audit_error_count": len(errors),
        "observed_total": total.model_dump(),
    }


def _dicom_value_and_vr(dataset: Any, tag: str) -> tuple[str, str]:
    """Return a normalized selected DICOM value and its actual VR."""

    element = dataset.data_element(tag)
    if element is None:
        return "", "missing"
    value = "" if element.value is None else str(element.value).strip().upper()
    return value, str(element.VR).upper()


def _update_accumulator(
    aggregate: AggregateAccumulator,
    *,
    row: Mapping[str, str],
    columns: ManifestColumns,
    age: AgeObservation,
    sex: str,
    projection: str,
    tag_vrs: Mapping[str, str],
) -> None:
    """Add one validated study to an aggregate accumulator."""

    aggregate.studies += 1
    aggregate.patient_groups.add(row[columns.patient_group])
    aggregate.opacity_positive_studies += int(row[columns.positive])
    aggregate.bounding_boxes += int(row[columns.box_count])
    aggregate.age_syntax[age.syntax] += 1
    if age.encoded_unit is not None:
        aggregate.age_units[age.encoded_unit] += 1
    if age.years is not None:
        aggregate.age_years.append(age.years)
    else:
        aggregate.age_unusable += 1
    aggregate.age_tag_missing += int(age.tag_missing)
    aggregate.age_unit_missing += int(age.unit_missing)
    aggregate.age_out_of_range += int(age.out_of_range)
    aggregate.sex[sex or "missing"] += 1
    aggregate.projection[projection or "missing"] += 1
    for tag, vr in tag_vrs.items():
        aggregate.tag_vrs.setdefault(tag, Counter())[vr] += 1


def _percent(count: int, total: int) -> float:
    """Return an all-study percentage with deterministic six-decimal precision."""

    return round(100.0 * count / total, 6) if total else 0.0


def _aggregate_row(
    name: str,
    aggregate: AggregateAccumulator,
    metadata: MetadataSettings,
) -> dict[str, Any]:
    """Convert one accumulator into a publication-safe flat row."""

    ages = np.asarray(aggregate.age_years, dtype=np.float64)
    if ages.size:
        q1, median, q3 = np.percentile(
            ages,
            (25, 50, 75),
            method=metadata.age.percentile_method,
        )
        age_values: dict[str, Any] = {
            "age_mean_years": round(float(np.mean(ages)), 6),
            "age_sd_years": round(float(np.std(ages, ddof=1)), 6) if ages.size > 1 else "",
            "age_median_years": round(float(median), 6),
            "age_q1_years": round(float(q1), 6),
            "age_q3_years": round(float(q3), 6),
            "age_min_years": round(float(np.min(ages)), 6),
            "age_max_years": round(float(np.max(ages)), 6),
        }
    else:
        age_values = {
            key: ""
            for key in (
                "age_mean_years",
                "age_sd_years",
                "age_median_years",
                "age_q1_years",
                "age_q3_years",
                "age_min_years",
                "age_max_years",
            )
        }
    row: dict[str, Any] = {
        "split": name,
        "studies": aggregate.studies,
        "patient_groups": len(aggregate.patient_groups),
        "opacity_positive_studies": aggregate.opacity_positive_studies,
        "bounding_boxes": aggregate.bounding_boxes,
        "age_summary_n": len(aggregate.age_years),
        **age_values,
        "age_tag_missing_count": aggregate.age_tag_missing,
        "age_tag_missing_percent": _percent(aggregate.age_tag_missing, aggregate.studies),
        "age_unusable_count": aggregate.age_unusable,
        "age_unusable_percent": _percent(aggregate.age_unusable, aggregate.studies),
        "age_conformant_as_count": aggregate.age_syntax["dicom_as_conformant"],
        "age_numeric_without_unit_count": aggregate.age_syntax[
            "numeric_without_unit_assumed_years"
        ],
        "age_unit_missing_count": aggregate.age_unit_missing,
        "age_unit_missing_percent": _percent(aggregate.age_unit_missing, aggregate.studies),
        "age_out_of_range_count": aggregate.age_out_of_range,
        "age_out_of_range_percent": _percent(aggregate.age_out_of_range, aggregate.studies),
        "age_invalid_other_count": aggregate.age_syntax["invalid_other"],
    }
    for category in metadata.sex.categories:
        count = aggregate.sex[category]
        key = category.lower()
        row[f"sex_{key}_count"] = count
        row[f"sex_{key}_percent"] = _percent(count, aggregate.studies)
    sex_other = sum(
        count
        for value, count in aggregate.sex.items()
        if value not in {*metadata.sex.categories, "missing"}
    )
    row["sex_other_count"] = sex_other
    row["sex_other_percent"] = _percent(sex_other, aggregate.studies)
    row["sex_missing_count"] = aggregate.sex["missing"]
    row["sex_missing_percent"] = _percent(aggregate.sex["missing"], aggregate.studies)
    for category in metadata.projection.categories:
        count = aggregate.projection[category]
        key = category.lower()
        row[f"projection_{key}_count"] = count
        row[f"projection_{key}_percent"] = _percent(count, aggregate.studies)
    projection_other = sum(
        count
        for value, count in aggregate.projection.items()
        if value not in {*metadata.projection.categories, "missing"}
    )
    row["projection_other_count"] = projection_other
    row["projection_other_percent"] = _percent(projection_other, aggregate.studies)
    row["projection_missing_count"] = aggregate.projection["missing"]
    row["projection_missing_percent"] = _percent(aggregate.projection["missing"], aggregate.studies)
    return row


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic LF CSV atomically."""

    if not rows:
        raise ValueError("aggregate cohort table cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic LF JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def generate_cohort_characteristics(
    config: CohortCharacteristicsConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate inputs, inspect selected headers, and write aggregate-only artifacts."""

    import pydicom

    dataset_path = config.resolve(config.inputs.dataset_config)
    dataset_config = load_dataset_config(dataset_path)
    manifests, prerequisite = _immutable_gate(config, dataset_config)
    dicom_root, root_source = resolve_dicom_root(config, dataset_config, environ=environ)

    metadata = config.metadata
    tags = (metadata.age.tag, metadata.sex.tag, metadata.projection.tag)
    expected_vrs = {
        metadata.age.tag: metadata.age.expected_vr,
        metadata.sex.tag: metadata.sex.expected_vr,
        metadata.projection.tag: metadata.projection.expected_vr,
    }
    columns = config.manifest_columns
    total = AggregateAccumulator()
    by_split: dict[str, AggregateAccumulator] = {}
    sex_by_patient_group: dict[str, set[str]] = {}
    metadata_fingerprint = hashlib.sha256()
    missing_dicom_count = 0

    for split_name, rows in manifests.items():
        split_aggregate = AggregateAccumulator()
        by_split[split_name] = split_aggregate
        for row in rows:
            source_name = row[columns.source_file]
            dicom_path = dicom_root / source_name
            if not dicom_path.is_file():
                missing_dicom_count += 1
                continue
            dataset = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=True,
                specific_tags=list(tags),
            )
            values_and_vrs = {tag: _dicom_value_and_vr(dataset, tag) for tag in tags}
            age_value, _ = values_and_vrs[metadata.age.tag]
            sex_value, _ = values_and_vrs[metadata.sex.tag]
            projection_value, _ = values_and_vrs[metadata.projection.tag]
            vrs = {tag: value[1] for tag, value in values_and_vrs.items()}
            age = parse_patient_age(age_value, metadata.age)
            if sex_value:
                sex_by_patient_group.setdefault(row[columns.patient_group], set()).add(sex_value)
            for aggregate in (split_aggregate, total):
                _update_accumulator(
                    aggregate,
                    row=row,
                    columns=columns,
                    age=age,
                    sex=sex_value,
                    projection=projection_value,
                    tag_vrs=vrs,
                )
            fingerprint_record = {
                "split": split_name,
                "source_file": source_name,
                "age": age_value,
                "sex": sex_value,
                "projection": projection_value,
                "vrs": vrs,
            }
            metadata_fingerprint.update(
                (json.dumps(fingerprint_record, sort_keys=True) + "\n").encode("utf-8")
            )

    if missing_dicom_count:
        variable = config.inputs.dicom_root_environment_variable
        raise FileNotFoundError(
            f"The resolved DICOM root is missing {missing_dicom_count} selected studies. "
            f"Set {variable} to the complete Stage 2 training DICOM directory."
        )
    if total.studies != config.expected_cohort.total.studies:
        raise ValueError("DICOM header count differs from the immutable selected cohort")

    vr_mismatches = {
        tag: {
            vr: count
            for vr, count in total.tag_vrs[tag].items()
            if vr not in {expected_vrs[tag], "missing"}
        }
        for tag in tags
    }
    if any(vr_mismatches.values()):
        raise ValueError("selected DICOM metadata VRs differ from the configured contract")

    rows = [
        _aggregate_row(config.inputs.output_split_names[name], by_split[name], metadata)
        for name in config.inputs.split_manifests
    ]
    rows.append(_aggregate_row("total", total, metadata))
    table_path = config.resolve(config.outputs.aggregate_table)
    summary_path = config.resolve(config.outputs.summary_json)
    _atomic_csv(table_path, rows)

    age_units = dict(sorted(total.age_units.items()))
    age_syntax = dict(sorted(total.age_syntax.items()))
    sex_categories = dict(sorted(total.sex.items()))
    projection_categories = dict(sorted(total.projection.items()))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": config.analysis_id,
        "status": "complete",
        "privacy": {
            "aggregate_outputs_only": True,
            "dicom_pixels_read": False,
            "identifiers_or_patient_rows_written": False,
            "selected_tags_only": list(tags),
        },
        "prerequisite_gate": prerequisite,
        "input_contract": {
            "config_path": config.source_path.relative_to(config.project_root).as_posix(),
            "config_sha256": sha256_file(config.source_path),
            "dicom_root_resolution": root_source,
            "selected_dicom_count": total.studies,
            "selected_dicom_metadata_sha256": metadata_fingerprint.hexdigest(),
        },
        "age_validation": {
            "tag": metadata.age.tag,
            "expected_vr": metadata.age.expected_vr,
            "observed_vr_counts": dict(sorted(total.tag_vrs[metadata.age.tag].items())),
            "syntax_counts": age_syntax,
            "encoded_unit_counts": age_units,
            "tag_missing_count": total.age_tag_missing,
            "unit_missing_count": total.age_unit_missing,
            "out_of_range_count": total.age_out_of_range,
            "unusable_count": total.age_unusable,
            "numeric_without_unit_policy": metadata.age.numeric_without_unit_policy,
            "plausible_year_range": [
                metadata.age.minimum_years,
                metadata.age.maximum_years,
            ],
            "interpretation": (
                "Numeric-only AS values are reported as nominal years with an explicit "
                "nonconformant-unit caveat; out-of-range values are excluded from age summaries."
            ),
        },
        "sex_validation": {
            "tag": metadata.sex.tag,
            "expected_vr": metadata.sex.expected_vr,
            "observed_vr_counts": dict(sorted(total.tag_vrs[metadata.sex.tag].items())),
            "configured_categories": list(metadata.sex.categories),
            "observed_category_counts": sex_categories,
            "missing_count": total.sex["missing"],
            "patient_groups_with_conflicting_nonmissing_values": sum(
                len(values) > 1 for values in sex_by_patient_group.values()
            ),
        },
        "projection_validation": {
            "tag": metadata.projection.tag,
            "expected_vr": metadata.projection.expected_vr,
            "observed_vr_counts": dict(sorted(total.tag_vrs[metadata.projection.tag].items())),
            "configured_categories": list(metadata.projection.categories),
            "observed_category_counts": projection_categories,
            "missing_count": total.projection["missing"],
        },
        "aggregates": rows,
        "artifacts": {
            "aggregate_table": config.outputs.aggregate_table.as_posix(),
            "aggregate_table_sha256": sha256_file(table_path),
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate aggregate-only RSNA cohort characteristics from DICOM headers."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cohort_characteristics.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    config = load_cohort_characteristics_config(args.config)
    summary = generate_cohort_characteristics(config)
    total = summary["aggregates"][-1]
    print(
        "Cohort characteristics complete: "
        f"{total['studies']} studies, {total['patient_groups']} patient groups, "
        f"{total['opacity_positive_studies']} opacity-positive studies, "
        f"{total['bounding_boxes']} boxes."
    )
    print(f"Aggregate table: {config.outputs.aggregate_table.as_posix()}")
    print(f"Aggregate summary: {config.outputs.summary_json.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
