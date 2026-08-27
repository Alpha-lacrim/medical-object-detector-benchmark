"""Prepare the RSNA Stage 2 annotations and patient-safe benchmark splits.

The module deliberately keeps image decoding optional.  Mapping and CSV metadata are
sufficient to audit annotations, construct patient-disjoint splits, and write canonical
COCO JSON.  When source DICOM files are present, the same command can materialize the
selected images as 8-bit PNG files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """One valid absolute-coordinate bounding box."""

    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""

        return self.width * self.height


@dataclass(frozen=True, slots=True)
class ExamRecord:
    """Audited metadata and boxes for one RSNA image/exam."""

    exam_id: str
    nih_patient_id: str
    study_stratum: str
    is_positive: bool
    boxes: tuple[BoundingBox, ...]
    valid: bool


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """A machine-readable annotation or metadata audit finding."""

    code: str
    severity: Severity
    message: str
    exam_id: str | None = None
    row_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable representation."""

        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Complete metadata audit result before selection and splitting."""

    records: tuple[ExamRecord, ...]
    issues: tuple[AuditIssue, ...]
    input_counts: dict[str, int]

    @property
    def valid_records(self) -> tuple[ExamRecord, ...]:
        """Return only records that are safe to export."""

        return tuple(record for record in self.records if record.valid)


@dataclass(frozen=True, slots=True)
class PatientGroup:
    """An indivisible patient group used during selection and splitting."""

    group_id: str
    records: tuple[ExamRecord, ...]
    stratum_counts: dict[str, int]

    @property
    def size(self) -> int:
        """Return the number of images in the patient group."""

        return len(self.records)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Validate and return a configuration mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, location: str) -> str:
    """Read a required, non-empty string from a mapping."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _require_int(mapping: Mapping[str, Any], key: str, location: str) -> int:
    """Read a required positive integer from a mapping."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}.{key} must be a positive integer")
    return value


def _require_seed(mapping: Mapping[str, Any], key: str, location: str) -> int:
    """Read a required non-negative deterministic seed from a mapping."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}.{key} must be a non-negative integer")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str, location: str) -> bool:
    """Read a required boolean without accepting integer lookalikes."""

    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be a boolean")
    return value


def _require_strings(mapping: Mapping[str, Any], key: str, location: str) -> tuple[str, ...]:
    """Read a required sequence of unique, non-empty strings."""

    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty list")
    strings = tuple(str(item).strip() for item in value)
    if any(not item for item in strings) or len(set(strings)) != len(strings):
        raise ValueError(f"{location}.{key} must contain unique, non-empty strings")
    return strings


def _dataset_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated top-level dataset section."""

    return _require_mapping(config.get("dataset"), "dataset")


def _path_from_config(dataset: Mapping[str, Any], key: str) -> Path:
    """Return an expanded path from ``dataset.paths``."""

    paths = _require_mapping(dataset.get("paths"), "dataset.paths")
    return Path(_require_string(paths, key, "dataset.paths")).expanduser()


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file_metadata(dataset: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Hash required metadata inputs and validate configured official digests."""

    result: dict[str, dict[str, Any]] = {}
    for key in ("labels_csv", "class_info_csv", "mapping_json"):
        path = _path_from_config(dataset, key)
        result[key] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    supplements = dataset.get("official_supplements")
    if supplements is not None:
        supplement_config = _require_mapping(supplements, "dataset.official_supplements")
        expected = _require_string(
            supplement_config,
            "mapping_sha256",
            "dataset.official_supplements",
        ).casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(
                "dataset.official_supplements.mapping_sha256 must be 64 hexadecimal digits"
            )
        actual = str(result["mapping_json"]["sha256"])
        if actual != expected:
            raise ValueError(
                f"official RSNA mapping SHA-256 mismatch: expected {expected}, found {actual}"
            )
        result["mapping_json"]["expected_sha256"] = expected
        result["mapping_json"]["sha256_verified"] = True
    return result


def load_dataset_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML dataset configuration and validate its top-level shape."""

    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is required to read a non-JSON dataset configuration"
            ) from exc
    else:
        loaded = yaml.safe_load(text)
    config = dict(_require_mapping(loaded, "configuration"))
    _dataset_section(config)
    return config


def _read_csv_rows(path: Path, required_fields: set[str]) -> list[tuple[int, dict[str, str]]]:
    """Read CSV rows while retaining human-readable one-based line numbers."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required_fields - fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows: list[tuple[int, dict[str, str]]] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                (row_number, {key: "" if value is None else value for key, value in row.items()})
            )
    return rows


def _add_issue(
    issues: list[AuditIssue],
    code: str,
    severity: Severity,
    message: str,
    *,
    exam_id: str | None = None,
    row_number: int | None = None,
) -> None:
    """Append one consistently structured audit issue."""

    issues.append(
        AuditIssue(
            code=code,
            severity=severity,
            message=message,
            exam_id=exam_id,
            row_number=row_number,
        )
    )


def _load_patient_mapping(
    dataset: Mapping[str, Any], issues: list[AuditIssue]
) -> tuple[dict[str, str], int]:
    """Map RSNA UUIDs to true NIH patient identifiers from the official JSON."""

    mapping_config = _require_mapping(dataset.get("mapping"), "dataset.mapping")
    exam_field = _require_string(mapping_config, "exam_id_field", "dataset.mapping")
    original_field = _require_string(mapping_config, "original_image_field", "dataset.mapping")
    pattern_text = _require_string(mapping_config, "patient_id_pattern", "dataset.mapping")
    pattern = re.compile(pattern_text)
    if "patient_id" not in pattern.groupindex:
        raise ValueError("dataset.mapping.patient_id_pattern must define patient_id")

    mapping_path = _path_from_config(dataset, "mapping_json")
    raw_entries = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise ValueError(f"{mapping_path} must contain a JSON list")

    result: dict[str, str] = {}
    for index, entry in enumerate(raw_entries, start=1):
        if not isinstance(entry, Mapping):
            _add_issue(
                issues,
                "malformed_mapping_entry",
                "error",
                "Mapping entry is not an object",
                row_number=index,
            )
            continue
        exam_id = str(entry.get(exam_field, "")).strip()
        original_name = str(entry.get(original_field, "")).strip()
        match = pattern.fullmatch(original_name)
        if not exam_id or match is None:
            _add_issue(
                issues,
                "malformed_mapping_entry",
                "error",
                "Mapping entry is missing an exam ID or has an invalid original image name",
                exam_id=exam_id or None,
                row_number=index,
            )
            continue
        patient_id = match.group("patient_id")
        previous = result.get(exam_id)
        if previous is not None and previous != patient_id:
            _add_issue(
                issues,
                "inconsistent_mapping",
                "error",
                f"Exam maps to both NIH patient {previous} and {patient_id}",
                exam_id=exam_id,
                row_number=index,
            )
            continue
        if previous is not None:
            _add_issue(
                issues,
                "duplicate_mapping",
                "warning",
                "Duplicate mapping entry repeats the same NIH patient",
                exam_id=exam_id,
                row_number=index,
            )
        result[exam_id] = patient_id
    return result, len(raw_entries)


def _parse_box(
    row: Mapping[str, str],
    fields: tuple[str, str, str, str],
) -> BoundingBox | None:
    """Parse a box, returning ``None`` when any coordinate is malformed."""

    try:
        values = tuple(float(row[field].strip()) for field in fields)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return BoundingBox(*values)


def load_and_audit_records(config: Mapping[str, Any]) -> AuditResult:
    """Load RSNA metadata, derive NIH groups, and audit every annotation row."""

    dataset = _dataset_section(config)
    annotation = _require_mapping(dataset.get("annotation"), "dataset.annotation")
    image = _require_mapping(dataset.get("image"), "dataset.image")
    classes = _require_mapping(dataset.get("classes"), "dataset.classes")

    exam_field = _require_string(annotation, "exam_id_field", "dataset.annotation")
    target_field = _require_string(annotation, "target_field", "dataset.annotation")
    box_fields = (
        _require_string(annotation, "x_field", "dataset.annotation"),
        _require_string(annotation, "y_field", "dataset.annotation"),
        _require_string(annotation, "width_field", "dataset.annotation"),
        _require_string(annotation, "height_field", "dataset.annotation"),
    )
    class_exam_field = _require_string(annotation, "class_info_exam_id_field", "dataset.annotation")
    class_field = _require_string(annotation, "class_info_class_field", "dataset.annotation")
    positive_target = _require_string(annotation, "positive_target", "dataset.annotation")
    negative_target = _require_string(annotation, "negative_target", "dataset.annotation")
    if positive_target == negative_target:
        raise ValueError("Positive and negative targets must differ")

    width = _require_int(image, "width", "dataset.image")
    height = _require_int(image, "height", "dataset.image")
    study_strata = _require_strings(classes, "study_strata", "dataset.classes")
    positive_stratum = _require_string(classes, "positive_study_stratum", "dataset.classes")
    if positive_stratum not in study_strata:
        raise ValueError("positive_study_stratum must be listed in study_strata")

    issues: list[AuditIssue] = []
    patient_mapping, mapping_entries = _load_patient_mapping(dataset, issues)
    label_rows = _read_csv_rows(
        _path_from_config(dataset, "labels_csv"),
        {exam_field, target_field, *box_fields},
    )
    class_rows = _read_csv_rows(
        _path_from_config(dataset, "class_info_csv"),
        {class_exam_field, class_field},
    )

    labels_by_exam: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_number, row in label_rows:
        exam_id = row[exam_field].strip()
        if not exam_id:
            _add_issue(
                issues,
                "missing_exam_id",
                "error",
                "Annotation row has an empty exam ID",
                row_number=row_number,
            )
            continue
        labels_by_exam[exam_id].append((row_number, row))

    classes_by_exam: dict[str, set[str]] = defaultdict(set)
    for row_number, row in class_rows:
        exam_id = row[class_exam_field].strip()
        study_class = row[class_field].strip()
        if not exam_id or not study_class:
            _add_issue(
                issues,
                "malformed_class_info",
                "error",
                "Class-info row has an empty exam ID or class",
                exam_id=exam_id or None,
                row_number=row_number,
            )
            continue
        classes_by_exam[exam_id].add(study_class)

    records: list[ExamRecord] = []
    for exam_id in sorted(labels_by_exam):
        rows = labels_by_exam[exam_id]
        valid = True
        patient_id = patient_mapping.get(exam_id, "")
        if not patient_id:
            _add_issue(
                issues,
                "missing_patient_mapping",
                "error",
                "Exam is absent from the official NIH-to-RSNA mapping",
                exam_id=exam_id,
            )
            valid = False

        targets = {row[target_field].strip() for _, row in rows}
        unknown_targets = sorted(targets - {positive_target, negative_target})
        if unknown_targets:
            _add_issue(
                issues,
                "unknown_target",
                "error",
                f"Unknown target values: {unknown_targets}",
                exam_id=exam_id,
            )
            valid = False
        if len(targets) != 1:
            _add_issue(
                issues,
                "inconsistent_targets",
                "error",
                "Exam has multiple target values",
                exam_id=exam_id,
            )
            valid = False
        is_positive = targets == {positive_target}

        study_classes = classes_by_exam.get(exam_id, set())
        if len(study_classes) != 1:
            code = "missing_class_info" if not study_classes else "inconsistent_class_info"
            _add_issue(
                issues,
                code,
                "error",
                "Exam must have exactly one detailed study class",
                exam_id=exam_id,
            )
            study_stratum = ""
            valid = False
        else:
            study_stratum = next(iter(study_classes))
            if study_stratum not in study_strata:
                _add_issue(
                    issues,
                    "unknown_study_stratum",
                    "error",
                    f"Study class is not configured: {study_stratum}",
                    exam_id=exam_id,
                )
                valid = False
            if is_positive != (study_stratum == positive_stratum):
                _add_issue(
                    issues,
                    "target_class_mismatch",
                    "error",
                    "Detection target conflicts with the detailed study class",
                    exam_id=exam_id,
                )
                valid = False

        boxes: list[BoundingBox] = []
        seen_boxes: set[tuple[float, float, float, float]] = set()
        if is_positive:
            for row_number, row in rows:
                box = _parse_box(row, box_fields)
                if box is None:
                    _add_issue(
                        issues,
                        "malformed_box",
                        "error",
                        "Positive row has missing, non-numeric, or non-finite coordinates",
                        exam_id=exam_id,
                        row_number=row_number,
                    )
                    valid = False
                    continue
                if box.width <= 0 or box.height <= 0:
                    _add_issue(
                        issues,
                        "nonpositive_box",
                        "error",
                        "Box width and height must be positive",
                        exam_id=exam_id,
                        row_number=row_number,
                    )
                    valid = False
                    continue
                if (
                    box.x < 0
                    or box.y < 0
                    or box.x + box.width > width
                    or box.y + box.height > height
                ):
                    _add_issue(
                        issues,
                        "off_image_box",
                        "error",
                        "Box extends outside the configured image dimensions",
                        exam_id=exam_id,
                        row_number=row_number,
                    )
                    valid = False
                    continue
                box_key = (box.x, box.y, box.width, box.height)
                if box_key in seen_boxes:
                    _add_issue(
                        issues,
                        "duplicate_box",
                        "warning",
                        "Exact duplicate box was removed",
                        exam_id=exam_id,
                        row_number=row_number,
                    )
                    continue
                seen_boxes.add(box_key)
                boxes.append(box)
            if not boxes:
                _add_issue(
                    issues,
                    "positive_without_valid_box",
                    "error",
                    "Positive exam has no valid bounding box",
                    exam_id=exam_id,
                )
                valid = False
        else:
            for row_number, row in rows:
                if any(row[field].strip() for field in box_fields):
                    _add_issue(
                        issues,
                        "negative_with_box",
                        "error",
                        "Negative row unexpectedly contains box coordinates",
                        exam_id=exam_id,
                        row_number=row_number,
                    )
                    valid = False
            if len(rows) > 1:
                _add_issue(
                    issues,
                    "duplicate_negative_row",
                    "warning",
                    "Repeated negative rows were collapsed to one image record",
                    exam_id=exam_id,
                )

        records.append(
            ExamRecord(
                exam_id=exam_id,
                nih_patient_id=patient_id,
                study_stratum=study_stratum,
                is_positive=is_positive,
                boxes=tuple(boxes),
                valid=valid,
            )
        )

    for exam_id in sorted(set(classes_by_exam) - set(labels_by_exam)):
        _add_issue(
            issues,
            "orphan_class_info",
            "warning",
            "Detailed class info has no matching annotation exam",
            exam_id=exam_id,
        )

    valid_patient_counts = Counter(record.nih_patient_id for record in records if record.valid)
    input_counts = {
        "mapping_entries": mapping_entries,
        "mapped_exam_ids": len(patient_mapping),
        "label_rows": len(label_rows),
        "class_info_rows": len(class_rows),
        "annotation_exam_ids": len(labels_by_exam),
        "class_info_exam_ids": len(classes_by_exam),
        "valid_exam_ids": sum(record.valid for record in records),
        "invalid_exam_ids": sum(not record.valid for record in records),
        "valid_boxes": sum(len(record.boxes) for record in records if record.valid),
        "unique_nih_patients": len(valid_patient_counts),
        "multi_exam_nih_patients": sum(
            exam_count > 1 for exam_count in valid_patient_counts.values()
        ),
        "maximum_exams_per_nih_patient": max(valid_patient_counts.values(), default=0),
    }
    return AuditResult(tuple(records), tuple(issues), input_counts)


def _record_field(record: ExamRecord, field: str) -> str:
    """Read a configured grouping or stratification field from a record."""

    if not hasattr(record, field):
        raise ValueError(f"ExamRecord has no configured field {field!r}")
    value = getattr(record, field)
    return str(value)


def _make_groups(
    records: Sequence[ExamRecord], group_field: str, stratum_field: str
) -> list[PatientGroup]:
    """Collapse records into indivisible configured groups."""

    grouped: dict[str, list[ExamRecord]] = defaultdict(list)
    for record in records:
        grouped[_record_field(record, group_field)].append(record)
    groups: list[PatientGroup] = []
    for group_id in sorted(grouped):
        members = tuple(sorted(grouped[group_id], key=lambda item: item.exam_id))
        counts = Counter(_record_field(record, stratum_field) for record in members)
        groups.append(PatientGroup(group_id, members, dict(counts)))
    return groups


def _stable_key(seed: int, stage: str, value: str) -> bytes:
    """Return a process-independent seeded ordering key."""

    payload = f"{seed}\0{stage}\0{value}".encode()
    return hashlib.sha256(payload).digest()


def _balanced_group_order(
    groups: Sequence[PatientGroup],
    strata: Sequence[str],
    seed: int,
    stage: str,
) -> list[PatientGroup]:
    """Order groups so each prefix tracks the overall image-level strata mix."""

    if not groups:
        return []
    configured = list(strata)
    observed = sorted({name for group in groups for name in group.stratum_counts})
    for name in observed:
        if name not in configured:
            configured.append(name)
    stratum_index = {name: index for index, name in enumerate(configured)}
    overall = Counter[str]()
    for group in groups:
        overall.update(group.stratum_counts)
    overall_total = sum(overall.values())

    buckets: dict[str, list[PatientGroup]] = defaultdict(list)
    for group in groups:
        dominant = max(
            configured,
            key=lambda name: (group.stratum_counts.get(name, 0), -stratum_index[name]),
        )
        buckets[dominant].append(group)
    for name, bucket in buckets.items():
        bucket.sort(key=lambda group: _stable_key(seed, f"{stage}:{name}", group.group_id))

    offsets = {name: 0 for name in buckets}
    emitted = Counter[str]()
    emitted_total = 0
    ordered: list[PatientGroup] = []
    while len(ordered) < len(groups):
        candidates = [
            bucket[offsets[name]] for name, bucket in buckets.items() if offsets[name] < len(bucket)
        ]
        scored: list[tuple[int, bytes, PatientGroup]] = []
        for candidate in candidates:
            new_total = emitted_total + candidate.size
            imbalance = sum(
                abs(
                    (emitted[name] + candidate.stratum_counts.get(name, 0)) * overall_total
                    - overall[name] * new_total
                )
                for name in configured
            )
            scored.append(
                (imbalance, _stable_key(seed, f"{stage}:choice", candidate.group_id), candidate)
            )
        _, _, chosen = min(scored, key=lambda item: (item[0], item[1]))
        dominant = max(
            configured,
            key=lambda name: (chosen.stratum_counts.get(name, 0), -stratum_index[name]),
        )
        offsets[dominant] += 1
        emitted.update(chosen.stratum_counts)
        emitted_total += chosen.size
        ordered.append(chosen)
    return ordered


def _select_groups_by_size(
    ordered_groups: Sequence[PatientGroup], target: int, *, allow_under: bool
) -> list[PatientGroup]:
    """Select whole groups with an exact target size using deterministic subset sum."""

    if target < 0:
        raise ValueError("Group-selection target cannot be negative")
    if target == 0:
        return []
    if sum(group.size for group in ordered_groups) == target:
        return list(ordered_groups)

    reachable = bytearray(target + 1)
    reachable[0] = 1
    previous_total = [-1] * (target + 1)
    previous_group = [-1] * (target + 1)
    for group_index, group in enumerate(ordered_groups):
        if group.size > target:
            continue
        for total in range(target, group.size - 1, -1):
            if not reachable[total] and reachable[total - group.size]:
                reachable[total] = 1
                previous_total[total] = total - group.size
                previous_group[total] = group_index
        if reachable[target]:
            break

    actual_target = target
    if not reachable[target]:
        if not allow_under:
            raise ValueError(
                f"Patient groups cannot satisfy the exact requested image count {target}"
            )
        actual_target = next((total for total in range(target - 1, -1, -1) if reachable[total]), 0)
    selected_indices: set[int] = set()
    cursor = actual_target
    while cursor > 0:
        group_index = previous_group[cursor]
        if group_index < 0:
            raise RuntimeError("Internal subset-sum reconstruction failure")
        selected_indices.add(group_index)
        cursor = previous_total[cursor]
    return [group for index, group in enumerate(ordered_groups) if index in selected_indices]


def subsample_records(records: Sequence[ExamRecord], config: Mapping[str, Any]) -> list[ExamRecord]:
    """Create a deterministic, group-safe, image-stratified bounded subsample."""

    dataset = _dataset_section(config)
    subsample = _require_mapping(dataset.get("subsample"), "dataset.subsample")
    valid_records = sorted((record for record in records if record.valid), key=lambda r: r.exam_id)
    if not bool(subsample.get("enabled", False)):
        return valid_records
    max_images = _require_int(subsample, "max_images", "dataset.subsample")
    if len(valid_records) <= max_images:
        return valid_records
    seed = _require_seed(subsample, "seed", "dataset.subsample")
    group_field = _require_string(subsample, "group_by", "dataset.subsample")
    stratum_field = _require_string(subsample, "stratify_by", "dataset.subsample")
    classes = _require_mapping(dataset.get("classes"), "dataset.classes")
    strata = _require_strings(classes, "study_strata", "dataset.classes")
    groups = _make_groups(valid_records, group_field, stratum_field)
    ordered = _balanced_group_order(groups, strata, seed, "subsample")
    selected_groups = _select_groups_by_size(ordered, max_images, allow_under=True)
    selected = [record for group in selected_groups for record in group.records]
    if not selected:
        raise ValueError("No complete patient group fits within the configured subsample bound")
    return sorted(selected, key=lambda record: record.exam_id)


def _allocate_exact_counts(total: int, ratios: Mapping[str, Any]) -> dict[str, int]:
    """Convert ratios to deterministic integer counts using largest remainders."""

    if not ratios:
        raise ValueError("dataset.split.ratios cannot be empty")
    names = list(ratios)
    numeric = [float(ratios[name]) for name in names]
    if any(not math.isfinite(value) or value < 0 for value in numeric):
        raise ValueError("dataset.split.ratios must be finite and non-negative")
    ratio_sum = sum(numeric)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("dataset.split.ratios must sum to 1")
    raw = [total * value for value in numeric]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(names)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return dict(zip(names, counts, strict=True))


def split_records(
    records: Sequence[ExamRecord], config: Mapping[str, Any]
) -> dict[str, list[ExamRecord]]:
    """Create exact-ratio deterministic splits without patient-group leakage."""

    dataset = _dataset_section(config)
    split = _require_mapping(dataset.get("split"), "dataset.split")
    ratios = _require_mapping(split.get("ratios"), "dataset.split.ratios")
    targets = _allocate_exact_counts(len(records), ratios)
    seed = _require_seed(split, "seed", "dataset.split")
    group_field = _require_string(split, "group_by", "dataset.split")
    stratum_field = _require_string(split, "stratify_by", "dataset.split")
    classes = _require_mapping(dataset.get("classes"), "dataset.classes")
    strata = _require_strings(classes, "study_strata", "dataset.classes")

    remaining_groups = _make_groups(records, group_field, stratum_field)
    split_names = list(targets)
    result: dict[str, list[ExamRecord]] = {}
    for split_name in split_names[:-1]:
        ordered = _balanced_group_order(remaining_groups, strata, seed, f"split:{split_name}")
        selected = _select_groups_by_size(ordered, targets[split_name], allow_under=False)
        selected_ids = {group.group_id for group in selected}
        result[split_name] = sorted(
            (record for group in selected for record in group.records),
            key=lambda record: record.exam_id,
        )
        remaining_groups = [
            group for group in remaining_groups if group.group_id not in selected_ids
        ]
    final_name = split_names[-1]
    result[final_name] = sorted(
        (record for group in remaining_groups for record in group.records),
        key=lambda record: record.exam_id,
    )
    for split_name, expected in targets.items():
        if len(result[split_name]) != expected:
            raise RuntimeError(
                f"Split {split_name} has {len(result[split_name])} images; expected {expected}"
            )
    return result


def build_coco(records: Sequence[ExamRecord], config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert audited exam records to canonical COCO detection JSON."""

    dataset = _dataset_section(config)
    classes = _require_mapping(dataset.get("classes"), "dataset.classes")
    image = _require_mapping(dataset.get("image"), "dataset.image")
    foreground = _require_strings(classes, "foreground", "dataset.classes")
    positive_stratum = _require_string(classes, "positive_study_stratum", "dataset.classes")
    category_by_name = {name: index for index, name in enumerate(foreground, start=1)}
    if positive_stratum not in category_by_name:
        raise ValueError("positive_study_stratum must also be a foreground class")
    category_id = category_by_name[positive_stratum]
    processed_extension = _require_string(image, "processed_extension", "dataset.image")
    width = _require_int(image, "width", "dataset.image")
    height = _require_int(image, "height", "dataset.image")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(sorted(records, key=lambda item: item.exam_id), start=1):
        images.append(
            {
                "id": image_id,
                "file_name": f"{record.exam_id}{processed_extension}",
                "width": width,
                "height": height,
                "rsna_exam_id": record.exam_id,
                "nih_patient_id": record.nih_patient_id,
                "study_stratum": record.study_stratum,
            }
        )
        for box in record.boxes:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [box.x, box.y, box.width, box.height],
                    "area": box.area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": str(dataset.get("display_name", dataset.get("id", ""))),
            "schema_version": config.get("schema_version"),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_by_name[name], "name": name} for name in foreground],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, human-readable JSON, creating parent directories first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_split_manifest(
    path: Path,
    split_name: str,
    records: Sequence[ExamRecord],
    config: Mapping[str, Any],
) -> None:
    """Write one stable, image-level CSV split manifest."""

    dataset = _dataset_section(config)
    image = _require_mapping(dataset.get("image"), "dataset.image")
    width = _require_int(image, "width", "dataset.image")
    height = _require_int(image, "height", "dataset.image")
    source_extension = _require_string(image, "source_extension", "dataset.image")
    processed_extension = _require_string(image, "processed_extension", "dataset.image")
    fieldnames = [
        "split",
        "image_id",
        "nih_patient_id",
        "study_stratum",
        "is_positive",
        "source_file",
        "processed_file",
        "width",
        "height",
        "box_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.exam_id):
            writer.writerow(
                {
                    "split": split_name,
                    "image_id": record.exam_id,
                    "nih_patient_id": record.nih_patient_id,
                    "study_stratum": record.study_stratum,
                    "is_positive": int(record.is_positive),
                    "source_file": f"{record.exam_id}{source_extension}",
                    "processed_file": f"{record.exam_id}{processed_extension}",
                    "width": width,
                    "height": height,
                    "box_count": len(record.boxes),
                }
            )


def scale_radiograph_to_uint8(
    pixels: Any,
    *,
    photometric_interpretation: str,
    invert_monochrome1: bool,
) -> Any:
    """Apply the canonical per-image min-max scaling to one radiograph array.

    This is the shared implementation used by the original DICOM-to-PNG
    conversion and by raw-array sensitivity analyses.  Keeping the transform in
    one function prevents those paths from drifting numerically.
    """

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Radiograph scaling requires numpy") from exc

    array = np.asarray(pixels, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D radiograph, found shape {array.shape}")
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Radiograph array contains no finite values")
    low = float(array[finite].min())
    high = float(array[finite].max())
    array = np.nan_to_num(array, nan=low, posinf=high, neginf=low)
    if invert_monochrome1 and photometric_interpretation.upper() == "MONOCHROME1":
        array = high + low - array
    array = (array - low) * (255.0 / (high - low)) if high > low else np.zeros_like(array)
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _convert_one_dicom(
    source: Path,
    destination: Path,
    *,
    normalization: str,
    output_bit_depth: int,
    invert_monochrome1: bool,
) -> tuple[int, int]:
    """Convert one DICOM pixel array to a min-max normalized 8-bit PNG."""

    if normalization != "per_image_minmax":
        raise ValueError(f"Unsupported DICOM normalization mode: {normalization}")
    if output_bit_depth != 8:
        raise ValueError("Only 8-bit processed PNG output is currently supported")

    try:
        import pydicom
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("DICOM conversion requires pydicom, numpy, and Pillow") from exc

    dataset = pydicom.dcmread(source)
    output = scale_radiograph_to_uint8(
        dataset.pixel_array,
        photometric_interpretation=str(getattr(dataset, "PhotometricInterpretation", "")),
        invert_monochrome1=invert_monochrome1,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(destination)
    return int(output.shape[1]), int(output.shape[0])


def convert_available_dicoms(
    records: Sequence[ExamRecord],
    config: Mapping[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Convert available selected DICOMs and summarize missing or failed images."""

    dataset = _dataset_section(config)
    image = _require_mapping(dataset.get("image"), "dataset.image")
    source_dir = _path_from_config(dataset, "source_images_dir")
    destination_dir = _path_from_config(dataset, "processed_images_dir")
    source_extension = _require_string(image, "source_extension", "dataset.image")
    processed_extension = _require_string(image, "processed_extension", "dataset.image")
    expected_width = _require_int(image, "width", "dataset.image")
    expected_height = _require_int(image, "height", "dataset.image")
    conversion = _require_mapping(image.get("conversion"), "dataset.image.conversion")
    normalization = _require_string(conversion, "normalization", "dataset.image.conversion")
    output_bit_depth = _require_int(conversion, "output_bit_depth", "dataset.image.conversion")
    invert_monochrome1 = _require_bool(conversion, "invert_monochrome1", "dataset.image.conversion")

    available = 0
    converted = 0
    existing = 0
    missing = 0
    errors: list[dict[str, str]] = []
    for record in sorted(records, key=lambda item: item.exam_id):
        source = source_dir / f"{record.exam_id}{source_extension}"
        destination = destination_dir / f"{record.exam_id}{processed_extension}"
        if destination.is_file():
            existing += 1
            continue
        if not source.is_file():
            missing += 1
            continue
        available += 1
        if not enabled:
            continue
        try:
            actual_width, actual_height = _convert_one_dicom(
                source,
                destination,
                normalization=normalization,
                output_bit_depth=output_bit_depth,
                invert_monochrome1=invert_monochrome1,
            )
            if (actual_width, actual_height) != (expected_width, expected_height):
                raise ValueError(
                    "DICOM dimensions "
                    f"{actual_width}x{actual_height} do not match configured "
                    f"{expected_width}x{expected_height}"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append({"exam_id": record.exam_id, "message": str(exc)})
        else:
            converted += 1

    ready = converted + existing
    if not enabled or (available == 0 and ready == 0):
        mode = "metadata_only"
    elif ready == len(records) and not errors:
        mode = "converted"
    else:
        mode = "partial"
    return {
        "mode": mode,
        "selected_images": len(records),
        "source_dicoms_available": available,
        "pngs_converted": converted,
        "pngs_existing": existing,
        "source_dicoms_missing": missing,
        "conversion_errors": errors,
        "normalization": normalization,
        "output_bit_depth": output_bit_depth,
        "invert_monochrome1": invert_monochrome1,
    }


def _record_distribution(records: Sequence[ExamRecord], field: str) -> dict[str, int]:
    """Count records by a configured metadata field."""

    return dict(sorted(Counter(_record_field(record, field) for record in records).items()))


def _group_count(records: Sequence[ExamRecord], field: str) -> int:
    """Count distinct configured groups in a record collection."""

    return len({_record_field(record, field) for record in records})


def _audit_summary(
    config: Mapping[str, Any],
    audit: AuditResult,
    selected: Sequence[ExamRecord],
    splits: Mapping[str, Sequence[ExamRecord]],
    image_processing: Mapping[str, Any],
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the complete reproducible preparation and audit summary."""

    dataset = _dataset_section(config)
    subsample = _require_mapping(dataset.get("subsample"), "dataset.subsample")
    split = _require_mapping(dataset.get("split"), "dataset.split")
    subsample_group = _require_string(subsample, "group_by", "dataset.subsample")
    subsample_stratum = _require_string(subsample, "stratify_by", "dataset.subsample")
    split_group = _require_string(split, "group_by", "dataset.split")
    split_stratum = _require_string(split, "stratify_by", "dataset.split")
    issue_codes = Counter(issue.code for issue in audit.issues)
    issue_severities = Counter(issue.severity for issue in audit.issues)
    split_summary = {
        name: {
            "images": len(records),
            "groups": _group_count(records, split_group),
            "strata": _record_distribution(records, split_stratum),
            "positive_images": sum(record.is_positive for record in records),
            "negative_images": sum(not record.is_positive for record in records),
            "boxes": sum(len(record.boxes) for record in records),
        }
        for name, records in splits.items()
    }
    split_groups = {
        name: {_record_field(record, split_group) for record in records}
        for name, records in splits.items()
    }
    group_overlap = {
        f"{first}__{second}": len(split_groups[first] & split_groups[second])
        for first, second in combinations(splits, 2)
    }
    return {
        "schema_version": config.get("schema_version"),
        "dataset_id": dataset.get("id"),
        "input_files": _input_file_metadata(dataset),
        "input_counts": audit.input_counts,
        "audit": {
            "issue_count": len(audit.issues),
            "by_severity": dict(sorted(issue_severities.items())),
            "by_code": dict(sorted(issue_codes.items())),
            "issues": [issue.to_dict() for issue in audit.issues],
        },
        "subsample": {
            "enabled": bool(subsample.get("enabled", False)),
            "configured_max_images": subsample.get("max_images"),
            "seed": subsample.get("seed"),
            "selected_images": len(selected),
            "selected_groups": _group_count(selected, subsample_group),
            "strata": _record_distribution(selected, subsample_stratum),
        },
        "splits": split_summary,
        "split_group_overlap": group_overlap,
        "image_processing": dict(image_processing),
        "outputs": dict(outputs),
    }


def prepare_dataset(
    config_path: str | Path,
    *,
    metadata_only: bool = False,
    convert_images: bool | None = None,
) -> dict[str, Any]:
    """Run the complete metadata audit, selection, split, and COCO export pipeline."""

    if metadata_only and convert_images is True:
        raise ValueError("metadata_only and convert_images=True are mutually exclusive")
    config = load_dataset_config(config_path)
    dataset = _dataset_section(config)
    _input_file_metadata(dataset)
    audit = load_and_audit_records(config)
    selected = subsample_records(audit.valid_records, config)
    splits = split_records(selected, config)

    splits_dir = _path_from_config(dataset, "splits_dir")
    annotations_dir = _path_from_config(dataset, "annotations_dir")
    outputs: dict[str, str] = {}
    for split_name, records in splits.items():
        manifest_path = splits_dir / f"{split_name}.csv"
        coco_path = annotations_dir / f"instances_{split_name}.json"
        write_split_manifest(manifest_path, split_name, records, config)
        _write_json(coco_path, build_coco(records, config))
        outputs[f"manifest_{split_name}"] = str(manifest_path)
        outputs[f"coco_{split_name}"] = str(coco_path)

    conversion_enabled = not metadata_only and convert_images is not False
    image_processing = convert_available_dicoms(selected, config, enabled=conversion_enabled)
    audit_path = _path_from_config(dataset, "audit_json")
    outputs["audit_summary"] = str(audit_path)
    summary = _audit_summary(config, audit, selected, splits, image_processing, outputs)
    _write_json(audit_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset preparation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Dataset YAML path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write metadata artifacts without decoding DICOM files",
    )
    mode.add_argument(
        "--convert-images",
        action="store_true",
        help="Convert all available selected DICOM files to PNG",
    )
    args = parser.parse_args(argv)
    convert_images = True if args.convert_images else None
    summary = prepare_dataset(
        args.config,
        metadata_only=args.metadata_only,
        convert_images=convert_images,
    )
    print(
        json.dumps(
            {
                "audit_summary": summary["outputs"]["audit_summary"],
                "selected_images": summary["subsample"]["selected_images"],
                "image_mode": summary["image_processing"]["mode"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
