import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from pydicom.dataset import Dataset

from src.data.cohort_characteristics import (
    AggregateAccumulator,
    _aggregate_row,
    _dicom_value_and_vr,
    load_cohort_characteristics_config,
    parse_patient_age,
    resolve_dicom_root,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "cohort_characteristics.yaml"


def test_config_freezes_the_immutable_5000_study_cohort() -> None:
    config = load_cohort_characteristics_config(CONFIG_PATH)

    assert config.inputs.dicom_root_environment_variable == "RSNA_DICOM_ROOT"
    assert config.expected_cohort.total.studies == 5000
    assert config.expected_cohort.total.patient_groups == 2136
    assert config.expected_cohort.total.opacity_positive_studies == 1136
    assert config.expected_cohort.total.bounding_boxes == 1812
    assert set(config.inputs.split_manifests) == {"train", "val", "test"}


def test_environment_dicom_root_overrides_dataset_config(tmp_path: Path) -> None:
    config = load_cohort_characteristics_config(CONFIG_PATH)
    dataset_config = {"dataset": {"paths": {"source_images_dir": "missing"}}}

    root, source = resolve_dicom_root(
        config,
        dataset_config,
        environ={"RSNA_DICOM_ROOT": str(tmp_path)},
    )

    assert root == tmp_path.resolve()
    assert source == "environment:RSNA_DICOM_ROOT"


def test_missing_dicom_root_requests_the_named_environment_variable(tmp_path: Path) -> None:
    config = load_cohort_characteristics_config(CONFIG_PATH)
    config = config.model_copy(update={"project_root": tmp_path})
    dataset_config = {"dataset": {"paths": {"source_images_dir": "missing"}}}

    with pytest.raises(FileNotFoundError, match="Set RSNA_DICOM_ROOT"):
        resolve_dicom_root(config, dataset_config, environ={})


def test_patient_age_parser_distinguishes_standard_units_and_dataset_caveat() -> None:
    settings = load_cohort_characteristics_config(CONFIG_PATH).metadata.age

    years = parse_patient_age("047Y", settings)
    months = parse_patient_age("006M", settings)
    bare = parse_patient_age("47", settings)

    assert years.years == 47.0
    assert years.syntax == "dicom_as_conformant"
    assert years.encoded_unit == "Y"
    assert months.years == 0.5
    assert months.encoded_unit == "M"
    assert bare.years == 47.0
    assert bare.syntax == "numeric_without_unit_assumed_years"
    assert bare.unit_missing is True


def test_patient_age_parser_excludes_missing_invalid_and_out_of_range_values() -> None:
    settings = load_cohort_characteristics_config(CONFIG_PATH).metadata.age

    missing = parse_patient_age("", settings)
    invalid = parse_patient_age("forty", settings)
    impossible = parse_patient_age("155", settings)

    assert missing.tag_missing is True and missing.years is None
    assert invalid.syntax == "invalid_other" and invalid.years is None
    assert impossible.out_of_range is True and impossible.years is None


def test_selected_dicom_tags_retain_actual_vr_without_patient_identifiers() -> None:
    dataset = Dataset()
    dataset.PatientAge = "047Y"
    dataset.PatientSex = "F"
    dataset.ViewPosition = "PA"

    assert _dicom_value_and_vr(dataset, "PatientAge") == ("047Y", "AS")
    assert _dicom_value_and_vr(dataset, "PatientSex") == ("F", "CS")
    assert _dicom_value_and_vr(dataset, "ViewPosition") == ("PA", "CS")


def test_aggregate_row_contains_counts_percentages_and_no_identifier_fields() -> None:
    config = load_cohort_characteristics_config(CONFIG_PATH)
    aggregate = AggregateAccumulator(
        studies=4,
        patient_groups={"group-a", "group-b"},
        opacity_positive_studies=1,
        bounding_boxes=2,
        age_years=[20.0, 40.0, 60.0],
        age_syntax=Counter({"numeric_without_unit_assumed_years": 4}),
        age_unit_missing=4,
        age_out_of_range=1,
        age_unusable=1,
        sex=Counter({"F": 3, "M": 1}),
        projection=Counter({"AP": 1, "PA": 3}),
    )

    row = _aggregate_row("train", aggregate, config.metadata)

    assert row["studies"] == 4
    assert row["age_median_years"] == 40.0
    assert row["sex_f_count"] == 3
    assert row["sex_f_percent"] == 75.0
    assert row["projection_pa_count"] == 3
    assert row["projection_pa_percent"] == 75.0
    assert not any("patient_id" in name or "image_id" in name for name in row)


def test_committed_outputs_are_aggregate_only_and_match_the_immutable_total() -> None:
    table_path = ROOT / "results" / "tables" / "rsna_cohort_characteristics.csv"
    summary_path = ROOT / "results" / "logs" / "phase44_cohort_characteristics" / "summary.json"
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [row["split"] for row in rows] == [
        "train",
        "validation",
        "internal_test",
        "total",
    ]
    assert rows[-1]["studies"] == "5000"
    assert rows[-1]["patient_groups"] == "2136"
    assert summary["privacy"] == {
        "aggregate_outputs_only": True,
        "dicom_pixels_read": False,
        "identifiers_or_patient_rows_written": False,
        "selected_tags_only": ["PatientAge", "PatientSex", "ViewPosition"],
    }
    assert summary["prerequisite_gate"]["passed"] is True
    assert (
        summary["prerequisite_gate"]["selected_studies_matching_official_patient_mapping"] == 5000
    )
    assert summary["prerequisite_gate"]["source_metadata_sha256"]["mapping_json"] == (
        "803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27"
    )
    assert not any(
        forbidden in row
        for row in rows
        for forbidden in ("image_id", "nih_patient_id", "source_file")
    )
