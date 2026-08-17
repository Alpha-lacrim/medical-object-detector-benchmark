from pathlib import Path

from src.robustness.run_robustness import (
    SamplingSettings,
    _largest_remainder_allocation,
    draw_stratified_subsample,
    load_robustness_config,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "corruptions.yaml"


def test_phase6_config_is_primary_seed_and_matches_required_grid() -> None:
    config = load_robustness_config(CONFIG_PATH)

    assert config.seed == 17
    assert config.sampling.size == 300
    assert {item.detector for item in config.detectors} == {"faster_rcnn", "yolo11s"}
    assert len(config.corruptions) == 7
    assert all(len(item.levels) == 5 for item in config.corruptions)
    assert config.evaluation.score_threshold == 0.25
    assert config.evaluation.coco_minimum_score == 0.001


def test_largest_remainder_allocation_is_exact_and_proportional() -> None:
    allocation = _largest_remainder_allocation(
        {
            "Lung Opacity": 169,
            "No Lung Opacity / Not Normal": 331,
            "Normal": 250,
        },
        300,
    )

    assert allocation == {
        "Lung Opacity": 68,
        "No Lung Opacity / Not Normal": 132,
        "Normal": 100,
    }
    assert sum(allocation.values()) == 300


def test_stratified_sample_is_reproducible_and_preserves_allocations(tmp_path: Path) -> None:
    source = ROOT / "data" / "splits" / "rsna-pneumonia-5000" / "test.csv"
    first_settings = SamplingSettings(
        source_manifest=source,
        output_manifest=tmp_path / "first.csv",
        size=300,
        split_column="split",
        split_value="test",
        id_column="processed_file",
        stratum_column="study_stratum",
        allocation="proportional_largest_remainder",
    )
    second_settings = first_settings.model_copy(update={"output_manifest": tmp_path / "second.csv"})

    first, first_audit = draw_stratified_subsample(first_settings, project_root=ROOT, seed=17)
    second, second_audit = draw_stratified_subsample(second_settings, project_root=ROOT, seed=17)

    assert [row["processed_file"] for row in first] == [row["processed_file"] for row in second]
    assert first_audit["sample_stratum_counts"] == {
        "Lung Opacity": 68,
        "No Lung Opacity / Not Normal": 132,
        "Normal": 100,
    }
    assert first_audit["sample_positive_images"] == 68
    assert first_audit["sample_box_count"] == 111
    assert first_audit["output_manifest_sha256"] == second_audit["output_manifest_sha256"]
