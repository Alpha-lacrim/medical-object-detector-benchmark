import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.explainability.sanity_checks import (
    DetectorLayerGroups,
    LayerGroup,
    ParameterInitializationSettings,
    _atomic_csv,
    aggregate_sanity,
    assert_checkpoint_immutability,
    checkpoint_hashes,
    load_xai_sanity_config,
    map_similarity_metrics,
    pearson_map_correlation,
    randomize_cumulative_model,
    select_nested_stratified_rows,
    select_panel_cases,
    shuffle_pixel_vectors,
    xavier_reinitialize_model,
)


def test_csv_artifacts_use_repository_lf_line_endings(tmp_path: Path) -> None:
    output = tmp_path / "artifact.csv"

    _atomic_csv(output, ["name", "value"], [{"name": "case", "value": 1}])

    assert output.read_bytes() == b"name,value\ncase,1\n"


def test_config_binds_nested_fifty_image_pool_and_outputs() -> None:
    config = load_xai_sanity_config(Path("configs/xai_sanity.yaml"))

    assert config.seed == 17
    assert config.sampling.size == 50
    assert config.sampling.source_pool_manifest.name == "test_robustness_seed17_n300.csv"
    assert config.target.map_target == "pre_activation_foreground_score_at_trained_reference_region"
    assert config.similarity.methods == ("pearson", "spearman", "ssim")
    assert config.similarity.evaluation_size == 40
    assert config.outputs.summary_table.name == "gradcam_sanity_v2_summary.csv"
    assert config.outputs.panel_figure.name == "gradcam_sanity_v2_panel.png"


def test_nested_sampler_uses_proportional_largest_remainder() -> None:
    rows = [{"image": f"opacity-{index:03d}", "stratum": "Lung Opacity"} for index in range(68)]
    rows.extend(
        {"image": f"other-{index:03d}", "stratum": "No Lung Opacity / Not Normal"}
        for index in range(132)
    )
    rows.extend({"image": f"normal-{index:03d}", "stratum": "Normal"} for index in range(100))

    selected, allocation = select_nested_stratified_rows(
        rows,
        size=50,
        id_column="image",
        stratum_column="stratum",
        seed=17,
    )

    assert allocation == {
        "Lung Opacity": 11,
        "No Lung Opacity / Not Normal": 22,
        "Normal": 17,
    }
    assert Counter(row["stratum"] for row in selected) == allocation
    assert selected == sorted(selected, key=lambda row: row["image"])


def test_pixel_shuffle_is_deterministic_and_preserves_rgb_vectors() -> None:
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    image = Image.fromarray(pixels)

    first = np.asarray(shuffle_pixel_vectors(image, seed=31, image_id="exam-a"))
    second = np.asarray(shuffle_pixel_vectors(image, seed=31, image_id="exam-a"))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, pixels)
    assert sorted(map(tuple, first.reshape(-1, 3))) == sorted(map(tuple, pixels.reshape(-1, 3)))


def test_pearson_map_correlation_handles_identity_inverse_and_constant() -> None:
    heatmap = np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)

    assert pearson_map_correlation(heatmap, heatmap, epsilon=1e-12) == pytest.approx(1.0)
    assert pearson_map_correlation(heatmap, 1 - heatmap, epsilon=1e-12) == pytest.approx(-1.0)
    assert pearson_map_correlation(np.ones((2, 2)), heatmap, epsilon=1e-12) is None


def test_similarity_metrics_handle_rank_structure_ssim_and_degenerate_maps() -> None:
    settings = load_xai_sanity_config(Path("configs/xai_sanity.yaml")).similarity
    heatmap = np.arange(64, dtype=np.float32).reshape(8, 8)

    identity = map_similarity_metrics(heatmap, heatmap, settings=settings)
    inverse = map_similarity_metrics(heatmap, heatmap.max() - heatmap, settings=settings)
    degenerate = map_similarity_metrics(np.ones((8, 8)), heatmap, settings=settings)
    nonfinite = heatmap.copy()
    nonfinite[0, 0] = np.nan

    assert identity["valid"] is True
    assert identity["pearson_correlation"] == pytest.approx(1.0)
    assert identity["spearman_correlation"] == pytest.approx(1.0)
    assert identity["ssim"] == pytest.approx(1.0)
    assert inverse["pearson_correlation"] == pytest.approx(-1.0)
    assert inverse["spearman_correlation"] == pytest.approx(-1.0, abs=1e-5)
    assert inverse["ssim"] < identity["ssim"]
    assert degenerate == {
        "valid": False,
        "failure_reason": "nonfinite_or_degenerate_map_after_similarity_preprocessing",
        "pearson_correlation": None,
        "spearman_correlation": None,
        "ssim": None,
    }
    assert map_similarity_metrics(nonfinite, heatmap, settings=settings)["valid"] is False


def test_xavier_randomization_covers_one_dimensional_weights_and_preserves_buffers() -> None:
    torch = pytest.importorskip("torch")

    class FrozenAffine(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", torch.ones(2))
            self.register_buffer("bias", torch.ones(2))
            self.register_buffer("running_mean", torch.zeros(2))

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 3, bias=True),
        torch.nn.BatchNorm2d(2),
        FrozenAffine(),
    )
    before_weight = model[0].weight.detach().clone()
    before_running_mean = model[1].running_mean.detach().clone()
    before_frozen_running_mean = model[2].running_mean.detach().clone()

    audit = xavier_reinitialize_model(model, seed=21017, gain=1.0)

    assert audit["xavier_weight_tensor_count"] == 3
    assert audit["one_dimensional_row_view_tensor_count"] == 2
    assert audit["xavier_weight_buffer_tensor_count"] == 1
    assert audit["zeroed_bias_tensor_count"] == 3
    assert audit["zeroed_bias_buffer_tensor_count"] == 1
    assert not torch.equal(model[0].weight, before_weight)
    assert torch.count_nonzero(model[0].bias) == 0
    assert torch.count_nonzero(model[1].bias) == 0
    assert torch.count_nonzero(model[2].bias) == 0
    assert torch.equal(model[1].running_mean, before_running_mean)
    assert torch.equal(model[2].running_mean, before_frozen_running_mean)


def test_cumulative_layer_group_randomization_is_deterministic_and_nonmutating() -> None:
    torch = pytest.importorskip("torch")

    class ToyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(4, 2)
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.BatchNorm1d(4),
            )

    groups = DetectorLayerGroups(
        detector="faster_rcnn",
        groups=(
            LayerGroup(name="head", description="output head", module_prefixes=("head",)),
            LayerGroup(
                name="backbone",
                description="feature extractor",
                module_prefixes=("backbone",),
            ),
        ),
    )
    settings = ParameterInitializationSettings(
        copy_method="deep_copy_trained_model",
        weight_initialization="xavier_normal",
        one_dimensional_weight_view="row_vector",
        bias_initialization="zeros",
        preserve_non_weight_non_bias_buffers=True,
        autocast="disabled_for_randomized_weight_numerical_validity",
        gain=1.0,
        seed=310,
        rng_reset_per_cumulative_stage=True,
    )
    torch.manual_seed(9)
    original = ToyDetector()
    original_state = {name: value.detach().clone() for name, value in original.state_dict().items()}

    first, first_audit = randomize_cumulative_model(
        original,
        groups=groups,
        stage_index=1,
        settings=settings,
    )
    second, second_audit = randomize_cumulative_model(
        original,
        groups=groups,
        stage_index=1,
        settings=settings,
    )

    assert first_audit == second_audit
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert not torch.equal(first.head.weight, original.head.weight)
    assert torch.equal(first.backbone[0].weight, original.backbone[0].weight)
    assert all(
        torch.equal(original.state_dict()[name], value) for name, value in original_state.items()
    )


def test_checkpoint_files_remain_immutable_during_in_memory_randomization(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    faster_path = tmp_path / "faster.pt"
    yolo_path = tmp_path / "yolo.pt"
    faster_path.write_bytes(b"frozen-faster-checkpoint")
    yolo_path.write_bytes(b"frozen-yolo-checkpoint")
    paths = {"faster_rcnn": faster_path, "yolo11s": yolo_path}
    before = checkpoint_hashes(paths)

    model = torch.nn.Linear(3, 2)
    xavier_reinitialize_model(model, seed=31, gain=1.0)

    after = checkpoint_hashes(paths)
    assert_checkpoint_immutability(before, after)
    assert before == after
    with pytest.raises(RuntimeError, match="checkpoint immutability violated"):
        assert_checkpoint_immutability(before, {**after, "yolo11s": "changed"})


def test_v2_labels_input_control_and_marks_training_data_randomization_not_performed() -> None:
    config = load_xai_sanity_config(Path("configs/xai_sanity.yaml"))

    assert config.input_pixel_randomization_control.interpretation == (
        "input_perturbation_stress_control_only"
    )
    assert config.claim_boundaries.adebayo_training_data_randomization == "not_performed"
    assert not hasattr(config, "data_randomization")
    assert "input_pixel_randomization_control" in config.model_dump()

    with Path("results/tables/gradcam_sanity_v2_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["control"] for row in rows} == {
        "cascading_model_parameter_randomization",
        "input_pixel_randomization_control",
    }
    assert all(row["training_data_randomization_performed"] == "False" for row in rows)


def test_aggregation_reports_declared_k_and_failure_rates() -> None:
    records = []
    for detector in ("faster_rcnn", "yolo11s"):
        for control, stage_index, stage_name, groups in (
            ("input_pixel_randomization_control", 0, "input_pixel_permutation", []),
            ("cascading_model_parameter_randomization", 1, "head", ["head"]),
            (
                "cascading_model_parameter_randomization",
                2,
                "backbone",
                ["head", "backbone"],
            ),
        ):
            records.extend(
                [
                    {
                        "detector": detector,
                        "control": control,
                        "cascade_stage_index": stage_index,
                        "cascade_stage_name": stage_name,
                        "randomized_group_names": json.dumps(groups),
                        "randomized_module_prefixes": "[]",
                        "full_model_parameter_randomization": stage_index == 2,
                        "trained_valid": True,
                        "randomized_valid": True,
                        "similarity_valid": True,
                        "pearson_correlation": 0.25,
                        "spearman_correlation": 0.20,
                        "ssim": 0.10,
                    },
                    {
                        "detector": detector,
                        "control": control,
                        "cascade_stage_index": stage_index,
                        "cascade_stage_name": stage_name,
                        "randomized_group_names": json.dumps(groups),
                        "randomized_module_prefixes": "[]",
                        "full_model_parameter_randomization": stage_index == 2,
                        "trained_valid": True,
                        "randomized_valid": False,
                        "similarity_valid": False,
                        "pearson_correlation": None,
                        "spearman_correlation": None,
                        "ssim": None,
                    },
                ]
            )
    sampling = {
        "subset_image_count": 2,
        "subset_patient_count": 2,
        "subset_stratum_counts": {"A": 1, "B": 1},
    }

    rows = aggregate_sanity(records, sampling_audit=sampling)

    assert len(rows) == 6
    assert all(row["k_valid_similarity_pairs"] == 1 for row in rows)
    assert all(row["map_pair_failure_rate"] == pytest.approx(0.5) for row in rows)
    assert all(row["pearson_mean"] == pytest.approx(0.25) for row in rows)
    assert all(row["spearman_mean"] == pytest.approx(0.20) for row in rows)
    assert all(row["ssim_mean"] == pytest.approx(0.10) for row in rows)


def test_panel_cases_are_selected_before_results() -> None:
    rows = [
        {"image": "z.png", "stratum": "B"},
        {"image": "a.png", "stratum": "B"},
        {"image": "c.png", "stratum": "A"},
    ]

    cases = select_panel_cases(rows, id_column="image", stratum_column="stratum")

    assert cases == [
        {"study_stratum": "A", "image_id": "c.png"},
        {"study_stratum": "B", "image_id": "a.png"},
    ]
