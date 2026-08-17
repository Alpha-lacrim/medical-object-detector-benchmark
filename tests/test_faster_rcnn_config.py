from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.models.faster_rcnn_config import config_fingerprint, load_faster_rcnn_config

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "faster_rcnn.yaml"
EXPECTED_CONFIG_SHA256 = "ef1e3ebe1fbe3cf1a6e27bf8b9c12f61719c2ea8771c9758f64dc278dd0e2633"


def _config_payload() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "faster_rcnn.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_real_config_locks_batch_2_hardware_and_training_invariants() -> None:
    config = load_faster_rcnn_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.experiment_id == "faster-rcnn-rsna-baseline-seed17"
    assert config.seed == 17
    assert config.model.architecture == "fasterrcnn_resnet50_fpn_v2"
    assert config.model.weights == "DEFAULT"
    assert (config.model.min_size, config.model.max_size) == (640, 640)
    assert config.runtime.device == "cuda"
    assert config.runtime.amp is True
    assert config.runtime.amp_dtype == "float16"
    assert config.runtime.batch_size == 2
    assert config.runtime.gradient_accumulation_steps == 2
    assert config.runtime.effective_batch_size == 4
    assert config.runtime.validation_persistent_workers is False
    assert config.profiling.num_workers == 0
    assert config.profiling.persistent_workers is False
    assert config.training.early_stopping.metric == "val_map_50_95"
    assert config.training.early_stopping.mode == "max"
    assert config.evaluation.coco_minimum_score == 0.0
    assert config.benchmark.epochs == 3
    assert config.benchmark.require_complete_dataset is True
    assert config.outputs.resolved_config_path == Path("resolved_config.json")
    assert config.outputs.epoch_csv_path == Path("epochs.csv")
    assert config.outputs.epoch_jsonl_path == Path("epochs.jsonl")
    assert config.outputs.summary_path == Path("summary.json")
    assert config.outputs.benchmark_estimate_path == Path("benchmark_estimate.json")
    assert config.outputs.best_checkpoint_path == Path(
        "results/checkpoints/faster_rcnn_rsna_seed17_full/best_model.pt"
    )
    assert config.outputs.last_checkpoint_path == Path(
        "results/checkpoints/faster_rcnn_rsna_seed17_full/last_state.pt"
    )
    assert config.outputs.benchmark_timing_checkpoints_dir == Path(
        "results/checkpoints/faster_rcnn_rsna_seed17_benchmark_timing"
    )
    assert config.outputs.validation_table_path == Path(
        "results/tables/faster_rcnn_baseline_validation.csv"
    )
    assert config.outputs.compute_table_path == Path("results/tables/faster_rcnn_compute.csv")
    assert config.outputs.training_curves_path == Path(
        "results/figures/faster_rcnn_training_curves.png"
    )


def test_run_scoped_output_paths_and_finalize_alias_use_configured_names() -> None:
    config = load_faster_rcnn_config(CONFIG_PATH)

    assert config.outputs.run_name("finalize") == config.outputs.train_run_name
    assert config.run_dir("finalize") == config.run_dir("train")
    assert config.run_artifact_path("benchmark", config.outputs.epoch_csv_path) == (
        PROJECT_ROOT / "results" / "logs" / "faster_rcnn_rsna_seed17_benchmark" / "epochs.csv"
    )


def test_real_config_fingerprint_is_pinned_to_exact_source_bytes() -> None:
    config = load_faster_rcnn_config(CONFIG_PATH)
    expected_from_source = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()

    assert expected_from_source == EXPECTED_CONFIG_SHA256
    assert config_fingerprint(config) == EXPECTED_CONFIG_SHA256
    assert config_fingerprint(load_faster_rcnn_config(CONFIG_PATH)) == EXPECTED_CONFIG_SHA256


def test_non_amp_runtime_is_rejected(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["runtime"]["amp"] = False

    with pytest.raises(ValueError, match="enable float16 AMP"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize("batch_size", [1, 5])
def test_batch_size_outside_hardware_scope_is_rejected(
    tmp_path: Path,
    batch_size: int,
) -> None:
    payload = _config_payload()
    payload["runtime"]["batch_size"] = batch_size

    with pytest.raises(ValueError, match="batch_size must be between 2 and 4"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


def test_wrong_architecture_is_rejected(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["model"]["architecture"] = "fasterrcnn_resnet50_fpn"

    with pytest.raises(ValueError, match="must be fasterrcnn_resnet50_fpn_v2"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


def test_coco_minimum_score_must_match_model_output_threshold(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["evaluation"]["coco_minimum_score"] = 0.1

    with pytest.raises(ValueError, match=r"must equal model\.box_score_threshold"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["runtime"]["silent_fallback_to_cpu"] = True

    with pytest.raises(ValueError, match="runtime contains unknown keys"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("runtime", "validation_persistent_workers"),
        ("profiling", "persistent_workers"),
    ],
)
def test_loader_persistence_settings_require_booleans(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    payload = _config_payload()
    payload[section][field] = 1

    with pytest.raises(ValueError, match=rf"{section}\.{field} must be boolean"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


def test_profiling_persistence_requires_a_worker(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["profiling"]["persistent_workers"] = True

    with pytest.raises(ValueError, match=r"require profiling\.num_workers > 0"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "persistent_field",
    ["persistent_workers", "validation_persistent_workers"],
)
def test_runtime_persistence_requires_a_worker(
    tmp_path: Path,
    persistent_field: str,
) -> None:
    payload = _config_payload()
    payload["runtime"]["num_workers"] = 0
    payload["runtime"]["persistent_workers"] = False
    payload["runtime"]["validation_persistent_workers"] = False
    payload["runtime"][persistent_field] = True

    with pytest.raises(ValueError, match=r"require runtime\.num_workers > 0"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "escaped_path",
    [
        "../outside/images",
        "C:\\outside\\images",
        "C:/outside/images",
        ".",
        "data/cache:alternate/images",
    ],
)
def test_path_escape_is_rejected(tmp_path: Path, escaped_path: str) -> None:
    payload = deepcopy(_config_payload())
    payload["data"]["images_dir"] = escaped_path

    with pytest.raises(ValueError, match="portable project-relative POSIX path"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "output_field",
    [
        "resolved_config_path",
        "epoch_csv_path",
        "epoch_jsonl_path",
        "summary_path",
        "benchmark_estimate_path",
        "best_checkpoint_path",
        "last_checkpoint_path",
        "benchmark_timing_checkpoints_dir",
        "validation_table_path",
        "compute_table_path",
        "training_curves_path",
    ],
)
def test_output_artifact_paths_reject_project_escape(
    tmp_path: Path,
    output_field: str,
) -> None:
    payload = deepcopy(_config_payload())
    payload["outputs"][output_field] = "../outside/artifact"

    with pytest.raises(ValueError, match="portable project-relative POSIX path"):
        load_faster_rcnn_config(_write_config(tmp_path, payload))
