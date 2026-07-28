from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from meddet_benchmark.config import (
    ExperimentConfig,
    assert_run_allowed,
    canonical_json,
    config_fingerprint,
    load_experiment,
)

SMOKE_PATH = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def smoke_payload() -> dict:
    return yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))


def test_smoke_config_loads_and_track_a_is_structurally_shared() -> None:
    config = load_experiment(SMOKE_PATH)

    assert config.track_a.training.effective_batch_size == 2
    assert set(config.models) == {"faster_rcnn", "yolo"}
    assert config.data.class_names == ("lesion",)


def test_fingerprint_is_stable_and_json_is_strict() -> None:
    first = load_experiment(SMOKE_PATH)
    second = ExperimentConfig.model_validate(smoke_payload())

    assert canonical_json(first) == canonical_json(second)
    assert config_fingerprint(first) == config_fingerprint(second)
    assert len(config_fingerprint(first)) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"unexpected": True}), "Extra inputs"),
        (lambda data: data["models"].pop("yolo"), "exactly Faster R-CNN and YOLO"),
        (lambda data: data.update({"seeds": [17, 17]}), "seeds must be unique"),
        (lambda data: data["data"].update({"root": "../outside"}), "portable"),
    ],
)
def test_invalid_protocol_is_rejected(mutation, message: str) -> None:
    payload = deepcopy(smoke_payload())
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(payload)


def test_test_access_requires_real_frozen_protocol() -> None:
    config = load_experiment(SMOKE_PATH)
    assert_run_allowed(config, "smoke")

    with pytest.raises(RuntimeError, match="cannot train"):
        assert_run_allowed(config, "train")
    with pytest.raises(RuntimeError, match="cannot train"):
        assert_run_allowed(config, "test")

    payload = smoke_payload()
    payload["status"] = "draft"
    payload["data"]["kind"] = "manual"
    draft = ExperimentConfig.model_validate(payload)
    assert_run_allowed(draft, "train")
    with pytest.raises(RuntimeError, match="frozen"):
        assert_run_allowed(draft, "test")
