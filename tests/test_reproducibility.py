import json
import random
from pathlib import Path

import numpy as np
import pytest

from meddet_benchmark.__main__ import main
from meddet_benchmark.reproducibility import configure_reproducibility, seed_worker

SMOKE_PATH = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def sample_random_values() -> tuple[float, float]:
    return random.random(), float(np.random.random())


def test_reseeding_repeats_python_and_numpy_streams() -> None:
    first_report = configure_reproducibility(42, deterministic=True)
    first = sample_random_values()
    second_report = configure_reproducibility(42, deterministic=True)
    second = sample_random_values()

    assert first == second
    assert first_report == second_report
    assert first_report.seed == 42


def test_worker_seed_is_repeatable_without_assuming_torch_availability() -> None:
    configure_reproducibility(2026, deterministic=True)
    seed_worker(3)
    first = sample_random_values()
    seed_worker(3)
    second = sample_random_values()

    assert first == second


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_seed_is_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        configure_reproducibility(seed, deterministic=True)


def test_cli_smoke_emits_strict_json(capsys) -> None:
    assert main(["smoke", str(SMOKE_PATH)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["experiment_id"] == "synthetic-smoke-v1"
    assert output["operation"] == "smoke"
    assert len(output["config_sha256"]) == 64
    assert output["reproducibility"]["seed"] == 17
