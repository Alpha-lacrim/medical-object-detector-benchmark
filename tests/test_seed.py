import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.utils import seed as seed_utils


def _random_sample() -> tuple[float, float]:
    return random.random(), float(np.random.random())


def test_seed_everything_repeats_python_and_numpy_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    first_report = seed_utils.seed_everything(42)
    first_sample = _random_sample()
    second_report = seed_utils.seed_everything(42)
    second_sample = _random_sample()

    assert first_sample == second_sample
    assert first_report == second_report
    assert os.environ["PYTHONHASHSEED"] == "42"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(":16:8", ":16:8"), ("invalid", ":4096:8")],
)
def test_seed_everything_accepts_only_valid_cublas_workspace_settings(
    configured: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", configured)
    monkeypatch.setattr(seed_utils, "_optional_torch", lambda: None)

    report = seed_utils.seed_everything(17)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == expected
    assert report.cublas_workspace_config == expected


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_everything_rejects_out_of_range_values(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        seed_utils.seed_everything(seed)


def test_initialize_reproducibility_seeds_torch_and_writes_run_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manual_seeds: list[int] = []
    deterministic_calls: list[tuple[bool, bool]] = []
    cudnn = SimpleNamespace(
        deterministic=False,
        benchmark=True,
        is_available=lambda: False,
        version=lambda: None,
    )
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda=None),
        manual_seed=manual_seeds.append,
        use_deterministic_algorithms=lambda enabled, warn_only: deterministic_calls.append(
            (enabled, warn_only)
        ),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda _: pytest.fail("CPU-only path must not seed CUDA"),
            device_count=lambda: pytest.fail("CPU-only path must not enumerate CUDA devices"),
            get_device_properties=lambda _: pytest.fail(
                "CPU-only path must not inspect CUDA devices"
            ),
        ),
        backends=SimpleNamespace(cudnn=cudnn),
    )

    def fake_run_command(command: Any, *, timeout_seconds: int) -> dict[str, Any]:
        del timeout_seconds
        if command[0] == "nvidia-smi":
            return {
                "status": "unavailable",
                "returncode": None,
                "stdout": "",
                "stderr": "FileNotFoundError: nvidia-smi",
            }
        return {
            "status": "ok",
            "returncode": 0,
            "stdout": "numpy==2.5.1\npytest==9.1.1\n",
            "stderr": "",
        }

    monkeypatch.setattr(seed_utils, "_optional_torch", lambda: fake_torch)
    monkeypatch.setattr(seed_utils, "_run_command", fake_run_command)

    report = seed_utils.initialize_reproducibility(17, tmp_path)

    assert manual_seeds == [17]
    assert deterministic_calls == [(True, True)]
    assert cudnn.deterministic is True
    assert cudnn.benchmark is False
    assert report.torch_available is True
    assert report.cuda_available is False

    freeze_path = tmp_path / "pip_freeze.txt"
    environment_path = tmp_path / "run_environment.json"
    assert freeze_path.read_text(encoding="utf-8") == "numpy==2.5.1\npytest==9.1.1\n"

    metadata = json.loads(environment_path.read_text(encoding="utf-8"))
    assert metadata["seed"]["seed"] == 17
    assert metadata["torch"]["version"] == "test-torch"
    assert metadata["torch"]["cuda_available"] is False
    assert metadata["nvidia_smi"]["status"] == "unavailable"
    assert metadata["pip_freeze"]["status"] == "ok"


def test_seed_worker_and_generator_use_pytorch_seed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_seeds: list[int] = []
    generator = SimpleNamespace(manual_seed=generator_seeds.append)
    fake_torch = SimpleNamespace(
        initial_seed=lambda: 2**32 + 123,
        Generator=lambda: generator,
    )
    monkeypatch.setattr(seed_utils, "_optional_torch", lambda: fake_torch)

    seed_utils.seed_worker(0)
    first_sample = _random_sample()
    seed_utils.seed_worker(0)
    second_sample = _random_sample()
    returned_generator = seed_utils.make_torch_generator(2026)

    assert first_sample == second_sample
    assert returned_generator is generator
    assert generator_seeds == [2026]


def test_initialization_can_defer_environment_writes_until_after_a_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(seed_utils, "_optional_torch", lambda: None)

    report = seed_utils.initialize_reproducibility(
        17, tmp_path, log_environment=False
    )

    assert report.seed == 17
    assert list(tmp_path.iterdir()) == []


def test_nvidia_smi_parser_tolerates_quoted_and_unavailable_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nvidia_smi_capture(command: Any, *, timeout_seconds: int) -> dict[str, Any]:
        del command, timeout_seconds
        return {
            "status": "ok",
            "returncode": 0,
            "stdout": '0,"GPU, Test",610.47,N/A\nmalformed\n',
            "stderr": "",
        }

    monkeypatch.setattr(seed_utils, "_run_command", nvidia_smi_capture)

    metadata = seed_utils._collect_nvidia_smi()

    assert metadata["gpus"] == [
        {
            "index": 0,
            "name": "GPU, Test",
            "driver_version": "610.47",
            "memory_mib": None,
        }
    ]
    assert len(metadata["parse_errors"]) == 2


def test_package_logging_falls_back_when_pip_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_command(command: Any, *, timeout_seconds: int) -> dict[str, Any]:
        del command, timeout_seconds
        return {
            "status": "unavailable",
            "returncode": None,
            "stdout": "",
            "stderr": "No module named pip",
        }

    monkeypatch.setattr(seed_utils, "_optional_torch", lambda: None)
    monkeypatch.setattr(seed_utils, "_run_command", unavailable_command)
    monkeypatch.setattr(seed_utils, "_fallback_freeze", lambda: "numpy==2.5.1\n")

    report = seed_utils.seed_everything(7, deterministic=False)
    environment_path, freeze_path = seed_utils.log_run_environment(tmp_path, report)

    assert freeze_path.read_text(encoding="utf-8") == "numpy==2.5.1\n"
    metadata = json.loads(environment_path.read_text(encoding="utf-8"))
    assert metadata["pip_freeze"]["status"] == "fallback"
    assert metadata["pip_freeze"]["method"] == "importlib.metadata"
    assert metadata["pip_freeze"]["stderr"] == "No module named pip"
    assert metadata["determinism_notes"][-1] == (
        "Deterministic PyTorch algorithms were disabled for this run."
    )
