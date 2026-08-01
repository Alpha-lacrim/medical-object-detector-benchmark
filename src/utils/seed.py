"""Deterministic RNG setup and per-run environment capture."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import numpy as np

_MAX_NUMPY_SEED = 2**32 - 1
_PIP_FREEZE_FILENAME = "pip_freeze.txt"
_RUN_ENVIRONMENT_FILENAME = "run_environment.json"
_VALID_CUBLAS_WORKSPACE_CONFIGS = frozenset({":16:8", ":4096:8"})


@dataclass(frozen=True)
class SeedReport:
    """Summary of the reproducibility settings applied to the current process."""

    seed: int
    deterministic: bool
    warn_only: bool
    torch_available: bool
    cuda_available: bool
    cublas_workspace_config: str | None
    python_hash_seed_applies_on_restart: bool = True


def _optional_torch() -> Any | None:
    """Import PyTorch lazily so lightweight and CPU-only tooling still runs."""

    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            return None
        raise
    return torch


def _validate_seed(seed: int) -> None:
    """Validate a seed against NumPy's accepted legacy RNG range."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_NUMPY_SEED}")


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
) -> SeedReport:
    """Seed Python, NumPy, and PyTorch and configure deterministic Torch behavior.

    ``PYTHONHASHSEED`` only affects hashing in interpreters started after the
    variable is set. Call this function before initializing CUDA. With
    ``warn_only=True``, PyTorch warns when an operation has no deterministic
    implementation instead of aborting the run; the choice is captured in the
    run metadata.
    """

    _validate_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured_workspace not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    random.seed(seed)
    np.random.seed(seed)

    torch = _optional_torch()
    if torch is None:
        return SeedReport(
            seed=seed,
            deterministic=deterministic,
            warn_only=warn_only,
            torch_available=False,
            cuda_available=False,
            cublas_workspace_config=cublas_workspace_config,
        )

    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    return SeedReport(
        seed=seed,
        deterministic=deterministic,
        warn_only=warn_only,
        torch_available=True,
        cuda_available=cuda_available,
        cublas_workspace_config=cublas_workspace_config,
    )


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker process."""

    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError("worker_id must be a non-negative integer")
    torch = _optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to seed a DataLoader worker")
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> Any:
    """Return a seeded generator for a PyTorch DataLoader's ``generator`` argument."""

    _validate_seed(seed)
    torch = _optional_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required to create a DataLoader generator")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _run_command(command: Sequence[str], *, timeout_seconds: int) -> dict[str, Any]:
    """Run a metadata command without allowing collection failures to stop a run."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "unavailable",
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
        }

    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _collect_torch_environment(torch: Any | None) -> dict[str, Any]:
    """Collect Torch, CUDA runtime, cuDNN, and visible GPU information safely."""

    if torch is None:
        return {
            "available": False,
            "version": None,
            "cuda_build_version": None,
            "cudnn_version": None,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )

    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_available": cuda_available,
        "device_count": len(devices),
        "devices": devices,
    }


def _collect_nvidia_smi() -> dict[str, Any]:
    """Collect NVIDIA GPU names, driver versions, and memory from ``nvidia-smi``."""

    command = (
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    capture = _run_command(command, timeout_seconds=10)
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    if capture["status"] == "ok":
        for line_number, fields in enumerate(
            csv.reader(capture["stdout"].splitlines(), skipinitialspace=True),
            start=1,
        ):
            if len(fields) != 4:
                parse_errors.append(f"line {line_number}: expected 4 fields, got {len(fields)}")
                continue
            index_text, name, driver_version, memory_text = (field.strip() for field in fields)
            try:
                index = int(index_text)
            except ValueError:
                parse_errors.append(f"line {line_number}: invalid GPU index {index_text!r}")
                continue
            try:
                memory_mib = int(memory_text)
            except ValueError:
                memory_mib = None
                parse_errors.append(f"line {line_number}: invalid memory value {memory_text!r}")
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "driver_version": driver_version,
                    "memory_mib": memory_mib,
                }
            )

    return {
        "status": capture["status"],
        "returncode": capture["returncode"],
        "stderr": capture["stderr"].strip(),
        "gpus": rows,
        "parse_errors": parse_errors,
    }


def _fallback_freeze() -> str:
    """List installed distributions when the current interpreter has no pip module."""

    packages = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }
    return "\n".join(sorted(packages, key=str.casefold)) + "\n"


def _collect_package_versions() -> dict[str, Any]:
    """Capture ``pip freeze``, falling back to installed distribution metadata."""

    capture = _run_command(
        (sys.executable, "-m", "pip", "freeze"),
        timeout_seconds=60,
    )
    if capture["status"] == "ok":
        capture["method"] = "pip freeze"
        return capture

    capture["status"] = "fallback"
    capture["method"] = "importlib.metadata"
    capture["stdout"] = _fallback_freeze()
    return capture


def _pip_freeze_text(capture: dict[str, Any]) -> str:
    """Render a pip-freeze capture as a useful artifact even when pip fails."""

    if capture["status"] in {"ok", "fallback"}:
        return capture["stdout"].rstrip() + "\n"
    detail = capture["stderr"].strip() or "pip freeze returned no diagnostic"
    return f"# pip freeze unavailable ({capture['status']}): {detail}\n"


def _determinism_note(report: SeedReport) -> str:
    """Describe the effective deterministic-algorithm policy for run metadata."""

    if not report.deterministic:
        return "Deterministic PyTorch algorithms were disabled for this run."
    if report.warn_only:
        return (
            "Unsupported deterministic PyTorch operations emit warnings and must be recorded "
            "with the affected run."
        )
    return "Unsupported deterministic PyTorch operations abort the run."


def log_run_environment(output_dir: str | Path, report: SeedReport) -> tuple[Path, Path]:
    """Write package, platform, GPU, driver, and seed metadata to a run directory."""

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pip_capture = _collect_package_versions()
    freeze_path = run_dir / _PIP_FREEZE_FILENAME
    freeze_path.write_text(_pip_freeze_text(pip_capture), encoding="utf-8")

    torch = _optional_torch()
    metadata = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "seed": asdict(report),
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "torch": _collect_torch_environment(torch),
        "nvidia_smi": _collect_nvidia_smi(),
        "pip_freeze": {
            "artifact": _PIP_FREEZE_FILENAME,
            "method": pip_capture["method"],
            "status": pip_capture["status"],
            "returncode": pip_capture["returncode"],
            "stderr": pip_capture["stderr"].strip(),
        },
        "determinism_notes": [
            "PYTHONHASHSEED affects hashing only in subsequently started interpreters.",
            _determinism_note(report),
        ],
    }
    environment_path = run_dir / _RUN_ENVIRONMENT_FILENAME
    environment_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return environment_path, freeze_path


def initialize_reproducibility(
    seed: int,
    output_dir: str | Path,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
) -> SeedReport:
    """Seed all supported RNGs and capture the run environment in ``output_dir``."""

    report = seed_everything(seed, deterministic=deterministic, warn_only=warn_only)
    log_run_environment(output_dir, report)
    return report
