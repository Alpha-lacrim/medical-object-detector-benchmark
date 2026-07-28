"""Deterministic seeding without requiring Torch in lightweight environments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

_MAX_NUMPY_SEED = 2**32 - 1


@dataclass(frozen=True)
class ReproducibilityReport:
    seed: int
    deterministic: bool
    warn_only: bool
    allow_tf32: bool
    torch_available: bool
    cuda_available: bool
    hash_seed_applies_on_restart: bool = True


def _optional_torch() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            return None
        raise
    return torch


def configure_reproducibility(
    seed: int,
    *,
    deterministic: bool,
    warn_only: bool = False,
    allow_tf32: bool = False,
) -> ReproducibilityReport:
    """Seed supported libraries and configure deterministic Torch behavior.

    ``PYTHONHASHSEED`` affects child processes but cannot retroactively change
    hashing in the current interpreter.
    """

    if not 0 <= seed <= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_NUMPY_SEED}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["MEDDET_BASE_SEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch = _optional_torch()
    if torch is None:
        return ReproducibilityReport(
            seed=seed,
            deterministic=deterministic,
            warn_only=warn_only,
            allow_tf32=allow_tf32,
            torch_available=False,
            cuda_available=False,
        )

    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    return ReproducibilityReport(
        seed=seed,
        deterministic=deterministic,
        warn_only=warn_only,
        allow_tf32=allow_tf32,
        torch_available=True,
        cuda_available=cuda_available,
    )


def seed_worker(worker_id: int) -> None:
    """Seed a spawned data-loader worker; kept top-level for Windows pickling."""

    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    torch = _optional_torch()
    if torch is not None:
        worker_seed = int(torch.initial_seed() % (2**32))
    else:
        base_seed = int(os.environ.get("MEDDET_BASE_SEED", "0"))
        worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
