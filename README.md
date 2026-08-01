# Medical Object Detector Benchmark

Controlled comparison of Faster R-CNN and one modern Ultralytics YOLO detector on a medical
imaging detection dataset. The study will compare predictive performance, computational cost,
robustness to corruption, Grad-CAM explainability, and statistical uncertainty. The authoritative
requirements and batch checkpoints are in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) and
[`BATCHES.md`](BATCHES.md).

## Current status

Batch 0 infrastructure is complete and awaiting review. The repository tree, exact dependency
baseline, reproducibility utility, run-environment capture, and unit tests are in place. No dataset
has been selected or downloaded, no detector has been implemented or trained in this workflow,
and no experimental result has been produced.

The dataset, class map, patient-level split policy, and YOLO generation are deliberately `TBD`.
Ultralytics therefore remains a commented placeholder in `requirements.txt` until the Batch 1
literature review records and justifies an exact version.

## Setup and verification

The project targets Python 3.13 on Windows. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

The existing `pyproject.toml`, `uv.lock`, and `src/meddet_benchmark/` predate this phased workflow
and contain deferred decisions. Do not use them as the Batch 0 dependency or implementation
baseline until they are reconciled in a later reviewed batch.

## Reproducibility contract

Every experiment entry point must initialize RNGs and write its environment snapshot before CUDA
initialization:

```python
from pathlib import Path

from src.utils.seed import initialize_reproducibility

initialize_reproducibility(seed=17, output_dir=Path("results/logs/example_run"))
```

The run directory receives `pip_freeze.txt` and `run_environment.json`, which record the seed,
determinism settings, package versions, Python/platform details, and detected CUDA/GPU/driver
information. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for DataLoader worker setup
and determinism limitations.

## Session workflow

Before each batch, read:

- [`AGENTS.md`](AGENTS.md) for the standing protocol;
- [`CODEX.md`](CODEX.md) for current decisions and the file map; and
- [`HANDOFF.md`](HANDOFF.md) for the newest session state.

Raw/processed datasets, credentials, model weights, and checkpoints must not be committed. This
repository is a research benchmark and does not establish clinical validity.
