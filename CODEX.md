# CODEX.md — Living Project Context

> This is *not* the requirements doc — see `PROJECT_SPEC.md` for that, it doesn't change. This file tracks *current state*: decisions actually made, files that actually exist, and where things are. Update it every session per `AGENTS.md`. Never delete history from the Decisions Log — append and mark superseded entries as such if a decision changes.

## Project one-liner

Controlled comparative study of Faster R-CNN vs. a modern YOLO version on a medical imaging detection dataset, evaluated on accuracy, robustness to corruption, and Grad-CAM explainability, with statistical significance testing. Full spec: `PROJECT_SPEC.md`.

## Hardware (fixed, see PROJECT_SPEC.md §3)

ASUS ROG Strix G16 — Intel i7-13650HX, RTX 4060 Laptop GPU (8GB VRAM), 16GB DDR5-4800 RAM.

## Decisions Log

*(Newest first. Each entry: what was decided, why, which session/batch, and the file it's documented in.)*

- **Phase 0 reproducibility contract:** every experiment entry point must call `initialize_reproducibility(seed, output_dir)` before CUDA initialization. Each run records `pip_freeze.txt` plus structured seed, platform, Torch/CUDA, GPU, and driver metadata in `run_environment.json`. Deterministic algorithms use warning mode so unsupported kernels are visible without silently changing the run. See `src/utils/seed.py` and `docs/REPRODUCIBILITY.md` (Session 1 / Batch 0).
- **Batch 0 dependency baseline:** `numpy==2.5.1`, `torch==2.13.0`, `torchvision==0.28.0`, `pycocotools==2.0.11`, `albumentations==2.0.8`, `scipy==1.18.0`, and `pytest==9.1.1` are pinned in `requirements.txt`. Ultralytics deliberately remains a commented placeholder until Batch 1 selects the YOLO generation (Session 1 / Batch 0).
- **Dataset choice:** `TBD` — pending Batch 1. Candidates: RSNA Pneumonia Detection Challenge 2018 (single-class, ~26k images), Kaggle Medical Image Dataset: Brain Tumor Detection, Kaggle MRI for Brain Tumor with Bounding Boxes. See `PROJECT_SPEC.md` §1 for links and selection criteria.
- **Actual class list / count:** `TBD` — depends on dataset choice above. Do not assume 5 classes (see `PROJECT_SPEC.md` §0 on the "verify five classes" placeholder).
- **YOLO version pinned:** `TBD` — pending the Batch 1 literature review. Candidates per `PROJECT_SPEC.md` §5 Phase 1: YOLO11 (anchor-free, more literature-comparable) vs. YOLO26 (current SOTA, NMS-free, less studied).
- **YOLO augmentation-asymmetry handling:** `TBD` — pending Batch 3. Must be explicitly decided (disable extras vs. document as validity threat) per `PROJECT_SPEC.md` §5 Phase 4, not left implicit.
- **Patient-level split strategy:** `TBD` — pending Batch 1, depends on whether the chosen dataset has multiple images per patient/study.

## File Map

*(Update as files are created. This is Codex's map back to itself across sessions.)*

| Path | Purpose | Status |
|---|---|---|
| `PROJECT_SPEC.md` | Full requirements/spec | done |
| `AGENTS.md` | Session bootstrap + protocol | done |
| `CODEX.md` | This file | living |
| `HANDOFF.md` | Session log | living |
| `BATCHES.md` | Batch instruction sequence | done |
| `README.md` | Batch 0 status, requirements-based setup, and reproducibility entry point | done |
| `.gitignore` | Excludes raw/processed data, credentials, generated artifacts, and checkpoints while retaining skeleton placeholders | done |
| `requirements.txt` | Exact Batch 0 dependency pins; Ultralytics version intentionally deferred | done |
| `configs/dataset.yaml` | Decision-gated dataset-config placeholder | skeleton |
| `configs/faster_rcnn.yaml` | Future Faster R-CNN config placeholder | skeleton |
| `configs/yolo.yaml` | Future YOLO config and Ultralytics-version placeholder | skeleton |
| `data/raw/`, `data/processed/` | Gitignored dataset storage with tracked directory placeholders | skeleton |
| `data/splits/` | Committed split-manifest directory | skeleton |
| `src/data/`, `src/models/` | Future dataset and detector modules; no implementation added in Batch 0 | skeleton |
| `src/robustness/`, `src/explainability/`, `src/stats/` | Future evaluation-phase module directories | skeleton |
| `src/utils/seed.py` | Python/NumPy/PyTorch seeding, deterministic settings, and per-run environment capture | done |
| `src/utils/__init__.py` | Public exports for reproducibility utilities | done |
| `notebooks/` | EDA-only notebook directory | skeleton |
| `results/tables/`, `results/figures/`, `results/logs/`, `results/checkpoints/` | Experiment-output directories; checkpoints ignored | skeleton |
| `report/report.md` | Final report placeholder | skeleton |
| `tests/test_seed.py` | RNG repeatability, validation, Torch configuration, metadata, and package-inventory fallback tests | done |
| `docs/DATASET_CHOICE.md`, `docs/LITERATURE_REVIEW.md`, `docs/LIMITATIONS.md` | Decision-gated documentation placeholders | skeleton |
| `docs/REPRODUCIBILITY.md` | Phase 0 determinism and run-metadata contract | done |
| `pyproject.toml`, `uv.lock`, `src/meddet_benchmark/`, existing tests/configs | Pre-existing implementation from earlier Git history; untouched by Batch 0 and not authoritative for deferred decisions | reconcile before use |

## Current phase

**Batch 0 complete; stopped for user review.** Next action, only after approval: Batch 1 dataset investigation, literature review, choice, and preparation.

## Known open issues / risks

- Kaggle datasets require API credentials — download script must read from environment/`~/.kaggle/kaggle.json`, never hardcoded (Batch 1).
- RSNA dataset (~26k images) likely needs stratified subsampling to be tractable on this hardware within project timeframe — decision folded into the Batch 1 dataset choice.
- YOLO's built-in augmentations vs. Faster R-CNN's pipeline is an unresolved fairness confound until Batch 3 makes an explicit call.
- This checkout already had Git history and implementation files beyond Batch 0. They were deliberately left untouched. In particular, the pre-existing `pyproject.toml`/`uv.lock` contain an old `ultralytics==8.4.108` pin that conflicts with the current `TBD` decision; use `requirements.txt` as the Batch 0 dependency baseline and reconcile/remove the legacy manifests before relying on them.
- The existing `.venv` omits the `pip` module. Run metadata attempts `python -m pip freeze` first and records an `importlib.metadata` name/version inventory plus the pip diagnostic when that command is unavailable.
- Git was already initialized with 12 commits before Batch 0, so the Batch 0 commit is the first commit for this workflow, not the repository's root commit.
- `AGENTS.md`, `CODEX.md`, and `HANDOFF.md` were already supplied as authoritative replacements at session start, despite differing substantially from the legacy versions in `HEAD`. This Batch 0 commit records that workflow reset; the displaced history remains recoverable from earlier Git commits.
