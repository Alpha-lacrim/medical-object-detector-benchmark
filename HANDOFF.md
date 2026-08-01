# HANDOFF.md — Session Log

> Read the top entry at the **start** of every session, before doing anything else — that's the state you're picking up from. Write a new entry at the **end** of every session, or right before stopping for the user's review — not at the start, since nothing's happened yet at that point. Newest entry goes on top, below the template.

---

## Template for new entries

```
## Session N — <date> — Batch <X>

**What I did:**
-
-

**What's still incomplete / next step:**
-

**Needs the user's review before proceeding:**
-

**Files touched:**
-
```

---

## Session 2 — 2026-08-02 — Batch 1

**What I did:**
- Inspected all three linked dataset candidates for advertised/verified image inventory, class semantics, annotation format/integrity, balance, license terms, and recoverable patient/study identity. Selected RSNA Stage 2 and documented why the two MRI exports cannot establish patient-disjoint splits.
- Selected YOLO11s with `ultralytics==8.4.110`; wrote the detector/Grad-CAM/medical-detection/corruption literature review and a 22-entry BibTeX file.
- Added a secret-safe Kaggle downloader that reads a complete environment-variable pair or `~/.kaggle/kaggle.json`/`KAGGLE_CONFIG_DIR`, redacts secrets, and distinguishes missing credentials, unaccepted rules, and geographic storage failures.
- Audited all 30,227 RSNA annotation rows: 26,684 valid studies, 9,555 valid boxes, zero malformed/non-positive/off-image/duplicate boxes, and zero target/class/mapping inconsistencies. Verified and recorded input SHA-256 values.
- Recovered true NIH patient keys from the official RSNA mapping. Deterministically selected 5,000 studies from 2,136 patient groups and created exact 3,500/750/750 splits with zero patient overlap. Wrote committed CSV manifests and generated per-split canonical COCO JSON.
- Converted 12 authentic review DICOMs, generated the split-distribution and labeled bounding-box sample figures, visually checked box placement, and wrote the dataset-choice report, datasheet, and limitations.
- Verified 80 repository tests pass, Ruff passes over `src`/`tests`, `git diff --check` is clean, and a repeated real-data preparation produces identical hashes for all three manifests and all three COCO files. No model training or Batch 2 implementation was started.

**What's still incomplete / next step:**
- Stop here until the user approves the RSNA choice, one-class task definition, patient-safe 5,000-study split, and EDA.
- Before Batch 2 can train, use authorized Kaggle credentials to acquire the other 4,988 selected DICOMs, rerun conversion, and compare the official-download CSV hashes with the committed audit.
- Batch 2 is Faster R-CNN only after this review gate. YOLO augmentation-parity handling remains deferred to Batch 3.

**Needs the user's review before proceeding:**
- Approve or reject the RSNA selection and fixed 5,000-study hardware scope.
- Review `results/figures/rsna_class_distribution.png` and `results/figures/rsna_annotation_samples.png`, plus the disclosed interim mirror provenance/full-image-download requirement in `docs/DATASHEET.md`.
- Confirm acceptance of the actual class map (`Lung Opacity` only) and the NIH-patient-group split strategy.

**Files touched:**
- `README.md`, `requirements.txt`, `configs/{dataset,yolo}.yaml`, `data/README.md`
- `src/data/{__init__,download,prepare,visualize}.py`
- `tests/test_{download,prepare,visualize}.py`
- `docs/{DATASET_CHOICE,DATASHEET,LITERATURE_REVIEW,LIMITATIONS}.md`, `report/references.bib`
- `data/manifests/rsna-pneumonia-5000-audit.json`, `data/splits/rsna-pneumonia-5000/*.csv`
- `results/figures/rsna_{class_distribution,annotation_samples}.png`, `results/figures/rsna_eda_summary.json`
- `CODEX.md`, `HANDOFF.md`

---

## Session 1 — 2026-08-02 — Batch 0

**What I did:**
- Created the §4 repository skeleton with decision-gated config/document placeholders and tracked empty directories; raw/processed data and checkpoints remain gitignored.
- Added exact dependency pins in `requirements.txt`, leaving Ultralytics as a commented Batch 1 placeholder.
- Implemented `src/utils/seed.py` to seed Python, NumPy, and available PyTorch/CUDA RNGs; enable deterministic behavior; and write package, platform, GPU, and driver metadata into every supplied run directory.
- Added `tests/test_seed.py` and `docs/REPRODUCIBILITY.md`. Verified 50 repository tests pass, Ruff passes on all new Python files, and a local smoke run detects the RTX 4060 Laptop GPU with driver 610.47.
- Preserved all pre-existing dataset/evaluator/corruption/model-adjacent files without modification. The repo was already Git-initialized with 12 commits, so no reinitialization or history rewrite was attempted.
- Treated the user-supplied `AGENTS.md`, `CODEX.md`, and `HANDOFF.md` replacements as the authoritative workflow reset. Their legacy content remains recoverable from earlier commits. Normalized the two memory filenames to uppercase for case-sensitive checkouts and replaced the stale README with Batch 0 instructions.

**What's still incomplete / next step:**
- Batch 1 must perform the literature review, select and pin the Ultralytics/YOLO version, investigate the three datasets, and stop again for dataset/EDA review.
- Reconcile the pre-existing `pyproject.toml`/`uv.lock` and `src/meddet_benchmark/` implementation before using them; they predate this workflow and include an Ultralytics pin that is not an approved project decision.

**Needs the user's review before proceeding:**
- Approve the skeleton, dependency versions, and `initialize_reproducibility(seed, output_dir)` artifact contract before Batch 1.
- Note that this checkout already had Git history; the new Batch 0 commit is not a literal root commit.

**Files touched:**
- `.gitignore`, `README.md`, `requirements.txt`, `configs/{dataset,faster_rcnn,yolo}.yaml`
- `data/{raw,processed,splits}/`, `src/{data,models,robustness,explainability,stats}/`, `notebooks/`, `results/{tables,figures,logs,checkpoints}/`
- `src/utils/{__init__,seed}.py`, `tests/test_seed.py`
- `docs/{DATASET_CHOICE,LITERATURE_REVIEW,LIMITATIONS,REPRODUCIBILITY}.md`, `report/report.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 0 — (not yet run)

No sessions completed yet. First session starts with **Batch 0** from `BATCHES.md`.
