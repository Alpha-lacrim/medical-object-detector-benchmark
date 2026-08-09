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

## Session 10 — 2026-08-10 — Batch 3 YOLO11s baseline

**What I did:**
- Confirmed the Batch 1 choice of YOLO11s at small scale and pinned
  `ultralytics==8.4.110`. Before any accepted training, resolved the required
  augmentation asymmetry by explicitly disabling every Ultralytics stochastic
  extra so the primary YOLO and Faster R-CNN arms both use deterministic
  resize-only inputs. Pinned the official pretrained checkpoint at SHA-256
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
- Implemented the strict config-driven YOLO data view, trainer, timing gate,
  checkpointing, early stopping, shared validation, curves, model/FLOP/latency
  profiling, recovery finalizer, and regression tests. The materialized views
  contain exactly 3,500 train images/1,267 boxes and 750 validation images/277
  boxes, including 2,702/581 negative images; the held-out test split was not
  accessed.
- Diagnosed and excluded the pre-benchmark numerical failures. The accepted
  policy uses batch/effective batch 4, 640 pixels, SGD LR 0.001 with one-epoch
  warmup, momentum 0.9, weight decay 0.0005, no Nesterov, bfloat16 AMP with
  float32 assignment/loss, native YOLO BatchNorm updates, ordinary seeded
  shuffle, two workers, and minimum-8/patience-5/maximum-30 validation-mAP
  stopping. All precision, normalization, LR, warmup, and scheduler asymmetries
  are disclosed in `docs/LIMITATIONS.md` and `docs/YOLO_BASELINE.md`.
- Completed the valid three-epoch benchmark in 141.79/134.99/136.20 seconds.
  Its 135.59-second steady epoch projected 18.18 minutes for eight epochs and
  67.90 minutes at the 30-epoch ceiling. Gate artifact SHA-256:
  `c339db91c05b1c8a1398dbbdcc7470ef1fd1932ddf1c374a87529faca45e1587`;
  config SHA-256:
  `5a9bd54c730a42db166d8e5c7075f863f914b5be7c66567f5bc91a70b50ef8d2`.
  The full-run training source identity matches the gate exactly.
- Completed the clean full seed-17 run from pretrained weights. It stopped at
  epoch 15 after patience reached 5/5; epoch 10 is best. Epoch-loop training
  took 1,975.64 seconds (32.93 minutes) and peak allocated VRAM was 1,148.16
  MiB. The native epoch-10 mAP50:95 used for selection was 0.07335.
- Final shared evaluation on all 750 validation images/277 boxes produced AP50
  0.26464 and AP50:95 0.08692. At score 0.25/match IoU 0.50 it produced
  precision 0.57143, recall 0.20217, and F1 0.29867 (56 TP, 42 FP, 221 FN).
  Batch-1 bfloat16 profiling over 100 synchronized images measured 65.24 FPS
  and mean/p50/p95 latency 15.33/14.49/19.82 ms. The model has 9,428,179 total
  and 9,428,163 training-time trainable parameters, 21.42 estimated GFLOPs,
  and a 19,172,819-byte/18.28-MiB checkpoint. Best checkpoint SHA-256:
  `65909164e82c1ef53c0d38e0d898d37bbbec5f46cb9f5cd029e76ba486c0371c`.
- Recovered reporting without retraining after Ultralytics materialized a
  750-path list as one inference batch and exhausted VRAM. The finalizer now
  streams only the audited validation directory, verifies all filenames,
  derives best epoch from immutable `results.csv` because stripped checkpoints
  store epoch `-1`, and records training-source and later reporting-source
  identities separately. Checkpoint copies/hashes, tables, summary, 15-row
  timing log, and the visually inspected best-epoch-10 curve all validate.
  Repository verification is 172 passed/1 expected metadata-only skip; Ruff
  and `git diff --check` pass.

**What's still incomplete / next step:**
- Batch 3 is complete. Do not start Batch 4 until the user reviews this
  one-seed result, curve, shared metrics, compute profile, and disclosed
  asymmetries.
- After approval, Batch 4 may implement the formal unified evaluator and the
  additional-seed headline comparison specified in Phase 5; no Batch 4 work
  has started.

**Needs the user's review before proceeding:**
- Review `results/figures/yolo_training_curves.png`,
  `results/tables/yolo_baseline_validation.csv`, and
  `results/tables/yolo_compute.csv`. In particular, confirm acceptance of the
  epoch-10 checkpoint, the shared AP50:95 0.08692 result, and the documented
  bfloat16/loss/BatchNorm/LR/scheduler asymmetries before Batch 4.

**Files touched:**
- Batch 3 config/code/tests: `configs/yolo.yaml`, `src/models/yolo_*.py`,
  `src/models/train_yolo.py`, and `tests/test_yolo_*.py`.
- Documentation/state: `README.md`, `docs/YOLO_BASELINE.md`,
  `docs/LIMITATIONS.md`, `CODEX.md`, and `HANDOFF.md`.
- Generated artifacts: `results/logs/yolo11s_rsna_seed17_{smoke,benchmark,full}/`,
  `results/checkpoints/yolo11s_rsna_seed17_full/`,
  `results/tables/yolo_{baseline_validation,compute}.csv`, and
  `results/figures/yolo_training_curves.png`; rejected numerical diagnostics
  and the preserved wrapper-collision attempt remain clearly named under
  `results/logs/` and are excluded from results.
- The pre-existing deletion of `Final project 1405.v1.pdf` was left untouched.

---

## Session 9 — 2026-08-09 — Batch 4 (blocked before start)

**What I did:**
- Read the newest `HANDOFF.md` entry, `CODEX.md`, `PROJECT_SPEC.md` §3 and §5 Phases 4–5, and the Batch 3–4 checkpoint language in `BATCHES.md` before taking any Batch 4 action.
- Verified that Batch 3 has not been run: `configs/yolo.yaml` contains only the Batch 1 model/version decision, the augmentation-asymmetry choice remains `TBD`, and no YOLO training implementation, checkpoint, log, curve, or baseline/compute table exists.
- Stopped without creating the unified evaluator or launching extra-seed runs because Batch 4 explicitly depends on the reviewed Batch 3 YOLO config and one-seed baseline; silently implementing Batch 3 inside Batch 4 would violate the repository's phase scope and review gate.

**What's still incomplete / next step:**
- Run Batch 3 first: make and document the augmentation-parity decision, implement YOLO11s training under the matched hardware protocol, complete the one-seed run, and stop for review.
- After Batch 3 is reviewed, rerun Batch 4 to build `src/evaluate.py`, evaluate both models identically, train seeds 2–3 for both detectors, and produce the mean ± std comparison tables.

**Needs the user's review before proceeding:**
- Confirm that Batch 3 should be run next. If the intended order has changed, explicitly resolve the conflict with the recorded sequential review gates before requesting Batch 4 again.

**Files touched:**
- `HANDOFF.md`

---

## Session 8 — 2026-08-09 — Batch 2 full Faster R-CNN baseline

**What I did:**
- Received explicit approval of Session 7's timing estimate and launched the documented full seed-17 command with the exact approved `benchmark_estimate.json`. Train mode independently revalidated the config, data, source, Python/Torch/CUDA/driver/GPU, AMP, batch, BatchNorm, and resolution identities, then restarted from COCO weights rather than timing-run weights.
- Completed 11 full train-plus-validation epochs and stopped by the configured validation-AP50:95 rule (minimum 8, patience 5, maximum 30). The scheduler reduced LR from 0.005 to 0.0005 at epoch 10. Epoch 6 is best: AP50:95 0.12764, AP50 0.33144, precision 0.14138, recall 0.68953, F1 0.23464 on all 750 validation images/277 boxes. The held-out test split was not accessed.
- Recorded 7,017.8 seconds (1.95 hours) of epoch-loop training and 1,556.6 MiB peak allocated GPU memory. Best-checkpoint batch-1 float16 AMP profiling over 100 synchronized batches measured 11.00 FPS and mean/p50/p95 latency 90.92/90.78/92.18 ms. The model has 43,256,153 total parameters, 43,030,809 trainable parameters, 450.76 estimated GFLOPs under the documented registered-operation convention, and a 173,412,170-byte/165.38-MiB checkpoint.
- Produced and validated the full CSV/JSONL history, summary, validation table, compute table, four-panel training curve, exact best checkpoint, and last restart state. All recorded sizes/hashes match the files; the visually inspected curve correctly marks epoch 6 and the epoch-10 LR reduction. Best checkpoint SHA-256: `9ec35c5d761f8e4bf7a43f7999f388ac1ffc0d533f62746409db280706dffab4`.

**What's still incomplete / next step:**
- Batch 2 is complete. Do not start Batch 3 until the user reviews the Faster R-CNN validation curves, final benchmark metrics, compute measurements, and one-seed limitation.
- Once approved, Batch 3 must implement only the pinned YOLO11s arm and make the required augmentation-asymmetry decision before training.

**Needs the user's review before proceeding:**
- Review `results/figures/faster_rcnn_training_curves.png`, `results/tables/faster_rcnn_baseline_validation.csv`, and `results/tables/faster_rcnn_compute.csv`, including the low-precision/high-recall operating point and validation variability. Approve or request changes before Batch 3.

**Files touched:**
- Generated artifacts: `results/logs/faster_rcnn_rsna_seed17_full/`, `results/checkpoints/faster_rcnn_rsna_seed17_full/{best_model,last_state}.pt`, `results/tables/faster_rcnn_{baseline_validation,compute}.csv`, `results/figures/faster_rcnn_training_curves.png`.
- Documentation/state: `docs/FASTER_RCNN_BASELINE.md`, `CODEX.md`, `HANDOFF.md`.

---

## Session 7 — 2026-08-09 — Batch 2 recovery and timing gate

**What I did:**
- Verified the official Kaggle aggregate ZIP (3,932,287,530 bytes; SHA-256 `133acacf95aa68c4d219124b17937f31cec073052096b9f9b122180df9d9af18`) by full CRC/path audit: 26,684 train DICOMs, 3,000 competition-test DICOMs, and exact official CSV hashes. Regenerated all 5,000 selected PNGs with zero missing/errors; the 12 earlier review conversions were byte-identical.
- Adopted the user-authorized local `torch-gpu` runtime before timed work: Python 3.11.15, Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, CUDA 12.4, and driver 610.47. Aligned dependency manifests/docs, verified CUDA NMS/AMP/FLOP support, passed preflight on all 3,500 train and 750 validation images, passed 160 tests with one expected metadata-only skip, passed Ruff, and completed the bounded smoke test.
- Diagnosed a first official-data attempt that finished all epoch-1 training batches but failed before validation with Windows `WinError 1455`: persistent training workers overlapped the new validation pool and exceeded the 16 GB host commit limit. It produced no epoch record/checkpoint. Preserved the failed metadata, kept six workers, changed train/validation pools to non-persistent, documented the decision, updated the pinned config hash, reran tests/smoke, and restarted cleanly from COCO weights.
- Completed exactly three clean benchmark epochs in 770.3/551.5/473.0 seconds (29.91 minutes total). Steady-state time is 512.2 seconds/epoch; eight epochs project to 1.21 hours and the 30-epoch upper bound to 4.34 hours (conservative 4.02--4.66-hour range). Peak allocated GPU memory was 1,556.6 MiB. Epoch 3 diagnostic validation AP50:95 was 0.10993; it is not a final result.
- Validated all CSV/JSONL/summary/projection invariants, both timing checkpoints, dataset/execution/implementation identities, and the same approval function used by train mode. The competition test split was not accessed. Approval artifact: `results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json`, SHA-256 `232460ae09827dfb780b0f5c6506bf9f545bbdc0e1483082c2c440035e8e8e8b`; config SHA-256 `ef1e3ebe1fbe3cf1a6e27bf8b9c12f61719c2ea8771c9758f64dc278dd0e2633`.

**What's still incomplete / next step:**
- Full one-seed Faster R-CNN training has not started. After explicit approval, run the documented `--mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json` command; early stopping uses validation AP50:95 with minimum 8, patience 5, maximum 30 epochs.
- After the full run, verify final metrics/curves/FPS/parameters/GFLOPs/model size/checkpoint/training time and stop again for Batch 2 result review. Do not start Batch 3 yet.

**Needs the user's review before proceeding:**
- Approve or reject the measured 4.34-hour maximum estimate (4.02--4.66-hour conservative range), bound to the artifact/hash above. No full training may start without explicit approval.

**Files touched:**
- Runtime/dependency state: `.python-version`, `requirements.txt`, `pyproject.toml`, `uv.lock`, `README.md`.
- Official-data provenance/docs: `data/README.md`, `data/manifests/rsna-pneumonia-5000-audit.json`, `docs/DATASHEET.md`.
- Batch 2 config/code/tests/docs: `configs/faster_rcnn.yaml`, `src/models/`, `src/meddet_benchmark/`, `src/utils/seed.py`, corresponding tests, `docs/FASTER_RCNN_BASELINE.md`, `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`, `CODEX.md`, `HANDOFF.md`.
- Local ignored artifacts: official raw/extracted DICOMs, 5,000 processed PNGs/COCO files, smoke/benchmark log directories, timing checkpoints, and preserved aborted-attempt metadata directories. The pre-existing deletion of `Final project 1405.v1.pdf` was left untouched.

---

## Session 6 — 2026-08-04 — Batch 2 Kaggle OAuth retry

**What I did:**
- Reopened Kaggle OAuth after the user completed browser authentication and kept the local callback alive through authorization.
- Received the browser authorization code successfully, but Kaggle denied the subsequent OAuth token exchange with HTTP 403. Verified that the CLI remains on `LEGACY_API_KEY`, the official archive is absent, and the checkout still contains only 12 DICOMs and 12 processed PNGs.
- Made no Torch-environment substitution and did not start conversion, smoke testing, or training.

**What's still incomplete / next step:**
- From an authorized available network/location, manually download the official `stage_2_train_images.zip`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` through the authenticated Kaggle website and place them in `data/raw/rsna-pneumonia/`.
- Explicitly authorize either repinning to the verified existing Anaconda Python 3.11 / Torch 2.6.0+cu124 / Torchvision 0.21.0+cu124 environment, or retaining and redownloading the current Python 3.13 / Torch 2.13.0+cu130 pins.
- Then verify sources, prepare 5,000 PNGs, run tests/preflight/smoke, and run exactly three benchmark epochs.

**Needs the user's review before proceeding:**
- Confirm when the three official files are present and state the environment choice. Kaggle API retries are not useful while OAuth token exchange itself is denied.

**Files touched:**
- `CODEX.md`, `HANDOFF.md`

---

## Session 5 — 2026-08-04 — Batch 2 recovery

**What I did:**
- Resumed the interrupted setup, verified no installer remained active, and confirmed that the project `.venv` still lacks Torch/Torchvision while the interrupted `uv` cache contains 3.81 GB that cannot resolve the pinned wheels offline.
- Audited the user's Anaconda `torch-gpu` environment read-only: Python 3.11.15, Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, working RTX 4060 CUDA, Torchvision CUDA NMS, and AMP. Did not adopt it because the recorded project identity is Python 3.13 / Torch 2.13.0+cu130 / Torchvision 0.28.0+cu130.
- Installed only `kaggle==2.2.3` into `.venv`, validated the credential structure, and preserved the review-source metadata before attempting the official download. Kaggle returned HTTP 403 before downloading any bytes; public dataset lists and competition lists/files also return 403, so the failure is API-wide rather than an absent archive or downloader bug.
- Attempted Kaggle's forced browser OAuth flow; it remained pending without completing and was cancelled cleanly. Restored the two review-source CSVs and verified their exact recorded SHA-256 hashes. No official archive, new DICOM, processed PNG, smoke artifact, or benchmark artifact was produced.

**What's still incomplete / next step:**
- Complete a fresh Kaggle OAuth/login for account `alphalacrim`, or manually download the official `stage_2_train_images.zip`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` into `data/raw/rsna-pneumonia/`.
- Explicitly choose whether to retain the pinned Python 3.13 / Torch 2.13.0+cu130 stack and redownload it, or authorize repinning the experiment to the verified existing Python 3.11 / Torch 2.6.0+cu124 / Torchvision 0.21.0+cu124 Anaconda stack.
- After those two prerequisites, convert and verify exactly 5,000 PNGs, run tests/preflight/smoke, then run exactly three complete benchmark epochs and stop for timing approval.

**Needs the user's review before proceeding:**
- Resolve Kaggle authentication/manual official-file placement and make the explicit environment-identity choice above. The benchmark cannot start with mirror pixels or a silently substituted Torch stack.

**Files touched:**
- `CODEX.md`, `HANDOFF.md`
- Local ignored environment/data state: `.venv` gained `kaggle==2.2.3`; review CSVs were temporarily moved and then restored byte-identically.

---

## Session 4 — 2026-08-02 — Batch 3 (blocked before start)

**What I did:**
- Read `HANDOFF.md`, `CODEX.md`, `PROJECT_SPEC.md` §3 and §5 Phase 4, and the Batch 2–4 checkpoint language in `BATCHES.md` before taking any Batch 3 action.
- Rechecked the recorded prerequisites: only 12 of the fixed 5,000 processed PNGs are present, and `.venv` has no Torch, Torchvision, or Ultralytics installation.

**What's still incomplete / next step:**
- Batch 2 still needs the official selected pixels, CUDA dependencies, its complete three-epoch timing gate, explicit approval, and the one-seed full Faster R-CNN run.
- Batch 3 must remain unstarted until the user resolves or explicitly overrides that phase-order conflict; no YOLO config, augmentation decision, implementation, or training artifact was created in this session.

**Needs the user's review before proceeding:**
- Provide/authorize the missing official RSNA pixels and dependency installation, and direct completion of Batch 2 first; or explicitly override the recorded Batch 2 review gate and accept that Batch 3 still cannot complete a real-data training run until the same data/dependency blockers are resolved.

**Files touched:**
- `HANDOFF.md`

---

## Session 3 — 2026-08-02 — Batch 2

**What I did:**
- Implemented the strict `fasterrcnn_resnet50_fpn_v2` Batch 2 configuration and pipeline: canonical COCO train/validation adapter, derived class mapping, negative-image support, RGB tensor conversion, transfer-learning head replacement, float16 AMP, physical batch 2 with accumulation 2, frozen BatchNorm statistics, SGD/plateau scheduling, exact-best checkpoints, and validation-AP early stopping.
- Reused the shared COCO/operating-point evaluator so every epoch records loss components, precision, recall, F1, AP50, and AP50:95. Added configured atomic CSV/JSONL logs, best/last state checkpoints, validation/compute tables, four-panel curves, synchronized FPS/latency, parameter counts, mandatory GFLOPs, checkpoint size/hash, peak GPU memory, and training time.
- Made the three-epoch gate representative and hard to reuse accidentally: its epoch durations include equivalent checkpoint I/O, and full-run approval compares exact YAML, train/validation annotation and pixel manifests, implementation sources, dependency/CUDA/driver/GPU identity, AMP, batch, BatchNorm policy, and resolution. The full run always restarts from COCO weights.
- Added an idempotent `--mode finalize` path so profiling/table/plot failures after optimization can be recovered from the saved best checkpoint. Corrected exact-best versus early-stopping patience semantics, learning-rate logging, operating-point prediction counts, DataLoader worker-pool cleanup, Windows path containment, full image decoding, crowd rejection, and clean package-relative imports.
- Audited the real local data without touching test annotations: train is 7/3,500 images with 1,267 boxes (3,493 missing), validation is 5/750 with 277 boxes (745 missing). The fixed 5,000-study set is still missing 4,988 images overall. Benchmark/full modes now fail before Torch/CUDA with all exact missing train/validation paths.
- Added Batch 2 protocol/reproduction documentation and exact Pydantic/Matplotlib pins. Verified 160 tests pass, one Torch tensor-contract test is skipped because Torch is absent, Ruff passes, `git diff --check` is clean, and the RTX 4060 Laptop GPU is visible with 8,188 MiB VRAM and driver 610.47. No model training was started.

**What's still incomplete / next step:**
- Provide authorized Kaggle access (accepted competition rules plus `KAGGLE_USERNAME`/`KAGGLE_KEY` or `C:\Users\Pouyan\.kaggle\kaggle.json`), or place the official Stage 2 archive/DICOMs locally; then run configured conversion and confirm all 5,000 PNGs.
- Complete the pinned CUDA Torch 2.13.0/Torchvision 0.28.0 installation. Two attempts made slow partial-download progress but did not install either package; then run the bounded CUDA/AMP/spawn-DataLoader smoke check and the currently skipped tensor test.
- Run exactly three complete benchmark epochs, report the measured time projection, and stop for user sign-off. Only after approval may the one-seed full run, final profiling/tables/curves, final CODEX metrics, and checkpoint/training-time record be produced.

**Needs the user's review before proceeding:**
- No training-time estimate exists yet: a 12-image estimate would be misleading and violates the complete-real-data gate. The user must supply/authorize the official pixels, then review and explicitly approve the real three-epoch estimate before full training.
- No Batch 3 work may start; Batch 2 still requires the approved full run and later review of its curves and compute numbers.

**Files touched:**
- `configs/faster_rcnn.yaml`, `requirements.txt`, `README.md`
- `src/models/{__init__,faster_rcnn_config,faster_rcnn_data,faster_rcnn_model,faster_rcnn_training,faster_rcnn_evaluation,faster_rcnn_reporting,train_faster_rcnn}.py`
- `tests/test_faster_rcnn_{config,data,model,training,reporting}.py`, `tests/test_train_faster_rcnn.py`
- `src/meddet_benchmark/{__init__,__main__,coco_evaluation}.py`
- `docs/FASTER_RCNN_BASELINE.md`, `docs/LIMITATIONS.md`
- `CODEX.md`, `HANDOFF.md`

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
