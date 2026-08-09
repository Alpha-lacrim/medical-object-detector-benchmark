# CODEX.md — Living Project Context

> `PROJECT_SPEC.md` is the static source of truth. This file records decisions
> actually made, current files, phase state, and open risks. Read it and the
> newest `HANDOFF.md` entry at the start of every session.

## Project one-liner

Controlled study of Faster R-CNN versus YOLO11s on medical object detection,
evaluated for accuracy, compute, common-corruption robustness, explainability,
and statistical uncertainty under one unified protocol.

## Fixed hardware

ASUS ROG Strix G16: Intel i7-13650HX, RTX 4060 Laptop GPU (8 GB VRAM),
16 GB DDR5-4800 RAM. AMP remains mandatory for both detectors.

## Decisions Log

Newest entries appear first; superseded decisions remain recorded.

- **YOLO11s one-seed baseline complete:** the exact benchmark-approved seed-17
  run restarted from pinned COCO weights and early-stopped at epoch 15 after
  1,975.64 seconds (32.93 minutes); epoch 10 is the selected checkpoint. Shared
  validation on all 750 images/277 boxes produced AP50 0.26464, AP50:95
  0.08692, precision 0.57143, recall 0.20217, and F1 0.29867 at score 0.25 and
  match IoU 0.50. Peak allocated training memory was 1,148.16 MiB. Batch-1
  bfloat16 profiling measured 65.24 FPS and mean/p50/p95 latency
  15.33/14.49/19.82 ms; the model has 9,428,179 total and 9,428,163 training-
  time trainable parameters, 21.42 estimated GFLOPs, and an 18.28 MiB best
  checkpoint. Best checkpoint SHA-256:
  `65909164e82c1ef53c0d38e0d898d37bbbec5f46cb9f5cd029e76ba486c0371c`.
  Ultralytics' native epoch-10 mAP50:95 was 0.07335 and is retained only as the
  checkpoint-selection metric; headline validation uses the shared evaluator.
  All artifacts validate and the test split was not accessed. Stop for Batch 3
  review before Batch 4 (Session 10 / Batch 3).
- **YOLO three-epoch timing gate complete:** the accepted official-data
  benchmark completed in 141.79, 134.99, and 136.20 seconds. The 135.59-second
  steady-state estimate projected 18.18 minutes for eight epochs and 67.90
  minutes for the 30-epoch ceiling. Approval artifact:
  `results/logs/yolo11s_rsna_seed17_benchmark/benchmark_estimate.json`, SHA-256
  `c339db91c05b1c8a1398dbbdcc7470ef1fd1932ddf1c374a87529faca45e1587`;
  config SHA-256
  `5a9bd54c730a42db166d8e5c7075f863f914b5be7c66567f5bc91a70b50ef8d2`.
  The full run's recorded training source identity matches this gate exactly
  (Session 10 / Batch 3).
- **YOLO finalization recovery is weight-preserving and auditable:** training
  completed before reporting encountered an OOM caused by Ultralytics treating
  a 750-path Python list as one inference batch. Finalization now streams the
  exact audited validation directory with batch 4 and verifies every filename.
  Best-epoch reporting is derived from immutable `results.csv` because stripped
  Ultralytics checkpoints record epoch `-1`. The final summary preserves the
  benchmark-approved training-source hash and separately records the corrected
  reporting-source hash; neither fix changed or resumed training (Session 10 /
  Batch 3).
- **YOLO native BatchNorm updates restored before the valid benchmark/training:**
  bfloat16 plus float32 loss kept arithmetic stable, but forced frozen BN still
  drove the head to zero scores during epoch 3. YOLO therefore updates its
  native BN statistics for COCO-to-radiograph adaptation; Faster R-CNN retains
  its frozen-normalization backbone. This architecture-specific asymmetry is
  disclosed (Session 10 / Batch 3, before valid training).
- **YOLO bfloat16 AMP adopted before the valid benchmark/training:** casting
  only target assignment/loss to float32 was insufficient because float16 head
  logits underflowed before reaching the loss (epoch 2, batch 229; AMP scale
  0.0625). Use RTX-4060-supported bfloat16 autocast for model forward/backward
  and float32 assignment/loss. AMP remains mandatory; ordinary seeded shuffle,
  batch, data, augmentation, and model remain unchanged. Faster R-CNN uses
  float16, so this precision asymmetry is disclosed (Session 10 / Batch 3,
  before valid training).
- **YOLO AMP validation adapted for bfloat16 before training:** Ultralytics'
  built-in probe is tied to its default float16 output-equivalence tolerance
  and disabled AMP when bfloat16 was selected. The custom trainer substitutes
  an RTX/CUDA bfloat16 support-and-active-dtype gate; a real GPU smoke plus
  permanent per-batch non-finite and zero-loss guards validate the actual path,
  and training still aborts if AMP is off (Session 10 / Batch 3, before valid
  training).
- **YOLO float32 loss under AMP adopted before the valid benchmark/training:**
  a positive-spread batch diagnostic still collapsed at epoch 2, batch 138,
  proving ordering was not the root cause. Ultralytics task assignment used
  float16 sigmoid scores that underflowed to zero on the one-class head. This
  initially retained float16 forward/backward; the later bfloat16 decision
  above supersedes that detail. Compute assignment and detector losses in
  float32, and restore ordinary seeded shuffle. The
  discarded sampler and all invalid runs are excluded; mixed loss precision
  is documented (Session 10 / Batch 3, before valid training).
- **YOLO target LR reduced to 0.001 before the valid benchmark/training:** the
  one-epoch warmup to 0.005 removed `NaN` but the classifier saturated to
  near-zero scores and epochs 2--3 had all-zero losses/mAP. Retain SGD,
  momentum 0.9, weight decay 0.0005, no Nesterov, and the one-epoch zero-to-
  target warmup, but use target LR 0.001. A per-batch guard now rejects both
  non-finite loss and five consecutive exactly-zero classification losses.
  This is disclosed as an optimizer asymmetry; invalid diagnostics are
  archived and excluded (Session 10 / Batch 3, before valid training).
- **YOLO one-epoch LR warmup adopted before the valid benchmark/training:**
  no-warmup full-data diagnostics reproducibly produced `NaN` classification
  loss at epoch 1, batch 29 while the AMP scale collapsed to 64. Use a linear
  one-epoch ramp from 0 to the same 0.005 target, with momentum fixed at 0.9;
  keep AMP, batch, model, data, and augmentation parity unchanged. Invalid
  attempts are archived and excluded from results. This is disclosed as an
  optimizer-schedule asymmetry (Session 10 / Batch 3, before valid training).
- **YOLO pretrained identity pinned before Batch 3 training:** official
  Ultralytics v8.4.0 `yolo11s.pt`, SHA-256
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`,
  stored locally at ignored path `results/checkpoints/pretrained/yolo11s.pt`
  (Session 10 / Batch 3, before training).
- **YOLO augmentation parity decided before Batch 3 training:** disable every
  Ultralytics stochastic extra (mosaic, mixup, cutmix, copy-paste, HSV jitter,
  flips, affine/perspective transforms, erasing, auto-augmentation, and
  multi-scale training) to match Faster R-CNN's deterministic resize-only
  pipeline. This prioritizes a controlled detector comparison over the usual
  augmentation-rich YOLO recipe. YOLO11s remains pinned to
  `ultralytics==8.4.110`; seed 17, 640-pixel input, AMP, SGD LR 0.005/momentum
  0.9/weight decay 0.0005, effective batch 4, and originally frozen BatchNorm
  statistics. The later stability decisions above supersede the LR and BN
  details. Validation-AP50:95 early stopping (minimum 8, patience 5, maximum
  30) matches the Faster R-CNN protocol as closely as the framework allows. The remaining
  loss/scheduler/architecture differences will be reported rather than hidden.
  (Session 10 / Batch 3, before training).
- **Faster R-CNN one-seed baseline complete:** after explicit approval, the
  full seed-17 run restarted from COCO weights and early-stopped at epoch 11
  after 7,017.8 seconds (1.95 hours). Epoch 6 is the exact best checkpoint:
  validation AP50:95 0.12764, AP50 0.33144, precision 0.14138, recall 0.68953,
  and F1 0.23464 on 750 images/277 boxes at score 0.25 and match IoU 0.50.
  Peak allocated training memory was 1,556.6 MiB. Best-checkpoint profiling is
  11.00 FPS, mean/p50/p95 latency 90.92/90.78/92.18 ms, 43,256,153 total and
  43,030,809 trainable parameters, 450.76 estimated GFLOPs, and a 165.38 MiB
  model. Best checkpoint:
  `results/checkpoints/faster_rcnn_rsna_seed17_full/best_model.pt`, SHA-256
  `9ec35c5d761f8e4bf7a43f7999f388ac1ffc0d533f62746409db280706dffab4`.
  All run/table/curve/checkpoint identities and hashes validate; the test split
  was not accessed. Stop for Batch 2 result review before Batch 3 (Session 8 /
  Batch 2).
- **Faster R-CNN three-epoch timing gate complete:** the clean official-data
  benchmark completed epochs 1--3 in 770.3, 551.5, and 473.0 seconds (29.91
  minutes total), with a 512.2-second steady-state estimate. The configured
  minimum eight epochs project to 1.21 hours; the 30-epoch sign-off upper bound
  is 4.34 hours with a conservative 4.02--4.66-hour range. Peak allocated GPU
  memory was 1,556.6 MiB. Epoch 3 had the best diagnostic validation AP50:95,
  0.10993; this is not a final baseline metric. The approval artifact is
  `results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json`
  (SHA-256
  `232460ae09827dfb780b0f5c6506bf9f545bbdc0e1483082c2c440035e8e8e8b`),
  bound to config SHA-256
  `ef1e3ebe1fbe3cf1a6e27bf8b9c12f61719c2ea8771c9758f64dc278dd0e2633`.
  Its exact train-mode approval check passes; stop pending explicit user
  sign-off before the one-seed full run (Session 7 / Batch 2 recovery).
- **Windows DataLoader memory adjustment before the timing gate:** the first
  official-data benchmark attempt completed all 1,750 epoch-1 training batches
  but failed before validation when a second six-worker pool hit `WinError
  1455` (paging file/commit limit). No epoch record or checkpoint was produced.
  Keep `num_workers: 6`, but use non-persistent train and validation pools so
  only one PyTorch worker pool exists at a time. Pool startup remains inside
  each measured epoch; the clean three-epoch gate restarts from COCO weights
  under the new configuration fingerprint (Session 7 / Batch 2 recovery).
- **Adopted Batch 2 runtime before training:** following the user's request to
  use the existing local CUDA environment and later directive to proceed, both
  detector arms are repinned before any smoke/benchmark/training result to
  Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Torch 2.6.0+cu124, and Torchvision
  0.21.0+cu124. CUDA 12.4 is wheel-bundled and uses driver 610.47. CUDA NMS,
  float16 AMP/GradScaler, negative-image Faster R-CNN training, and FLOP
  profiling passed direct probes. `requirements.txt`, `pyproject.toml`,
  `.python-version`, `uv.lock`, README, and reproducibility/baseline docs are
  aligned; Ultralytics remains 8.4.110 (Session 7 / Batch 2 recovery).
- **Official pixels complete:** the manually downloaded Kaggle aggregate ZIP is
  3,932,287,530 bytes with SHA-256
  `133acacf95aa68c4d219124b17937f31cec073052096b9f9b122180df9d9af18`.
  Its full CRC test passes, it has no duplicate/unsafe paths, and it contains
  exactly 26,684 training plus 3,000 competition-test DICOMs. Both metadata CSV
  hashes match the committed audit. All 5,000 selected DICOMs are present; a
  fresh official-source conversion produced exactly 5,000 manifest-matching
  PNGs, zero missing sources/errors, and byte-identical results for the 12
  earlier review images (Session 7 / Batch 2 recovery).
- **2026-08-04 Kaggle OAuth confirmation:** the browser authorization callback
  completed successfully, but Kaggle returned HTTP 403 while exchanging the
  authorization code at `security.OAuthService/ExchangeOAuthToken`. The CLI
  therefore remains on the legacy key and still receives 403 for all tested
  API calls. This confirms the blocker is at Kaggle's API/account/network
  policy layer rather than a missed browser authorization. No official file
  bytes were downloaded; use the authenticated Kaggle website from an
  authorized available network/location and place the three official files
  manually (Session 6 / Batch 2 recovery).
- **2026-08-04 Batch 2 recovery audit:** `kaggle==2.2.3` is now installed in
  `.venv`, and `C:\Users\Pouyan\.kaggle\kaggle.json` is structurally valid for
  account `alphalacrim`, but Kaggle returns HTTP 403 for public dataset lists,
  competition lists/files, and the RSNA download. A forced browser OAuth login
  was opened but did not complete; no official file bytes were downloaded. The
  existing Anaconda `torch-gpu` environment was also verified: Python 3.11.15,
  Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, CUDA available, CUDA NMS working,
  and AMP enabled. It has not been adopted because it conflicts with the pinned
  Python 3.13 / Torch 2.13.0+cu130 / Torchvision 0.28.0+cu130 experiment
  identity; changing that identity requires an explicit user decision
  (Session 5 / Batch 2 recovery).
- **Faster R-CNN Batch 2 configuration:** the one-seed baseline is
  `fasterrcnn_resnet50_fpn_v2` with Torchvision default COCO weights, seed 17,
  640×640 transform bounds, physical batch 2, two-step gradient accumulation
  (effective optimizer batch 4), float16 AMP, SGD at LR 0.005, and no stochastic
  augmentation. Pretrained BatchNorm running statistics are frozen while
  affine parameters remain trainable. Validation AP50:95 drives both the
  plateau scheduler and early stopping (minimum 8, patience 5, maximum 30
  epochs). See `configs/faster_rcnn.yaml` and
  `docs/FASTER_RCNN_BASELINE.md` (Session 3 / Batch 2).
- **Faster R-CNN timing and artifact gate:** a complete three-epoch
  train-plus-validation benchmark is mandatory before full training. Epoch
  timing includes equivalent best/last checkpoint I/O; approval is bound to
  the exact YAML, train/validation annotation and pixel manifests, source
  manifest, Torch/Torchvision/CUDA/driver/GPU identity, AMP, batch, and
  resolution. Full mode cannot run without that approved artifact. Final
  metrics use the shared COCO/operating-point evaluator, and a recoverable
  `finalize` mode regenerates tables, curves, FPS, parameters, checkpoint size,
  and mandatory finite GFLOPs without retraining (Session 3 / Batch 2).
- **Batch 2 data readiness:** canonical train/validation COCO metadata and
  pixels are complete (3,500/750 images; 1,267/277 boxes), and the separate
  750-image held-out split is also materialized for later batches. Timed modes
  still aggregate and report any missing train/validation path before importing
  Torch or initializing CUDA. The held-out test annotation is neither opened
  nor evaluated in Batch 2 (updated Session 7 / Batch 2 recovery).
- **RSNA patient-safe benchmark split:** the Kaggle `patientId` is an exam UUID,
  so grouping uses the official RSNA mapping to the original NIH filename and
  parses its patient prefix. Seed 17 selects 5,000 studies from 2,136 NIH
  patient groups, then splits them exactly 3,500/750/750. All three patient-key
  intersections are empty. Study strata are train 798/1,554/1,148, validation
  169/331/250, and test 169/331/250 for opacity/no-opacity-not-normal/normal.
  See `configs/dataset.yaml`, committed split manifests, and `docs/DATASHEET.md`
  (Session 2 / Batch 1).
- **Actual class list/count:** exactly one foreground detection class,
  `Lung Opacity` (`category_id=1`); background is implicit. `Normal` and
  `No Lung Opacity / Not Normal` are study-level sampling strata, not detector
  classes. Downstream class counts must be read from config/COCO rather than
  hardcoded (Session 2 / Batch 1).
- **Dataset choice:** RSNA Pneumonia Detection Challenge 2018 Stage 2 was chosen
  over both brain-MRI exports because it has stronger expert annotation
  provenance, coherent negative-image semantics, bespoke terms suitable for
  course research, and an official mapping that enables patient-safe grouping.
  The full 26,684-study labeled set is reduced to the fixed stratified 5,000
  subset required by §3. See `docs/DATASET_CHOICE.md` (Session 2 / Batch 1).
- **Canonical data contract:** both future detectors will read per-split COCO
  JSON generated by `src/data/prepare.py`. Preparation performs the full
  metadata audit, verifies the official mapping digest, preserves negative
  images with zero annotations, and records input hashes. Configured DICOM
  conversion uses `MONOCHROME1` inversion plus per-image min–max scaling to
  8-bit PNG. See `docs/DATASHEET.md` (Session 2 / Batch 1).
- **YOLO selection and pin:** YOLO11s (`yolo11s.pt`) with
  `ultralytics==8.4.110`. YOLO11 was selected over YOLO26 for its conventional
  anchor-free/NMS pipeline, greater continuity with medical-detector literature,
  and clearer comparison against two-stage Faster R-CNN; the small scale fits
  8 GB VRAM. See `docs/LITERATURE_REVIEW.md` and `configs/yolo.yaml`
  (Session 2 / Batch 1).
- **Phase 0 reproducibility contract:** every experiment entry point must call
  `initialize_reproducibility(seed, output_dir)` before CUDA initialization.
  Runs record `pip_freeze.txt` and structured seed/platform/Torch/CUDA/GPU/driver
  metadata in `run_environment.json`. Deterministic algorithms use warning mode.
  See `src/utils/seed.py` and `docs/REPRODUCIBILITY.md` (Session 1 / Batch 0).
- **Dependency baseline:** the Batch 0 core pins remain exact in
  `requirements.txt`; Batches 1–2 added exact Pillow, PyYAML, Pydantic, pydicom,
  Kaggle, Ruff, Matplotlib, and Ultralytics pins. The former Ultralytics
  placeholder is superseded by `ultralytics==8.4.110` (Sessions 1–3 /
  Batches 0–2).
- **YOLO augmentation-asymmetry handling:** resolved by disabling all extra
  Ultralytics stochastic augmentation for the primary comparison. See
  `configs/yolo.yaml`, `docs/YOLO_BASELINE.md`, and `docs/LIMITATIONS.md`.

## File Map

| Path | Purpose | Status |
|---|---|---|
| `PROJECT_SPEC.md`, `BATCHES.md`, `AGENTS.md` | Static requirements, sequence, and session protocol | authoritative |
| `CODEX.md`, `HANDOFF.md` | Living decisions and append-only session state | living |
| `README.md`, `data/README.md` | Exact setup, official aggregate preparation, benchmark-gate, training, and finalization commands | documented through Batch 3 |
| `.python-version`, `requirements.txt`, `pyproject.toml`, `uv.lock` | Exact Python 3.11 / CUDA 12.4 phased-workflow dependency pins | aligned for adopted runtime |
| `configs/dataset.yaml` | RSNA paths, class/stratum map, conversion, subset, split, and EDA settings | done |
| `configs/yolo.yaml` | Strict YOLO11s data/model/runtime/training/evaluation/profile/artifact config | implemented; smoke, timing gate, full run, and profiling complete |
| `configs/faster_rcnn.yaml` | Strict Faster R-CNN model/runtime/training/evaluation/profile/artifact config | implemented; timing gate and full run complete |
| `src/data/download.py` | Secret-safe Kaggle competition downloader and clear failure diagnostics | done |
| `src/data/prepare.py` | Metadata audit, digest check, patient grouping, subset/split, COCO, DICOM conversion | done |
| `src/data/visualize.py` | Deterministic distribution and annotation-sample EDA | done |
| `data/manifests/rsna-pneumonia-5000-audit.json` | Committed machine-readable audit and exact counts | done |
| `data/splits/rsna-pneumonia-5000/*.csv` | Patient-safe image manifests | done |
| `data/processed/rsna-pneumonia-5000/annotations/*.json` | Generated canonical COCO annotations | local/ignored; regenerable |
| `results/figures/rsna_class_distribution.png` | Selected split class/stratum distribution | done |
| `results/figures/rsna_annotation_samples.png` | Twelve real radiographs with labels/boxes | done |
| `docs/DATASET_CHOICE.md` | Three-candidate inspection and selection rationale | done |
| `docs/DATASHEET.md` | Collection, composition, patient split, audit, processing, terms, bias | done |
| `docs/LITERATURE_REVIEW.md` | YOLO11, detector paradigms, Grad-CAM/XAI, related work, robustness | done |
| `report/references.bib` | Phase 1 primary/official BibTeX sources | done |
| `docs/LIMITATIONS.md` | Dataset and predeclared compute limitations | living |
| `docs/FASTER_RCNN_BASELINE.md` | Batch 2 architecture, optimization, metrics, timing, and profiling protocol | complete with final measurements |
| `docs/YOLO_BASELINE.md` | Batch 3 architecture, parity/stability decisions, timing, metrics, and profiling | complete with final measurements |
| `src/utils/seed.py`, `docs/REPRODUCIBILITY.md` | Reproducibility utilities and contract | done |
| `tests/test_{download,prepare,visualize}.py` | Batch 1 acquisition/conversion/split/EDA tests | done |
| `src/models/faster_rcnn_*.py`, `src/models/train_faster_rcnn.py` | Strict data adapter, model, AMP trainer, unified validation, reporting, profiling, and gates | implemented; smoke, benchmark, full run, and profiling complete |
| `src/models/yolo_*.py`, `src/models/train_yolo.py` | Strict YOLO data view, mixed-precision trainer, unified validation, reporting, profiling, and gates | implemented; smoke, benchmark, full run, and profiling complete |
| `tests/test_faster_rcnn_*.py`, `tests/test_train_faster_rcnn.py`, `tests/test_yolo_*.py` | Batch 2/3 config/data/model/control/reporting/gate regressions | 172 pass repository-wide; one expected metadata-only skip |
| `src/robustness/`, `src/explainability/`, `src/stats/` | Later-batch implementation directories | deferred |
| `pyproject.toml`, `uv.lock`, older tests/configs | Pre-workflow legacy implementation | non-authoritative; reconcile before use |
| `src/meddet_benchmark/` | Shared tested operating-point and COCO evaluator used by both detector baselines | reconciled for package-relative imports; other legacy surfaces non-authoritative |

## Current phase

**Batch 3 / Phase 4 is complete.** YOLO11s is pinned, augmentation parity was
decided before training, the official train/validation views validate, and the
stable bfloat16/fp32-loss run passed smoke and the three-epoch timing gate. The
one-seed full run early-stopped at epoch 15; final shared validation, compute,
speed, checkpoint, and curve artifacts all validate. Stop here for the user's
review before starting Batch 4 / Phase 5.

## Known open issues / risks

- Kaggle API/OAuth remains denied on the current route, but it no longer blocks
  Batch 2 because the complete official aggregate archive was acquired manually
  and verified. Future clean acquisitions may still need manual browser download
  from an authorized available network/location.
- The adopted Anaconda runtime is external to the repository, so reproducible
  commands and run-level `pip_freeze.txt`/`run_environment.json` are mandatory.
  Both detector arms must use this same locked Python 3.11/Torch 2.6/CUDA 12.4
  identity; changing it after the timing benchmark invalidates approval.
- The official Kaggle CSV hashes reproduce the values recorded in the committed
  audit; the earlier mirror-provenance uncertainty for metadata is resolved.
- Per-image min–max DICOM conversion is deterministic and config-declared but
  does not reproduce vendor-specific window/VOI display processing. This remains
  a preprocessing limitation.
- `pyproject.toml`/`uv.lock` now match the reviewed Ultralytics 8.4.110 decision
  and adopted Python 3.11/CUDA 12.4 runtime.
- YOLO augmentation parity is resolved: all Ultralytics-only stochastic
  augmentations are disabled for the controlled baseline. This may understate
  performance under YOLO's conventional augmentation-rich recipe.
- Both detectors currently have only one completed training seed, as
  predeclared by the hardware scope. Their validation curves and point metrics
  do not characterize across-seed variance; two additional seeds per detector
  remain reserved for the later headline quantitative comparison.
- YOLO required bfloat16 forward/backward with float32 assignment/loss, native
  BatchNorm updates, a lower LR, and a one-epoch warmup for numerical stability;
  Faster R-CNN used float16, frozen normalization, LR 0.005, and a plateau
  scheduler. These disclosed optimization/precision asymmetries limit a purely
  architecture-only interpretation of the comparison.
