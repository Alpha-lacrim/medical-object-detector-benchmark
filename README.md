# Medical Object Detector Benchmark

This repository is a controlled comparison of Faster R-CNN and YOLO11s on the
RSNA Pneumonia Detection Challenge. It measures clean detection performance,
compute, common-corruption robustness, Grad-CAM localization, and paired
statistical uncertainty under one patient-safe data and evaluation protocol.

The project is complete. The 12-section study is in
[`report/report.md`](report/report.md), the consolidated scope statement is in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), and the exact generated evidence
is under [`results/`](results/). This is a retrospective benchmark, not a
clinical device or diagnostic system.

## Headline result

Across seeds 17, 42, 137, 271, and 314, Faster R-CNN achieves
mAP@0.5:0.95 of 0.0995 ± 0.0067 versus 0.0542 ± 0.0060 for YOLO11s.
At thresholds selected by maximum mean validation F1 in the frozen original
n=3 analysis, final test precision/recall/F1 is
0.3543/0.3607/0.3492 versus 0.3096/0.2438/0.2718. Faster R-CNN has higher
precision at 96 of 101 official AP@0.5 recall positions; YOLO11s' apparent
precision advantage at the original shared score threshold of 0.25 is a
score-scale/selectivity artifact, not a frontier advantage. The primary
training-procedure bootstrap gives wholly positive Faster-R-CNN-minus-YOLO11s
intervals for recall, F1, mAP@0.5, and mAP@0.5:0.95. Fixed-threshold precision
is different: its primary interval crosses zero (`-0.2423` to `0.0553`), while
the separate Holm p-value conditional on the observed checkpoints is `0.0020`
in favor of YOLO11s. These target different randomness and are not
interchangeable significance claims. Conditional IoU and Dice also cross zero.

YOLO11s seed 271 is retained as a legitimate all-attempt result. Training
converged normally and test AP@0.5/AP@0.5:0.95 was 0.1587217/0.0555799, but its
maximum test confidence was only 0.0412735, so it emitted no detection at the
frozen score threshold 0.25 and contributed precision/recall/F1 of zero.
Matched-only IoU and Dice are therefore undefined for that run, not zero:
descriptive localization uses Faster R-CNN n=5 versus YOLO11s n=4, and paired
localization inference uses the four complete seed pairs 17, 42, 137, and 314.
All other clean endpoints retain all five attempted seeds.

The defensible trade-off is detection quality versus implementation-specific
computational cost. Across all five seeds, YOLO11s delivers
60.29 ± 12.62 FPS versus 20.28 ± 5.62, with 9.43 M parameters versus
43.26 M. The frozen n=3 FROC analysis gives Faster R-CNN higher sensitivity at
every reported false-positive budget, and neither detector strictly dominates
the frozen n=3 accuracy-efficiency Pareto panels.

On the seed-17 300-image common-corruption sample, mean mAP@0.5:0.95 retention
is 0.7638 for Faster R-CNN and 0.7091 for YOLO11s. Both detectors have weak
Grad-CAM localization: mean energy-in-box is 0.0869 and 0.0975, with pointing
accuracy 0.1091 and 0.1261. These values do not establish clinical validity.

## Reproduction assumptions

Run every command below from the repository root in Windows PowerShell. A clean
reproduction requires:

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/);
- an NVIDIA GPU/driver compatible with the pinned CUDA 12.4 Torch wheels;
- sufficient disk space for the RSNA archive, 5,000 processed images,
  checkpoints, prediction bundles, and logs;
- a Kaggle account that has joined the RSNA competition and accepted its rules;
  and
- either `KAGGLE_USERNAME`/`KAGGLE_KEY` or a valid
  `$HOME\.kaggle\kaggle.json`. Credentials must never be committed.

Raw images, processed images, downloaded pretrained weights, and trained
checkpoints are intentionally Git-ignored. The commands below regenerate them.
The measured workstation used an RTX 4060 Laptop GPU with 8 GB VRAM, 16 GB RAM,
and an i7-13650HX; timing will vary on other machines.

## 1. Create the pinned environment

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv --default-index https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps --editable .

$benchmarkPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $benchmarkPython -m pip check
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests
```

`requirements.txt` is the authoritative exact dependency list. Each experiment
also writes `pip_freeze.txt` and `run_environment.json` before CUDA
initialization. Deterministic algorithms use warning mode because the pinned
Torchvision CUDA ROI Align backward used by Grad-CAM is not bitwise
deterministic.

## 2. Acquire and prepare the dataset

The downloader requests the three official Stage 2 files declared in
`configs/dataset.yaml`. The official mapping is fetched directly from RSNA.

```powershell
& $benchmarkPython -m src.data.download --check-credentials
& $benchmarkPython -m src.data.download --config configs/dataset.yaml
Expand-Archive -LiteralPath data/raw/rsna-pneumonia/stage_2_train_images.zip -DestinationPath data/raw/rsna-pneumonia -Force
Invoke-WebRequest -Uri "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json" -OutFile data/raw/rsna-pneumonia/mappings.json
& $benchmarkPython -m src.data.prepare --config configs/dataset.yaml --convert-images
& $benchmarkPython -m src.data.visualize --config configs/dataset.yaml
```

Preparation verifies the pinned mapping digest, audits all 26,684 labeled
studies, reconstructs true NIH patient groups, writes the exact 5,000-study
70/15/15 split, converts the selected DICOMs, and creates canonical COCO JSON.
Visualization regenerates:

- `results/figures/rsna_class_distribution.png` (report Figure 1);
- `results/figures/rsna_annotation_samples.png` (report Figure 2); and
- `results/figures/rsna_eda_summary.json`.

The dataset table in report Section 3 is a rounded rendering of
`data/manifests/rsna-pneumonia-5000-audit.json` and the generated/committed split
manifests from the preparation command. To regenerate metadata, splits, and
COCO annotations without decoding pixels, use:

```powershell
& $benchmarkPython -m src.data.prepare --config configs/dataset.yaml --metadata-only
```

If Kaggle supplies the aggregate competition archive instead of the individual
image ZIP, extract only its training-image member before preparation:

```powershell
tar -xf data/raw/rsna-pneumonia/rsna-pneumonia-detection-challenge.zip -C data/raw/rsna-pneumonia stage_2_train_images
```

## 3. Train and finalize the primary Faster R-CNN run

`configs/faster_rcnn.yaml` fixes the seed-17 ResNet-50/FPN model, data,
optimizer, float16 AMP, 640-pixel transform, physical batch 2, accumulation 2,
and output identity. Run the required readiness, smoke, and three-epoch timing
gate before the full run:

```powershell
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode preflight
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode smoke
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode benchmark
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json
```

The train command regenerates the report Section 5 evidence:

- `results/tables/faster_rcnn_baseline_validation.csv` (report Table 2);
- `results/tables/faster_rcnn_compute.csv` (report Table 2); and
- `results/figures/faster_rcnn_training_curves.png` (report Figure 3).

If optimization completed but final profiling or plotting was interrupted,
regenerate the derived artifacts from the saved best checkpoint without
retraining:

```powershell
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode finalize
```

## 4. Train and finalize the primary YOLO11s run

The prepare mode makes an audited YOLO view of the canonical data. The remaining
commands verify the pinned pretrained weight, run the GPU smoke/timing gate, and
train seed 17:

```powershell
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode prepare
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode preflight
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode smoke
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode benchmark
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode train
```

The train command regenerates the report Section 6 evidence:

- `results/tables/yolo_baseline_validation.csv` (report Table 3);
- `results/tables/yolo_compute.csv` (report Table 3); and
- `results/figures/yolo_training_curves.png` (report Figure 4).

If reporting was interrupted after training, use:

```powershell
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode finalize
```

## 5. Train the additional seeds and run the five-seed unified test evaluator

The additional configs change only seed and artifact identity. First derive
seed-specific timing approvals from the accepted seed-17 Faster R-CNN gate,
then train seeds 42, 137, 271, and 314 for both detectors. These are the exact
commands for every non-primary run:

```powershell
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode seed-gates

& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed42.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed42_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed137.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed137_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed271.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed271_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed314.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed314_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed42.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed137.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed271.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed314.yaml --mode train

& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode preflight
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode evaluate
```

Only the final `evaluate` command opens the held-out test annotation. Both detector
adapters feed one pycocotools/operating-point evaluator; framework-native mAP is
not used in the comparison. The evaluate command regenerates:

- `results/tables/detector_comparison.csv` (report Tables 4a and 4b);
- `results/tables/detector_comparison_mean_std.csv`;
- `results/tables/detector_comparison_per_seed.csv`; and
- `results/logs/phase5_evaluation/summary.json` plus the ten frozen prediction
  bundles used by statistics.

The n=5 evaluator retains null matched-only IoU/Dice only when a run has zero
true positives. It reports the affected seed and metric-specific n in every
comparison table and in the summary; it never coerces undefined localization
to zero. Regenerate the seed-271 convergence and score-scale diagnostic from
the frozen evidence with:

```powershell
& $benchmarkPython -m src.analyze_yolo_seed_stability --config configs/yolo_seed_stability.yaml
```

This writes `results/tables/yolo_seed_stability.csv` and
`results/logs/phase16_yolo_seed_stability/summary.json`. The superseded clean
n=3 evidence remains available at:

- `results/tables/detector_comparison_n3_archive.csv`;
- `results/tables/detector_comparison_mean_std_n3_archive.csv`;
- `results/tables/detector_comparison_per_seed_n3_archive.csv`;
- `results/tables/statistical_clean_comparison_n3_archive.csv`; and
- `results/logs/phase5_evaluation/summary_n3_archive.json`.

### 5a. Reprocess the frozen bundles across confidence thresholds

The research-track threshold analysis is CPU-only and remains frozen to the six
original n=3 Phase 5 bundles for seeds 17, 42, and 137. Its config reads
`configs/evaluation_n3_archive.yaml` and the archived n=3 Phase 5 summary; it is
not an n=5 analysis. It does not load a checkpoint, train a model, or run
inference.
Preflight verifies every bundle and annotation hash before the run exposes the
official pycocotools precision-recall tensors and evaluates 99 common confidence
thresholds through the same unified matcher:

```powershell
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep.yaml --mode preflight
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep.yaml --mode run
```

The run regenerates `results/tables/threshold_sweep*.csv`,
`results/tables/precision_recall_curves*.csv`,
`results/tables/threshold_operating_targets.csv`,
`results/figures/precision_recall_curves.png`,
`results/figures/f1_vs_threshold.png`, and the hashed summary under
`results/logs/phase10_threshold_sweep/`. Definitions and interpretation are in
`docs/THRESHOLD_ANALYSIS.md`.

### 5b. Select final operating thresholds on validation

The original training runs retained validation aggregates but not the raw scored
detections needed for a threshold sweep. The frozen n=3 workflow materializes
those records from the six immutable best checkpoints for seeds 17, 42, and 137
on the 750-image validation split, then performs selection and the final test
application offline:

```powershell
& $benchmarkPython -m src.evaluate_threshold_selection --config configs/threshold_selection.yaml --mode preflight
& $benchmarkPython -m src.evaluate_threshold_selection --config configs/threshold_selection.yaml --mode collect-validation
& $benchmarkPython -m src.evaluate_threshold_selection --config configs/threshold_selection.yaml --mode run
```

`collect-validation` is inference-only: it neither trains nor changes a
checkpoint, and it must reproduce each run's archived validation
precision/recall/F1 before writing a hash-bound bundle. The offline run maximizes
mean validation F1, freezes one threshold per detector, and applies each once to
the frozen test bundles. It regenerates `validation_threshold_sweep*.csv`,
`selected_operating_points*.csv`, and the manifest/summary under
`results/logs/phase14_threshold_selection/`.

### 5c. Reparameterize the test sweep as FROC curves

This CPU-only step reads the unchanged Batch 10 per-seed table and describes the
full test sweep as sensitivity versus false positives per image. It does not
select a deployment threshold or perform training, checkpoint loading, or
inference. Its inputs and outputs remain n=3-only:

```powershell
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n3_archive.yaml --mode preflight
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n3_archive.yaml --mode run
```

The archive-safe run writes
`results/figures/froc_curves_n3_archive_reproduction.png`,
`results/tables/froc_operating_points_n3_archive_reproduction.csv`, and a
summary under `results/logs/phase14_froc_n3_archive_reproduction/`; it does not
overwrite the frozen primary FROC artifacts. Definitions and interpretation
are in `docs/FROC_ANALYSIS.md`.

### 5d. Regenerate the accuracy-efficiency Pareto figure

This CPU-only step joins the archived n=3 Phase 5 accuracy rows, all six original
seed-specific compute tables, and the validation-selected, test-evaluated
operating points. It performs no training or inference. The recall panels use
the thresholds selected in section 5b, while the mAP panels remain
threshold-independent. It must not be interpreted as a five-seed Pareto result:

```powershell
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode preflight
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode run
```

The run regenerates `results/figures/pareto_frontier.png`; definitions and the
scenario-conditional interpretation are in `docs/PARETO_ANALYSIS.md`.

### 5e. Measure five-seed detection calibration

This CPU-only analysis reads all ten frozen Phase 5 prediction bundles. It
matches every detection retained at the 0.001 bundle floor through the canonical
IoU-0.50 matcher and computes the full box-sensitive Detection Expected
Calibration Error over confidence, relative center, width, and height. It does
not fit a calibrator, load checkpoints, run inference, or train a model:

```powershell
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode run
```

The run regenerates `results/tables/calibration_summary.csv`,
`results/figures/reliability_diagrams.png`, and the exact provenance record at
`results/logs/phase18_calibration/summary.json`. Definitions, the five-seed
finding, and the distinction from threshold selectivity are in
`docs/CALIBRATION_ANALYSIS.md`.

### 5f. Run recall-weighted F-beta threshold sensitivity

This CPU-only sensitivity analysis reuses the six frozen Batch 14 validation
prediction bundles. It evaluates beta values 1, 3, 5, and 10 over the same
0.01–0.99 threshold grid, computes hierarchical patient-cluster/seed bootstrap
intervals, and selects the threshold with the largest lower 95% confidence
bound. Beta is a recall-versus-precision preference parameter, not a measured
clinical-harm ratio. The run also writes validation-only plateau and bootstrap
selection-frequency diagnostics plus a separate hypothetical linear
`r * FN / N + FP / N` loss sweep. It performs no training, inference,
checkpoint loading, or test access:

```powershell
& $benchmarkPython -m src.stats.threshold_calibration --config configs/threshold_calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.threshold_calibration --config configs/threshold_calibration.yaml --mode run
```

The run regenerates
`results/tables/recall_weighted_fbeta_threshold_summary.csv`,
`results/tables/recall_weighted_fbeta_threshold_stability.csv`,
`results/tables/hypothetical_detection_error_loss_summary.csv`,
`results/figures/recall_weighted_fbeta_threshold_sensitivity.png`, and the
provenance record at
`results/logs/phase29_threshold_sensitivity/summary.json`. Per D-006, these
remain separate from Batch 14's primary operating points and are never selected
from or applied to test outcomes. Definitions and findings are in
`docs/THRESHOLD_CALIBRATION.md`.

### 5g. Run full-test decision curve analysis

This CPU-only analysis reduces each detector to an exam-level flag based on the maximum
emitted box confidence, then computes net benefit over thresholds 0.01--0.99 from all ten
frozen Phase 5 test bundles. The prevalence input is derived from the complete 750-image
test manifest (169 positive images; 22.533%), not from the 300-image robustness sample. The
2,000 common draws reuse the patient-cluster/seed bootstrap machinery. No checkpoint,
inference, or training is involved:

```powershell
& $benchmarkPython -m src.clinical.decision_curve --config configs/decision_curve.yaml --mode preflight
& $benchmarkPython -m src.clinical.decision_curve --config configs/decision_curve.yaml --mode run
```

The run regenerates `results/tables/dca_summary.csv`,
`results/figures/dca_curves.png`, and the provenance record at
`results/logs/phase20_decision_curve/summary.json`. The exam-level decision rule,
patient-cluster uncertainty, prevalence scope, findings, and clinical-interpretation limits
are in `docs/DCA_ANALYSIS.md`.

### 5h. Regenerate the seed-level raincloud comparison

This CPU-only reporting step reads the five-seed publication table and its ten
run-level records. Before plotting, it recomputes every configured mean, sample
standard deviation, finite seed count, and attempted seed count from the
per-seed table and rejects any mismatch. No checkpoint, prediction bundle,
inference, resampling, or training is involved:

```powershell
& $benchmarkPython -m src.plot_raincloud_metrics --config configs/raincloud_metrics.yaml --mode preflight
& $benchmarkPython -m src.plot_raincloud_metrics --config configs/raincloud_metrics.yaml --mode run
```

The run regenerates `results/figures/raincloud_metrics.png` and the input,
figure-hash, and panel-count record at
`results/logs/phase23_reporting/raincloud_metrics_summary.json`. All 14 panels
state their actual finite seed count; conditional IoU and Dice explicitly show
Faster R-CNN `n=5` versus YOLO11s `n=4`, while the other panels use `n=5` per
detector.

## 6. Run the common-corruption benchmark

The run command deterministically draws or verifies the 300-image sample,
filters the clean seed-17 prediction evidence, and infers every one of the 70
corrupted detector conditions. Completed condition bundles are resumable.

```powershell
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode preflight
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode run
```

This regenerates the report Section 8 evidence:

- `results/tables/robustness_results.csv` (report Table 5 and all raw/relative
  metrics);
- `results/tables/robustness_curves.csv`;
- `results/tables/robustness_family_mean_curves.csv`;
- `results/figures/robustness_map_50_95_raw.png` (report Figure 5);
- `results/figures/robustness_map_50_95_relative.png` (report Figure 6); and
- `results/logs/phase6_robustness/summary.json` plus 72 prediction bundles.

## 6a. Run the raw-radiography acquisition-shift analysis

This checkpoint-only extension reuses the frozen 300-image robustness sample
and both seed-17 checkpoints. Preflight directly verifies all ignored raw DICOM
files, their metadata contract, and exact pixel equivalence between clean raw
reconversion and the canonical PNGs before any GPU work. The full run applies
standard DICOM `LINEAR` VOI Window Center/Width alternatives,
signal-dependent Poisson count noise, and finite Gaussian detector/processing
blur kernels before the shared per-image min-max scaler.

```powershell
& $benchmarkPython -m pytest tests/test_radiography_shifts.py tests/test_prepare.py -q
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode preflight
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode smoke
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode run
```

This generates:

- `results/tables/acquisition_shift_results.csv`, including clean and shifted
  performance plus `DSI = 1 - shifted / clean` for all seven unified metrics;
- `results/logs/phase22_acquisition_shifts/summary.json`; and
- 20 resumable, hash-bound prediction bundles.

The method, findings, metadata constraints, and explicit distinction from the
post-conversion digital corruption grid are documented in
[`docs/ACQUISITION_SHIFTS.md`](docs/ACQUISITION_SHIFTS.md). Both analyses remain
internal synthetic sensitivity studies, not clinical-robustness evidence.

## 7. Run the Grad-CAM analysis

Preflight binds the analysis to the completed robustness sample and primary
checkpoint hashes. The smoke pass is bounded to two positive images per model;
the full run covers all 111 boxes and the predeclared qualitative cases. This
analysis remains explicitly scoped to the seed-17 checkpoints and was not
extended to five seeds.

```powershell
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode preflight
& $benchmarkPython -m pytest tests/test_gradcam.py tests/test_pointing_game.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode smoke
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode run
```

This regenerates the report Section 9 evidence:

- `results/tables/gradcam_localization_summary.csv` (report Table 6);
- `results/tables/gradcam_localization_per_target.csv`;
- `results/tables/gradcam_qualitative_cases.csv`;
- `results/figures/gradcam_good_predictions.png` (report Figure 7);
- `results/figures/gradcam_bad_predictions.png` (report Figure 8);
- `results/figures/gradcam_failure_cases.png` (report Figure 9); and
- `results/logs/phase7_explainability/summary.json`.

## 7a. Run the Grad-CAM sanity checks

This checkpoint-only extension selects 50 images from the already-frozen
robustness manifest, then compares trained maps with maps after Xavier weight
randomization and deterministic within-image pixel shuffling. It performs no
training. Preflight validates the Phase 6/7 provenance and materializes the
nested 50-image manifest before CUDA is initialized.

```powershell
& $benchmarkPython -m pytest tests/test_xai_sanity.py tests/test_gradcam.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode preflight
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode run
```

This generates:

- `results/tables/gradcam_sanity_summary.csv`;
- `results/tables/gradcam_sanity_per_image.csv`;
- `results/figures/gradcam_sanity_panel.png`; and
- `results/logs/phase21_xai_sanity/summary.json` plus the nested-subset manifest.

The method, denominators, zero-map failures, and interpretation are documented
in [`docs/XAI_SANITY.md`](docs/XAI_SANITY.md).

## 8. Run the estimand-separated statistical analysis

The current clean-only CPU refresh reads the ten frozen clean bundles and the
committed Batch 1 image-to-patient mapping. It does not rerun model inference,
robustness, or explainability. The pre-existing 72 robustness bundles and
seed-17 robustness table remain frozen and are verified by hash rather than
recomputed.

```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode preflight
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run --scope clean
```

This regenerates:

- `results/tables/statistical_clean_comparison.csv` (report Table 7);
- `results/tables/statistical_clean_per_run_metrics.csv`;
- `results/tables/statistical_clean_leave_one_run_out.csv`;
- `results/tables/statistical_clean_leave_one_seed_label_out.csv`;
- `results/logs/phase8_statistics/summary.json`.

It preserves `results/tables/statistical_robustness_comparison.csv` and the
seed-17 corruption inference already recorded in the summary. A deliberate
full from-scratch reproduction after Section 6 can instead run the same command
without `--scope clean`; that was not done for the five-seed clean refresh.

The primary training-procedure estimand uses 2,000 shared patient-cluster draws
and independent within-detector trained-run draws. Every sampled patient carries
all observed images together, and every nonlinear metric—including aggregate
AP—is reconstructed from the sampled predictions. Unconditional endpoints use
five runs per detector. Conditional IoU and Dice use five defined Faster R-CNN
runs and four defined YOLO11s runs because seed 271 has no YOLO11s
fixed-threshold match.

The separate 5,000-draw patient-cluster permutation p-values condition on the
observed checkpoints and receive Holm correction across seven clean endpoints;
they are secondary sensitivity results, not seed-aware p-values. The old
common-seed-index bootstrap remains at
`statistical_clean_comparison_paired_seed_sensitivity_archive.csv` with its
provenance summary. The first patient-cluster correction also preserves the superseded
image-level CSVs and summary under `*_image_level_archive.*` paths for audit;
reruns never replace those archives. The clean-only refresh also preserves the
corrected n=3 clean table at
`results/tables/statistical_clean_comparison_n3_archive.csv`.

Any derivative paper must name the inferential target. For broad pipeline
claims, the training-procedure intervals are primary. It must carry the
detector-specific run counts and seed-271 role: 5/5 for unconditional clean
endpoints and 5/4 for conditional IoU/Dice, with no replacement seed.
Threshold selection, FROC, and Pareto remain n=3-only;
robustness and explainability remain seed-17-only.

## Report artifact-to-command index

Report and paper-draft values are rounded views of committed machine-readable
artifacts; manuscript assembly does not recompute them. The index explicitly
includes every executable analysis module added in Batches 18--23.

| Report or paper item | Generated source | Regenerating command |
|---|---|---|
| Paper §3.1; report Table 1 and Figures 1–2 | audit/split manifests; `rsna_*.png`; `rsna_eda_summary.json` | `src.data.prepare`, then `src.data.visualize` in §2 |
| Paper §§3.2–3.3 protocol parameters | run-level `resolved_config.json` / `resolved_experiment.json`; Phase 5 summary | detector train/finalize in §§3–4, then `src.evaluate --mode evaluate` in §5 |
| Table 2; Figure 3 | `faster_rcnn_*.csv`; Faster curve | seed-17 Faster R-CNN train/finalize in §3 |
| Table 3; Figure 4 | `yolo_*.csv`; YOLO curve | seed-17 YOLO train/finalize in §4 |
| Paper §§4.1 and 4.5; report Tables 4a–4b | `detector_comparison*.csv` | unified `src.evaluate --mode evaluate` in §5 |
| YOLO seed-stability diagnostic | `yolo_seed_stability.csv` | `src.analyze_yolo_seed_stability` in §5 |
| Paper §4.2 PR/F1 evidence (frozen n=3) | `threshold_sweep*.csv`; `precision_recall_curves*.csv` | offline `src.evaluate_threshold_sweep --mode run` in §5a |
| Paper §4.2 validation-selected operating points (frozen n=3) | `validation_threshold_sweep*.csv`; `selected_operating_points*.csv` | inference-only materialization plus offline `src.evaluate_threshold_selection --mode run` in §5b |
| Paper §4.2 FROC evidence (frozen n=3) | primary `froc_operating_points.csv` / `froc_curves.png`; archive-safe `*_n3_archive_reproduction` copies | offline `src.plot_froc_curves --mode run` in §5c |
| Paper §4.5 Pareto evidence (frozen n=3) | `pareto_frontier.png` | offline `src.plot_pareto_frontier --mode run` in §5d |
| Paper §4.4 five-seed detection calibration (Batch 18) | `calibration_summary.csv`; `reliability_diagrams.png`; Phase 18 summary | offline `src.stats.calibration --mode run` in §5e |
| Paper §4.3 recall-weighted F-beta and hypothetical-loss sensitivity (Batch 29; frozen n=3 validation) | `recall_weighted_fbeta_threshold_summary.csv`; `recall_weighted_fbeta_threshold_stability.csv`; `hypothetical_detection_error_loss_summary.csv`; corrected sensitivity figure; Phase 29 summary | offline `src.stats.threshold_calibration --mode run` in §5f |
| Paper §4.6 full-test decision curves (Batch 20) | `dca_summary.csv`; `dca_curves.png`; Phase 20 summary | offline `src.clinical.decision_curve --mode run` in §5g |
| Paper Figure 1 seed-level predictive/compute rainclouds (Batch 23) | `detector_comparison.csv`; `detector_comparison_per_seed.csv`; `raincloud_metrics.png`; Phase 23 summary | audited `src.plot_raincloud_metrics --mode run` in §5h |
| Paper §4.7; report Table 5 and Figures 5–6 | `robustness*.csv`; robustness plots | robustness `--mode run` in §6 |
| Paper §4.8 acquisition-shift sensitivity (Batch 22) | `acquisition_shift_results.csv`; 20 prediction bundles; Phase 22 summary | checkpoint-only `src.robustness.radiography_shifts --mode run` in §6a |
| Table 6; Figures 7–9 | `gradcam*.csv`; Grad-CAM plots | explainability `--mode run` in §7 |
| Paper §4.9 Grad-CAM parameter/data sanity checks (Batch 21) | `gradcam_sanity*.csv`; `gradcam_sanity_panel.png`; Phase 21 summary | checkpoint-only `src.explainability.sanity_checks --mode run` in §7a |
| Paper §4.10; report Table 7 | `statistical_clean_comparison.csv` | statistics `--mode run --scope clean` in §8 |
| Frozen seed-17 corruption inference | `statistical_robustness_comparison.csv` | prior full-scope statistics `--mode run` after §6; not rerun for n=5 |

## Definition of Done audit

Every item in the benchmark's Definition of Done is satisfied:

- [x] **Two detectors under matched conditions.** Faster R-CNN and YOLO11s are
  trained on identical patient-safe splits, resolution, seed grid, evaluation
  protocol, and no-augmentation policy. Necessary precision, normalization,
  learning-rate, batch, and scheduler asymmetries are documented in the model
  reports and `docs/LIMITATIONS.md`.
- [x] **Standardized predictive and compute benchmark.** `src/evaluate.py`
  routes both adapters through the same operating-point matcher and
  pycocotools evaluator; comparison artifacts include accuracy, latency/FPS,
  parameters, GFLOPs, memory, and training time.
- [x] **Multi-family, multi-severity robustness.** Seven corruption types in
  four families are evaluated at five severities, with raw and clean-relative
  curves and complete result tables.
- [x] **Qualitative and quantitative Grad-CAM.** Three paired figure categories
  cover good predictions, false positives, and false-negative proxies; all 111
  boxes receive energy-in-box and pointing-game analysis, with one explicit
  zero-energy map.
- [x] **Statistical inference.** The frozen clean and corruption predictions
  have patient-cluster inference. Clean pipeline claims use primary
  patient-cluster/independent-run bootstrap CIs; separately labeled
  permutation p-values condition on the observed checkpoints. Raw-difference
  effects, Holm correction, non-estimable rows, and the reason McNemar is
  inapplicable are explicit.
- [x] **Scenario-grounded discussion.** Report Section 11 weighs measured
  accuracy, robustness, interpretability, and compute for high-sensitivity
  retrospective screening, constrained point-of-care assistance, and
  autonomous use.
- [x] **Honest consolidated limitations.** `docs/LIMITATIONS.md` covers the
  single dataset, five-seed clean headline scope with four complete conditional
  localization pairs, seed-271 confidence/output-score instability,
  primary-seed 300-image/111-box robustness and explainability scope,
  augmentation choice, detector asymmetries, and RTX 4060 8 GB / 16 GB RAM
  constraints.
- [x] **Clean-checkout reproduction commands.** Sections 1–8 above provide the
  exact ordered commands and the artifact index maps every report table and
  figure to its generating command.

## Repository verification

After a reproduction or code change, run:

```powershell
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests
git diff --check
```

See `docs/REPRODUCIBILITY.md` for the seed/environment contract and the
phase-specific documents under `docs/` for complete metric, corruption,
Grad-CAM, and inference definitions.

## License

Copyright (C) 2026 Pouyan Delivandani.

Except where otherwise noted, repository-authored software and documentation
are licensed under the [GNU Affero General Public License, version 3.0
only](LICENSE) (`AGPL-3.0-only`). This choice is compatible with the
[upstream licensing requirements for Ultralytics
YOLO](https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed).

The repository license does not replace third-party terms. In particular, the
RSNA/NIH dataset and dataset-derived image content remain governed by the
[RSNA challenge terms](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf);
pretrained model weights and external dependencies remain governed by their
respective licenses. Raw datasets and trained weights are not distributed in
this repository.
