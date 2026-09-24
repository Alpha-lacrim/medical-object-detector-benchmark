# Medical Object Detector Benchmark

This repository is a controlled, multi-axis benchmark of two disclosed Faster
R-CNN and YOLO11s pipelines on a patient-disjoint internal subset of the RSNA
Pneumonia Detection Challenge. It characterizes clean detection performance,
operating points, calibration, compute, common-corruption robustness, Grad-CAM
localization, and statistical uncertainty under one common evaluator. Frozen
external testing applies all ten RSNA checkpoints without adaptation to the
official VinDr-CXR test release under its strict local opacity ontology.

The documented benchmark experiments are complete, while the manuscript remains
a living document. This is a retrospective benchmark, not a clinical device or
diagnostic system.

## Research artifact release

This tree declares release version `2.0.0`. The Python project, importable
package, and research artifact release use one version; the corresponding Git
tag is `v2.0.0`. The release is identified by the exact reviewed repository
commit named by that tag. Until the tag is created by the release owner, an
untagged worktree is only a release candidate.

The tag preserves a historical snapshot of every tracked file at that commit,
including [`report/paper_draft.md`](report/paper_draft.md). That file is the
current, living manuscript: release 2.0.0 does not declare it final, frozen, or
separately versioned, and later manuscript edits on `main` do not by themselves
require another repository release. The release-controlled scientific boundary
is the frozen artifact and provenance inventory described below.

The release includes repository source and the Git-tracked frozen evidence
described below. It excludes raw/processed patient images and model binaries.
See [`RELEASE_NOTES_v2.0.0.md`](RELEASE_NOTES_v2.0.0.md) for the release scope
and delta from `v1.0.0`, [`CHANGELOG.md`](CHANGELOG.md) for release history, and
[`CITATION.cff`](CITATION.cff) for the available software citation metadata.

## Canonical document hierarchy

| Location | Canonical role |
|---|---|
| [`report/paper_draft.md`](report/paper_draft.md) | **Current, living manuscript.** Reporting audits, hypothesis links, and submission preparation must be assessed against this file. A tag records its historical state but does not freeze or version the manuscript as a publication. |
| [`report/report.md`](report/report.md) | **Historical/full technical report.** It is preserved as a detailed project record and is not the current paper. Do not rewrite it merely to mirror the manuscript. |
| [`docs/`](docs/) | **Analysis, methodology, decision, and audit record.** These files explain provenance and scope but do not replace the manuscript. |
| [`results/`](results/) | **Numerical source of truth.** Manuscript prose and tables are rounded views of these generated artifacts. |
| [`results/scientific_artifact_manifest.json`](results/scientific_artifact_manifest.json) | **Frozen critical-artifact inventory.** Hashes, schemas, generators, configs, inputs, phases, and regeneration requirements. |
| [`results/publication_artifact_manifest.json`](results/publication_artifact_manifest.json) | **Final publication inventory.** Extends the unchanged internal inventory with frozen external aggregate evidence and explicit private-input boundaries. |
| [`report/provenance/batch46/`](report/provenance/batch46/) | **Immutable editorial baseline only.** Exact pre-external manuscript/claim bytes bound by the original protocol; never independently maintained. |
| [`report/paper_claim_sources.yaml`](report/paper_claim_sources.yaml) | **Numerical claim bindings.** Exact source cells/calculations and manuscript rounding tolerances. |

The consolidated scope statement is
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), and the current reporting,
hypothesis, citation, and author-declaration audits are indexed from
[`docs/REPORTING_CHECKLIST.md`](docs/REPORTING_CHECKLIST.md). If prose and a
generated result disagree, resolve the discrepancy from `results/` rather than
treating either report document as a numerical source.

## Headline result

The final manuscript separates ranking, operating-point behavior and external
transportability. On the strict VinDr target (3,000 images; 84 positive images;
95 boxes), equal-run AP@0.50 was 0.002505/0.000501 and historical-threshold
recall 0.018947/0.004211 (Faster R-CNN/YOLO11s). Preserved observed AP ordering
does not mean preserved performance: absolute AP and operating-point performance
collapsed. Most external FROC contrasts and all historical-threshold metric
contrasts include zero. The external upper-budget missing-support bound permits
a reversal beyond retained candidates; Faster R-CNN cap saturation is another
limit. See [complete transportability evidence](docs/VINDR_STATISTICS_RESULTS.md)
and [final manuscript audit](docs/FINAL_MANUSCRIPT_AUDIT.md).

For the **internal benchmark**, across seeds 17, 42, 137, 271, and 314, Faster R-CNN achieves
mAP@0.5:0.95 of 0.0995 ± 0.0067 versus 0.0542 ± 0.0060 for YOLO11s.
At thresholds selected by maximum mean validation F1 in the frozen original
n=3 analysis and applied unchanged to all five test bundles, sensitivity test
precision/recall/F1 is 0.3624/0.3507/0.3511 versus
0.2524/0.1948/0.2192. Faster R-CNN has higher precision at 97 of 101 official
AP@0.5 recall positions in the n=5 sensitivity; YOLO11s' apparent
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
computational cost. Under the matched decoded-host v1 boundary on the reporting
laptop, YOLO11s achieved 57.56 FPS versus 20.91 FPS and median latency
17.29 ms versus 47.20 ms, using the same 100 images and three technical repeats
of the primary frozen checkpoints. Parameter counts are 9.43 M versus 43.26 M.
The historical five-run asymmetric profiles remain preserved separately. On the
internal observed exact-score FROC frontier, Faster R-CNN has higher sensitivity at all five
prespecified FP/image operating budgets. User-approved inference at a 0.0001
candidate floor and then 0.00001 materially narrows the higher-budget gap.
YOLO11s seed 137 still ends just below 2 FP/image, but even a mathematical-
maximum bound for its missing sensitivity cannot reverse the detector ordering.
Neither detector strictly
dominates the n=5 accuracy-efficiency Pareto panels. Original n=3 and n=5 grid
artifacts remain unchanged as historical provenance.

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
checkpoints are intentionally Git-ignored. The commands below document their
generation, but a Git checkout alone cannot regenerate inference or training
evidence without those licensed/external inputs. The exact ten trained
checkpoints still existed locally at the 2026-08-31 audit and are hash-bound in
`results/checkpoint_release_manifest.json`; they have not been uploaded and no
public download URL is claimed. The measured workstation used an RTX 4060
Laptop GPU with 8 GB VRAM, 16 GB RAM, and an i7-13650HX; timing will vary on
other machines.

### Reproducibility boundary

These are distinct claims; success at one level does not establish the next:

| Level | Scope |
|---|---|
| Software tests pass | CI checks the locked CPU install, formatting/lint, tests, package smoke path, critical-artifact hashes/schemas/references, and selected manuscript claim bindings. It performs no training, large download, or GPU inference. |
| Committed-analysis reproduction | CPU analysis commands replay derived evidence from frozen committed predictions/provenance. This does not recreate the predictions. |
| Exact inference reproduction | Requires the exact ten checkpoint hashes, licensed/processed RSNA data, configs, pinned CUDA stack, and a compatible GPU. It is outside standard CI. |
| Exact retraining reproducibility | Requires the original data and pretrained initialization plus the full ten-run GPU budget. Fixed seeds and environment make the protocol reproducible, but warning-only CUDA ROI Align nondeterminism prevents a bitwise-identity guarantee. It is outside standard CI. |

Green CI means that the software and committed evidence snapshot pass their
declared checks. It never means that all published results were regenerated
from raw data. See `docs/REPRODUCIBILITY.md` for the full contract, environment
lock roles, RNG controls, known nondeterminism, and checkpoint release audit.

## 1. Create the pinned environment

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv --default-index https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps --editable .

$benchmarkPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $benchmarkPython -m pip check
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py scripts/build_scientific_artifact_manifest.py
```

`pyproject.toml` declares project dependencies, `uv.lock` is the CI resolver
lock, `.python-version` fixes Python 3.11.15, and `requirements.txt` records the
exact adopted CUDA-environment package versions for the setup above. Each
experiment also writes `pip_freeze.txt` and `run_environment.json` before CUDA
initialization. Deterministic algorithms use warning mode because the pinned
Torchvision CUDA ROI Align backward used by Faster R-CNN/Grad-CAM is not
bitwise deterministic.

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

The split counts in report Section 3 are bound to
`data/manifests/rsna-pneumonia-5000-audit.json` and the committed split
manifests. To regenerate metadata, splits, and COCO annotations without
decoding pixels, use:

```powershell
& $benchmarkPython -m src.data.prepare --config configs/dataset.yaml --metadata-only
```

### 2a. Generate aggregate cohort characteristics

The original DICOM root defaults to `dataset.paths.source_images_dir` in
`configs/dataset.yaml`. If the files are stored elsewhere, supply the root at
runtime rather than editing or hardcoding a machine-specific path:

```powershell
$env:RSNA_DICOM_ROOT = "<directory-containing-stage-2-training-DICOMs>" # optional override
& $benchmarkPython -m src.data.cohort_characteristics --config configs/cohort_characteristics.yaml
```

If neither the config path nor `RSNA_DICOM_ROOT` resolves to the complete
selected source, the command stops with an instruction to set
`RSNA_DICOM_ROOT`. It verifies the immutable split hashes and counts, checks all
5,000 rows against the official NIH patient mapping, then reads only
`PatientAge`, `PatientSex`, and `ViewPosition` without decoding pixels. It
writes only the four-row aggregate table
`results/tables/rsna_cohort_characteristics.csv` and aggregate provenance
summary under `results/logs/phase44_cohort_characteristics/`; no identifier- or
patient-level demographic output is created.

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

## Standardized end-to-end inference timing (Batch 45)

The [v1 protocol](docs/COMPUTE_TIMING.md) starts from decoded host images and
includes preprocessing, transfer, forward, NMS, coordinate restoration and CPU
outputs for both detectors, excluding disk I/O. It uses the identical 100-image
subset, batch 1, 10 warm-up images per detector/repetition, and three complete
technical repetitions of the seed-17 checkpoints. No training runs.

The exact command used for publication timing on the project's verified laptop
uses its existing CUDA interpreter (the current repository `.venv` is CPU-only):

```powershell
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.benchmark_inference --config configs/inference_timing_v1.yaml --mode preflight
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.benchmark_inference --config configs/inference_timing_v1.yaml --mode run
& .\.venv\Scripts\python.exe -m src.benchmark_inference --config configs/inference_timing_v1.yaml --mode verify
& .\.venv\Scripts\python.exe -m pytest tests/test_inference_timing.py -q --basetemp=tmp/pytest-timing -p no:cacheprovider
```

`run` refuses to overwrite accepted outputs and refuses non-reporting hardware
or a mismatched CUDA/framework stack. Use `--mode report` to regenerate tables
and the figure from verified saved intervals. Full inference reproduction
requires the hash-matched checkpoints and licensed processed images. For a new
timing campaign preserve this version and review a new output/protocol version.
Primary outputs are `results/tables/inference_timing_v1{,_repetitions,_images}.csv`,
`results/figures/inference_timing_v1.png`, and the summary/environment/historical
preservation manifest in `results/logs/phase45_inference_timing_v1/`.

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

The historical threshold analysis is CPU-only and remains frozen to the six
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

The Batch 35 n=5 all-attempt sensitivity reads all ten already-frozen test
prediction bundles and writes only `*_n5_sensitivity` artifacts. It performs
no training, checkpoint loading, or inference:

```powershell
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep_n5_sensitivity.yaml --mode run
```

This produces versioned aggregate/per-run threshold and official PR tables,
the n=5 PR/F1 figures, and
`results/logs/phase35_operating_regime_n5/threshold_summary.json`. It does not
overwrite any unsuffixed n=3 artifact.

### 5b. Select primary single operating thresholds on validation

The original training runs retained validation aggregates but not the raw scored
detections needed for a threshold sweep. The frozen n=3 workflow materializes
those records from the six immutable best checkpoints for seeds 17, 42, and 137
on the 750-image validation split, then performs selection and the one-shot
test application offline:

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

Apply those unchanged 0.69/0.05 thresholds to the n=5 test sweep without
reselection:

```powershell
& $benchmarkPython -m src.analyze_operating_regime_sensitivity --config configs/operating_regime_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.analyze_operating_regime_sensitivity --config configs/operating_regime_n5_sensitivity.yaml --mode fixed-thresholds
```

The outputs visibly separate `threshold_selection_run_count=3` from
`test_run_count=5` and retain seed 271's zero-detection row at 0.05.

#### 5b.1 Run the post-hoc five-validation-run selection sensitivity

This separate workflow leaves every historical n=3 artifact unchanged. It
uses validation evidence from seeds 17, 42, 137, 271, and 314 with the same
99-point maximum-mean-F1 rule and then applies the selected thresholds once to
all five frozen internal-test bundles. The result is a **post-hoc validation
sensitivity**, never a prospectively frozen threshold choice.

```powershell
& $benchmarkPython -m src.analyze_validation_threshold_sensitivity --config configs/threshold_selection_n5_validation_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.analyze_validation_threshold_sensitivity --config configs/threshold_selection_n5_validation_sensitivity.yaml --mode collect-validation
& $benchmarkPython -m src.analyze_validation_threshold_sensitivity --config configs/threshold_selection_n5_validation_sensitivity.yaml --mode run
```

`preflight` is read-only. `collect-validation` loads only the exact frozen best
checkpoints for missing seed-271/314 validation bundles and performs no
training. Once those four bundles exist, `run` is CPU-only and reproduces the
historical 0.69/0.05 selection before selecting 0.70/0.01 over all five
validation runs. The versioned summary is
`results/logs/phase43_threshold_selection_n5_validation_sensitivity/summary.json`;
the aggregate comparison is
`results/tables/threshold_selection_test_operating_points_n5_validation_sensitivity.csv`.

### 5c. Reparameterize the test sweep as FROC curves

This CPU-only step reads the unchanged Batch 10 per-seed table and describes the
full test sweep as sensitivity versus false positives per image. It does not
select a deployment threshold or perform training, checkpoint loading, or
inference. The archive-safe historical reproduction remains:

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

The separately labeled n=5 path writes aggregate and per-run FROC curves, five
budget summaries, a figure, and provenance:

```powershell
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n5_sensitivity.yaml --mode run
```

The current exact-score repair uses user-approved inference-only bundles at a
0.00001 candidate floor, evaluates every unique retained score, adds explicit
empty and candidate-floor endpoints, and compares the observed frontier with
the historical grid. The first two commands require the pinned CUDA runtime;
the final exact-score analysis is CPU-only:

```powershell
& $benchmarkPython -m src.collect_froc_lower_floor_predictions --config configs/evaluation_froc_lower_floor_v4.yaml --source-config configs/evaluation.yaml --prior-config configs/evaluation_froc_lower_floor_v3.yaml --checkpoint-manifest results/checkpoint_release_manifest.json --mode preflight
& $benchmarkPython -m src.collect_froc_lower_floor_predictions --config configs/evaluation_froc_lower_floor_v4.yaml --source-config configs/evaluation.yaml --prior-config configs/evaluation_froc_lower_floor_v3.yaml --checkpoint-manifest results/checkpoint_release_manifest.json --mode run
& $benchmarkPython -m src.analyze_exact_score_froc --config configs/froc_exact_score_v4.yaml --mode preflight
& $benchmarkPython -m src.analyze_exact_score_froc --config configs/froc_exact_score_v4.yaml --mode run
```

The 0.00001 result removes the previous 1-FP/image boundary, but YOLO11s seed
137 reaches only 1.9907 FP/image and remains floor-limited at 2. The observed
YOLO11s aggregate is 0.6090; its conservative maximum is 0.6963 versus Faster
R-CNN's observed 0.6978, so the qualitative ordering cannot reverse. No still-
lower inference is authorized; see `docs/FROC_ANALYSIS.md`.

### 5d. Regenerate the accuracy-efficiency Pareto figure

The historical CPU-only step joins the archived n=3 Phase 5 accuracy rows, all
six original seed-specific compute tables, and validation-selected,
test-evaluated operating points:

```powershell
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode preflight
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode run
```

The run regenerates `results/figures/pareto_frontier.png`; definitions and the
scenario-conditional interpretation are in `docs/PARETO_ANALYSIS.md`.

The n=5 sensitivity joins all ten run-specific accuracy/compute rows to test
recall at the unchanged n=3-selected thresholds. It writes a visibly labeled
figure, exact point table, equal-run aggregate table, and provenance:

```powershell
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto_n5_sensitivity.yaml --mode run
& $benchmarkPython -m src.analyze_operating_regime_sensitivity --config configs/operating_regime_n5_sensitivity.yaml --mode run
```

The final command also writes the four-analysis inventory, the complete
19-row n=3-versus-n=5 conclusion table, and the aggregate Batch 35 provenance
summary. No n=3 and n=5 metric is mixed into one Pareto frontier.

### 5e. Measure five-seed detection calibration

This CPU-only analysis reads all ten frozen Phase 5 prediction bundles. It
matches every detection retained at the 0.001 bundle floor through the canonical
IoU-0.50 matcher and computes the full box-sensitive Detection Expected
Calibration Error over confidence, relative center, width, and height. Class is
a categorical stratum. The run also records per-run cell support/occupancy and
executes the predeclared bin/minimum-cell and confidence-floor sensitivity grid.
It does not fit a calibrator, load checkpoints, run inference, or train a model:

```powershell
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode run
```

The run regenerates the versioned `calibration_summary_v2.csv`,
`calibration_support_v2.csv`, and `calibration_sensitivity_v2.csv` tables; the
confidence-only marginal reliability, support/occupancy, binning/support, and
floor-sensitivity v2 figures; and exact provenance at
`results/logs/phase33_calibration_support_v2/summary.json`. Definitions,
descriptive findings, population limits, and the separation from exam-level
probability calibration/DCA are in `docs/CALIBRATION_ANALYSIS.md`.

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

### 5g. Audit the non-standard raw-score threshold utility calculation

Batch 30 classified the historical Batch 20 calculation as non-standard for
conventional decision-curve interpretation: it used maximum detector confidence both to
define action and to supply the `tau/(1-tau)` false-positive weight. The CPU-only audit
recomputes that exact arithmetic from all ten frozen Phase 5 test bundles, verifies it
against immutable historical archives, and emits explicitly relabeled exploratory
raw-score utility/sensitivity artifacts. It performs no checkpoint inference, training,
probability calibration, or standard DCA:

```powershell
& $benchmarkPython -m src.clinical.raw_score_utility --config configs/raw_score_utility.yaml --mode preflight
& $benchmarkPython -m src.clinical.raw_score_utility --config configs/raw_score_utility.yaml --mode run
```

The run regenerates `results/tables/raw_score_threshold_utility_summary.csv`,
`results/figures/raw_score_threshold_utility_sensitivity.png`, and
`results/logs/phase30_raw_score_utility/summary.json`. The original code, config, table,
figure, and provenance remain under explicit `pre_batch30_nonstandard` archive names.
`docs/DCA_ANALYSIS.md` records the classification, six-of-ten validation-prediction
coverage that prevented probability-based salvage, prevalence boundary, and hashes.

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

## 6a. Audit the radiography-motivated synthetic acquisition/display analysis

Batch 32 reuses the frozen 300-image robustness sample and the historical
Batch 22 predictions without GPU inference. Every sampled object is a lossy,
workstation-converted Secondary Capture object (`ConversionType=WSD`), despite
recording `Modality=CR`. Pixel Intensity Relationship/Sign, Modality LUT or
Rescale, and VOI fields are absent in all 300 headers. The stored values
therefore do not establish an approximately linear X-ray-signal scale.

The four DICOM `LINEAR` Window Center/Width conditions are class-A display
transform sensitivities. The signal-dependent Poisson-like conditions are
physically unsupported as dose/quantum-noise proxies for these values and are
retained only as class-B generic intensity perturbations. Gaussian kernels are
class-C spatial-resolution-motivated blur proxies, not scanner- or
reconstruction-specific models. The audit also measures every perturbation
before and after canonical per-image min-max scaling. `DSI = 1 - shifted /
clean` remains a descriptive performance-retention/domain-sensitivity index,
not an estimate of inter-site transportability.

```powershell
& $benchmarkPython -m pytest tests/test_radiography_shifts.py tests/test_prepare.py -q
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode preflight
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode audit
```

This CPU-only audit generates:

- `results/tables/acquisition_shift_dicom_metadata_audit.csv`, one row per
  sampled image with field-level missingness;
- `results/tables/acquisition_shift_preprocessing_per_image.csv` and
  `acquisition_shift_preprocessing_summary.csv`, with pre/post-min-max
  perturbation diagnostics;
- `results/tables/radiography_synthetic_shift_results.csv`, which relabels the
  unchanged historical detector results under the corrected scientific scope;
  and
- `results/logs/phase32_acquisition_shift_audit/summary.json`, including hashes
  of all audit products and the 20 unchanged historical prediction bundles.

The original `acquisition_shift_results.csv`, Phase 22 summary, and prediction
bundles remain preserved as superseded historical evidence. Re-running
`--mode smoke` or `--mode run` is unnecessary for this correction.

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

## 7a. Run the Grad-CAM sensitivity controls

This checkpoint-only extension selects 50 images from the already-frozen
robustness manifest, then compares trained maps with six detector-specific,
cumulative head-to-input model-parameter randomization stages and a
deterministic within-image pixel-vector permutation. The latter is an
input-pixel perturbation control, not Adebayo training-label data
randomization. That randomized-label retraining experiment was not performed.
Preflight validates Phase 6/7 provenance and the historical artifact hashes
before CUDA is initialized; the run also verifies checkpoint immutability.

```powershell
& $benchmarkPython -m pytest tests/test_xai_sanity.py tests/test_gradcam.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode preflight
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode run
```

This generates:

- `results/tables/gradcam_sanity_v2_summary.csv`;
- `results/tables/gradcam_sanity_v2_per_image.csv`;
- `results/figures/gradcam_sanity_v2_panel.png`; and
- `results/logs/phase31_xai_sanity_v2/summary.json` plus the nested-subset manifest.

The unversioned Batch 21 table/detail/panel and Phase 21 summary remain frozen
historical artifacts. Their legacy `data_randomization` label means the
input-pixel control only.

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
Threshold selection remains n=3, while test-side threshold/PR and Pareto have
separately versioned n=5 sensitivities and FROC now uses the five-run observed
exact-score frontier; robustness and explainability remain seed-17-only.

## Report artifact-to-command index

Report and paper-draft values are rounded views of committed machine-readable
artifacts; manuscript assembly does not recompute them. The index explicitly
includes every executable analysis module added in Batches 18--23. The
scientific artifact manifest makes the manuscript-critical subset and its full
hash/schema/config/input boundary machine-readable; the claim-source manifest
adds exact bindings for central numerical prose and table claims.

| Report or paper item | Generated source | Regenerating command |
|---|---|---|
| Paper §3.1 cohort construction; report Figures 1–2 | audit/split manifests; `rsna_*.png`; `rsna_eda_summary.json` | `src.data.prepare`, then `src.data.visualize` in §2 |
| Paper Table 1 / §3.1 cohort table | `rsna_cohort_characteristics.csv`; Phase 44 aggregate summary | `src.data.cohort_characteristics` in §2a |
| Paper §§3.2–3.3 protocol parameters | run-level `resolved_config.json` / `resolved_experiment.json`; Phase 5 summary | detector train/finalize in §§3–4, then `src.evaluate --mode evaluate` in §5 |
| Historical report Table 2; Figure 3 | `faster_rcnn_*.csv`; Faster curve | seed-17 Faster R-CNN train/finalize in §3 |
| Historical report Table 3; Figure 4 | `yolo_*.csv`; YOLO curve | seed-17 YOLO train/finalize in §4 |
| Paper Table 2 / §4.1; historical report Tables 4a–4b | `detector_comparison*.csv` | unified `src.evaluate --mode evaluate` in §5 |
| YOLO seed-stability diagnostic | `yolo_seed_stability.csv` | `src.analyze_yolo_seed_stability` in §5 |
| Paper Figures 1a/1b / §§4.1–4.2 PR/F1 evidence | principal `*_n5_sensitivity` threshold/PR tables and figures; unsuffixed n=3 history retained | offline `src.evaluate_threshold_sweep --mode run` in §5a |
| Paper §4.2 validation-selected operating points | n=3 validation selection plus `selected_operating_points*_n5_sensitivity.csv` test application | `src.evaluate_threshold_selection` plus offline `src.analyze_operating_regime_sensitivity` in §5b |
| Paper §4.3 internal exact-score FROC evidence | approved lower-floor bundles and `froc_exact_score_*_v4` curves, prespecified-budget tables, comparisons, bound, figure, and summary; v2/v3 and historical grids retained | inference-only collection plus offline `src.analyze_exact_score_froc --mode run` in §5c |
| Paper Table 4 / Figure 3 / §4.5: primary matched timing (Batch 45) | `inference_timing_v1{,_repetitions,_images}.csv`; `inference_timing_v1.png`; Phase 45 summary and preservation manifest | `src.benchmark_inference --mode run` and offline `--mode report`; exact commands in the Batch 45 section above |
| Supplementary historical Pareto evidence | `pareto_{points,summary}_n5_sensitivity.csv`; `pareto_frontier_n5_sensitivity.png`; n=3 figure retained | offline `src.plot_pareto_frontier --mode run` in §5d; historical asymmetric timing axes |
| Supplementary five-seed detection calibration and support sensitivity (Batch 33 v2) | `calibration_summary_v2.csv`; `calibration_support_v2.csv`; `calibration_sensitivity_v2.csv`; four v2 figures; Phase 33 summary | offline `src.stats.calibration --mode run` in §5e |
| Supplementary recall-weighted F-beta and hypothetical-loss sensitivity (Batch 29; frozen n=3 validation) | `recall_weighted_fbeta_threshold_summary.csv`; `recall_weighted_fbeta_threshold_stability.csv`; `hypothetical_detection_error_loss_summary.csv`; corrected sensitivity figure; Phase 29 summary | offline `src.stats.threshold_calibration --mode run` in §5f |
| Supplementary non-standard raw-score utility audit (Batch 30) | `raw_score_threshold_utility_summary.csv`; `raw_score_threshold_utility_sensitivity.png`; Phase 30 summary; exact pre-Batch-30 archives | offline `src.clinical.raw_score_utility --mode run` in §5g |
| Supplementary seed-level predictive/historical-compute rainclouds (Batch 23) | `detector_comparison.csv`; `detector_comparison_per_seed.csv`; `raincloud_metrics.png`; Phase 23 summary | audited `src.plot_raincloud_metrics --mode run` in §5h |
| Supplementary corruption evidence; historical report Table 5 and Figures 5–6 | `robustness*.csv`; robustness plots | robustness `--mode run` in §6 |
| Supplementary radiography-motivated synthetic acquisition/display sensitivity (Batch 32) | per-image DICOM audit; pre/post-min-max diagnostics; `radiography_synthetic_shift_results.csv`; Phase 32 summary; unchanged historical bundles | CPU-only `src.robustness.radiography_shifts --mode audit` in §6a |
| Historical report Table 6; Figures 7–9 | `gradcam*.csv`; Grad-CAM plots | explainability `--mode run` in §7 |
| Supplementary Grad-CAM v2 parameter cascade and input-pixel control (Batch 31) | `gradcam_sanity_v2*.csv`; `gradcam_sanity_v2_panel.png`; Phase 31 summary; frozen Batch 21 artifacts | checkpoint-only `src.explainability.sanity_checks --mode run` in §7a |
| Paper Table 3 / §4.4; historical report Table 7 | `statistical_clean_comparison.csv` | statistics `--mode run --scope clean` in §8 |
| Frozen seed-17 corruption inference | `statistical_robustness_comparison.csv` | prior full-scope statistics `--mode run` after §6; not rerun for n=5 |
| Paper Table 5–7 / Figures 2 and 4 / §§4.6–4.8 | VinDr adapter/inference/statistical summaries and five CSVs; internal/external FROC and threshold-transport figures | Frozen external and statistics commands below; current read-only replay commands in Batch 51 verification |

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
  parameters, incomplete profiler-registered GFLOPs, memory, and training time.
  Batch 45 separately replaces primary asymmetric timing with the matched
  decoded-host v1 protocol, verified on the intended reporting laptop.
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
- [x] **Scope-grounded discussion.** Historical report Section 11 addresses
  the course brief's hypothetical screening and constrained-assistance
  scenarios while rejecting autonomous use. The current paper limits its
  interpretation to the measured benchmark dimensions and makes no
  clinical-utility recommendation.
- [x] **Honest consolidated limitations.** `docs/LIMITATIONS.md` covers the
  single dataset, five-seed clean headline scope with four complete conditional
  localization pairs, seed-271 confidence/output-score instability,
  primary-seed 300-image/111-box robustness and explainability scope,
  augmentation choice, detector asymmetries, and RTX 4060 8 GB / 16 GB RAM
  constraints.
- [x] **Documented reproduction commands and boundary.** Sections 1–8 provide
  the exact ordered commands and the artifact index maps report evidence to
  generators. A clean checkout can run software verification and the committed
  offline analyses whose frozen inputs are present; exact inference and
  retraining additionally require the external data/checkpoints described
  above.

## VinDr-CXR frozen external testing (Batches 47–49)

Review [`docs/VINDR_EXTERNAL_PROTOCOL.md`](docs/VINDR_EXTERNAL_PROTOCOL.md)
and [`configs/vindr_external_v1.yaml`](configs/vindr_external_v1.yaml) before
Batch 49. The official v1.0.0 test set is fixed at 3,000 images. The strict
target concept `Lung opacity` has the exact CSV spelling `Lung Opacity`; no
other finding is merged. Historical RSNA thresholds remain 0.69/0.05;
0.70/0.01 is a separately labeled post-hoc n=5 sensitivity. Full inference
and external performance inspection were not performed in Batch 47.

The user must personally obtain credentialed access, complete the required
training/DUA and download the official release. The current user confirmed
this prerequisite on 2026-09-22. No credentials belong in this repository.
Set the root to the parent of `test/` and the two annotation CSVs. For the
user-supplied repository layout, from the repository root:

```powershell
$env:VINDR_CXR_ROOT = (Resolve-Path -LiteralPath 'data/raw/vindr-cxr').Path
```

There is no config fallback root. On another machine set `VINDR_CXR_ROOT` to
that machine's authorized release directory. Restricted source and derived
row-level data stay in ignored `data/`; only reviewed nonidentifying aggregates
may enter the configured result directory.

Run this **offline, no-inference** freeze check from the repository root before
implementing the adapter and again before full inference. It checks both frozen
files and their internal definition/provenance dependencies; a mismatch means
stop and review, not regenerate the sidecar:

```powershell
$vindrFreeze = Get-Content -Raw -Encoding utf8 'docs/VINDR_EXTERNAL_PROTOCOL_v1.sha256.json' | ConvertFrom-Json
foreach ($vindrArtifact in $vindrFreeze.artifacts) {
    if (-not (Test-Path -LiteralPath $vindrArtifact.path -PathType Leaf)) {
        throw "Missing frozen input: $($vindrArtifact.path)"
    }
    $vindrActual = (Get-FileHash -LiteralPath $vindrArtifact.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($vindrActual -ne $vindrArtifact.sha256) {
        throw "Frozen input changed: $($vindrArtifact.path)"
    }
}
Write-Output "VinDr v1 freeze verified: $($vindrFreeze.artifacts.Count) files; no inference."
```

The Batch 48 adapter consumes the unchanged frozen config plus operational
settings in `configs/vindr_adapter_v1.yaml`. Run from the repository root with
the authorized `VINDR_CXR_ROOT` set as above. The CPU preparation environment
is sufficient; this does not change the mandatory CUDA/AMP contract for later
detector inference.

```powershell
# Full data-only audit: checksums, headers, all decoded pixels, PNG round trips,
# strict targets, deterministic repeat sample and native coordinate restoration.
& .\.venv\Scripts\python.exe -m src.data.prepare_vindr --config configs/vindr_external_v1.yaml --adapter-config configs/vindr_adapter_v1.yaml --mode preflight
# Includes the same full audit and additionally writes all native PNGs, the
# sorted manifest, canonical COCO annotations and common-loader integration.
& .\.venv\Scripts\python.exe -m src.data.prepare_vindr --config configs/vindr_external_v1.yaml --adapter-config configs/vindr_adapter_v1.yaml --mode prepare
& .\.venv\Scripts\python.exe -m pytest tests/test_prepare_vindr.py tests/test_prepare.py tests/test_faster_rcnn_data.py tests/test_evaluation.py tests/test_coco_evaluation.py -q --basetemp=tmp/pytest-vindr-adapter -p no:cacheprovider
```

`prepare` subsumes `preflight`; running both is optional. Neither loads a
detector. Both write the restricted detailed audit under the private root and
the nonidentifying aggregate `results/vindr_external_v1/adapter_preflight.json`.
Only `prepare` writes `manifest.csv`, `instances_test.json`, `images/*.png` and
`dataset_loader.yaml` under ignored `data/processed/vindr-cxr-external-v1/`.
The generated loader config works with `CocoDetectionDataset(..., "test")`;
canonical evaluator IDs are COCO `file_name` values (including `.png`), with
native `(height, width)` and original-coordinate boxes. The manifest also
retains the original source image ID locally. No patient grouping is invented.
Source/PNG hashes, source-row box order, sorted image order, decoder versions,
adapter/config hashes and the unchanged scientific contract are recorded.
The technical sample is the first sorted image per transfer-syntax/polarity/
bit-depth stratum plus every strict-target-positive image. All images still
undergo checksum, decode, dimension and PNG checks; none is excluded.

See [the adapter preflight review](docs/VINDR_ADAPTER_PREFLIGHT.md). Restricted
images, headers, identifiers, annotations and predictions must remain private.
These commands stop on integrity/schema/semantics failures or existing external
prediction/result artifacts; do not reuse them after performance collection
without reviewing that gate. A failed/interrupted conversion is incomplete and
must not be used for inference; require a successful `prepare` summary and
matching private artifact hashes. Repeat verification may replace the summary
with `mode: preflight`, which is not a completed preparation receipt.

Batch 49 uses `src.evaluate_vindr_external` with the unchanged scientific
protocol and the operational paths in `configs/vindr_inference_v1.yaml`.
The user explicitly requested Batch 49 on 2026-09-23 after the Batch 48 review
checkpoint. Run sequentially on the reporting RTX 4060 laptop in the pinned
CUDA environment, with the authorized dataset root set above:

```powershell
# Read-only: 38 frozen definitions, ten checkpoints, all source/PNG hashes,
# source-to-COCO equality and complete common-loader decoding.
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.evaluate_vindr_external --config configs/vindr_external_v1.yaml --operation-config configs/vindr_inference_v1.yaml --mode preflight
# Full run repeats the gate before any detector inference; never overwrites.
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.evaluate_vindr_external --config configs/vindr_external_v1.yaml --operation-config configs/vindr_inference_v1.yaml --mode run
# No inference: verify hashes and recompute every metric from private bundles.
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.evaluate_vindr_external --config configs/vindr_external_v1.yaml --operation-config configs/vindr_inference_v1.yaml --mode verify
& .\.venv\Scripts\python.exe -m pytest tests/test_evaluate_vindr_external.py tests/test_exact_score_froc.py tests/test_inference_timing.py -q --basetemp=tmp/pytest-vindr-inference -p no:cacheprovider
# Format the verified aggregate evidence as readable per-run tables (no analysis).
& .\.venv\Scripts\python.exe -m scripts.report_vindr_inference --config configs/vindr_inference_report_v1.yaml
# PLANNED Batch 50, verified predictions only; no model inference:
# & $benchmarkPython -m src.analyze_vindr_external --config configs/vindr_external_v1.yaml --mode run
```

Both native candidate filters use strict `score > floor`; common evaluation
uses `score >= threshold` on emitted candidates. Synthetic equality tests
record this boundary without changing the frozen floor. Actual convolution
dtypes must be float16 for Faster R-CNN and bfloat16 for YOLO11s, using the
Batch 45 explicit autocast path with `quantize=None`.

The private prediction tree retains all images, including empty outputs,
exact scores and native coordinates, exact-score FROC curves, environment
files, execution metadata and hashes. The nonidentifying
`results/vindr_external_v1/inference_summary.json` contains all ten per-run
AP/FROC/threshold/score results and equal-run descriptive mean/sample SD with
defined counts. Historical n=3 threshold transport is primary; post-hoc n=5
transport is secondary. Study percentages use released images, not inferred
patient counts. No external threshold is selected. The runner refuses to
overwrite or automatically resume existing output; a failure requires review
with all existing evidence preserved. Never manually edit generated artifacts.

The initial attempt exposed a one-ULP Torchvision inverse-resize boundary
overshoot before any complete run or performance calculation. The corrected
handoff repairs only that numerical upper-bound error and preserves raw
coordinates privately; larger errors still stop. See the
[implementation and failure record](docs/VINDR_INFERENCE.md). The bounded
diagnostic command used against the stopped first-attempt receipt was:

```powershell
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m scripts.diagnose_vindr_bounds --config configs/vindr_external_v1.yaml --operation-config configs/vindr_inference_v1.yaml --maximum-images 200
```

This diagnostic intentionally refuses successful or already-diagnosed runs.
The initial receipt and diagnostic are now preserved in the private
`superseded/batch49_initial_bounds_failure/` archive, with original source and
config bytes and a hash manifest; do not move them back over accepted outputs.

The Batch 49 commands above do not execute Batch 50 statistics; its separately
authorized offline analysis and commands are documented below.

Batch 49 completed all ten runs and passed full hash/metric replay verification.
See [per-run results](docs/VINDR_INFERENCE_RESULTS.md) and the
[execution and findings record](docs/VINDR_INFERENCE.md). Both pipelines show
severely reduced external performance under the strict ontology; the historical
thresholds yield near-zero recall, and YOLO seed 271 emits no detections at its
historical cutoff. All adverse results, candidate-floor/cap limits and both
threshold policies are retained. No frozen scientific setting was tuned.
The freeze record is local provenance, not a public registration or trusted
timestamp. Preserve v1; record amendments under a new version and retain any
superseded evidence.

## VinDr statistics and cross-dataset transportability (Batch 50)

[Methods and estimands](docs/VINDR_STATISTICS.md) and the generated
[results and three figures](docs/VINDR_STATISTICS_RESULTS.md) compare the
separate internal and external datasets. All five runs per detector remain.
The observed AP/FROC ordering is unchanged, while absolute performance and
RSNA-frozen operating-point recall collapse externally. The two external AP
contrast intervals remain positive; four of five FROC-budget contrasts and
all four historical-threshold contrasts include zero. This is cross-dataset
transportability, without attributing the shift to a specific confounded factor.

The analysis inherits all scientific choices from the Batch 47 freeze. It
uses 2,000 shared observation/independent detector-run bootstrap draws per
dataset and separate seed-17 checkpoint-conditional sensitivity. RSNA resamples
its 323 NIH patient groups; VinDr resamples 3,000 released images because no
defensible patient grouping exists. Historical n=3 thresholds remain primary;
Batch 43's post-hoc n=5 policy stays descriptive. No external threshold or
ontology tuning, training, new inference, dataset pooling or new hypothesis
tests occur. Floor-limited FROC values remain observed lower bounds.

Run from the repository root in the pinned environment. The initial Batch 49
read-only verification requires the authorized local source dataset; the
offline statistics use existing local annotations, prediction bundles and
checkpoint hashes. They do not require a GPU. These commands reproduce every
new displayed number and figure:

```powershell
# Verify the complete prerequisite experiment, including all source and metric replays.
$env:VINDR_CXR_ROOT = (Resolve-Path 'data/raw/vindr-cxr').Path
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.evaluate_vindr_external --config configs/vindr_external_v1.yaml --operation-config configs/vindr_inference_v1.yaml --mode verify

# Verify frozen inputs and run inventory; generate statistics once in a fresh output location.
& .\.venv\Scripts\python.exe -m src.stats.run_vindr_statistics --config configs/vindr_statistics_v1.yaml --mode preflight
& .\.venv\Scripts\python.exe -m src.stats.run_vindr_statistics --config configs/vindr_statistics_v1.yaml --mode run

# For the completed analysis: read-only hashes, aggregate and saved-draw interval replay.
& .\.venv\Scripts\python.exe -m src.stats.run_vindr_statistics --config configs/vindr_statistics_v1.yaml --mode verify

# Full deterministic recomputation from bundles: both datasets and every bootstrap draw.
& .\.venv\Scripts\python.exe -m src.stats.run_vindr_statistics --config configs/vindr_statistics_v1.yaml --mode replay

# Reproduce the three PNGs and review document from verified evidence, then check their hashes.
& .\.venv\Scripts\python.exe -m src.stats.run_vindr_statistics --config configs/vindr_statistics_v1.yaml --mode render

# Synthetic COCO-copy/FROC, joint-observation, independent-run and provenance regression tests.
& .\.venv\Scripts\python.exe -m pytest tests/test_transportability.py tests/test_statistics.py tests/test_evaluate_vindr_external.py tests/test_exact_score_froc.py tests/test_verify_scientific_artifacts.py -q --basetemp=tmp/pytest-batch50-focused -p no:cacheprovider
& .\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest-batch50-full -p no:cacheprovider
& .\.venv\Scripts\python.exe -m ruff check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
& .\.venv\Scripts\python.exe -m ruff format --check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
& .\.venv\Scripts\python.exe -m scripts.verify_paper_claims
& .\.venv\Scripts\python.exe -m scripts.verify_scientific_artifacts
git diff --check
```

`run` refuses to replace completed statistics or saved bootstrap draws; use
`verify` or `replay` on this completed checkout. The public summary, five CSVs
and three figures are under `results/vindr_external_v1/statistics/`. Source
hashes and all 44 marginal-interval rows are bound in its summary. Public
outputs contain no image identifiers. Restricted inputs and private plot/draw
working files stay in `data/processed/vindr-cxr-external-v1/statistics/` and
the existing ignored prediction tree. A public checkout without authorized
private inputs cannot run the complete external replay; ordinary CI tests
synthetic statistical logic and preserves the existing internal verifiers.
The historical manuscript and prior results are unchanged. Stop after Batch
50 review; manuscript integration is a separately requested Batch 51.

## Final manuscript verification (Batch 51)

`report/paper_draft.md` is the only editable journal manuscript. The final
scientific draft is ready for review, not a completed submission: author
ethics/consent, funding, conflicts, contributions, availability and involvement
statements still require human completion. The [audit](docs/FINAL_MANUSCRIPT_AUDIT.md)
records evidence gates and remaining reporting gaps.

Portable CI runs synthetic/unit tests and checks committed evidence; it does
not need licensed RSNA/VinDr images, local annotations, checkpoints or private
bootstrap files. Seven full-data exact-FROC integration tests are explicitly
marked `scientific_data` and deselected by default. Enable them in an authorized
local environment; missing inputs then fail rather than being skipped:

```powershell
# Portable default (the normal CI test command).
uv run --locked --extra cpu python -m pytest -q
# Full suite including authorized-data scientific integration checks.
& .\.venv\Scripts\python.exe -m pytest -q --run-scientific --basetemp=tmp/pytest-scientific -p no:cacheprovider
# Only those full-data integration checks.
& .\.venv\Scripts\python.exe -m pytest -q --run-scientific -m scientific_data --basetemp=tmp/pytest-scientific-only -p no:cacheprovider
# Final publication evidence, claims, bibliography and original freeze.
& .\.venv\Scripts\python.exe scripts/verify_scientific_artifacts.py --manifest results/publication_artifact_manifest.json
& .\.venv\Scripts\python.exe scripts/verify_paper_claims.py
& .\.venv\Scripts\python.exe scripts/check_bibliography.py --bibliography report/references.bib --manuscripts report/paper_draft.md report/report.md report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md
& .\.venv\Scripts\python.exe -m scripts.verify_frozen_external --mode freeze
```

Two split CSVs use file-specific Git attributes to preserve their original
cohort-bound byte hashes. Their rows and partition memberships are unchanged;
the earlier Git blobs differed only by newline normalization. Include those
byte-preservation diffs with the CI repair rather than refreshing expected hashes.

The Batch 47 freeze includes the then-current manuscript and claim manifest as
**editorial baseline** records. Their exact bytes are preserved under
`report/provenance/batch46/`; the original freeze, scientific configs, result
summaries and original scientific inventory remain unchanged. The read-only
wrapper below explicitly resolves only those two baseline reads to the archives.
It neither swaps files on disk nor changes a scientific check, source file,
threshold, numerical result or hash. It rejects baseline tampering and any
attempt to redirect scientific inputs. Current manuscript claims are checked
separately against the new publication inventory. Raw legacy verification
commands above refer to the pre-rewrite tree; use these current replay commands:

```powershell
# Use the existing authorized raw-data root; do not download or accept terms here.
$env:VINDR_CXR_ROOT = (Resolve-Path data/raw/vindr-cxr).Path
# Full source/checkpoint/provenance and per-run metric replay, without inference.
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m scripts.verify_frozen_external --mode inference
# All source bindings, aggregate replay and saved-draw interval reconstruction.
& .\.venv\Scripts\python.exe -m scripts.verify_frozen_external --mode statistics
# Additionally recompute every draw and all twenty internal/external point rows.
& .\.venv\Scripts\python.exe -m scripts.verify_frozen_external --mode replay
```

These replay modes require authorized local evidence. The inference verifier
retains a historical absolute adapter-source path binding and must run at the
recorded local project location; relocation needs a separately documented
provenance treatment, not a silent hash rewrite. The public freeze, artifact,
claim and bibliography checks remain portable. Restricted external images,
annotations, IDs, image-linked predictions and working draws are never added
to the public checkout.

The publication inventory is generated only by a deliberate maintenance
command, `python -m scripts.build_publication_manifest`, and reviewed as a diff.
CI never runs that builder. It retains every original internal inventory entry
and binds the external public summaries/tables/figures to their existing sources.
The [audit](docs/FINAL_MANUSCRIPT_AUDIT.md) maps manuscript methods, tables,
figures and inferential statements to deterministic commands and artifacts.

## Repository verification

The final internal/external manuscript checks are recorded in
[`docs/FINAL_MANUSCRIPT_AUDIT.md`](docs/FINAL_MANUSCRIPT_AUDIT.md).
The earlier [`internal focus audit`](docs/INTERNAL_PAPER_FOCUS_AUDIT.md)
is preserved as pre-external provenance, not current manuscript status. The shared
bibliography also supports the preserved historical report and long alternate;
the offline check below resolves citations in all three without editing them.

```powershell
& .\.venv\Scripts\python.exe scripts/check_bibliography.py --bibliography report/references.bib --manuscripts report/paper_draft.md report/report.md report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md
& .\.venv\Scripts\python.exe scripts/verify_paper_claims.py
& .\.venv\Scripts\python.exe scripts/verify_scientific_artifacts.py
& .\.venv\Scripts\python.exe -m pytest tests/test_check_bibliography.py tests/test_verify_paper_claims.py tests/test_verify_scientific_artifacts.py -q --basetemp=tmp/pytest-paper-review -p no:cacheprovider
& .\.venv\Scripts\python.exe -m ruff check scripts/check_bibliography.py tests/test_check_bibliography.py
& .\.venv\Scripts\python.exe -m ruff format --check scripts/check_bibliography.py tests/test_check_bibliography.py
```

The release-candidate software and committed-evidence checks are:

```powershell
uv lock --check
uv sync --locked --group dev --extra cpu
uv run --locked --extra cpu ruff format --check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
uv run --locked --extra cpu ruff check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
uv run --locked --extra cpu python -m pytest -q
uv run --locked --extra cpu python scripts/verify_scientific_artifacts.py --manifest results/publication_artifact_manifest.json
uv run --locked --extra cpu python scripts/verify_paper_claims.py
uv run --locked --extra cpu python -m meddet_benchmark smoke configs/smoke.yaml
git diff --check
```

Foundation CI runs these checks on Ubuntu and Windows and also applies Ruff to
both manifest builders, the bibliography checker and frozen-baseline verifier.
It checks the bibliography and original external freeze as well. The verifiers are
lightweight integrity/consistency checks over committed evidence; they do not
regenerate models or predictions. Green CI therefore verifies the repository's
declared software/evidence boundary, not full training or end-to-end scientific
regeneration. See `docs/REPRODUCIBILITY.md` for the complete four-level boundary
and the phase-specific documents under `docs/` for metric, corruption,
Grad-CAM, and inference definitions.

## Citation

Use [`CITATION.cff`](CITATION.cff) for the software citation metadata available
for release 2.0.0. No DOI, ORCID, venue, acceptance/publication status, final
paper citation, or release date is asserted there because those fields are not
yet established in authoritative project sources. This repository citation is
not evidence that the manuscript has been submitted, accepted, or published.
The current manuscript is [`report/paper_draft.md`](report/paper_draft.md);
its eventual authorship and submission declarations are independent of the
software/research-artifact release metadata.

## License

Copyright (C) 2026 Pouyan Delivandani, Mohammadamin Haji-Alirezaei.

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
this repository. The 2026-08-31 audit found all ten exact best checkpoints
locally and recorded their sizes/hashes in
`results/checkpoint_release_manifest.json`. Public release is assessed as
feasible only after the attribution, AGPL, Torchvision-pretraining permission,
and serialized-metadata review described in `docs/REPRODUCIBILITY.md`. No
checkpoint download link exists at present.
