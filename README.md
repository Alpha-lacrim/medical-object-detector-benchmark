# Medical Object Detector Benchmark

Controlled comparison of Faster R-CNN and YOLO11s on medical object detection,
covering predictive performance, compute, common-corruption robustness,
explainability, and statistical uncertainty. The authoritative requirements and
review checkpoints are in `PROJECT_SPEC.md` and `BATCHES.md`.

## Current status

Batch 6 / Phase 7 is complete and paused for explainability review. Faster
R-CNN and YOLO11s were each trained with seeds 17, 42, and 137 for the Phase 5
headline comparison. Phases 6 and 7 then use the primary seed-17 checkpoints on
the same fixed, stratified 300-image/111-box test subset for common-corruption
robustness and Grad-CAM localization respectively.

Across seeds, Faster R-CNN achieved mAP@0.5:0.95 of 0.1023 ± 0.0036 versus
0.0549 ± 0.0080 for YOLO11s, while YOLO11s measured 52.94 ± 10.65 FPS versus
17.42 ± 5.69 FPS. The full accuracy/compute comparison, metric definitions,
and framework-timing caveat are in `docs/QUANTITATIVE_COMPARISON.md`. Across the
35 corrupted conditions, clean-relative mAP@0.5:0.95 retention averages 0.7638
for Faster R-CNN and 0.7091 for YOLO11s. The sampling procedure, curves, and
interpretation are in `docs/ROBUSTNESS.md`. Phase 7 mean Grad-CAM energy-in-box
is 0.0869 for Faster R-CNN and 0.0975 for YOLO11s, versus box-area references
of 0.0713 and 0.0718. Pointing-game accuracy is only 0.1091 and 0.1261. The
paired maps, definitions, and cautious interpretation are in
`docs/EXPLAINABILITY.md`. Batch 7 must not start until these explainability
results are reviewed.

Review artifacts:

- `results/figures/gradcam_{good_predictions,bad_predictions,failure_cases}.png`
  — paired stride-16 heatmaps with target and candidate boxes;
- `results/tables/gradcam_localization_{summary,per_target}.csv` — aggregate
  and all 222 detector-target explainability records;
- `results/tables/gradcam_qualitative_cases.csv` — the objective case rubric
  and selected evidence;
- `results/logs/phase7_explainability/summary.json` — method, source,
  checkpoint, sample, table, and figure provenance;
- `docs/EXPLAINABILITY.md` — explicit where-is-it-looking answer and caveats;
- `results/figures/robustness_map_50_95_{raw,relative}.png` — per-corruption
  five-severity curves for both detectors;
- `results/tables/robustness_results.csv` — all clean/corrupted raw and
  relative metrics;
- `results/tables/robustness_family_mean_curves.csv` — mean curves within the
  four corruption families;
- `results/logs/phase6_robustness/summary.json` — sample, config, checkpoint,
  bundle, table, and figure provenance;
- `docs/ROBUSTNESS.md` — exact sampling/corruption protocol, results, and
  interpretation;
- `results/tables/detector_comparison.csv` — side-by-side three-seed mean ± SD;
- `results/tables/detector_comparison_per_seed.csv` — all six run-level rows;
- `results/logs/phase5_evaluation/summary.json` — hashes, settings, and full
  evaluation provenance;
- `docs/QUANTITATIVE_COMPARISON.md` — metric definitions, caveats, results,
  interpretation, and exact commands;
- `docs/DATASET_CHOICE.md` — all three candidate inspections and the decision;
- `docs/DATASHEET.md` — composition, patient grouping, preprocessing, terms,
  exclusions, provenance, and biases;
- `data/manifests/rsna-pneumonia-5000-audit.json` — machine-readable full
  metadata audit and selected/split counts;
- `results/figures/rsna_class_distribution.png` — split distribution;
- `results/figures/rsna_annotation_samples.png` — 12 real radiographs with
  labels and opacity boxes; and
- `data/splits/rsna-pneumonia-5000/` — committed patient-safe manifests.

## Setup and verification

The measured detector runs target Python 3.11 on Windows. For a clean local
environment from PowerShell:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv --default-index https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps --editable .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

`requirements.txt` is the authoritative phased-workflow environment. For the
GPU run, the CUDA 12.4 Torch wheels carry their runtime and use the installed
NVIDIA driver; the machine-wide CUDA toolkit is not used. This workstation's
Batch 2 run reuses the verified Anaconda environment without replacing its
working CUDA Torch build:

```powershell
$benchmarkPython = 'C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe'
uv pip install --python $benchmarkPython -r requirements.txt
uv pip install --python $benchmarkPython --no-deps --editable .
& $benchmarkPython -m pip check
```

The tested operating-point and COCO metric modules under
`src/meddet_benchmark/` are now reused by the Faster R-CNN validation adapter;
the legacy experiment schema in that package and `configs/smoke.yaml` remain
non-authoritative.

## Reproduce dataset preparation and EDA

First join the Kaggle competition and accept its rules. Configure either both
`KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`; credentials must never
be placed in this repository.

```powershell
python -m src.data.download --check-credentials
python -m src.data.download --config configs/dataset.yaml
Expand-Archive -LiteralPath data/raw/rsna-pneumonia/stage_2_train_images.zip -DestinationPath data/raw/rsna-pneumonia -Force
Invoke-WebRequest -Uri "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json" -OutFile data/raw/rsna-pneumonia/mappings.json
python -m src.data.prepare --config configs/dataset.yaml --convert-images
python -m src.data.visualize --config configs/dataset.yaml
```

The measured local run used Kaggle's manually downloaded aggregate archive
after the API route was denied. The equivalent extraction command, restricted
to training pixels, is:

```powershell
tar -xf data/raw/rsna-pneumonia/rsna-pneumonia-detection-challenge.zip -C data/raw/rsna-pneumonia stage_2_train_images
```

For a metadata-only audit, COCO conversion, and split regeneration before DICOM
pixels are available:

```powershell
python -m src.data.prepare --config configs/dataset.yaml --metadata-only
```

Preparation verifies the configured official-mapping SHA-256, audits all label
rows, derives true NIH patient groups, rebuilds the fixed subset/manifests, and
writes per-split canonical COCO JSON under the ignored processed-data tree.
Image conversion streams selected DICOMs to PNG without holding the dataset in
RAM. See `data/README.md` for acquisition behavior and the disclosed local
mirror used only to make the review EDA when Kaggle access was unavailable.

The metadata audit is complete for all 26,684 labeled studies. This checkout
contains and has verified the full official training archive, while conversion
materializes only the configured patient-safe 5,000-study subset.

## Faster R-CNN baseline and timing gate

The chosen configuration is fully declared in `configs/faster_rcnn.yaml`:
`fasterrcnn_resnet50_fpn_v2`, COCO transfer weights, 640-pixel short/long edge
for these square radiographs, physical batch size 2, accumulation 2 (effective
optimizer batch 4), float16 AMP, and frozen BatchNorm running statistics.
Accumulation is an optimizer-batch mechanism; it does not combine BatchNorm
statistics between forwards. No stochastic training augmentation is applied.

Check readiness without importing Torch or accessing the test split:

```powershell
python -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode preflight
```

After all 5,000 PNGs exist and the pinned CUDA environment is installed, run a
two-batch GPU/AMP smoke check, then the required three complete train+validation
benchmark epochs:

```powershell
python -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode smoke
python -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode benchmark
```

Benchmark mode writes per-epoch loss, precision, recall, F1, AP50, AP50:95,
timing, learning rate, and peak allocated GPU memory under
`results/logs/faster_rcnn_rsna_seed17_benchmark/`. Its
`benchmark_estimate.json` projects the configured 30-epoch upper bound using
epoch one plus the median steady-state duration from epochs two and three. Each
timed epoch includes equivalent best/last checkpoint writes. The approval
artifact is bound to the YAML, train/validation annotation and image digests,
implementation sources, Torch/torchvision versions, GPU identity, AMP, batch,
and resolution. Stop and obtain the user's approval of that estimate before
continuing.

Only after approval, run the one-seed full baseline from the original COCO
weights (the benchmark weights are deliberately not resumed):

```powershell
python -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json
```

Full mode early-stops on validation AP50:95, keeps the test split untouched,
writes best/last checkpoints, and generates the validation/compute tables and
training-curves figure. See `docs/FASTER_RCNN_BASELINE.md` for metric and
profiling definitions.

If training finishes but final profiling or plotting fails, the run remains in
`trained_pending_finalization` state and can be finalized without retraining:

```powershell
python -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode finalize
```

## YOLO11s baseline

Batch 3 uses the same seed-17 train/validation split, 640-pixel resolution,
effective batch 4, SGD settings, bfloat16 AMP, native YOLO BatchNorm updates, and
validation-mAP early-stopping policy as the Faster R-CNN arm. All additional
Ultralytics stochastic augmentations are explicitly disabled in
`configs/yolo.yaml`; see `docs/YOLO_BASELINE.md` for the controlled-comparison
rationale and the documented constant-LR framework difference.

Materialize the hardlinked YOLO view of the canonical COCO data and verify the
pinned runtime, official pretrained checkpoint, CUDA device, and manifests:

```powershell
python -m src.models.train_yolo --config configs/yolo.yaml --mode prepare
python -m src.models.train_yolo --config configs/yolo.yaml --mode preflight
```

Run the bounded smoke test, three-epoch full-data timing benchmark, and then the
one-seed run. Full training restarts from the original `yolo11s.pt`; it never
continues from timing-benchmark weights.

```powershell
python -m src.models.train_yolo --config configs/yolo.yaml --mode smoke
python -m src.models.train_yolo --config configs/yolo.yaml --mode benchmark
python -m src.models.train_yolo --config configs/yolo.yaml --mode train
```

The train command also evaluates the best checkpoint with the shared COCO and
operating-point evaluator, profiles synchronized batch-1 model-plus-NMS
latency/FPS and model-forward GFLOPs, writes tables and training curves, and
keeps the test split untouched. If training finishes but finalization is
interrupted, regenerate those derived artifacts without retraining:

```powershell
python -m src.models.train_yolo --config configs/yolo.yaml --mode finalize
```

## Unified three-seed comparison

Phase 5 fixes seeds 17, 42, and 137 in `configs/evaluation.yaml`. The additional
configs preserve the accepted detector hyperparameters and change only seed and
artifact identity. Materialize auditable seed-only timing approvals from the
accepted seed-17 measurements, then run the four additional trainings:

```powershell
$benchmarkPython = 'C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe'
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode seed-gates
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed42.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed42_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed137.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed137_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed42.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed137.yaml --mode train
```

Only after all validation-selected checkpoints exist, open the held-out test
split through the one shared evaluator:

```powershell
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode preflight
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode evaluate
```

`src/evaluate.py` converts both frameworks' outputs to the same canonical
records and applies one pycocotools/operating-point metric path. It writes all
six per-seed rows, sample mean ± standard deviation tables, and hashed raw
prediction bundles for the Phase 8 paired statistics. Framework-native mAP is
retained only for validation checkpoint selection. Definitions and caveats are
in `docs/QUANTITATIVE_COMPARISON.md`.

Phase 6 draws the fixed stratified sample, validates the frozen Phase 5
identities, and runs/resumes every corruption condition with:

```powershell
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode preflight
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode run
```

Each condition is saved immediately as a hashed prediction bundle, so an
interrupted grid resumes without recomputing completed inference. The clean
reference is filtered from the frozen Phase 5 seed-17 predictions; all 70
corrupted detector conditions are inferred afresh. Exact severity values and
outputs are defined in `configs/corruptions.yaml` and explained in
`docs/ROBUSTNESS.md`.

## Explainability analysis

Phase 7 verifies the completed Phase 6 identities and reuses its exact
300-image sample. It first validates the paired population and objective
qualitative case rubric, runs focused unit tests and a bounded two-image GPU
smoke, then generates the full quantitative and qualitative artifacts:

```powershell
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode preflight
& $benchmarkPython -m pytest tests/test_gradcam.py tests/test_pointing_game.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode smoke
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode run
```

Both models hook a 40 by 40 stride-16 backbone tensor, target the ground-truth-
associated retained foreground score, and use the same energy-in-box and
pointing-game implementation. False negatives use clearly labeled low-threshold
proxy targets rather than inventing a score for an absent detection. Exact
target semantics, zero-map handling, objective case selection, and results are
in `docs/EXPLAINABILITY.md`.

## Statistical analysis

Phase 8 reads the six frozen Phase 5 bundles and all 72 Phase 6 bundles; it does
not rerun model inference. Preflight verifies upstream completion, configuration
and annotation hashes, and the complete paired grid. Run the exact aggregate-
metric bootstrap and permutation analysis with:

```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode preflight
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run
```

The config fixes 2,000 paired bootstrap draws, 5,000 paired image-label
permutations, 95% percentile intervals, paired jackknife Cohen's d, and Holm
correction. The clean pass resamples the three paired seeds and 750 matched
images. The 35-condition corruption pass reports both raw metrics and clean-
relative retention on the matched 300-image sample. AP is recomputed as a
dataset-level score in every draw rather than averaged from a per-image
surrogate. See `docs/STATISTICAL_ANALYSIS.md` for the protocol, complete clean
table, corruption summary, McNemar non-applicability decision, and limitations.

## Reproducibility contract

Every future experiment entry point must initialize RNGs and write its
environment snapshot before CUDA initialization:

```python
from pathlib import Path

from src.utils.seed import initialize_reproducibility

initialize_reproducibility(seed=17, output_dir=Path("results/logs/example_run"))
```

The run directory receives `pip_freeze.txt` and `run_environment.json`, including
seed, determinism, package, platform, CUDA, GPU, and driver information. See
`docs/REPRODUCIBILITY.md` for worker setup and determinism limitations.

Raw/processed images, credentials, weights, and checkpoints are excluded from
Git. This repository is a research benchmark and does not establish clinical
validity.
