# Medical Object Detector Benchmark

Controlled comparison of Faster R-CNN and YOLO11s on medical object detection,
covering predictive performance, compute, common-corruption robustness,
explainability, and statistical uncertainty. The authoritative requirements and
review checkpoints are in `PROJECT_SPEC.md` and `BATCHES.md`.

## Current status

Batch 1 Phases 1–2 are complete and the Faster R-CNN Batch 2 pipeline is
implemented. The selected dataset is the RSNA Pneumonia Detection Challenge
2018 Stage 2 set, with one actual foreground class: `Lung Opacity`. A
deterministic patient-grouped 5,000-study subset is split 3,500/750/750 with no
NIH patient-key overlap. No model training has started.

The official Kaggle "download all" archive has been CRC-checked and contains
all 26,684 Stage 2 training DICOMs. Its two metadata CSVs match the committed
audit hashes. All 5,000 fixed manifest studies were freshly converted from the
official DICOMs with zero missing sources or decode errors. The real-data gate
is ready for tests, the bounded smoke check, and the required three-epoch timing
benchmark.

YOLO11s is selected for the later one-stage arm and the implementation is pinned
to `ultralytics==8.4.110`. The detector decision and literature basis are in
`docs/LITERATURE_REVIEW.md`.

Review artifacts:

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
