# Medical Object Detector Benchmark

Controlled comparison of Faster R-CNN and YOLO11s on medical object detection,
covering predictive performance, compute, common-corruption robustness,
explainability, and statistical uncertainty. The authoritative requirements and
review checkpoints are in `PROJECT_SPEC.md` and `BATCHES.md`.

## Current status

Batch 1 Phases 1–2 are complete and stopped at the required dataset/EDA review
gate. The selected dataset is the RSNA Pneumonia Detection Challenge 2018 Stage
2 set, with one actual foreground class: `Lung Opacity`. A deterministic
patient-grouped 5,000-study subset is split 3,500/750/750 with no NIH patient-key
overlap. No model training has started.

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

The project targets Python 3.13 on Windows. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check src tests
```

`requirements.txt` is the authoritative phased-workflow environment. The
pre-existing `pyproject.toml`, `uv.lock`, and `src/meddet_benchmark/` remain a
legacy implementation and are not used by this batch.

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
decoded 12 review DICOMs; the full authorized Kaggle image download and a rerun
of conversion are required before training.

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
