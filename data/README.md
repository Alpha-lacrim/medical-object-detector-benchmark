# Dataset workspace

Raw data is deliberately excluded from Git. This directory keeps only download
instructions and future versioned manifests.

## Selected source

Provisional source:
[Medical Image DataSet: Brain Tumor Detection](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection).

Public Kaggle API metadata retrieved on 2026-07-28 reported:

- dataset version `5`;
- last update `2025-02-10T21:20:45.74Z`;
- archive size `313,038,935` bytes;
- license `CC BY 4.0`; and
- public visibility.

These are source metadata, not verified image or annotation counts. The archive
and extracted files must still be hashed and audited locally.

## Download

Authenticate with Kaggle outside this repository. Never place a token or
`kaggle.json` under the project directory. With the official `kagglehub`
package, download the exact version into the ignored raw-data directory:

```powershell
uv run --with kagglehub python -c "import kagglehub; print(kagglehub.dataset_download('pkdarabi/medical-image-dataset-brain-tumor-detection/versions/5', output_dir='data/raw/brain-tumor-v5'))"
```

Alternatively, download version 5 in a browser and extract it under
`data/raw/brain-tumor-v5/`. Preserve the original archive until its SHA-256 is
recorded.

The unauthenticated direct API probe returned HTTP 403 on 2026-07-28, so no
dataset is currently present.

## Audit

Read the extracted `data.yaml` first and pass its actual ordered class names;
do not infer a fifth class or manufacture a `no tumor` box:

```powershell
uv run --locked python -m meddet_benchmark audit-data data/raw/brain-tumor-v5/EXTRACTED_ROOT --class-name ACTUAL_CLASS_0 --class-name ACTUAL_CLASS_1
```

The command emits deterministic JSON containing:

- image and box counts by split and class;
- missing, empty, invalid, duplicate, and orphan annotations;
- image readability and dimensions;
- image and label SHA-256 values;
- exact within-split and cross-split duplicate groups; and
- a canonical manifest fingerprint.

The default expected layout is `train/`, `valid/`, and `test/`, each containing
`images/` and `labels/`. Near-duplicate and patient-level leakage checks remain
required after the exact-hash audit.
