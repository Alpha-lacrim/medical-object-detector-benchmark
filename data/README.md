# Dataset workspace

Raw images, credentials, and processed pixels are excluded from Git. Only this
guide and deterministic split manifests are versioned.

## Selected source

The selected source is the
[RSNA Pneumonia Detection Challenge 2018](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018),
Stage 2 training data, hosted as the Kaggle competition
`rsna-pneumonia-detection-challenge`. Its one foreground detection class is
`Lung Opacity`; background images include both normal studies and abnormal
studies without opacity.

The experiment uses a deterministic, patient-grouped 5,000-study subset. See
`docs/DATASET_CHOICE.md` and `docs/DATASHEET.md` for the selection rationale,
composition, terms, and leakage controls.

## Credentials and competition access

Join the Kaggle competition and accept its rules first. Supply credentials by
either:

- setting both `KAGGLE_USERNAME` and `KAGGLE_KEY`; or
- placing `kaggle.json` in `~/.kaggle/` (or under `KAGGLE_CONFIG_DIR`).

Never put `kaggle.json` inside this repository. Check discovery without making
a network request:

```powershell
python -m src.data.download --check-credentials
```

If credentials are absent, partial, or malformed, the command exits with an
actionable message and never prints a secret.

## Acquire and prepare

From the repository root:

```powershell
python -m src.data.download --config configs/dataset.yaml
Expand-Archive -LiteralPath data/raw/rsna-pneumonia/stage_2_train_images.zip -DestinationPath data/raw/rsna-pneumonia -Force
Invoke-WebRequest -Uri "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json" -OutFile data/raw/rsna-pneumonia/mappings.json
python -m src.data.prepare --config configs/dataset.yaml
python -m src.data.visualize --config configs/dataset.yaml
```

The expected SHA-256 of the official mapping is configured in
`configs/dataset.yaml`; preparation refuses a mismatched mapping. Kaggle may
report either unaccepted competition rules or a storage-region restriction.
The downloader explains those cases separately rather than treating them as
missing credentials.

Preparation writes canonical COCO JSON, a machine-readable annotation audit,
and CSV split manifests. DICOM-to-PNG conversion is streaming and does not load
the full dataset into RAM. The committed manifests and figures can be reviewed
without committing the raw/processed image files.

## Locally inspected source files

The 2026-08-02 audit used the canonical Stage 2 CSV filenames and the official
RSNA mapping. Re-running preparation records the exact input hashes in its
audit output. The locally inspected mapping hash was:

```text
803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27  mappings.json
```

The local environment had no Kaggle credentials and its signed Google Storage
route was region-blocked. For the review-only EDA, a public mirror supplied the
canonical-named label CSVs and a small set of DICOMs; their hashes and this
provenance are disclosed in `docs/DATASHEET.md`. A full authorized Kaggle
download remains the required source before detector training.
