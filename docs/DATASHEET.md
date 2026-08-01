# Datasheet: RSNA Pneumonia Detection Challenge 2018

## Summary and intended use

This project uses the Stage 2 labeled training portion of the RSNA Pneumonia
Detection Challenge 2018 for a controlled comparison of Faster R-CNN and
YOLO11s. The benchmark studies localization accuracy, compute, robustness to
configured digital corruptions, and post-hoc explanation behavior. It is not a
clinical diagnostic dataset validation, and neither the data nor model outputs
may be used to guide patient care.

The canonical task in this repository has **one foreground class**:
`Lung Opacity`. Background is implicit. `Normal` and
`No Lung Opacity / Not Normal` are study-level sampling strata, not detection
categories. The second label means no opacity suspicious for pneumonia; the
image may still contain another abnormality or opacity.

## Collection and annotation

The images are frontal chest radiographs selected from an NIH clinical archive.
The full challenge cohort contained 30,000 exams from 12,274 patients. Eighteen
board-certified radiologists from 16 academic institutions participated in the
annotation effort, including 12 thoracic-imaging specialists. Of those 30,000
cases, 4,527 were triple-read and 1,380 were individually adjudicated. The
annotator network was multi-institutional; the image cohort itself was from one
NIH institution.

Stage 2 exposes 26,684 labeled studies as DICOM images. The label CSV gives
top-left pixel coordinates and width/height for each opacity box. A negative
study has one row with an empty box. The detailed-class CSV supplies the three
study strata used for sampling. The official RSNA mapping links each Kaggle exam
UUID to its original NIH filename.

Primary source documentation is the
[RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
and [resource paper](https://pubs.rsna.org/doi/10.1148/ryai.2019180041).

## Population composition

| Study-level stratum | Studies | Share | Detection boxes |
|---|---:|---:|---:|
| Lung Opacity | 6,012 | 22.53% | 9,555 |
| No Lung Opacity / Not Normal | 11,821 | 44.30% | 0 |
| Normal | 8,851 | 33.17% | 0 |
| **Total** | **26,684** | **100.00%** | **9,555** |

The official mapping recovers 11,452 NIH patient keys in the Stage 2 labeled
set. Of these, 4,355 have multiple exams and the largest group contains 65.
These counts are specific to Stage 2; they should not be confused with the full
30,000-exam cohort counts above.

## Hardware-scoped benchmark subset

`configs/dataset.yaml` fixes a 5,000-study maximum and seed 17. Preparation
orders patient groups deterministically with seeded SHA-256 keys, chooses whole
groups while closely tracking the population study-stratum proportions, and
then performs another grouped stratified allocation into exact 70%/15%/15%
image counts.

| Split | Studies | NIH patient groups | Opacity | No opacity / not normal | Normal | Boxes |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,500 | 1,492 | 798 | 1,554 | 1,148 | 1,267 |
| Validation | 750 | 321 | 169 | 331 | 250 | 277 |
| Test | 750 | 323 | 169 | 331 | 250 | 268 |
| **Selected total** | **5,000** | **2,136** | **1,136** | **2,216** | **1,648** | **1,812** |

The three patient-key intersections are all empty. The Kaggle field called
`patientId` is an exam UUID and is **not** used as the grouping key. The split
uses the NIH patient prefix parsed from the official mapping's original
filename, so all known repeat exams for a patient remain together. Committed
CSV manifests under `data/splits/rsna-pneumonia-5000/` record every assignment.

The other 21,684 valid labeled studies are excluded solely by the predeclared
compute scope. No study was excluded for a metadata audit failure.

## Annotation conversion and image processing

`src/data/prepare.py` converts each split to canonical COCO JSON. Every COCO
image includes its RSNA exam UUID, NIH patient key, study stratum, configured
1024×1024 dimensions, and processed PNG name. Positive boxes are represented as
COCO `[x, y, width, height]` annotations with category ID 1; negative images are
retained with zero annotations. Both detector loaders will consume these same
COCO records in later batches.

Available DICOMs are converted lazily rather than loaded as one in-memory
array. Conversion reads the 2-D pixel array, inverts `MONOCHROME1` images,
applies per-image finite min–max scaling, and writes 8-bit grayscale PNG. This
fixed conversion does not apply a vendor-specific DICOM display window or VOI
LUT; that simplification can change displayed contrast and is a limitation to
carry into interpretation.

## Integrity checks

The complete metadata audit found:

- 30,227 label rows and 30,227 detailed-class rows covering 26,684 studies;
- 26,684 valid and zero invalid study records;
- 9,555 valid positive boxes;
- zero missing/non-numeric, non-positive-area, or off-image boxes;
- zero exact duplicate boxes;
- zero target/class, class-info, or patient-mapping inconsistencies; and
- zero patient-key overlap across train, validation, and test.

Blank coordinates on the 20,672 negative label rows are expected. These checks
establish metadata consistency within the documented 1024×1024 frame; they do
not establish that every box is clinically correct.

For the review EDA, 12 authentic selected DICOMs—four from each study
stratum—were decoded successfully at 1024×1024, with no conversion errors.
Their opacity boxes visually fall on plausible pulmonary findings. The local
checkout does not yet cache the other 4,988 selected DICOMs, so full pixel-file
integrity must be rerun after the authorized Kaggle download and before model
training. The metadata audit covers the entire labeled population.

The committed machine-readable audit is
`data/manifests/rsna-pneumonia-5000-audit.json`. Its locally inspected source
digests are:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `stage_2_train_labels.csv` | 1,490,034 | `bb40b7e956e9922a6b275ed4a158197568cf9ab618017d53db6159b1b624bb65` |
| `stage_2_detailed_class_info.csv` | 1,647,396 | `c004c12dea2042cc23e3b848f65e8cb2e725799afaa90f13ee81f854bcc9614d` |
| official `mappings.json` | 17,587,143 | `803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27` |

The mapping digest is pinned in config and must match before preparation writes
outputs.

## Acquisition provenance

The authoritative acquisition route is the Kaggle competition downloader in
`src/data/download.py`, after the user accepts the competition rules. This local
environment had no Kaggle credentials, and the signed Google Storage route was
unavailable from its region. To complete the requested metadata review, the two
canonical-named Stage 2 CSVs and 12 review DICOMs were obtained from the public
Hugging Face mirror `Baldezo313/rsna-pneumonia-dataset`; the patient mapping came
directly from official RSNA storage. The exact local file hashes above make that
interim input state auditable. A full authorized Kaggle acquisition is required
before training.

Raw DICOMs, extracted PNGs, credentials, and generated COCO files are excluded
from Git. They can be regenerated with the commands in `README.md`.

## License and access conditions

The dataset uses bespoke
[RSNA challenge terms](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf),
not a named open-source license. The terms allow academic research, education,
and other commercial or non-commercial purposes; prohibit re-identification;
and specify NIH/RSNA attribution when the data are shared or redistributed.
Users must read and follow the terms themselves. This repository redistributes
no patient images.

## Known biases and risks

- The single-institution, historical NIH imaging cohort may encode site,
  equipment, workflow, demographic, and acquisition-position effects.
- Challenge selection was enriched with pre-existing labels and is not a
  prevalence-representative sample. Deployment precision cannot be inferred
  from this class balance.
- `Lung Opacity` is a non-specific radiographic observation, not a confirmed
  pneumonia diagnosis. Boxes are coarse rectangles and reader certainty varies.
- Patient grouping controls known repeated exams but cannot remove every hidden
  correlation between encounters, devices, or acquisition sessions.
- The fixed 5,000-study subset increases sampling uncertainty and limits claims
  about the full challenge population.
- The 12-image visual review is illustrative, not expert re-annotation or a
  substitute for the full post-download pixel audit.

These limitations are also tracked in `docs/LIMITATIONS.md` and must accompany
all later benchmark conclusions.
