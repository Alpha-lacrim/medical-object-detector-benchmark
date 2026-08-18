# Common-Corruption Robustness

## Scope and frozen inputs

Phase 6 evaluates the primary seed-17 Faster R-CNN and YOLO11s checkpoints on
one fixed subset of the held-out test split. It does not average robustness over
the three training seeds. The clean reference predictions are filtered from the
frozen Phase 5 seed-17 prediction bundles because the checkpoint, images,
thresholds, NMS, and unified evaluator are identical. Every corrupted condition
is inferred afresh and scored through the same operating-point matcher and
pycocotools path used in Phase 5.

The main robustness measure is mAP@0.5:0.95. For every reported metric,
`relative performance = corrupted performance / clean performance`; the table
also stores `relative degradation = 1 - relative performance`. A retention
above 1, and therefore a negative degradation, is possible when a mild
transformation happens to improve a finite-sample score. It is not clipped.

## Stratified test sample

The source test manifest has 750 images distributed as 169 Lung Opacity, 331 No
Lung Opacity / Not Normal, and 250 Normal. With seed 17, the sampler:

1. computes proportional quotas and assigns the one remainder by the largest
   fractional remainder;
2. sorts strata and `processed_file` identifiers lexicographically;
3. draws without replacement sequentially from one NumPy PCG64 generator; and
4. sorts the selected identifiers before writing the manifest.

This gives exactly 300 images: 68 Lung Opacity, 132 No Lung Opacity / Not
Normal, and 100 Normal. The sample has 68 positive and 232 negative images, 111
boxes, and 183 distinct NIH patient identifiers. Its committed manifest is
`data/splits/rsna-pneumonia-5000/test_robustness_seed17_n300.csv` (SHA-256
`63b4dd706dc2fcd8a528a935957ccb318ed2cde51a6fd87d20feca348d00fc5e`).
Sampling uses only the fixed study stratum, never detector performance.

## Corruption grid

Albumentations 2.0.8 applies geometry-preserving uint8 RGB transforms; boxes
remain unchanged. Stochastic noise and motion direction use a deterministic
seed derived from the experiment seed, image filename, and condition ID. Thus
both detectors receive the same corrupted image instance independently of
evaluation order.

| Corruption type | Family | Severity 1 -> 5 |
|---|---|---|
| Darker | Lighting | intensity multipliers 0.90, 0.80, 0.70, 0.60, 0.50 |
| Brighter | Lighting | intensity multipliers 1.10, 1.20, 1.30, 1.40, 1.50 |
| Gaussian noise | Noise | uint8-range standard deviations 0.010, 0.020, 0.035, 0.050, 0.075 |
| Salt and pepper | Noise | affected-pixel fractions 0.0025, 0.005, 0.010, 0.020, 0.040 |
| Gaussian blur | Blur | sigma 0.5, 1.0, 1.5, 2.0, 3.0 pixels |
| Motion blur | Blur | centered kernels 3, 5, 9, 13, 17 pixels |
| JPEG | Compression | quality 90, 70, 50, 35, 20 percent |

This is an ImageNet-C-style ordered stress-test design, not a claim that the
severity levels reproduce equal or clinically calibrated physical changes
[@hendrycks2019corruptions].

## Results

The 300-image clean subset mAP@0.5:0.95 is 0.147802 for Faster R-CNN and
0.076295 for YOLO11s. Averaged equally across all 35 corrupted conditions,
Faster R-CNN records raw mAP@0.5:0.95 of 0.112898 and clean-relative retention
of 0.763846 (23.62% mean degradation). YOLO11s records 0.054099 and 0.709083
(29.09% mean degradation). Faster R-CNN has higher raw mAP@0.5:0.95 in every
clean/corrupted matched condition; this is conditional on the primary
checkpoints and this fixed sample.

Severity-5 mAP@0.5:0.95 retention is:

| Corruption | Faster R-CNN | YOLO11s |
|---|---:|---:|
| Darker | 0.8674 | 0.1645 |
| Brighter | 0.6706 | 0.6882 |
| Gaussian noise | 0.4643 | 0.3692 |
| Salt and pepper | 0.2025 | 0.0570 |
| Gaussian blur | 0.6936 | 0.6949 |
| Motion blur | 0.6788 | 0.5780 |
| JPEG quality 20 | 0.6825 | 0.7662 |

Impulse noise is the most damaging tested corruption for both detectors. The
YOLO11s darkest condition emits no true positives at the 0.25 operating
threshold, so its conditional matched-box IoU/Dice and their ratios are
undefined there; COCO mAP remains defined because it uses predictions retained
down to the shared 0.001 minimum. Mild darkening raises Faster R-CNN mAP by
about 1.5--1.7%, illustrating why the measured curves are reported rather than
forced to be monotone.

Review artifacts:

- `results/figures/robustness_map_50_95_raw.png` and
  `results/figures/robustness_map_50_95_relative.png`: per-type raw and
  clean-relative curves;
- `results/tables/robustness_results.csv`: all 72 clean/corrupted detector rows
  and wide raw/relative metrics;
- `results/tables/robustness_curves.csv`: tidy per-type curves for all seven
  predictive metrics;
- `results/tables/robustness_family_mean_curves.csv`: severity curves averaged
  within lighting, noise, blur, and compression families; and
- `results/logs/phase6_robustness/summary.json` plus 72 hashed prediction
  bundles: complete provenance and the per-image sufficient evidence used by
  Phase 8's patient-cluster paired tests in `docs/STATISTICAL_ANALYSIS.md`.

## Interpretation limits

The benchmark tests deterministic digital transformations after the project's
PNG conversion. Brightness and JPEG changes do not simulate scanner physics,
DICOM window/VOI processing, reconstruction, population shift, or a new site.
The 300-image sample contains only 111 boxes and repeated exams from some
patients. Batch 13's primary inference keeps all exams from each patient
together during resampling and permutation, but the 183 patient groups and one
checkpoint per detector still make relative rankings sample-sensitive. These
results support a scoped digital common-corruption comparison, not clinical
robustness or safety.
