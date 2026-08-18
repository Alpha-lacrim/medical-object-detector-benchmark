# Paired statistical analysis

All unqualified inferential results below are the Batch 13 patient-cluster
results. The former image-level analysis is superseded and retained only in
explicitly named archive artifacts for audit.

## Questions and estimands

Phase 8 uses only the frozen prediction evidence from Phases 5 and 6; it does
not run either detector again. Faster R-CNN is detector A, YOLO11s is detector
B, and every reported difference is `A - B`.

The clean comparison covers precision, recall, F1, conditional matched-box IoU
and Dice, mAP@0.5, and mAP@0.5:0.95 on all 750 test images. Each statistic is
first computed for each paired training seed (17, 42, and 137), then averaged
over seeds. The corruption analysis uses the primary seed-17 checkpoints on the
fixed 300-image sample. For each of 35 conditions it tests both:

- raw performance, `metric(A, corruption) - metric(B, corruption)`; and
- clean-relative retention,
  `metric(A, corruption) / metric(A, clean) - metric(B, corruption) /
  metric(B, clean)`.

Retention is a ratio rather than a percentage-point degradation. It can exceed
one when a corruption improves a finite-sample estimate. Conditional IoU and
Dice remain undefined when a detector has no matched true positives.

## Inference protocol

All resampling parameters are fixed in `configs/statistics.yaml`.

### Bootstrap confidence intervals

The tables report two-sided pointwise 95% percentile intervals from 2,000
paired bootstrap draws. The sampling unit is the NIH patient group recorded in
the committed Batch 1 test manifest (`nih_patient_id` in
`data/splits/rsna-pneumonia-5000/test.csv`). A draw samples patient groups with
replacement; each sampled group contributes every observed image belonging to
that patient with one shared multiplicity. The 750-image clean set contains 323
patient groups, and the frozen 300-image robustness subset contains 183. The
clean analysis also resamples the three paired seed indices, producing a
two-stage empirical patient-group-and-seed bootstrap. For retention, the same
patient-group draw is applied jointly to the clean and corrupted evidence
before either ratio is calculated.

The aggregate metric is recomputed in every draw. Precision, recall, F1, IoU,
and Dice are rebuilt from image-level TP/FP/FN and localization sums. AP is
rebuilt from score-ordered per-image matches at all ten COCO IoU thresholds and
the 101 recall thresholds. Thus the analysis does not substitute a mean of
ill-defined “per-image AP” values for dataset-level mAP. The fast reconstruction
was checked against the frozen pycocotools and operating-point results for all
78 Phase 5/6 bundles to an absolute tolerance of `5e-12`.

### Paired permutation test and effect size

The two-sided paired permutation test uses 5,000 Monte Carlo detector-label
swaps plus the standard plus-one correction. A detector label is swapped by
patient group, so every image belonging to one patient moves together. The
group swap is consistent across all three seeds in the clean analysis, and
retention uses the same group swap for clean and corrupted evidence. The
p-values therefore test detector exchangeability conditional on the observed
patient clusters and trained checkpoints; the hierarchical confidence
intervals additionally show the coarse empirical variation from resampling the
three available seed pairs.

The effect size is the unstandardized paired aggregate difference (`A - B`),
reported with its patient-cluster bootstrap interval. The former image-level
jackknife Cohen's d is retained only in the archived tables. It is not carried
forward because a standardized pseudovalue interpretation is not clearly
established for these nonlinear pooled metrics with unequal patient-cluster
sizes. This avoids presenting a guessed cluster-level Cohen's d; it does not
redefine mAP as a per-image or per-patient mean.

Holm's step-down correction is applied to the seven clean predictive endpoints.
For the corruption grid, it is applied separately within each metric and
estimand across the 35 conditions. The darkest-condition IoU and Dice families
contain 34 tests because one comparison is not estimable. Confidence intervals
are pointwise, not multiplicity-adjusted, so family-wise claims use the Holm
p-value rather than CI exclusion alone.

McNemar's test is not applied. This benchmark has multiple possible targets per
image plus negative-image false positives, not one independent binary outcome
per image. Collapsing it to “correct/incorrect” would discard detection outcome
structure, while target-level success indicators would be nested within images
and repeated across seeds/conditions.

## Clean three-seed results

Values are estimate `[pointwise 95% CI]`. The comparison interval is also the
unstandardized effect-size interval; raw and Holm p-values accompany every
endpoint.

| Metric | Faster R-CNN | YOLO11s | Difference/effect (A - B) | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| Precision | 0.1626 [0.1161, 0.2197] | 0.3730 [0.2552, 0.4754] | -0.2105 [-0.3027, -0.1047] | 0.0002 | 0.0014 |
| Recall | 0.6381 [0.5576, 0.7067] | 0.1356 [0.0708, 0.1937] | 0.5025 [0.4311, 0.5712] | 0.0002 | 0.0014 |
| F1 | 0.2558 [0.1955, 0.3221] | 0.1981 [0.1125, 0.2674] | 0.0577 [-0.0116, 0.1448] | 0.0554 | 0.0972 |
| Conditional IoU | 0.6732 [0.6545, 0.6899] | 0.6971 [0.6663, 0.7270] | -0.0239 [-0.0560, 0.0038] | 0.0334 | 0.0972 |
| Conditional Dice | 0.7997 [0.7861, 0.8117] | 0.8172 [0.7950, 0.8389] | -0.0174 [-0.0408, 0.0024] | 0.0324 | 0.0972 |
| mAP@0.5 | 0.3084 [0.2407, 0.3795] | 0.1643 [0.1056, 0.2322] | 0.1441 [0.0947, 0.1914] | 0.0034 | 0.0170 |
| mAP@0.5:0.95 | 0.1023 [0.0756, 0.1342] | 0.0549 [0.0324, 0.0833] | 0.0474 [0.0300, 0.0649] | 0.0082 | 0.0328 |

After correction, Faster R-CNN retains evidence for higher recall and both AP
metrics, while YOLO retains evidence for higher precision. F1 and the
conditional localization metrics do not cross the 0.05 Holm threshold. The
localization comparison is conditional on a true-positive match and does not
counterbalance YOLO's much lower recall.

Precision, recall, and F1 in this analysis use the original Phase 5 score
threshold of 0.25. The significant YOLO precision difference is therefore a
fixed-threshold result. The complete official precision-recall curves show that
it is a score-scale/selectivity artifact rather than evidence of a superior
YOLO11s frontier; validation-selected final operating points are reported in
`THRESHOLD_ANALYSIS.md`.

This clean significance pattern is unchanged from the archived image-level
analysis: the same four of seven endpoints are Holm-significant, and no clean
endpoint gains or loses significance.

## Corruption-grid results

The full machine-readable table contains 497 comparison rows: seven raw clean
references, 245 raw corrupted comparisons, and 245 retention comparisons. Four
darkest-condition localization rows (raw and retention IoU/Dice) are explicitly
not estimable because YOLO has no operating-point true positive. Every other row
contains both model intervals, a difference interval, raw and Holm p-values,
and the paired raw-difference effect.

Faster R-CNN has higher point-estimate raw mAP@0.5:0.95 in all 35 corrupted
conditions, but no raw AP comparison now survives the 35-condition Holm family.
At darkness severity 5, raw mAP@0.5:0.95 is 0.1282 `[0.0762, 0.1914]` for
Faster R-CNN and 0.0126 `[0.0038, 0.0365]` for YOLO, a difference of 0.1156
`[0.0609, 0.1689]` (raw `p=0.0022`, Holm `p=0.0770`). Raw mAP@0.5 is 0.3156
`[0.1880, 0.4526]` versus 0.0439 `[0.0127, 0.1142]`, a difference of 0.2717
`[0.1464, 0.3748]` (raw `p=0.0050`, Holm `p=0.1750`). Their pointwise
intervals exclude zero, but those intervals are not multiplicity-adjusted; the
Holm tests determine the grid-wide conclusion.

The strongest relative-robustness result remains the darkest condition. Faster
R-CNN retains 0.8674 of clean mAP@0.5:0.95 and YOLO retains 0.1645; the
retention difference is 0.7029 `[0.2734, 0.8158]` (raw `p=0.0002`, Holm
`p=0.0070`). For mAP@0.5, retention is 0.8355 versus 0.2076 and the difference
is 0.6279 `[0.2313, 0.7759]` (raw `p=0.0002`, Holm `p=0.0070`). Darkness
severity 2 mAP@0.5 retention also becomes significant: difference 0.1797
`[0.0005, 0.3135]` (raw `p=0.0014`, Holm `p=0.0476`).

Several operating-point retention comparisons survive within their own
metric families, chiefly severe darkness and selected noise conditions. Mild
motion blur favors YOLO retention for recall and F1, and JPEG severity 1 favors
YOLO recall retention. These are endpoint-specific results, not evidence that
YOLO has higher raw AP: the exact rows and all adjusted p-values are in
`results/tables/statistical_robustness_comparison.csv`.

### Significance changes from patient clustering

The clean family does not change, but the secondary robustness pattern does.
Across the complete 497-row robustness table, 88 rows were Holm-significant in
the archived image-level analysis and 87 are significant with patient-cluster
exchange. The net count hides 11 changed decisions:

| Change | Condition and estimand | Metric | Image-level Holm p | Patient-cluster Holm p |
|---|---|---:|---:|---:|
| Lost significance | Darkness severity 5, raw | mAP@0.5 | 0.0070 | 0.1750 |
| Lost significance | Darkness severity 5, raw | mAP@0.5:0.95 | 0.0070 | 0.0770 |
| Lost significance | Salt-and-pepper severity 1, retention | F1 | 0.0136 | 0.2356 |
| Lost significance | Salt-and-pepper severity 2, retention | F1 | 0.0248 | 0.0576 |
| Lost significance | Salt-and-pepper severity 2, retention | Precision | 0.0272 | 0.0952 |
| Lost significance | Salt-and-pepper severity 3, raw | Precision | 0.0344 | 0.1212 |
| Gained significance | Darkness severity 2, retention | Dice | 0.5099 | 0.0136 |
| Gained significance | Darkness severity 2, retention | IoU | 0.5207 | 0.0136 |
| Gained significance | Darkness severity 2, retention | mAP@0.5 | 0.1020 | 0.0476 |
| Gained significance | Darkness severity 3, retention | Dice | 0.4685 | 0.0264 |
| Gained significance | Darkness severity 3, retention | IoU | 0.4355 | 0.0136 |

Thus patient clustering changes secondary corruption conclusions in both
directions. Most importantly, the prior claim that darkness severity 5 had a
Holm-significant *raw* AP advantage no longer holds. Its AP retention advantage
does remain significant. The newly significant darkness localization-retention
rows are conditional on true-positive matches and do not overturn the raw
coverage results.

## Interpretation limits

The patient-cluster bootstrap and permutation now keep every observed exam from
one NIH patient together, correcting the former within-patient independence
error. This is still a nonparametric analysis of 323 observed clean patient
groups and 183 robustness-sample groups, not a guarantee of transportability to
a new clinical population. Three training seeds only coarsely represent
training variability, and corruption inference remains primary-seed-only. The
corruption conditions are repeated digital transformations of the same images,
not independent deployment cohorts. Holm controls each declared family of
p-values but does not make the pointwise intervals simultaneous or make the
corruption conditions independent cohorts.

## Reproduction and artifacts

Run the CPU statistical pass after Phases 5 and 6 are complete:

```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode preflight
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run
```

The outputs are `results/tables/statistical_clean_comparison.csv`,
`results/tables/statistical_robustness_comparison.csv`, and
`results/logs/phase8_statistics/summary.json`. The summary records input and
source hashes, patient-group counts, methods, all result rows, artifact hashes,
and the exact before/after Holm-decision comparison. The superseded image-level
outputs remain available for audit at
`results/tables/statistical_clean_comparison_image_level_archive.csv`,
`results/tables/statistical_robustness_comparison_image_level_archive.csv`, and
`results/logs/phase8_statistics/summary_image_level_archive.json`; they are not
primary results because they treat correlated exams as independent units.
