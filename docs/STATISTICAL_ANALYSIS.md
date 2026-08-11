# Paired statistical analysis

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
paired bootstrap draws. A draw resamples image indices and applies the same
multiplicity to both detectors. The clean analysis also resamples the three
paired seed indices, producing a two-stage empirical image-and-seed bootstrap.
For retention, the same image draw is applied jointly to the clean and
corrupted evidence before either ratio is calculated.

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
image, consistently across all three seeds in the clean analysis. Retention
uses the same swap for an image's clean and corrupted evidence. The p-values
therefore test detector exchangeability conditional on the observed images and
trained checkpoints; the hierarchical confidence intervals additionally show
the coarse empirical variation from resampling the three available seed pairs.

The effect size is paired jackknife Cohen's d. For each aggregate metric, the
analysis removes one matched image at a time, forms the jackknife pseudovalue of
the Faster-minus-YOLO difference, and divides the mean pseudovalue by its sample
standard deviation. This supplies an image-level standardized effect without
redefining mAP as a per-image mean. Its sign follows the table difference.

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

Values are estimate `[pointwise 95% CI]`. The comparison interval, raw p-value,
Holm p-value, and standardized effect accompany every endpoint.

| Metric | Faster R-CNN | YOLO11s | Difference (A - B) | Raw p | Holm p | d |
|---|---:|---:|---:|---:|---:|---:|
| Precision | 0.1626 [0.1236, 0.2150] | 0.3730 [0.2802, 0.4737] | -0.2105 [-0.3050, -0.1177] | 0.0002 | 0.0014 | -0.182 |
| Recall | 0.6381 [0.5582, 0.7119] | 0.1356 [0.0956, 0.1802] | 0.5025 [0.4301, 0.5760] | 0.0002 | 0.0014 | 0.624 |
| F1 | 0.2558 [0.2064, 0.3173] | 0.1981 [0.1437, 0.2536] | 0.0577 [-0.0130, 0.1355] | 0.0250 | 0.0510 | 0.076 |
| Conditional IoU | 0.6732 [0.6567, 0.6906] | 0.6971 [0.6637, 0.7307] | -0.0239 [-0.0530, 0.0047] | 0.0180 | 0.0510 | -0.075 |
| Conditional Dice | 0.7997 [0.7875, 0.8122] | 0.8172 [0.7933, 0.8405] | -0.0174 [-0.0377, 0.0026] | 0.0170 | 0.0510 | -0.076 |
| mAP@0.5 | 0.3084 [0.2519, 0.3771] | 0.1643 [0.1180, 0.2221] | 0.1441 [0.0967, 0.1949] | 0.0002 | 0.0014 | 0.095 |
| mAP@0.5:0.95 | 0.1023 [0.0798, 0.1343] | 0.0549 [0.0374, 0.0800] | 0.0474 [0.0311, 0.0683] | 0.0004 | 0.0016 | 0.236 |

After correction, Faster R-CNN retains evidence for higher recall and both AP
metrics, while YOLO retains evidence for higher precision. F1 and the
conditional localization metrics do not cross the 0.05 Holm threshold. The
localization comparison is conditional on a true-positive match and does not
counterbalance YOLO's much lower recall.

## Corruption-grid results

The full machine-readable table contains 497 comparison rows: seven raw clean
references, 245 raw corrupted comparisons, and 245 retention comparisons. Four
darkest-condition localization rows (raw and retention IoU/Dice) are explicitly
not estimable because YOLO has no operating-point true positive. Every other row
contains both model intervals, a difference interval, raw and Holm p-values,
and Cohen's d.

Faster R-CNN has higher point-estimate raw mAP@0.5:0.95 in all 35 corrupted
conditions, but the grid-wide correction matters: only darkness severity 5
survives Holm correction for that endpoint. There, raw mAP@0.5:0.95 is 0.1282
for Faster R-CNN and 0.0126 for YOLO, a difference of 0.1156 `[0.0712,
0.1702]` (raw `p=0.0002`, Holm `p=0.0070`, `d=0.241`). Raw mAP@0.5 shows the
same condition-level conclusion: difference 0.2717 `[0.1842, 0.3651]`, raw
`p=0.0002`, Holm `p=0.0070`, `d=0.295`.

The primary relative-robustness result is also the darkest condition. Faster
R-CNN retains 0.8674 of clean mAP@0.5:0.95 and YOLO retains 0.1645; the
retention difference is 0.7029 `[0.3982, 0.8424]` (raw `p=0.0002`, Holm
`p=0.0070`, `d=0.433`). For mAP@0.5, retention is 0.8355 versus 0.2076 and the
difference is 0.6279 `[0.3564, 0.7834]` (raw `p=0.0002`, Holm `p=0.0070`,
`d=0.286`). No other AP retention comparison survives grid-wide correction.

Several operating-point retention comparisons survive within their own
metric families, chiefly severe darkness and selected noise conditions. Mild
motion blur favors YOLO retention for recall and F1, and JPEG severity 1 favors
YOLO recall retention. These are endpoint-specific results, not evidence that
YOLO has higher raw AP: the exact rows and all adjusted p-values are in
`results/tables/statistical_robustness_comparison.csv`.

## Interpretation limits

The bootstrap is explicitly per image as required by the analysis plan, but the
750 test images and 300-image robustness sample contain repeated exams from
some NIH patients. The intervals therefore do not fully account for within-
patient dependence. Three training seeds only coarsely represent training
variability, and corruption inference remains primary-seed-only. The
corruption conditions are repeated digital transformations of the same images,
not independent deployment cohorts. Holm controls each declared family of
p-values but does not make the pointwise intervals simultaneous or remove these
sampling limitations.

## Reproduction and artifacts

Run the CPU statistical pass after Phases 5 and 6 are complete:

```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode preflight
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run
```

The outputs are `results/tables/statistical_clean_comparison.csv`,
`results/tables/statistical_robustness_comparison.csv`, and
`results/logs/phase8_statistics/summary.json`. The summary records input and
source hashes, methods, all result rows, and artifact hashes.
