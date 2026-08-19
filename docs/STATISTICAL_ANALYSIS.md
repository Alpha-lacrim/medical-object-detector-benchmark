# Paired statistical analysis

All unqualified inferential results below use the patient-cluster methodology
corrected in Batch 13; the clean results were refreshed in Batch 16. The former
image-level analysis is superseded and retained only in explicitly named
archive artifacts for audit.

## Questions and estimands

Phase 8 uses only the frozen prediction evidence from Phases 5 and 6; it does
not run either detector again. Faster R-CNN is detector A, YOLO11s is detector
B, and every reported difference is `A - B`.

The clean comparison covers precision, recall, F1, conditional matched-box IoU
and Dice, mAP@0.5, and mAP@0.5:0.95 on all 750 test images. Each statistic is
first computed within an eligible paired training seed, then averaged over
seeds. Precision, recall, F1, and both AP endpoints use all five predeclared
pairs (17, 42, 137, 271, and 314). Conditional IoU and Dice use the four
complete pairs 17, 42, 137, and 314 because YOLO11s seed 271 has no
score-0.25 true positive and its matched-only localization is undefined. The
corruption analysis uses the primary seed-17 checkpoints on the fixed 300-image
sample. For each of 35 conditions it tests both:

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
clean analysis also resamples the eligible paired seed indices, producing a
two-stage empirical patient-group-and-seed bootstrap: five pairs for the
unconditional endpoints and four complete pairs for conditional IoU/Dice. For
retention, the same patient-group draw is applied jointly to the clean and
corrupted evidence before either ratio is calculated. The additional
endpoint-specific seed filter changes only cross-seed eligibility; the patient
IDs, cluster construction, group multiplicities, and resampling mechanics from
Batch 13 are unchanged. The five-pair and four-pair endpoint groups use
separate deterministic comparison labels, so their realized bootstrap draws
and label-swap sequences need not be byte-identical; common random draws are
not required for endpoint-specific intervals or the final Holm correction.

The aggregate metric is recomputed in every draw. Precision, recall, F1, IoU,
and Dice are rebuilt from image-level TP/FP/FN and localization sums. AP is
rebuilt from score-ordered per-image matches at all ten COCO IoU thresholds and
the 101 recall thresholds. Thus the analysis does not substitute a mean of
ill-defined “per-image AP” values for dataset-level mAP. The fast reconstruction
was checked against the frozen pycocotools and operating-point results for all
82 Phase 5/6 bundles to an absolute tolerance of `5e-12`.

### Paired permutation test and effect size

The two-sided paired permutation test uses 5,000 Monte Carlo detector-label
swaps plus the standard plus-one correction. A detector label is swapped by
patient group, so every image belonging to one patient moves together. The
group swap is consistent across every eligible seed in the clean analysis, and
retention uses the same group swap for clean and corrupted evidence. The
p-values therefore test detector exchangeability conditional on the observed
patient clusters and trained checkpoints; the hierarchical confidence
intervals additionally show the coarse empirical variation from resampling the
five available pairs, or the four complete conditional-localization pairs.

The effect size is the unstandardized paired aggregate difference (`A - B`),
reported with its patient-cluster bootstrap interval. The former image-level
jackknife Cohen's d is retained only in the archived tables. It is not carried
forward because a standardized pseudovalue interpretation is not clearly
established for these nonlinear pooled metrics with unequal patient-cluster
sizes. This avoids presenting a guessed cluster-level Cohen's d; it does not
redefine mAP as a per-image or per-patient mean.

Holm's step-down correction is applied once to the same seven clean predictive
endpoints. IoU and Dice remain in that family even though their paired seed
count is four rather than five.

For the corruption grid, Holm correction is applied separately within each
metric and estimand across the 35 conditions. The darkest-condition IoU and
Dice families contain 34 tests because one comparison is not estimable.
Confidence intervals are pointwise, not multiplicity-adjusted, so family-wise
claims use the Holm p-value rather than CI exclusion alone.

McNemar's test is not applied. This benchmark has multiple possible targets per
image plus negative-image false positives, not one independent binary outcome
per image. Collapsing it to “correct/incorrect” would discard detection outcome
structure, while target-level success indicators would be nested within images
and repeated across seeds/conditions.

## Clean all-attempt results

Values are estimate `[pointwise 95% CI]`. The comparison interval is also the
unstandardized effect-size interval; raw and Holm p-values accompany every
endpoint. Precision, recall, F1, and AP use all five paired attempts. The two
conditional rows use the complete pairs 17, 42, 137, and 314 (`n=4`), so the
Faster R-CNN localization estimates in this inferential table are also reduced
to those same four seeds to preserve pairing.

| Metric | Paired seed n | Faster R-CNN | YOLO11s | Difference/effect (A - B) | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Precision | 5 | 0.1959 [0.1394, 0.2505] | 0.2983 [0.1372, 0.4420] | -0.1024 [-0.2529, 0.0802] | 0.0004 | **0.0020** |
| Recall | 5 | 0.5799 [0.4747, 0.6670] | 0.0955 [0.0347, 0.1625] | 0.4843 [0.4065, 0.5555] | 0.0002 | **0.0014** |
| F1 | 5 | 0.2845 [0.2205, 0.3413] | 0.1427 [0.0549, 0.2317] | 0.1419 [0.0338, 0.2526] | 0.0002 | **0.0014** |
| Conditional IoU | **4** | 0.6746 [0.6550, 0.6919] | 0.6985 [0.6655, 0.7317] | -0.0239 [-0.0567, 0.0074] | 0.0562 | 0.1064 |
| Conditional Dice | **4** | 0.8007 [0.7864, 0.8131] | 0.8181 [0.7944, 0.8409] | -0.0173 [-0.0410, 0.0050] | 0.0532 | 0.1064 |
| mAP@0.5 | 5 | 0.3042 [0.2378, 0.3725] | 0.1626 [0.1029, 0.2239] | 0.1416 [0.0997, 0.1856] | 0.0052 | **0.0208** |
| mAP@0.5:0.95 | 5 | 0.0995 [0.0722, 0.1292] | 0.0542 [0.0326, 0.0816] | 0.0453 [0.0325, 0.0595] | 0.0114 | **0.0342** |

After correction, Faster R-CNN has evidence for higher recall, F1, and both AP
metrics, while YOLO11s has higher fixed-threshold precision. Conditional IoU
and Dice do not cross the 0.05 Holm threshold (`p_Holm=0.1064`). Their
comparison is conditional on a true-positive match and cannot counterbalance
YOLO11s's much lower recall.

Precision, recall, and F1 use the original Phase 5 score threshold of 0.25.
YOLO11s seed 271 contributes observed zeros to all three: it converged normally
but its maximum test score was `0.0412735`, so it emitted no thresholded
detection. Its AP@0.5 (`0.1587217`) and AP@0.5:0.95 (`0.0555799`) remain
plausible because AP uses ranked predictions retained down to 0.001. This is an
operational confidence-score degeneracy, not classic head/loss collapse and not
an unlucky IoU-matching event. The significant YOLO precision result remains a
fixed-threshold score-scale result rather than evidence of a superior frontier;
the official precision-recall analysis remains the frozen three-seed Batch 10
analysis.

The corrected patient-cluster Holm pattern changed from four of seven endpoints
at n=3 to five of seven at n=5. F1 became significant
(`p_Holm=0.0971805639` to `0.0013997201`); precision, recall, and both AP
endpoints remain significant, while conditional IoU and Dice remain
non-significant. All seven endpoints were corrected together despite the
conditional rows' four-pair eligibility.

The exact n=5 Holm-adjusted p-values are `0.0019996001` (precision),
`0.0013997201` (recall), `0.0013997201` (F1), `0.0207958408` (AP@0.5), and
`0.0341931614` (AP@0.5:0.95); conditional IoU and Dice are both
`0.1063787243`.

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

In the archived n=3 image-level versus patient-cluster audit, the clean family
did not change, but the secondary robustness pattern did. This is separate from
the Batch 16 n=3-to-n=5 clean comparison above.

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
a new clinical population. Five training seeds still only coarsely represent
training variability, and conditional localization has only four complete
pairs. Corruption inference remains primary-seed-only. The
corruption conditions are repeated digital transformations of the same images,
not independent deployment cohorts. Holm controls each declared family of
p-values but does not make the pointwise intervals simultaneous or make the
corruption conditions independent cohorts.

## Reproduction and artifacts

Run the CPU clean-only statistical refresh after Phase 5 is complete. This
preserves the existing seed-17 robustness table and does not rerun robustness:

```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode preflight
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run --scope clean
```

The outputs are `results/tables/statistical_clean_comparison.csv`,
`results/tables/statistical_robustness_comparison.csv`, and
`results/logs/phase8_statistics/summary.json`. The summary records input and
source hashes, patient-group counts, methods, all result rows, artifact hashes,
and the exact n=3-to-n=5 Holm-decision comparison. The corrected n=3 clean table
remains at `results/tables/statistical_clean_comparison_n3_archive.csv`. The
superseded image-level
outputs remain available for audit at
`results/tables/statistical_clean_comparison_image_level_archive.csv`,
`results/tables/statistical_robustness_comparison_image_level_archive.csv`, and
`results/logs/phase8_statistics/summary_image_level_archive.json`; they are not
primary results because they treat correlated exams as independent units.
