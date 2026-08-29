# Statistical analysis

The clean analysis uses only the ten frozen Phase 5 prediction bundles and the
unchanged 750-image, 323-patient test set. It does not retrain either detector.
Faster R-CNN is detector A, YOLO11s is detector B, and every difference is
`A - B`. The former image-level and paired-seed analyses remain only in
explicitly named audit or sensitivity archives.

## Inferential targets

Two inferential targets are reported and must not be collapsed into one
generic statement of “statistical significance.”

### Primary: training-procedure estimand

The **training-procedure estimand** represents the expected difference between
the two disclosed pipelines over both a new held-out patient sample and a new
stochastic training realization under each pipeline's fixed recipe. The 95%
bootstrap intervals belong to this estimand. Each draw resamples NIH patient
clusters once, shared across detectors, and resamples trained runs separately
within Faster R-CNN and YOLO11s.

This is the primary target for broad pipeline claims. Precision, recall, F1,
and both AP endpoints use five runs per detector. Conditional matched-box IoU
and Dice use five defined Faster R-CNN runs and four defined YOLO11s runs.
YOLO11s seed 271 emitted no score-0.25 detection, so its matched-only metrics
are undefined, not zero; Faster R-CNN seed 271 remains defined and contributes.

### Secondary: checkpoint-conditional estimand

The **checkpoint-conditional estimand** conditions on the already-trained
checkpoints and represents held-out patient sampling only. The patient-cluster
permutation p-values belong here. They are a secondary sensitivity analysis in
columns explicitly labeled **conditional on observed checkpoints**. For
conditional localization, the historical label-swap calculation uses the four
complete same-label checkpoint pairs so its exact prior result is reproducible.
This does not make those numeric labels matched training blocks.

### Why the CI and p-value can differ

In plain language, the CI asks, “Would the pipeline difference persist if we
sampled new patients and repeated each training procedure?” The permutation
p-value asks, “For these already observed checkpoints, how unusual is the
patient-level detector-label contrast under exchangeability?” A small
checkpoint-conditional p-value can coexist with a training-procedure CI that
crosses zero when observed checkpoints differ consistently across patients but
trained-run variability is appreciable. That is exactly the precision result:
the primary interval is `[-0.2423, 0.0553]`, while the secondary
checkpoint-conditional Holm p-value is `0.0020`. Neither arithmetic result is
wrong, but the p-value is not evidence that the training-procedure difference
is robust.

## Seed-block audit

Seed 17 for Faster R-CNN and seed 17 for YOLO11s are not a scientifically
meaningful matched stochastic block. Both pipelines disable stochastic
augmentation, so there is no augmentation RNG draw to couple. Their remaining
randomness is also not coupled:

- Faster R-CNN uses the project PyTorch `DataLoader`, batch size 2, and a
  per-run PyTorch generator; YOLO11s uses the Ultralytics data stack and batch
  size 4. Equal integers do not produce the same minibatch blocks.
- Model initialization occurs in different architectures, parameter shapes,
  constructors, and framework call sequences. The random streams do not
  represent aligned initialization perturbations.
- The training loops consume RNG state through different frameworks and stop
  at different epochs. No code saves or shares random variates between arms.

The five runs per detector are independent realizations that happen to share
reproducibility labels. The old common seed-multiplicity bootstrap is retained
in `statistical_clean_comparison_paired_seed_sensitivity_archive.csv`, but it is
not primary.

## Endpoints

The clean comparison covers precision, recall, F1, conditional matched-box IoU
and Dice, mAP@0.5, and mAP@0.5:0.95. The corruption analysis remains a separate
seed-17 checkpoint analysis on the fixed 300-image sample. For each of 35
conditions it compares both:

- raw performance, `metric(A, corruption) - metric(B, corruption)`; and
- clean-relative retention,
  `metric(A, corruption) / metric(A, clean) - metric(B, corruption) /
  metric(B, clean)`.

Retention is a ratio rather than a percentage-point degradation. It can exceed
one when a corruption improves a finite-sample estimate. Conditional IoU and
Dice remain undefined when a detector has no matched true positives.

## Inference protocol

All resampling parameters are fixed in `configs/statistics.yaml`.

### Training-procedure bootstrap confidence intervals

The clean table reports two-sided pointwise 95% percentile intervals from
2,000 draws. The sampling unit is the NIH patient group recorded in the
committed Batch 1 test manifest (`nih_patient_id` in
`data/splits/rsna-pneumonia-5000/test.csv`). A draw samples the 323 patient
groups with replacement; each sampled group contributes every observed image
belonging to that patient with one shared multiplicity. Two additional
multinomial draws independently sample eligible Faster R-CNN and YOLO11s runs.
The sample size within each run pool equals that pool's observed run count.

Every nonlinear metric is recomputed within each sampled run from the sampled
patient predictions. Precision, recall, and F1 are rebuilt from TP/FP/FN; IoU
and Dice from matched localization sums; and AP from score-ordered detections
at all ten COCO IoU thresholds and 101 recall thresholds. Run-level metrics are
then averaged using the sampled run multiplicities. The implementation does not
bootstrap precomputed summary means and does not create a per-image AP
surrogate. Reconstruction matches every frozen Phase 5 bundle's unified metric
vector to absolute tolerance `5e-12`.

### Checkpoint-conditional patient-cluster permutation and effect

The two-sided paired permutation test uses 5,000 Monte Carlo detector-label
swaps plus the standard plus-one correction. A detector label is swapped by
patient group, so every image belonging to one patient moves together. The
group swap is consistent across every eligible observed checkpoint. The
p-values test detector exchangeability conditional on the observed patient
clusters and checkpoints. They do not incorporate stochastic retraining
variability. No new seed-aware p-value or hierarchical hypothesis test is
introduced.

The effect is the unstandardized training-procedure difference (`A - B`),
reported with its training-procedure bootstrap interval. The former image-level
jackknife Cohen's d is retained only in the archived tables. It is not carried
forward because a standardized pseudovalue interpretation is not clearly
established for these nonlinear pooled metrics with unequal patient-cluster
sizes. This avoids presenting a guessed cluster-level Cohen's d; it does not
redefine mAP as a per-image or per-patient mean.

Holm's step-down correction is applied once to the seven checkpoint-conditional
clean permutation p-values. IoU and Dice remain in that family, with four
complete checkpoint pairs for this secondary calculation.

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

## Clean training-procedure results

Values are point estimate `[pointwise 95% training-procedure CI]`. All rows use
750 images and 323 patient clusters. `Runs A/B` gives the eligible Faster
R-CNN/YOLO11s run counts. The last column is deliberately separate and names
the secondary inferential target.

| Endpoint | Runs A/B | Conditioning | Faster R-CNN | YOLO11s | Difference (A - B) | Seed 271 contribution | Holm p, conditional on observed checkpoints |
|---|---:|---|---:|---:|---:|---|---:|
| Precision | 5/5 | Unconditional | 0.1959 [0.1418, 0.2528] | 0.2983 [0.1412, 0.4384] | -0.1024 [-0.2423, 0.0553] | Both runs; YOLO contributes observed zero | **0.0020** |
| Recall | 5/5 | Unconditional | 0.5799 [0.4780, 0.6693] | 0.0955 [0.0400, 0.1621] | 0.4843 [0.3830, 0.5830] | Both runs; YOLO contributes observed zero | **0.0014** |
| F1 | 5/5 | Unconditional | 0.2845 [0.2223, 0.3470] | 0.1427 [0.0638, 0.2307] | 0.1419 [0.0559, 0.2290] | Both runs; YOLO contributes observed zero | **0.0014** |
| Conditional IoU | 5/4 | Conditional on a matched detection | 0.6749 [0.6561, 0.6919] | 0.6985 [0.6650, 0.7296] | -0.0236 [-0.0585, 0.0089] | Faster contributes; YOLO undefined | 0.1064 |
| Conditional Dice | 5/4 | Conditional on a matched detection | 0.8010 [0.7876, 0.8132] | 0.8181 [0.7944, 0.8400] | -0.0171 [-0.0420, 0.0066] | Faster contributes; YOLO undefined | 0.1064 |
| mAP@0.5 | 5/5 | Unconditional | 0.3042 [0.2347, 0.3706] | 0.1626 [0.1058, 0.2233] | 0.1416 [0.1013, 0.1845] | Both ranked-prediction bundles contribute | **0.0208** |
| mAP@0.5:0.95 | 5/5 | Unconditional | 0.0995 [0.0725, 0.1287] | 0.0542 [0.0332, 0.0785] | 0.0453 [0.0313, 0.0599] | Both ranked-prediction bundles contribute | **0.0342** |

Under the primary training-procedure estimand, Faster R-CNN's recall, F1, and
both AP differences remain wholly above zero. Precision does not: its interval
crosses zero and is not robust evidence of a training-procedure difference,
even though the observed-checkpoint permutation favors YOLO11s at the fixed
threshold. Conditional IoU and Dice also cross zero and remain descriptive
matched-detection comparisons.

Precision, recall, and F1 use the original Phase 5 score threshold of 0.25.
YOLO11s seed 271 contributes observed zeros to all three: it converged normally
but its maximum test score was `0.0412735`, so it emitted no thresholded
detection. Its AP@0.5 (`0.1587217`) and AP@0.5:0.95 (`0.0555799`) remain
plausible because AP uses ranked predictions retained down to 0.001. This is an
operational confidence-score degeneracy, not classic head/loss collapse and not
an unlucky IoU-matching event. The checkpoint-conditional YOLO precision result
remains a fixed-threshold score-scale result rather than evidence of a superior frontier;
the official precision-recall analysis remains the frozen three-seed Batch 10
analysis.

The checkpoint-conditional Holm-adjusted p-values are `0.0019996001`
(precision), `0.0013997201` (recall), `0.0013997201` (F1), `0.0207958408`
(AP@0.5), and `0.0341931614` (AP@0.5:0.95); conditional IoU and Dice are both
`0.1063787243`. This is a statement about the observed checkpoints, not a count
of training-procedure differences.

## Pairing sensitivity and seed influence

Replacing common seed-index draws with independent detector-run draws changed
interval endpoints but did not change zero exclusion for any endpoint. The
paired-seed interval is retained only for sensitivity: precision changed from
`[-0.2529, 0.0802]` to `[-0.2423, 0.0553]`, F1 from `[0.0338, 0.2526]` to
`[0.0559, 0.2290]`, and mAP@0.5:0.95 from `[0.0325, 0.0595]` to
`[0.0313, 0.0599]`.

The per-run and deletion diagnostics are descriptive. Omitting the seed-271
label from both detectors changes fixed-threshold precision difference from
`-0.1024` to `-0.1892` and F1 difference from `0.1419` to `0.0938`; the
mAP@0.5:0.95 difference changes only from `0.04533` to `0.04504`. These values
show influence, not a corrected result. Seed 271 remains in every endpoint for
which its metric is defined.

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

The bootstrap and permutation keep every observed exam from one NIH patient
together. This is still a nonparametric analysis of 323 observed clean patient
groups and 183 robustness-sample groups, not a guarantee of transportability to
a new clinical population. Five runs per detector only coarsely represent
training variability; conditional localization has five defined Faster R-CNN
runs and four defined YOLO11s runs. The checkpoint-conditional localization
permutation has four complete pairs. Corruption inference remains
primary-seed-only. The
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

The primary table is `results/tables/statistical_clean_comparison.csv`. Seed
influence is recorded in `statistical_clean_per_run_metrics.csv`,
`statistical_clean_leave_one_run_out.csv`, and
`statistical_clean_leave_one_seed_label_out.csv`. The provenance summary is
`results/logs/phase8_statistics/summary.json`; it records hashes, inferential
targets, the seed-block audit, every clean row, the paired-versus-independent
interval comparison, and all diagnostic artifacts. The seed-17 robustness
table remains byte-identical at
`results/tables/statistical_robustness_comparison.csv`.

The former paired-seed calculation is preserved at
`results/tables/statistical_clean_comparison_paired_seed_sensitivity_archive.csv`
and `results/logs/phase8_statistics/summary_paired_seed_sensitivity_archive.json`.
The corrected n=3 clean table remains at
`results/tables/statistical_clean_comparison_n3_archive.csv`. The superseded image-level
outputs remain available for audit at
`results/tables/statistical_clean_comparison_image_level_archive.csv`,
`results/tables/statistical_robustness_comparison_image_level_archive.csv`, and
`results/logs/phase8_statistics/summary_image_level_archive.json`; they are not
primary results because they treat correlated exams as independent units.
