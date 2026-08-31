# Confidence-Threshold and Precision-Recall Analysis

## Historical n=3 analysis and n=5 sensitivity

The original analysis remains frozen at **n=3** (seeds 17, 42, and 137), and
none of its files was overwritten. `configs/threshold_sweep.yaml` and
`configs/threshold_selection.yaml` deliberately route their Phase 5 identity
checks through `configs/evaluation_n3_archive.yaml` and
`results/logs/phase5_evaluation/summary_n3_archive.json`; the archived Phase 5
comparison inputs are the corresponding `*_n3_archive.csv` tables. Output files
whose historical names do not contain `n3` remain three-seed artifacts and
must not be cited as five-seed results.

Batch 35 adds a separately named **n=5 all-attempt sensitivity** using seeds
17, 42, 137, 271, and 314 from the ten already-frozen test prediction bundles.
It performs no training, checkpoint loading, or model inference. The complete
test sweep and official PR curves are recomputed across five runs. The
detector-specific thresholds remain exactly 0.69 and 0.05, as selected on the
original three validation runs; they are applied to all five test bundles
without reselection or test-set feedback. Decision D-008 makes the n=5 curves
the manuscript's principal operating-regime display while retaining every n=3
artifact as the prespecified historical analysis and provenance reference.

## Plain finding

The fixed confidence threshold of 0.25 exaggerated YOLO11s's low-recall appearance, but it
did not create the underlying performance gap. Lowering the threshold to 0.01 raises
YOLO11s mean recall from 0.1356 to 0.3321 and mean F1 from 0.1981 to 0.2840. Its best F1 in
the requested sweep is therefore at the lower boundary, not at 0.25.

The apparent YOLO11s *precision advantage*, however, does not persist on the matched
precision-recall curve. Faster R-CNN has higher mean interpolated precision at 96 of the 101
official AP@0.5 recall positions; the remaining five positions are equal and YOLO11s is
higher at none. For the IoU-averaged AP@0.5:0.95 curve, Faster R-CNN is higher at 96
positions, YOLO11s at one near-zero-recall position, and four are equal. The corresponding
AP values reproduce the frozen comparison: 0.3084 +/- 0.0123 versus 0.1643 +/- 0.0226 at
IoU 0.50 and 0.1023 +/- 0.0036 versus 0.0549 +/- 0.0080 across IoU 0.50:0.95.

The n=5 sensitivity preserves the conclusion. Faster R-CNN is higher at 97 of
101 AP@0.5 recall positions (four ties, zero YOLO11s-higher positions) and at
96 AP@0.5:0.95 positions (four ties and one YOLO11s-higher position). Equal-run
mean AP is 0.3042 +/- 0.0189 versus 0.1626 +/- 0.0162 at IoU 0.50 and
0.0995 +/- 0.0067 versus 0.0542 +/- 0.0060 across IoU 0.50:0.95. The AP gaps
are slightly smaller than at n=3, so those quantitative margins are classified
as **weakened**, even though the AP@0.5 recall-position count strengthens from
96 to 97 and the threshold-free ordering remains unchanged.

The defensible conclusion is therefore narrower than "distinct operating regimes." The two
models have a score-scale/selectivity mismatch at the same nominal threshold, and YOLO11s
remains coverage-limited across the full curve, but it does not define a generally superior
high-precision regime. At matched recall or precision constraints, Faster R-CNN is better on
the tested targets. Later paper text should describe the 0.25 result as a fixed-threshold
operating-point difference, not as an architecture-inherent precision advantage. This sweep
does not assess probabilistic calibration: it does not test whether, for example, predictions
scored 0.8 are correct approximately 80% of the time.

## Protocol

The analysis reads the six frozen clean prediction bundles for Faster R-CNN and YOLO11s at
seeds 17, 42, and 137. Their SHA-256 hashes, test-annotation hash, detector/seed identities,
and evaluator settings are checked against the archived n=3 Phase 5 config and summary
before any metric is accepted. It performs no training, checkpoint loading, model
inference, NMS, or other prediction processing.

For each seed, 99 confidence thresholds from 0.01 through 0.99 in increments of 0.01 are
passed to the same `evaluate_operating_point` function used by `src/evaluate.py`. That
function retains the frozen post-NMS detections with score greater than or equal to the
threshold, caps each image at the same 100 detections, and applies the same score-ordered,
same-class greedy matching at IoU 0.50. Precision, recall, and F1 are micro-aggregated across
all 750 test images within a seed. The published threshold curve is the arithmetic mean and
sample standard deviation across the three seeds, consistent with Tables 4a and 4b.

This complete test-set sweep is descriptive and exploratory. Its peak-F1 thresholds must not
be treated as selected deployment settings. Phase 14 separately applies the same 99-point
protocol to validation predictions from the six already-trained, immutable best checkpoints.
Because the earlier training artifacts retained validation aggregates but not raw scored
detections, a one-time inference-only pass materializes hash-bound validation bundles; it
does not train, resume, or alter either model. All later threshold selection is offline.

The precision-recall figure does not reconstruct or approximate AP from the threshold grid.
It exposes pycocotools' official 101-recall-point interpolated precision tensor directly,
using the all-area, maximum-100-detections slice. The left panel is the IoU 0.50 slice
underlying AP@0.5; the right panel averages the ten IoU slices from 0.50 through 0.95 that
underlie AP@0.5:0.95. Lines and bands are the mean and sample standard deviation across the
three historical or five sensitivity run-specific official curves. The two
scopes live at separate paths and are never pooled into one unlabeled result.

## Threshold behavior

The table in this section describes the exploratory **test** sweep. In particular, the final
column is not used to select the final operating threshold.

| Detector | Threshold 0.25: precision / recall / F1 | Maximum mean recall in 0.01-0.99 | Peak mean F1 (threshold) |
|---|---:|---:|---:|
| Faster R-CNN | 0.1626 / 0.6381 / 0.2558 | 0.8532 at 0.01 | 0.3549 at 0.63 |
| YOLO11s | 0.3730 / 0.1356 / 0.1981 | 0.3321 at 0.01 | 0.2840 at 0.01 |

The separately versioned n=5 sensitivity is:

| Detector | Threshold 0.25: precision / recall / F1 | Maximum mean recall in 0.01-0.99 | Peak mean F1 (threshold) |
|---|---:|---:|---:|
| Faster R-CNN | 0.1959 / 0.5799 / 0.2845 | 0.7910 at 0.01 | 0.3562 at 0.63 |
| YOLO11s | 0.2983 / 0.0955 / 0.1427 | 0.2925 at 0.01 | 0.2657 at 0.01 |

Seed 271 is not filtered because it has no detections at 0.25. It contributes
defined zeros to YOLO11s precision, recall, and F1 there, but contributes its
observed low-score detections below 0.05 and its full official COCO PR curve
from the unchanged 0.001 bundle floor. Relative to n=3, the shared-threshold
YOLO precision margin and Faster R-CNN recall margin weaken, while the Faster
R-CNN F1 margin strengthens; all three directions remain unchanged.

The YOLO11s F1 optimum lies on the lower sweep boundary, so 0.2840 is the best *observed*
value in the predeclared range rather than a claimed global optimum. The official COCO
curves, which use the frozen predictions down to the Phase 5 minimum score of 0.001, retain
positive mean precision through recall 0.54 for YOLO11s and 0.96 for Faster R-CNN. Thus the
complete curve confirms that YOLO11s can recover more recall below 0.01, but also confirms a
substantial remaining coverage gap.

## Fixed operating targets

Targets are selected on each detector's three-seed mean threshold curve. For recall at fixed
precision, the reported point is the maximum mean recall among swept thresholds whose mean
precision meets the target. Precision at fixed recall is selected analogously. There is no
interpolation or extrapolation. The standard deviation is the across-seed sample SD at the
selected common threshold.

| Constraint on mean curve | Faster R-CNN response | YOLO11s response |
|---|---:|---:|
| Precision >= 0.50 | Recall 0.2189 +/- 0.1490 at threshold 0.78 | Recall 0.0759 +/- 0.0533 at threshold 0.44 |
| Precision >= 0.90 | Not reachable at any threshold | Not reachable at any threshold |
| Recall >= 0.30 | Precision 0.4208 +/- 0.1483 at threshold 0.73 | Precision 0.2547 +/- 0.0360 at threshold 0.01 |
| Recall >= 0.50 | Precision 0.2532 +/- 0.0343 at threshold 0.53 | Not reachable at any threshold |

"Not reachable" here means that no common threshold in the 0.01-0.99 grid makes the
three-seed *mean* satisfy the target. It does not conceal seed heterogeneity: at least one
threshold reaches precision 0.90 in two of three Faster R-CNN seeds and all three YOLO11s
seeds, but not at one common threshold whose detector-level mean reaches 0.90. For the
recall-0.30 target, all three Faster R-CNN seeds and two of three YOLO11s seeds reach the
target somewhere in the grid. No YOLO11s seed reaches recall 0.50.

In the n=5 sensitivity, precision >=0.50 yields recall 0.2381 +/- 0.1133 at
0.79 for Faster R-CNN and 0.0425 +/- 0.0553 at 0.46 for YOLO11s. At recall
>=0.30, Faster R-CNN reaches precision 0.4073 +/- 0.1083 at 0.73, whereas the
five-run YOLO11s mean does not reach the target anywhere on the grid. Neither
detector's mean reaches precision 0.90; Faster R-CNN alone reaches mean recall
0.50. These are exploratory test-curve descriptions, not newly selected
deployment thresholds.

These fixed-target comparisons reinforce the official curves. When mean precision is held
to at least 0.50, Faster R-CNN retains about 2.9 times the recall of YOLO11s. When mean recall
is held to at least 0.30, Faster R-CNN has about 1.65 times the precision. YOLO11s's higher
precision at the shared threshold 0.25 is therefore principally a score-scale/selectivity
mismatch rather than a superior precision-recall frontier.

## Validation-selected final operating points

The predeclared selection rule maximizes arithmetic mean F1 across the three validation
seeds on the same 0.01--0.99 grid. F1 gives precision and recall equal weight without
asserting an unmeasured clinical-harm function and retains the already-reviewed Batch 10
protocol. Exact mean-F1 ties are resolved toward the higher, more selective threshold. The
rule selects 0.69 for Faster R-CNN and 0.05 for YOLO11s. These values are then frozen and
each is applied exactly once to each corresponding frozen test bundle; the test results do
not feed back into selection.

| Detector | Validation precision / recall / F1 (n=3) | Frozen threshold | Historical test precision / recall / F1 (n=3) |
|---|---:|---:|---:|
| Faster R-CNN | 0.4164 +/- 0.0769 / 0.4404 +/- 0.0663 / 0.4202 +/- 0.0078 | 0.69 | 0.3543 +/- 0.0746 / 0.3607 +/- 0.0608 / 0.3492 +/- 0.0135 |
| YOLO11s | 0.4076 +/- 0.0107 / 0.3213 +/- 0.0315 / 0.3588 +/- 0.0209 | 0.05 | 0.3096 +/- 0.0134 / 0.2438 +/- 0.0302 / 0.2718 +/- 0.0181 |

These are the authoritative single-threshold operating-point results for the frozen n=3
threshold-selection, FROC, and Pareto analyses. The YOLO11s threshold 0.05 was selected from
only the three validation seeds 17, 42, and 137. It was not reselected at n=5: the later
seed-271 checkpoint has a maximum test confidence of only 0.0412735 and therefore produces
zero detections even at 0.05. That later observation is evidence of seed-specific confidence-
score instability, not a reason to revise this historical selection with test feedback.
The original 0.25 comparison and complete test sweep remain explicitly labeled protocol
sensitivity analyses.

Applying the same, unchanged thresholds to all five frozen test bundles gives:

| Detector | Threshold-selection n | Frozen threshold | Sensitivity test n | Test precision / recall / F1, mean +/- SD |
|---|---:|---:|---:|---:|
| Faster R-CNN | 3 | 0.69 | 5 | 0.3624 +/- 0.0581 / 0.3507 +/- 0.0463 / 0.3511 +/- 0.0184 |
| YOLO11s | 3 | 0.05 | 5 | 0.2524 +/- 0.1418 / 0.1948 +/- 0.1110 / 0.2192 +/- 0.1233 |

All three Faster-R-CNN-minus-YOLO11s margins strengthen relative to n=3. This
is a sensitivity of test performance to the all-attempt run set, not a new
threshold-selection analysis: validation evidence remains n=3, test evidence
is n=5, and seed 271 contributes 0/0/0 with zero emitted detections at 0.05.

## Artifacts and reproduction

The immutable Phase 5 sources for every artifact below are
`configs/evaluation_n3_archive.yaml` and
`results/logs/phase5_evaluation/summary_n3_archive.json`. The archived comparison CSVs are
`results/tables/detector_comparison_n3_archive.csv` and
`results/tables/detector_comparison_per_seed_n3_archive.csv` (with the corresponding
mean/SD archive). The threshold output filenames below predate the seed expansion, but their
contents remain n=3.

- `results/tables/threshold_sweep.csv`: detector-level mean and sample SD at all 99
  thresholds.
- `results/tables/threshold_sweep_per_seed.csv`: the 594 seed-specific operating points.
- `results/tables/precision_recall_curves.csv`: detector-level mean and sample SD on both
  official 101-point COCO curves.
- `results/tables/precision_recall_curves_per_seed.csv`: the 1,212 seed-specific COCO curve
  points.
- `results/tables/threshold_operating_targets.csv`: the fixed-target results and per-seed
  reachability counts.
- `results/figures/precision_recall_curves.png` and
  `results/figures/f1_vs_threshold.png`: the review figures.
- `results/logs/phase10_threshold_sweep/summary.json`: source hashes, bundle hashes,
  reproduction checks, target results, finding summary, and artifact hashes.
- `results/tables/validation_threshold_sweep*.csv`: validation-only threshold curves used
  for selection.
- `results/tables/selected_operating_points*.csv`: validation-selected thresholds and their
  one-shot test precision, recall, and F1, in aggregate and per seed.
- `results/logs/phase14_threshold_selection/`: the validation-bundle manifest, hashes,
  inference environment, and final selection summary.

From the repository root in the pinned Python 3.11 environment:

```powershell
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep.yaml --mode preflight
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep.yaml --mode run

# One-time inference-only materialization from the immutable best checkpoints.
& $benchmarkPython -m src.evaluate_threshold_selection --config configs/threshold_selection.yaml --mode preflight
& $benchmarkPython -m src.evaluate_threshold_selection --config configs/threshold_selection.yaml --mode collect-validation

# Offline validation sweep, threshold freeze, and one-shot test evaluation.
& $benchmarkPython -m src.evaluate_threshold_selection --config configs/threshold_selection.yaml --mode run
```

The Batch 10 run and the final selection run take only frozen n=3 JSON prediction records as
model evidence. With the current configs, these commands reproduce the archived three-seed
analysis only; they do not regenerate a five-seed threshold study. The separate
materialization command performs validation inference because raw validation scores were
not archived during training; it cannot train or update weights.

The n=5 sensitivity is entirely offline and writes only versioned paths:

```powershell
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.evaluate_threshold_sweep --config configs/threshold_sweep_n5_sensitivity.yaml --mode run
& $benchmarkPython -m src.analyze_operating_regime_sensitivity --config configs/operating_regime_n5_sensitivity.yaml --mode fixed-thresholds
```

Its main tables are `threshold_sweep_n5_sensitivity.csv`,
`threshold_sweep_per_seed_n5_sensitivity.csv`,
`precision_recall_curves_n5_sensitivity.csv`,
`precision_recall_curves_per_seed_n5_sensitivity.csv`, and
`selected_operating_points*_n5_sensitivity.csv`; figures use the matching n=5
suffix. `results/logs/phase35_operating_regime_n5/threshold_summary.json`
hashes the config, annotations, all ten bundles, source code, and outputs. The
final Batch 35 summary additionally binds these files to FROC, Pareto, the
inventory, and the complete n=3-versus-n=5 conclusion table.
