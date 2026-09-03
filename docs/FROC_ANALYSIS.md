# Free-Response ROC Analysis

## Historical n=3 scope and n=5 sensitivity

This FROC analysis is frozen at **n=3** (seeds 17, 42, and 137) and was not
regenerated after the clean comparison expanded to five seeds.
`configs/froc.yaml` consumes the archived Batch 10 threshold outputs;
`configs/threshold_sweep.yaml` in turn binds those outputs to
`configs/evaluation_n3_archive.yaml` and
`results/logs/phase5_evaluation/summary_n3_archive.json`. The Phase 5 source
tables are the `*_n3_archive.csv` comparison files. Historical FROC and
threshold filenames without an `n3` suffix remain three-seed artifacts.

Batch 35 retains those files unchanged and adds a separately named n=5
all-attempt sensitivity from the same 0.01--0.99 grid. It uses all five frozen
test bundles per detector and performs no training, checkpoint loading,
inference, interpolation, threshold reselection, or test-set tuning.

## Plain finding

Faster R-CNN has higher mean test sensitivity than YOLO11s at every reported false-positive
budget in the frozen 0.01--0.99 sweep. The gap is modest at the two lowest budgets and grows
as more false positives are permitted: sensitivity is 0.2699 versus 0.1803 at 0.125 FP/image,
0.3607 versus 0.2749 at 0.25, 0.4801 versus 0.3321 at 0.5, 0.6032 versus 0.3321 at 1, and
0.6928 versus 0.3321 at 2 FP/image.

YOLO11s plateaus at mean sensitivity 0.3321 from the 0.5-FP/image budget upward because its
least selective available point is the lower sweep boundary, threshold 0.01, with mean 0.36
FP/image. This is a boundary of the archived Batch 10 sweep, not evidence that the detector's
true FROC curve cannot extend further below threshold 0.01. Faster R-CNN continues trading
more false positives for sensitivity across the reported range. The result reinforces the
full precision-recall finding: the shared-threshold precision difference reflects a
score-scale/selectivity mismatch, while Faster R-CNN retains broader detection coverage.

The n=5 sensitivity reaches the same directional conclusion at all five
budgets. Sensitivity is 0.2672 versus 0.1761 at 0.125 FP/image, 0.3642 versus
0.2455 at 0.25, 0.4836 versus 0.2925 at 0.5, 0.5970 versus 0.2925 at 1, and
0.6873 versus 0.2925 at 2. Each Faster-R-CNN-minus-YOLO11s gap is slightly
larger than at n=3, so all five budget conclusions are classified as
**strengthened**. This does not turn the descriptive curve into a selected or
clinically validated operating point.

## Protocol

This comparison does not use the test set to choose one deployment operating
threshold. It reparameterizes every existing Batch 10 test threshold by two
quantities already produced by the canonical matcher. The comparison remains
conditional on the retained predictions, the evaluated 0.01--0.99 grid, and
the common matching/post-processing protocol:

- sensitivity is micro-aggregated recall at match IoU 0.50; and
- average false positives per image is the total false-positive detection count divided by
  all 750 test images, including images with no annotated opacity.

The inputs are the unchanged 594 seed-threshold rows in
`results/tables/threshold_sweep_per_seed.csv`: 99 thresholds for each detector and each of
seeds 17, 42, and 137. The script verifies that table's hash against the completed Batch 10
summary and checks every sensitivity against TP / (TP + FN). It performs no training,
checkpoint loading, inference, NMS, rematching, interpolation, or extrapolation. The plotted
line uses the mean FP/image and mean sensitivity at each common threshold; the shaded band is
the sample SD of sensitivity across three seeds.

YOLO11s seed 271 is not part of the historical curves. Its training converged normally, but its
maximum test confidence is 0.0412735 and it produces zero detections even at the frozen
n=3-selected threshold 0.05. The n=5 path includes the run exactly as observed:
sensitivity is 0.1493 at threshold 0.01 with 0.12 FP/image, 0.0896 at 0.02,
0.0448 at 0.03, and zero at 0.04 and above (one false positive remains at
0.04). Thus the run is not automatically filtered for having no detections at
0.25 or 0.05.

For budget summaries, each seed independently contributes the highest observed sensitivity
whose FP/image is at or below the requested budget. Exact sensitivity ties prefer fewer
false positives and then the higher threshold. The reported value is the arithmetic mean and
sample SD of those three seed-specific sensitivities. This conservative, non-interpolated
rule never claims performance at an unobserved threshold.

## False-positive budgets

The predeclared set is 0.125, 0.25, 0.5, 1, and 2 FP/image. This doubling sequence spans one
false alert per eight images through two false alerts per image, giving a compact range of
clinically interpretable candidate-review burdens without implying that any point is
clinically validated, acceptable, or recommended for deployment.

| FP/image budget | Faster R-CNN sensitivity | YOLO11s sensitivity |
|---:|---:|---:|
| 0.125 | 0.2699 +/- 0.0151 | 0.1803 +/- 0.0212 |
| 0.25 | 0.3607 +/- 0.0248 | 0.2749 +/- 0.0151 |
| 0.5 | 0.4801 +/- 0.0326 | 0.3321 +/- 0.0529 |
| 1 | 0.6032 +/- 0.0155 | 0.3321 +/- 0.0529 |
| 2 | 0.6928 +/- 0.0206 | 0.3321 +/- 0.0529 |

The versioned n=5 sensitivity table is:

| FP/image budget | Faster R-CNN sensitivity (n=5) | YOLO11s sensitivity (n=5) | n=3 to n=5 |
|---:|---:|---:|---|
| 0.125 | 0.2672 +/- 0.0120 | 0.1761 +/- 0.0217 | Strengthened |
| 0.25 | 0.3642 +/- 0.0259 | 0.2455 +/- 0.0556 | Strengthened |
| 0.5 | 0.4836 +/- 0.0316 | 0.2925 +/- 0.0886 | Strengthened |
| 1 | 0.5970 +/- 0.0221 | 0.2925 +/- 0.0886 | Strengthened |
| 2 | 0.6873 +/- 0.0267 | 0.2925 +/- 0.0886 | Strengthened |

Because the threshold grid is discrete, the actual mean false-positive load selected under a
budget can be lower than the budget. The required CSV reports those achieved means and SDs,
plus the seed-specific threshold range. In particular, YOLO11s's achieved mean is 0.36
FP/image for the 0.5, 1, and 2 budgets because threshold 0.01 is the available boundary.

## Artifacts and reproduction

- `results/figures/froc_curves.png`: both detector FROC curves with across-seed sensitivity
  bands and the five non-interpolated budget summaries.
- `results/tables/froc_operating_points.csv`: mean sensitivity, sample SD, achieved FP/image,
  and threshold range at every requested budget.
- `results/logs/phase14_froc/summary.json`: input hashes, protocol, all aggregate and per-seed
  budget selections, counts, and output hashes.

From the repository root in the pinned Python 3.11 environment:

```powershell
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n3_archive.yaml --mode preflight
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n3_archive.yaml --mode run
```

The archive config validates the exact historical Batch 10 config hash through
`configs/threshold_sweep_n3_frozen.yaml` and writes to separate
`*_n3_archive_reproduction` paths, so it cannot overwrite the frozen primary artifacts.
It reproduces only the three-seed FROC values; it does not create an n=5 FROC analysis.

The separate n=5 commands are:

```powershell
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.plot_froc_curves --config configs/froc_n5_sensitivity.yaml --mode run
```

They write aggregate and per-run curve tables
`froc_curves_n5_sensitivity.csv` and
`froc_curves_per_seed_n5_sensitivity.csv`, the five-budget table
`froc_operating_points_n5_sensitivity.csv`, the explicitly labeled figure
`froc_curves_n5_sensitivity.png`, and the hash-bound
`phase35_operating_regime_n5/froc_summary.json`. Every aggregate row carries
`seed_count=5`; the summary also retains all 50 per-run budget selections.
