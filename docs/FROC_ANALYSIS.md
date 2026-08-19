# Free-Response ROC Analysis

## Frozen three-seed scope

This FROC analysis is frozen at **n=3** (seeds 17, 42, and 137) and was not
regenerated after the clean comparison expanded to five seeds.
`configs/froc.yaml` consumes the archived Batch 10 threshold outputs;
`configs/threshold_sweep.yaml` in turn binds those outputs to
`configs/evaluation_n3_archive.yaml` and
`results/logs/phase5_evaluation/summary_n3_archive.json`. The Phase 5 source
tables are the `*_n3_archive.csv` comparison files. Historical FROC and
threshold filenames without an `n3` suffix remain three-seed artifacts.

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

## Protocol

This is a threshold-free comparison in the operating-point-selection sense: it does not use
the test set to choose one deployment threshold. It reparameterizes every existing Batch 10
test threshold by two quantities already produced by the canonical matcher:

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

YOLO11s seed 271 is not part of these curves. Its training converged normally, but its
maximum test confidence is 0.0412735 and it produces zero detections even at the frozen
n=3-selected threshold 0.05. Because the archived FROC grid extends to 0.01 but contains no
seed-271 row, this document makes no claim about an n=5 FROC curve or the seed's behavior
below 0.05.

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
