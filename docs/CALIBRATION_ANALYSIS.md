# Detection-Specific Confidence Calibration

## Finding

Faster R-CNN is better calibrated than YOLO11s under the full five-dimensional Detection
Expected Calibration Error (D-ECE) protocol. Across all five frozen training seeds, mean
D-ECE is `0.0320 +/- 0.0058` for Faster R-CNN and `0.0990 +/- 0.0232` for YOLO11s. The
seed ranges do not overlap: `0.0266--0.0403` versus `0.0714--0.1313`.

YOLO11s seed 271 is retained and is the worst-calibrated YOLO seed. Its 962 emitted test
detections have mean confidence `0.00538`, yet `0.15073` are true positives under the
canonical matcher. The resulting absolute global confidence--correctness gap is `0.14535`,
and the box-sensitive D-ECE is `0.13130`, 1.46 times the median D-ECE of the other four
YOLO11s seeds. Its maximum confidence remains only `0.0412735`. This is direct calibration
evidence for the documented score pathology: the scores substantially understate empirical
correctness rather than merely falling below an arbitrary operating threshold.

These values are descriptive calibration measurements on the held-out test detections. No
confidence transformation is fitted, and no claim is made that either model supplies
clinically calibrated risk.

## Method

The analysis follows Küppers et al., ["Multivariate Confidence Calibration for Object
Detection"](https://doi.org/10.1109/CVPRW50498.2020.00171) (CVPR Workshops, 2020). For an
emitted detection with confidence \(p\), predicted class \(y\), and relative box encoding
\(r=(c_x,c_y,w,h)\), black-box detection calibration requires

\[
P(m=1 \mid \hat p=p, \hat y=y, \hat r=r)=p,
\]

where \(m=1\) means the detection is assigned to an unmatched same-class ground-truth box
at the stated IoU threshold. Precision is the relevant correctness outcome because a missed
target has no emitted confidence and cannot enter black-box confidence calibration.
The benchmark has one foreground class, so this class condition is fixed rather than added
as another histogram dimension.

For each of the ten frozen Phase 5 bundles, `src/stats/calibration.py`:

1. retains every post-NMS prediction with score at least `0.001`, capped at 100 per image;
2. applies the project's canonical stable score-order, same-class greedy matcher at IoU
   `0.50`;
3. labels every retained prediction as a matched true positive or unmatched false positive;
4. converts each box to relative center and scale
   `(confidence, center_x, center_y, width, height)` in `[0, 1]^5`; and
5. partitions all five dimensions into five equal-width bins.

For multivariate cell \(b\), let \(n_b\) be its detection count, \(N\) the number of retained
detections, \(\operatorname{prec}(b)\) its true-positive fraction, and
\(\operatorname{conf}(b)\) its mean score. The reported metric is

\[
\operatorname{D\text{-}ECE}
=\sum_b \frac{n_b}{N}
\left|\operatorname{prec}(b)-\operatorname{conf}(b)\right|.
\]

As in the paper's full five-dimensional evaluation, each dimension has five bins and cells
with fewer than eight samples do not contribute to D-ECE. The CSV reports both the total and
included detection counts so this robustness threshold is visible. D-ECE values are
comparable here because every detector/seed uses the same prediction floor, IoU, feature
dimensions, bin edges, minimum cell count, image set, and matching rule.

The [public reference framework](https://github.com/EFS-OpenSource/calibration-framework)
is currently Apache-2.0 licensed, which is compatible with this repository's
AGPL-3.0-only license under [GNU's version-3 compatibility
guidance](https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility). The analysis
nevertheless uses an independent NumPy implementation of the paper's D-ECE equation rather
than adding `netcal` and its broader dependency stack; no reference-framework source was
copied.

## Per-seed results

| Detector | Seed | Predictions | TP fraction | Mean confidence | Global gap | D-ECE |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 17 | 18,260 | 0.01396 | 0.05590 | 0.04194 | 0.03507 |
| Faster R-CNN | 42 | 7,682 | 0.03176 | 0.07502 | 0.04325 | 0.03129 |
| Faster R-CNN | 137 | 25,030 | 0.01035 | 0.04017 | 0.02982 | 0.02663 |
| Faster R-CNN | 271 | 6,588 | 0.03643 | 0.06981 | 0.03338 | 0.02695 |
| Faster R-CNN | 314 | 3,663 | 0.05678 | 0.10609 | 0.04931 | 0.04027 |
| YOLO11s | 17 | 1,019 | 0.14230 | 0.06107 | 0.08123 | 0.07142 |
| YOLO11s | 42 | 773 | 0.16041 | 0.06745 | 0.09296 | 0.08830 |
| YOLO11s | 137 | 394 | 0.24619 | 0.20187 | 0.04432 | 0.11227 |
| **YOLO11s** | **271** | **962** | **0.15073** | **0.00538** | **0.14535** | **0.13130** |
| YOLO11s | 314 | 905 | 0.14586 | 0.04569 | 0.10016 | 0.09185 |

The global gap is included only as a transparent signed-scale diagnostic summarized in
absolute value; it is not substituted for D-ECE. In particular, YOLO11s seed 137's global
gap is modest while its D-ECE is high, showing why conditioning on predicted location and
scale adds information beyond a confidence-only average.

The reliability diagrams plot mean confidence against the fraction of matched true positives
in ten equal-width confidence bins. Thin lines show individual seeds, thick lines pool the
emitted detections within each detector, and the seed-271 point is called out explicitly.
These one-dimensional diagrams are visual summaries; the numerical endpoint remains the
five-dimensional D-ECE.

## Calibration is not threshold selectivity

This analysis **measures probabilistic detection calibration**: whether detections assigned
probability \(p\) are empirically correct at approximately rate \(p\), conditional on score,
class, box location, and box scale. It therefore answers the question that
[`THRESHOLD_ANALYSIS.md`](THRESHOLD_ANALYSIS.md) explicitly left open.

The earlier threshold analysis measures a different property. It asks how precision, recall,
F1, and false positives per image change as a score cutoff selects more or fewer boxes, and
it established that the same numerical threshold selects different operating regimes for the
two detectors. A detector may have a shifted score scale yet preserve ranking, or may select
a useful operating point while its scores do not equal empirical probabilities. Conversely,
a calibrated score does not guarantee high AP, recall, or utility. The threshold/selectivity
and calibration results are therefore complementary, not duplicate formulations of one
claim.

## Scope and limitations

- Calibration is evaluated on the test set without fitting a post-hoc calibrator. A future
  calibration map would need to be fitted on validation data and evaluated once on untouched
  test data.
- The estimand is conditional on post-NMS detections retained at score `0.001`. It measures
  emitted-detection precision calibration, not recall calibration, missed-target probability,
  pre-NMS proposal calibration, or localization uncertainty.
- Absolute D-ECE depends on the IoU threshold, feature dimensions, binning, and minimum cell
  count. Values under another protocol must not be compared directly.
- Five seeds characterize recipe-level variation only coarsely. The pooled visual curves
  weight every detection equally, while the headline detector summary gives each seed equal
  weight by reporting the mean and sample standard deviation of seed-level D-ECE.
- The result is specific to this one-class, single-site-derived benchmark and is not evidence
  of prospective, external-site, subgroup, or clinical-risk calibration.

## Artifacts and reproduction

- [`calibration_summary.csv`](../results/tables/calibration_summary.csv): ten seed rows plus
  equal-seed-weight detector mean/sample-SD rows, including explicit seed-271 diagnostics.
- [Reliability diagrams](../results/figures/reliability_diagrams.png): both detectors, pooled
  and seed-specific reliability curves.
- `results/logs/phase18_calibration/summary.json`: exact config, annotation, Phase 5 summary,
  prediction-bundle and output hashes; per-cell D-ECE and reliability records.

From the repository root in the pinned Python 3.11 environment:

```powershell
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode run
```

Both commands are CPU-only and operate on the frozen prediction bundles. They do not load a
checkpoint, run inference, or train either detector.
