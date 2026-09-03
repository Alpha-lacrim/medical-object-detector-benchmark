# Detection-Confidence Calibration

## Descriptive finding

Under the original, common D-ECE setting, the observed equal-run mean was
`0.0320 +/- 0.0058` for Faster R-CNN and `0.0990 +/- 0.0232` for YOLO11s.
The run ranges were `0.0266--0.0403` and `0.0714--0.1313`, respectively. This
is a descriptive comparison of ten frozen runs; no patient- or run-level
inferential analysis was added for D-ECE.

D-ECE is a calibration-error measure: under the same binning, support,
matching, and prediction-floor protocol, a lower value denotes lower observed
calibration error. It is not a score for which a larger value is better.

All configured runs went through the same code path. No run was excluded or
handled specially. The largest observed YOLO11s value was `0.13130` for the run
with seed 271. That run contributed all 962 detections available at the 0.001
bundle floor; its mean confidence was `0.00538` and its matched fraction was
`0.15073`. These are results, not a predeclared expectation that this run had to
be an outlier.

D-ECE here concerns confidence attached to emitted detections. It is not
exam-level disease-risk calibration, clinical-risk calibration, or evidence of
clinical utility.

## Definition and verified implementation

The implementation follows Kuppers et al., ["Multivariate Confidence
Calibration for Object Detection"](https://doi.org/10.1109/CVPRW50498.2020.00171)
(CVPR Workshops, 2020). For emitted confidence (p), predicted class (y),
relative box (r=(c_x,c_y,w,h)), and binary matching outcome (m), the
detection-calibration condition is

\[
P(m=1 \mid \hat p=p, \hat y=y, \hat r=r)=p.
\]

The audited implementation is:

- **Confidence:** the post-NMS detector score, with inclusion by score
  `>= 0.001`.
- **Class:** a categorical stratum. The benchmark has one foreground class,
  so the primary analysis has one class stratum; class is not treated as an
  ordered sixth numeric dimension.
- **Box dimensions:** relative center x, center y, width, and height. Together
  with confidence, these are the five numeric dimensions used by D-ECE.
- **Matching:** stable descending-score order, at most 100 predictions per
  image, same predicted/target class, highest-IoU currently unmatched target,
  no target reuse, and IoU `>= 0.50`. Unmatched emitted predictions are false
  positives.
- **Binning:** equal-width bins on `[0,1]`; the index is
  `floor(value * bin_count)`, with exactly 1.0 clipped into the last bin.
  Internal edges enter the upper bin.
- **Weighting:** for supported cell (b), the contribution is
  ((n_b/N)|\operatorname{precision}(b)-\operatorname{confidence}(b)|), where
  (N) is **all** detections entering calibration, not only detections in
  supported cells.
- **Minimum-cell rule:** the original full-dimensional setting uses five bins
  per numeric dimension and cells with fewer than eight detections contribute
  zero. Unsupported detections remain in (N). A population with no supported
  cell is reported as undefined, not as zero.
- **Confidence floor:** `0.001` is the immutable minimum stored in the frozen
  Phase 5 bundles. Lower-floor sensitivity cannot be recovered from those
  bundles. Higher floors are evaluated descriptively without deleting a run to
  stabilize the metric.

Thus the original one-class grid has (5^5=3,125) possible cells and

\[
\operatorname{D\text{-}ECE}
=\sum_{b:\,n_b\ge 8}\frac{n_b}{N}
\left|\operatorname{precision}(b)-\operatorname{confidence}(b)\right|.
\]

The paper's full-dimensional experiment also used five bins per numeric
dimension and omitted cells below eight observations. The project's 0.001
floor and IoU-0.50 correctness rule are frozen project evaluation choices, not
universal constants in the D-ECE definition. They do not reproduce the paper's
detector demonstration protocol, which used a 0.3 probability threshold, NMS
IoU 0.6, and correctness IoU 0.6 with an additional 0.75 evaluation. The
present implementation is definition-aligned while remaining bound to this
project's frozen prediction/evaluation protocol.

## Primary support and occupancy

The high-dimensional histogram is sparse. The table reports every requested
support diagnostic under the original 5-bin/minimum-8 setting.

| Detector | Run | Detections | Possible cells | Occupied | Supported | Detection fraction supported | Supported-cell size median [min, max] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 17 | 18,260 | 3,125 | 354 | 169 | 0.972 | 28 [8, 1,483] |
| Faster R-CNN | 42 | 7,682 | 3,125 | 298 | 95 | 0.933 | 22 [8, 605] |
| Faster R-CNN | 137 | 25,030 | 3,125 | 318 | 164 | 0.983 | 34 [8, 2,292] |
| Faster R-CNN | 271 | 6,588 | 3,125 | 263 | 84 | 0.933 | 21.5 [8, 629] |
| Faster R-CNN | 314 | 3,663 | 3,125 | 247 | 70 | 0.879 | 18 [8, 252] |
| YOLO11s | 17 | 1,019 | 3,125 | 102 | 29 | 0.819 | 21 [9, 93] |
| YOLO11s | 42 | 773 | 3,125 | 81 | 27 | 0.834 | 17 [8, 84] |
| YOLO11s | 137 | 394 | 3,125 | 101 | 15 | 0.543 | 12 [8, 31] |
| YOLO11s | 271 | 962 | 3,125 | 68 | 21 | 0.906 | 22 [8, 143] |
| YOLO11s | 314 | 905 | 3,125 | 81 | 29 | 0.881 | 19 [8, 80] |

Only 68--354 of 3,125 cells were occupied, and only 15--169 met minimum
support. The most support-sensitive primary run was YOLO11s seed 137: 54.3% of
its detections contributed through supported cells. D-ECE must therefore be
read with occupancy rather than as a support-invariant property.

## Predeclared descriptive sensitivity

The configuration declares equal bin counts `{3, 5, 7}` and minimum-cell
thresholds `{1, 4, 8, 16}`. The original 5-bin/minimum-8 estimate is explicitly
flagged; no grid point selects a preferred detector or replaces it.

Across the full grid, Faster R-CNN's equal-run mean D-ECE ranged from `0.0131`
to `0.0531`, and YOLO11s's ranged from `0.0500` to `0.1551`. Faster R-CNN was
descriptively lower at all 12 common settings, but the absolute estimates fell
as stricter support rules left more detections in cells contributing zero. At
the original setting, the mean supported-detection fractions were 0.940 and
0.797; with seven bins/minimum 16 they were 0.788 and 0.400. This sensitivity
is why the original value is not presented without its support fraction.

The confidence floor materially determines the emitted population. Raising it
from 0.001 to 0.005 retained an equal-run mean of 46.6% of Faster R-CNN
detections and 53.1% of YOLO11s detections; at 0.05 the means were 18.1% and
23.7%. The YOLO11s seed-271 run retained no detection at 0.05, so its D-ECE is
explicitly undefined there. It is not imputed, removed, or converted to zero.
The floor-sensitivity values describe different emitted-detection populations
and are not candidates for choosing the most favorable score floor.

## Reliability-diagram scope

The versioned reliability figure is a **confidence-only marginal** plot. It
uses ten equal-width confidence bins, shows every run uniformly, and overlays a
pooled detection curve for visual context. It is not a visualization of the
five-dimensional D-ECE cells and does not replace the numeric endpoint or its
support diagnostics.

## Population and decision-analysis boundary

A missed ground-truth object has no emitted confidence. It therefore cannot
enter this black-box emitted-detection calibration population. D-ECE evaluates
precision calibration among emitted detections; it does not evaluate missed
target probability, recall calibration, exam-level risk, or downstream harm.

Any valid decision-curve analysis requires a separately defined, validation-
frozen **exam-level outcome probability**. Detection-level D-ECE is neither an
input to nor an output from that calibration. Batch 30 created only a strict
probability-semantic DCA guard because complete run-specific validation
predictions were unavailable; it did not fit exam-level calibrators or produce
standard DCA results.

## Versioned artifacts and reproduction

- [Primary descriptive table](../results/tables/calibration_summary_v2.csv)
- [Per-run support table](../results/tables/calibration_support_v2.csv)
- [Binning, minimum-support, and floor sensitivity table](../results/tables/calibration_sensitivity_v2.csv)
- [Confidence-only marginal reliability figure](../results/figures/reliability_diagrams_confidence_marginal_v2.png)
- [Support/occupancy figure](../results/figures/calibration_support_occupancy_v2.png)
- [Binning/support sensitivity figure](../results/figures/calibration_binning_sensitivity_v2.png)
- [Confidence-floor sensitivity figure](../results/figures/calibration_confidence_floor_sensitivity_v2.png)
- `results/logs/phase33_calibration_support_v2/summary.json`: exact config,
  code, annotation, Phase 5 summary, prediction-bundle, artifact hashes, primary
  cell records, and sensitivity provenance.

From the repository root in the pinned environment:

```powershell
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.calibration --config configs/calibration.yaml --mode run
```

Both commands are CPU-only and reuse frozen prediction bundles. They do not
load a checkpoint, run inference, fit a calibrator, or train a detector.
