# Decision Curve Analysis

## Question and frozen scope

This analysis asks whether acting on an exam-level detector flag has greater net benefit
than the treat-all and treat-none references over nominal threshold probabilities
$\tau=0.01,0.02,\ldots,0.99$. It reads all ten hash-bound Phase 5 test prediction bundles
(Faster R-CNN and YOLO11s, seeds 17, 42, 137, 271, and 314). It performs no training,
checkpoint loading, model inference, threshold fitting, or use of the 300-image robustness
subsample.

The generated evidence is
[`dca_summary.csv`](../results/tables/dca_summary.csv),
[`dca_curves.png`](../results/figures/dca_curves.png), and the exact provenance record in
`results/logs/phase20_decision_curve/summary.json`.

## Population and empirical prevalence

The population is the **full held-out RSNA test split: 750 radiographs from 323 NIH patient
groups**. The committed split manifest contains 169 `Lung Opacity`-positive images and 581
negative images (331 `No Lung Opacity / Not Normal` and 250 `Normal`). The same 169 images
have at least one annotation in the frozen test COCO file, providing an independent
cross-check of the manifest outcome. The empirical exam-level disease prevalence used by
the treat-all calculation is therefore

$$
\hat{\pi}=\frac{169}{750}=0.225333\quad (22.533\%).
$$

This is not a literature prevalence, a patient-level prevalence, or the prevalence of the
separate 300-image/183-patient robustness sample. It describes only the constructed,
stratified held-out test population and should not be transported to another clinical
population.

## Method

Decision-curve analysis requires a binary outcome and a binary action. The unit here is one
radiograph. An image is outcome-positive when it contains at least one lung-opacity
annotation. For each frozen detector/seed bundle, the image is flagged when its maximum
emitted box confidence is at least $\tau$; an image with no emitted box has score zero. Thus
TP and FP below are counts of positive and negative **images flagged**, not matched boxes:

$$
\operatorname{NB}(\tau)
=\frac{\operatorname{TP}(\tau)}{N}
-\frac{\operatorname{FP}(\tau)}{N}\frac{\tau}{1-\tau}.
$$

The point curve is the arithmetic mean of the five seed-specific net benefits. Treat-none
has net benefit zero. Treat-all uses the empirical full-test prevalence:

$$
\operatorname{NB}_{\mathrm{all}}(\tau)
=\hat{\pi}-(1-\hat{\pi})\frac{\tau}{1-\tau}.
$$

The uncertainty analysis reuses the established Phase 8/19 hierarchical resampling helper.
Each of 2,000 common draws samples all 323 NIH patient groups with replacement, moves every
exam from a sampled patient together, and resamples the five frozen training seeds. The
same patient/seed multiplicities are used for both detectors at every threshold. Ribbons
and CSV limits are pointwise two-sided 95% percentile intervals. The CSV also reports the
paired detector difference and its pointwise interval; these are not simultaneous bands
over the 99-threshold grid.

Collapsing detections to an exam flag is appropriate for the retrospective screening and
human-reviewed assistance framing already used in the report. It does not measure whether
the emitted box correctly localizes an opacity, and it is not a utility analysis for
box-guided intervention.

## Results

At the lowest thresholds, treating every exam is competitive because the test prevalence is
22.53%. Treat-all has the largest point estimate from $\tau=0.01$ to 0.03. At 0.04,
Faster R-CNN becomes the largest point-estimate strategy: net benefit is 0.1949 (95% CI
0.1450 to 0.2484), versus 0.0891 (0.0389 to 0.1435) for YOLO11s and 0.1931 for treat-all.
At 0.20 the corresponding values are 0.1185 (0.0603 to 0.1786), 0.0481 (0.0183 to
0.0859), and 0.0317.

The grid-level ranges need two distinct readings:

- **Detector versus detector, by point estimate:** Faster R-CNN is higher from 0.01--0.41;
  YOLO11s is higher from 0.42--0.96; Faster R-CNN is higher only at the isolated 0.97
  point; and both are zero at 0.98--0.99.
- **Paired detector difference:** the 95% interval excludes zero in favor of Faster R-CNN
  from 0.01--0.27. It excludes zero in favor of YOLO11s at 0.60--0.62 and 0.64--0.88.
  The other thresholds do not distinguish the detectors at the pointwise 95% level.
- **Against both reference strategies, by point estimate:** treat-all is largest at
  0.01--0.03, Faster R-CNN at 0.04--0.41, and YOLO11s at 0.42--0.62. Treat-none is largest
  at 0.63--0.84. YOLO11s is nominally largest again at 0.85--0.87, but its net benefit is
  only 0.00053 to 0.00027 and both lower confidence limits equal zero. YOLO11s and
  treat-none tie at zero from 0.88--0.96. Faster R-CNN's isolated 0.97 estimate is 0.0016
  with a lower limit of zero; all non-treat-all strategies are zero at 0.98--0.99.

The high-threshold paired advantage for YOLO11s must therefore not be reported as uniformly
positive clinical utility. Across 0.64--0.84, YOLO11s is statistically less harmful than
Faster R-CNN in the paired comparison while still having negative net benefit; treat-none
is the better strategy.

## Clinical-relevance interpretation

For the report's **high-sensitivity retrospective screening or server-side case
prioritization** scenario, the decision curves support the existing preference for Faster
R-CNN over the practically informative low-to-middle threshold region. Its point net
benefit is the best strategy from 0.04--0.41, and its paired advantage over YOLO11s is
supported through 0.27. At extremely low thresholds (0.01--0.03), however, this constructed
test population favors treating or reviewing every exam rather than using either detector.

For **resource-constrained point-of-care assistance in which a human reviews every image**,
YOLO11s has the larger detector net benefit from 0.42--0.96, but the defensible positive
range is narrower: it is the best point-estimate strategy from 0.42--0.62, while the paired
difference becomes distinguishable only at 0.60--0.62 in that positive-benefit band. This
does not turn YOLO11s into a sole triage gate or rule-out system. It remains the conditional
compute-driven option already described in the report, and only when every image receives
human review.

These threshold values are **raw detector confidences used as nominal threshold
probabilities**, not validated clinical-risk probabilities. Batch 18 found material
detection-calibration error and a marked YOLO11s seed-271 score-scale pathology; no
validation-fitted risk calibration map was learned here. The same held-out test data provide
the prevalence and evaluate the curves, and the test subset was deliberately stratified.
Consequently, this DCA is a retrospective, internal decision-analytic description of the
frozen benchmark, not evidence of prospective benefit, safety, transportability, or a
clinically recommended threshold.

## Reproduction

```powershell
& $benchmarkPython -m src.clinical.decision_curve --config configs/decision_curve.yaml --mode preflight
& $benchmarkPython -m src.clinical.decision_curve --config configs/decision_curve.yaml --mode run
```

Preflight verifies the complete full-test image/patient grid, manifest-versus-COCO outcome
agreement, all ten Phase 5 bundle identities and hashes, the five-seed factorial grid, and
the frozen 0.001 score floor before any result artifact is written.
