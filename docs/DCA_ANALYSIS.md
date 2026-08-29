# Exploratory Raw-Score Threshold Utility/Sensitivity (Non-Standard DCA)

## Status and corrective decision

**The Batch 20 calculation is non-standard and cannot be interpreted as
conventional decision-curve analysis (DCA).** The implementation defined action as
`maximum emitted detector confidence >= tau` and simultaneously used
`tau / (1 - tau)` as the false-positive weight. Although detector confidence is bounded
between zero and one, it is not automatically a predicted exam-level outcome probability
or an elicited decision threshold.

Conventional DCA defines threshold probability as the decision-maker's harm/benefit
trade-off [@vickers2006decisioncurve]. For a continuous diagnostic test or marker, the
marker should first be converted to predicted outcome probability and the same probability
threshold should then define action and false-positive weighting
[@vickers2008decisioncurveextensions]. The original analysis did not perform that mapping.

Batch 30 therefore removed the raw-score curves from the main manuscript Results and
retained the calculation only as an exploratory raw-score threshold utility/sensitivity
artifact. It provides no standard net-benefit, clinical-utility, beneficial-range,
deployment-readiness, or threshold-recommendation evidence.

## Verified historical construction

The historical calculation read ten frozen Phase 5 test bundles: Faster R-CNN and YOLO11s
at seeds 17, 42, 137, 271, and 314. For each exam and run, its marker was exactly the maximum
emitted post-NMS detection score; an exam with no emitted detection received score zero. At
each raw threshold $\tau=0.01,0.02,\ldots,0.99$, the exploratory action was marker
$\geq\tau$. It then calculated

$$
U_{\mathrm{raw}}(\tau)
=\frac{\mathrm{TP}(\tau)}{N}
-\frac{\mathrm{FP}(\tau)}{N}\frac{\tau}{1-\tau}.
$$

Here $U_{\mathrm{raw}}$ is only an arbitrary-scale sensitivity index. Its weighting reuses a
detector-score cutoff; it does not encode an elicited harm/benefit trade-off. TP and FP are
positive and negative **exam images flagged**, not matched boxes. Localization does not
enter this calculation.

The point curve averages five seed-specific values. The preserved uncertainty calculation
uses 2,000 common hierarchical draws that resample 323 NIH patient groups and five seeds;
its percentile limits are pointwise, not simultaneous across 99 raw-score cutoffs. Better
or worse values at a raw cutoff are not conventional DCA evidence, and the two detectors'
raw cutoffs do not represent a common probability scale.

## Why probability-based salvage was not performed

The required salvage route would define an exam-level marker separately for every
detector/run, fit and freeze a probability mapping on validation data only, apply that
mapping once to test, and threshold the calibrated predicted outcome probability using the
same $p_t$ used in the DCA harm weight. Calibrator family and hyperparameters could not be
selected from test behavior.

That route is incomplete for the retained experiment:

| Coverage check | Available |
|---|---:|
| Retained detector/test runs | 10 |
| Runs with frozen validation predictions | 6 |
| Runs missing frozen validation predictions | 4 |

Frozen validation predictions exist for both detectors at seeds 17, 42, and 137. They do
not exist for Faster R-CNN or YOLO11s at seeds 271 and 314. Therefore four retained test
runs cannot receive the required separate, validation-frozen probability mapping. No
calibrator family or hyperparameters were selected, no calibrator was fitted, and test
outcomes were not used to choose one. A partial three-seed salvage was not substituted for
the stated five-seed analysis.

`src/clinical/decision_curve.py` now makes the conventional-DCA input contract explicit.
Its standard path accepts only a typed, validation-frozen outcome-probability object and a
separately typed, elicited decision-threshold probability. It rejects raw detector
confidence or an untyped numeric vector as the predictor and rejects a raw confidence
cutoff or plain scalar passed directly as $p_t$. Regression tests enforce both boundaries.

## Population and prevalence boundary

The historical population is the full held-out RSNA test subset: **750 radiographs from
323 NIH patient groups**, including 169 positive and 581 negative images. Its image-level
prevalence is $169/750=0.225333$ (22.533%). This subset was deliberately stratified and
enriched; it is not a deployment-prevalence sample, a natural patient-level prevalence, or
an external cohort. Even a future probability-valid analysis on this test set would remain
internal to this study population and could not establish external clinical utility or
deployment readiness.

## Preserved and relabeled artifacts

The exact pre-correction arithmetic and files remain hash-audited historical archives:

- `src/clinical/archive/decision_curve_pre_batch30_nonstandard.py`
- `configs/decision_curve_pre_batch30_nonstandard_archive.yaml`
- `results/tables/dca_summary_pre_batch30_nonstandard_archive.csv`
- `results/figures/dca_curves_pre_batch30_nonstandard_archive.png`
- `results/logs/phase20_decision_curve/archive/summary_pre_batch30_nonstandard.json`

The original table has 198 rows (99 cutoffs for each detector). Batch 30 regenerates the
same numerical arithmetic and verifies every historical field before emitting clearly
relabeled artifacts:

- `results/tables/raw_score_threshold_utility_summary.csv`
- `results/figures/raw_score_threshold_utility_sensitivity.png`
- `results/logs/phase30_raw_score_utility/summary.json`

The new table, figure, captions, and provenance explicitly identify the calculation as
non-standard. The original source, config, table, figure, and provenance hashes are
recorded in `configs/raw_score_utility.yaml` and rechecked before generation.

## Reproduction

```powershell
& $benchmarkPython -m src.clinical.raw_score_utility --config configs/raw_score_utility.yaml --mode preflight
& $benchmarkPython -m src.clinical.raw_score_utility --config configs/raw_score_utility.yaml --mode run
```

Preflight verifies all five archives, the complete 750-image/323-patient test grid, all ten
test prediction bundles, and the incomplete six-of-ten validation-prediction coverage. The
run performs no training, checkpoint inference, probability calibration, or standard DCA.

## Method references

- Vickers AJ, Elkin EB. *Decision Curve Analysis: A Novel Method for Evaluating Prediction
  Models.* Medical Decision Making. 2006;26(6):565--574.
  https://doi.org/10.1177/0272989X06295361
- Vickers AJ, Cronin AM, Elkin EB, Gonen M. *Extensions to Decision Curve Analysis, a Novel
  Method for Evaluating Diagnostic Tests, Prediction Models and Molecular Markers.* BMC
  Medical Informatics and Decision Making. 2008;8:53.
  https://doi.org/10.1186/1472-6947-8-53
