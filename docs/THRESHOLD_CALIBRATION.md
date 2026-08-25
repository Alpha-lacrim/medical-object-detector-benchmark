# Patient-Cluster-Aware Threshold Calibration

## Relationship to the primary operating points

Decision [D-004](DECISION_LOG.md#d-004--keep-cost-weighted-threshold-calibration-as-a-separate-sensitivity-analysis)
states that this analysis **stands alongside and does not supersede** the
validation-selected operating points in [THRESHOLD_ANALYSIS.md](THRESHOLD_ANALYSIS.md).
The Batch 14 thresholds—0.69 for Faster R-CNN and 0.05 for YOLO11s—remain the
authoritative primary single-threshold results and the inputs to the existing FROC and
Pareto analyses. The results below are a distinct cost-sensitivity extension. No beta is
adopted as a clinically definitive setting, and none is applied retrospectively to the test
set or to another downstream artifact.

This separation matters even at beta 1. Batch 14 maximizes the point estimate of arithmetic
mean F1, whereas this analysis maximizes a lower bootstrap confidence bound. Agreement or
disagreement between those rules does not silently change which artifact is primary.

## Mathematical objective

At confidence threshold $\tau$, let $P(\tau)$ and $R(\tau)$ denote precision and recall under
the canonical same-class, score-ordered matcher at IoU 0.50. The cost-weighted score is

$$
F1_\beta(\tau) =
\frac{(1+\beta^2)P(\tau)R(\tau)}{\beta^2P(\tau)+R(\tau)},
\qquad
\beta = \sqrt{\frac{C_{FN}}{C_{FP}}}.
$$

Thus beta values 1, 3, 5, and 10 correspond to assumed false-negative/false-positive cost
ratios of 1, 9, 25, and 100. These ratios are scenario assumptions, not quantities estimated
from this dataset. The complete sweep is reported to expose that dependence.

## Method

The analysis is validation-only and CPU-only. It reads the six immutable validation
prediction bundles created for Batch 14: Faster R-CNN and YOLO11s at seeds 17, 42, and 137.
The bundles contain 750 validation images from 321 NIH patient groups. Their hashes,
annotation identity, evaluator settings, detector/seed grid, and the archived Phase 14
configuration are verified before use. This batch loads no checkpoint, performs no model
inference, accesses no test prediction, and retrains nothing.

For each detector, the outer loop evaluates 99 thresholds from 0.01 through 0.99 in steps of
0.01. Predictions are capped at the frozen maximum of 100 per image and passed through the
same operating-point matcher used by the unified evaluator. Within each seed, TP, FP, and FN
are micro-aggregated over the resampled validation images; precision, recall, and
$F1_\beta$ are then calculated. The detector point estimate is the arithmetic mean of the
three seed-specific metrics, matching Batch 14's across-seed reduction.

The inner uncertainty analysis uses 2,000 deterministic draws from the shared Phase 8
hierarchical resampling helper in `src/stats/paired.py`. Each draw samples the 321 NIH patient
groups with replacement, moves every repeated exam from a patient together, and independently
resamples the three frozen training seeds with replacement. The same random draws are reused
at every threshold and beta to avoid Monte Carlo noise from different resamples changing the
argmax. The reported interval is the two-sided 95% percentile interval of the bootstrap
mean-across-seed $F1_\beta$ distribution.

For each detector and beta, $\tau^*$ is the swept threshold with the largest lower 95%
confidence bound. Exact lower-bound ties are resolved toward the higher, more selective
threshold. These are pointwise intervals, not a simultaneous confidence band over all 99
candidate thresholds.

## Finding

| Detector | Beta (cost ratio) | Selected threshold | Precision | Recall | $F1_\beta$ (95% CI) |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 1 (1:1) | 0.69 | 0.4164 | 0.4404 | 0.4202 (0.3322–0.4732) |
| Faster R-CNN | 3 (9:1) | 0.33 | 0.1958 | 0.6534 | 0.5247 (0.4342–0.5803) |
| Faster R-CNN | 5 (25:1) | 0.12 | 0.1247 | 0.7413 | 0.6136 (0.5359–0.6634) |
| Faster R-CNN | 10 (100:1) | 0.03 | 0.0753 | 0.8207 | 0.7374 (0.6781–0.7790) |
| YOLO11s | 1 (1:1) | 0.02 | 0.3492 | 0.3670 | 0.3544 (0.2788–0.4195) |
| YOLO11s | 3 (9:1) | 0.01 | 0.3025 | 0.3971 | 0.3823 (0.2879–0.4579) |
| YOLO11s | 5 (25:1) | 0.01 | 0.3025 | 0.3971 | 0.3912 (0.2877–0.4765) |
| YOLO11s | 10 (100:1) | 0.01 | 0.3025 | 0.3971 | 0.3956 (0.2884–0.4860) |

Faster R-CNN shows the expected monotone operating shift: as false negatives receive more
assumed weight, the conservative-LCB optimum moves from 0.69 to 0.33, 0.12, and 0.03,
trading precision for recall. At beta 1 it exactly reproduces the Batch 14 primary threshold,
although D-004 still keeps the two selection rules conceptually separate.

YOLO11s selects 0.02 at beta 1 and reaches the predeclared lower sweep boundary of 0.01 by
beta 3. The beta 3, 5, and 10 results are therefore the best observed values in the requested
grid, not demonstrated interior or global optima. This boundary behavior reinforces the
previous score-scale/coverage finding: increasing the assumed false-negative cost demands
lower thresholds, but YOLO11s cannot move farther within the declared range. It does not
establish a clinically preferable cost ratio or deployment setting.

## Scope and limitations

- The cost ratios are hypothetical. No patient outcome, treatment utility, reader study, or
  clinical harm valuation was used to estimate them.
- Selection uses three historical validation seeds. The bootstrap includes seed resampling,
  but three seed identities still provide a coarse view of training variability.
- Patient-cluster resampling addresses repeated exams within the observed validation cohort;
  it does not establish external-site transportability or prospective clinical utility.
- Optimizing a pointwise lower interval over 99 thresholds is a conservative selection rule,
  not a simultaneous 95% guarantee for the selected maximum.
- The YOLO11s beta 3–10 optima lie at 0.01. Behavior below the predeclared sweep is not
  extrapolated even though the frozen bundles retain lower-scored predictions.
- These validation-only thresholds were not applied to test. The table describes threshold
  sensitivity, not final held-out performance, and does not replace the test results already
  reported for Batch 14's primary thresholds.

## Artifacts and reproduction

- `results/tables/threshold_calibration_summary.csv` contains the eight selected
  detector/beta rows, cost ratios, point estimates, confidence bounds, boundary status, and
  D-004 relationship.
- `results/figures/threshold_calibration_sensitivity.png` plots the lower confidence-bound
  curve over all 99 thresholds for every beta, with selected points and the Batch 14 primary
  thresholds marked.
- `results/logs/phase19_threshold_calibration/summary.json` records the complete threshold
  curves, all frozen input hashes, the resampling contract, source identities, and output
  hashes.

From the repository root in the pinned Python 3.11 environment:

```powershell
& $benchmarkPython -m src.stats.threshold_calibration --config configs/threshold_calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.threshold_calibration --config configs/threshold_calibration.yaml --mode run
```
