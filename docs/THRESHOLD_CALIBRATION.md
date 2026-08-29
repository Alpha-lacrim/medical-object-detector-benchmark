# Recall-Weighted F-beta Threshold Sensitivity

## Relationship to the primary operating points

Decision [D-006](DECISION_LOG.md#d-006--treat-beta-as-an-f-beta-preference-parameter-and-separate-linear-loss)
corrects D-004's interpretation while preserving its precedence rule. The
Batch 14 validation-selected thresholds—0.69 for Faster R-CNN and 0.05 for
YOLO11s—remain the authoritative primary single-threshold results and the
inputs to the existing FROC and Pareto analyses. The analyses below are
validation-only sensitivity descriptions. They do not replace the primary
thresholds, and none of their thresholds is applied to test or another
downstream artifact.

This separation matters even at beta 1. Batch 14 maximizes the point estimate
of arithmetic mean F1, whereas the F-beta analysis maximizes a pointwise lower
bootstrap bound. Agreement or disagreement between those rules does not change
which artifact is primary.

## What beta does

At confidence threshold $\tau$, let $P(\tau)$ and $R(\tau)$ denote precision
and recall under the canonical same-class, score-ordered matcher at IoU 0.50.
The implemented objective is

$$
F_\beta(\tau)=
\frac{(1+\beta^2)P(\tau)R(\tau)}
{\beta^2P(\tau)+R(\tau)}.
$$

This is a weighted harmonic mean: beta 1 weights precision and recall equally,
and beta greater than 1 places more preference on recall. In this harmonic
mean, $\beta^2$ is the relative recall weight. The retained beta values
1, 3, 5, and 10 therefore produce relative recall weights 1, 9, 25, and 100.
They are hypothetical preference settings frozen before test evaluation, not
quantities estimated from patient outcomes.

Substituting $P=TP/(TP+FP)$ and $R=TP/(TP+FN)$ gives the exact implemented
count form:

$$
F_\beta(\tau)=
\frac{(1+\beta^2)TP(\tau)}
{(1+\beta^2)TP(\tau)+\beta^2FN(\tau)+FP(\tau)}.
$$

The appearance of weighted FN and FP terms in this denominator does not turn
the ratio into a linear clinical-harm function: TP also appears in the
numerator and denominator, and the optimum depends on the achieved F-beta
level. The repository contains no citation, clinical outcome valuation, or
derivation supporting the former identity
$\beta=\sqrt{C_{FN}/C_{FP}}$. It is therefore not used in the canonical
analysis.

## Validation-only method

The CPU-only workflow reads the six immutable validation prediction bundles
created for Batch 14: Faster R-CNN and YOLO11s at seeds 17, 42, and 137. The
bundles contain 750 validation images from 321 NIH patient groups. Before use,
the runner verifies the selection config and manifest hashes, the declared
model-development validation role, the upstream validation split, a false
test-access flag, exact agreement between annotation and validation-manifest
image identities, evaluator settings, and the detector/seed grid. It loads no
checkpoint, performs no inference, accesses no test label or prediction, and
retrains nothing.

For each detector, 99 thresholds from 0.01 through 0.99 are evaluated through
the shared matcher. Within each seed, TP, FP, and FN are micro-aggregated over
the resampled validation images. Precision, recall, and $F_\beta$ are computed
per seed and then averaged over the three seed-specific metrics, matching the
Batch 14 reduction.

The uncertainty analysis uses 2,000 deterministic hierarchical draws. Each
draw samples 321 NIH patient groups with replacement, moves every repeated exam
from a patient together, and resamples the three frozen training seeds with
replacement. The original Batch 19 random stream is explicitly retained so
the terminology correction cannot change a historical selected threshold. The
same draws are used at every threshold and beta. The reported interval is the
two-sided 95% percentile interval of mean-across-seed $F_\beta$.

For each detector and beta, the canonical sensitivity threshold is the
candidate with the largest lower pointwise 95% bound; exact ties favor the
higher threshold. These are pointwise intervals, not a simultaneous band.

## F-beta findings and threshold stability

The selected thresholds and original F-beta estimates are unchanged. The new
stability diagnostics are descriptive:

- The **near-optimal plateau** is the contiguous grid interval containing the
  canonical threshold whose lower-bound objective stays within an absolute
  0.01 of its maximum.
- For each bootstrap draw, a separate diagnostic selects the threshold with
  the largest draw-specific mean $F_\beta$, with the same high-threshold tie
  break. Its distribution and candidate frequencies describe sampling
  sensitivity; they do not replace the lower-bound selection rule.

| Detector | Beta (recall weight $\beta^2$) | Canonical $\tau$ | Near-optimal plateau (width) | Bootstrap selected $\tau$, median (95% interval) | Modal $\tau$ (frequency) |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 1 (1) | 0.69 | 0.64–0.70 (0.06) | 0.69 (0.63–0.75) | 0.70 (28.50%) |
| Faster R-CNN | 3 (9) | 0.33 | 0.27–0.39 (0.12) | 0.35 (0.13–0.51) | 0.38 (16.35%) |
| Faster R-CNN | 5 (25) | 0.12 | 0.07–0.16 (0.09) | 0.14 (0.04–0.29) | 0.23 (15.25%) |
| Faster R-CNN | 10 (100) | 0.03 | 0.03–0.04 (0.01) | 0.03 (0.01–0.09) | 0.03 (49.70%) |
| YOLO11s | 1 (1) | 0.02 | 0.01–0.05 (0.04) | 0.05 (0.01–0.10) | 0.02 (20.90%) |
| YOLO11s | 3 (9) | 0.01 | 0.01–0.01 (0.00) | 0.01 (0.01–0.01) | 0.01 (99.70%) |
| YOLO11s | 5 (25) | 0.01 | 0.01–0.01 (0.00) | 0.01 (0.01–0.01) | 0.01 (100.00%) |
| YOLO11s | 10 (100) | 0.01 | 0.01–0.01 (0.00) | 0.01 (0.01–0.01) | 0.01 (100.00%) |

The wide Faster R-CNN beta-3 and beta-5 bootstrap distributions show that a
single grid argmax is sample-sensitive even though the canonical thresholds
are reproducible under the frozen rule. YOLO11s beta 3–10 remains pinned to the
0.01 lower boundary, so the apparent stability at those settings is boundary
stability rather than evidence of an interior or global optimum.

The corresponding validation precision, recall, and F-beta values remain:

| Detector | Beta | Precision | Recall | $F_\beta$ (95% pointwise CI) |
|---|---:|---:|---:|---:|
| Faster R-CNN | 1 | 0.4164 | 0.4404 | 0.4202 (0.3322–0.4732) |
| Faster R-CNN | 3 | 0.1958 | 0.6534 | 0.5247 (0.4342–0.5803) |
| Faster R-CNN | 5 | 0.1247 | 0.7413 | 0.6136 (0.5359–0.6634) |
| Faster R-CNN | 10 | 0.0753 | 0.8207 | 0.7374 (0.6781–0.7790) |
| YOLO11s | 1 | 0.3492 | 0.3670 | 0.3544 (0.2788–0.4195) |
| YOLO11s | 3 | 0.3025 | 0.3971 | 0.3823 (0.2879–0.4579) |
| YOLO11s | 5 | 0.3025 | 0.3971 | 0.3912 (0.2877–0.4765) |
| YOLO11s | 10 | 0.3025 | 0.3971 | 0.3956 (0.2884–0.4860) |

## Separate hypothetical detection-error loss

To show what a genuinely linear weighted-error objective looks like, a separate
validation-only analysis minimizes

$$
L(\tau;r)=r\frac{FN(\tau)}{N}+\frac{FP(\tau)}{N},
$$

where $N$ is the number of validation images and the assumed FN:FP loss ratios
are $r\in\{1,9,25,100\}$. Each unmatched annotated target receives penalty
$r$ and each false-positive detection receives penalty 1. These are explicit,
hypothetical detection-error penalties. They are not measured clinical harms,
do not include downstream actions or outcomes, and do not establish deployment
utility. Selection minimizes the three-seed validation mean; exact ties favor
the higher threshold. Bootstrap intervals are pointwise descriptions.

| Detector | Assumed $r$ | Selected $\tau$ | Mean validation loss per image (95% CI) |
|---|---:|---:|---:|
| Faster R-CNN | 1 | 0.87 | 0.3596 (0.2479–0.4673) |
| Faster R-CNN | 9 | 0.61 | 2.0040 (1.5320–2.4036) |
| Faster R-CNN | 25 | 0.23 | 4.2058 (3.2933–5.0022) |
| Faster R-CNN | 100 | 0.04 | 10.9031 (8.1162–13.4461) |
| YOLO11s | 1 | 0.35 | 0.3529 (0.2421–0.4682) |
| YOLO11s | 9 | 0.01 | 2.3604 (1.6511–3.0451) |
| YOLO11s | 25 | 0.01 | 5.9231 (4.2415–7.5682) |
| YOLO11s | 100 | 0.01 | 22.6231 (16.2481–29.0648) |

The different beta-1 and r-1 thresholds—0.69 versus 0.87 for Faster R-CNN and
0.02 versus 0.35 for YOLO11s—are direct empirical evidence that F-beta
optimization and linear error-loss minimization are different objectives.
None of these loss-selected thresholds is applied to test or adopted as a
deployment rule.

## Scope and limitations

- Both sensitivity analyses use only three historical validation seeds; seed
  variability remains coarsely characterized.
- Patient-cluster resampling addresses repeat exams in this cohort but does not
  establish external transportability or prospective utility.
- The lower-bound F-beta selection uses pointwise intervals over 99 candidates,
  not a simultaneous guarantee for the selected maximum.
- The 0.01 plateau tolerance is a declared descriptive convention, not a new
  tuning criterion.
- Draw-specific bootstrap argmax frequencies estimate selection instability
  under the resampling model; they are not probabilities that a threshold is
  clinically optimal.
- The hypothetical loss treats every missed target and every false-positive
  box as exchangeable within its type. It omits severity, patient consequences,
  reader workflow, treatment effects, and prevalence transport.

## Artifacts and reproduction

- `results/tables/recall_weighted_fbeta_threshold_summary.csv` contains the
  eight canonical F-beta sensitivity rows and plateau/distribution summaries.
- `results/tables/recall_weighted_fbeta_threshold_stability.csv` contains all
  792 candidate-threshold selection counts and frequencies.
- `results/tables/hypothetical_detection_error_loss_summary.csv` contains the
  separate eight-row linear loss sensitivity.
- `results/figures/recall_weighted_fbeta_threshold_sensitivity.png` plots the
  lower confidence-bound curves with recall-weight labels.
- `results/logs/phase29_threshold_sensitivity/summary.json` records the full
  F-beta, stability, and hypothetical-loss curves plus input/output hashes.
- The exact pre-Batch-29 table, figure, and provenance are retained under
  explicit `threshold_calibration_pre_batch29_archive` and
  `phase19_threshold_calibration/archive` paths. Their former clinical-cost
  interpretation is superseded and must not be cited as canonical.

From the repository root in the pinned Python 3.11 environment:

```powershell
& $benchmarkPython -m src.stats.threshold_calibration --config configs/threshold_calibration.yaml --mode preflight
& $benchmarkPython -m src.stats.threshold_calibration --config configs/threshold_calibration.yaml --mode run
```
