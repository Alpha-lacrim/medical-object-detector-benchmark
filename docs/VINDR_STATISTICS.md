# Frozen VinDr statistics and cross-dataset transportability

Batch 50 implements the statistical plan already frozen in
[protocol v1, section 7](VINDR_EXTERNAL_PROTOCOL.md#7-observational-unit-aggregation-and-uncertainty).
The operational configuration is
[`vindr_statistics_v1.yaml`](../configs/vindr_statistics_v1.yaml).
It inherits scientific settings from the unchanged Batch 47 protocol/config;
it does not select an ontology, confidence threshold, candidate floor or model.
Results and the three figures are in
[VINDR_STATISTICS_RESULTS.md](VINDR_STATISTICS_RESULTS.md).

## Estimands and conditioning

For dataset D and detector d, the primary target is the equal-run expected
cohort metric of the disclosed training procedure on D, approximated by its
five retained trained runs. The estimand is conditional on the fixed training
data and training recipe, checkpoint selection rule, dataset-specific target
annotations, retained prediction support, matching/evaluation implementation,
and the already selected RSNA thresholds. It is neither an ensemble metric
nor the average of image-level APs, ratios or detection probabilities.

Within each dataset, the contrast is Faster R-CNN minus YOLO11s. These are
separate dataset-specific estimands; neither the cohorts nor their predictions
are pooled. Cross-dataset differences in those contrasts are described, without
a new interaction test or a confidence interval for a difference of differences.
The shared checkpoints across datasets are therefore not treated as independent
retraining realizations in a cross-dataset test.

| Endpoint | Population and metric | Inference |
| --- | --- | --- |
| AP@0.50, AP@0.50:0.95 | Dataset-specific target boxes; official COCO bbox matches, all areas, 101 recall positions, maxDets [1,10,100], common score support >=0.001 | Primary training-procedure and secondary fixed seed-17 intervals |
| FROC sensitivity at 0.125/0.25/0.5/1/2 FP per image | All retained candidates at 0.00001; exact-score greedy IoU-0.50 matching, tied scores enter together; all sampled images in FP denominator and all sampled targets in sensitivity denominator | Same two estimands, conditional on observed floor/cap support |
| Historical RSNA n=3 threshold precision, recall, F1, FP/image | Unchanged 0.69 Faster R-CNN and 0.05 YOLO11s thresholds on all five runs | Same two estimands; threshold-selection uncertainty is not resampled |
| Batch 43 post-hoc n=5 policy | Unchanged 0.70 Faster R-CNN and 0.01 YOLO11s thresholds on all five runs | Descriptive sensitivity only; no secondary-policy bootstrap interval |
| Confidence distributions and detection counts | Per-run emitted detections on each frozen support; equal-run summaries with defined-run counts | Descriptive; no detection-level confidence interval or clinical probability interpretation |

The seed-17 secondary target holds those two checkpoints fixed. Its uncertainty
comes only from sampled observations and cannot substitute for uncertainty
over a training procedure. Five trained runs remain a coarse empirical estimate
of run variability, conditional on one fixed training dataset; the bootstrap
does not resample or retrain on training patients.

## Observation units and joint resampling

The VinDr annotation CSVs lack a patient key, and the Batch 47/48 audit found
no nonempty PatientID, StudyInstanceUID, SeriesInstanceUID or SOPInstanceUID
in any of the 3,000 released headers. No patient groups are inferred from
image names, hashes, apparent similarity or demographics. The unit is the
**released study/image**, not a claimed independent patient. Unknown repeated
patients may make image-based intervals too narrow.

The RSNA comparison retains its 323 known NIH patient groups across 750 test
studies. Every study from a sampled patient receives the same multiplicity;
the total number of studies can vary between patient bootstrap draws. FP/image
therefore divides by the *sampled image total*, not the original 750 or the
patient count. Internal and external observations are sampled separately.

For each dataset and each of the 2,000 draws:

1. Sample the original number of observation units with replacement. Use
   known patients internally and released images externally. Share this one
   draw across both detectors and all their runs.
2. Independently sample five trained runs with replacement within each
   detector. Reuse those run multiplicities across endpoints in that draw.
   Same-number seeds are not coupled stochastic blocks.
3. Recompute each run's cohort-level AP, FROC and historical-threshold ratios
   from the sampled observation multiplicities. Only then average runs using
   their sampled multiplicities, with equal weight per sampled run.
4. Form the within-dataset detector contrast. For the secondary sensitivity,
   retain only the seed-17 metric on the same observation draw and hold both
   checkpoints fixed.

This reuses the project's existing independent-run draw helper, NIH patient
draw helper, stable RNG derivation, operating-ratio conventions and percentile
interval helper in `src/stats/paired.py`. Base seed is 20260811. The existing
SHA-256-based `stable_rng_seed` derives separate streams from
`batch50-cross-dataset-transportability-v1:internal` and
`batch50-cross-dataset-transportability-v1:external`; the resulting integer
seeds are recorded. Within each draw the observation draw precedes the two
independent detector-run draws. Endpoint evaluation consumes no randomness.

Percentile intervals use the 2.5th and 97.5th percentiles, with linear
quantiles. They are marginal, not simultaneous or familywise confirmatory
intervals. There are no new p-values or hypothesis-test families. Undefined
replicates are recorded, never retried. Intervals use finite draws, with valid
and undefined counts per detector and contrast; no finite draws yields NA.
An undefined selected run propagates rather than being discarded. With targets
present, no predictions gives defined zero AP/precision/recall/F1, as in the
frozen evaluator. AP and FROC are undefined if a draw contains no target boxes.

## Exact reconstruction and technical verification

[`transportability.py`](../src/stats/transportability.py) adds a numerical
cache for resampling, not a competing metric definition. Official pycocotools
per-image COCO matches are cached at every configured IoU. AP is reconstructed
from score-ordered true-positive ranks and the official recall grid. Integer
multiplicities are equivalent to distinct COCO image copies, with canonical
source-image order, then copy order, then stable within-image score ties.
No sampled copy is deduplicated. Exact within-image score ties and COCO's
target-IoU tie behavior are preserved. This extension leaves historical
Phase 8 code, estimates and bootstrap artifacts unchanged.

FROC caches use the existing canonical greedy matcher. Image weights apply
to both matched and unmatched detections. Every tied-score block enters
atomically; no fractional block, interpolation or extrapolation can satisfy
a budget. The reported sensitivity is the maximum observed sensitivity at
or below each fixed budget. Its value agrees with the existing selector,
including its fewer-FP/higher-score tie-breakers. No score coordinate from
this calculation is exported as an external operating rule.

Synthetic tests compare cached AP with actual pycocotools evaluation after
creating distinct image copies, and cached FROC with the existing exact-score
curve evaluated on those copies. They cover repeated and omitted images,
cross-image and within-image score ties, competing equal-IoU targets, negatives,
caps, threshold equality, empty predictions, no-target draws, invalid weights,
unit sharing, known patient clusters, independent detector-run draws, and
undefined intervals. Before bootstrapping, every observed endpoint is replayed
against its canonical internal or Batch 49 source. This is metric verification,
not a new GPU inference run.

## Internal comparison and support limits

Internal AP, historical and secondary threshold metrics, and common-AP-support
scores use the original Phase 5 bundles. Internal lower-floor FROC and the
all-retained score support use Batch 42 v4 bundles. External endpoints use
Batch 49 bundles. Source hashes and checkpoint identity are checked across
these families; original internal artifacts are preserved byte for byte.

Batch 50 adds aligned *internal* patient/run intervals for the same AP, FROC,
and historical-threshold endpoints to make the comparison interpretable.
They use the already established internal inferential principles. Historical
Phase 8 intervals concern their original endpoints and RNG streams; these new
intervals are explicitly separate and do not overwrite them. Both n=3 and n=5
refer to validation **selection** counts; test evaluation always includes
five runs per detector in each dataset.

Score distributions are separate per-run ECDFs at the common AP floor.
Arithmetic score means and quantiles first describe each emitted population;
equal-run means/SDs then weight runs equally, with explicit contributing-run
counts. Empty populations have count zero and undefined score quantiles.
Counts per image accompany raw counts because the cohorts differ in size.
The score analysis is conditional on emission and does not establish
probability calibration. Missing targets have no emitted score.

An observed FROC point beyond a run's candidate-floor endpoint remains a
lower bound. The interval resamples that observed lower-bound estimand; it
does not resolve missing prediction support. Conservative upper bounds replace
each limited run's contribution with 1.0, preserving other runs. These are
missing-support bounds, not bootstrap confidence bounds. Detection-cap
saturation is reported independently and is not repaired by this bound.
No floor or cap is changed after external results.

Change labels are descriptive: unchanged within the frozen numerical tolerance,
strengthened for an increased same-sign absolute A-minus-B gap, weakened for
a smaller same-sign gap, and reversed for a sign reversal. A tie transition
is explicitly identified in the ordering column. Score and detection-count
contrasts have no intrinsic preferred direction; lower FP/image must be read
alongside recall. Neither an unchanged rank nor a narrow interval implies
successful absolute performance transport or clinical equivalence.

Historical YOLO internal accuracy used FP32 inference; new VinDr inference
uses mandatory verified bfloat16 AMP. Dataset annotations, acquisition,
institution, country, population and implementation differences are confounded.
Use **cross-dataset transportability**, without attributing differences
specifically to any one factor. No clinical utility, diagnostic validation,
architecture-family causal effect or deployment recommendation follows.

## Artifacts and replay

The public `results/vindr_external_v1/statistics/summary.json` binds code,
operational and frozen configs, inference receipts, annotations, patient map,
prediction bundles, source summaries, checkpoints, bootstrap arrays and figure
outputs by SHA-256. Generated text uses explicit LF bytes. Only aggregate
nonidentifying tables and three figures are public. Restricted annotations,
IDs and image-linked predictions remain in the existing ignored data tree;
plot working data and bootstrap arrays also remain there.

The runner's `preflight` verifies all frozen files and complete ten-run
provenance; the Batch 49 read-only verifier separately replays full source
integrity and inference metrics. `verify` checks every hash and replays
aggregations and intervals from saved draws. `replay` additionally reloads
prediction bundles and deterministically recomputes all 2,000 draws in both
datasets. `render` regenerates the three figures and review document from
hash-verified evidence, then verifies their identities. `run` refuses to
overwrite existing completed statistics or private draws. Exact commands are
in [README](../README.md#vindr-statistics-and-cross-dataset-transportability-batch-50).

The manuscript remains unchanged. Batch 51 integration requires its own
user request after the Batch 50 review checkpoint.

### Preserved rendering-order correction

The initial completed analysis passed full bootstrap replay. A subsequent
`render` check caught a report-only ordering difference: sorted JSON keys
changed the display order of the two cohort rows. The original report was
recovered exactly with the original renderer and its recorded hash verified.
That report, the reordered rendering, all initial public/private outputs and
the implementation/config were preserved under ignored
`superseded/batch50_report_order_fix/`, with an archive hash manifest. Only
the report's iteration order was corrected. A regression test checks JSON
object-order invariance. Regeneration must preserve every numerical summary
and bootstrap draw from the initial analysis exactly; no scientific setting
was amended.
