# Frozen VinDr-CXR inference — Batch 49

The user explicitly requested Batch 49 on 2026-09-23 following the completed
Batch 48 review checkpoint. This authorizes the frozen inference experiment;
it does not authorize Batch 50 statistics or a manuscript rewrite.

The scientific contract remains the byte-bound
[v1 protocol](VINDR_EXTERNAL_PROTOCOL.md) and
[scientific config](../configs/vindr_external_v1.yaml). The new
[operational config](../configs/vindr_inference_v1.yaml) supplies filenames,
progress reporting, authorization provenance and the intended GPU identity.
The exact commands are in [README](../README.md#vindr-cxr-frozen-external-testing-batches-4749).

**Completed and verified.** All ten checkpoints finished on the intended GPU
on 2026-09-23. After the conversation interruption, the complete generated
receipt was recovered without rerunning inference or editing any result.
The full read-only verifier passed all ten runs and all 3,000 images per run,
including every metric and saved raw-boundary correction. Public summary
SHA-256: `d15070ebcafdfb2711b07db5090e67ef556b60d0042c87ac047b243bdf8fc656`.
[Generated per-run tables](VINDR_INFERENCE_RESULTS.md) expose all endpoints,
transport policies, support limits, score summaries and execution records.

## Observed findings and adverse results

Both pipelines show severe external performance collapse under this strict
annotation ontology. Equal-run mean AP@0.50 is 0.002505 for Faster R-CNN and
0.000501 for YOLO11s; mean AP@0.50:0.95 is 0.000618 and 0.000119. Faster R-CNN
remains higher on these observed aggregate AP endpoints and all five observed
mean FROC sensitivities; there is no observed aggregate AP/FROC rank reversal.
This does not establish ordering beyond the retained candidate support or
identify a causal reason for the external performance change.

Historical n=3 RSNA thresholds transport poorly: mean precision/recall/F1 are
0.010830/0.018947/0.013580 for Faster R-CNN and
0.004271/0.004211/0.004214 for YOLO11s. Thus mean box recall is only 1.89% and
0.42%. Mean emitting-image percentages are 4.26% and 2.14%. Poor threshold
transport is a descriptive characterization of these near-zero values; no
formal success/failure cutoff was prespecified. This is not a clinical
deployment assessment or an identical-task validation result.

YOLO seed 271 has maximum emitted confidence 0.03857421875, below its historical
0.05 threshold, so it emits zero detections there. At the separately labeled
post-hoc n=5 threshold of 0.01 it emits 54 detections across 45 images, all
unmatched to the strict target (zero TP, 54 FP, 95 FN). It remains in every
defined endpoint and aggregate. Faster R-CNN seed 271 remains in all results.

The secondary n=5 thresholds do not rescue performance: mean recall stays
0.018947 for Faster R-CNN and becomes 0.010526 for YOLO11s. The observed
FP/image ordering does reverse across policies: historical thresholds give
0.047667 versus 0.027067 (Faster R-CNN versus YOLO), while the secondary policy
gives 0.044200 versus 0.060067. These are two transported RSNA policies, not
external optimization or evidence to replace the primary policy.

YOLO seed 137's candidate-floor endpoint is 0.793333 FP/image, so its 1 and
2 FP/image contributions remain lower-bound observations. Faster R-CNN cap
saturation is substantial: 3,000, 2,635, 3,000, 3,000 and 209 images for seeds
17/42/137/271/314 respectively. No YOLO image reaches the cap. The observed
FROC is conditional on these fixed support limits; neither floor nor cap was
changed after results. Per-run overlap is visible in the tables; no paired-seed
or complete run-cloud dominance claim is made. Bootstrap uncertainty and
internal/external synthesis remain Batch 50.

## Prerequisite and implementation checks

Before inference the read-only gate verifies all 38 frozen definitions, all
ten checkpoint sizes/hashes and training configs, the completed Batch 48
receipt and code/config bindings, the official release metadata/inventory,
all 3,000 DICOM and PNG hashes, source-to-COCO equality and complete common
loader decoding/dimensions. The population remains 3,000 released images,
84 strict-target positive images, 95 boxes and 2,916 strict-target negatives.
There is no defensible patient grouping.

The adapter suite passed 71 tests with one expected metadata-only skip.
The inference/exact-FROC/timing suite passed 48 tests. All 12 new inference
tests also passed separately in the pinned CUDA environment. Existing manuscript
verification passed 66 numerical claims and semantic guards; existing
scientific verification passed 72 artifacts, 346 present inputs, no missing
inputs and 201 result references. These checks precede the new experiment.

Synthetic native-filter tests establish that both Torchvision ROI heads and
Ultralytics NMS filter with strict `score > floor`. A candidate exactly equal
to the native cutoff is not emitted. The common evaluator uses `>=` on
retained candidates. The configured floor remains 0.00001; no nextafter
adjustment is applied in real inference. The synthetic test itself uses
neighboring values only to establish the API boundary.

The initial synthetic development test caught a wrapper error: the existing
FROC selection/aggregation helper requires at least two runs for sample SD.
The new single-run wrapper now applies the identical frozen selection key
and budget tolerance directly. A regression test verifies equality to the
five-run helper. This correction occurred before any external model was
loaded or predictions/results existed. No scientific choice or historical
artifact changed.

Inference reuses Batch 45's `Predictor`, explicit CUDA autocast and
`quantize=None` path. Each checkpoint must pass an actual convolution-dtype
check before its full run: float16 for Faster R-CNN, bfloat16 for YOLO11s.
The native preprocessing, input resolution, proposal settings, NMS and cap
remain fixed. Only Faster R-CNN's native score cutoff is set to the already
prespecified collection floor. Deterministic algorithms and disabled TF32
are checked after native predictor initialization.

## Evidence and replay

### Preserved initial implementation failure

The initial Faster R-CNN seed-17 attempt stopped before its first complete
bundle or any metric calculation. The strict common box validator rejected
an upper image-boundary overshoot. A bounded diagnostic reproduced the same
failure, without computing performance, using the exact checkpoint and native
AMP path. Its saved raw float32 prediction contains one edge exceeding the
source boundary by 0.000244140625 pixels (one float32 ULP). A synthetic
Torchvision `resize_boxes` test independently reproduces this inverse-scale
rounding behavior: Torchvision clips boxes in resized coordinates before
multiplying by a float32 source/resized ratio.

The correction canonicalizes only an upper-bound overshoot of at most one
float32 ULP from the Faster R-CNN path. Negative coordinates, larger errors,
other dtypes and other detector paths still fail. In-bound coordinates,
scores, labels and all candidates are unchanged. Replaying the actual failed
prediction preserves its 100 candidates and every score while repairing
exactly one coordinate. The raw coordinate and its canonical counterpart
are saved privately for every affected future prediction and replayed by
verification. This is a numerical handoff bug fix, not a change to scientific
preprocessing, NMS, score support, ontology, endpoints or thresholds.

The failed receipt, environment, raw diagnostic and original source/config
are preserved under the private `superseded/batch49_initial_bounds_failure/`
directory. Its archive manifest maps original paths to preserved files and
binds their hashes. No generated metric or prediction bundle existed to
supersede. The corrected runner must use a new prediction directory after
preservation, never overwrite the failed receipt. The affected focused suite
passes 54 tests in the pinned CUDA environment after the repair, including
replay and tamper rejection for saved raw-coordinate corrections. All seven
initial-attempt files were archived and independently hash-verified before
the corrected attempt.

The diagnostic command is recorded in README. No external performance was
computed or inspected to choose this correction.

Restricted output is local under
`data/processed/vindr-cxr-external-v1/predictions/`. Each run saves all 3,000
records, including empty outputs, full emitted scores, category IDs and
original-coordinate boxes. Private exact-score frontiers retain all distinct
scores and both endpoints. Execution metadata bind the dataset version,
protocol/config/adapter/source/checkpoint hashes, environment files, actual
AMP dtype, start/end times and peak allocated CUDA memory. Runs execute
sequentially with batch one; decoded images are not cached as a cohort.

The aggregate `results/vindr_external_v1/inference_summary.json` contains
per-run AP@0.50/AP@0.50:0.95 at the fixed 0.001 AP support; five prespecified
FROC sensitivities with achieved FP/image and floor limitations; detection-cap
saturation; both threshold policies; and score summaries at four fixed
supports. Each threshold policy includes TP/FP/FN, precision/recall/F1,
FP/image, detections/image, emitting-image percentages and denominators.
Score summaries retain empty populations and give equal-run descriptive
means/sample SDs with defined-run counts, without pooling detections.
No FROC score coordinate is exported as a deployment threshold.

The primary policy transports historical **n=3 RSNA-validation thresholds**
0.69/0.05 unchanged to all five external runs. The secondary policy transports
the **post-hoc n=5 RSNA-validation sensitivity thresholds** 0.70/0.01.
Neither policy is selected using VinDr. All five seeds, including 271, remain.

The `verify` command repeats the integrity gate, checks all bundle/environment/
receipt hashes and independently recomputes every AP, exact FROC frontier,
budget, threshold metric, score summary and equal-run aggregate from saved
predictions. It performs no model inference and writes no scientific artifact.
Existing run output is never overwritten or automatically resumed. An error
stops execution with a private error receipt and preserves completed evidence.

## Interpretation boundaries

This is external testing across datasets and annotation ontologies. Other
findings remain strict-target negatives and do not create ignore regions.
Their unmatched detections count as false positives for this target, not
necessarily clinical errors. Historical YOLO internal accuracy used FP32;
the mandatory new AMP path is an additional disclosed comparison limitation.
Image percentages do not imply patient-level percentages or clinical utility.

Batch 49 provides per-run and descriptive aggregate results. Prespecified
bootstrap intervals and internal/external comparison figures belong to
separately authorized Batch 50. No training, fine-tuning, calibration,
external threshold optimization or outcome-driven protocol changes are allowed.
