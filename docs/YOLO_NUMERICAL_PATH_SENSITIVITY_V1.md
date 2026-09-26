# YOLO numerical inference-path sensitivity, v1

**Status:** Secondary sensitivity, Batch 53. Historical FP32 internal analysis
and frozen VinDr results remain immutable primary provenance. No training,
threshold selection, checkpoint selection or external re-inference is permitted.

## Frozen design and execution gate

The [path audit](YOLO_NUMERICAL_PATH_AUDIT_V1.md) found common native loading,
preprocessing, letterboxing, fusion, NMS and coordinate-restoration semantics.
The intervention reproduces the external **numerical path**: explicit CUDA
bfloat16 autocast and disabled TF32, with FP32 model weights. Historical TF32
flags were not recorded and the old seed helper left them at defaults. This
is therefore not an activation-dtype-only causal ablation. No adapter is rewritten.

[`configs/yolo_numerical_path_sensitivity_v1.yaml`](../configs/yolo_numerical_path_sensitivity_v1.yaml)
binds 51 prerequisite files, including all five checkpoint hashes, both original
prediction supports, immutable annotations/splits, thresholds, evaluator and
adapter source. All five runs (17, 42, 137, 271, 314) use the same 750 internal
images and 268 target boxes. Candidate floor 0.00001, AP floor 0.001, 640 input,
class-aware native NMS IoU 0.50 and cap 100 are unchanged. Native collection is
strict `>`; evaluation is `>=`. Historical 0.05 and post-hoc 0.01 cutoffs are
applied unchanged, alongside the historical shared 0.25 operating point.
FROC retains every exact score, tied scores together, the empty/floor endpoints
and budgets 0.125/0.25/0.5/1/2 FP/image without interpolation or extrapolation.

The original lower-floor bundles filtered to AP support are byte-value identical
to the original AP bundles for all five runs. Paired numerical differences
therefore use the same checkpoints and images at each relevant support. Pairing
does not extend to same-number seeds across detector families. Summary means and
sample SDs weight each run equally. No new significance tests, confidence
intervals, or outcome-dependent materiality cutoffs are introduced.

Before substantive collection, the pinned NVIDIA environment passed preflight,
including hashes, split/annotation membership and observed bfloat16 convolution
output with FP32 weights. The inference API is guarded against training calls;
all modules must be in evaluation mode with gradients disabled. Every new output
is created exclusively; an existing versioned root is a hard refusal, including
partial attempts. Preflight performs only two diagnostic calls on the first
internal image; the collection records fresh per-run activation checks.

The prespecified collection is 5 x 750 = 3,750 model-image evaluations, plus
two initialization/dtype calls per run. Expected prediction output was roughly
2 MB compressed, based on 1,700,443 bytes for the five historical lower-floor
bundles; tables and provenance add several MB. Ordinary single-GPU inference
was expected to take several minutes, with integrity/replay overhead. No
external image enters this collection.

## Candidate agreement

Before operating-threshold filtering, compare the two post-NMS retained sets
at the same 0.00001 floor. Within each image and class, consider box pairs with
IoU >=0.90; greedily accept descending-IoU pairs, breaking ties by original
FP32 index and then bfloat16 index. Each candidate participates at most once.
Remaining candidates are reported as unmatched/appeared/disappeared under this
rule; those labels do not establish identity among pre-NMS proposals.

Report image-level count agreement, matched absolute coordinate differences
in source pixels, signed/absolute score differences, matched IoU and Spearman
correlation of matched scores within each run (average-rank ties, undefined for
constant/insufficient scores). This correlation describes the matched emitted
population, not every latent candidate. At 0.001/0.01/0.05/0.25 report matched
upward/downward crossings and their fractions of matched candidates, plus
unmatched candidates above each cutoff. Overall emitted counts/proportions are
also reported so matching attrition cannot conceal emission changes.

## Reproduction

Run from the repository root with authorized RSNA images and exact local
checkpoints. Preflight/collection require the pinned CUDA environment; replay
uses CPU and does not regenerate predictions. Portable CI exercises synthetic
tests and artifact bindings only.

```powershell
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.analyze_yolo_numerical_path --config configs/yolo_numerical_path_sensitivity_v1.yaml --mode preflight
& C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe -m src.analyze_yolo_numerical_path --config configs/yolo_numerical_path_sensitivity_v1.yaml --mode run
& .\.venv\Scripts\python.exe -m src.analyze_yolo_numerical_path --config configs/yolo_numerical_path_sensitivity_v1.yaml --mode verify
```

Accepted outputs belong under `results/logs/phase53_yolo_numerical_path_v1/`:
five prediction bundles, exact protocol/image identities, environment receipts,
per-run metric deltas, equal-run aggregates, agreement table and complete summary.
After collection, use `verify`; rerunning `run` or `preflight` on an occupied
version refuses rather than replacing evidence.

## Completed result and interpretation

All five internal runs completed and passed full CPU replay of metrics,
candidate agreement, aggregates, hashes and original image bytes. Model
collection took approximately 168 seconds in total, excluding integrity and
analysis overhead. Each run recorded a bfloat16 convolution and FP32 weights.
The canonical secondary summary SHA-256 is
`f76ac0cb438f31e1721ebe60e596db01a5b613cc702a9bc5fdf6e3ea2f028830`.
The [generated detailed results](YOLO_NUMERICAL_PATH_RESULTS_V1.md) contain all
per-run/aggregate metrics and score distributions; no primary number changed.

| Question / observed conclusion | Classification | Evidence and scope |
|---|---|---|
| Does precision materially change internal AP or detector ordering? | Unchanged ordering | AP50 mean 0.162612 to 0.158412 (delta -0.004200, about 2.58% relative reduction); AP50:95 0.054168 to 0.053137. All five AP50 values decline. This is a modest, nonzero effect, not numerical equivalence. Faster R-CNN remains higher. |
| Does it change the score-scale interpretation? | Unchanged | At AP support, equal-run mean score 0.076292 to 0.076924 and median score 0.010906 to 0.011761. Emitted populations differ; these are not probability-calibration estimates. The broad score-scale/low-score-run interpretation persists. |
| Does it change frozen-threshold emissions/recall? | Unchanged major comparisons | Historical recall 0.194776 to 0.191791; detections/image 0.221333 to 0.218667. Post-hoc n=5 recall 0.292537 to 0.285821; aggregate detections/image stays 0.414933 while TP/FP composition changes. No cutoff is reselected. |
| Does it change observed exact-score FROC? | Unchanged ordering | Equal-run sensitivity deltas span -0.005970 to +0.002239; Faster R-CNN remains higher at every prespecified budget. Seed 137 still lacks full support at 2 FP/image. |
| Does seed 271 remain low-score with nonzero AP? | Unchanged | Maximum score 0.041273523 to 0.040771484; new AP50 0.151761500 and AP50:95 0.053629264. No detections at historical 0.05; the run remains in every applicable aggregate. |
| Do major internal detector comparisons change? | Unchanged | AP/FROC direction, higher YOLO precision but lower recall at shared 0.25, and Faster R-CNN precision/recall/F1 advantages under both detector-specific policies persist. The policy-dependent FP/image and detection-count ordering reversal also persists. These are observed comparisons, not re-estimated confidence intervals. |
| Does numerical harmonisation alter severe RSNA-to-VinDr failure? | Unchanged | Harmonised internal YOLO AP50 0.158412 and historical-cutoff recall 0.191791 remain far above frozen external 0.000501 and 0.004211. No external output changed. |
| Is numerical asymmetry still an equally unresolved threat to that descriptive interpretation? | Weakened concern | The internal matched sensitivity shows modest aggregate effects relative to the severe transport loss. It does not identify a causal dtype effect independently of TF32 policy, prove equivalence, or establish that all remaining change is dataset shift. |

The internal lower-bound qualification remains. Seed 137 reaches only
1.966667 FP/image at the new floor endpoint. At budget 2, replacing that run's
observed sensitivity with the mathematical maximum gives
`0.6059701492537313 + (1 - 0.5708955223880597) / 5 = 0.6917910447761193`,
still below the frozen Faster R-CNN observation 0.6977611940298508. This is a
support bound, not a confidence bound or a change to the primary published bound.
External missing-support bounds and detection-cap limitations remain unchanged.

Aggregate stability does not mean candidate stability. Total retained
post-NMS candidates increased from 38,577 to 39,235 across the five runs.
Under the prespecified IoU >=0.90 rule, 9.95%--20.68% of bfloat16 candidates
and 10.83%--19.68% of FP32 candidates were unmatched. Matched-score Spearman
correlations range from 0.989001 to 0.996880. Threshold crossings and unmatched
emissions are both visible in the detailed results. The maximum per-run
historical recall change is -0.014925 (seed 17); the largest post-hoc recall
decline is -0.018657 (seed 42). Thus a small equal-run difference should not be
read as exact per-run agreement.

Dataset/site/population/annotation/preprocessing differences remain confounded.
No new bootstrap or equivalence test was performed; existing uncertainty
statements still refer to the frozen primary analysis. The sensitivity remains
secondary and does not authorize replacing the historical internal numbers.

The repository advanced externally during this session from starting HEAD
`ac6975d` to `4e66e3e` (the separate user-authorized Batch 52 commit recorded in
Session 113). All bound scientific inputs stayed exact. Execution receipts
record the actual latter HEAD plus hashes for this uncommitted implementation.
No staging, commit or push was performed as part of Batch 53.

## Verification and review gate

- Focused new tests: 27 passed. Relevant existing evaluator/FROC/timing tests:
  66 passed, seven authorized-data tests deliberately deselected in that command.
- Final portable suite: 465 passed, one expected metadata-only skip, seven
  scientific-data deselections. Full authorized suite: 472 passed, one expected
  skip. These commands do not regenerate GPU inference.
- Full five-run CPU metric/agreement replay passed, with all 11 recorded output
  hashes and original image-byte identities. Generated report replay passed.
- Scientific publication verifier: 88 artifacts, 646 present input bindings,
  212 result references. All 83 pre-existing entries are exactly preserved.
  Original 38-file external freeze still passes; original 417 manuscript claim
  sources/tolerances are unchanged. Current manuscript: 427 numeric claims and
  36 semantic guards, all passing.
- Ruff check/format for all four new Python files and `git diff --check` pass.
  Existing unrelated file bytes are preserved, including the reviewed PDF and
  archive; the index is empty. Detailed command logs are local in `tmp/batch53/`.

Stop for user review of this secondary interpretation. No Batch 54 work is
included. The Markdown manuscript contains the new evidence; the PDF and prior
numeric trace remain preserved review snapshots requiring a future refresh.
