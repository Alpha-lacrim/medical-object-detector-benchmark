# Decision log

Entries are append-only records of project-level choices. A later entry may
supersede part of an earlier entry without rewriting the historical text.

## D-001 — Two-track detector comparison design

- **Original proposal date:** 2026-07-28
- **Recorded from the reconciled project architecture:** 2026-08-14
- **Status:** Historical proposal; only Track A was delivered

### Context

The proposed study compared `fasterrcnn_resnet50_fpn_v2` and YOLO11s on the
patient-grouped RSNA Pneumonia subset. One comparison needed to stay close to
the assignment's controlled-experiment requirement, while a second comparison
was proposed to distinguish controlled conditions from each architecture's
best-effort performance.

### Proposal

- **Track A — assignment-aligned controlled comparison:** use the same fixed
  train/validation/test manifests, 640 x 640 input, COCO initialization, no
  stochastic training augmentation, seeds 17/42/137, matched early-stopping
  budget, and the same model-independent evaluator. Detector-intrinsic RPN,
  RoI, loss, assignment, precision-stability, and postprocessing behavior may
  differ, but every such asymmetry must be disclosed.
- **Track B — separately tuned, architecture-optimized comparison:** keep the
  same frozen data splits, test-access boundary, and evaluator, while allowing
  model-specific schedules, resolutions, augmentations, and other tuning under
  an equal-opportunity trial budget and an auditable trial ledger. Report this
  as a secondary comparison, separate from Track A.

### Outcome

The course submission implemented only Track A: three training seeds per
detector, unified clean evaluation, compute profiling, robustness,
explainability, and paired statistics. It contains no Track B search or Track B
results. This entry reconstructs the proposal from the reconciled architecture;
it does not represent Track B as preregistered or completed.

## D-002 — Descope Track B and reframe the research contribution

- **Date:** 2026-08-14
- **Status:** Accepted; supersedes only the Track B portion of D-001

### Context

The completed controlled comparison already consumes the appropriate scope for
one RTX 4060 Laptop GPU with 8 GB VRAM, 16 GB system RAM, and the course
timeline. A separately tuned search would require additional model-specific
trials, validation decisions, and full reruns, while weakening the clean
interpretation of the delivered controlled comparison.

### Decision

Track B is descoped for hardware- and time-budget reasons and is not required
for the research-paper direction. The contribution is reframed around an
assignment-aligned controlled comparison and multi-axis trade-off analysis
across detection accuracy, operating-point behavior, compute, robustness,
Grad-CAM localization, and statistical uncertainty—not best-effort leaderboard
performance.

### Consequences

- The course-submission evidence remains the single controlled comparison; no
  absent architecture-optimized result is implied.
- Research-track work may deepen the frozen comparison with analyses that reuse
  its evidence, but it does not need a separate Track B training/tuning program.
- Reviving architecture-specific tuning would be a new, explicitly approved
  study with its own budget and claims, not completion work for D-001.

## D-003 — Decline to revive a Track-B-style native/best-practice comparison

- **Date:** 2026-08-18
- **Status:** Accepted; does not reopen D-002

### Context

An independent technical review recommended reviving a second,
architecture-optimized comparison (each detector under its own best-practice
recipe, equal tuning budget) to isolate “one-stage vs two-stage detectors” from
“this Faster R-CNN pipeline vs this YOLO11s pipeline.”

### Decision

D-002's descoping is reaffirmed. The project's current claims do not depend on
isolating architecture family in the abstract—they depend on the
already-disclosed comparison of two specific, documented pipelines under
shared constraints, which remains scientifically sound. A full Track B under
an equal-opportunity tuning budget would multiply training compute
substantially on the project's fixed hardware (RTX 4060 Laptop, 8 GB VRAM), for
a claim-strengthening addition rather than a correctness fix, while several
correctness fixes (patient-cluster statistics, threshold-selection
methodology, CI) are still outstanding and higher priority.

### Consequences

- A much cheaper partial alternative remains available as optional future
  work: a single native-defaults run per detector (not a tuning search),
  explicitly framed as an exploratory robustness-of-conclusion check rather
  than a formal second track. This is not committed work and should only be
  considered after Batches 13–15 are complete and if compute time remains.
- The paper's contribution stays framed around the controlled comparison, the
  multi-axis trade-off characterization, and the threshold/score-scale
  mismatch finding—not an architecture-family claim.

## D-004 — Keep cost-weighted threshold calibration as a separate sensitivity analysis

- **Date:** 2026-08-25
- **Status:** Accepted; stands alongside and does not supersede the Batch 14
  validation-selected operating points

### Context

Batch 14 selected one operating threshold per detector by maximizing arithmetic
mean, equal-weight F1 across the three frozen validation seeds. The new threshold-
calibration analysis instead varies the assumed false-negative/false-positive cost
ratio through $\beta = \sqrt{C_{FN}/C_{FP}}$ and selects the threshold that maximizes
the patient-cluster-bootstrap lower 95% confidence bound of $F1_\beta$. Allowing both
selection documents to appear without an explicit precedence rule would make the
downstream operating-point claims contradictory.

### Decision

The Batch 14 thresholds remain the project's authoritative primary single-threshold
operating points and the source for the existing downstream threshold, FROC, and
Pareto results. This batch's full $\beta \in \{1, 3, 5, 10\}$ sweep is a distinct
cost-sensitivity extension and will be reported separately, including in any later
paper draft. It does not select one clinically definitive $\beta$ and does not replace
the Batch 14 thresholds. Even the $\beta=1$ result is not a replacement because its
lower-confidence-bound objective differs from Batch 14's mean-F1 point objective.

### Consequences

- `docs/THRESHOLD_ANALYSIS.md` and its selected operating-point tables remain the
  authoritative source for the primary validation-selected operating points.
- `docs/THRESHOLD_CALIBRATION.md` and its outputs must be labeled as a sensitivity
  analysis conditional on assumed costs, the available frozen validation predictions,
  and the patient-cluster bootstrap protocol.
- A later paper may compare the threshold shifts across the four assumed cost ratios,
  but must not present any one setting as a measured clinical utility or silently feed
  it into existing FROC, Pareto, or other downstream artifacts.

## D-005 — Separate inferential targets and resample detector runs independently

- **Date:** 2026-08-29
- **Status:** Accepted; supersedes the paired-seed bootstrap interpretation of
  the clean confidence intervals but preserves that calculation as a
  historical sensitivity artifact

### Context

The clean table combined confidence intervals that resampled patient groups
and same-number training seeds with permutation p-values that held the trained
checkpoints fixed. Those procedures answer different questions. In addition,
the shared seed integers 17, 42, 137, 271, and 314 were treated as matched
cross-detector blocks without a common-random-number design.

### Decision

The paper distinguishes two estimands:

- **Checkpoint-conditional estimand.** This conditions on the already-trained
  checkpoints and represents uncertainty from held-out NIH patient sampling.
  The patient-cluster detector-label permutation p-values belong to this
  estimand and are a **secondary sensitivity analysis conditional on the
  observed checkpoints**.
- **Training-procedure estimand.** This represents both held-out patient
  uncertainty and stochastic retraining/run variability. Its intervals use
  shared patient-cluster draws and separate within-detector trained-run draws.
  This is the **primary estimand** for broad claims about the two disclosed
  pipelines.

The same-number detector seeds are not scientifically meaningful matched
stochastic blocks. Both programs disable stochastic augmentation, so there is
no augmentation draw to couple. Faster R-CNN uses the project PyTorch loader
with batch size 2 and a seeded generator; YOLO11s uses Ultralytics' loader with
batch size 4. Their dataset abstractions, initialization shapes and calls,
training frameworks, RNG-consumption paths, and early-stopping trajectories
differ. A numeric seed is reproducibility metadata within each program, not a
shared random variate across programs. The five detector runs are therefore
treated as independent realizations within each detector.

### Consequences

- The primary bootstrap resamples 323 patient clusters and trained runs
  independently within detector, reconstructing TP/FP/FN ratios, matched-box
  localization, and COCO AP from the resampled predictions in every draw.
- Unconditional endpoints use five runs per detector. Conditional IoU and
  Dice use all five defined Faster R-CNN runs and the four defined YOLO11s
  runs; YOLO11s seed 271 remains undefined for those endpoints rather than
  being converted to zero. No model is retrained and the frozen test set and
  ten prediction bundles are unchanged.
- The former paired-seed table and provenance summary are retained at explicit
  `*_paired_seed_sensitivity_archive.*` paths. They are not a second proof of
  the training-procedure claim.
- No seed-aware p-value is introduced. Holm-adjusted patient-cluster
  permutation p-values remain checkpoint-conditional and are labeled as such
  wherever they accompany the primary intervals.
- Per-run, leave-one-training-run-out, and leave-one-seed-label-out artifacts
  are descriptive influence diagnostics. In particular, deleting seed 271 is
  not a corrected analysis and does not replace the all-attempt result.
