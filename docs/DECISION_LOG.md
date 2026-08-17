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
