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

## D-004 — Keep later validation-threshold sensitivity separate from the primary thresholds

- **Date:** 2026-08-25
- **Status:** Threshold-precedence decision remains accepted; the cost
  interpretation and terminology are superseded by D-006

### Context

Batch 14 selected one operating threshold per detector by maximizing arithmetic
mean, equal-weight F1 across the three frozen validation seeds. The later
threshold-sensitivity analysis instead varies $\beta\in\{1,3,5,10\}$ and selects
the threshold that maximizes the patient-cluster-bootstrap lower 95% confidence
bound of $F_\beta$. Allowing both selection documents to appear without an
explicit precedence rule would make the downstream operating-point claims
contradictory. This entry originally interpreted beta through a direct
false-negative/false-positive clinical-cost identity; D-006 records why that
interpretation was unsupported and supersedes it.

### Decision

The Batch 14 thresholds remain the project's authoritative primary
single-threshold operating points and the source for the existing downstream
threshold, FROC, and Pareto results. This batch's full
$\beta\in\{1,3,5,10\}$ sweep is a distinct recall-preference sensitivity and
will be reported separately, including in any later paper draft. It does not
select one clinically definitive beta and does not replace the Batch 14
thresholds. Even the $\beta=1$ result is not a replacement because its
lower-confidence-bound objective differs from Batch 14's mean-F1 point
objective.

### Consequences

- `docs/THRESHOLD_ANALYSIS.md` and its selected operating-point tables remain the
  authoritative source for the primary validation-selected operating points.
- `docs/THRESHOLD_CALIBRATION.md` and its outputs must be labeled as a
  recall-weighted F-beta sensitivity conditional on explicit preference
  settings, the available frozen validation predictions, and the
  patient-cluster bootstrap protocol.
- A later paper may compare threshold shifts across the four beta settings but
  must not present any one setting as measured clinical utility or silently
  feed it into existing FROC, Pareto, or other downstream artifacts. Any
  hypothetical linear misclassification-loss analysis must be reported as a
  separate objective, as formalized by D-006.

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

## D-006 — Treat beta as an F-beta preference parameter and separate linear loss

- **Date:** 2026-08-29
- **Status:** Accepted; supersedes D-004's cost interpretation while preserving
  its threshold-precedence and validation-only decisions

### Context

D-004 correctly prevented the later sensitivity thresholds from replacing the
Batch 14 primary operating points, but it identified
$\beta^2$ with an assumed false-negative/false-positive cost ratio. The
implementation actually maximizes a pointwise lower bootstrap bound of
$F_\beta$, a weighted harmonic mean of precision and recall. The repository
contains no citation, patient-outcome valuation, or derivation that makes this
ratio objective equivalent to minimizing an empirically measured clinical-harm
function.

### Decision

The frozen $\beta\in\{1,3,5,10\}$ sweep is retained because it was specified
before test evaluation, but beta is now a recall-versus-precision preference
parameter. In the weighted harmonic mean, $\beta^2\in\{1,9,25,100\}$ is the
relative recall weight; it is not a measured clinical harm ratio. The canonical
name is **recall-weighted F-beta validation-threshold sensitivity**.

The existing F-beta thresholds remain unchanged because the Batch 19 bootstrap
stream is explicitly preserved. Stability diagnostics are descriptive and use
validation only: a contiguous near-optimal lower-bound plateau within 0.01 of
the maximum, the bootstrap distribution of draw-specific selected thresholds,
and selection frequency for every candidate threshold. They neither tune nor
alter any test result.

A separate hypothetical detection-error analysis minimizes
$L(\tau;r)=rFN(\tau)/N+FP(\tau)/N$ for assumed
$r\in\{1,9,25,100\}$. Here one unmatched target and one false-positive
detection are assigned explicit linear penalties and $N$ is the number of
validation images. These ratios are hypothetical methodological assumptions,
not clinical valuations or deployment utilities. Selection remains
validation-only, and the analysis is not substituted for F-beta or the Batch 14
primary thresholds.

### Consequences

- D-004 still governs precedence: Batch 14's 0.69/0.05 thresholds remain the
  authoritative primary single-threshold operating points and no Batch 29
  sensitivity threshold is applied to test, FROC, Pareto, or DCA.
- Canonical tables and the figure use recall-preference terminology and live at
  new Batch 29 paths. The exact pre-Batch-29 table, figure, and provenance are
  retained only as explicitly named historical archives.
- The full threshold-selection path verifies the model-development validation
  role, upstream validation split, manifest test-isolation flag, and exact
  agreement between annotation and validation-manifest image identities.

## D-007 — Remove raw detector-score curves from conventional DCA interpretation

- **Date:** 2026-08-29
- **Status:** Accepted; supersedes the Batch 20 interpretation while preserving
  its exact arithmetic and provenance as historical exploratory evidence

### Context

The Batch 20 implementation defined an exam action as maximum emitted detector
confidence `>= tau` and simultaneously used `tau/(1-tau)` as the false-positive
weight. Conventional DCA instead treats threshold probability as the
decision-maker's harm/benefit trade-off. Extensions for continuous diagnostic
tests and markers require conversion to predicted outcome probability before
the same probability threshold defines action and weighting. A bounded raw
detector score is not automatically that probability.

### Decision

The Batch 20 calculation is classified as
`NON_STANDARD_RAW_SCORE_THRESHOLD_UTILITY`. It is removed from the main
manuscript Results and retained only in the Supplementary/Limitations as an
exploratory raw-score threshold utility/sensitivity calculation. No detector
comparison, cutoff range, or reference-strategy comparison from it may be
described as standard net benefit, clinical utility, a beneficial decision
range, deployment readiness, or a clinical threshold.

Probability-based salvage is not performed. The retained test grid contains
ten detector/runs, whereas frozen validation predictions exist for only six:
both detectors at seeds 17, 42, and 137. Faster R-CNN and YOLO11s seeds 271 and
314 lack the run-specific validation predictions needed to fit and freeze a
separate exam-outcome probability mapping before test evaluation. No
calibrator family or hyperparameters are selected and test behavior is not
used for calibration decisions.

### Consequences

- The exact original code, config, 198-row table, figure, and provenance are
  retained under explicit `pre_batch30_nonstandard` archive names and their
  SHA-256 hashes are verified before regeneration.
- New canonical supplementary artifacts reproduce the same arithmetic under
  raw-score utility/sensitivity terminology and record the non-standard
  classification in every table row, the figure, and provenance summary.
- The public conventional-DCA helper accepts only a typed outcome-probability
  vector carrying a validation-frozen mapping identifier and a separately
  typed, elicited decision-threshold probability. Raw detector scores, raw
  detector-confidence cutoffs passed as `p_t`, and untyped numeric values are
  rejected, with regression tests guarding both boundaries.
- The 750-image test subset (169 positive, 581 negative; 323 patient groups;
  22.533% image-level prevalence) is enriched and internal, not a deployment-
  prevalence sample. Even a future probability-valid DCA on this population
  would not establish external clinical utility or deployment readiness.

## D-008 — Use n=5 as the principal operating-regime sensitivity display

- **Date:** 2026-08-31
- **Status:** Accepted; preserves the historical n=3 analysis and does not
  supersede D-004/D-006 threshold precedence

### Context

The threshold sweep, official PR curves, FROC, validation-selected operating
points, and Pareto figure were originally frozen over seeds 17, 42, and 137.
The later clean comparison retained five predeclared attempts per detector,
including YOLO11s seed 271, but the operating-regime artifacts were not then
expanded. All ten hash-bound test prediction bundles and all ten run-specific
compute tables now exist, so the test-side analyses can be recomputed without
training, inference, threshold reselection, or test-set tuning. Validation
predictions still exist only for the original three runs per detector.

### Decision

The manuscript's principal operating-regime figures use the separately
versioned n=5 all-attempt sensitivity. The original n=3 figures and tables
remain unchanged, explicitly historical/prespecified provenance artifacts.

This choice follows design and provenance rather than visual preference. The
five-run set is the complete predeclared attempt population already governing
the clean comparison and avoids presenting a central curve after omitting an
adverse but valid run. Every n=5 test result is a deterministic offline
redescription of predictions frozen before Batch 35. The detector thresholds
remain exactly 0.69 and 0.05, selected by maximum mean F1 on the original n=3
validation bundles; the n=5 fixed-threshold and recall-Pareto results are
sensitivity evaluations of those frozen rules, not n=5 threshold selection.

### Consequences

- Official PR and exploratory 0.01--0.99 threshold figures, FROC, and Pareto
  use visibly labeled n=5 sensitivity paths in the current manuscript.
- Historical filenames without an n=5 suffix remain n=3 and are never
  overwritten or relabeled. Their values remain available for direct
  n=3-versus-n=5 comparison.
- Seed 271 contributes nonzero COCO AP from predictions retained down to the
  frozen 0.001 floor and low-threshold FROC points, but defined zero
  precision/recall/F1 at score 0.25 and at the frozen YOLO11s threshold 0.05.
  It is not filtered because of those zeros.
- The threshold-selection sample size remains n=3 even where test evaluation
  uses n=5. No later F-beta or hypothetical-loss threshold enters these
  analyses, consistent with D-004 and D-006.
- The dedicated conclusion table reports every unchanged, strengthened,
  weakened, or reversed finding. It records weakened shared-threshold
  precision/recall margins and AP gaps as well as strengthened F1,
  fixed-threshold, FROC, and AP@0.5 position-count results; no reversal occurs.

## D-009 — Use one version for software and research-artifact releases

- **Date:** 2026-09-02
- **Status:** Accepted for the v2.0.0 release candidate; publication and tagging
  remain owner-reviewed actions

### Context

The historical public research release is `v1.0.0`, while the Python project,
importable package, and root entry in `uv.lock` remained at the bootstrap value
`0.1.0`. No repository policy documented a separate package-version line or a
different lifecycle for research releases. The research-artifact candidate is a
material expansion of the first stable baseline: it adds five-run and
estimand-corrected analysis, calibration/robustness/XAI sensitivity work,
artifact/claim verification, and a cross-platform locked CI boundary. The
tracked `report/paper_draft.md` is the current working manuscript, not a
separately versioned release artifact.

### Decision

From release 2.0.0 onward, `pyproject.toml`,
`meddet_benchmark.__version__`, the editable project entry in `uv.lock`, and the
research release use the same semantic version. Public Git tags use the
`v<version>` form. Version 2.0.0 communicates a materially expanded
research-artifact release rather than a compatible patch to the original
research baseline.

The version labels the repository release snapshot; it does not replace the
per-artifact hashes, study phases, generator/config bindings, or checkpoint
identities in the scientific manifests. A tag retains the manuscript text at
that commit for historical traceability, but it does not make the manuscript
final, frozen, or version 2.0.0. Future manuscript edits on `main` do not by
themselves require a new repository release.

### Consequences

- The v2.0.0 candidate aligns all three package declarations at `2.0.0` and
  proposes tag `v2.0.0`.
- Historical tag `v1.0.0` remains immutable at commit
  `3a3808841795938a296d48ae3b379b0d10ef3d48`; its historical internal package
  value is documented by history rather than rewritten.
- Release metadata changes do not trigger a mechanical refresh of frozen
  scientific/checkpoint hashes.
- The final tag and GitHub release may be created only after owner review of an
  exact committed candidate and successful CI for that SHA.
