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

## D-010 — Keep `paper_draft.md` as the canonical editable journal manuscript

- **Date:** 2026-09-05
- **Status:** Accepted for the V7 baseline; manuscript content remains
  provisional pending later focused editing and human declarations

### Context

After V5, the repository gained a longer alternate Markdown manuscript and a
25-page rendered article PDF. Their additional length and polish do not by
themselves establish scientific or editorial authority. The current README,
scientific-artifact manifest, claim-source manifest, reporting checklist, and
final-submission audit all bind the verified manuscript workflow to
`report/paper_draft.md`. The historical `report/report.md` has a separate,
preserved role.

The long manuscript and its rendered PDF reproduce much of the current
numerical evidence but add unsupported declaration assertions, stronger causal/
superiority rhetoric, stale internal process terms, and a less complete
limitations boundary. Text-level and visual audit show that the PDF represents
the long manuscript line rather than `report/paper_draft.md`; its exact LaTeX
source/build recipe is not committed.

### Decision

`report/paper_draft.md` remains the sole canonical editable journal manuscript.
`report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md` is a noncanonical long
alternate draft that may be consulted, but nothing migrates from it without
later evidence review. `report/A_Controlled_Comparative_Study___article.pdf` is
a noncanonical read-only rendering of that alternate line and must not be
manually edited. `report/report.md` remains the historical/full technical
report.

### Consequences

- Scientific-claim binding and manuscript verification continue to target only
  `report/paper_draft.md` unless a later explicit decision changes the hierarchy
  and updates all bindings atomically.
- The long manuscript and PDF cannot supply or resolve funding, interest,
  ethics, consent, contribution, availability, or PPI declarations.
- Potential Methods/Supplement detail from the alternate drafts may be migrated
  only in a later manuscript batch after source checking; greater length is not
  a promotion criterion.
- Batch 41 changes no scientific result, threshold, artifact, or manuscript
  claim. Its complete baseline evidence is recorded in
  `docs/V7_Q2_VINDR_BASELINE_AUDIT.md`.

## D-011 — Use the observed exact-score frontier and retain the candidate-floor boundary

- **Date:** 2026-09-05
- **Status:** Accepted for current frozen-bundle reporting; lower-floor
  inference remains proposed only and requires user approval

### Context

The historical FROC analysis sampled 99 thresholds from 0.01 through 0.99,
while the frozen Phase 5 prediction bundles retained candidates down to 0.001.
YOLO11s reached the grid's lower boundary, so treating the grid as the complete
low-score frontier understated observable sensitivity and overstated several
between-detector margins.

### Decision

The current FROC display and manuscript claims use the **observed exact-score
frontier**: every unique score emitted into each of the ten frozen prediction
bundles, plus explicit empty and candidate-floor endpoints. The prespecified
FP/image operating budgets, all five seeds, images, annotations, class, IoU
0.50 matching, native NMS IoU 0.50, and maximum 100 detections/image remain
unchanged. Historical n=3 and n=5 grid artifacts remain immutable provenance.

The observed detector ordering does not reverse. Faster R-CNN remains higher
at all five budgets, but the exact-score gap strengthens only at 0.125 and
weakens at 0.25, 0.5, 1, and 2 FP/image relative to the historical n=5 grid.
The 0.001 candidate floor still limits YOLO11s in one run at 0.5, two runs at
1, and all five runs at 2 FP/image; affected aggregate values are lower-bound
observations.

### Consequences

- `configs/froc_exact_score_v2.yaml` and
  `src/analyze_exact_score_froc.py` define the current offline analysis; no
  publication parameter is embedded in the script.
- The historical grid is a comparison/provenance artifact, not a complete or
  threshold-independent frontier.
- A detector-neutral 0.0001 candidate floor is technically accepted by both
  inference adapters and is proposed for a ten-checkpoint, 750-image-per-model
  inference-only sensitivity. It was selected as a one-decade reduction, not
  for detector advantage.
- That lower-floor inference was not run. It requires explicit user approval
  and must write new versioned bundles and provenance rather than replace the
  current Phase 5 predictions.

## D-012 — Adopt the approved 0.0001 FROC sensitivity and retain the residual boundary

- **Date:** 2026-09-06
- **Status:** Accepted for current reporting; a further 0.00001 inference-only
  proposal requires new explicit user approval

### Context

The user approved D-011's detector-neutral 0.0001 checkpoint-inference
sensitivity. All ten frozen checkpoints were therefore evaluated on the same
750 internal-test images with the same annotations, class, preprocessing,
matching, native NMS, maximum detections, and seed scope. No training or weight
update occurred, and all outputs were written to a new v3 namespace.

### Decision

Use the v3 0.0001 bundles and exact-score artifacts as the current FROC
evidence. Faster R-CNN remains more sensitive at every prespecified budget.
YOLO11s changes relative to the 0.001 exact-score analysis by 0.0000, 0.0000,
+0.0052, +0.0246, and +0.1082 at ascending budgets. No conclusion reverses.

The v3 frontier is still not called exhaustive. YOLO11s seed 137 reaches only
0.7427 FP/image at the bundle boundary and therefore remains floor-limited at
1 and 2 FP/image. Those two YOLO11s aggregate values remain lower-bound
observations.

### Consequences

- `configs/evaluation_froc_lower_floor_v3.yaml`,
  `src/collect_froc_lower_floor_predictions.py`, and
  `configs/froc_exact_score_v3.yaml` define the current reproducible path.
- The original Phase 5, historical grid, and 0.001 exact-score artifacts remain
  preserved as provenance.
- A next floor of 0.00001 is technically supported by both adapters and was
  selected by the same one-decade, detector-neutral rule. The executable
  proposal is `configs/evaluation_froc_lower_floor_v4_proposal.yaml`.
- No v4 inference has been run. Its exact scope is the same ten models and 750
  images per model; it must create new v4 bundles and downstream artifacts only
  after a new explicit user approval.

## D-013 — Adopt the approved 0.00001 FROC sensitivity and stop at the residual bound

- **Date:** 2026-09-06
- **Status:** Accepted for current reporting; no still-lower inference is authorized

### Context

The user approved D-012's detector-neutral 0.00001 inference-only sensitivity
under the same ten checkpoint hashes, five seeds per detector, 750-image test
set, preprocessing, adapters, NMS, maximum detections, IoU matching, and
prespecified budgets. The only inference change from v3 was the candidate floor
from 0.0001 to 0.00001. No training, model selection, test-label threshold
tuning, or overwrite of v2/v3 evidence occurred.

### Decision

Use v4 as the current observed exact-score FROC evidence. Faster R-CNN remains
higher at all five budgets. Relative to v3, both detectors are unchanged
through 0.5 FP/image; YOLO11s increases from 0.4985 to 0.5075 at 1 and from
0.5881 to 0.6090 at 2, weakening but not reversing the detector gap.

The frontier is still technically incomplete only for YOLO11s seed 137 at
2 FP/image: its candidate-floor endpoint is sensitivity 0.5634 at 1.9907
FP/image. A conservative bound assigns sensitivity 1.0 to the entire
unobserved contribution. The resulting maximum YOLO11s aggregate is 0.6963,
still below Faster R-CNN's observed 0.6978. Therefore the missing frontier
cannot theoretically reverse the detector ordering at 1 or 2 FP/image.

### Consequences

- `configs/evaluation_froc_lower_floor_v4.yaml` and
  `configs/froc_exact_score_v4.yaml` define the current reproducible path.
- Historical grids and v2/v3 bundles, tables, figures, and summaries remain
  preserved provenance.
- A still-lower inference pass could complete the numeric curve near 2
  FP/image, but has limited scientific value for the qualitative detector
  ordering. It must not run without separate review and approval.
- Current prose uses “observed exact-score frontier” and “prespecified
  FP/image operating budgets”; it does not call the frontier exhaustive or
  threshold-independent.

## D-014 — Retain n=3 threshold provenance and report n=5 selection only as post-hoc sensitivity

- **Date:** 2026-09-06
- **Status:** Accepted for sensitivity reporting; historical precedence unchanged

### Context

Batch 35 expanded test-side operating-regime evidence to all five retained
runs but deliberately kept threshold selection on validation seeds 17, 42,
and 137. Batch 41 confirmed that raw validation predictions were missing for
both detectors at seeds 271 and 314. Batch 43 generated only those four bundles
from the exact frozen checkpoints under the original validation and inference
contract, then repeated the historical 0.01--0.99 maximum equal-run mean-F1
rule with the higher-threshold exact-tie rule. Test labels did not enter bundle
validation or threshold selection.

### Decision

Keep 0.69 for Faster R-CNN and 0.05 for YOLO11s as frozen historical n=3
provenance. Report the five-validation-run result only as **post-hoc validation
sensitivity** and never as prospectively frozen. It selects 0.70 for Faster
R-CNN and 0.01 for YOLO11s. Applying those thresholds unchanged to all five
internal-test bundles preserves Faster R-CNN's mean precision, recall, and F1
lead, although each margin weakens; the FP/image and detection-count orderings
reverse.

### Consequences

- Phase 14 files and the historical threshold configs remain byte-for-byte
  unchanged and authoritative for provenance.
- Batch 43 writes only versioned config, bundle, table, and summary paths.
- The four new validation bundles may support later explicitly authorized
  validation analyses, but they do not retroactively make earlier work
  prospective or validate a calibration/DCA protocol.
- All mean and sample-SD comparisons remain descriptive over five retained
  runs and are not inferential evidence of deployment stability.

## D-015 — Report aggregate DICOM-header cohort characteristics with an age-unit caveat

- **Date:** 2026-09-07
- **Status:** Accepted for descriptive cohort reporting only

### Context

The original RSNA DICOMs were locally available through the repository-relative
path already declared in `configs/dataset.yaml`. All 5,000 immutable split rows
mapped one-to-one to those source files and to the official NIH patient mapping.
The selected `PatientAge`, `PatientSex`, and `ViewPosition` tags could therefore
improve cohort reporting without model training or any change to membership.

All age elements use DICOM VR `AS`, but every selected value is numeric-only and
lacks an encoded D/W/M/Y suffix. Sex is complete with only F/M values, and
projection is complete with only AP/PA values. One numeric age is outside the
configured 0--120 plausible-year range.

### Decision

Report plausible numeric-only age values as **nominal years under an explicit
dataset-specific assumption**, and exclude the single out-of-range value from
age summaries. Report sex and AP/PA projection exactly as encoded. Percentages
use all studies in each partition as the denominator. These are descriptive
study-level header characteristics, not independently verified clinical
demographics, subgroup performance, or fairness evidence.

### Consequences

- `RSNA_DICOM_ROOT` is an optional environment override; the config-declared
  repository-relative DICOM path remains the default. A missing or incomplete
  root produces an actionable stop rather than path guessing.
- The generator hash-checks the dataset contract and all three immutable split
  manifests, reconstructs the official patient mapping, and reads headers with
  pixel decoding disabled.
- Only a four-row aggregate CSV and aggregate provenance JSON may be tracked.
  No source filename, exam ID, patient key, patient-level demographic row, raw
  DICOM, or pixel output is created.
- No model training, inference, subgroup comparison, or fairness claim is part
  of this decision.


## D-016 — Matched decoded-host timing v1 supersedes primary asymmetric timing

Date: 2026-09-07. Batch 45 verified the historical timing code/results and the
intended RTX 4060 Laptop GPU/i7-13650HX/16 GB reporting machine. Adopt
`decoded-host-to-source-detections-v1` as the canonical manuscript's primary
runtime comparison. Both detectors include resize/letterbox, conversion,
transfer, forward, native NMS, source-coordinate restoration and CPU outputs,
starting from the same decoded host image and excluding disk I/O/decoding.

Use the two exact primary seed-17 checkpoints, identical deterministic 100-image
test subset, batch 1, 10 warm-up images per detector/repetition, and three
complete technical repetitions. Explicit float16/bfloat16 AMP is checked
against active convolution dtypes. All 600 timed predictions agree exactly
with ordinary file inference under the same AMP. This is not a claim that
historical FP32 YOLO bundles are identical to the v1 AMP path.

Accepted Faster R-CNN / YOLO11s throughput is 20.91 / 57.56 FPS, median latency
47.20 / 17.29 ms, and image-latency IQR 1.22 / 1.19 ms. No training, test-label
selection, or training-replicate inference is introduced. Preserve the 46
hash-bound historical files and existing Pareto/raincloud timing axes with
their original labels. Keep parameters and incomplete profiler-registered
operation counts; remove the approximate 21-fold ratio as a headline fact.
Detailed provenance and an excluded warning-emitting development trial are
documented in `docs/COMPUTE_TIMING.md`. No later batch is authorized here.

## D-017 — Freeze VinDr-CXR external testing v1 before performance

Date: 2026-09-22. Status: frozen for user review; full inference not authorized
in Batch 47. Protocol: [VINDR_EXTERNAL_PROTOCOL.md](VINDR_EXTERNAL_PROTOCOL.md).
Config: [vindr_external_v1.yaml](../configs/vindr_external_v1.yaml).
Exact byte bindings: [v1 freeze manifest](VINDR_EXTERNAL_PROTOCOL_v1.sha256.json).

The user confirmed approved PhysioNet access, DUA acceptance and an official
download before restricted contents were read. Public v1.0.0 documentation,
local annotation/checksum/license files and all 3,000 test DICOM headers were
checked without decoding pixels or invoking a detector. Annotation files match
the supplied official checksum manifest; full image checksums and conversion
remain Batch 48 gates. Batch 46 paper/scientific/bibliography verification passes.

Freeze the official 3,000-image consensus test set and the single local target
concept `Lung opacity`, explicitly mapped to its actual CSV spelling
`Lung Opacity`. No synonym matching or merging of other local/global findings.
Every image without a strict-target box remains a negative for this target.
This is external testing and cross-dataset/cross-annotation-ontology
transportability, not an identical annotation task.

Freeze AP@0.50 and AP@0.50:0.95 at the internal 0.001 AP floor, exact-score
FROC from candidates down to 0.00001, and sensitivity at the same internal
0.125/0.25/0.5/1/2 FP/image budgets. NMS and matching IoU are 0.50, cap 100.
Historical RSNA n=3 validation thresholds 0.69/0.05 remain primary and apply
unchanged to all five external runs per detector. Batch 43's 0.70/0.01 applies
only as the separately labeled post-hoc n=5 sensitivity. Never optimize an
external threshold, including from FROC score coordinates. No VinDr training
data, fine-tuning, calibration, model/seed selection or outcome-guided changes
to ontology, preprocessing, support, endpoints or hyperparameters.

No defensible patient identifier exists in the inspected release: both CSVs
lack a patient key and all headers have no nonempty patient/study/series/SOP
identity tags. Freeze released study/image-level resampling, jointly across
detectors, and independent run resampling within detector. Do not invent
patient groups or pair seed numbers. All five runs, including seed 271, stay.
Document unknown within-patient dependence and the historical FP32 versus new
mandatory AMP YOLO execution difference. Per-run score count/range/quantiles
and planned tables/figures are specified before any external results.

Restricted derivatives remain local in ignored `data/`; only reviewed
nonidentifying aggregates may be published. User protocol review plus passing
Batch 48 adapter/integrity tests and a separate Batch 49 authorization are
required before full inference. No inference or external detector results
were inspected in this session. Changes require a versioned amendment with
reason and timing; never silently replace v1 or its hash record.

## D-018 — Retain complete frozen VinDr external inference and adverse results

Date: 2026-09-23. Batch 49 completed after the user's explicit request and
passing protocol, adapter, authorization, checkpoint and release gates.
Canonical generated aggregate evidence is
[`inference_summary.json`](../results/vindr_external_v1/inference_summary.json),
SHA-256 `d15070ebcafdfb2711b07db5090e67ef556b60d0042c87ac047b243bdf8fc656`.
All ten checkpoints and 3,000 official images per run are retained. Full
hash/metric/raw-coordinate-repair replay passed. The protocol/config and
38 frozen dependencies remain byte-identical; no scientific amendment occurred.

The first attempt stopped before any completed bundle or performance metric
because float32 inverse resizing could restore a clipped Faster R-CNN boundary
one ULP outside the original image. The exact native error was reproduced;
only one-ULP upper-bound overshoots are now canonicalized. Larger/negative
errors still stop, and raw corrected coordinates are preserved and replayed.
Seven initial-attempt files, including the original implementation/config and
raw diagnostic, remain hash-verified in the private superseded archive.
No ontology, preprocessing, resolution, score floor, NMS, cap, checkpoint or
threshold changed. See [the implementation record](VINDR_INFERENCE.md).

Both pipelines have severely reduced external performance under the strict
target ontology. Retain the near-zero historical-threshold recall, YOLO
seed-271 zero-output historical result, secondary-policy FP/image ordering
reversal, YOLO seed-137 floor limits and Faster R-CNN cap saturation exactly
as observed. Historical n=3 threshold transport remains primary; post-hoc n=5
transport remains secondary. No external threshold is selected. These results
do not establish an architecture-family cause, a patient-level clinical
interpretation or ordering beyond observed candidate support.

Readable [per-run tables](VINDR_INFERENCE_RESULTS.md) are deterministically
rendered from the generated aggregate summary. Restricted image-linked evidence
stays in ignored local storage. Batch 50 uncertainty/internal-external synthesis
and later manuscript updates require separate authorization; stop for review.
