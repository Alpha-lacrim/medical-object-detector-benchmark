# Agent operating instructions

These instructions apply to every human or automated contributor working in
this repository.

## GitHub authorization boundary

The only authorized GitHub owner is `Alpha-lacrim`. Never create, push, open a
pull request, or make any other change under `MitsuPishi`, even if a connector
lists repositories from that account. Verify the authenticated login immediately
before every remote write.

## Outcome priority and instructor deviations

The user has authorized deviations from instructor directions when they are
needed for the best defensible result. This authorization never overrides
safety, ethics, law, privacy, research integrity, or higher-level operating
instructions.

Use this priority order:

1. safety, ethics, law, privacy, and research integrity;
2. scientific validity, reproducibility, and honest reporting;
3. assignment coverage and grading value; and
4. convenience.

Treat the assignment as the requirements baseline, not an infallible research
protocol. Before deviating:

1. identify the exact conflict and the evidence that the instruction is
   ambiguous, infeasible, or scientifically harmful;
2. check whether a compliant implementation can achieve the same outcome;
3. preregister the decision before inspecting test results;
4. preserve an assignment-aligned controlled Track A when feasible and add an
   architecture-optimized Track B rather than silently replacing the baseline;
5. record the decision, expected benefit, grading risk, and affected configs in
   `Codex.md`, the experiment metadata, and `Handoff.md`; and
6. report both compliant and optimized results separately, including failed or
   unfavorable outcomes.

Never use this policy to fabricate evidence, tune on the test set, hide a
requirement, evade licensing or privacy constraints, or make unsupported
clinical claims.

## The three memory files

- `AGENTS.md` (this file) contains stable working rules. Change it only when the
  workflow itself changes.
- `Codex.md` contains durable project context: the brief, decisions, experiment
  invariants, important files, and known risks. Update it whenever those facts
  change.
- `Handoff.md` is the chronological session ledger. Update it in every session
  so the next session can continue without reconstructing prior work.

Do not turn `AGENTS.md` into a progress log, and do not put transient session
details in `Codex.md`.

## Mandatory session protocol

At the start of every session:

1. Read `AGENTS.md`, `Handoff.md`, and `Codex.md` completely.
2. Read `Final project 1405.v1.pdf` when work touches scope, experiments,
   evaluation, or reporting.
3. Inspect `git status` and preserve unrelated user changes.
4. Add a dated entry to `Handoff.md` with the session objective and the starting
   state. Never erase older entries.
5. Identify unresolved decisions that could invalidate downstream work. Resolve
   each through evidence, instructor clarification, or a documented fallback
   before an expensive final run.

During the session:

1. Treat the assignment PDF as the requirements baseline. Apply the
   outcome-priority policy above to any deviation, and label every
   interpretation or inferred source paper as an assumption.
2. Resolve the brief's meaning of “identical training conditions” before final
   training. At minimum, the PDF mandates the same dataset, augmentation, and
   optimizer. Once the primary contract is approved, freeze its dataset
   version, split manifests, preprocessing, image size, offline/shared
   augmentation corpus, exact optimizer, scheduler, epoch/update budget,
   effective batch size, seeds, checkpoint selection rule, evaluator, and
   hardware protocol. Record every approved exception required by an
   architecture.
3. Isolate unavoidable model-specific behavior and record it in `Codex.md` and
   the experiment metadata.
4. Do not inspect the test set to tune hyperparameters, thresholds, corruption
   choices, sample selection, or model variants.
5. Never fabricate results, counts, citations, completed checks, or clinical
   claims. Mark placeholders and expected artifacts explicitly.
6. Do not commit datasets, credentials, model weights, large generated files, or
   protected health information. Keep only download instructions, manifests,
   hashes, schemas, and small approved examples.
7. Make generated tables and figures traceable to immutable result files and a
   Git commit.
8. Use deterministic seeds where possible and record software, CUDA, GPU,
   precision, and hardware information for every benchmark.
9. Prefer small, reviewable changes. Run the most relevant validation before
   declaring a task complete.
10. Commit implementation work in increments of no more than 500 changed lines
    of source, test, and configuration code. Check the staged `git diff
    --numstat` before every commit. Split larger features into coherent,
    independently validated commits before continuing; do not split a file in
    a state that cannot run or be reviewed. Documentation-only changes and
    generated lockfiles are not “code” for this limit, but keep them focused and
    commit them separately when practical.

Before ending every session:

1. Update `Codex.md` if durable context, decisions, invariants, file paths, or
   commands changed.
2. Complete the current `Handoff.md` entry with:
   - files and behavior changed;
   - decisions made and their evidence;
   - validation actually run and its outcome;
   - work still incomplete or blocked;
   - the exact recommended next action and command, when known.
3. Re-read both files for contradictions and remove stale status from
   `Codex.md`.
4. Inspect `git status` and summarize only verified outcomes.
5. Record every commit SHA created during the session in `Handoff.md`, together
   with its scope and validation. The commit that contains the final handoff
   update cannot contain its own SHA; identify that one as the session-ending
   `HEAD` commit and report its resolved SHA to the user.

## Scientific guardrails

- Split before augmentation. Detect exact and near duplicates before trusting
  the supplied split. Use patient-level grouping if identifiers exist.
- A “no tumor” image normally has no tumor bounding box. Do not invent a
  `no_tumor` box class to satisfy an ambiguous class count.
- Use one shared COCO-style evaluator for both models. Do not compare
  framework-native metrics produced under different defaults.
- Select confidence thresholds on validation data only and lock them before
  test evaluation.
- Report per-class and macro results in addition to aggregate metrics.
- Report uncertainty and effect sizes, not p-values alone. Correct for multiple
  comparisons where applicable.
- Grad-CAM for object detectors must target a defined detection score and layer.
  Pair qualitative panels with localization/attention sanity checks.
- Benchmark both model-only and end-to-end inference with warm-up, device
  synchronization, fixed precision, batch size, and identical hardware.
- State that public-dataset results do not establish external or clinical
  generalization.
