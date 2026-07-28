# Decision and deviation log

This log preregisters protocol decisions that interpret or depart from
`Final project 1405.v1.pdf`. Add an entry before test-set access. Never rewrite
an old decision to match later results; append a superseding entry instead.

Each entry records the requirement affected, evidence, compliant alternative,
expected benefit, grading or validity risk, controls, affected artifacts, and
whether test data had been accessed.

## D-001 — Outcome-first policy and two-track comparison

- **Date:** 2026-07-28
- **Status:** Accepted policy; no specific instructor requirement overridden yet
- **Trigger:** The user authorized deviations from instructor directions when
  required for the best defensible result.
- **Requirement affected:** Project-wide interpretation of the assignment
  brief, especially its “identical training conditions” language.
- **Evidence:** The brief requires identical data, augmentation, and optimizer
  while comparing architectures with different training conventions; it also
  contains unresolved dataset-count and class-count inconsistencies documented
  in `Codex.md` and `docs/PROJECT_PLAN.md`.
- **Decision:** Use an assignment-aligned controlled Track A and a separately
  reported, equal-tuning-opportunity architecture-optimized Track B. Prioritize
  safety, ethics, law, privacy, and research integrity; then scientific validity
  and reproducibility; then assignment coverage; then convenience.
- **Compliant alternative considered:** Run only the strict shared-settings
  comparison. It is retained as Track A rather than discarded.
- **Expected benefit:** Track A preserves grading traceability while Track B
  answers the practically meaningful question of how each detector performs
  when configured competently.
- **Risks:** Additional compute and possible reader confusion. Mitigate with
  staged runs, explicit resource accounting, and separate tables and claims.
- **Controls:** Shared audited data, immutable splits and test set, canonical
  evaluator, metrics, reporting schema, and final hardware. All tuning uses only
  training and validation data.
- **Affected artifacts:** `AGENTS.md`, `Codex.md`, `README.md`,
  `docs/PROJECT_PLAN.md`, and future experiment configs and reports.
- **Test-set state:** No dataset downloaded and no result produced at acceptance.

## Entry template

### D-NNN — Short decision title

- **Date:**
- **Status:** Proposed, accepted, rejected, or superseded
- **Trigger:**
- **Requirement affected:**
- **Evidence:**
- **Decision:**
- **Compliant alternative considered:**
- **Expected benefit:**
- **Risks:**
- **Controls:**
- **Affected artifacts:**
- **Test-set state:**
