# Focused Internal Manuscript Audit

**Date:** 2026-09-08
**Scope:** Batch 46 only; internal paper before external integration.
**Canonical file:** [`report/paper_draft.md`](../report/paper_draft.md).

## Prerequisite gate

Batch 41 explicitly retained the canonical file above. Neither the long
alternate nor `report/report.md` was used as the editing base. The initial
Git state included staged CODEX/HANDOFF changes and unrelated untracked
records; these were not staged, reset, committed, or published in this session.

Before editing, the paper verifier passed 64 claims and all configured guards.
The scientific verifier passed 72 artifacts, 346 present inputs, zero missing
external/ignored inputs, and 201 referenced result files. The following
summaries were read and checked against that evidence:

| Prerequisite | Verified output | Disposition and remaining boundary |
|---|---|---|
| Batch 42 | [`phase42_froc_exact_score_v4/summary.json`](../results/logs/phase42_froc_exact_score_v4/summary.json), ten lower-floor bundles, exact operating points and conservative bounds | Complete at candidate floor 0.00001. All runs reach 1 FP/image; YOLO11s seed 137 ends at 1.9907. Its 2-FP/image contribution remains floor-limited. The aggregate mathematical-maximum bound (0.6963) stays below Faster R-CNN (0.6978). Further inference has limited completeness value, is not authorized, and was not launched. |
| Batch 43 | [`phase43_threshold_selection_n5_validation_sensitivity/summary.json`](../results/logs/phase43_threshold_selection_n5_validation_sensitivity/summary.json), ten validation/test bundles, historical reproduction and selected-point tables | Complete. The historical three-validation-run rule remains 0.69/0.05. Identical selection across five validation runs yields 0.70/0.01, explicitly post-hoc, with no test-label selection. |
| Batch 44 | [`phase44_cohort_characteristics/summary.json`](../results/logs/phase44_cohort_characteristics/summary.json), aggregate cohort table and mapping/split gate | Complete. Zero patient-group overlap; all 5,000 studies mapped to official metadata. Unitless numeric ages are nominal years; one value is excluded. Sex/projection are header descriptors, not subgroup or fairness evidence. |
| Batch 45 | [`phase45_inference_timing_v1/summary.json`](../results/logs/phase45_inference_timing_v1/summary.json), raw times, repetitions and aggregate table | Complete on intended hardware. All 600 timed outputs exactly match ordinary inference under the same AMP. Offline timing verification passes historical preservation, raw statistics and provenance. Three technical repeats do not establish a five-training-run matched Pareto frontier; that optional extension remains outside scope. |

SHA-256 of the reviewed prerequisite summaries, in the same order:

```text
6614b49208a635e902b6b77e4b2351be86a9700ca3bcec3152c6c1a831799898
0cbd04874192ac659eef645f64a8352e2de48a8b867bd990be26e17f59e4dcd0
4a4d205599a6bc3142d33d928d9251cd02a3fb3b231e62c58f5219f2f4adc397
1e33816ce4519083f139c3e05a81ca5857ed67c6cf9242af2bf7f224bdae5df8
```

## Editorial changes

The manuscript decreased from 11,890 to 4,940 whitespace-delimited words
(about 58% shorter; the same count includes Markdown tables, captions,
front matter, and declarations). The title/abstract explicitly remain
provisional pending external results. The central question is operating-point
dependence under a patient-disjoint common protocol, not first discovery of
score-scale mismatch or an architecture-family causal effect.

Main evidence remains five retained runs, AP/PR, the shared cutoff,
historical and separately post-hoc validation selection, exact-score FROC,
training-procedure uncertainty, and standardized implementation timing.
Four image embeds form three figure groups: PR/F1-threshold, exact-score FROC,
and matched timing. The cohort table was transposed for readability without
changing its values. Patient/run/bootstrap/conditional-test distinctions and
the AMP timing reference boundary are explicit.

D-ECE now has only a compact secondary methods/result account. It is
detection-level, conditional on the emitted-detection population,
support/binning dependent, descriptive, and not clinical-risk calibration.
Grad-CAM, common corruptions, acquisition/display stress, F-beta, hypothetical
loss, and historical Pareto/decision-analysis details are routed to the
existing supplementary index and underlying documents. No calibration model,
new experiment, external dataset integration, or inference was added.

Journal prose has no project batch identifiers, D-xxx decisions, course-project
framing, inflated certainty language, or architecture-family causal claims.
The old hypothesis/decision history and scientific provenance documents were
preserved. The prior citation audit is retained verbatim below its new dated
mapping. All seven `AUTHOR ACTION REQUIRED` declarations remain textually
identical; no ethics, consent, funding, COI, contributions, or involvement fact
was inferred. Checkpoint availability is still a factual release gap.

## Literature and claim bindings

The current [citation audit](CITATION_AUDIT.md) covers all 16 used references.
Five additions address ECCV 2024 detector-threshold/calibration pitfalls,
the 2026 direct RSNA detector comparison, CLAIM 2024, the official RSNA
dataset description, and challenge evaluation. Existing Wu et al. RSNA work
was rechecked and retained. IEEE full text was inaccessible; the coauthor's
uploaded conference paper was the primary-text fallback, with its resolution
inconsistency explicitly recorded rather than silently reconciled. No
cross-paper performance values were imported.

The shared bibliography retains 37 entries so the untouched historical report
and long alternate still resolve their 19 and 27 keys. The new offline checker
validates the repository's literal-field BibTeX syntax, required fields, years,
DOI syntax/duplicates, key uniqueness, and cited-key resolution. It is not a
general BibTeX processor or a substitute for source verification.

The numerical claim manifest now has 66 bindings: 58 retained/rebound claims,
plus eight bindings completing both detectors' five FROC budgets. Six bindings
were removed because their numerical prose left the main paper:
`faster_corruption_retention`, `yolo_corruption_retention`,
`faster_gradcam_energy_in_box`, `yolo_gradcam_energy_in_box`,
`faster_full_randomization_ssim`, and `yolo_full_randomization_ssim`.
Their underlying tables, figures, summaries, and scientific-manifest bindings
are unchanged. D-ECE's two retained numerical claims are labeled secondary.
Semantic guards now protect the revised scope, provisional status, historical
timing separation, and exclusions from threshold-selection/FROC inference.

The [reporting crosswalk](REPORTING_CHECKLIST.md) was remapped to current
sections and its manuscript hash refreshed. No missing evidence was promoted
to a completed CLAIM item merely because wording was shortened. The historic
H1--H6 records remain retrospective repository provenance rather than journal
prose. Broader reporting and author-action gaps remain visible.

## Completion checks and review boundary

Exact offline paper/scientific/bibliography/test commands are in
[README](../README.md#repository-verification). The timing gate additionally ran:

```powershell
& .\.venv\Scripts\python.exe -m src.benchmark_inference --config configs/inference_timing_v1.yaml --mode verify
```

Paper claims and semantic guards, scientific hashes, bibliography checks, the
20 affected tests, focused Ruff lint/format, local manuscript/crosswalk
links and anchors, and Git whitespace pass. Every pre-existing scientific
artifact and the historical/alternate manuscripts were preserved by a
working-tree SHA-256 comparison captured before editing. The scientific
artifact manifest itself did not need regeneration. The pre-existing
train.csv Git-versus-working-tree serialization issue remains separately
recorded in the handoff; this editing session does not repair or certify it.

Stop for review of the focused internal draft. External results and all later
batches remain outside this session. Author declarations may remain unresolved
until the authors choose to supply factual determinations. No commit or push
is part of this batch.
