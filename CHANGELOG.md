# Changelog

This file records release-level changes. It does not replace the scientific
provenance in `results/`, the decision history in `docs/DECISION_LOG.md`, or the
commit history.

## [2.0.0] — Research Artifact Release

### Scientific analysis

- Expanded the clean detector comparison from three to five attempted runs per
  detector while retaining the low-confidence YOLO11s seed-271 run and reporting
  endpoint-specific sample sizes for conditional localization metrics.
- Added validation-selected operating points, five-run test-side PR/FROC and
  Pareto sensitivities, detection-calibration analyses, recall-weighted F-beta
  and hypothetical-loss sensitivities, and an audit that removes raw detector
  scores from conventional decision-curve interpretation.
- Corrected the inferential boundary to patient-cluster resampling with an
  independent-run primary estimand and separately labeled
  checkpoint-conditional permutation results.
- Added radiography-motivated synthetic acquisition/display sensitivity and
  expanded Grad-CAM parameter/randomization controls with explicit failure
  handling.

### Reproducibility and verification

- Added a hash- and schema-bound scientific-artifact manifest covering the
  manuscript-critical evidence and a verifier that does not refresh reviewed
  hashes in CI.
- Added source bindings and tolerances for central numerical manuscript claims,
  with deterministic verification against frozen result files.
- Added checkpoint identity/provenance metadata without distributing the model
  binaries, plus a four-tier reproducibility contract separating software
  checks, committed analysis, exact inference, and retraining.
- Hardened clean-checkout verification, including clone-stable artifact-input
  hashes and tests for missing declared external inputs.

### Documentation

- Added the current working manuscript and made its relationship to the
  historical technical report and frozen numerical sources explicit. Its tagged
  copy is historical traceability, not a frozen or separately versioned
  publication.
- Consolidated limitations, retrospective hypotheses, reporting checks,
  citation provenance, project decisions, and the release/reproduction boundary.
- Added software citation metadata and research-oriented release documentation.

### Infrastructure

- Added an AGPL-3.0-only repository license and clarified third-party data,
  dependency, pretrained-weight, and trained-checkpoint terms.
- Added a locked `uv` CPU environment and cross-platform GitHub Actions checks
  for formatting, lint, tests, artifact/claim verification, and package smoke.
- Pinned GitHub Actions to immutable revisions and verified clean-checkout
  behavior on Ubuntu and Windows.

## [1.0.0]

Historical first stable benchmark release, frozen by tag `v1.0.0` at commit
`3a3808841795938a296d48ae3b379b0d10ef3d48`.
