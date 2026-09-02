# Medical Object Detector Benchmark v2.0.0 — Research Artifact Release

## Overview

Version 2.0.0 is the research-artifact release of the controlled Faster R-CNN
and YOLO11s comparison on the RSNA Pneumonia Detection Challenge. It identifies
an exact repository snapshot and the committed scientific-evidence/provenance
boundary described below. It is research software and is not a medical
diagnostic system.

The final `v2.0.0` tag must point to the exact owner-reviewed release commit.
This file does not itself publish a release or authorize a tag.

## Major changes since v1.0.0

### Scientific and analysis changes

- The clean comparison now retains all five attempted runs per detector,
  including the valid low-confidence YOLO11s seed-271 result, with explicit
  endpoint-specific denominators where matched-only localization is undefined.
- Patient-cluster inference now separates the primary training-procedure
  estimand from secondary permutation results conditional on the observed
  checkpoints.
- The artifact set adds validation-selected operating points, five-run test-side
  PR/FROC and Pareto sensitivities, detection calibration, recall-weighted
  F-beta and hypothetical-loss sensitivities, a raw-score utility audit,
  radiography-motivated synthetic sensitivities, and expanded XAI sanity checks.
- The manuscript interpretation was hardened to preserve descriptive,
  retrospective, internal-testing, and non-clinical boundaries.

### Reproducibility and verification changes

- `results/scientific_artifact_manifest.json` binds 46 critical artifacts to
  reviewed hashes, schemas, generators, configs, inputs, and regeneration tiers.
- `report/paper_claim_sources.yaml` binds 35 central numerical manuscript claims
  to frozen sources and explicit rounding tolerances.
- CI verifies a locked CPU environment, formatting/lint, the test suite (300
  passed and one declared environment-conditional skip in the latest exact-SHA
  run), package smoke, artifact integrity, and manuscript-claim consistency on
  Ubuntu and Windows.
- `results/checkpoint_release_manifest.json` records the identity of ten exact
  trained checkpoints without treating the manifest as permission or a download
  mechanism for the checkpoint binaries.

### Documentation and infrastructure changes

- `report/paper_draft.md` is the current working manuscript;
  `report/report.md` remains the historical full technical report.
- The repository adds consolidated reproducibility, limitation, decision,
  hypothesis, reporting, citation-audit, and release documentation.
- The project adds the AGPL-3.0-only license, a locked `uv` dependency model,
  cross-platform Foundation CI, and pinned GitHub Actions revisions.

## Scientific artifact scope

The release includes the Git-tracked source, configs, split metadata, frozen
prediction/analysis inputs, numerical tables, figures, and provenance records
identified by the scientific-artifact manifest. The manifest and claim-source
file are integrity/traceability controls; neither is a claim that CI regenerated
the evidence from raw radiographs or trained weights.

The numerical source of truth is `results/`. The current manuscript is a rounded
presentation of that evidence. Release preparation did not alter
scientific results, manuscript numerical claims, selected thresholds, seeds,
estimands, model weights, or scientific/checkpoint hashes.

## Manuscript status

`report/paper_draft.md` is the current working manuscript. The copy present in
this tagged repository is retained for historical traceability and represents
the manuscript state at the release commit. The manuscript is not declared
final or frozen by this release and may continue to evolve on the repository's
`main` branch.

The release-controlled scientific evidence consists of the frozen scientific
artifacts and provenance identified by the repository's scientific-artifact
machinery, not the future prose state of the manuscript. The manuscript has no
2.0.0 version, and manuscript edits alone do not require another repository
release.

## Reproducibility and verification

Run the release-candidate checks from the repository root:

```powershell
uv lock --check
uv sync --locked --group dev --extra cpu
uv run --locked --extra cpu ruff format --check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
uv run --locked --extra cpu ruff check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
uv run --locked --extra cpu python -m pytest -q
uv run --locked --extra cpu python scripts/verify_scientific_artifacts.py
uv run --locked --extra cpu python scripts/verify_paper_claims.py
uv run --locked --extra cpu python -m meddet_benchmark smoke configs/smoke.yaml
git diff --check
```

CI also applies Ruff to `scripts/build_scientific_artifact_manifest.py`; it
does not rebuild the manifest because mechanically refreshing hashes would
approve changed scientific evidence.

Passing these checks establishes that the locked CPU software paths work, the
committed critical artifacts retain their reviewed hashes and schemas, declared
references exist, and the bound manuscript numbers agree with their frozen
sources. It does **not** download the RSNA dataset, train either detector,
recreate checkpoints, run GPU inference, reproduce timing on the study machine,
or regenerate every experiment from raw data. See `docs/REPRODUCIBILITY.md` for
the four-tier contract.

## Data availability

Raw and processed patient images are not included in Git or in this release.
Users must obtain the RSNA Pneumonia Detection Challenge data separately,
accept the applicable access/competition terms, and run the documented
preparation pipeline. The repository license does not grant rights to the
RSNA/NIH data or dataset-derived image content.

## Model and checkpoint availability

Trained and pretrained model binaries are not included in Git or in this
release candidate. The checkpoint release manifest records filenames, sizes,
SHA-256 identities, and provenance for ten locally audited trained checkpoints;
its `public_download_url` is `null`. Those hashes allow an independently
supplied file to be checked for identity, but they do not make the weights
available. Upload remains blocked pending explicit human review of attribution,
license, pretrained-weight, and serialized-metadata/privacy conditions.

## Known limitations

The benchmark is retrospective internal testing on one curated RSNA/NIH-derived
cohort. It has no external or prospective validation, demographic subgroup
analysis, clinical-utility demonstration, safety case, or deployment claim.
Clean results use five runs per detector, while several robustness and XAI
analyses use only seed 17. Full limitations and reporting gaps are maintained in
`docs/LIMITATIONS.md` and `docs/REPORTING_CHECKLIST.md`.

## Citation

`CITATION.cff` provides the available software citation metadata for version
2.0.0. No DOI, ORCID, venue, acceptance/publication status, final paper citation,
or release date is asserted because those fields are not yet established in
authoritative project sources. This metadata cites the repository/software and
research artifact; it does not assert that the living manuscript is published.

## License

Repository-authored software and documentation are licensed AGPL-3.0-only,
except where otherwise noted. The repository license does not replace the
separate terms governing the RSNA/NIH data, dataset-derived content, pretrained
weights, or external dependencies. See `LICENSE` and the README license section.

## Previous release

The prior release is `v1.0.0`, titled “Medical Object Detector Benchmark
v1.0.0,” and freezes commit
`3a3808841795938a296d48ae3b379b0d10ef3d48`. Its tag must not be moved or
retargeted.
