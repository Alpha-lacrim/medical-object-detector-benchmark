# Project handoff

This is an append-only session ledger. The next session must read the newest
entry first, verify it against Git and the filesystem, and continue from the
listed next action. Stable context belongs in `Codex.md`; stable workflow rules
belong in `AGENTS.md`.

## 2026-07-28 — Repository initialization

**Objective**

Read the final-project brief carefully, design the strongest feasible project
plan, create persistent context files, initialize a local repository, and
publish it to the user's GitHub account.

**Starting state**

- The workspace contained only `Final project 1405.v1.pdf`.
- No local Git repository or remote was configured.
- The connected GitHub app exposed a repository owned by `MitsuPishi`, but the
  user later clarified that this is not their account and must never be used.
- The user's only authorized GitHub account is `Alpha-lacrim`.
- GitHub CLI (`gh`) and Windows Package Manager (`winget`) were not installed.

**Changes**

- Extracted and audited all 8 pages of the PDF, including its three embedded
  dataset URLs.
- Identified the likely missing source paper:
  DOI `10.3934/bioeng.2025001`; this remains an inference requiring instructor
  confirmation.
- Verified that the likely paper is internally inconsistent: its dataset
  section reports 3,903 images/four groups, while its experiment and conclusion
  report 9,900 images/three groups. Exact reproduction is blocked until this is
  resolved.
- Compared the candidate datasets and provisionally selected the 3,903-image
  brain-tumor dataset because it matches the likely paper and includes
  train/validation/test partitions.
- Documented the conflict between the brief's five-class wording and the
  dataset's four image categories.
- Designed a controlled experiment covering clean detection, compute,
  robustness, detection-aware Grad-CAM, paired statistics, and deployment.
- Recorded the assignment PDF SHA-256 and detected the local RTX 4060 Laptop
  GPU with 8 GB VRAM for feasibility planning.
- Added the permanent rule that implementation commits contain no more than 500
  changed lines of source, test, and configuration code; future handoffs must
  record each commit SHA and its validation.
- Added a permanent authorization boundary: remote writes may target only
  `Alpha-lacrim`; `MitsuPishi` must never be used.
- Added `README.md`, `AGENTS.md`, `Codex.md`, `Handoff.md`,
  `docs/PROJECT_PLAN.md`, `.gitignore`, and `.gitattributes`.
- Created the private GitHub repository
  `Alpha-lacrim/medical-object-detector-benchmark` and pushed `main`.

**Validation performed**

- PDF metadata and full text were read with `pdfinfo` and `pdftotext`.
- Raw PDF hyperlinks were extracted and checked against their dataset pages.
- Current model documentation was checked against official Torchvision and
  Ultralytics sources.
- The likely source paper and its dataset/training details were checked on the
  publisher's page and PDF.
- Required files were checked for existence and nonzero size.
- All relative links in the five Markdown files resolved locally.
- `git diff --check` passed after adding a repository-wide LF policy.
- A second independent requirements review was applied to tighten scope,
  source-paper reproduction, YOLO26 head mode, named deliverables, input size,
  IoU/Dice aggregation, corruption weighting, and the statistical endpoint.
- Official portable GitHub CLI v2.94.0 was downloaded and its SHA-256 matched
  GitHub's published checksum.
- GitHub CLI/API authentication was verified as exactly `Alpha-lacrim` before
  the remote write.
- The remote was verified as private with `main` as its default branch.
- Local `main` was verified to track and match `origin/main` after the initial
  push.

**Commits**

- `f9ff63c2a9227178e5b752a9819d43798ef80423` — initialized the benchmark,
  assignment brief, plan, and memory files; validation: relative-link check and
  `git diff --check`.
- `904acf42706f7076e7ae0bcf4d9664cc4f444610` — recorded the corrected GitHub
  authorization boundary; validation: staged diff check and authenticated-owner
  verification.
- Session-ending `HEAD` — records the verified remote and final handoff. Its
  resolved SHA must be reported to the user because a commit cannot contain its
  own hash.

**Incomplete or blocked**

- Confirm the source paper, dataset, class semantics, allowed YOLO generation,
  fairness interpretation, deadline, and hardware budget with the instructor.
- No data download, implementation, environment lock, training, or result
  generation has begun.

**Next action**

1. Obtain instructor answers to the blocking questions in `Codex.md`.
2. Record the approved dataset, classes, model/head, source-paper
   interpretation, and fairness rules before implementation.
3. Create the environment/config foundation in coherent commits containing no
   more than 500 changed lines of source, test, and configuration code.

## 2026-07-28 — Outcome-first research policy

**Objective**

Record the user's authorization to override instructor directions when necessary
for the best scientific outcome, while preserving grading value where possible.

**Starting state**

- `main` was clean and matched `origin/main` at `9f22641`.
- The plan treated unresolved instructor questions as blockers and defined only
  one strict assignment-aligned comparison track.

**Changes**

- Added a durable outcome-priority policy to `AGENTS.md` and `Codex.md`, with
  safety/research-integrity limits and a requirement to preregister every
  instructor deviation before test access.
- Replaced indefinite instructor blockers with evidence-based fallback
  decisions for the source paper, dataset, class semantics, image counts, YOLO
  version, fairness interpretation, and unknown compute/deadline.
- Upgraded the protocol to v0.2 with an assignment-aligned Track A and a
  separately reported, equal-tuning-opportunity architecture-optimized Track B.
- Added `docs/DECISION_LOG.md` and accepted D-001, recording the policy,
  rationale, compliant alternative, risks, controls, affected artifacts, and
  pre-test state.
- Updated `README.md` so the public roadmap matches the two-track protocol.

**Validation performed**

- Re-read the assignment PDF, all three memory files, and the detailed project
  plan before changing the research policy.
- `git diff --check` passed.
- All relative links in the six project Markdown files resolved locally.
- The diff is documentation-only; it adds no source, test, or configuration
  code, so the 500-code-line commit ceiling is satisfied.

**Incomplete or blocked**

- No dataset download, environment lock, implementation, training, or result
  generation has begun.
- Instructor answers are still welcome, but every known ambiguity now has a
  non-blocking fallback that must be frozen in config before final test access.
- The exact Track B search spaces and equal compute/trial budget remain to be
  defined after the dataset and dependency preflights.

**Next action**

1. Implement Milestone 1: pin the environment, add the configuration schema,
   and create deterministic seed and tiny CPU smoke-test foundations.
2. Encode Track A and Track B as separate validated configs and add a generated
   difference check before any detector training.
3. Continue in coherent commits containing no more than 500 changed lines of
   source, test, and configuration code.

**Commits**

- Session-ending `HEAD` — records the outcome-first policy, two-track protocol,
  D-001, and this handoff; validation: relative-link check and
  `git diff --check`. Its resolved SHA must be reported to the user because a
  commit cannot contain its own hash.

## 2026-07-28 — Reproducible implementation foundation

**Objective**

Begin implementation with the strongest reproducible foundation: pinned project
metadata, validated Track A/Track B configuration, deterministic utilities,
shared evaluation primitives, and fast tests suitable for continuous
integration.

**Starting state**

- `main` was clean and matched `origin/main` at
  `317dca0f2791b9fe58337bca2c045a4010c40c2e`.
- The repository contained the brief, protocol, decision log, and memory files,
  but no Python package, environment definition, configuration, tests, or CI.
- No dataset had been downloaded and no test-set result existed.

**Changes**

- Added a Python 3.13.14 `uv` project with an exact hash-bearing lock, bounded
  lightweight dependencies, and mutually exclusive official PyTorch CPU/CUDA
  13.0 detector extras.
- Added a strict immutable experiment schema, canonical configuration
  fingerprint, explicit run gates, and a synthetic Track A/Track B smoke
  contract. A smoke config cannot train/test final models, and test access
  requires an explicitly frozen real-data config.
- Added deterministic Python, NumPy, optional-Torch, CUDA, TF32, cuDNN, and
  Windows-worker seeding plus a machine-readable offline smoke CLI.
- Added commit-pinned Windows/Linux GitHub CI.
- Added immutable canonical prediction/target records, continuous-coordinate
  IoU, score-first class-aware matching, micro/macro/per-class metrics, and
  explicitly named localization-only matched box IoU/Dice.
- Added official pycocotools AP50/AP50:95 evaluation using the same canonical
  records, including correct empty-class and negative-image handling.
- Added a deterministic YOLO dataset audit and CLI covering image/label
  pairing, readability, dimensions, normalized boxes, classes, duplicate
  boxes, orphans, exact hashes, split leakage, counts, and manifest
  fingerprinting.
- Added all required deterministic lighting, noise, blur, and JPEG corruption
  levels, including JPEG qualities 20 and 50.
- Queried current public Kaggle metadata for dataset version 5 and documented
  safe download and audit instructions. No raw data was committed.
- Updated `README.md`, `Codex.md`, and the protocol checkpoint to match the
  implemented state.

**Validation performed**

- Re-read the complete assignment PDF and verified its recorded SHA-256.
- Resolved 70 packages for both accelerator forks and installed the isolated
  lightweight environment under standard-GIL Python 3.13.14.
- `uv lock --check`, Ruff formatting, Ruff lint, and all 41 unit/fixture tests
  passed.
- `uv build --wheel` produced
  `meddet_benchmark-0.1.0-py3-none-any.whl`; its contents contain only the
  intended package modules and distribution metadata.
- The offline smoke CLI emitted strict JSON with configuration fingerprint
  `9f26a579a1e208e8953c14840a795ce8607489c68dcc813299b4ce7b4f94bc0b`.
- Official COCO tests covered perfect, missed, wrong-class, class-without-GT,
  negative-only, and identity-contract cases.
- Each implementation commit was staged and inspected separately; changed
  source/test/config lines were 428, 249, 485, 266, 360, and 276 respectively,
  all below the 500-line ceiling. Generated `uv.lock` changes were excluded as
  permitted by `AGENTS.md`.
- The public Kaggle API reported version 5, last update
  `2025-02-10T21:20:45.74Z`, 313,038,935 bytes, and CC BY 4.0. An
  unauthenticated archive probe returned HTTP 403, so no data was claimed.
- All relative links in the seven project Markdown files resolved locally, and
  `git diff --check` passed.
- Draft PR
  [`#1`](https://github.com/Alpha-lacrim/medical-object-detector-benchmark/pull/1)
  was opened from `agent/implementation-foundation` into `main`. Both Ubuntu
  jobs and both Windows jobs passed for `f547f89`.

**Incomplete or blocked**

- Dataset acquisition is blocked on Kaggle authentication or a manual version-5
  download. Consequently, actual counts, class order, no-tumor semantics,
  patient identifiers, and split leakage remain unverified.
- The CUDA 13.0 extra is locked but not installed; model weights have not been
  downloaded or hashed, and no Faster R-CNN/YOLO GPU preflight has run.
- The final real-data config, offline shared augmentations, model adapters,
  training loops, profiling, detection-aware union-mask IoU/Dice, near-duplicate
  audit, Grad-CAM, statistical resampling, and reports remain incomplete.
- No test-set access, training, or result generation occurred.

**Next action**

1. Authenticate to Kaggle outside the repository or manually place exact
   version 5 under `data/raw/brain-tumor-v5/`, then follow `data/README.md`.
2. Read the downloaded `data.yaml`, pass its actual ordered names to
   `python -m meddet_benchmark audit-data`, and commit only the resulting small
   manifests and report.
3. Install and preflight the locked GPU stack with
   `uv sync --locked --group dev --extra cu130`; record package, CUDA, driver,
   GPU, and pretrained-weight hashes before implementing detector adapters.

**Commits**

- `ff2d4c8` — strict configuration, accelerator-aware dependency lock, and
  seven config tests; 428 source/test/config lines.
- `8fddac8` — deterministic runtime, offline smoke CLI, and cross-platform CI;
  249 changed source/test/config lines.
- `30e3b69` — canonical records and operating-point evaluator; 485
  source/test/config lines.
- `1950803` — official COCO AP evaluation; 266 changed source/test/config
  lines excluding generated lock changes.
- `a845ba6` — deterministic dataset audit and CLI; 360 changed
  source/test/config lines excluding generated lock changes.
- `c222632` — frozen deterministic robustness corruptions; 276
  source/test/config lines.
- `f547f89` — dataset instructions, implementation status, durable context, and
  the completed local-validation handoff.
- Session-ending `HEAD` — records the verified draft PR and remote CI result.
  Its resolved SHA must be reported because a commit cannot contain its own
  hash.
