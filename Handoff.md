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
