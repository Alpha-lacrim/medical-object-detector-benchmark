# CODEX.md — Living Project Context

> `PROJECT_SPEC.md` is the static source of truth. This file records decisions
> actually made, current files, phase state, and open risks. Read it and the
> newest `HANDOFF.md` entry at the start of every session.

## Project one-liner

Controlled comparison of Faster R-CNN and YOLO11s for lung-opacity localization.
The current internal manuscript centers on operating-point dependence, five
retained runs, patient/run uncertainty, and standardized implementation timing;
calibration, robustness, and explainability remain secondary evidence.

## Fixed hardware

ASUS ROG Strix G16: Intel i7-13650HX, RTX 4060 Laptop GPU (8 GB VRAM),
16 GB DDR5-4800 RAM. AMP remains mandatory for both detectors.

## Decisions Log

- **VinDr Batch 48 adapter complete (2026-09-23):**
  `src/data/prepare_vindr.py` and operational `configs/vindr_adapter_v1.yaml`
  implement the unchanged Batch 47 scientific contract. All 38 frozen files,
  official v1.0.0 metadata and all 3,000 DICOM checksums pass. Strict exact
  `Lung Opacity` yields 84 positive images, 95 boxes and 2,916 negatives,
  retaining negatives with other findings; no invalid/duplicate target boxes.
  All 3,000 native PNGs pass decoding, lossless round-trip and common-loader
  integration. A fixed 94-image repeat/native-geometry sample is deterministic,
  with maximum inverse error 0.00019315886765980395 native pixels (<0.001).
  Actual unsigned 16-bit storage contains 10/12/14/16 stored bits and both
  polarities; JPEG 2000 is supported by Pillow/OpenJPEG. Present rescale is
  identity; windows vary; orientation/projection tags are absent. The frozen
  stored-pixel min-max transform remains unchanged. No constant/nonfinite
  images were found. PNG compression level 1 affects encoding only.
  Restricted manifest/COCO/headers/PNG data and loader YAML remain under
  ignored `data/processed/vindr-cxr-external-v1/`; only the nonidentifying
  `results/vindr_external_v1/adapter_preflight.json` is a public result artifact.
  `docs/VINDR_ADAPTER_PREFLIGHT.md` and README record reproduction and boundaries.
  Focused checks pass 71 tests plus one declared skip, Ruff, 66 paper claims,
  and the unchanged 72-artifact scientific verifier. No detector inference,
  external performance analysis, protocol amendment, commit or push occurred.
  Stop for user review; Batch 49 still requires separate authorization and
  native candidate-floor/AMP/checkpoint/prediction checks.

- **VinDr external protocol v1 frozen before performance (Batch 47):**
  `docs/VINDR_EXTERNAL_PROTOCOL.md`, `configs/vindr_external_v1.yaml` and
  `docs/VINDR_EXTERNAL_PROTOCOL_v1.sha256.json` freeze the official v1.0.0
  3,000-image consensus test set and 38 protocol/internal dependency hashes.
  D-017 records strict target concept `Lung opacity`, explicitly mapped to
  actual CSV `Lung Opacity` without merging findings. Collection/FROC floor
  0.00001, AP floor 0.001, NMS/matching IoU 0.50, cap 100 and internal
  0.125/0.25/0.5/1/2 FP/image budgets are fixed. Historical 0.69/0.05 threshold
  transport remains primary; Batch 43's 0.70/0.01 stays post-hoc sensitivity.
  All five runs per detector including 271 remain; independent detector-run
  resampling and paired released-image resampling are prespecified. No
  defensible patient grouping: both CSVs lack a patient key and all 3,000
  headers have no nonempty PatientID/Study/Series/SOP identity tags. Unknown
  patient dependence remains a limitation. Mandatory verified AMP also
  retains the disclosed difference from historical FP32 YOLO accuracy outputs.
  User confirmed approved access/DUA/official download before restricted reads.
  Local root is supplied via `VINDR_CXR_ROOT`; the provided layout is
  `data/raw/vindr-cxr/` with CSVs at root and DICOMs under `test/`. Annotation,
  license and supplement hashes agree with the supplied official manifest;
  the 3,000 filenames agree with its test inventory. Full DICOM checksums,
  decoding, target counts and adapter tests remain Batch 48. Restricted
  row-level derivatives stay under ignored `data/`, with only reviewed
  nonidentifying aggregate publication. Batch 46 verifiers, all ten checkpoint
  hashes and 38 focused tests pass. No pixels decoded, detector inference,
  external results inspected, model/source/internal-result changes, commit
  or push. User must review the protocol before separately authorized Batch 49.

- **Windows download tooling (2026-09-16, Session 98):** Explicit standalone
  environment-support request; no scientific batch executed. Bare `wget` is
  PowerShell's Invoke-WebRequest alias. Installed portable GNU Wget 1.21.4 x64
  in ignored `.tools/wget/wget.exe`. `docs/VINDR_WGET_WINDOWS.md` records source,
  local hash, `--no-config`, and the resumable test-only command. HTTPS probe
  reaches PhysioNet and returns expected unauthenticated HTTP 401. User must
  run with their approved account and enter their password locally. No dataset
  files downloaded; authenticated access and available disk space unverified.

Newest entries appear first; superseded decisions remain recorded.

- **Focused internal manuscript complete before external integration:** Batch 46
  passed the Batch 42--45 gate and retained Batch 41's canonical
  `report/paper_draft.md`. It reduced the draft from 11,890 to 4,940
  whitespace-delimited words (about 58%), retaining patient-disjoint data,
  common evaluation, five runs, AP/PR, shared thresholds, historical n=3 and
  separately post-hoc n=5 validation selection, floor-bounded exact-score FROC,
  training-procedure uncertainty, and matched timing. D-ECE is explicitly
  secondary, detection-level, emitted-population conditional, descriptive,
  support/binning dependent, and not clinical-risk calibration; Grad-CAM,
  corruption/acquisition stress, F-beta, hypothetical loss, and project history
  are routed to existing supplementary/provenance records. No research,
  calibration fitting, inference, retraining, or external integration was added.
  Current primary sources were verified, including Kuzucu et al. ECCV 2024,
  Chinnam et al. CCIC 2026, CLAIM 2024, RSNA methodology, and Wu et al. 2024.
  The citation audit records inaccessible IEEE text and the primary coauthor-
  uploaded-paper fallback without importing performance claims. All 16 current
  citations and the historical manuscripts' 19/27 keys resolve against the
  shared 37-entry bibliography. New offline `scripts/check_bibliography.py`
  and its tests document syntax/identifier resolution separately from source
  review. Paper verification passes 66 numerical bindings and revised guards;
  six removed secondary prose bindings retain their scientific artifacts, and
  eight additions cover every central FROC budget. Scientific verification
  remains 72 artifacts/346 present inputs/201 referenced results, with no
  manifest regeneration. The 20 affected tests, focused Ruff, timing verifier,
  links/anchors, SHA preservation and whitespace pass. Title/abstract remain
  provisional; seven unchanged AUTHOR ACTION REQUIRED declarations remain.
  `docs/INTERNAL_PAPER_FOCUS_AUDIT.md`, citation/reporting audits and README
  record the review state. Historical/alternate manuscripts and scientific
  provenance are unchanged. No staging, commit or push occurred (Session 96).

- **Matched decoded-host inference timing v1 is primary runtime evidence:**
  Batch 45 verified the actual ASUS ROG Strix G16/i7-13650HX/16 GB/RTX 4060
  Laptop GPU reporting machine and the existing Python 3.11.15, Torch
  2.6.0+cu124, Torchvision 0.21.0+cu124, Ultralytics 8.4.110, CUDA 12.4,
  cuDNN 90100, driver 610.47 Windows runtime. The repository `.venv` is CPU;
  publication timing used `C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe`.
  New `configs/inference_timing_v1.yaml` and `src/benchmark_inference.py` start
  both detectors from the identical decoded uint8 RGB host image and include
  resize/letterbox, conversion, transfer, native forward/NMS, source-coordinate
  restoration and CPU output extraction, excluding disk I/O/decoding. Each
  frozen seed-17 checkpoint uses the identical hash-ranked 100-image test
  subset, batch 1, 10 warm-up images per repetition, and three full technical
  repetitions with alternate model order. Actual convolution dtypes verify
  float16/bfloat16 AMP; all 600 timed outputs exactly match ordinary file
  inference under the same explicit AMP, including all counts, labels, boxes
  and scores. This does not claim parity with historical FP32 YOLO bundles.
  Accepted Faster R-CNN / YOLO11s totals over 300 calls are 14.3465654 /
  5.2123790 seconds, FPS 20.91093 / 57.55529, median latency 47.20395 /
  17.28700 ms, and IQR 1.22010 / 1.19335 ms. Repetitions are technical,
  not biological/patient/training replicates; no new inferential interval or
  standardized five-run Pareto frontier is asserted. D-016 and
  `docs/COMPUTE_TIMING.md` preserve 46 hash-bound historical files in place,
  keep parameter/training profiles, and retain GFLOPs only as incomplete
  profiler-registered operations, removing the approximate 21-fold ratio as
  a headline architecture fact. An initial development trial was excluded
  because deprecated `half=False` emitted warnings inside each YOLO timer;
  it remains in ignored `tmp/inference_timing_trial_deprecated_half/`. The
  corrected `quantize=None` run supplies all publication values. Primary
  table/Figure 5, manuscript, limitations, README command, claim bindings,
  scientific manifest and crosswalk are aligned. Verification passes: 357
  tests plus one declared skip, repository-wide Ruff, 72 scientific artifacts/
  346 present inputs/201 references, 64 manuscript claims/guards, exact
  historical preservation, and deterministic offline table/figure rendering.
  No training, frozen-prediction replacement, commit or push occurred
  (Session 93 / Batch 45).
- **RSNA cohort characteristics reported from immutable-split DICOM headers:**
  Batch 44 verified the unchanged 5,000-study train/validation/internal-test
  split (3,500/750/750 studies; 1,492/321/323 NIH patient groups; zero group
  overlap) and every study-to-patient/stratum/label/box mapping against the
  hash-bound official CSVs and committed audit. The configured raw DICOM root
  contains all 26,684 Stage 2 files. Aggregate-only extraction reads
  `PatientAge`, `PatientSex`, and `ViewPosition` without pixel data. All 5,000
  age values are numeric `AS` strings lacking D/W/M/Y units, so D-015 treats
  them transparently as nominal years for descriptive reporting only and
  excludes one value above the configured 0--120-year range; the remaining
  4,999 have median 49 years (IQR 36--60). Sex is 2,362 female and 2,638 male;
  projection is 2,363 AP and 2,637 PA; neither tag is missing or outside those
  observed categories. These are limited header descriptors, not clinically
  verified demographics and not subgroup, fairness, transportability, or
  causal evidence. Only aggregate CSV/JSON outputs are retained. Verification
  passes: 333 tests plus one declared skip, repository-wide Ruff, 67 scientific
  artifacts/224 inputs/197 references, 49 manuscript claims/guards, and Git
  whitespace. No training, inference, split mutation, staging, commit, or push
  occurred (Session 91 / Batch 44).
- **Five-run validation threshold-selection sensitivity complete; historical
  n=3 provenance retained:** Batch 43 verified the frozen Phase 14 selection
  rule and independently reproduced thresholds 0.69/0.05 from validation seeds
  17/42/137. It also verified the Batch 35 five-run test analysis, all ten
  checkpoint hashes (962,924,817 bytes), the 750-image/277-box validation
  source, and Batch 41's four-bundle gap. Inference-only collection generated
  Faster R-CNN and YOLO11s validation bundles for seeds 271/314 under the
  original split, preprocessing, adapters, candidate floor 0.001, NMS 0.50,
  100-detection cap, coordinates, and frozen checkpoints. Three score-0.25
  count checks are exact; deterministic Faster seed 314 preserves TP/FN but
  has one fewer FP/prediction than its training-time row, recorded under a
  config-bounded audit. The unchanged 99-point maximum equal-run mean-F1 rule
  selects 0.70 for Faster R-CNN and 0.01 for YOLO11s across all five validation
  runs. Applied unchanged to all five test bundles, Faster R-CNN versus
  YOLO11s mean precision is 0.3686 versus 0.2631, recall 0.3388 versus 0.2925,
  F1 0.3458 versus 0.2657, FP/image 0.2205 versus 0.3104, and detections/run
  256.2 versus 311.2. Threshold separation strengthens; mean precision,
  recall, and F1 advantages weaken; FP/image and detection-count orderings
  reverse. Run-level precision-SD ordering reverses, recall/F1 stability
  advantages weaken, and FP/image/count stability advantages strengthen.
  D-014 keeps 0.69/0.05 as historical provenance and labels 0.70/0.01 only
  **post-hoc validation sensitivity**, never prospectively frozen. Historical
  artifacts remain hash-identical. Verification passes: 325 tests plus one
  declared skip, repository-wide Ruff, 65 scientific artifacts/216 inputs/196
  references, 43 manuscript claims/guards, GPU preflight, and Git whitespace.
  No training, historical overwrite, staging, commit, or push occurred
  (Session 90 / Batch 43).
- **Approved 0.00001 FROC inference complete; residual cannot reverse the
  ordering:** the user approved the detector-neutral v4 continuation under the
  same ten immutable checkpoint hashes, five seeds/detector, 750-image/268-box
  test set, preprocessing, adapters, IoU-0.50 NMS/matching, 100 detections per
  image, and prespecified budgets. The only inference change from v3 was the
  candidate floor, 0.0001 to 0.00001; no retraining, selection, tuning, or
  overwrite occurred. Exact-score sensitivities at budgets
  0.125/0.25/0.5/1/2 are Faster R-CNN
  0.2776/0.3664/0.4858/0.6000/0.6978 and YOLO11s
  0.1799/0.2664/0.3821/0.5075/0.6090. Nine runs exceed 2 FP/image; YOLO seed
  137 ends at sensitivity 0.5634 and 1.9907 FP/image, so only its 2-FP/image
  contribution is floor-limited. Assigning that missing contribution the
  mathematical maximum sensitivity 1.0 gives a YOLO aggregate upper bound of
  0.6963, below Faster R-CNN's observed 0.6978; ordering at 1 or 2 cannot
  reverse. D-013 adopts v4 for current reporting. Another lower-floor run has
  limited quantitative-completeness value and is not authorized (Session 89 /
  Batch 42 continuation). Verification passes: 51 relevant tests, 320 full-
  suite tests with one declared skip, 26 final focused/verifier tests, Ruff
  lint/format, 58 scientific artifacts/170 inputs/187 references, 37 paper
  claims/guards, historical v2/v3 hash preservation, figure review, and Git
  whitespace. Nothing was staged, committed, pushed, or retrained.
- **Approved 0.0001 FROC inference complete; residual 0.00001 proposal is
  approval-gated:** after the user's explicit approval, Batch 42 hash-checked
  all ten immutable checkpoints (962,924,817 bytes) and ran inference only on
  the same 750-image/268-box test split for each model. New v3 bundles preserve
  seeds 17/42/137/271/314, Lung Opacity, preprocessing, IoU 0.50 matching,
  native NMS IoU 0.50, and 100 detections/image; no retraining or weight update
  occurred and no Phase 5 artifact was overwritten. The 0.0001 exact-score
  sensitivities are Faster R-CNN 0.2776/0.3664/0.4858/0.6000/0.6978 versus
  YOLO11s 0.1799/0.2664/0.3821/0.4985/0.5881 at budgets
  0.125/0.25/0.5/1/2. Relative to the 0.001 analysis, YOLO changes by
  0/0/+0.0052/+0.0246/+0.1082; the detector ordering never reverses. All ten
  runs are monotone and all 50 budget selections match the canonical matcher.
  The new floor removes the former 0.5-FP/image limit, but YOLO seed 137 still
  ends at 0.7427 FP/image and therefore limits the 1- and 2-FP/image
  aggregates. D-012 adopts v3 for current reporting and retains those values as
  lower bounds. A detector-neutral 0.00001 v4 config is prepared for the same
  ten models/750 images per model but was not run; new explicit approval is
  required. The 56-artifact/163-input/184-reference verifier and 35-claim
  verifier pass; 45 relevant tests and the full 316-test suite pass with one
  declared metadata-only skip. Nothing was staged, committed, pushed, or
  retrained (Session 88 / Batch 42).
- **Exact-score FROC repair complete; lower-floor inference requires user
  approval:** Batch 42 verified the Batch 41 baseline, reproduced the
  historical n=3 operating-point table byte for byte, and reverified the
  unchanged Batch 35 five-run grid inputs. Added a config-driven offline
  analysis that evaluates every unique score in all ten frozen Phase 5 test
  bundles with deterministic descending thresholds, an empty upper sentinel,
  and a 0.001 candidate-floor endpoint. Images, 268 annotations, Lung Opacity
  class, IoU 0.50 matching, NMS IoU 0.50, maximum 100 detections/image, five
  seeds including 271, and FP/image budgets 0.125/0.25/0.5/1/2 are unchanged.
  Faster R-CNN remains higher at all five budgets: 0.2776/0.3664/0.4858/
  0.6000/0.6978 versus YOLO11s 0.1799/0.2664/0.3769/0.4739/0.4799. Relative
  to the n=5 historical grid, the gap strengthens at 0.125, weakens at the
  other four budgets, and never reverses. The frozen 0.001 floor genuinely
  truncates YOLO11s for one run at 0.5, two at 1, and all five at 2 FP/image;
  those affected aggregate values are lower-bound observations. D-011 adopts
  “observed exact-score frontier” and “prespecified FP/image operating
  budgets.” A detector-neutral 0.0001 floor is supported by both adapters and
  proposed for inference over the exact ten checkpoints and same 750 images
  per model, but was not run. Full focused/relevant/full tests (310 passed,
  one declared skip), deterministic rerun hashes, Ruff, the 50-artifact
  verifier, the 35-claim paper verifier, and Git whitespace pass. No training,
  lower-floor inference, checkpoint/prediction replacement, commit, or push
  occurred (Session 87 / Batch 42).
- **V7 baseline re-audited and manuscript hierarchy adjudicated:** Batch 41
  began from `main`/`origin/main` at
  `42826df37468a1c0b28439a7120cc5f3d83f9168`; the only change after the latest
  handoff SHA was a tracked PDF-only commit, so the completed V5 scientific
  state remains consistent. All ten local Git-ignored release checkpoints are
  present and match the release manifest, Phase 5 checkpoint references,
  configs, sizes, and SHA-256 values. All ten tracked test prediction bundles
  are present and hash-valid. The Phase 14 validation inventory remains n=3:
  seeds 17/42/137 exist for both detectors, while seeds 271/314 are absent for
  both and require four inference-only bundles in Batch 43. Historical n=3
  threshold selection and current n=5 operating-regime sensitivity are intact;
  exact original config bytes survive in the frozen n=3 copies despite later
  archive-routing changes to the live paths. Current FROC is a non-interpolated
  0.01--0.99 grid redescription, while frozen bundles retain candidates down to
  0.001, leaving exact-score/floor adjudication for Batch 42. D-010 confirms
  `report/paper_draft.md` as the sole canonical editable journal manuscript;
  the long Markdown is a noncanonical alternate, its 25-page PDF is a rendered
  derivative without a committed source/build recipe, and `report/report.md`
  remains historical. The long/PDF line has unsupported declarations,
  overclaiming, stale process language, and incomplete limitations, so no text
  was migrated. Both production verifiers pass and the focused integrity suite
  passes 21/21 using an external pytest temp root to avoid the known local ACL.
  No numerical result, manuscript claim, threshold, scientific artifact,
  checkpoint, prediction, config, training, inference, stage, commit, or push
  occurred. Full evidence and remaining V7/inference/external/human work are in
  `docs/V7_Q2_VINDR_BASELINE_AUDIT.md` (Session 86 / Batch 41).
- **Surgical manuscript/reporting correction complete; stop for owner diff
  review:** started from tracked-clean `main` at tagged release merge
  `0c94924` and preserved every pre-existing untracked local instruction,
  handoff, audit, and historical log item. Corrected the current paper's AP,
  D-ECE, FROC, and throughput wording without changing any scientific value:
  AP is independent only of the selected single operating threshold and remains
  conditional on retained predictions/common evaluation; FROC claims are
  bounded to the evaluated 0.01--0.99 score sweep and retain the explicit
  non-global-asymptote statement; higher YOLO11s D-ECE is named higher
  calibration error; and the approximate threefold throughput result is tied
  to the detector-specific profiling procedure, stated laptop/software stack,
  and implementation asymmetry. Synchronized only the affected manuscript-
  facing documentation, three textual claim regexes, the reporting-checklist
  manuscript hash, and historical-audit tense. `report/report.md`,
  `docs/LIMITATIONS.md`, declarations, all `results/`, frozen n=3 artifacts,
  code, configs, and numerical claim bindings/tolerances remain unchanged.
  Verification passes: 35 paper claims; 46 scientific artifacts/110 inputs/159
  references; 300 tests with one expected skip; Ruff format/lint; lock check;
  package smoke; manuscript-hash check; and Git whitespace. Nothing was staged,
  committed, pushed, regenerated, trained, or inferred (Session 84 /
  BATCHES_V6 Batch 1).
- **v2.0.0 paper-artifact release candidate prepared, not published:** began
  from clean tracked `main`/`origin/main` at
  `226034c4f8b74b1bc3edfcd3da2dce7bd8c74d98`, whose exact-sha Foundation CI
  run 57 (`33586066060`) is green on Ubuntu and Windows. Current evidence shows
  annotated tags `v1.0.0` and `v1-course-submission` both peel to
  `3a3808841795938a296d48ae3b379b0d10ef3d48`; GitHub has one release,
  `v1.0.0`, with no assets, and its body defines that commit as the immutable
  first stable research baseline. The v1-to-HEAD delta materially expands the
  paper artifact across five-run/estimand-corrected analysis, operating-regime/
  calibration/acquisition/XAI sensitivities, scientific/claim verification,
  locked cross-platform CI, and the current manuscript. D-009 therefore adopts
  one version for the Python project, importable package, lockfile root, and
  research release: `2.0.0`, with proposed tag `v2.0.0` and title `Medical
  Object Detector Benchmark v2.0.0 — Paper Artifact Release`. Prepared
  `CITATION.cff`, a research-oriented changelog, release notes, README and
  reproducibility landing updates, and aligned package versions. The locked
  local gate passes: lock/sync, exact Ruff scopes including the manifest
  builder, 300 tests plus one declared conditional skip, both verifiers (46
  artifacts/159 references and 35 claims), package smoke, YAML citation/version
  checks, whitespace, and tracked large/private-file audits. Scientific and
  checkpoint manifests, all results, canonical/historical manuscripts, and
  claim bindings remain byte-unchanged. The candidate changes are intentionally
  uncommitted, so there is no exact candidate SHA or CI run yet; no tag, GitHub
  release, dataset/checkpoint upload, push, or publication occurred (Session 83
  / paper-artifact release preparation).
- **Foundation CI repair published and remotely green on both platforms:** at
  the user's explicit direction, fast-forwarded `main` from the Batch 38 audit
  base through the reviewed test repair `e5d44fd` and workflow modernization
  `470d958`, then pushed directly. GitHub run 55 proved the actual pytest
  regression was fixed on both Ubuntu and Windows (300 passed, one expected
  skip), but the newly reached scientific verifier exposed CRLF-derived
  manifest hashes recorded from an old Windows worktree rather than the LF
  bytes committed under `.gitattributes`. Corrected the audit-JSON binding in
  `de6d1f6`, used run 56 to identify the remaining test-split and Grad-CAM CSV
  bindings, and corrected all occurrences in `226034c`. Semantic JSON
  comparison, Git blob/index checks, and a temporary clean-archive verification
  established that only the binding metadata was stale; the underlying
  scientific artifacts and CSV/JSON meaning were unchanged. `foundation-ci`
  run 57 (`33586066060`) completed successfully at
  `226034c4f8b74b1bc3edfcd3da2dce7bd8c74d98`: every lock, sync, Ruff, 300-test,
  scientific-verifier, paper-verifier, and package-smoke step passed on both
  `ubuntu-latest` and `windows-latest`. `main` and `origin/main` are identical
  with zero divergence. No scientific result, metric, threshold, seed,
  tolerance, frozen artifact content, checkpoint, figure, or manuscript claim
  changed. The software-integrity release gate is restored; the separate human
  declaration and checkpoint-availability submission blockers remain (Session
  82 / Foundation CI publication).
- **Foundation CI regression repaired on a dedicated local branch:** the batch
  started on `main` at `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`; while the
  investigation was running, the separately authored/pushed Batch 38 audit
  commit advanced `main`/`origin/main` to
  `b858772fe0d6abdcd68269b5163b4776a3acae0b`, which was preserved as the
  repair base. GitHub runs 51--54 and a local archive of the tracked tree all
  identify one failure on both Ubuntu and Windows:
  `tests/test_threshold_sweep.py::test_n5_sensitivity_config_retains_all_five_runs_and_seed_271`.
  The Batch 35 test added at `fe304f7` called production
  `_validate_upstream`, which correctly requires the gitignored held-out COCO
  annotation file; a populated research workspace masked that clean-checkout
  fixture defect. The test now checks the tracked Phase 5 config and summary,
  all ten tracked prediction-bundle hashes, bundle identity/evaluator/
  annotation bindings, and the retained seed-271 zero-detection observation
  without requiring ignored data. Production validation and every scientific
  safeguard remain unchanged. `actions/checkout` moved from the pinned v4
  commit `11d5960a326750d5838078e36cf38b85af677262` (Node 20) to pinned v7.0.1
  commit `3d3c42e5aac5ba805825da76410c181273ba90b1` (Node 24), and
  `astral-sh/setup-uv` moved from pinned v7 tag-object SHA
  `94527f2e458b27549849d47d273a16bec83a01e9` to pinned v10.0.1 commit
  `20cfd1bf945f4377ade1205e4dbc17946fc9a30d`. The workflow still installs uv
  `0.11.27`, retains the Ubuntu/Windows matrix, and uses the unchanged lockfile.
  The complete lock/sync/Ruff/300-test/two-verifier/smoke/whitespace gate passes.
  On this managed Windows Codex host, setting `UV_CACHE_DIR` to a writable temp
  directory avoids an orchestration-only ACL mismatch between sandbox and
  elevated identities; clean GitHub runners do not show that issue. Local
  branch `fix/foundation-ci` ends at commits `e5d44fd` and `470d958`; it has not
  been pushed or merged, so remote CI still requires a branch push/review run
  before release preparation (Session 81 / CI software-integrity batch).
- **Batch 38 final audit published directly to `main`:** fetched `origin` and
  confirmed local/remote started at audited manuscript commit `9aaf414b` with
  zero divergence. Staged exactly one reviewed public file,
  `docs/FINAL_SUBMISSION_AUDIT.md`; internal rule/spec/state files and unrelated
  historical logs remained outside the commit. The staged snapshot passed Git
  whitespace validation, and the audit records the same-session 300-test,
  Ruff, smoke, lock, scientific-verifier, and paper-verifier results. Commit
  `b858772fe0d6abdcd68269b5163b4776a3acae0b` (`Add final submission audit`) was
  pushed directly to `origin/main`; local and remote hashes are identical with
  zero divergence. The audit disposition remains NOT READY FOR SUBMISSION
  pending human declarations and the availability/release decision (Session 80
  / Batch 38 publication).
- **Batch 38 final adversarial audit complete; not ready for submission:**
  audited committed HEAD `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`
  without assuming prior-batch success. Re-ran the full 300-test suite (one
  expected metadata-only skip), Ruff format/lint, package smoke, `uv
  lock --check`, the 46-artifact scientific verifier, and the 35-claim paper
  verifier; all substantive gates pass. Traced 29 manuscript numbers from the
  abstract through limitations to exact committed cells/calculations with no
  mismatch outside declared tolerances. Rechecked statistical/DCA/threshold/
  XAI/radiography/DICOM/calibration language, every major analysis scope and
  seed-271 boundary, H1--H6, all 29 current citation keys, the CLAIM-manuscript
  hash, README hierarchy/commands, 13 manuscript links, 72 supplement links,
  and all ten locally referenced checkpoints. No scientific or manuscript
  language blocker was found. The final disposition is **NOT READY FOR
  SUBMISSION** because all human-only declaration decisions remain unresolved
  and the checkpoint archive has no public URL; the correct state is ready for
  author review. Full evidence and exact commands are in
  `docs/FINAL_SUBMISSION_AUDIT.md`. No scientific code, result, manuscript,
  commit, or remote state changed (Session 79 / Batch 38).
- **Batch 37 published directly to `main`:** fetched `origin` and confirmed
  local/remote both started at Batch 36 commit `23b766d` with zero divergence.
  Staged exactly four reviewed public files: the current manuscript, its claim
  bindings, the regenerated reporting checklist, and refreshed hypothesis
  traceability. Internal rule/spec/state files and unrelated historical logs
  remained outside the commit. The staged snapshot passed 300 tests with one
  expected metadata-only skip, Ruff format/lint, `uv lock --check`, the 35-claim
  paper verifier, the 46-artifact/110-input/159-reference scientific verifier,
  and staged whitespace. Commit
  `9aaf414bdbab337a7c45283d4b32fe58b1e1700d` (`Align manuscript with corrected
  evidence`) was pushed directly to `origin/main`; a post-push fetch confirmed
  identical local/remote hashes and zero divergence (Session 78 / Batch 37
  publication).
- **Batch 37 final evidence-aligned manuscript rewrite complete; stop for
  review:** read the Batch 27--36 decisions, outputs, and source artifacts before
  editing. Rewrote `report/paper_draft.md` as a bounded retrospective internal-
  testing comparison of the two disclosed lung-opacity pipelines while leaving
  `report/report.md` unchanged. The structured abstract now reports the 5,000-
  study/2,136-patient-group cohort, 3,500/750/750 patient-disjoint partitions,
  five attempts per detector, primary training-procedure estimand, absolute AP,
  score/threshold instability, and the nonsignificant primary precision
  interval without a clinical-use claim. Methods now explicitly cover canonical
  preprocessing, asymmetric training recipes, seed/run scope, estimand
  separation, validation-only threshold selection, F-beta preference versus
  hypothetical loss, emitted-detection D-ECE support limits, exclusion of
  invalid DCA, digital versus radiography-motivated synthetic shifts, corrected
  Adebayo controls, and retrospective hypotheses. Results visibly label n=3,
  n=5, n=5/n=4, single-checkpoint, and image scopes; retain seed 271; separate
  descriptive evidence from inference; report absolute endpoints; and disclose
  every n=3-to-n=5 change. Discussion/conclusion preserve the defensible score-
  scale/operating-regime contribution without generalizing to detector families
  or clinical populations. Limitations and explicit author-action declaration
  placeholders cover all adjudicated gaps. Citation-audit corrections remain
  present. Updated three claim-manifest anchors for internal-testing wording,
  refreshed hypothesis links, and regenerated the 44-row CLAIM/STARD-AI/
  TRIPOD+AI crosswalk against manuscript SHA-256
  `0b4cbb4e9effb1c055f0da885ba23636e7adbfd1b885504d20b9a2715943f0f5`
  (24 Yes, 17 No, 3 Not Applicable). Validation passes `uv lock --check`, Ruff
  format/lint, 300 tests with one expected metadata-only skip, the 35-claim paper
  verifier, the 46-artifact/110-input/159-reference scientific verifier, and Git
  whitespace. No training, inference, artifact regeneration, commit, push, or
  historical-report edit occurred; published `main` remains at `23b766d`
  (Session 77 / Batch 37).
- **Batch 36 published directly to `main`:** fetched `origin` and confirmed
  local/remote started identical at Batch 35 commit `fe304f7` with zero
  divergence. Staged exactly 13 reviewed public files: CI, README/reproducibility
  wording, the single machine-readable manuscript wording change, three
  manifests, four scripts, and two verifier test modules. Internal rule/state
  files and unrelated failed, aborted, smoke, and orchestration logs remained
  outside the commit. The staged snapshot passed 300 tests with one expected
  metadata-only skip, Ruff format/lint, `uv lock --check`, package smoke, both
  verifiers, staged whitespace, and the staged-byte checkpoint-manifest binding.
  Commit `23b766dde47784764e14d29c76e0b23f107e33f6` (`Add scientific artifact
  verification`) was pushed directly to `origin/main`; a post-push fetch
  confirmed identical local/remote hashes and zero divergence (Session 76 /
  Batch 36 publication).
- **Batch 36 artifact-level reproducibility boundary complete; stop for review:**
  added a reviewed 46-item scientific artifact manifest that binds every
  manuscript-critical CSV/JSON/PNG to its SHA-256, exact generator/config
  hashes, input hashes and availability, frozen schema, study phase,
  reproduction tier, and GPU/training requirements. A strict verifier now
  checks artifact/generator/config/input staleness, CSV/JSON/PNG schemas, and
  159 declared result references. A separate claim manifest binds 35 central
  manuscript numbers to exact rows/cells/JSON pointers or allow-listed
  deterministic calculations with explicit rounding tolerances; its verifier
  requires unique manuscript matches and scientific-manifest-listed sources.
  Standard Ubuntu/Windows CI runs both lightweight checks but performs no data
  download, checkpoint loading, GPU inference, or training. README and the
  reproducibility contract now distinguish software verification, committed-
  analysis replay, exact inference, and exact retraining; green CI is explicitly
  not evidence of end-to-end scientific regeneration. The exact ten Phase 5
  best checkpoints all remain local, match their recorded hashes, and total
  962,924,817 bytes. Their release manifest contains hashes/instructions and a
  conditional license/data-policy assessment, but the binaries were not
  committed/uploaded and the public URL is null. The pinned Python/Torch/CUDA/
  cuDNN/driver stack, lockfile roles, RNG controls, `PYTHONHASHSEED` timing,
  DataLoader seeding, Windows worker constraint, and warning-only CUDA ROI Align
  nondeterminism are documented. Validation passes 300 tests with one expected
  metadata-only skip, both verifiers, Ruff format/lint, `uv lock --check`, package
  smoke, Git whitespace, and all ten checkpoint hashes. No scientific analysis,
  model inference/training, artifact auto-refresh in CI, commit, push, or binary
  upload occurred (Session 75 / Batch 36).
- **Batch 35 published directly to `main`:** fetched `origin/main` and confirmed
  local/remote started identical at Batch 34 commit `2f0dc6e` with zero
  divergence. Staged exactly 49 reviewed Batch 35 entries: the four n=5 configs,
  implementation and tests, canonical docs/current manuscript, 16 versioned n=5
  tables/figures, two operating-regime audit tables, and four provenance JSONs.
  Internal rule/state files and unrelated failed, aborted, smoke, and
  orchestration logs remained outside the commit. Staging exposed CRLF-to-LF
  normalization in three provenance JSONs; their atomic writers were corrected
  to emit deterministic LF bytes, the affected summaries were regenerated, and
  all recorded SHA-256 bindings revalidated before publication. The final
  snapshot passed 292 tests with one expected metadata-only skip, Ruff lint and
  format, all four preflights, hash/row/seed/classification/link/whitespace
  checks, visual review, and historical-artifact preservation. Commit
  `fe304f7c2bf26c59e204e7cd83c8cf6190f57857` (`Add five-run operating-regime
  sensitivity`) was pushed directly to `origin/main`; a post-push fetch confirmed
  identical local/remote hashes and zero divergence (Session 74 / Batch 35
  publication).
- **Batch 35 five-run operating-regime sensitivity complete; stop for review:**
  inventoried the ten frozen test-prediction bundles and confirmed that official
  PR, the exploratory threshold sweep/FROC, unchanged validation-selected-
  threshold application, and Pareto can all be recomputed over five runs per
  detector without training, inference, threshold reselection, or test tuning.
  Added separately versioned n=5 configs, tables, figures, and hash-bound
  provenance while leaving every historical unsuffixed n=3 artifact unchanged.
  Seed 271 is retained exactly as observed: nonzero AP and low-threshold FROC,
  but zero detections and defined zero operating metrics at 0.25 and the frozen
  YOLO11s threshold 0.05. D-008 makes the n=5 sensitivity the current
  manuscript's principal operating-regime display because it covers the complete
  predeclared attempt population; the n=3 analysis remains visible as historical/
  pre-specified provenance. The thresholds remain 0.69/0.05 selected from n=3
  validation evidence. A 19-row comparison records 10 strengthened, four
  weakened, five unchanged, and zero reversed conclusions. Regression tests
  explicitly require all five runs and prevent automatic filtering of seed 271.
  No retraining, checkpoint/model loading, inference, or test-set selection
  occurred (Session 73 / Batch 35).
- **Batches 31--34 published directly to `main`:** fetched `origin/main` and
  confirmed local/remote started identical at Batch 30 commit `fed66b2` with
  zero divergence. Staged exactly 46 reviewed Git entries covering the Batch
  31 XAI correction, Batch 32 DICOM/acquisition audit, Batch 33 calibration
  support remediation, and Batch 34 current-manuscript reporting, hypothesis,
  citation, declaration, and limitation audit. The public release included the
  canonical implementation/config/tests/docs/manuscript and versioned
  result/provenance artifacts; `report/report.md` retained its historical role
  and contributed only the previously reviewed three-line Batch 32 claim
  correction. Internal protocol/state/audit files and unrelated failed,
  aborted, smoke, and orchestration logs remained outside the commit. The
  release passed 286 tests with one expected metadata-only skip, Ruff lint and
  format checks, `uv lock --check`, all three module preflights, Git whitespace,
  JSON/CSV parsing, exact citation-key and local-link resolution, sensitive-data
  signature screening, and file-size review. Commit
  `2f0dc6e7cb1ce0b232ef6fe5d1c1ca2d7f7a90a8` (`Correct evidence audits and
  reporting`) was pushed directly to `origin/main`; a post-push fetch confirmed
  identical local/remote hashes and zero divergence (Session 72 / Batches
  31--34 publication).
- **Batch 34 current-manuscript reporting/hypothesis/citation audit complete;
  stop for review:** made the document hierarchy explicit: `report/paper_draft.md`
  is the current manuscript, `report/report.md` is the preserved historical/full
  technical report, `docs/` is the analysis/methodology/audit record, and
  `results/` is the numerical source of truth. Regenerated the reporting audit
  against the paper draft with all 44 CLAIM 2024 rows in official Yes/No/Not
  Applicable form and evidence/explanations. Final STARD-AI 2025 was assessed as
  outside a clean primary diagnostic-accuracy fit and TRIPOD+AI as outside the
  primary individualized prediction-model scope; both are selected crosswalks
  by analogy, not compliance claims. Added complete H1--H6 traceability without
  relabeling any statement as preregistered, an exhaustive audit of all 29
  manuscript citation keys plus the three guideline sources, and unresolved
  author-declaration placeholders. The manuscript now discloses the deterministic
  5,000-study selection, absence of a formal sample-size/power calculation,
  internal-testing terminology, and canonical COCO/SSIM/bootstrap/permutation/
  Holm/DICOM sources. `docs/LIMITATIONS.md` now includes subset/power,
  retrospective-hypothesis/multiplicity, framework-applicability, and declaration
  boundaries. All new local links resolve; 44 CLAIM rows, all H1--H6 fields, 29
  cited-key audit mappings, 32 unique BibTeX keys, and zero unresolved citations
  were verified. `report/report.md`, code, configs, and numerical results were
  not changed in Batch 34; no commit or push occurred (Session 71 / Batch 34).
- **Batch 33 detection-calibration definition/support remediation complete;
  stop for review:** verified the implementation against Küppers et al. The
  canonical endpoint uses emitted confidence plus relative center x/y and
  width/height, with predicted class as a categorical stratum; this one-class
  dataset has 3,125 possible primary cells. Matching is stable descending
  score, same class, highest-IoU unmatched target, no target reuse, IoU
  `>=0.50`, score `>=0.001`, and 100 detections/image. Equal-width bins use
  upper-edge assignment and final-bin clipping at 1.0. Supported cells are
  weighted by their share of all emitted detections; cells below eight
  contribute zero without removing their detections from the denominator.
  The provenance distinguishes definition alignment from the paper's detector
  demonstration settings (0.3 probability threshold, NMS IoU 0.6, and
  correctness IoU 0.6/0.75); the project's floor 0.001 and matcher IoU 0.50
  remain frozen project choices.
  Removed the seed-specific plot/provenance branches and the retrospective
  method expectation that seed 271 should be worst; every configured run is
  dynamically discovered and uses the same pipeline. Versioned v2 outputs add
  exact per-run occupancy/support, a predeclared 3/5/7-bin by 1/4/8/16-minimum
  grid, and 0.001/0.005/0.01/0.05 floor sensitivity while retaining the
  original 5-bin/minimum-8/floor-0.001 estimate. Only 68--354/3,125 cells are
  occupied and 15--169 supported per run; supported detection fractions span
  0.543--0.983. The 0.005 floor retains only 46.6%/53.1% of Faster/YOLO
  detections on average, so population sensitivity is material; a run with no
  0.05 detection is undefined, not removed or zero. The reliability figure is
  explicitly confidence-only marginal. Manuscript/docs call D-ECE descriptive,
  exclude missed targets and clinical/exam risk, and keep exam-level
  probability calibration/DCA separate. Validation passes 286 tests with one
  expected skip, focused tests, Ruff, preflight, provenance/hash audits, visual
  figure review, and Git whitespace. No training, inference, calibrator fit,
  commit, or push occurred (Sessions 69--70 / Batch 33).
- **Batch 32 DICOM/acquisition-claim correction complete; stop for review:**
  audited every DICOM in the frozen 300-image acquisition-shift subset against
  current DICOM PS3.3/PS3.4 2026c. All 300 objects are Secondary Capture Image
  Storage, `Modality=CR`, `MONOCHROME2`, workstation-converted (`WSD`), and
  marked as previously lossily JPEG-compressed. Pixel Intensity Relationship/
  Sign, Modality LUT/rescale, VOI LUT/Window Center/Width/function,
  Presentation Intent Type, and processing-description fields are missing in
  all 300, so the stored 8-bit values cannot be shown to be approximately
  linear or logarithmic in incident X-ray signal or reliably inverted to that
  scale. The DICOM `LINEAR` alternatives are class-A display-transform
  sensitivities; the Poisson-like conditions are class D for dose/quantum
  interpretation and retained only as class-B generic signal-dependent
  intensity perturbations; Gaussian kernels are class-C generic blur/spatial-
  resolution proxies. Standard numeric examples at center/width 2048/4096,
  2048/1, and 0/100 plus MONOCHROME polarity regressions pass. A 3,000-row
  diagnostic quantifies per-image perturbations before/after canonical min-max:
  the wider window is almost completely cancelled (264/300 exactly identical),
  the center shifts are partly cancelled, and blur differences can be re-
  stretched. DSI remains descriptive and does not estimate site
  transportability. CPU-only audit artifacts and corrected manuscript/docs
  were produced; historical Phase 22 results and all 20 prediction bundles
  remain unchanged and hash-bound, so no GPU inference ran. Validation passes
  279 tests with one expected skip, Ruff, lock, preflight, source/artifact
  identity, link/citation, and whitespace checks. No commit or push occurred
  (Session 68 / Batch 32).
- **Batch 31 XAI sanity-method correction complete; stop for review:** audited
  the historical implementation and confirmed that its so-called data
  randomization shuffled RGB pixel vectors only at inference, while its model
  control reinitialized all weights once. The former is now named an
  input-pixel randomization control and explicitly is not Adebayo et al.'s
  training-label data-randomization test; the latter is retained as a full
  model-parameter randomization control, not a cascading reproduction. The
  canonical training-label test was not performed because it would require
  retraining identical detectors on randomized annotations and demonstrating
  task fit. A versioned v2 run adds six cumulative, detector-specific
  output-to-backbone parameter groups (137 Faster R-CNN and 169 YOLO11s
  weight/bias-bearing modules fully partitioned), fresh deterministic model
  copies per stage, checkpoint before/after hashes, and Pearson, tie-aware
  Spearman, and Gaussian-window SSIM after 40-by-40 resampling and independent
  min-max normalization. The same frozen 50-image subset produced 700 detail
  rows and 14 summaries. Full-randomization Pearson/Spearman/SSIM means were
  `0.0025/0.0287/0.0381` for Faster R-CNN and
  `0.0242/0.0795/0.0476` for YOLO11s; these support parameter sensitivity only.
  Input-pixel-control means were `-0.0206/-0.0188/0.2468` and
  `-0.0021/0.0133/0.0067`, respectively, and remain perturbation-stress
  evidence rather than data-label-dependence evidence. Intermediate cascade
  behavior is non-monotonic and descriptive. Phase 7 energy-in-box, pointing,
  and box-area values remain descriptive; small absolute overlaps/lifts are
  not characterized as strong localization. Historical Batch 21 artifacts
  remain byte-identical. The v2 summary/detail/panel/summary-JSON hashes are
  `23a8b47c...`, `4e765237...`, `7eac3805...`, and `88d386c2...`; checkpoint
  hashes were unchanged. Validation passes 274 tests with one expected skip,
  Ruff format/lint, lock, preflight, provenance, link/citation, and whitespace
  checks. No detector training, canonical data randomization, commit, or push
  occurred (Session 67 / Batch 31).
- **Batch 30 published directly to `main`:** reloaded Session 65 state,
  fetched `origin/main`, and confirmed local/remote started identical at
  `22a814007a062347fd683645e4e9eb71367b14bc` with zero divergence. Staged
  exactly 27 reviewed Batch 30 filesystem paths, represented as 23 Git entries
  because the original config, table, figure, and Phase 20 summary became 100%
  archive renames. The original DCA source was also preserved as a byte-
  identical archive copy while its canonical path became the probability-
  semantic guard. Internal protocol/state/audit files and unrelated run
  diagnostics remained unstaged. The release snapshot passed 270 tests with
  one expected skip, Ruff format/lint, `uv lock --check`, raw-score preflight,
  Git whitespace, exact-scope and archive-to-HEAD checks, stable artifact/
  provenance hashes, 198-row zero-difference audit, 322 local links, citation
  resolution, and sensitive/stale-path screening. Commit
  `fed66b22b2bfda37a665260c276402a18e8d83c7` (`Correct decision-curve
  interpretation`) was pushed directly to `origin/main`; a post-push fetch
  confirmed identical hashes and zero divergence (Session 66 / Batch 30
  publication).
- **Batch 30 DCA correction complete; stop for review:** verified that the
  Batch 20 implementation used `maximum emitted confidence >= tau` to define
  action and reused that raw `tau` in the `tau/(1-tau)` false-positive weight.
  Per Vickers/Elkin 2006 and Vickers et al. 2008, this is non-standard because
  the detector score was never converted to a predicted exam-outcome
  probability on the threshold-probability scale. D-007 removes the curves and
  every scenario-linked claim from the main paper Results and retains the
  calculation only as an exploratory raw-score threshold utility/sensitivity
  artifact in Supplementary/Limitations. Probability-based salvage was not
  forced: frozen validation predictions cover only six of ten retained runs,
  with both detectors' seeds 271 and 314 missing. No calibrator family or
  hyperparameters were selected; no calibration, training, or inference ran.
  The exact Batch 20 source/config/table/figure/provenance remain hash-audited
  pre-Batch-30 archives. The relabeled 198-row table is numerically identical
  over all 39 historical fields (maximum absolute difference zero), and the
  new figure/captions/provenance declare the arbitrary raw-score scale and
  non-standard status. The conventional-DCA helper now accepts only a typed
  validation-frozen outcome-probability input and a separately typed elicited
  decision threshold; regression tests reject raw detector scores, untyped
  arrays, raw confidence cutoffs passed as `p_t`, and plain scalars. Validation
  passes 270 tests with one expected skip, focused tests, Ruff format/lint,
  preflight, archive/numeric/
  provenance checks, figure review, lock and Git whitespace checks. Canonical
  table/figure/summary hashes are `0bad365d...`, `8d91b2ed...`, and
  `c7ae8123...`. `docs/HYPOTHESES.md` required no edit because it contained no
  DCA claim. No commit or push occurred (Session 65 / Batch 30).
- **Batch 29 published directly to `main`:** reloaded Session 63 state,
  fetched GitHub, and verified local/remote both started at `fb4709b4` with
  zero divergence. Staged exactly 24 Batch 29 public filesystem paths (21 Git
  entries after three 100% archive renames): implementation/config/tests,
  corrected decisions/docs/manuscripts, five canonical artifacts, and three
  content-identical historical archives. Internal protocol/state/audit files
  and unrelated failed/aborted/orchestration diagnostics remained unstaged.
  The release snapshot passed exact-scope and staged-byte checks, 266 tests
  with one expected skip, Ruff format/lint, `uv lock --check`, validation-only
  threshold preflight, Git whitespace, 352 local links, JSON/CSV/hash/frequency
  audits, archive-to-HEAD identity, and sensitive/stale-path screening. Commit
  `22a814007a062347fd683645e4e9eb71367b14bc` (`Correct F-beta threshold
  sensitivity`) was pushed directly to `origin/main`; a post-push fetch
  confirmed identical hashes and zero divergence. GitHub Actions
  `foundation-ci` run `33242465877` passed on Ubuntu and Windows (Session 64 /
  Batch 29 publication).
- **Batch 29 threshold terminology and stability remediation complete; stop
  for review:** D-006 supersedes D-004's unsupported clinical-cost
  interpretation while retaining its validation-only precedence rule. The
  implementation is now documented exactly as recall-weighted F-beta
  sensitivity,
  $F_\beta=(1+\beta^2)TP/[(1+\beta^2)TP+\beta^2FN+FP]$, where beta is a
  recall-versus-precision preference and beta squared is a harmonic-mean
  recall weight, not measured harm. The frozen beta 1/3/5/10 sweep and original
  bootstrap stream reproduce the eight historical thresholds exactly.
  Validation-only stability output adds contiguous near-optimal-LCB plateaus,
  draw-specific selected-tau intervals, and all-candidate selection
  frequencies. A separate explicitly hypothetical linear detection-error
  analysis minimizes `r * FN / N + FP / N` for assumed r 1/9/25/100; it is not
  substituted for F-beta or described as deployment utility. The selection
  path now rejects a non-validation role/split, any manifest without a false
  test-access flag, and annotation/validation-manifest identity mismatch.
  Canonical artifacts use new Batch 29 paths; exact pre-remediation artifacts
  remain content-identical archives. Documentation, tables, figure, decision
  log, README, and both manuscripts are synchronized; `docs/HYPOTHESES.md`
  required no edit because it contained no clinical-cost claim. A second run
  reproduced all five canonical hashes byte-for-byte. Validation passes 266
  tests with one expected skip, Ruff format/lint, `uv lock --check`, preflight,
  Git whitespace, row/frequency/provenance audits, and visual figure review.
  No test labels, training, inference, threshold propagation, commit, or push
  occurred (Session 63 / Batch 29).
- **Batch 28 published directly to `main`:** reloaded the Session 61 handoff,
  audited the exact public scope, fetched GitHub, and confirmed local/remote
  started identical at `df2f35f`. Staged exactly 22 Batch 28 files: the
  implementation/config/test changes, synchronized statistical/manuscript
  documentation, regenerated clean table/provenance, paired-seed sensitivity
  archive, and three seed-influence tables. Internal protocol/state files,
  `docs/REVIEW_REMEDIATION_AUDIT.md`, host logs outside the tracked Phase 8
  provenance, and unrelated failed/aborted diagnostics remained excluded. The
  staged snapshot passed 262 tests with one expected skip, Ruff format/lint,
  `uv lock --check`, statistics preflight, Git whitespace, 152 local-link
  checks, JSON validation, credential-pattern screening, LF/hash checks, and
  exact-scope checks. Commit `fb4709b4995d85a05b6acfb20d3d1946d7cefdfa`
  (`Correct statistical inferential targets`) was pushed directly to
  `origin/main`; a post-push fetch confirms identical hashes and zero
  divergence. GitHub Actions `foundation-ci` run `33224313738` passed both
  Ubuntu and Windows jobs (Session 62 / Batch 28 publication).
- **Batch 28 estimand separation and independent-run inference complete; stop
  for review:** D-005 defines the primary training-procedure estimand as
  uncertainty over held-out NIH patients and stochastic trained-run
  variability, and the secondary checkpoint-conditional estimand as patient
  uncertainty with the observed checkpoints fixed. The seed audit rejects
  same-number Faster R-CNN/YOLO11s seeds as matched stochastic blocks: although
  stochastic augmentation is disabled in both arms, loaders/batches,
  initialization paths, frameworks, RNG-consumption sequences, and stopping
  trajectories are not coupled into a common-random-number design. The clean
  bootstrap therefore resamples 323 patient clusters jointly and trained runs
  independently within detector, reconstructing every nonlinear endpoint from
  frozen sampled predictions. Unconditional endpoints use 5/5 runs;
  matched-detection IoU/Dice use 5 Faster R-CNN and 4 YOLO11s defined runs.
  The old paired-seed table/summary are frozen as historical sensitivity
  archives; checkpoint-conditional patient-cluster permutation p-values remain
  separate, Holm-adjusted sensitivity columns rather than a second proof of the
  training-procedure claim. Under the primary intervals, recall, F1, mAP@0.5,
  and mAP@0.5:0.95 remain wholly positive; precision crosses zero despite its
  small checkpoint-conditional p-value; IoU/Dice cross zero. Seed 271 is
  retained wherever defined, and per-run, leave-one-run-out, and descriptive
  leave-one-label-out artifacts expose its influence without treating deletion
  as correction. No retraining, new p-value, robustness recomputation, or test-
  set change occurred. Manuscripts and statistical documentation now name the
  target for every inference. Validation passes 262 tests with one expected
  skip, Ruff format/lint, uv lock, statistics preflight, Git whitespace, and
  artifact-column/hash audits; the clean table is `cf17182a...` and the frozen
  robustness table remains `59c9ec17...` (Session 61 / Batch 28).
- **Independent review-remediation audit complete; stop for review:** created
  `docs/REVIEW_REMEDIATION_AUDIT.md` without changing scientific code,
  committed result artifacts, or either manuscript. The audit records `main`
  at `df2f35f43170699cca32540a283a50caa1764aee`, a baseline worktree with
  zero tracked modifications and 37 pre-existing untracked entries, the sole
  software-QA CI workflow, a fresh local pass (257 tests, one expected skip,
  Ruff format/lint, lock check, deterministic smoke), and the currently
  documented canonical-file hierarchy. It independently adjudicates all 17
  requested findings, maps H1--H6, verifies all 24 repository Markdown
  citation keys plus CLAIM/TRIPOD+AI/STARD-AI/DICOM standards sources, and
  checks 18 manuscript quantities directly against committed tables/manifests;
  every numerical value matches. Three findings are blocking before
  submission: the same-number detector seeds are not demonstrated matched
  stochastic blocks despite paired seed resampling; `beta^2` is not
  automatically a calibrated clinical FN:FP cost ratio; and the XAI
  “data-randomization” test is pixel-vector shuffling rather than Adebayo
  training-label randomization. The DICOM header audit cannot establish stored
  values proportional to beam intensity, so the existing synthetic-proxy
  wording survives. The public v1.0.0 release exists but exposes no attached
  model assets; ten final best checkpoints remain present only at ignored local
  paths. No remediation was implemented pending user review (Session 60 /
  audit-only review-remediation batch).
- **Batch 26 final consistency pass complete and published:** Batch 25 remained
  intentionally unrun under its optional gate. Audited the quantitative claims
  in `report/paper_draft.md` across 12 evidence domains against tracked
  machine-readable artifacts covering cohort construction, clean metrics,
  seed-271 stability, PR/threshold/FROC, cost-sensitive thresholds,
  calibration, compute, DCA, digital robustness, acquisition shifts,
  Grad-CAM/sanity, and patient-cluster statistics; no manuscript value required
  correction. Repository-wide citation auditing found 24 unique BibTeX entries
  and 24 unique citation keys, with no missing, unused, duplicate-key,
  duplicate-DOI, duplicate-title, or duplicate-URL records; Küppers et al. and
  Vickers/Elkin both resolve. Added the missing Batch 20 DCA prevalence,
  nominal-score, same-test, localization, interval, and utility boundaries plus
  the Batch 23 reporting-governance gaps to `docs/LIMITATIONS.md`. Regenerated
  README's paper/report command index and explicitly covered all six executable
  modules added in Batches 18--23, including the previously omitted Batch 22
  acquisition-shift runner. Local validation passed 257 tests with one expected
  metadata-only skip, Ruff format/lint, uv lock, deterministic smoke, and Git
  whitespace checks. Committed exactly `README.md` and
  `docs/LIMITATIONS.md` as
  `df2f35f43170699cca32540a283a50caa1764aee` (`Complete final consistency
  pass`) and pushed to `origin/main`; GitHub Actions run `33173026686` passed
  both Ubuntu and Windows matrix jobs (Session 59 / Batch 26).
- **Batch 24 published directly to `main`:** fetched live `origin/main` and
  confirmed zero starting divergence at `d992cf8`. Staged exactly
  `report/paper_draft.md` and `report/references.bib`; internal state/protocol
  files, host environment records, smoke artifacts, orchestration output, and
  historical failed/aborted diagnostics remained excluded. The staged snapshot
  contained the audited 7,855-word manuscript and 257-word abstract; all 14
  local links and 20 citation keys resolved, no sensitive path/credential
  pattern matched, staged blobs equaled the audited working bytes, and Git's
  whitespace check passed. Committed the two-file snapshot as
  `ce166a59aeec69f6585ad5071286f4eb16fbee8d` (`Add controlled comparison
  paper draft`) and pushed it directly to GitHub `origin/main`. A post-push
  fetch confirms identical local/remote hashes and zero divergence (Session 58
  / Batch 24 publication).
- **Batch 24 first full paper draft complete; stop for review:** created
  `report/paper_draft.md` as a publication-style IMRaD manuscript while leaving
  the complete technical/reproducibility report byte-unchanged. The 7,855-word
  draft has a 257-word methods-first abstract and integrates every Batch 9--23
  result domain: clean n=5/n=4 evidence, frozen n=3 threshold/PR/FROC/Pareto
  analyses, validation-selected and D-006 recall-weighted F-beta thresholds,
  five-seed D-ECE and DCA, patient-cluster inference, digital and raw-array
  robustness, Grad-CAM localization/sanity, and the Batch 23 raincloud and
  reporting gaps. The Discussion frames scenario-conditional trade-offs and
  explicitly limits attribution to the two disclosed pipelines per D-002 and
  D-003; the Limitations preserve the consolidated hedging and absent
  ethics/demographic/external evidence. Exhaustive records are routed to
  `docs/SUPPLEMENTARY.md`. Added the missing Vickers/Elkin DCA reference to
  `report/references.bib`. All 14 manuscript-local links resolve, all actual
  citation keys resolve, selected-threshold/calibration/DCA spot checks match
  committed CSVs, and `report/report.md` has no diff (Session 57 / Batch 24).
- **Batch 23 published directly to `main`:** fetched live `origin/main` and
  confirmed zero starting divergence, then staged exactly the 11 audited public
  Batch 23 paths while excluding internal protocol/state files and unrelated
  historical logs. The staging gate exposed Windows CRLF normalization in the
  new provenance JSON; changed the atomic writer to force clone-stable LF,
  added a regression test, and regenerated the artifact. The final figure hash
  remains `0ffdfcf...`; the clone-stable provenance hash is `e50ebf7...`.
  Exact-scope, sensitive-pattern, local-link, aggregate/seed, artifact-byte,
  lockfile, Ruff, format, and whitespace checks pass; 257 tests pass with one
  expected metadata-only skip. Committed the exact snapshot as
  `d992cf852387cd4ecd95b27df5067b3fdb4646bd` (`Add standards reporting and
  raincloud metrics`) and pushed it directly to GitHub `origin/main`. A
  post-push fetch confirms identical local/remote hashes and zero divergence
  (Session 56 / Batch 23 publication).
- **Batch 23 standards reporting crosswalk and supplementary raincloud complete;
  stop for review:** audited the current repository item by item against CLAIM
  2024 (44 items), TRIPOD+AI 2024 (52 subitems), and STARD-AI 2025 (40 main
  items represented by 48 item/subitem rows). Every checked row links to
  repository evidence; incomplete items stay explicitly partial, absent, or
  not applicable. The main gaps are the absent abstract/paper draft, source
  accrual dates, demographic/fairness evidence, local ethics/consent statement,
  registration, funding/conflict disclosures, patient/public involvement, and
  external evaluation. Added a pointer-only supplementary index for full
  seed-level tables, frozen n=3 archives, threshold/calibration/DCA outputs,
  complete 72-row corruption and 20-row acquisition grids, XAI/statistics
  evidence, and the decision log. The config-driven Seaborn raincloud workflow
  audits all 14 aggregate rows against their ten seed records before drawing
  seven predictive and seven compute/hardware panels. It uses 138 finite
  observations: n=5 per detector except conditional YOLO IoU/Dice n=4, with
  seed 271 and its undefined reason printed in both panels; historical n=3
  evidence is not mixed into the figure. Exact pandas/seaborn pins and lockfile
  entries were added. Figure and provenance reproduce byte-for-byte (SHA-256
  `0ffdfcf...` and `990624f...`); 256 tests pass with one expected skip, Ruff
  lint/format, Git whitespace, link/evidence, source-count, and lock checks are
  clean (Session 55 / Batch 23).
- **Batch 22 published directly to `main`:** performed a final publication
  audit, replaced host-absolute artifact paths with repository-relative paths,
  and regenerated the full 6,000-inference GPU analysis without changing any
  scientific field. All 20 compressed 300-image prediction bundles pass
  path/count/hash and sensitive-value checks; smoke output, host environment
  captures, internal state/protocol files, and unrelated historical logs remain
  excluded. The exact 29-file snapshot passed 254 tests with one expected skip,
  Ruff lint/format, Git whitespace, clone-stable hash, and scope checks. Fetched
  live `origin/main`, confirmed zero starting divergence, committed
  `af24c100d9f8e3bc1a3735035948b4fc89d704fb` (`Add radiography
  acquisition-shift analysis`), and pushed it directly to `origin/main` as
  requested. A post-push fetch confirmed identical local/remote hashes and zero
  divergence (Session 54 / Batch 22 publication).
- **Batch 22 raw-radiography acquisition-shift analysis complete; stop for
  review:** the prerequisite filesystem audit found the full ignored raw
  archive still present (26,684 training plus 3,000 competition-test DICOMs),
  and all 300 frozen Phase 6 studies map one-to-one to local training DICOMs.
  Every sampled file is an 8-bit unsigned `CR`/`MONOCHROME2` radiograph with
  no native Window Center/Width, VOI LUT Sequence, Modality LUT, Rescale
  Slope/Intercept, or pixel padding. Reusing the extracted canonical per-image
  min-max scaler reproduces all 300 processed PNGs pixel-for-pixel. A strict,
  config-driven checkpoint-only GPU runner applies four exact DICOM PS3.3
  default-`LINEAR` center/width alternatives, three deterministic
  signal-dependent Poisson count conditions, and three finite Gaussian
  detector/processing blur kernels to raw arrays before scaling. It writes all
  seven shifted/clean ratios and `DSI = 1 - shifted/clean`, with mAP@0.5:0.95
  primary. At the strongest Poisson condition, Faster R-CNN/YOLO11s DSI is
  `0.314529/0.436641`; at 9x9 blur it is `0.248532/0.282869`. VOI results are
  smaller/mixed, including near-neutral wide-window DSI
  `-0.000119/0.000906`. All 20 300-image bundles pass count/hash/identity
  audits; clone-stable CSV and summary hashes are `2cd1428...` and `4782ac7...`
  and are
  byte-identical across a cache-only rerun. The write-up explicitly separates
  this acquisition-motivated internal stress test from Phase 6's generic
  post-conversion digital corruptions and rejects calibrated-dose, vendor-
  preset, scanner, clinical-robustness, or transportability claims. Validation
  is 254 tests passed with one expected metadata-only skip, Ruff lint/format
  and whitespace checks clean (Session 53 / Batch 22).
- **Batch 21 published directly to `main`:** after the user approved export of
  the disclosed de-identified 50-study manifest and qualitative panel derived
  from medical radiographs, fetched live `origin/main` and confirmed local
  `main` was safely one commit ahead and zero behind. Pushed the exact audited
  12-file commit `5cc7d081f66c77698a6c29ca4b8cc3131ef22ab7` (`Add Grad-CAM sanity
  checks`) directly to `origin/main`. A post-push fetch confirmed identical
  local/remote hashes and zero divergence. Internal protocol/state files,
  host-specific environment captures, orchestration output, and historical
  diagnostics remain excluded (Session 52 / Batch 21 publication).
- **Batch 21 committed locally; push awaits explicit sensitive-artifact
  approval:** fetched live `origin/main` and confirmed zero starting divergence
  at `cd155b2`. Staged exactly the 12 public Batch 21 paths and excluded
  internal protocol/state files, host-specific environment captures,
  orchestration output, and historical diagnostics. A final audit caught Git
  LF normalization invalidating Windows-generated CSV hashes; the atomic CSV
  writer now emits LF explicitly, a regression test covers it, and a fresh
  checkpoint-only GPU run preserved all numerical/visual results while making
  the committed hashes clone-stable. The snapshot passed 248 tests with one
  expected skip, Ruff lint/format, whitespace, exact-scope, sensitive-pattern,
  artifact-hash, LF, row-count, and PNG checks. Local `main` commit
  `5cc7d081f66c77698a6c29ca4b8cc3131ef22ab7` (`Add Grad-CAM sanity checks`)
  exists and is one commit ahead of `origin/main`. The push was rejected by the
  safety gate pending explicit user approval after disclosure that it exports
  a 50-study de-identified manifest and a qualitative panel derived from
  medical radiographs to the configured GitHub repository (Session 51 / Batch
  21 publication attempt).
- **Batch 21 Grad-CAM parameter/data sanity checks complete; stop for review:**
  added a config-driven, checkpoint-only GPU extension over a 50-image nested
  sample of the frozen Phase 6 robustness pool. Proportional allocation and
  seed-17 within-pool selection yield 11 Lung Opacity, 22 No Lung Opacity / Not
  Normal, and 17 Normal studies from 41 NIH patients, with 18 boxes; the source
  300-image manifest hash remains `63b4dd70...`. Each image's trained
  highest-score candidate defines one fixed reference region. Both conditions
  use a pre-activation foreground target at that region, the existing stride-16
  layers, and ordinary ReLU Grad-CAM: a fixed ROI classifier for Faster R-CNN
  and the closest raw-anchor center for YOLO11s. Parameter randomization deep-
  copies the trained model, applies seeded `torch.nn.init.xavier_normal_` to
  every module weight tensor, zeros biases, and preserves other buffers; data
  randomization identically permutes RGB pixel vectors for both detectors while
  preserving their multiset. Mean full-resolution Pearson correlations are
  Faster R-CNN `0.001360` (parameter, K=50) and `-0.015859` (data, K=43), and
  YOLO11s `0.023442` (parameter, K=46) and `-0.003357` (data, K=50). Seven
  Faster shuffled-input and four YOLO randomized-weight maps are zero/constant
  and explicitly excluded; no valid map reaches the predeclared `r >= 0.50`
  descriptive sanity-failure threshold. The result passes a basic model/input
  sensitivity check and reinforces Grad-CAM's cautious failure-analysis role,
  but does not alter weak localization or support clinical-reasoning/causal
  claims. Required CSV/PNG plus per-image/provenance artifacts are complete and
  byte-identical across repeated runs; the panel was visually inspected; 248
  tests pass with one expected skip and all static/artifact checks are clean
  (Session 50 / Batch 21).
- **Batch 20 published directly to `main`:** fetched live `origin/main` and
  confirmed zero starting divergence at `cdfcdc0`. Staged exactly the nine
  public DCA code/config/test/document/result/provenance paths; internal
  protocol/state files, orchestration output, and historical failed/aborted
  diagnostics remained excluded. The staged snapshot passed 240 tests with one
  expected skip, Ruff lint/format, whitespace, exact-scope, file-size,
  sensitive-pattern, artifact, and local-link checks. Commit
  `cd155b2ac4b434f664397674bf4ac6528364dd5e` (`Add decision curve
  analysis`) was pushed directly to `origin/main`; the post-push fetch shows
  identical local/remote hashes and zero divergence (Session 49 / Batch 20
  publication).
- **Batch 20 full-test decision-curve analysis complete; stop for review:**
  added a CPU-only, hash-bound analysis of all ten frozen Phase 5 test bundles.
  The binary action unit is one radiograph: a detector flags an exam when its
  maximum emitted box confidence reaches tau, and outcome positivity means at
  least one lung-opacity annotation. The prevalence input is computed from the
  complete 750-image/323-patient held-out test split, not the robustness sample:
  169 positive and 581 negative images, prevalence `0.225333` (`22.533%`). The
  five-seed point curves use 2,000 common established patient-cluster/seed draws
  for pointwise percentile intervals at all 99 thresholds. Treat-all is the
  largest point-estimate strategy at `0.01--0.03`, Faster R-CNN at
  `0.04--0.41`, YOLO11s at `0.42--0.62`, and treat-none at `0.63--0.84`;
  the sparse higher-threshold reversals have zero lower bounds or zero net
  benefit. Pairwise intervals favor Faster R-CNN at `0.01--0.27` and YOLO11s
  at `0.60--0.62` plus `0.64--0.88`, but the latter interval mostly says YOLO
  is less harmful than Faster R-CNN while treat-none remains preferable. Raw
  detector scores are explicitly nominal threshold probabilities, not fitted
  clinical risks, and localization is outside this exam-action estimand. The
  CSV/PNG/provenance artifacts are byte-identical across repeated runs, the
  figure was visually inspected, and 240 tests pass with one expected skip;
  Ruff, format, artifact, and whitespace checks are clean (Session 48 / Batch
  20).
- **Batches 18--19 published directly to `main`:** fetched live
  `origin/main`, confirmed zero starting divergence at `5defa77`, and staged
  exactly the 21 public calibration/threshold-calibration files. Internal
  protocol/state files, orchestration output, and historical failed/aborted
  diagnostics were excluded. The staged snapshot passed 235 tests with one
  expected skip, Ruff lint/format, whitespace, local-link, JSON, file-size, and
  sensitive-path checks. Commit
  `cdfcdc025d495dc7db144aad6a0fd0471b468878` was pushed directly to
  `origin/main`;
  the post-push fetch shows identical local/remote hashes and zero divergence
  (Session 47 / Batch 18--19 publication). Batch 29 supersedes that release's
  threshold terminology without changing its frozen F-beta selections.
- **Historical Batch 19 F-beta sensitivity completed:** D-004 kept Batch 14's
  validation-selected 0.69/0.05 thresholds as the authoritative primary
  operating points. A CPU-only runner reused the six frozen Phase 14
  validation bundles (seeds 17/42/137), the 750-image/321-patient validation
  split, the canonical IoU-0.50 matcher, and a hierarchical patient-cluster/
  seed bootstrap. Across 2,000 common draws and the 0.01--0.99 grid, the
  maximum-lower-95%-bound thresholds for beta 1/3/5/10 were Faster R-CNN
  `0.69/0.33/0.12/0.03` and YOLO11s `0.02/0.01/0.01/0.01`. Batch 29 preserves
  those exact numbers while superseding the former interpretation: beta is a
  recall-preference parameter, not an empirical clinical-harm ratio. YOLO's
  beta 3--10 results are lower-boundary optima, not claimed global settings.
  No sensitivity threshold was applied to test or fed into FROC/Pareto
  (Session 46 / Batch 19; corrected by Session 63 / Batch 29).
- **Batch 18 five-seed detection calibration complete; stop for review:** added
  a hash-bound, CPU-only analysis of all ten frozen Phase 5 prediction bundles.
  The primary endpoint is the Küppers et al. full five-dimensional D-ECE over
  confidence, relative box center, width, and height, with five equal-width
  bins per dimension and the paper's eight-sample cell minimum. Every post-NMS
  detection retained at the 0.001 bundle floor is matched through the canonical
  same-class greedy matcher at IoU 0.50; missed targets have no emitted score
  and are outside this black-box precision-calibration estimand. Faster R-CNN
  versus YOLO11s mean D-ECE is `0.032043 +/- 0.005761` versus
  `0.099027 +/- 0.023160` across five seeds. YOLO11s seed 271 is retained and
  ranks worst at D-ECE `0.131298`: its 962 detections have mean confidence
  `0.005382`, matched-TP fraction `0.150728`, global gap `0.145346`, and maximum
  score `0.041274`. The reference framework is Apache-2.0, compatible with the
  repository's AGPL-3.0-only license; the project uses an independent NumPy
  implementation to avoid a new dependency and copied code. H6, the method,
  explicit distinction from n=3 threshold selectivity, scope limitations,
  README commands, table, figure, and full provenance summary are complete.
  Repeated runs are byte-identical; 231 tests pass with one expected skip and
  all static/artifact checks are clean (Session 45 / Batch 18).
- **Repository license is GNU AGPL-3.0-only and is published on `main`:** the
  benchmark integrates Ultralytics YOLO, whose upstream open-source terms are
  AGPL-3.0. The repository now carries the canonical GNU AGPL v3 text in
  `LICENSE`, declares SPDX `AGPL-3.0-only` in `pyproject.toml`, and scopes the
  grant in `README.md` to repository-authored software and documentation.
  RSNA/NIH data and dataset-derived image content, pretrained weights, and
  external dependencies retain their own terms. The exact three-file change
  passed canonical-text, TOML, scope, and whitespace checks and was committed
  as `5defa77d50e3988b3045271e4b1c2ef1588ffc85` (`Add AGPL-3.0
  repository license`) directly on `main`; post-push local/remote hashes match
  with zero divergence (Session 44 / licensing follow-up).
- **Batch 17 published directly to `main`:** fetched live `origin/main` at
  `1876044`, staged the exact ten-file writing allowlist, and excluded local
  state/protocol files and diagnostic logs. The staged snapshot passed scope,
  whitespace, citation-key, local-link, hypothesis-scope, and sensitive-path
  checks. Commit `032436632f674138162d2038b698fc2caba14db7`
  (`Reframe literature review and hypotheses`) was pushed to `origin/main`;
  the post-push fetch showed identical local/remote hashes and zero divergence.
  No branch, pull request, or tag was created (Session 43 / Batch 17
  publication).
- **Batch 17 literature reframing and hypotheses complete; stop for review:**
  `docs/LITERATURE_REVIEW.md` now identifies a specific, citation-grounded
  comparability gap across the medical-detector studies reviewed: dataset and
  split construction, preprocessing, augmentation, operating thresholds, and
  metric definitions/aggregation move alongside architecture, so published
  point estimates do not isolate a detector-family effect. The contribution is
  scoped to two disclosed pipelines under one patient-disjoint data protocol,
  common canonical inputs/augmentation policy, model-independent evaluator,
  and one-machine compute protocol. It also states the project's empirical
  operating-point contribution: a shared numerical score threshold did not
  produce a shared precision-recall/FROC regime. New
  `docs/HYPOTHESES.md` records one research question and five retrospective,
  result-linked hypotheses, each with an operational check, exact existing
  table/figure links, and explicit n=5, paired n=4, frozen n=3, or seed-17
  scope. A terminology pass now describes project claims as lung-opacity
  detection or annotated-opacity localization while preserving the official
  RSNA challenge title, publication titles, BibTeX keys, URLs, filenames, and
  explicit non-diagnostic warnings. No experiment, metric, result table,
  figure, config, or code changed. Citation-key, local-link, terminology, and
  whitespace/diff checks pass (Session 42 / Batch 17).
- **Batch 16 all-attempt n=5 clean analysis complete; stop for review:** the
  user accepted retention of the predeclared YOLO11s seed 271 rather than
  outcome-selecting a replacement. The unified evaluator completed all ten
  checkpoints. AP and fixed-threshold precision/recall/F1 include all five
  attempts; descriptive conditional IoU/Dice use Faster R-CNN `n=5` and
  YOLO11s `n=4`, with seed 271 explicitly null rather than zero. Paired clean
  inference uses all five pairs for nonconditional endpoints and complete
  pairs `17/42/137/314` for IoU/Dice, while keeping all seven endpoints in one
  Holm family. The patient IDs, 323-group construction, cluster-resampling
  algorithm, and patient-label-swap algorithm in `src/stats/paired.py` are
  unchanged; the two endpoint groups use separate deterministic random streams.
  The corrected Holm pattern strengthened from 4/7 at n=3 to 5/7:
  F1 changed from non-significant (`0.0971805639`) to significant
  (`0.0013997201`); precision, recall, and both AP endpoints remain
  significant, while IoU/Dice remain non-significant (`0.1063787243`). A
  hash-bound offline stability table confirms seed 271 converged normally but
  had zero detections at 0.25 and 0.05, versus 962 predictions and 145 matches
  at the 0.001 COCO floor. This is operational confidence-score degeneracy,
  not classic loss/head collapse or an IoU-matching accident. The n=3 Phase 5
  tables/config/summary are archived; threshold/PR/FROC/Pareto remain frozen
  n=3; robustness and explainability remain seed-17-only and were not rerun.
  Phase 6 summary/table hashes remain `4fe09e19...` / `59c9ec17...`. Mandatory
  Batch 18 paper carry-forward: abstract, Methods, Results, Discussion, and
  Limitations must disclose all-attempt retention, n=5/n=4, confidence-score
  instability, and n=3-only threshold/Pareto/FROC scope (Session 39 / Batch 16).
- **Batch 16 all-attempt recommendation (accepted; historical rationale):**
  do not outcome-select a
  replacement for seed 271 in the primary analysis. It is a valid predeclared
  training attempt with normal convergence but an operational confidence-scale
  failure, which is part of the augmentation-disabled recipe's seed
  variability. Silently replacing it would condition the comparison on a
  favorable result. Recommended recipe-level analysis: retain all five seeds
  for AP and fixed-threshold precision/recall/F1 (seed 271 contributes its
  observed zeros); report descriptive conditional IoU/Dice with explicit
  detector-specific n (Faster R-CNN 5, YOLO11s 4); use the four complete seed
  pairs for conditional IoU/Dice inference while retaining all five pairs for
  other endpoints; and keep the seven-endpoint Holm family. This changes only
  endpoint-specific cross-seed reduction/eligibility, not patient identifiers,
  clustering, bootstrap draws, or label-swap mechanics. At score 0.25, this
  all-attempt summary would move YOLO precision/recall/F1 from the n=3 values
  `0.3730/0.1356/0.1981` to approximately
  `0.2983/0.0955/0.1427`, with much larger seed SD, while AP50:95 remains
  `0.05417 +/- 0.00603`; at the already-frozen score 0.05, YOLO becomes
  `0.2524/0.1948/0.2192`. This AP-versus-operating-point contrast is the
  stability finding. An additional detector pair may be trained only as an
  explicitly supplementary sixth-seed sensitivity analysis, never as an
  unreported replacement (Session 38 / recommendation only; not executed).
- **Batch 16 seed-271 validity gate (superseded diagnostic stop):** the run did
  not show
  the previously documented optimizer/head-collapse signature: its three train
  losses fell from `1.58274/4.82425/1.61948` to
  `0.96834/1.09339/0.99961`, validation mAP50:95 peaked at `0.08958` (the
  highest of the five YOLO seeds), and it completed cleanly by validation-map
  early stopping. However, it emitted **no prediction at the frozen score-0.25
  operating point on either validation or test**. On test, all 962 predictions
  retained at the COCO floor 0.001 were below 0.25 (maximum `0.0412735`), so
  TP/FP/FN were `0/0/268` and precision/recall/F1 were all zero. This was not
  an unlucky IoU-matching event. Ranking/localization remained plausible:
  test AP50 `0.1587217`, AP50:95 `0.0555799`, both inside the sibling range,
  and the low-score boxes include 145 IoU>=0.5 matches. The evidence therefore
  indicates a seed-specific confidence/output-score degeneracy despite normal
  curve convergence. Per the user's gate, seed 271 has not been folded into an
  n=5 result and aggregation/statistics recovery has not been implemented;
  choose a replacement seed or report this as a YOLO stability finding. The
  complete bundle is `yolo11s_seed271_test_predictions.json.gz`, SHA-256
  `8790c679...f1047da5b` (Session 37 / Batch 16 validity stop).
- **Batch 16 recovery-scope audit (resolved):** reporting conditional IoU/Dice
  over four
  finite seeds is a cross-seed missingness rule. The Phase 5 table aggregator
  needs that rule, and Batch 13's current `estimate_pair` seed-reduction loop
  also propagates one NaN across all seeds. A future n=4 conditional
  inferential result would therefore require metricwise finite-seed reduction
  there as well. Patient identifiers, patient clustering, cluster bootstrap
  draws, and patient-level permutation swaps need no changes, but it would be
  inaccurate to call the complete fix evaluator-only. No statistics code has
  been changed (Session 37 / Batch 16 scope audit).
- **Batch 16 training complete; downstream undefined-metric failure diagnosis
  (resolved):** all four approved trainings completed serially with exit code 0
  at 2026-08-19 04:35 UTC, and evaluation preflight validated all ten
  detector/seed contracts. The unified evaluator completed inference and wrote
  all ten prediction bundles, then failed during mean/SD aggregation because
  YOLO11s seed 271 had zero fixed-threshold true positives: precision, recall,
  and F1 are numeric zeros, while conditional matched-box IoU/Dice are
  correctly `null`. `aggregate_rows` incorrectly calls `float()` on these
  valid nulls. Statistics therefore did not run. The primary clean comparison
  and statistics tables remain byte-identical to their n=3 archives; the
  robustness table and Phase 6 summary also retain hashes `59c9ec17...` and
  `4fe09e19...`. No retraining is needed. Recovery must preserve the documented
  conditional-metric semantics rather than coercing undefined localization to
  zero (Session 36 / Batch 16 stopped downstream).
- **Batch 16 approved run-state note (completed):** the user approved the frozen
  approximately five-hour budget. A hidden serial queue started at
  2026-08-18 23:50 UTC with Faster R-CNN seed 271, followed by Faster R-CNN
  seed 314 and YOLO11s seeds 271/314; the queue aborts on the first nonzero
  exit and never overlaps GPU trainings. Seed 271 completed epochs 1 and 2
  (validation mAP50:95 `0.1071597631` and `0.0984534389`) and is advancing
  normally through epoch 3, with only the already-documented CUDA ROI Align
  deterministic warning. A 60-second health watchdog records log freshness,
  process count, GPU load/VRAM/temperature, and orchestration failures; its
  initial state is healthy at 100% GPU, 3,128 MiB VRAM, and 62 C. A separate
  downstream watcher is waiting for all four
  successes before running Phase 5 preflight/evaluation and Phase 8
  patient-cluster clean-only inference. It records and rechecks the Phase 6
  summary and corrected robustness-table hashes, so robustness cannot be
  silently regenerated or altered. No n=5 result exists yet (Session 35 /
  Batch 16 active run).
- **Batch 16 clean-only execution path (executed):** `src/evaluate.py` now uses
  each completed Faster R-CNN training summary's immutable config hash when
  validating historical checkpoints, while the loaded configs must still pass
  detector-level semantic parity. `src/stats/run_statistics.py --mode run
  --scope clean` recomputes only the five-seed clean patient-cluster analysis,
  compares its Holm pattern to the corrected n=3 archive, and preserves the
  prior robustness results/table by hash. Ruff is clean and 14 focused
  evaluator/statistics tests pass (Session 34 / Batch 16 active run).
- **Batch 16 timing/sign-off gate was accepted (superseded run-state note):** seeds
  271 and 314 are frozen for both detectors. Their four configs are exact
  seed/output-identity copies of the accepted seed-42/137 contracts, and the
  unified evaluator now validates the complete five-seed factorial grid. The
  accepted seed-17 timing sources are reused through four new approval
  artifacts bound to each target config, the current runtime/data identity,
  and current source-manifest SHA-256
  `118775aba048fae32ff5e2ce2e0001c1388e2815bed1f9b60f78516eb338df4e`.
  The post-timing source drift was audited and frozen exactly: Ruff-only
  rewrites, optional downstream evaluation/robustness additions, the recorded
  YOLO reporting recovery, and the valid-input-neutral COCO path-safety fix.
  Any further source drift invalidates the new gates. All four trainers'
  internal approval checks pass. Historical completed runs average 1.725 h per
  Faster R-CNN seed and 0.509 h per YOLO11s seed, projecting 4.47 h of serial
  epoch-loop training and approximately 4.75--5 h including finalization,
  evaluation, and clean-only statistics. The observed-run envelope is roughly
  3--6 h; the predeclared 30-epoch hard ceiling is approximately 10.94 h before
  finalization. The user subsequently approved this budget and the queue is
  active (Session 33 / Batch 16 preflight; superseded by Sessions 34--35).
- **Batch 16 n=3 clean archives frozen before replacement:** the current
  corrected three-seed comparison, mean/std, per-seed, and patient-cluster
  clean-statistics tables were copied byte-for-byte to explicit
  `*_n3_archive.csv` paths. Their SHA-256 values remain respectively
  `6b467c70...`, `91affca6...`, `ab457458...`, and `cdd67407...`. No primary
  table, prediction bundle, checkpoint, robustness artifact, or explainability
  artifact has been changed (Session 33 / Batch 16 preflight).
- **Batches 13--15 published directly to `main`:** authenticated through the
  configured GitHub CLI keyring, fetched `origin/main`, and confirmed zero
  divergence at `48a5704` before publication. Staged exactly 50 patient-cluster
  statistics, validation-threshold/FROC/Pareto, code/test/config, result, and
  reconciled public-document paths. Local protocol/state files and all
  failed/aborted/rejected diagnostic directories remained excluded. The staged
  scope, largest-blob size, added-line credential patterns, and staged diff all
  passed audit; the already-completed validation remained 212 tests passed with
  one expected skip and Ruff clean. Commit `802cf75` (`Correct statistical and
  threshold methodology`) was created on `main` and pushed directly as
  requested. Local `HEAD`, `origin/main`, and GitHub's live `main` ref all
  resolve to full SHA `802cf75995680284ce5f34c2430155a84e6d0d7c` with zero
  divergence. No branch, PR, or tag was created (Session 32 / Batches 13--15
  publication).
- **Batch 15 narrative reconciliation complete:** the report and quantitative
  summary now state the corrected headline: Faster R-CNN has the stronger
  precision-recall frontier, YOLO11s's original score-0.25 precision advantage
  is a score-scale/selectivity artifact rather than a frontier advantage, and
  the defensible comparison is detection quality versus
  implementation-specific computational cost. The report adds the
  validation-selected 0.69/0.05 test operating points, links the Pareto and
  FROC evidence, replaces superseded image-level intervals/effect sizes with
  the primary patient-cluster results, and corrects the severe-darkness claim
  from significant raw AP to significant AP retention only. `LIMITATIONS.md`
  records the resolved Batch 13 clustering error, digital-not-clinical
  robustness, and implementation-specific-not-architecture-general compute
  scope. A repository-wide stale-claim audit also reconciled the README
  headline, the V1/current distinction in `PROJECT_PLAN.md`, and the
  robustness evidence wording. The requested six-document joint read found no
  contradictory threshold, frontier, FROC, Pareto, or statistical claim.
  Validation: cross-document/artifact assertions and local links pass, Ruff
  format/lint pass, 212 tests pass with one expected metadata-only skip, and
  `git diff --check` passes (Session 31 / Batch 15).
- **Batch 14 validation-only threshold selection complete:** raw validation
  scores were absent from the archived training tables, so one inference-only
  pass materialized six hash-bound 750-image validation bundles from the
  immutable best checkpoints; all six exactly reproduce their archived
  threshold-0.25 validation precision/recall/F1. The predeclared rule maximizes
  arithmetic mean F1 across seeds on the unchanged 0.01--0.99 grid and breaks
  exact ties toward the higher threshold. It selects 0.69 for Faster R-CNN and
  0.05 for YOLO11s. Applying each threshold exactly once to each frozen test
  bundle gives mean precision/recall/F1 0.3543/0.3607/0.3492 versus
  0.3096/0.2438/0.2718. The selection summary SHA-256 is
  `4e1d31e1dffb258fdb534b86677d912944df81cc226fde3bc9daf302505f7e50`
  (Session 30 / Batch 14).
- **Batch 14 Pareto contamination correction complete:** recall panels now
  consume the validation-selected, one-shot test operating points rather than
  the exploratory test-sweep peaks 0.63/0.01. Faster R-CNN test recall spans
  0.2910--0.4030 at threshold 0.69; YOLO11s spans 0.2090--0.2612 at threshold
  0.05. Neither detector strictly dominates any panel under the unchanged
  all-seeds rule. AP inputs and panels are unchanged, and the Batch 10 PR and
  F1 figure hashes remain exactly frozen (Session 30 / Batch 14).
- **Batch 14 FROC/terminology correction complete:** the unchanged 594-row
  Batch 10 test sweep is reparameterized as sensitivity versus false positives
  per image. At budgets 0.125/0.25/0.5/1/2 FP/image, Faster R-CNN sensitivity
  is 0.2699/0.3607/0.4801/0.6032/0.6928 versus YOLO11s
  0.1803/0.2749/0.3321/0.3321/0.3321. YOLO11s's plateau is the threshold-0.01
  sweep boundary, not a claimed global asymptote. The finding is a
  score-scale/selectivity mismatch, not measured probabilistic calibration.
  FROC summary SHA-256 is
  `f38873f5a1b00405d7a5e4f0ebf686338a61270981b08f7e7c89a4f1828c50f5`;
  validation is 212 tests passed with one expected skip, Ruff clean, and all
  source/artifact/hash/count checks passing (Session 30 / Batch 14).
- **Batch 13 patient-cluster statistical correction complete:** Phase 8 was
  reprocessed offline from the six frozen Phase 5 clean bundles and all 72
  frozen Phase 6 bundles; no training, checkpoint loading, or inference ran.
  The committed Batch 1 `nih_patient_id` mapping supplies 323 patient groups
  for the 750-image clean set and 183 groups for the 300-image robustness
  subset. All 2,000 bootstrap draws now sample patient groups with replacement
  and include every observed image in each sampled group; all 5,000 paired
  detector-label permutations swap complete patient clusters. The clean pass
  still resamples the three paired training seeds, and retention keeps clean
  and corrupted evidence coupled within each group (Session 29 / Batch 13).
- **Batch 13 significance decision:** the clean Holm pattern is unchanged at
  four of seven endpoints: Faster R-CNN remains significant for recall,
  mAP@0.5, and mAP@0.5:0.95, while YOLO11s remains significant for precision.
  The secondary robustness pattern changes: among all 497 rows, the significant
  count moves from 88 to 87, comprising six formerly significant comparisons
  that lose significance and five that gain it. Most importantly, darkness
  severity 5 raw mAP@0.5 (Holm 0.0070 to 0.1750) and raw mAP@0.5:0.95 (0.0070
  to 0.0770) are no longer significant. Their retention differences remain
  significant at Holm 0.0070; new significance appears for darkness severity
  2 mAP@0.5 retention and darkness severities 2--3 conditional IoU/Dice
  retention. See the complete 11-row decision delta in
  `docs/STATISTICAL_ANALYSIS.md` (Session 29 / Batch 13).
- **Batch 13 effect/archive decision:** the image-level jackknife Cohen's d is
  not adapted by guesswork to unequal patient clusters and nonlinear pooled
  metrics. The primary effect is the paired raw aggregate difference with its
  patient-cluster bootstrap CI. Exact superseded clean/robustness CSV hashes
  remain `9eb05cf8...` and `8b766f59...`, and the old summary hash remains
  `064bba13...`; all three are retained under explicit
  `*_image_level_archive.*` paths for audit. Corrected primary clean and
  robustness CSV hashes are
  `cdd674075ab6eefd5903891ec84da8b2999344aaafb325723fa8e63df4e6aa6d`
  and `59c9ec17abacef102f854e463039e569fa22bac97e6fbc55c99af7f20f2dcb02`.
  Validation is 208 tests passed with one expected metadata-only skip, Ruff
  clean, artifact/source hashes valid, and all point estimates byte-equivalent
  to the archived rows (Session 29 / Batch 13).
- **Pre-Batch 13 CI housekeeping published and green:** Ruff 0.16.0 formatted
  the exact 36 `src/` and `tests/` files reported by both failing matrix jobs;
  no Ruff rule, lint selection, expectation, or CI gate was relaxed. The
  workflow now installs the existing locked `cpu` extra and runs pytest as a
  module, which exposed and then fixed one latent POSIX/Windows discrepancy in
  COCO filename safety validation. Commits `9fee539` (`Restore foundation CI
  and record D-003`) and `48a5704` (`Validate COCO paths across platforms`) are
  published on `origin/main`. Foundation CI run
  `https://github.com/Alpha-lacrim/medical-object-detector-benchmark/actions/runs/32076898917`
  passed lock validation, dependency sync, Ruff formatting, Ruff lint, pytest,
  and deterministic smoke on both Ubuntu and Windows (Session 28 / pre-Batch
  13 housekeeping).
- **D-003 reaffirms D-002:** the project declines to revive a formal
  Track-B-style native/best-practice comparison. Current claims concern two
  specific documented pipelines under shared constraints, not detector-family
  effects in the abstract. A single native-defaults run per detector remains
  optional post-Batches 13--15 exploratory future work only if compute remains;
  it is neither committed work nor a second formal track (Session 28 / pre-Batch
  13 housekeeping).
- **Batch 11 published directly to `main`:** the audited six-file Pareto scope
  was committed as `e33fbcb` (`Add accuracy-efficiency Pareto analysis`) and
  pushed directly to `origin/main`. Local `main`, the remote-tracking ref, and
  the live GitHub branch all resolve to full SHA
  `e33fbcbd9d97e91fee75b4ffc5855ff83d0bb6f2`. The publication excludes
  local-only protocol/state files, static batch/spec files, and all preserved
  diagnostic directories; no branch, PR, tag, or unrelated worktree content
  changed (Session 27 / Batch 11 publication).
- **Batch 11 Pareto analysis complete:** the six frozen Phase 5 seed rows were
  joined to their six existing compute CSVs and the completed Batch 10 sweep;
  no training, checkpoint loading, or inference ran. The mAP panels remain
  threshold-independent, while recall is shown at each detector's best
  observed mean-F1 sweep point (Faster R-CNN threshold 0.63, YOLO11s threshold
  0.01). Under the conservative detector-level rule requiring every seed to be
  strictly better than every alternative seed on both axes, neither detector
  dominates in any of the four panels: Faster R-CNN occupies the higher-AP and
  higher-recall region, while YOLO11s occupies the higher-throughput,
  lower-latency, smaller-parameter, and lower-estimated-GFLOP region. The
  scenario-conditional interpretation is documented in
  `docs/PARETO_ANALYSIS.md` (Session 26 / Batch 11). The test-derived recall
  threshold choice in this historical entry is superseded by Batch 14.
- **Batch 10 published directly to `main`:** the audited 15-file threshold
  analysis scope was committed as `cddcaa8` (`Add threshold sweep analysis`)
  and pushed directly to `origin/main`; local and remote `main` both resolve to
  `cddcaa893262119fdeb3c550c02758322fe1da82`. A temporary publication branch
  and draft PR #3 were created before the user's direct-main instruction
  arrived; the PR was immediately closed and both temporary branch refs were
  deleted. The local-only protocol/state files and all preserved diagnostic
  directories remained outside the commit (Session 25 / Batch 10 publication).
- **Batch 10 threshold/PR analysis complete:** all six hash-verified Phase 5
  prediction bundles were reprocessed offline at 99 thresholds from 0.01 to
  0.99 through the existing unified matcher; no training or inference ran. The
  fixed 0.25 threshold materially exaggerates YOLO11s's low recall (mean recall
  rises from 0.1356 to 0.3321 at threshold 0.01), but its apparent precision
  advantage does not survive matched operating points: Faster R-CNN has higher
  mean precision at 96/101 official AP@0.5 recall positions and YOLO11s at none.
  YOLO11s also cannot reach mean recall 0.50 in the sweep. Treat the headline as
  a score-scale/selectivity fixed-threshold difference plus a persistent YOLO11s
  coverage limitation, not two generally superior but distinct PR regimes
  (Session 24 / Batch 10).
- **Batch 9 housekeeping published to GitHub:** authenticated against the
  configured GitHub keyring, fetched `origin/main`, and verified its two new
  commits changed only `docs/PROJECT_PLAN.md`. Local `main` was fast-forwarded
  to remote tip `d07f193`, accepting the cloud plan exactly while preserving
  the local Batch 9 documents. Staged scope was exactly
  `docs/DECISION_LOG.md` and `docs/QUANTITATIVE_COMPARISON.md`; staged diff and
  unchanged `results/`/`report/` checks passed. Commit `caeee58` (`Document
  course submission freeze`) was pushed directly to `origin/main`, and the
  annotated `v1-course-submission` tag was published to the remote. Unrelated
  untracked coordination/protocol files and diagnostic runs remained excluded
  (Session 23 / Batch 9 publication).
- **Course submission frozen at an annotated tag:**
  `v1-course-submission` is an annotated tag targeting the pre-research-track
  HEAD `3a3808841795938a296d48ae3b379b0d10ef3d48`. Its annotation identifies
  the completed two-detector course deliverable: Faster R-CNN and YOLO11s, the
  unified evaluator, robustness benchmark, Grad-CAM explainability, paired
  statistics, and 12-section report. The tag was created without moving the
  existing `v1.0.0` tag, rewriting history, or changing the worktree (Session
  22 / Batch 9).
- **D-002 formally descopes Track B:** `docs/DECISION_LOG.md` now preserves
  D-001 as the historical two-track proposal and adds append-only D-002,
  superseding only its architecture-optimized Track B portion. Track B is
  descoped for the fixed RTX 4060 Laptop/16 GB RAM hardware and course-time
  budget. The research-paper contribution is the delivered controlled
  comparison and multi-axis accuracy/operating-point/compute/robustness/
  explainability/statistical trade-off analysis, not best-effort leaderboard
  performance. A read-only replay also reproduced the frozen 450.7637248 and
  21.4198784 GFLOP values and documented the exact registered operations,
  1,000-proposal Faster R-CNN RoI path, module attribution, exclusions, and
  official-model-card sanity check without changing results (Session 22 /
  Batch 9).
- **Reconciled V1 planning record published:** the user-approved reconciliation
  of `docs/PROJECT_PLAN.md` and removal of `Final project 1405.v1.pdf` were
  committed on `main` as `3a38088` (`Reconcile V1 project plan`) and pushed to
  `origin/main`. The commit contains exactly those two tracked paths; local
  agent/coordination files and diagnostic run directories remain untracked and
  unpublished. No GitHub release was created and the existing `v1.0.0` tag was
  not moved; it still targets the earlier `df81f71` commit pending the user's
  separate release instruction (Session 21).
- **Project and final report complete:** Batch 8 assembled the frozen Phase
  1--8 evidence into `report/report.md` using the authoritative 12-section
  structure without rerunning inference, statistics, or artifact generation.
  The Discussion assigns Faster R-CNN to accuracy-/recall-sensitive,
  GPU-backed screening and YOLO11s conditionally to compute-constrained,
  human-in-the-loop assistance, while rejecting autonomous clinical use for
  both. It weighs the measured accuracy, robustness, Grad-CAM, and compute
  evidence and includes the required prospective/regulatory scope boundary.
  `docs/LIMITATIONS.md` is consolidated by validity domain, and `README.md`
  now provides a clean-checkout command sequence plus an explicit mapping from
  every report table/figure to its generating artifact and command. All eight
  Definition of Done items pass an evidence audit: 12 report sections, two
  detectors, 14 comparison metrics, four corruption families/seven types at
  five severities, three Grad-CAM figure categories over 111 targets per
  detector, and the complete seven-clean/497-robustness statistical tables.
  Validation is 198 tests passed, one expected metadata-only skip, Ruff clean,
  all local report/doc links and 17 BibTeX citation keys valid, all documented
  CLI parsers valid, and `git diff --check` clean (Session 18 / Batch 8).
- **Agent branch promoted to `main`:** GitHub's existing default branch `main`
  was a strict ancestor of `agent/implementation-foundation`, with zero commits
  unique to `main` and 14 commits unique to the agent branch. It was safely
  fast-forwarded from `317dca0` to the accepted Batch 4–7 publication tip
  `f0e0f00`; no force-push, history rewrite, merge commit, or source-tree change
  was used. The Session 17 bookkeeping record is synchronized to both refs.
  Batch 8 remains gated on the user's statistical-results review (Session 17).
- **Batches 4–7 publication checkpoint:** the accepted evaluation,
  multi-seed, robustness, explainability, and statistical-analysis scope was
  committed as `99ee5e3` (`Complete evaluation robustness explainability and
  statistics`) and pushed over HTTPS to
  `origin/agent/implementation-foundation`. Publication followed a 192-file
  index audit, 198 passing tests with one expected metadata-only skip, clean
  Ruff and staged-diff checks, and explicit exclusion of the local PDF deletion
  plus failed/aborted/rejected and orchestration diagnostic runs. Batch 8 remains
  blocked on the user's Batch 7 review (Session 16).
- **Superseded image-level statistical analysis (archived):** The original
  Phase 8 pass reconstructed the exact
  seven unified predictive metrics from all six frozen Phase 5 bundles and all
  72 Phase 6 bundles without model inference. The clean analysis uses 2,000
  paired hierarchical image/seed bootstrap draws, 5,000 paired image-label
  permutations, pointwise 95% percentile intervals, and paired jackknife
  Cohen's d. After Holm correction across seven endpoints, Faster R-CNN retains
  evidence for higher recall (difference 0.50249, 95% CI 0.43005 to 0.57596,
  adjusted p 0.00140), mAP@0.5 (0.14413, 0.09673 to 0.19488, adjusted p
  0.00140), and mAP@0.5:0.95 (0.04737, 0.03105 to 0.06833, adjusted p
  0.00160); YOLO11s retains higher precision (difference -0.21045, -0.30500 to
  -0.11773, adjusted p 0.00140). F1 and conditional IoU/Dice did not cross the
  0.05 Holm threshold. Its intervals, p-values, and Cohen's d are superseded by
  Batch 13; this entry remains as the historical Session 15 / Batch 7 record.
- **Superseded image-level corruption-grid inference:** Every one of the 35
  primary-seed conditions has raw and clean-relative detector comparisons for
  all seven metrics. Holm correction is separate by metric and estimand; IoU
  and Dice contain 34 estimable tests because YOLO has no true positive under
  darkness severity 5. That condition is the only AP comparison surviving
  grid-wide correction: raw mAP@0.5:0.95 difference 0.11565 (95% CI 0.07122 to
  0.17024, adjusted p 0.00700, d 0.241) and retention difference 0.70287
  (0.39816 to 0.84242, adjusted p 0.00700, d 0.433). The two final table hashes
  are `9eb05cf8df7e26e237d8bf7b0c5eb85cdc1130477db46213e125217b692d819f`
  and `8b766f59b3e69aa7d011f2a8ac5499636cbf30ed7eb943b3c1ad8c753a49940b`;
  summary SHA-256 is
  `064bba1317195e749562f29a8fa089ac59a40957b6205f51d27c65babc3c3937`
  (Session 15 / Batch 7). Batch 13 archives these exact artifacts and replaces
  their image-level inference with the patient-cluster results recorded above.
- **McNemar non-applicability decision:** no McNemar test is reported. The
  benchmark has multiple targets plus negative-image false positives rather
  than one independent binary image outcome. Collapsing to correct/incorrect
  would discard detection structure, while target decisions are nested within
  images and repeated across seeds/conditions. The aggregate-metric paired
  permutation directly tests the detector comparison without forcing a
  classification endpoint (Session 15 / Batch 7).
- **Primary-seed stride-matched Grad-CAM analysis complete:** Phase 7 reuses
  the exact Phase 6 seed-17 300-image manifest and both primary checkpoints.
  Ordinary ReLU Grad-CAM hooks 40 by 40 stride-16 backbone tensors: ResNet-50
  `backbone.body.layer3` before FPN and YOLO11s `model.6` before the stride-32
  stage/PAN neck. Every one of the sample's 111 ground-truth boxes receives a
  low-threshold highest-IoU retained-candidate target per detector; operating-
  point false negatives are explicitly labeled proxy targets. Faster R-CNN
  has 110 valid CAMs and one reported zero map; YOLO11s has 111. Mean energy in
  box is 0.08689 versus 0.09749 against box-area references 0.07129 versus
  0.07178, and pointing accuracy is 0.10909 versus 0.12613. Among 110 paired
  valid targets, YOLO has higher energy for 76 and Faster R-CNN for 34; mean
  Faster-minus-YOLO energy is -0.00910. Both remain weakly localized and often
  emphasize anatomy, borders, markers, and devices. The complete summary
  SHA-256 is
  `2b8d2d5835c113e8dc24af9eecbece62571cc5bcc689bcb245c1f74e1c23a848`.
  Stop for explainability review before Batch 7 (Session 14 / Batch 6).
- **Phase 7 qualitative and metric contract:** the primary metric is
  pixel-center energy-in-box, with pointing-game accuracy secondary and
  rasterized box area as a reference. Zero-energy maps are excluded and
  reported rather than coerced to zero. Three shared high-IoU true positives,
  three shared false positives on box-negative `No Lung Opacity / Not Normal`
  images, and three shared false negatives spanning proxy-IoU quantiles are
  selected from frozen predictions before CAM values are known. CUDA ROI Align
  backward is seeded but only deterministic-warn-only in the pinned
  Torchvision build. These maps are association diagnostics, not causal or
  clinical reasoning evidence (Session 14 / Batch 6).
- **Primary-seed common-corruption robustness grid complete:** Phase 6 fixes a
  seed-17 proportional stratified sample of 300 held-out test images: 68 Lung
  Opacity, 132 No Lung Opacity / Not Normal, and 100 Normal, containing 111
  boxes and 183 patients. The committed manifest SHA-256 is
  `63b4dd706dc2fcd8a528a935957ccb318ed2cde51a6fd87d20feca348d00fc5e`.
  Both primary checkpoints were scored on seven corruption types at five
  config-defined severities (35 conditions each) through the same Phase 5
  evaluator. Every condition has a resumable hashed 300-prediction bundle;
  clean references are exact filters of the frozen Phase 5 seed-17 bundles.
  Across all 35 conditions, Faster R-CNN versus YOLO11s mean raw
  mAP@0.5:0.95 is 0.11290 versus 0.05410, while mean clean-relative retention
  is 0.76385 versus 0.70908 (23.62% versus 29.09% degradation). Faster R-CNN
  has higher raw mAP in all 36 matched clean/corrupted conditions; salt-and-
  pepper severity 5 is the worst relative condition for both. The complete
  summary SHA-256 is
  `4fe09e19bc7b7d620ab9e6a3785ecae0bb2ef16cb517fce1bc287b2de2fafb2b`.
  Stop for robustness review before Batch 6 (Session 13 / Batch 5).
- **Phase 6 corruption and reporting contract:** Albumentations 2.0.8 applies
  geometry-preserving, deterministically seeded brightness, Gaussian/impulse
  noise, Gaussian/motion blur, and JPEG transforms. Each type has five ordered
  severities; JPEG explicitly includes qualities 50 and 20. Tables report raw
  precision/recall/F1/conditional IoU/Dice/mAP and corrupted/clean ratios plus
  `1 - ratio`, with null conditional ratios preserved when a condition has no
  true positives. Family means are descriptive equal-weight averages rather
  than physically calibrated deployment expectations (Session 13 / Batch 5).
- **Unified three-seed held-out comparison complete:** seeds 17, 42, and 137
  are complete for both detectors under the accepted Batch 2/3 hyperparameter
  contracts. The additional Faster R-CNN runs selected epochs 9 and 2 after
  14 and 8 completed epochs; the additional YOLO11s runs selected epochs 10
  and 14 after 14 and 19 completed epochs. Only after all six checkpoints were
  frozen, one `src/evaluate.py` path evaluated every checkpoint on the same 750
  test images/268 boxes with the same operating-point matcher and official
  pycocotools COCO evaluator. Across three seeds, Faster R-CNN versus YOLO11s
  produced mAP@0.5:0.95 0.1023 ± 0.0036 versus 0.0549 ± 0.0080, mAP@0.5
  0.3084 ± 0.0123 versus 0.1643 ± 0.0226, recall 0.6381 ± 0.0526 versus
  0.1356 ± 0.0094, and F1 0.2558 ± 0.0493 versus 0.1981 ± 0.0048. YOLO11s
  has higher precision (0.3730 ± 0.0395 versus 0.1626 ± 0.0439) and conditional
  matched-box IoU/Dice, plus 52.94 ± 10.65 FPS versus 17.42 ± 5.69. The final
  publication/per-seed/long-form table SHA-256 values are respectively
  `6b467c706dd39a9a240d99a552eb0218734c8b9eaf38b0bfbc70d347f921449c`,
  `ab4574589da9c63f4463e6ef13e4fef26dc565cd514cbd19118491ac0e7c09a8`,
  and `91affca6abe7fadcc70e0b5ca5836e74394d99df1b1af982ea7240dbcab9d482`.
  Six hashed prediction bundles preserve image-level evidence for the later
  paired tests. The final evaluation summary SHA-256 is
  `e6018a9fc2117ac41cc51ab395c22316e61ba40c030b8f54ce6e64c641ea8245`.
  Stop for the user's table review before Batch 5 (Session 12 / Batch 4).
- **Additional-seed timing gates reuse the accepted seed-17 measurements:**
  seeds 42 and 137 differ only in RNG and artifact identity, so Phase 5 derived
  explicit provenance-bearing timing approvals rather than spending four more
  redundant three-epoch benchmarks. Every full run measured its own training
  time and peak allocated memory. A briefly started Faster R-CNN seed-42
  redundant benchmark was stopped before any epoch/checkpoint and preserved as
  an excluded diagnostic; it contributes no result (Session 12 / Batch 4).
- **YOLO11s one-seed baseline complete:** the exact benchmark-approved seed-17
  run restarted from pinned COCO weights and early-stopped at epoch 15 after
  1,975.64 seconds (32.93 minutes); epoch 10 is the selected checkpoint. Shared
  validation on all 750 images/277 boxes produced AP50 0.26464, AP50:95
  0.08692, precision 0.57143, recall 0.20217, and F1 0.29867 at score 0.25 and
  match IoU 0.50. Peak allocated training memory was 1,148.16 MiB. Batch-1
  bfloat16 profiling measured 65.24 FPS and mean/p50/p95 latency
  15.33/14.49/19.82 ms; the model has 9,428,179 total and 9,428,163 training-
  time trainable parameters, 21.42 estimated GFLOPs, and an 18.28 MiB best
  checkpoint. Best checkpoint SHA-256:
  `65909164e82c1ef53c0d38e0d898d37bbbec5f46cb9f5cd029e76ba486c0371c`.
  Ultralytics' native epoch-10 mAP50:95 was 0.07335 and is retained only as the
  checkpoint-selection metric; headline validation uses the shared evaluator.
  All artifacts validate and the test split was not accessed. Stop for Batch 3
  review before Batch 4 (Session 10 / Batch 3).
- **YOLO three-epoch timing gate complete:** the accepted official-data
  benchmark completed in 141.79, 134.99, and 136.20 seconds. The 135.59-second
  steady-state estimate projected 18.18 minutes for eight epochs and 67.90
  minutes for the 30-epoch ceiling. Approval artifact:
  `results/logs/yolo11s_rsna_seed17_benchmark/benchmark_estimate.json`, SHA-256
  `c339db91c05b1c8a1398dbbdcc7470ef1fd1932ddf1c374a87529faca45e1587`;
  config SHA-256
  `5a9bd54c730a42db166d8e5c7075f863f914b5be7c66567f5bc91a70b50ef8d2`.
  The full run's recorded training source identity matches this gate exactly
  (Session 10 / Batch 3).
- **YOLO finalization recovery is weight-preserving and auditable:** training
  completed before reporting encountered an OOM caused by Ultralytics treating
  a 750-path Python list as one inference batch. Finalization now streams the
  exact audited validation directory with batch 4 and verifies every filename.
  Best-epoch reporting is derived from immutable `results.csv` because stripped
  Ultralytics checkpoints record epoch `-1`. The final summary preserves the
  benchmark-approved training-source hash and separately records the corrected
  reporting-source hash; neither fix changed or resumed training (Session 10 /
  Batch 3).
- **YOLO native BatchNorm updates restored before the valid benchmark/training:**
  bfloat16 plus float32 loss kept arithmetic stable, but forced frozen BN still
  drove the head to zero scores during epoch 3. YOLO therefore updates its
  native BN statistics for COCO-to-radiograph adaptation; Faster R-CNN retains
  its frozen-normalization backbone. This architecture-specific asymmetry is
  disclosed (Session 10 / Batch 3, before valid training).
- **YOLO bfloat16 AMP adopted before the valid benchmark/training:** casting
  only target assignment/loss to float32 was insufficient because float16 head
  logits underflowed before reaching the loss (epoch 2, batch 229; AMP scale
  0.0625). Use RTX-4060-supported bfloat16 autocast for model forward/backward
  and float32 assignment/loss. AMP remains mandatory; ordinary seeded shuffle,
  batch, data, augmentation, and model remain unchanged. Faster R-CNN uses
  float16, so this precision asymmetry is disclosed (Session 10 / Batch 3,
  before valid training).
- **YOLO AMP validation adapted for bfloat16 before training:** Ultralytics'
  built-in probe is tied to its default float16 output-equivalence tolerance
  and disabled AMP when bfloat16 was selected. The custom trainer substitutes
  an RTX/CUDA bfloat16 support-and-active-dtype gate; a real GPU smoke plus
  permanent per-batch non-finite and zero-loss guards validate the actual path,
  and training still aborts if AMP is off (Session 10 / Batch 3, before valid
  training).
- **YOLO float32 loss under AMP adopted before the valid benchmark/training:**
  a positive-spread batch diagnostic still collapsed at epoch 2, batch 138,
  proving ordering was not the root cause. Ultralytics task assignment used
  float16 sigmoid scores that underflowed to zero on the one-class head. This
  initially retained float16 forward/backward; the later bfloat16 decision
  above supersedes that detail. Compute assignment and detector losses in
  float32, and restore ordinary seeded shuffle. The
  discarded sampler and all invalid runs are excluded; mixed loss precision
  is documented (Session 10 / Batch 3, before valid training).
- **YOLO target LR reduced to 0.001 before the valid benchmark/training:** the
  one-epoch warmup to 0.005 removed `NaN` but the classifier saturated to
  near-zero scores and epochs 2--3 had all-zero losses/mAP. Retain SGD,
  momentum 0.9, weight decay 0.0005, no Nesterov, and the one-epoch zero-to-
  target warmup, but use target LR 0.001. A per-batch guard now rejects both
  non-finite loss and five consecutive exactly-zero classification losses.
  This is disclosed as an optimizer asymmetry; invalid diagnostics are
  archived and excluded (Session 10 / Batch 3, before valid training).
- **YOLO one-epoch LR warmup adopted before the valid benchmark/training:**
  no-warmup full-data diagnostics reproducibly produced `NaN` classification
  loss at epoch 1, batch 29 while the AMP scale collapsed to 64. Use a linear
  one-epoch ramp from 0 to the same 0.005 target, with momentum fixed at 0.9;
  keep AMP, batch, model, data, and augmentation parity unchanged. Invalid
  attempts are archived and excluded from results. This is disclosed as an
  optimizer-schedule asymmetry (Session 10 / Batch 3, before valid training).
- **YOLO pretrained identity pinned before Batch 3 training:** official
  Ultralytics v8.4.0 `yolo11s.pt`, SHA-256
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`,
  stored locally at ignored path `results/checkpoints/pretrained/yolo11s.pt`
  (Session 10 / Batch 3, before training).
- **YOLO augmentation parity decided before Batch 3 training:** disable every
  Ultralytics stochastic extra (mosaic, mixup, cutmix, copy-paste, HSV jitter,
  flips, affine/perspective transforms, erasing, auto-augmentation, and
  multi-scale training) to match Faster R-CNN's deterministic resize-only
  pipeline. This prioritizes a controlled detector comparison over the usual
  augmentation-rich YOLO recipe. YOLO11s remains pinned to
  `ultralytics==8.4.110`; seed 17, 640-pixel input, AMP, SGD LR 0.005/momentum
  0.9/weight decay 0.0005, effective batch 4, and originally frozen BatchNorm
  statistics. The later stability decisions above supersede the LR and BN
  details. Validation-AP50:95 early stopping (minimum 8, patience 5, maximum
  30) matches the Faster R-CNN protocol as closely as the framework allows. The remaining
  loss/scheduler/architecture differences will be reported rather than hidden.
  (Session 10 / Batch 3, before training).
- **Faster R-CNN one-seed baseline complete:** after explicit approval, the
  full seed-17 run restarted from COCO weights and early-stopped at epoch 11
  after 7,017.8 seconds (1.95 hours). Epoch 6 is the exact best checkpoint:
  validation AP50:95 0.12764, AP50 0.33144, precision 0.14138, recall 0.68953,
  and F1 0.23464 on 750 images/277 boxes at score 0.25 and match IoU 0.50.
  Peak allocated training memory was 1,556.6 MiB. Best-checkpoint profiling is
  11.00 FPS, mean/p50/p95 latency 90.92/90.78/92.18 ms, 43,256,153 total and
  43,030,809 trainable parameters, 450.76 estimated GFLOPs, and a 165.38 MiB
  model. Best checkpoint:
  `results/checkpoints/faster_rcnn_rsna_seed17_full/best_model.pt`, SHA-256
  `9ec35c5d761f8e4bf7a43f7999f388ac1ffc0d533f62746409db280706dffab4`.
  All run/table/curve/checkpoint identities and hashes validate; the test split
  was not accessed. Stop for Batch 2 result review before Batch 3 (Session 8 /
  Batch 2).
- **Faster R-CNN three-epoch timing gate complete:** the clean official-data
  benchmark completed epochs 1--3 in 770.3, 551.5, and 473.0 seconds (29.91
  minutes total), with a 512.2-second steady-state estimate. The configured
  minimum eight epochs project to 1.21 hours; the 30-epoch sign-off upper bound
  is 4.34 hours with a conservative 4.02--4.66-hour range. Peak allocated GPU
  memory was 1,556.6 MiB. Epoch 3 had the best diagnostic validation AP50:95,
  0.10993; this is not a final baseline metric. The approval artifact is
  `results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json`
  (SHA-256
  `232460ae09827dfb780b0f5c6506bf9f545bbdc0e1483082c2c440035e8e8e8b`),
  bound to config SHA-256
  `ef1e3ebe1fbe3cf1a6e27bf8b9c12f61719c2ea8771c9758f64dc278dd0e2633`.
  Its exact train-mode approval check passes; stop pending explicit user
  sign-off before the one-seed full run (Session 7 / Batch 2 recovery).
- **Windows DataLoader memory adjustment before the timing gate:** the first
  official-data benchmark attempt completed all 1,750 epoch-1 training batches
  but failed before validation when a second six-worker pool hit `WinError
  1455` (paging file/commit limit). No epoch record or checkpoint was produced.
  Keep `num_workers: 6`, but use non-persistent train and validation pools so
  only one PyTorch worker pool exists at a time. Pool startup remains inside
  each measured epoch; the clean three-epoch gate restarts from COCO weights
  under the new configuration fingerprint (Session 7 / Batch 2 recovery).
- **Adopted Batch 2 runtime before training:** following the user's request to
  use the existing local CUDA environment and later directive to proceed, both
  detector arms are repinned before any smoke/benchmark/training result to
  Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1, Torch 2.6.0+cu124, and Torchvision
  0.21.0+cu124. CUDA 12.4 is wheel-bundled and uses driver 610.47. CUDA NMS,
  float16 AMP/GradScaler, negative-image Faster R-CNN training, and FLOP
  profiling passed direct probes. `requirements.txt`, `pyproject.toml`,
  `.python-version`, `uv.lock`, README, and reproducibility/baseline docs are
  aligned; Ultralytics remains 8.4.110 (Session 7 / Batch 2 recovery).
- **Official pixels complete:** the manually downloaded Kaggle aggregate ZIP is
  3,932,287,530 bytes with SHA-256
  `133acacf95aa68c4d219124b17937f31cec073052096b9f9b122180df9d9af18`.
  Its full CRC test passes, it has no duplicate/unsafe paths, and it contains
  exactly 26,684 training plus 3,000 competition-test DICOMs. Both metadata CSV
  hashes match the committed audit. All 5,000 selected DICOMs are present; a
  fresh official-source conversion produced exactly 5,000 manifest-matching
  PNGs, zero missing sources/errors, and byte-identical results for the 12
  earlier review images (Session 7 / Batch 2 recovery).
- **2026-08-04 Kaggle OAuth confirmation:** the browser authorization callback
  completed successfully, but Kaggle returned HTTP 403 while exchanging the
  authorization code at `security.OAuthService/ExchangeOAuthToken`. The CLI
  therefore remains on the legacy key and still receives 403 for all tested
  API calls. This confirms the blocker is at Kaggle's API/account/network
  policy layer rather than a missed browser authorization. No official file
  bytes were downloaded; use the authenticated Kaggle website from an
  authorized available network/location and place the three official files
  manually (Session 6 / Batch 2 recovery).
- **2026-08-04 Batch 2 recovery audit:** `kaggle==2.2.3` is now installed in
  `.venv`, and `C:\Users\Pouyan\.kaggle\kaggle.json` is structurally valid for
  account `alphalacrim`, but Kaggle returns HTTP 403 for public dataset lists,
  competition lists/files, and the RSNA download. A forced browser OAuth login
  was opened but did not complete; no official file bytes were downloaded. The
  existing Anaconda `torch-gpu` environment was also verified: Python 3.11.15,
  Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, CUDA available, CUDA NMS working,
  and AMP enabled. It has not been adopted because it conflicts with the pinned
  Python 3.13 / Torch 2.13.0+cu130 / Torchvision 0.28.0+cu130 experiment
  identity; changing that identity requires an explicit user decision
  (Session 5 / Batch 2 recovery).
- **Faster R-CNN Batch 2 configuration:** the one-seed baseline is
  `fasterrcnn_resnet50_fpn_v2` with Torchvision default COCO weights, seed 17,
  640×640 transform bounds, physical batch 2, two-step gradient accumulation
  (effective optimizer batch 4), float16 AMP, SGD at LR 0.005, and no stochastic
  augmentation. Pretrained BatchNorm running statistics are frozen while
  affine parameters remain trainable. Validation AP50:95 drives both the
  plateau scheduler and early stopping (minimum 8, patience 5, maximum 30
  epochs). See `configs/faster_rcnn.yaml` and
  `docs/FASTER_RCNN_BASELINE.md` (Session 3 / Batch 2).
- **Faster R-CNN timing and artifact gate:** a complete three-epoch
  train-plus-validation benchmark is mandatory before full training. Epoch
  timing includes equivalent best/last checkpoint I/O; approval is bound to
  the exact YAML, train/validation annotation and pixel manifests, source
  manifest, Torch/Torchvision/CUDA/driver/GPU identity, AMP, batch, and
  resolution. Full mode cannot run without that approved artifact. Final
  metrics use the shared COCO/operating-point evaluator, and a recoverable
  `finalize` mode regenerates tables, curves, FPS, parameters, checkpoint size,
  and mandatory finite GFLOPs without retraining (Session 3 / Batch 2).
- **Batch 2 data readiness:** canonical train/validation COCO metadata and
  pixels are complete (3,500/750 images; 1,267/277 boxes), and the separate
  750-image held-out split is also materialized for later batches. Timed modes
  still aggregate and report any missing train/validation path before importing
  Torch or initializing CUDA. The held-out test annotation is neither opened
  nor evaluated in Batch 2 (updated Session 7 / Batch 2 recovery).
- **RSNA patient-safe benchmark split:** the Kaggle `patientId` is an exam UUID,
  so grouping uses the official RSNA mapping to the original NIH filename and
  parses its patient prefix. Seed 17 selects 5,000 studies from 2,136 NIH
  patient groups, then splits them exactly 3,500/750/750. All three patient-key
  intersections are empty. Study strata are train 798/1,554/1,148, validation
  169/331/250, and test 169/331/250 for opacity/no-opacity-not-normal/normal.
  See `configs/dataset.yaml`, committed split manifests, and `docs/DATASHEET.md`
  (Session 2 / Batch 1).
- **Actual class list/count:** exactly one foreground detection class,
  `Lung Opacity` (`category_id=1`); background is implicit. `Normal` and
  `No Lung Opacity / Not Normal` are study-level sampling strata, not detector
  classes. Downstream class counts must be read from config/COCO rather than
  hardcoded (Session 2 / Batch 1).
- **Dataset choice:** RSNA Pneumonia Detection Challenge 2018 Stage 2 was chosen
  over both brain-MRI exports because it has stronger expert annotation
  provenance, coherent negative-image semantics, bespoke terms suitable for
  course research, and an official mapping that enables patient-safe grouping.
  The full 26,684-study labeled set is reduced to the fixed stratified 5,000
  subset required by §3. See `docs/DATASET_CHOICE.md` (Session 2 / Batch 1).
- **Canonical data contract:** both future detectors will read per-split COCO
  JSON generated by `src/data/prepare.py`. Preparation performs the full
  metadata audit, verifies the official mapping digest, preserves negative
  images with zero annotations, and records input hashes. Configured DICOM
  conversion uses `MONOCHROME1` inversion plus per-image min–max scaling to
  8-bit PNG. See `docs/DATASHEET.md` (Session 2 / Batch 1).
- **YOLO selection and pin:** YOLO11s (`yolo11s.pt`) with
  `ultralytics==8.4.110`. YOLO11 was selected over YOLO26 for its conventional
  anchor-free/NMS pipeline, greater continuity with medical-detector literature,
  and clearer comparison against two-stage Faster R-CNN; the small scale fits
  8 GB VRAM. See `docs/LITERATURE_REVIEW.md` and `configs/yolo.yaml`
  (Session 2 / Batch 1).
- **Phase 0 reproducibility contract:** every experiment entry point must call
  `initialize_reproducibility(seed, output_dir)` before CUDA initialization.
  Runs record `pip_freeze.txt` and structured seed/platform/Torch/CUDA/GPU/driver
  metadata in `run_environment.json`. Deterministic algorithms use warning mode.
  See `src/utils/seed.py` and `docs/REPRODUCIBILITY.md` (Session 1 / Batch 0).
- **Dependency baseline:** the Batch 0 core pins remain exact in
  `requirements.txt`; Batches 1–2 added exact Pillow, PyYAML, Pydantic, pydicom,
  Kaggle, Ruff, Matplotlib, and Ultralytics pins. The former Ultralytics
  placeholder is superseded by `ultralytics==8.4.110`; Batch 23 adds exact
  `pandas==3.0.3` and `seaborn==0.13.2` pins for the audited raincloud workflow
  (Sessions 1–3 and 55 / Batches 0–2 and 23).
- **YOLO augmentation-asymmetry handling:** resolved by disabling all extra
  Ultralytics stochastic augmentation for the primary comparison. See
  `configs/yolo.yaml`, `docs/YOLO_BASELINE.md`, and `docs/LIMITATIONS.md`.

## File Map

- `configs/vindr_external_v1.yaml`: frozen external scientific contract; root
  via `VINDR_CXR_ROOT`, with no machine-path fallback.
- `docs/VINDR_EXTERNAL_PROTOCOL.md`: canonical Batch 47 external prespecification
  and user review document, including source/access and observational-unit evidence.
- `docs/VINDR_EXTERNAL_PROTOCOL_v1.sha256.json`: exact byte hashes for protocol,
  config and internal dependencies; aggregate metadata-only release audit.

| Path | Purpose | Status |
|---|---|---|
| `PROJECT_SPEC.md`, `BATCHES.md`, `AGENTS.md` | Static requirements, sequence, and session protocol | authoritative |
| `CODEX.md`, `HANDOFF.md` | Living decisions and append-only session state | living |
| `.github/workflows/ci.yml` | Locked cross-platform formatting, lint, pytest, deterministic smoke, frozen scientific-artifact, and manuscript-claim gates with the CPU model extra | Batch 36 extension passes locally; no model training, large download, or GPU inference in CI |
| `LICENSE` | Canonical GNU Affero General Public License v3 text | published as `AGPL-3.0-only` at `5defa77` |
| `CITATION.cff`, `CHANGELOG.md`, `RELEASE_NOTES_v2.0.0.md` | Authoritative available software citation metadata, research-oriented release history, and the review-only paper-artifact release scope/delta | v2.0.0 candidate prepared; no DOI/release date/final paper citation invented and no release/tag published |
| `README.md`, `data/README.md` | Release landing page, exact setup, four-level reproducibility boundary, complete experiment sequence, artifact-command index, headline result, Definition of Done audit, and license/third-party scope | v2.0.0 candidate identifies the version/tag relationship, distribution boundary, CI scope, release notes, and citation metadata |
| `.python-version`, `requirements.txt`, `pyproject.toml`, `uv.lock`, `src/meddet_benchmark/__init__.py` | Exact Python 3.11 / CUDA 12.4 phased-workflow dependency pins, SPDX package-license metadata, and release/package version | D-009 aligns project, package, and lock root at candidate version 2.0.0 |
| `configs/dataset.yaml` | RSNA paths, class/stratum map, conversion, subset, split, and EDA settings | done |
| `configs/cohort_characteristics.yaml`, `src/data/cohort_characteristics.py`, `tests/test_cohort_characteristics.py` | Hash- and count-gated aggregate DICOM-header cohort extraction with optional `RSNA_DICOM_ROOT`, explicit age-unit/range policy, privacy guards, and regression tests | Batch 44 complete; reads no pixels, retains no row-level identifiers, and performs no training or inference |
| `configs/inference_timing_v1.yaml`, `src/benchmark_inference.py`, `tests/test_inference_timing.py` | Matched decoded-host timing protocol, reporting-hardware gate, ordinary-inference parity and offline verifier | Batch 45 complete; inference only, strict batch 1, three technical repeats |
| `docs/COMPUTE_TIMING.md`, `results/tables/inference_timing_v1*.csv`, `results/figures/inference_timing_v1.png`, `results/logs/phase45_inference_timing_v1/` | Primary timing docs/table/Figure 5, raw intervals, repetition rows, environment and historical preservation manifest | Batch 45 complete on intended RTX 4060 Laptop GPU; 46 historical files unchanged |
| `configs/yolo.yaml` | Strict YOLO11s data/model/runtime/training/evaluation/profile/artifact config | implemented; smoke, timing gate, full run, and profiling complete |
| `configs/faster_rcnn.yaml` | Strict Faster R-CNN model/runtime/training/evaluation/profile/artifact config | implemented; timing gate and full run complete |
| `configs/{faster_rcnn,yolo}_seed{42,137,271,314}.yaml`, `configs/evaluation.yaml` | Seed-only rerun identities and the ten-run Phase 5 evaluation contract | all five seeds per detector trained and evaluated in Batch 16 |
| `configs/evaluation_n3_archive.yaml`, `configs/{threshold_sweep,threshold_selection}_n3_frozen.yaml` | Exact historical Phase 5/threshold identities plus archive-aware n=3 reproduction routing | frozen around Batch 16 replacement; hashes verified |
| `configs/{threshold_sweep_n5_sensitivity,froc_n5_sensitivity,pareto_n5_sensitivity,operating_regime_n5_sensitivity}.yaml` | Five-run frozen-bundle sensitivity routing with explicit historical n=3 inputs, fixed validation-selected thresholds, expected seeds, and versioned outputs | Batch 35 complete; no training, inference, threshold reselection, or overwrite of n=3 artifacts |
| `configs/threshold_sweep.yaml`, `src/evaluate_threshold_sweep.py`, `tests/test_threshold_sweep.py` | Hash-bound offline 99-threshold sweep, fixed operating targets, official COCO PR exposure, aggregation, plots, and regression tests | historical Batch 10 n=3 values preserved; Batch 35 adds explicit n=5 routing and seed-271 coverage |
| `configs/pareto.yaml`, `src/plot_pareto_frontier.py`, `tests/test_pareto_frontier.py` | Strict offline join, aggregate cross-checks, threshold-aware recall selection, run-level/aggregate outputs, detector-level dominance classification, and 2x2 seed plot | historical n=3 preserved; Batch 35 adds an explicitly labeled five-run equal-run sensitivity |
| `configs/corruptions.yaml` | Fixed Phase 6 sample, checkpoint, evaluator, 7×5 corruption, and output contract | complete |
| `src/data/download.py` | Secret-safe Kaggle competition downloader and clear failure diagnostics | done |
| `src/data/prepare.py` | Metadata audit, digest check, patient grouping, subset/split, COCO, DICOM conversion | done |
| `src/data/visualize.py` | Deterministic distribution and annotation-sample EDA | done |
| `data/manifests/rsna-pneumonia-5000-audit.json` | Committed machine-readable audit and exact counts | done |
| `data/splits/rsna-pneumonia-5000/*.csv` | Patient-safe image manifests | done |
| `data/processed/rsna-pneumonia-5000/annotations/*.json` | Generated canonical COCO annotations | local/ignored; regenerable |
| `results/figures/rsna_class_distribution.png` | Selected split class/stratum distribution | done |
| `results/figures/rsna_annotation_samples.png` | Twelve real radiographs with labels/boxes | done |
| `docs/DATASET_CHOICE.md` | Three-candidate inspection and selection rationale | done |
| `docs/DATASHEET.md` | Collection, composition, patient split, audit, aggregate header cohort characteristics, processing, terms, and bias | Batch 44 adds age/sex/projection totals with explicit age-unit and no-fairness caveats |
| `docs/DECISION_LOG.md` | Append-only project decisions through D-015 | D-015 permits explicitly caveated nominal-year interpretation of unitless numeric `PatientAge` solely for aggregate cohort description and prohibits fairness/subgroup inference from these tags |
| `docs/PROJECT_PLAN.md` | Reconciled pre-implementation plan preserving superseded proposals while distinguishing frozen historical analyses from research-track corrections | Batch 28 records independent within-detector run resampling and estimand separation |
| `docs/LITERATURE_REVIEW.md` | YOLO11, detector paradigms, Grad-CAM/XAI, related work, robustness, and the controlled-comparison/operating-regime gap | Batch 42 aligns the FROC contribution with the approved 0.00001 boundary and conservative bound; existing BibTeX citations unchanged |
| `docs/HYPOTHESES.md` | Primary question plus six retrospective, artifact-checkable hypotheses with endpoint-specific evidence scopes | Batch 42 routes H2 FROC evidence to the observed exact-score frontier while retaining historical grids |
| `report/paper_draft.md`, `report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md`, `report/A_Controlled_Comparative_Study___article.pdf`, `report/report.md`, `report/references.bib` | Canonical editable manuscript, noncanonical long alternate, its rendered PDF derivative, preserved historical/full technical report, and resolving bibliography | D-010 hierarchy retained; Batch 46 focuses and shortens only the canonical manuscript and refreshes primary references; alternate/PDF/historical report remain unchanged |
| `docs/INTERNAL_PAPER_FOCUS_AUDIT.md`, `scripts/check_bibliography.py`, `tests/test_check_bibliography.py` | Batch 46 prerequisite/editorial audit and repeatable offline bibliography syntax/citation checks | Complete; 16 canonical citations resolve in the shared 37-entry database; scientific provenance and author facts are unchanged |
| `docs/LIMITATIONS.md` | Consolidated dataset, cohort-header, compute, comparison, calibration, decision analysis, robustness, explainability, statistics, reporting, deployment, and regulatory limitations | Batch 44 adds unitless-age, limited-sex/projection, and no-subgroup/fairness caveats while retaining Batch 43 threshold scope |
| `docs/FASTER_RCNN_BASELINE.md` | Batch 2 architecture, optimization, metrics, timing, and profiling protocol | complete with final measurements |
| `docs/YOLO_BASELINE.md` | Batch 3 architecture, parity/stability decisions, timing, metrics, and profiling | complete with final measurements |
| `docs/QUANTITATIVE_COMPARISON.md` | Unified metric definitions, compute caveats, commands, five-seed held-out results, and exact GFLOP counting/sanity-check record | Batch 16 n=5/n=4 clean comparison complete; n=3 analyses explicitly historical |
| `docs/ROBUSTNESS.md`, `docs/LIMITATIONS.md` | Phase 6 sampling, corruption grid, raw/relative results, interpretation, and scope | Batch 15 reconciled to patient-cluster inference and digital-not-clinical robustness wording |
| `src/utils/seed.py`, `docs/REPRODUCIBILITY.md` | Reproducibility utilities plus software/committed-analysis/inference/retraining contract, lock roles, RNG/nondeterminism, release-candidate gate, checkpoint release procedure, and aggregate cohort command | Batch 44 adds the header-only cohort reproduction route and root override contract |
| `results/tables/rsna_cohort_characteristics.csv`, `results/logs/phase44_cohort_characteristics/summary.json` | Aggregate train/validation/internal-test/all-study cohort descriptors plus immutable-input, metadata-policy, and output provenance | Batch 44 complete; 5,000-study totals reconcile exactly and no row-level identifier is emitted |
| `results/scientific_artifact_manifest.json`, `scripts/{build_scientific_artifact_manifest,verify_scientific_artifacts}.py` | Fixed 67-artifact manuscript-critical inventory plus maintenance builder and CI-safe hash/schema/reference verifier | Batch 44 binds the aggregate cohort CSV and provenance summary; builder never runs in CI, so evidence changes require reviewed manifest diffs |
| `report/paper_claim_sources.yaml`, `scripts/verify_paper_claims.py` | Exact claim-to-source bindings with deterministic calculations, unique manuscript matching, and explicit rounding tolerance | Batch 44 binds six cohort claims plus the descriptive/no-fairness semantic guard; 49 claims/guards pass |
| `results/checkpoint_release_manifest.json` | Ten Phase 5 best-checkpoint paths, names, sizes, SHA-256/config bindings, local audit, conditional release assessment, and null public URL | Batch 36 audit: all ten local and matching, 962,924,817 bytes; binaries not committed or uploaded |
| `tests/test_{download,prepare,visualize}.py` | Batch 1 acquisition/conversion/split/EDA tests | done |
| `src/models/faster_rcnn_*.py`, `src/models/train_faster_rcnn.py` | Strict data adapter, model, AMP trainer, unified validation, reporting, profiling, and gates | implemented; smoke, benchmark, full run, and profiling complete |
| `src/models/yolo_*.py`, `src/models/train_yolo.py` | Strict YOLO data view, mixed-precision trainer, unified validation, reporting, profiling, and gates | implemented; smoke, benchmark, full run, and profiling complete |
| `src/evaluate.py`, `tests/test_evaluate.py` | One canonical adapter-to-pycocotools evaluation path and conditional-null-safe multi-seed aggregation | verified on ten checkpoints; nonconditional n=5 and detector-specific localization n |
| `results/tables/detector_comparison*.csv`, `results/logs/phase5_evaluation/` | Per-seed metrics, mean ± sample SD, full provenance, ten prediction bundles, and explicit n=3 archives | n=5 primary complete; n=3 config/summary/tables preserved byte-identically |
| `configs/yolo_seed_stability.yaml`, `src/analyze_yolo_seed_stability.py`, `tests/test_yolo_seed_stability.py`, `results/tables/yolo_seed_stability.csv` | Offline curve/score-threshold stability audit at 0.25, historical 0.05, and COCO floor 0.001 | complete; seed 271 classified as normal-convergence confidence degeneracy |
| `results/tables/{threshold_sweep*,precision_recall_curves*,threshold_operating_targets}.csv`, `results/figures/{precision_recall_curves,f1_vs_threshold}.png`, `results/logs/phase10_threshold_sweep/` | Three-seed exploratory test threshold/official-PR curves, fixed-target results, figures, and provenance hashes from the six frozen bundles | Historical Batch 10 n=3 artifacts unchanged |
| `results/tables/{threshold_sweep,threshold_sweep_per_seed,threshold_operating_targets,precision_recall_curves,precision_recall_curves_per_seed}_n5_sensitivity.csv`, `results/figures/{precision_recall_curves,f1_vs_threshold}_n5_sensitivity.png` | Five-run exploratory threshold and threshold-free PR sensitivity with per-run and aggregate outputs | Batch 35 complete; seed 271 retained exactly as observed |
| `configs/threshold_selection.yaml`, `src/evaluate_threshold_selection.py`, `tests/test_threshold_selection.py` | Inference-only validation bundle materialization, validation mean-F1 selection, one-shot frozen-test application, tables, and provenance | Batch 14 thresholds frozen at 0.69/0.05 over n=3; archive-routed after Batch 16 |
| `results/tables/{validation_threshold_sweep*,selected_operating_points*}.csv`, `results/logs/phase14_threshold_selection/` | Six validation bundles, 99-point validation sweeps, selected thresholds, final test precision/recall/F1, hashes, and environment | Batch 14 complete; primary single-threshold source |
| `configs/threshold_selection_n5_validation_sensitivity.yaml`, `src/analyze_validation_threshold_sensitivity.py`, `tests/test_validation_threshold_sensitivity.py` | Hash-gated completion of four missing validation bundles, exact historical-selector reproduction, five-run post-hoc validation selection, unchanged five-run test application, classification, and overwrite/test-selection/seed-271 guards | Batch 43 complete; selects 0.70/0.01 as post-hoc sensitivity only; 0.69/0.05 historical provenance unchanged |
| `results/logs/phase43_threshold_selection_n5_validation_sensitivity/`, `results/tables/*n5_validation_sensitivity.csv`, `results/tables/threshold_selection_n3_vs_n5_validation_conclusions.csv` | Four new hash-bound validation bundles, 990 per-run and 198 aggregate validation-sweep rows, 20 test rows, four aggregate operating rows, 11 classified conclusions, and end-to-end provenance | Batch 43 complete; no training or test-label selection; Faster seed-314 one-FP validation re-inference drift explicitly bounded and recorded |
| `docs/THRESHOLD_ANALYSIS.md` | Historical n=3 and five-run test/validation sensitivity results, frozen n=3 precedence, post-hoc n=5 selection, fixed-threshold application, comparison audit, and reproduction | Batch 43 complete; 0.70/0.01 never called prospectively frozen |
| `src/analyze_operating_regime_sensitivity.py`, `tests/test_operating_regime_sensitivity.py`, `results/tables/{operating_regime_n5_inventory,operating_regime_n3_vs_n5_conclusions,selected_operating_points_n5_sensitivity,selected_operating_points_per_seed_n5_sensitivity}.csv`, `results/logs/phase35_operating_regime_n5/` | Frozen-input inventory, unchanged-threshold application, 19-row n=3-versus-n=5 conclusion classification, and end-to-end provenance | Batch 35 complete; regression guards require all five runs and seed 271 |
| `configs/threshold_calibration.yaml`, `src/stats/threshold_calibration.py`, `tests/test_threshold_calibration.py` | Validation-only recall-weighted F-beta sweep, hierarchical intervals, lower-bound selection, stability diagnostics, separate hypothetical linear loss, plotting, provenance, and isolation regressions | Batch 29 complete; no training, inference, test-label selection, or threshold propagation |
| `results/tables/recall_weighted_fbeta_threshold_{summary,stability}.csv`, `results/tables/hypothetical_detection_error_loss_summary.csv`, `results/figures/recall_weighted_fbeta_threshold_sensitivity.png`, `results/logs/phase29_threshold_sensitivity/summary.json`, `docs/THRESHOLD_CALIBRATION.md` | Eight F-beta selections, 792 candidate stability rows, eight separate loss selections, full curves/hashes, D-006 method, findings, and scope | Batch 29 canonical; frozen F-beta thresholds unchanged and YOLO beta 3--10 boundary status explicit |
| `results/tables/threshold_calibration_pre_batch29_archive.csv`, `results/figures/threshold_calibration_pre_batch29_archive.png`, `results/logs/phase19_threshold_calibration/archive/summary_pre_batch29.json` | Exact historical pre-remediation threshold artifacts | Content-identical to the former tracked Batch 19 files; noncanonical terminology archive only |
| `src/clinical/decision_curve.py`, `tests/test_decision_curve.py` | Conventional-DCA probability-semantic gate, typed validation-frozen probability input, typed elicited decision threshold, standard formula primitive, and raw/untyped predictor or `p_t` rejection tests | Batch 30 guard complete; no project standard DCA is run because complete validation calibration inputs do not exist |
| `configs/raw_score_utility.yaml`, `src/clinical/raw_score_utility.py`, `tests/test_raw_score_utility.py` | Strict non-standard classification, archive hash audit, exact historical arithmetic reproduction/relabeling, incomplete calibration-salvage check, plotting, provenance, and tests | Batch 30 supplementary audit complete; no training, inference, calibrator fitting, or standard DCA |
| `src/clinical/archive/decision_curve_pre_batch30_nonstandard.py`, `configs/decision_curve_pre_batch30_nonstandard_archive.yaml`, `results/{tables,figures,logs}/...pre_batch30_nonstandard...` | Exact original Batch 20 source, config, 198-row table, figure, and provenance preserved with fixed SHA-256 identities | Historical only; not canonical conventional-DCA evidence |
| `results/tables/raw_score_threshold_utility_summary.csv`, `results/figures/raw_score_threshold_utility_sensitivity.png`, `results/logs/phase30_raw_score_utility/summary.json`, `docs/DCA_ANALYSIS.md` | Relabeled non-standard five-seed raw-score utility/sensitivity artifact, arbitrary-scale figure, zero-difference audit, prevalence boundary, salvage assessment, and exact provenance | Supplementary/limitations only; excluded from the paper Results and clinical interpretation |
| `configs/calibration.yaml`, `src/stats/calibration.py`, `tests/test_calibration.py` | Uniform class-conditioned five-dimensional D-ECE, canonical matching, edge-safe binning, occupancy/support, predeclared bin/minimum/floor sensitivity, four figures, provenance, and seed-branch regression | Batch 33 v2 canonical; original 5-bin/minimum-8 estimate preserved; no retraining or inference |
| `results/tables/calibration_{summary,support,sensitivity}_v2.csv`, `results/figures/{reliability_diagrams_confidence_marginal,calibration_support_occupancy,calibration_binning_sensitivity,calibration_confidence_floor_sensitivity}_v2.png`, `results/logs/phase33_calibration_support_v2/summary.json`, `docs/CALIBRATION_ANALYSIS.md` | Per-run descriptive endpoint/support, predeclared robustness grid, accurately scoped figures, exact hashes, method, findings, DCA separation, and limitations | Batch 33 canonical; historical Phase 18 artifacts retained but superseded for reporting |
| `configs/{froc,froc_n3_archive}.yaml`, `src/plot_froc_curves.py`, `tests/test_froc.py`, `results/tables/froc_operating_points.csv`, `results/figures/froc_curves.png`, `docs/FROC_ANALYSIS.md` | Test-sweep FROC redescription, five non-interpolated FP/image budgets, figure, tables, provenance, and interpretation | historical n=3 artifact preserved; Batch 35 adds n=5 per-run/aggregate curves and visibly labeled sensitivity outputs |
| `results/tables/{froc_curves,froc_curves_per_seed,froc_operating_points}_n5_sensitivity.csv`, `results/figures/froc_curves_n5_sensitivity.png` | Five-run FROC sensitivity, including seed 271's observed low-score behavior and explicit run count | Batch 35 complete |
| `configs/{evaluation_froc_lower_floor_v4,froc_exact_score_v4}.yaml`, `src/{collect_froc_lower_floor_predictions,analyze_exact_score_froc}.py`, `tests/test_{froc_lower_floor_inference,exact_score_froc}.py`, `results/tables/froc_*v4.csv`, `results/figures/froc_exact_score_v4.png`, `results/logs/phase42_froc_{lower_floor,exact_score}_v4/` | Hash-gated, restart-safe approved inference-only collection plus config-driven exact-score endpoints, per-run/aggregate budgets, historical/v3 comparisons, monotonicity/canonical-matcher audits, and conservative residual bound | Batch 42 v4 current; no retraining; original Phase 5/historical/v2/v3 artifacts preserved; no still-lower run authorized |
| `results/figures/pareto_frontier.png`, `docs/PARETO_ANALYSIS.md` | Four seed-level accuracy-efficiency panels, strict dominance labels, validation-selected test recall protocol, and scenario-conditional interpretation | historical n=3 artifact preserved; Batch 35 adds explicitly labeled n=5 run-level/aggregate sensitivity |
| `results/tables/{pareto_points,pareto_summary}_n5_sensitivity.csv`, `results/figures/pareto_frontier_n5_sensitivity.png` | Five-run equal-run Pareto aggregation with five hardware rows per detector, frozen n=3-selected recall thresholds, and four strict-cloud labels | Batch 35 complete; no n=3/n=5 metric mixing within a frontier |
| `src/robustness/`, `src/meddet_benchmark/corruptions.py`, `tests/test_{corruptions,robustness}.py` | Deterministic Albumentations grid, sample audit, resumable dual-detector inference, aggregation, plots, and tests | implemented; full grid complete |
| `results/tables/robustness*.csv`, `results/figures/robustness*.png`, `results/logs/phase6_robustness/` | Raw/relative/family curves, 72 bundles, hashes, and full Phase 6 summary | complete; incorporated in report |
| `configs/acquisition_shifts.yaml`, `src/robustness/radiography_shifts.py`, `tests/test_radiography_shifts.py` | Per-image DICOM metadata/missingness audit, exact DICOM LINEAR and MONOCHROME regressions, separately classified Poisson-like intensity and Gaussian-blur sensitivities, pre/post-min-max diagnostics, historical-bundle verification, and CPU-only audit mode | Batch 32 canonical interpretation; no training, inference, or checkpoint mutation |
| `results/tables/{acquisition_shift_dicom_metadata_audit,acquisition_shift_preprocessing_per_image,acquisition_shift_preprocessing_summary,radiography_synthetic_shift_results}.csv`, `results/logs/phase32_acquisition_shift_audit/`, `docs/ACQUISITION_SHIFTS.md` | 300 header rows, 3,000 per-image diagnostics, ten transformation summaries, corrected 20-row labels, current-standard interpretation, and SHA-256 provenance over unchanged Phase 22 bundles | Batch 32 complete; internal synthetic sensitivity only; Phase 22 outputs preserved as superseded history |
| `configs/raincloud_metrics.yaml`, `src/plot_raincloud_metrics.py`, `tests/test_raincloud_metrics.py` | Strict 14-metric layout, aggregate-to-seed mean/SD/n audit, conditional-null preservation, deterministic Seaborn cloud/box/rain rendering, provenance, and regression tests | Batch 23 complete; no model inference or metric recomputation |
| `results/figures/raincloud_metrics.png`, `results/logs/phase23_reporting/raincloud_metrics_summary.json` | Seven predictive and seven compute/hardware seed distributions with actual finite n in every panel, input hashes, counts, undefined reasons, and figure hash | Batch 23 complete; 138 finite observations and no n=3 mixing |
| `docs/REPORTING_CHECKLIST.md`, `docs/SUPPLEMENTARY.md` | Current-paper CLAIM 2024 primary crosswalk plus explicitly analogical STARD-AI 2025/TRIPOD+AI 2024 mappings and routes to exhaustive evidence | Batch 44 refreshes the manuscript hash and routes the aggregate cohort artifacts; it is not an official compliance claim |
| `docs/HYPOTHESIS_TRACEABILITY.md` | Exact H1--H6 wording, retrospective status, operational endpoints, split/seed scope, artifacts, manuscript sections, results, multiplicity, and limitations | Batch 43 distinguishes historical n=3 threshold precedence from post-hoc n=5 validation sensitivity; no hypothesis is labeled preregistered |
| `docs/CITATION_AUDIT.md` | Exhaustive method/standards/dataset/DICOM/statistics/calibration/DCA/XAI source verification and exact claim mapping | Batch 34 complete; all 29 manuscript keys audited, no unsupported or wrong-attribution row remains after corrections |
| `docs/AUTHOR_DECLARATIONS_TODO.md` | Human-only funding, interests, ethics/data-use, consent, contributions, availability, and PPI actions | Batch 34 complete as an unresolved action register; no declaration fact is invented |
| `docs/REVIEW_REMEDIATION_AUDIT.md` | Independent repository-first adjudication of 17 external-review findings, 18 numerical claims, H1--H6 coverage, citation support, Git/CI state, and submission priorities | Historical Session 60 audit plus Batch 33 R6/R15 calibration-remediation addendum |
| `docs/FINAL_SUBMISSION_AUDIT.md` | Batch 38 adversarial final gate: blockers, should-fix items, verified areas, 29-number traceability, analysis scopes, language/citation/artifact checks, and exact rerun commands | Complete; software/evidence checks are clean, but unresolved author declarations and the public-release decision make the disposition NOT READY FOR SUBMISSION |
| `docs/V7_Q2_VINDR_BASELINE_AUDIT.md` | Batch 41 repository/manuscript adjudication, checkpoint and prediction inventories, n=3/n=5/FROC baseline, and V7 work classification | Complete for owner review; no scientific or manuscript content changed |
| `report/paper_draft.md`, `report/report.md`, `report/references.bib` | Current controlled-comparison manuscript, preserved historical/full technical report, and resolving bibliography | Batch 44 adds aggregate cohort characteristics to the current manuscript; prior calibration, XAI, statistical, threshold, and citation decisions remain in force |
| `configs/explainability.yaml`, `src/explainability/`, `tests/test_{gradcam,pointing_game,explainability}.py` | Strict paired Grad-CAM targets, stride-matched hooks, localization metrics, case selection, figures, and regression tests | complete |
| `results/tables/gradcam*.csv`, `results/figures/gradcam*.png`, `results/logs/phase7_explainability/` | All 222 per-target records, six aggregate rows, 18 paired qualitative records, three figures, hashes, and full Phase 7 summary | complete; incorporated in report |
| `docs/EXPLAINABILITY.md` | Exact target/layer/metric protocol, results, explicit interpretation, reproduction, and caveats | Batch 31 keeps localization descriptive and rejects strong-localization wording from small absolute overlaps |
| `configs/xai_sanity.yaml`, `src/explainability/sanity_checks.py`, `tests/test_xai_sanity.py` | Six-stage detector-specific cascading parameter controls, accurately labeled input-pixel control, normalized Pearson/Spearman/SSIM, partition and checkpoint audits, versioned artifacts, provenance, and regressions | Batch 31 complete; canonical randomized-training test not performed and checkpoint files immutable |
| `results/tables/gradcam_sanity*.csv`, `results/figures/gradcam_sanity*.png`, `results/logs/phase{21_xai_sanity,31_xai_sanity_v2}/`, `docs/XAI_SANITY.md` | Historical four-row/200-record control artifacts retained plus v2 14-row/700-record cascade, panel, fixed-subset manifest, hashes, exact method, results, and claim boundary | Batch 31 v2 canonical; Batch 21 table/figure retained unchanged as historical |
| `configs/statistics.yaml`, `src/stats/`, `tests/test_statistics.py` | Independent within-detector trained-run/patient-cluster bootstrap, checkpoint-conditional permutation sensitivity, nonlinear prediction-level reconstruction, endpoint eligibility, Holm correction, archives, diagnostics, and regressions | Batch 28 estimand-separated implementation complete |
| `results/tables/statistical_*.csv`, `results/logs/phase8_statistics/` | Seven clean primary intervals plus checkpoint-conditional p-values, 497 frozen robustness rows, run-influence diagnostics, and paired-seed/n=3/image-level archives | Batch 28 clean n=5/5 or 5/4 primary complete; robustness byte-identical |
| `docs/STATISTICAL_ANALYSIS.md` | Named inferential targets, seed-coupling audit, independent-run bootstrap, endpoint-level results, CI/p-value explanation, diagnostics, archives, reproduction, and caveats | Batch 28 canonical statistical interpretation |
| `tests/` | Data/cohort, model, evaluator, threshold/Pareto/FROC/calibration/DCA scale guard/raw-score audit, corruption/acquisition shifts, explainability/sanity, statistics, reproducibility, artifact/claim verification, stability, raincloud, and reporting regressions | 333 passing; one declared environment-conditional skip after Batch 44 |
| older tests/configs | Pre-workflow legacy implementation | non-authoritative; reconcile before use |
| `src/meddet_benchmark/` | Shared tested operating-point and COCO evaluator used by both detector baselines | package version aligned at 2.0.0; package smoke passes |

## Current phase

**Batch 48 is complete locally; stop for adapter/protocol review.** All 3,000
official test DICOMs and derived PNGs are validated, with 84 strict-target
positive images, 95 boxes and 2,916 negatives. Canonical COCO and the manifest
remain private; `results/vindr_external_v1/adapter_preflight.json` and
`docs/VINDR_ADAPTER_PREFLIGHT.md` record the completed preflight. The Batch 47
protocol/config and all 38 frozen dependencies remain unchanged. README has
the executable preparation and test commands. No detector inference or
external performance analysis occurred. Review the protocol and preflight
before separately authorizing Batch 49; do not start it automatically. No
staging, commit or push occurred. Existing unrelated changes are preserved.

The user subsequently requested a scoped local commit of the completed VinDr
adapter and its required, still-uncommitted Batch 47 freeze (Session 101).
The commit includes the protocol/config/sidecar, D-017, adapter/config/tests,
aggregate preflight, review document, relevant README section and tracked
session-state history. Restricted data and unrelated local files remain
excluded; the separate wget README hunk stays unstaged. No push requested.

### Previous phase: Batch 47

**Batch 47 is complete locally; stop for protocol review.** The external
contract and 38 dependency hashes are frozen in the three VinDr v1 files
above, with D-017 and executable offline verification/planned later commands
in README. No external detector inference or performance inspection occurred.
Next is a separately requested Batch 48 adapter/data-integrity session. User
review of `docs/VINDR_EXTERNAL_PROTOCOL.md`, passing Batch 48 and separate
authorization are required before Batch 49. Existing Batch 46 manuscript,
historical artifacts, download-tooling edits and unrelated files are preserved.
No staging, commit or push occurred in Batch 47.

### Previous phase: Batch 46

**Batch 46 is complete locally; stop for review of the focused internal paper.**
`report/paper_draft.md` is the sole canonical editable manuscript. It is now
about 58% shorter, centered on operating-point definition and the five-run
internal evidence; title and abstract await external results. Batches 42--45
passed the prerequisite gate. Current literature and citation/reporting audits,
66 numerical claim bindings, bibliography syntax/resolution, and 20 affected
tests pass. No scientific output, training configuration, frozen prediction,
checkpoint, or historical/alternate manuscript was changed. Author declarations
remain placeholders. The next batch is not authorized by this session.
No staging, commit or push occurred; the pre-existing staged CODEX/HANDOFF
changes and unrelated untracked files were preserved. The publication-state
record below is historical and was not acted on during this editorial batch.

The user subsequently requested the scoped local Batch 46 commit with the
exact message `Focus the Internal Q2 Manuscript and Update Related Work`
(Session 97). The commit includes the eleven Session 96 paths and these
session-state notes, including previously staged publication-state history.
Unrelated untracked files stay outside the commit; no push is requested.

### Previous phase and publication state (through Session 95)

**Batch 45 is complete locally on `main`, based on Batch 43--44 commit
`21defcfe6a2c25cb7cfbaada0fb255755b9c111d`.** Primary manuscript runtime
now uses the versioned matched decoded-host boundary, measured on the intended
reporting laptop without training. The two primary checkpoints were timed on
100 identical images for three full technical repetitions at batch 1 after
10 warm-up images per repetition. Faster R-CNN / YOLO11s give 20.91 / 57.56
FPS and 47.20 / 17.29 ms median latency (IQR 1.22 / 1.19 ms). Every one of
600 timed outputs exactly agrees with ordinary file inference under the same
explicit AMP. All 46 protected historical files remain unchanged; old timing
and Pareto/raincloud panels are labeled historical. The 72-artifact manifest,
64 numerical claims/semantic guards, 357-test suite plus one declared skip,
Ruff and timing verifier pass. The exact command, hardware/software and raw
per-image/repetition provenance are recorded in README and COMPUTE_TIMING.
No retraining or change to frozen accuracy bundles occurred. The user
explicitly requested the scoped Batch 45 local commit with the exact message
`Standardized End-to-End Inference Timing` in Session 94, producing `1318ee3`.
In Session 95 the user requested a direct `main` push. The configured origin
is `https://github.com/Alpha-lacrim/medical-object-detector-benchmark.git`;
a fetch confirmed that `21defcf` and `1318ee3` were the two outgoing commits
with no remote divergence. Automatic approval review rejected the push before
execution because it requires explicit confirmation of this destination and
payload. No remote change occurred. A local session-state commit records the
block; pushing remains pending explicit destination/payload approval.
Narrow `.gitattributes` exceptions preserve the new timing artifacts' exact
bytes so Git normalization cannot invalidate their recorded SHA256 hashes.
Batch 46 and later work, external testing, author declarations, checkpoint
publication and further FROC inference remain outside this session.

## Residual limitations / reproducibility risks

- VinDr v1 is a frozen external testing plan, not evidence of transportability
  yet. Image-level resampling cannot account for unknown repeated patients.
  Strict local-target ontology differs from RSNA; FP32/AMP inference-path
  differences are also disclosed. Batch 48 passed full source-image checksum,
  decoder and geometry checks, including JPEG 2000 lossless support. This is
  ingestion evidence only; native inference/AMP checks remain Batch 49. Do not
  publish restricted image-linked derivatives or tune any protocol choice on
  external performance. Do not regenerate v1 hashes to conceal a change.

- Batch 45 standardizes timing boundaries but remains one primary checkpoint
  per detector and one hardware/software state. Three technical timing repeats
  do not extend training-replicate uncertainty. Different AMP dtypes, native
  transforms/API overhead, and excluded disk/clinical workflow costs remain
  explicit. Historical timing/Pareto axes cannot be relabeled as matched v1.
- Kaggle API/OAuth remains denied on the current route, but it no longer blocks
  Batch 2 because the complete official aggregate archive was acquired manually
  and verified. Future clean acquisitions may still need manual browser download
  from an authorized available network/location.
- The adopted Anaconda runtime is external to the repository, so reproducible
  commands and run-level `pip_freeze.txt`/`run_environment.json` are mandatory.
  Both detector arms must use this same locked Python 3.11/Torch 2.6/CUDA 12.4
  identity; changing it after the timing benchmark invalidates approval.
- The official Kaggle CSV hashes reproduce the values recorded in the committed
  audit; the earlier mirror-provenance uncertainty for metadata is resolved.
- Batch 44 adds limited aggregate `PatientAge`, `PatientSex`, and
  `ViewPosition` descriptions from DICOM headers. Every selected age value is a
  numeric `AS` string without an encoded unit; the nominal-year interpretation
  is disclosed, and one value above 120 is excluded. These tags are not
  clinically verified demographics, contain no race/ethnicity or other social
  variables, and support neither subgroup performance nor fairness inference.
- Per-image min–max DICOM conversion is deterministic and config-declared but
  does not reproduce vendor-specific window/VOI display processing. This remains
  a preprocessing limitation.
- Batch 32 establishes that all 300 pre-PNG objects are 8-bit, lossy,
  workstation-converted Secondary Capture images with `Modality=CR`, rather
  than the Computed Radiography Image Storage SOP Class. All signal-
  relationship, modality/rescale, VOI, presentation-intent, and documented
  processing fields needed to characterize or invert the stored scale are
  absent. The DICOM `LINEAR` settings are synthetic display sensitivities;
  Poisson-like perturbations are physically unsupported as dose/quantum-noise
  proxies and retained only generically; Gaussian kernels are generic blur/
  spatial-resolution proxies. Per-image min-max partly cancels center shifts,
  almost erases the wide-window condition, and can re-stretch blur differences.
  DSI remains one-sample/one-checkpoint descriptive sensitivity, not clinical
  robustness, scanner validation, or external transportability.
- `pyproject.toml`/`uv.lock` match the reviewed Ultralytics 8.4.110 decision and
  adopted Python 3.11/CUDA 12.4 runtime; D-009 also aligns the project, package,
  and candidate research-release version at 2.0.0.
- The ten exact best checkpoints are currently available only as local,
  Git-ignored files. Their release is feasible only after the human license,
  attribution, Torchvision-pretraining permission, and serialized-metadata
  review recorded in `results/checkpoint_release_manifest.json`; no public URL
  exists. A clean checkout can verify/replay committed evidence but cannot run
  exact checkpoint inference until matching external checkpoint assets and
  licensed processed data are supplied.
- YOLO augmentation parity is resolved: all Ultralytics-only stochastic
  augmentations are disabled for the controlled baseline. This may understate
  performance under YOLO's conventional augmentation-rich recipe.
- Both detectors now have five completed clean-analysis seeds. Five attempts
  remain a coarse estimate of seed variation, and YOLO11s seed 271 demonstrates
  substantial fixed-threshold score-scale instability despite stable AP.
  Robustness and explainability remain scoped to the primary seed-17
  checkpoints and do not characterize across-seed variability.
- Batch 10's n=3 and Batch 35's n=5 grids remain exploratory historical
  evidence, and five runs still give coarse mean ± sample-SD bands. Batch 42
  evaluates every unique score retained in the frozen bundles, not only the
  0.01--0.99 grid. The observed exact-score frontier demonstrates score-scale/
  selectivity rather than probabilistic calibration and remains bounded by the
  approved 0.00001 candidate floor. YOLO11s seed 137 alone is floor-limited at
  2 FP/image; that aggregate sensitivity is a lower-bound observation. Its
  conservative maximum remains below Faster R-CNN's observed sensitivity, so
  the qualitative detector ordering cannot reverse.
- Batch 33's v2 audit measures detection-confidence calibration descriptively
  on the same held-out bundles; it does not fit a validation calibration map,
  reserve a second independent holdout, or add D-ECE inference. D-ECE is
  conditional on emitted predictions at the 0.001 floor. Only 68--354 of 3,125
  primary cells are occupied and 15--169 meet minimum support; supported
  detection fractions span 0.543--0.983. The predeclared bin/minimum and floor
  grid shows material absolute-estimate and population sensitivity, so the
  original estimate must be read with support. Missed targets have no emitted
  confidence and are outside the population. D-ECE does not measure exam-level
  or clinical-risk calibration and is programmatically separate from any valid
  probability-based DCA. Its n=5 prediction population must not be merged with
  the n=3 validation-threshold selection or mistaken for the separately labeled
  n=5 test-side operating-regime sensitivity.
- Batch 14's detector-specific thresholds maximize equal-weight validation F1;
  that transparent rule does not encode an elicited clinical-harm function and
  uses only three validation seeds. Batch 35 applies 0.69/0.05 unchanged to all
  five test runs, so its fixed-threshold and recall-Pareto values are five-run
  test sensitivities but not five-run threshold selections. Batch 43 separately
  repeats the identical rule on all five validation runs and selects 0.70/0.01,
  but this is post-hoc validation sensitivity and does not replace the frozen
  n=3 provenance. Five validation runs still provide only coarse threshold
  stability evidence. The historical YOLO threshold produces no seed-271
  detection. The strict Pareto labels
  describe complete ordering of the visibly labeled n=3 or n=5 run clouds, not
  statistical uncertainty, deployment utility, or unmeasured hardware.
- Batch 29's corrected F-beta alternatives use only the three frozen
  validation seeds. Beta 1/3/5/10 are recall-preference parameters and beta
  squared is a harmonic-mean weight, not a measured clinical-harm ratio. The
  2,000-draw intervals resample 321 patient groups and three seeds but are
  pointwise over the 99-threshold grid, not simultaneous guarantees for the
  selected maxima. Near-optimal plateaus and draw-specific argmax frequencies
  are descriptive. The separate linear loss assumes r 1/9/25/100 and treats
  target misses/false-positive boxes as exchangeable within type; it omits
  patient outcomes and deployment utility. YOLO beta 3--10 and loss r 9--100
  reach the 0.01 lower boundary. D-006 keeps every sensitivity row separate
  from the primary thresholds, and none is applied to test or propagated into
  FROC/Pareto.
- Batch 30 establishes that the historical Batch 20 raw-score calculation is
  not conventional DCA: maximum detector confidence defined action and the
  same raw cutoff supplied the `tau/(1-tau)` weight without a validation-fitted
  exam-outcome probability mapping. It is excluded from the main Results and
  retained only as a non-standard supplementary sensitivity artifact. Four
  validation bundles were absent when that decision was made; Batch 43 later
  generated them solely for post-hoc threshold sensitivity. Their availability
  does not retroactively specify an outcome-probability calibration protocol:
  no calibrator was chosen or fitted and no DCA was rerun. The historical
  750-image subset is enriched and internal (169
  positive, 581 negative, 323 patient groups; 22.533% image-level prevalence),
  not a deployment-prevalence sample. Neither the archived nor relabeled curve
  supplies standard net-benefit, clinical-utility, beneficial-range, threshold-
  recommendation, or deployment-readiness evidence.
- Phase 8's primary clean hierarchical bootstrap resamples five trained runs
  independently within detector for AP/precision/recall/F1; conditional
  IoU/Dice use five defined Faster R-CNN runs and four defined YOLO11s runs.
  Five/four runs still provide coarse, empirical training-procedure uncertainty,
  not a parametric model of all retraining conditions. Corruption inference
  remains conditional on the primary seed-17 checkpoints.
- Batch 13 corrects within-patient dependence by resampling and exchanging the
  323 clean / 183 robustness-sample NIH patient groups, with every observed
  exam from one patient moving together. Permutation tests still condition on
  the observed checkpoints, while clean intervals resample only five or four
  eligible detector-specific runs; pointwise intervals are not simultaneous.
  Holm controls
  the declared p-value families but does not make the repeated digital
  corruption conditions independent deployment cohorts or guarantee
  population transportability.
- YOLO required bfloat16 forward/backward with float32 assignment/loss, native
  BatchNorm updates, a lower LR, and a one-epoch warmup for numerical stability;
  Faster R-CNN used float16, frozen normalization, LR 0.005, and a plateau
  scheduler. These disclosed optimization/precision asymmetries limit a purely
  architecture-only interpretation of the comparison.
- Phase 7 uses coarse 40 by 40 Grad-CAM maps and remains one-primary-seed
  evidence. Missed opacity annotations require annotation-guided proxy
  candidates; Grad-CAM does not explain proposal generation, NMS, box
  regression, or causality. Box metrics exclude 232 images without target
  boxes, one Faster R-CNN map has zero Grad-CAM energy, and CUDA ROI Align
  backward is not bitwise deterministic under the pinned Torchvision build.
  Batch 31's fixed 50-image/41-patient audit uses one deterministic
  randomization draw per cumulative stage, a severe pixel-permutation control,
  and a fixed-reference/pre-activation target distinct from Phase 7's box
  estimand. Seven Faster input-control pairs and four YOLO full-randomization
  pairs are degenerate and excluded consistently. Intermediate cascade stages
  are non-monotonic. Low full-model similarities support parameter sensitivity
  only; the input-pixel control cannot establish learned data-label dependence,
  and neither control proves anatomical correctness, causal faithfulness, or
  clinical reasoning. The canonical Adebayo randomized-training test was not
  performed.
- `report/paper_draft.md` now carries the all-attempt n=5/n=4 policy,
  seed-271 confidence-score instability, frozen n=3 threshold/PR/Pareto
  scope, D-006's primary-versus-preference/loss-sensitivity relationship, and
  D-007's exclusion of non-standard raw-score curves throughout the abstract,
  Methods, Results, Discussion, and Limitations. Batch 42 adds the approved
  0.00001 observed exact-score FROC result, residual seed-137 lower-bound, and
  conservative no-reversal proof. Batch 43 adds the separately labeled
  five-run post-hoc validation selection and its weakened/reversed operating-
  point comparisons. Batch 44 adds limited aggregate DICOM-header age, sex,
  and projection characteristics with explicit unit and fairness boundaries,
  but the manuscript still awaits the user's
  substantive review and author completion of the declaration placeholders; it
  is not yet a submission-ready journal artifact.
- The Batch 34 crosswalk audits the current paper but is not an official CLAIM
  submission checklist or certification. CLAIM 2024 is directly applicable;
  final STARD-AI 2025 and TRIPOD+AI 2024 are used only by analogy because the
  study is neither a participant-level diagnostic-accuracy analysis nor an
  individualized clinical prediction-model study. The Batch 37 structured
  abstract now covers design, partitions, estimand, outcomes, and implications
  but remains incomplete against CLAIM item 2 because release-ready software,
  data, and model availability is unresolved. Source accrual dates, broader
  clinical/demographic variables, demographic subgroup/fairness evaluation, a
  participant-flow diagram, external testing, and human-author declarations
  for ethics/consent, funding,
  competing interests, contributions, availability, and patient/public
  involvement remain unresolved. `docs/AUTHOR_DECLARATIONS_TODO.md` prevents
  those facts from being invented.
