# HANDOFF.md — Session Log

> Read the top entry at the **start** of every session, before doing anything else — that's the state you're picking up from. Write a new entry at the **end** of every session, or right before stopping for the user's review — not at the start, since nothing's happened yet at that point. Newest entry goes on top, below the template.

---

## Template for new entries

```
## Session N — <date> — Batch <X>

**What I did:**
-
-

**What's still incomplete / next step:**
-

**Needs the user's review before proceeding:**
-

**Files touched:**
-
```

---

## Session 94 - 2026-09-07 - Commit completed Batch 45 work

**What I did:**
- At the user's explicit request, prepared one scoped commit of the completed
  Batch 45 implementation, tests, publication timing artifacts, documentation,
  manuscript/provenance updates, and session state on top of
  `21defcfe6a2c25cb7cfbaada0fb255755b9c111d`.
- Used the user's exact commit message:
  `Standardized End-to-End Inference Timing`.
- Included the 29 Batch 45 paths enumerated by Session 93 and narrow
  `.gitattributes` rules preserving exact generated timing CSV/JSON/environment
  bytes. The repository's default LF normalization would otherwise change
  these new artifacts in the commit and invalidate their recorded SHA256s.
  Unrelated
  untracked specifications, review notes, orchestration records, historical
  aborted/smoke logs, and the ignored development timing trial stay outside
  the commit. No push was requested or performed.
- Rechecked the staged inventory and Git whitespace. The final commit hash is
  reported in the session response because a commit cannot contain its own
  resulting hash. Batch 45 verification remains as recorded in Session 93;
  no code, timing values, or scientific conclusions changed in this operation.

**What's still incomplete / next step:**
- No push. Batch 46 and later work remain outside this session.
- A broader read-only index check exposed a pre-existing serialization issue:
  the scientific manifest's `train.csv` hash matches its working-tree bytes
  but differs from the Git-stored LF blob. Both that binding and blob predate
  this commit and were left unchanged. Audit this separately; all new Batch 45
  inputs/outputs and their index hashes are checked within this commit's scope.

**Needs the user's review before proceeding:**
- None for this explicitly requested Git operation; Session 93 records the
  remaining scientific interpretation and scope limitations.

**Files touched:**
- `CODEX.md`, `HANDOFF.md`, and `.gitattributes`; the other committed Batch 45
  files are those listed in Session 93.

## Session 93 - 2026-09-07 - Batch 45 standardized end-to-end inference timing

**What I did:**
- Passed the prerequisite gate by auditing both historical profiling functions,
  all ten compute CSVs/training summaries, immutable checkpoint hashes and the
  intended reporting hardware. The real ASUS ROG Strix G16/i7-13650HX/16 GB/
  RTX 4060 Laptop GPU environment is available. Publication timing used its
  existing `C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe`; the repository
  `.venv` has CPU Torch and was used for tests/offline verification only.
- Documented the exact historical asymmetric boundary and algebraically
  reconstructed its available elapsed totals from mean latency and image
  count. Preserved 46 historical files by SHA256, including all compute tables,
  summary/config sources, split/annotations, timing code, comparison tables,
  and old Pareto/raincloud figures. No historical value was overwritten.
- Implemented strict versioned `decoded-host-to-source-detections-v1` with
  config-driven 100-image deterministic SHA256-ranked test subset, batch 1,
  10 warm-up images per detector/repetition, and three full repetitions with
  alternating model order. Both start from the same decoded uint8 RGB host
  source and include native resize/letterbox, tensor conversion, H2D, forward,
  postprocessing/NMS, source-box restoration and CPU detection extraction.
  Explicit CUDA synchronization brackets every timer; disk I/O/decoding,
  model setup/fusion, references and result checks/writes are outside timing.
- Verified actual float16/bfloat16 convolution AMP dtypes. Ordinary reference
  inference independently reads source files under identical explicit AMP and
  postprocessing. All 600 timed-image comparisons pass with exactly equal
  detection counts, category labels, box coordinates and scores; each repeat
  checks 2,471 Faster R-CNN and 165 YOLO11s detections. Historical FP32 YOLO
  evaluation bundles are unchanged and are not claimed as the AMP reference.
- Accepted totals over 300 calls per detector are Faster R-CNN 14.3465654 s and
  YOLO11s 5.2123790 s; FPS 20.91093 / 57.55529; median latency 47.20395 /
  17.28700 ms; image-latency IQR 1.22010 / 1.19335 ms. Recorded raw per-image
  times/agreement, each repetition, aggregate table, figure, source/image/
  checkpoint/environment hashes, hardware/CUDA/framework versions, and GPU
  telemetry. Three repetitions are technical timing repeats, not biological,
  patient or training replicates, and supply no inferential confidence interval.
- Excluded a first development trial because deprecated `half=False` printed
  warnings inside every YOLO timed call. Preserved that diagnostic in ignored
  `tmp/inference_timing_trial_deprecated_half/`, corrected to `quantize=None`,
  and reran the complete protocol; the accepted artifact uses only that clean
  run, without selecting repetitions based on speed.
- Updated the primary manuscript timing table/Figure 5, abstract, methods,
  discussion, limitations, D-016, compute/baseline/Pareto documentation,
  supplementary/reproducibility/checklist/traceability, README exact command,
  claim sources and scientific manifest. Parameter counts and training
  profiles remain; GFLOPs are explicitly incomplete profiler-registered
  operations and the approximate 21-fold ratio is no longer a headline fact.
- Verification passes: 24 timing tests and full 357-test suite with one expected
  metadata-only-environment skip; repository-wide Ruff format/lint; 72
  scientific artifacts, 346 present inputs and 201 referenced results; 64
  numerical paper claims and all semantic guards; historical SHA preservation;
  raw timing/statistics verifier; visually inspected figure; identical hashes
  after offline table/figure regeneration; and Git whitespace. No training,
  checkpoint/prediction replacement, staging, commit or push occurred.

**What's still incomplete / next step:**
- Batch 45 is complete and stops here. Review the new primary matched timing
  and the explicit same-AMP ordinary-reference scope. No local user timing
  action remains because the intended reporting machine was available and
  measured successfully in this session.
- Run only the next separately requested batch after review. Five-run matched
  timing, a standardized five-run Pareto frontier, external work, human author
  declarations, checkpoint release, and later FROC work remain out of scope.

**Needs the user's review before proceeding:**
- Review D-016 and the manuscript's replacement of the asymmetric main runtime
  comparison by one-checkpoint matched timing with technical repetitions.
  YOLO11s remains faster (2.75-fold aggregate FPS here); incomplete operation
  counts are supplementary. Different AMP dtypes and native API costs remain
  explicit measurement limits.

**Files touched:**
- `configs/inference_timing_v1.yaml`, `src/benchmark_inference.py`,
  `tests/test_inference_timing.py`, and `docs/COMPUTE_TIMING.md`
- `results/tables/inference_timing_v1.csv`,
  `results/tables/inference_timing_v1_images.csv`,
  `results/tables/inference_timing_v1_repetitions.csv`,
  `results/figures/inference_timing_v1.png`, and
  `results/logs/phase45_inference_timing_v1/`
- `README.md`, `docs/DECISION_LOG.md`, `docs/FASTER_RCNN_BASELINE.md`,
  `docs/YOLO_BASELINE.md`, `docs/QUANTITATIVE_COMPARISON.md`,
  `docs/PARETO_ANALYSIS.md`, `docs/HYPOTHESIS_TRACEABILITY.md`,
  `docs/LIMITATIONS.md`, `docs/REPORTING_CHECKLIST.md`,
  `docs/REPRODUCIBILITY.md`, and `docs/SUPPLEMENTARY.md`
- `report/paper_draft.md`, `report/paper_claim_sources.yaml`,
  `scripts/build_scientific_artifact_manifest.py`,
  `results/scientific_artifact_manifest.json`, `CODEX.md`, and `HANDOFF.md`

## Session 92 - 2026-09-07 - Commit completed Batch 43--44 work

**What I did:**
- At the user's explicit request, prepared one consolidated commit containing
  the completed Batch 43 threshold-sensitivity work and Batch 44 aggregate
  cohort-reporting work on top of Batch 42 commit
  `8f5c3508ccba3502287709a0848af14040349422`.
- Scoped the commit to the files enumerated by Sessions 90 and 91. Unrelated
  untracked specifications, review notes, orchestration records, and historical
  aborted/smoke run logs remain untracked and outside the commit.
- Rechecked Git whitespace and the staged inventory before commit. The exact
  resulting commit identifier is reported in the session response because a
  commit cannot contain its own final hash.

**What's still incomplete / next step:**
- No push was requested or performed. Batch 45 and all other later work remain
  outside this session.

**Needs the user's review before proceeding:**
- None for the Git operation; the scientific review points recorded in
  Sessions 90 and 91 remain applicable.

**Files touched:**
- `CODEX.md` and `HANDOFF.md`; all other committed files are the completed
  Batch 43--44 files listed in Sessions 90 and 91.

## Session 91 - 2026-09-07 - Batch 44 RSNA cohort characteristics

**What I did:**
- Passed the prerequisite gate without changing membership: rehashed the
  immutable train/validation/test split manifests and committed audit, verified
  exactly 3,500/750/750 studies, 1,492/321/323 disjoint NIH patient groups,
  1,136 total opacity-positive studies, 1,812 boxes, and exact agreement of all
  5,000 study-to-patient/stratum/label/box mappings with the official RSNA/NIH
  CSVs. The configured raw root contains all 26,684 Stage 2 DICOM files.
- Added `configs/cohort_characteristics.yaml`, a strict header-only aggregate
  extractor, and eight regression tests. The workflow supports the optional
  `RSNA_DICOM_ROOT` override, validates hashes/counts/mapping before reading
  metadata, uses `stop_before_pixels=True`, and writes no row-level identifiers.
- Read `PatientAge`, `PatientSex`, and `ViewPosition` for the selected cohort.
  All 5,000 age tags are numeric `AS` strings without a D/W/M/Y unit. Under the
  explicit nominal-years descriptive policy, 4,999 ages are usable after one
  value above 120 is excluded: total median 49 years (IQR 36--60). Study-level
  sex is 2,362 female and 2,638 male; projection is 2,363 AP and 2,637 PA; sex
  and projection have no missing/other values, and no patient group has
  conflicting nonmissing sex tags.
- Generated the aggregate cohort CSV and provenance summary, then updated the
  canonical manuscript, datasheet, limitations, supplementary material,
  reproducibility/docs commands, reporting checklist, D-015 decision, claim
  bindings, and scientific-artifact manifest. Wording is explicitly
  descriptive and makes no subgroup-performance or fairness inference.
- Final verification passes: 333 tests plus one expected environment-conditional
  skip; repository-wide Ruff format/lint; scientific verifier with 67 artifacts,
  224 present inputs, and 197 referenced result files; paper verifier with 49
  numerical claims and all semantic guards; deterministic aggregate generation;
  identifier-leak scan; and Git whitespace. No training or model inference ran,
  and nothing was staged, committed, or pushed.

**What's still incomplete / next step:**
- Batch 44 is complete and stops here. Broader clinical/demographic variables,
  subgroup-performance/fairness analyses, external testing, and later batches
  remain outside this batch.

**Needs the user's review before proceeding:**
- Review the disclosed interpretation of unitless numeric `PatientAge` strings
  as nominal years for descriptive reporting only, including exclusion of the
  single value above the configured 0--120-year range.

**Files touched:**
- `configs/cohort_characteristics.yaml`, `src/data/cohort_characteristics.py`,
  and `tests/test_cohort_characteristics.py`
- `results/tables/rsna_cohort_characteristics.csv` and
  `results/logs/phase44_cohort_characteristics/summary.json`
- `README.md`, `data/README.md`, `docs/DATASHEET.md`, `docs/DECISION_LOG.md`,
  `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`,
  `docs/REPORTING_CHECKLIST.md`, and `docs/SUPPLEMENTARY.md`
- `report/paper_draft.md`, `report/paper_claim_sources.yaml`,
  `scripts/build_scientific_artifact_manifest.py`,
  `results/scientific_artifact_manifest.json`, `CODEX.md`, and `HANDOFF.md`

## Session 90 - 2026-09-07 - Batch 43 five-run validation threshold-selection sensitivity

**What I did:**
- Passed the prerequisite gate without training: independently reconstructed
  the historical validation-only selector (seeds 17/42/137, thresholds
  0.01--0.99, maximum equal-run arithmetic mean F1, sample SD, higher-threshold
  exact-tie rule) and verified Faster R-CNN 0.69 / YOLO11s 0.05; verified the
  complete Batch 35 five-run test-side operating analysis, all ten checkpoint
  hashes (962,924,817 bytes), and Batch 41's four missing validation bundles.
  The original 750-image/277-box validation source and every referenced image
  were present, so the batch was allowed to continue.
- Added a strict versioned config, analyzer/collector, and regression tests.
  The guards prohibit test-set threshold selection, omission of seed 271, and
  any output overlap or hash drift in the historical Phase 14 artifacts. The
  workflow reuses the original Phase 5 adapters/evaluator and performs only
  validation inference for detector/seed pairs 271 and 314 when bundles are
  absent; no optimizer, resume, or training path exists.
- Generated and hash-bound four validation bundles from the exact frozen best
  checkpoints under the original validation split, preprocessing, candidate
  floor 0.001, NMS 0.50, maximum 100 detections, and coordinate logic. Faster
  seed 271 and both YOLO runs reproduce their training-time score-0.25 counts
  exactly. Faster seed 314 is deterministic across repeated inference and has
  identical TP/FN but one fewer FP/prediction (407/559 rather than 408/560);
  the bounded discrepancy and configured count audit are explicit in the
  manifest. No test label was used to accept or modify a bundle.
- Reproduced the historical sweep/tables numerically before selecting over all
  five validation runs. The post-hoc validation sensitivity selects Faster
  R-CNN 0.70 and YOLO11s 0.01. Applying these unchanged to all five test bundles
  gives Faster versus YOLO mean precision 0.3686 vs 0.2631, recall 0.3388 vs
  0.2925, F1 0.3458 vs 0.2657, FP/image 0.2205 vs 0.3104, and detections/run
  256.2 vs 311.2. The historical 0.69/0.05 five-test-run results were also
  reproduced exactly.
- Classified all 11 conclusions. Threshold separation strengthens; Faster's
  mean precision/recall/F1 leads remain but weaken; FP/image and detection-count
  orderings reverse. For run-level sample SD, precision ordering reverses,
  recall/F1 stability leads weaken, and FP/image/detection-count stability
  leads strengthen. The 0.70/0.01 result is labeled only **post-hoc validation
  sensitivity**, never prospectively frozen, and D-014 preserves historical
  0.69/0.05 precedence.
- Updated the threshold analysis, decision log, limitations, reproducibility,
  README commands, supplementary/hypothesis traceability, canonical manuscript,
  claim bindings, reporting-checklist manuscript hash, scientific manifest,
  CODEX, and this handoff. `report/report.md`, the alternate manuscript/PDF,
  and every historical n=3 artifact remain unchanged.
- Final verification passes: 325 tests plus one expected metadata-only skip;
  repository-wide Ruff format/lint; scientific verifier with 65 artifacts, 216
  present inputs, and 196 references; paper verifier with 43 numerical claims
  and all semantic guards; final CUDA preflight; historical hash protection;
  and Git whitespace. Nothing was staged, committed, pushed, or retrained.

**What's still incomplete / next step:**
- Batch 43 is complete and stops here. Do not promote 0.70/0.01 to primary or
  prospectively frozen thresholds. Run only the next separately requested
  batch after review.
- External/VinDr work, human declarations, checkpoint publication/release, and
  any still-lower FROC inference remain outside this batch.

**Needs the user's review before proceeding:**
- Review the scientific interpretation: five-run validation coverage changes
  YOLO11s to the 0.01 grid boundary and weakens Faster R-CNN's precision,
  recall, and F1 margins without reversing them, while FP/image ordering does
  reverse. Also review the explicitly disclosed one-FP Faster seed-314
  re-inference discrepancy.

**Files touched:**
- `configs/threshold_selection_n5_validation_sensitivity.yaml`,
  `src/analyze_validation_threshold_sensitivity.py`, and
  `tests/test_validation_threshold_sensitivity.py`
- `results/logs/phase43_threshold_selection_n5_validation_sensitivity/` and
  the five Batch 43 CSV tables under `results/tables/`
- `README.md`, `docs/THRESHOLD_ANALYSIS.md`, `docs/DECISION_LOG.md`,
  `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`,
  `docs/HYPOTHESIS_TRACEABILITY.md`, `docs/SUPPLEMENTARY.md`, and
  `docs/REPORTING_CHECKLIST.md`
- `report/paper_draft.md`, `report/paper_claim_sources.yaml`,
  `scripts/build_scientific_artifact_manifest.py`,
  `results/scientific_artifact_manifest.json`, `CODEX.md`, and `HANDOFF.md`

---

## Session 89 - 2026-09-06 - Batch 42 approved 0.00001 FROC continuation

**What I did:**
- Resumed from the completed v3 state after the user explicitly approved the
  detector-neutral 0.00001 inference-only sensitivity. Added final v4 configs
  and a prior-config contract that proves the only inference change from v3
  was the candidate floor, 0.0001 to 0.00001. Preflight reverified all ten
  checkpoint hashes (962,924,817 bytes), five seeds including 271, the same
  750-image/268-box test set, CUDA, preprocessing/adapters, matching, NMS,
  maximum detections, and nonoverlapping output paths.
- Ran the ten frozen checkpoints over 7,500 model-image evaluations with no
  training, weight change, model selection, or test-label threshold tuning.
  Wrote only new v4 bundles, inventory, boundaries, inference contract,
  progress, environment, and provenance. All historical, v2, and v3 artifacts
  remain intact; their recorded hashes were rechecked with zero failures.
- Generated the v4 exact-score frontier from every unique emitted confidence
  plus configured upper/lower endpoints. It contains 64,739 per-run rows,
  30,025 aggregate-curve rows, 50 per-run budget points, ten aggregate points,
  historical and v3 comparisons, a figure, and a summary. All ten monotonicity
  audits pass and all 50 selections match the canonical evaluator.
- At budgets 0.125/0.25/0.5/1/2, Faster R-CNN sensitivity is
  0.2776/0.3664/0.4858/0.6000/0.6978 and YOLO11s is
  0.1799/0.2664/0.3821/0.5075/0.6090. Relative to v3, the gap is unchanged at
  the first three budgets and weakened at 1 and 2; it never reverses. Relative
  to the historical grid, it strengthens at 0.125 and weakens at the other
  four budgets.
- Nine runs reach at least 2 FP/image. YOLO11s seed 137 ends at sensitivity
  0.5634 and 1.9907 FP/image, so only its 2-FP/image contribution remains
  floor-limited. The conservative mathematical-maximum bound replaces that
  run's missing sensitivity with 1.0, yielding a YOLO11s aggregate upper bound
  of 0.6963 versus Faster R-CNN's observed 0.6978. The ordering at 1 or 2
  FP/image therefore cannot theoretically reverse.
- Added D-013 and synchronized the canonical/alternate Markdown manuscripts,
  FROC/threshold/limitations/reproducibility/hypothesis/supporting docs,
  README, claim bindings, scientific manifest, CODEX, and this handoff. The
  noncanonical PDF and historical report remain untouched. No 0.000001 run was
  performed or authorized.
- Verification passes: 16 initial focused tests, 51 relevant tests, 320 full-
  suite tests plus one declared metadata-only skip, 26 final focused/verifier
  tests, repository-wide Ruff lint/format, scientific verifier (58 artifacts,
  170 inputs, 187 references), paper verifier (37 claims plus semantic guards),
  v2/v3 preservation audit, v4 provenance preflights, figure review, and Git
  whitespace. Nothing was staged, committed, pushed, or retrained.

**What's still incomplete / next step:**
- Batch 42 stops here because one run remains 0.0093 FP/image short of the
  2-FP/image budget. A still-lower inference pass could complete that numeric
  point, but cannot change the qualitative detector ordering under the
  conservative bound. Do not run 0.000001 automatically.
- Batch 43, external/VinDr work, human declarations, checkpoint publication,
  and release actions remain out of scope.

**Needs the user's review before proceeding:**
- Review the residual-bound conclusion. Another lower-floor run would have
  limited scientific value: it could refine YOLO11s sensitivity at 2 FP/image,
  but it cannot reverse Faster R-CNN's ordering even at the mathematical
  maximum.

**Files touched:**
- `configs/evaluation_froc_lower_floor_v4.yaml` and
  `configs/froc_exact_score_v4.yaml`
- `src/collect_froc_lower_floor_predictions.py`,
  `src/analyze_exact_score_froc.py`, `tests/test_froc_lower_floor_inference.py`,
  and `tests/test_exact_score_froc.py`
- `results/logs/phase42_froc_lower_floor_v4/`,
  `results/logs/phase42_froc_exact_score_v4/`, v4 FROC tables/comparisons/bound,
  and `results/figures/froc_exact_score_v4.png`
- `README.md`, `docs/FROC_ANALYSIS.md`, `docs/THRESHOLD_ANALYSIS.md`,
  `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`, `docs/HYPOTHESES.md`,
  `docs/HYPOTHESIS_TRACEABILITY.md`, `docs/LITERATURE_REVIEW.md`,
  `docs/SUPPLEMENTARY.md`, `docs/REPORTING_CHECKLIST.md`, and
  `docs/DECISION_LOG.md`
- `report/paper_draft.md`, `report/paper_claim_sources.yaml`, and only stale
  FROC sentences in `report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md`
- `scripts/build_scientific_artifact_manifest.py`,
  `results/scientific_artifact_manifest.json`, `CODEX.md`, and `HANDOFF.md`

---

## Session 88 - 2026-09-06 - Batch 42 approved lower-floor FROC sensitivity

**What I did:**
- Resumed the completed frozen-bundle Batch 42 state after the user explicitly
  approved its proposed 0.0001 inference-only sensitivity. Preserved the
  original Phase 5 bundles, historical n=3/n=5 FROC artifacts, 0.001
  exact-score v2 artifacts, every checkpoint, and all unrelated user files.
- Added a strict, restart-safe collector and versioned v3 inference config. Its
  preflight verified the same ten checkpoint hashes (962,924,817 bytes), Phase
  5 config/summary, release manifest, 750-image/268-box annotations, five seeds
  including 271, CUDA runtime, and nonoverlapping outputs. The only changed
  scientific parameter was the candidate floor, 0.001 to 0.0001; class,
  preprocessing, score-0.25 operating point, IoU-0.50 matching, class-aware
  NMS IoU 0.50, and maximum 100 detections/image stayed fixed.
- Ran all ten frozen checkpoints on the same 750 test images per model (7,500
  model-image evaluations) without training or weight updates. Wrote ten
  deterministic, hash-bound v3 bundles plus inventory, score-boundary,
  inference-contract, progress, environment, and completion artifacts. The
  collector reuses an existing bundle only after full identity/content checks.
- Added `configs/froc_exact_score_v3.yaml` and extended the exact-score analyzer
  without changing v2 behavior. The v3 run produced 35,894 per-run exact rows,
  16,211 aggregate-curve rows, 50 per-run budget selections, ten aggregate
  operating points, a historical comparison, figure, and provenance. All ten
  runs are monotone in sensitivity and FP/image as thresholds relax; all 50
  selections match the canonical matcher.
- Current sensitivities at budgets 0.125/0.25/0.5/1/2 are Faster R-CNN
  0.2776/0.3664/0.4858/0.6000/0.6978 and YOLO11s
  0.1799/0.2664/0.3821/0.4985/0.5881. Versus the 0.001 exact-score result,
  Faster is unchanged and YOLO changes by 0/0/+0.0052/+0.0246/+0.1082. Faster
  remains higher at every budget; the conclusion does not reverse. Relative to
  the historical grid, the gap strengthens at 0.125 and weakens at the other
  four budgets.
- Confirmed that 0.0001 is still a real boundary only for YOLO11s seed 137: its
  endpoint is sensitivity 0.4590 at 0.7427 FP/image, below the 1- and
  2-FP/image budgets. Those two aggregates remain lower-bound observations.
  Prepared the detector-neutral 0.00001 v4 proposal for the same ten
  checkpoints/images; its read-only preflight passes, but no v4 inference ran.
- Added D-012 and aligned FROC, threshold, limitation, reproducibility,
  hypothesis, supplementary, README, both Markdown manuscripts, claim-source,
  checklist, scientific-manifest, CODEX, and handoff records. The current
  manuscript uses v3 values/figure; the noncanonical PDF and historical report
  remain untouched. The scientific verifier now covers 56 artifacts, 163
  present inputs, and 184 references; all 35 paper claims/guards pass.
- Verification passes: 12 focused tests, 45 relevant tests, 316 full-suite
  tests plus one declared metadata-only skip, Ruff format/lint, scientific
  artifact verifier, paper verifier, visual figure review, and Git whitespace.
  One first relevant-test invocation hit the known repository `.pytest-tmp`
  Windows ACL and produced fixture setup errors after 35 test passes; the same
  suite passed 45/45 with a fresh system-temp root. Nothing was staged,
  committed, pushed, or retrained.

**What's still incomplete / next step:**
- Batch 42 must stop at the residual approval gate. If the user approves, run
  `configs/evaluation_froc_lower_floor_v4_proposal.yaml` into its separate v4
  namespace, then create v4 exact-score outputs and compare them with both v3
  and the historical grid. Do not overwrite any v2/v3/Phase 5 artifact.
- Batch 43, external/VinDr work, human declarations, checkpoint publication,
  and release actions remain out of scope.

**Needs the user's review before proceeding:**
- Approve or reject a further detector-neutral 0.00001 candidate floor for
  inference-only sensitivity over the same Faster R-CNN and YOLO11s seeds
  17/42/137/271/314 and the same 750 images per model. It is needed because
  YOLO11s seed 137 still cannot reach the prespecified 1- and 2-FP/image budgets
  at 0.0001. Expected outputs are ten v4 bundles/provenance records followed by
  v4 exact-score per-run/aggregate/budget/comparison tables, figure, and
  summary.

**Files touched:**
- `configs/evaluation_froc_lower_floor_v3.yaml`,
  `configs/evaluation_froc_lower_floor_v4_proposal.yaml`, and
  `configs/froc_exact_score_v3.yaml`
- `src/collect_froc_lower_floor_predictions.py`,
  `src/analyze_exact_score_froc.py`, `tests/test_froc_lower_floor_inference.py`,
  and `tests/test_exact_score_froc.py`
- `results/logs/phase42_froc_lower_floor_v3/`,
  `results/logs/phase42_froc_exact_score_v3/`,
  `results/tables/froc_*v3.csv`, and `results/figures/froc_exact_score_v3.png`
- Refreshed v2 exact-score outputs only to bind the finalized analyzer source;
  numerical CSV/figure content remains deterministic.
- `README.md`, `docs/FROC_ANALYSIS.md`, `docs/THRESHOLD_ANALYSIS.md`,
  `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`, `docs/HYPOTHESES.md`,
  `docs/HYPOTHESIS_TRACEABILITY.md`, `docs/LITERATURE_REVIEW.md`,
  `docs/SUPPLEMENTARY.md`, `docs/REPORTING_CHECKLIST.md`, and
  `docs/DECISION_LOG.md`
- `report/paper_draft.md`, `report/paper_claim_sources.yaml`, and only stale
  FROC sentences in `report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md`
- `scripts/build_scientific_artifact_manifest.py`,
  `results/scientific_artifact_manifest.json`, `CODEX.md`, and `HANDOFF.md`

---

## Session 87 - 2026-09-06 - Batch 42 exact-score FROC lower-bound repair

**What I did:**
- Resumed on `main` at the unchanged base
  `42826df37468a1c0b28439a7120cc5f3d83f9168`, preserving the pre-existing
  Batch 41 decision/audit edits and every unrelated untracked local file. The
  prerequisite gate passed before implementation: the scientific verifier,
  paper verifier, and focused historical FROC tests passed; all Batch 35 n=5
  FROC artifacts retained their hashes.
- Reproduced the archive-safe historical n=3 FROC operating-point CSV byte for
  byte (SHA-256
  `c5b1fad22afbaedac90060056066beca62be2464f79cfb716a0db998836527c8`).
  Recorded the 99-point 0.01--0.99 grid, 0.001 bundle floor, IoU 0.50 matcher,
  NMS IoU 0.50, 100-detection cap, 750 images/268 boxes, all five seeds
  including 271, and budgets 0.125/0.25/0.5/1/2 FP/image.
- Added strict versioned `configs/froc_exact_score_v2.yaml` and
  `src/analyze_exact_score_froc.py`. The offline path hash-checks all ten frozen
  bundles and upstream artifacts, evaluates every unique emitted score in
  deterministic descending order, includes an empty upper sentinel and 0.001
  lower endpoint, aggregates five runs equally, writes versioned per-run/
  aggregate/budget/comparison artifacts, audits monotonicity, and cross-checks
  all 50 selected run-budget points through the canonical matcher. It cannot
  train, load checkpoints, or perform inference.
- The observed exact-score sensitivities for Faster R-CNN are
  0.2776/0.3664/0.4858/0.6000/0.6978 and for YOLO11s are
  0.1799/0.2664/0.3769/0.4739/0.4799 at the five ascending budgets. Faster
  R-CNN remains higher at all five. Relative to the historical n=5 grid, the
  gap strengthens at 0.125, weakens at the other four budgets, and never
  reverses.
- Confirmed the 0.001 candidate floor still truncates the observable YOLO11s
  frontier: seed 137 is floor-limited at 0.5; seeds 42/137 at 1; all five seeds
  at 2 FP/image. Affected aggregate sensitivities are lower-bound observations.
  No monotonicity failure occurred in sensitivity or FP/image for any run.
- Prepared but did not execute a detector-neutral lower-floor inference-only
  protocol. The proposed floor is 0.0001, supported by both adapter/config
  validators. Scope is the exact ten Faster R-CNN/YOLO11s checkpoints at seeds
  17/42/137/271/314 and the same 750 test images per model (7,500 model-image
  evaluations). Expected outputs are ten new versioned post-NMS bundles plus
  hash-bound provenance, exact-score curve/budget/comparison tables, figure,
  and summary. Explicit user approval is required.
- Updated the FROC/threshold/limitations/reproducibility/hypothesis/
  supplementary documentation, README commands, D-011, canonical manuscript,
  only stale FROC sentences in the noncanonical Markdown alternate, claim
  bindings/semantic guard, and reporting-checklist manuscript hash. The
  historical report and noncanonical PDF remain unchanged.
- Added four Phase 42 records to the scientific manifest (now 50 artifacts,
  129 present inputs, and 165 references). Exact outputs reproduce with
  identical hashes across consecutive runs; the figure was visually checked
  for legibility. Final verification passes: 10 focused FROC tests, 38 full
  relevant tests, 310 full-suite tests plus one declared conditional skip,
  Ruff format/lint, artifact verifier, 35-claim paper verifier, and Git
  whitespace. One initial relevant-test command used two nonexistent legacy
  test filenames; the corrected command passed. Nothing was staged, committed,
  pushed, retrained, or inferred.

**What's still incomplete / next step:**
- Batch 42 is complete at the frozen-bundle boundary. Do not run lower-floor
  inference unless the user explicitly approves the proposed protocol.
- If approved in a later session, first create a separately versioned
  inference config/output namespace, revalidate all ten checkpoint hashes, and
  generate new bundles without overwriting Phase 5 evidence. Batch 43 and any
  external/VinDr work remain out of scope for this handoff.

**Needs the user's review before proceeding:**
- Approve or reject the proposed detector-neutral candidate floor of 0.0001
  for inference-only sensitivity over all ten exact models and the same 750
  images per model. The reason is that the present 0.001 bundle floor prevents
  observing the YOLO11s frontier at affected higher FP/image budgets.

**Files touched:**
- `configs/froc_exact_score_v2.yaml`
- `src/analyze_exact_score_froc.py`
- `tests/test_exact_score_froc.py`
- `results/tables/froc_*exact_score*_v2.csv`
- `results/figures/froc_exact_score_v2.png`
- `results/logs/phase42_froc_exact_score_v2/summary.json`
- Archive-safe reproduction outputs
  `results/tables/froc_operating_points_n3_archive_reproduction.csv`,
  `results/figures/froc_curves_n3_archive_reproduction.png`, and
  `results/logs/phase14_froc_n3_archive_reproduction/summary.json`
- `results/scientific_artifact_manifest.json`
- `scripts/build_scientific_artifact_manifest.py`
- `README.md`, `docs/FROC_ANALYSIS.md`, `docs/THRESHOLD_ANALYSIS.md`,
  `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`, `docs/HYPOTHESES.md`,
  `docs/HYPOTHESIS_TRACEABILITY.md`, `docs/LITERATURE_REVIEW.md`,
  `docs/SUPPLEMENTARY.md`, `docs/REPORTING_CHECKLIST.md`,
  `docs/DECISION_LOG.md`
- `report/paper_draft.md`, `report/paper_claim_sources.yaml`, and the stale
  FROC sentences in `report/Manuscript_FasterRCNN_vs_YO11s_LungOpacity.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 86 - 2026-09-05 - Batch 41 current-state re-audit and manuscript adjudication

**What I did:**
- Started from `main` at
  `42826df37468a1c0b28439a7120cc5f3d83f9168`, identical to `origin/main`,
  with a clean tracked tree and empty index. The one commit after Session 85's
  SHA adds only the requested article PDF, so it does not conflict with the
  completed V5 scientific state. Preserved every pre-existing untracked local
  state, batch/spec, audit, smoke, aborted-run, and orchestration file.
- Read and compared the canonical paper, new long manuscript, rendered PDF,
  historical report, README hierarchy, V5 final audit, decision/reporting
  records, claim bindings, and current scientific/checkpoint manifests across
  title, abstract, question, contributions, Methods, Results, statistics,
  limitations, declarations, references, overclaiming, and internal process
  terminology.
- Kept `report/paper_draft.md` as the sole canonical editable journal
  manuscript. Recorded the long Markdown as a noncanonical alternate, the PDF
  as its noncanonical rendered derivative, and `report/report.md` as historical.
  The long/PDF line was not promoted because it adds unsupported declarations,
  stronger overclaiming, stale process terms, incomplete limitations, and a
  less controlled reference boundary. No text was copied or edited.
- Rendered and visually inspected all 25 PDF pages. No obvious clipping,
  overlap, broken glyph, or missing-page issue was found. Metadata identifies a
  pdfTeX/LaTeX build but no source/build recipe is committed. Text-shingle
  comparison strongly binds the PDF to the long manuscript (about 79.94%
  containment) rather than the canonical paper (about 11.43%).
- Rehashed all ten local Git-ignored release checkpoints against
  `results/checkpoint_release_manifest.json`, the bound configs, and the Phase 5
  summary: all ten exist and pass, totaling 962,924,817 bytes. The distribution
  state remains binaries-not-published with no public URL.
- Inventoried and rehashed all ten tracked held-out-test prediction bundles:
  five Faster R-CNN and five YOLO11s bundles are present and match Phase 5.
  Confirmed the validation inventory is only six bundles (both detectors,
  seeds 17/42/137). Seeds 271 and 314 are absent for both pipelines, so Batch 43
  needs four validation inference-only bundles.
- Verified the historical n=3 validation-threshold artifact and current n=5
  operating-regime sensitivity. The apparent old-summary differences are
  limited to later-mutated routing/documentation paths; the exact Phase 14
  config bytes remain in the frozen n=3 files, and all scientific outputs pass
  the current verifier. Confirmed current FROC uses observed, non-interpolated
  0.01--0.99 grid points at five FP/image budgets while prediction bundles
  retain scores down to 0.001, leaving exact-score/floor adjudication for
  Batch 42.
- Created `docs/V7_Q2_VINDR_BASELINE_AUDIT.md` with the complete repository,
  manuscript, checkpoint, prediction, n=3/n=5/FROC, frozen-work, inference,
  external-work, and human-blocker record. Added D-010 to
  `docs/DECISION_LOG.md` and updated `CODEX.md`. No numerical result, threshold,
  prediction, checkpoint, configuration, figure, manuscript claim, training,
  or inference changed.
- Verification passes: scientific verifier (46 artifacts, 110 present inputs,
  159 referenced results), paper verifier (35 claims and semantic guards), and
  21 focused verifier/threshold/FROC tests. The first focused pytest attempt
  used a nonexistent filename, and the corrected default-temp attempt met the
  known `.pytest-tmp` Windows ACL; the final explicit external-temp run passed
  21/21. Final post-edit verification and Git integrity checks also pass.

**What's still incomplete / next step:**
- Stop for owner review. If approved, Batch 42 is next: exact-score FROC from
  the frozen test bundles, offline before any possible lower-floor protocol.
- Later V7 work remains exactly classified in the new audit: four validation
  inference bundles, cohort metadata, standardized timing, focused manuscript
  rewrite, frozen VinDr protocol/adapter, one external test experiment, and
  internal/external synthesis. No retraining is authorized.
- Submission remains blocked on the seven human declaration categories and the
  checkpoint publication policy/URL. VinDr work additionally requires the
  human-provided authorized official dataset and protocol review.

**Needs the user's review before proceeding:**
- Confirm the D-010 manuscript hierarchy and the Batch 41 baseline audit.
- Confirm that Batch 42 may begin; do not use the long manuscript or PDF as the
  canonical source in the meantime.

**Files touched:**
- Tracked/unstaged: `docs/DECISION_LOG.md`.
- New/unstaged: `docs/V7_Q2_VINDR_BASELINE_AUDIT.md`.
- Local session state only: `CODEX.md`, `HANDOFF.md` (intentionally untracked).
- Explicitly untouched: all manuscripts/PDFs, README, claim bindings, configs,
  source/tests, `results/`, checkpoints, prediction bundles, Git index/history,
  branches, tags, releases, remotes, and every pre-existing local-only file.

---

## Session 85 - 2026-09-05 - Repository synchronization

**What I did:**
- Read the required top handoff entry and living project state before changing
  repository state.
- Confirmed `main` was tracked-clean at `527e161`, aligned with `origin/main`,
  with only the established local-only untracked notes and historical logs.
- Pulled `origin/main` using `git pull --ff-only origin main`; the branch
  fast-forwarded cleanly to `9d0c4789ac0776cca6aa67280d5fa4d8ce87300b`.
- The fetched delta updated one README line and added
  `report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md`. No merge commit was
  created and no local untracked file was overwritten.
- Verified local `main` and `origin/main` are aligned after the pull.

**What's still incomplete / next step:**
- No project batch was started. The next implementation/review session should
  use the batch explicitly requested by the user.

**Needs the user's review before proceeding:**
- None for the repository synchronization itself.

**Files touched:**
- Pulled tracked files: `README.md` and
  `report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked).

---

## Session 84 - 2026-09-02 - BATCHES_V6 Batch 1 surgical manuscript fixes

**What I did:**
- Began with `git status --short` on tracked-clean `main` at
  `0c94924` (`v2.0.0`, aligned with `origin/main`) and preserved all existing
  untracked local instructions, handoff/audit files, and historical logs.
- Read the local protocol/state/spec, canonical document hierarchy, current
  paper, all 35 claim bindings, consolidated limitations/reporting/declaration
  audits, threshold/FROC/calibration/compute/Pareto documentation, and the
  relevant frozen numerical tables before editing. No provenance or integrity
  mismatch was found.
- Corrected the abstract and manuscript-facing documentation so COCO AP is
  described as independent of the selected single operating threshold but
  conditional on the retained prediction floor and common evaluation/
  post-processing protocol. Removed the literal `threshold-free` and
  `threshold-independent` terms from tracked manuscript-facing Markdown.
- Corrected the abstract from descriptively higher calibration to
  descriptively higher detection-specific calibration error for YOLO11s,
  preserving mean D-ECE 0.0990 versus 0.0320 and all sparse-cell/protocol
  qualifications. Clarified the later retrospective summary in the same way.
- Bounded the abstract, Results, Discussion, Conclusion, README headline,
  Figure 3 caption, hypotheses/traceability, and supporting FROC prose to the
  evaluated 0.01--0.99 score sweep. Preserved the statement that YOLO11s's
  lower-bound plateau is a grid boundary, not a demonstrated global asymptote.
- Qualified the approximate threefold throughput statement in the abstract,
  Discussion, Conclusion, README, quantitative/Pareto documentation as a
  detector-specific measurement on the stated laptop/software stack with
  asymmetric timing regions, not an architecture-family law. FPS, latency,
  parameter, and operation values are unchanged.
- Updated only three affected manuscript regex anchors in
  `report/paper_claim_sources.yaml`; source cells, operations, tolerances, and
  claim IDs are unchanged. Refreshed the reporting-checklist manuscript SHA-256
  to `0b18e05b...` and made the earlier final-submission audit's commit-specific
  tense explicit without rewriting its historical conclusions.
- Left `report/report.md`, `docs/LIMITATIONS.md`,
  `docs/AUTHOR_DECLARATIONS_TODO.md`, all `results/`, frozen n=3 artifacts,
  configs, scientific code, and declarations unchanged. Remaining literal
  `threshold-free` strings occur only in frozen Batch 35 result/provenance
  labels and their hash-bound generator; they were intentionally not mutated.
- Complete lightweight verification passed: `uv lock --check` (97 packages),
  paper verifier (35 claims), scientific verifier (46 artifacts, 110 present
  inputs, 159 referenced results), pytest (300 passed, one expected skip), Ruff
  format (99 files) and lint, deterministic package smoke, checklist/manuscript
  hash equality, seven preserved declaration placeholders, and
  `git diff --check`. The first paper-verifier attempt only hit the known
  managed-Windows uv-cache ACL; the approved retry ran and exposed one wrapped
  text anchor, which was corrected before the clean final pass.

**What's still incomplete / next step:**
- Owner review of the unstaged tracked diff. No stage, commit, push, tag,
  release, training, model inference, artifact regeneration, or result edit was
  performed.
- Submission remains blocked on author-controlled declarations and the final
  public checkpoint/data/code availability statement; no fact was inferred.

**Needs the user's review before proceeding:**
- Confirm the AP, D-ECE, bounded-FROC, and implementation-specific throughput
  phrasing and the deliberately unchanged frozen Batch 35 textual labels.
- Confirm the 14 tracked documentation/claim files are the intended review
  scope. Local `CODEX.md` and `HANDOFF.md` remain untracked session state and
  must not be staged or published.

**Files touched:**
- Tracked: `README.md`, `report/paper_draft.md`,
  `report/paper_claim_sources.yaml`, `docs/THRESHOLD_ANALYSIS.md`,
  `docs/FROC_ANALYSIS.md`, `docs/PARETO_ANALYSIS.md`,
  `docs/QUANTITATIVE_COMPARISON.md`, `docs/LITERATURE_REVIEW.md`,
  `docs/HYPOTHESES.md`, `docs/HYPOTHESIS_TRACEABILITY.md`,
  `docs/DECISION_LOG.md`, `docs/PROJECT_PLAN.md`,
  `docs/REPORTING_CHECKLIST.md`, and `docs/FINAL_SUBMISSION_AUDIT.md`.
- Local session state only: `CODEX.md`, `HANDOFF.md` (intentionally untracked).
- Explicitly untouched: `report/report.md`, `docs/LIMITATIONS.md`,
  `docs/AUTHOR_DECLARATIONS_TODO.md`, all `results/`, source/config/test code,
  model/checkpoint files, Git index/history, tags, releases, branches, and
  remotes.

---

## Session 83 - 2026-09-02 - Batch 40 paper-artifact release preparation

**What I did:**
- Read the complete `AGENTS.md`, `CODEX.md`, `HANDOFF.md`, and project spec,
  then audited the required README, package/lock/CI files, reproducibility,
  decision, limitation, hypothesis, manuscript/report, claim-binding,
  scientific/checkpoint manifest, verifier, Git-ignore/packaging, tag, release,
  and CI sources. Began on `main` at
  `226034c4f8b74b1bc3edfcd3da2dce7bd8c74d98`, exactly aligned with
  `origin/main` at `+0/-0`, with a clean tracked tree and the established
  local-only orchestration/spec/diagnostic files and historical logs untracked.
  Preserved every pre-existing worktree item.
- Confirmed the entry gate from GitHub evidence: Foundation CI run 57
  (`33586066060`) is successful at that exact starting SHA on both
  `ubuntu-latest` and `windows-latest`, including lock/sync, Ruff, pytest,
  scientific/claim verification, and package smoke. This is the latest
  completed CI state for the base commit; the uncommitted release-preparation
  work has no CI run or immutable candidate SHA yet.
- Audited release history. Annotated tags `v1.0.0` and
  `v1-course-submission` both peel to
  `3a3808841795938a296d48ae3b379b0d10ef3d48`; neither was moved. GitHub has
  one non-draft/non-prerelease release, `Medical Object Detector Benchmark
  v1.0.0`, published 2026-08-12 with no assets. Its body explicitly defines
  `3a380884...` as the immutable first stable Version 1 research baseline and
  requires later changes to use later versions. An older local untracked audit
  that names `df81f713...` is stale relative to the current tag object, remote
  tag refs, and GitHub release body.
- Classified the actual `v1.0.0..HEAD` delta (327 files; 371,627 insertions and
  9,460 deletions) from repository evidence rather than commit subjects:
  scientific/analysis changes include the five-attempt clean comparison,
  estimand-separated patient-cluster inference, n=5 operating-regime
  sensitivities, calibration/F-beta/raw-score-utility corrections,
  radiography-motivated sensitivities, and expanded XAI controls;
  reproducibility/verification changes include the 46-artifact manifest, 35
  claim bindings, checkpoint identity manifest, strict verifiers/tests, and
  clone-stable clean-checkout checks; infrastructure changes include the
  AGPL-3.0-only license, locked `uv` CPU environment, pinned cross-platform CI,
  and package smoke; documentation/manuscript changes include the canonical
  paper draft, consolidated limitations/hypotheses/decisions/reporting/citation
  audits, and the explicit four-tier reproduction boundary.
- Chose proposed version `2.0.0`, tag `v2.0.0`, and title `Medical Object
  Detector Benchmark v2.0.0 — Paper Artifact Release`. The major version is
  warranted by the materially expanded paper artifact relative to the first
  stable baseline. No SemVer/repository policy supports another value. Added
  D-009 to make the version model explicit: from v2 onward the Python project,
  importable package, lockfile root, and research release share one version;
  individual scientific artifacts/checkpoints remain identified by their own
  frozen hashes/provenance.
- Aligned `pyproject.toml`, `src/meddet_benchmark/__init__.py`, and the editable
  root entry in `uv.lock` from the unexplained bootstrap `0.1.0` value to
  `2.0.0`. `uv lock` resolved the existing 97-package graph and changed only
  the local project version. Historical v1 metadata was not rewritten.
- Created `CITATION.cff` from authoritative repository information only:
  software title `Medical Object Detector Benchmark`, author Pouyan
  Delivandani from the README copyright line, repository URL, version 2.0.0,
  and AGPL-3.0-only license. Omitted DOI, ORCID, affiliation, release date,
  venue, acceptance/publication status, and final paper citation. Locked
  PyYAML parsing plus required-field and package/release-version assertions
  passed; no dedicated CFF schema validator is installed in the locked
  environment.
- Created `CHANGELOG.md` with evidence-backed scientific, reproducibility,
  documentation, and infrastructure deltas plus the exact v1 commit. Created
  `RELEASE_NOTES_v2.0.0.md` covering release scope, exact verification
  commands and their limits, external data, undistributed checkpoints,
  limitations, citation, license, and previous release. Updated the README as
  the release landing page and `docs/REPRODUCIBILITY.md` with the single-version
  and immutable-tag/commit relationship plus exact release gate. No unsupported
  clinical, deployment, state-of-the-art, or full-reproducibility claim was
  added.
- Preserved the frozen scientific boundary. `report/paper_draft.md`,
  `report/report.md`, `report/paper_claim_sources.yaml`,
  `results/scientific_artifact_manifest.json`, and
  `results/checkpoint_release_manifest.json` remain byte-unchanged from HEAD.
  No result, CSV, figure, prediction, provenance record, threshold, seed,
  estimand, manuscript numerical claim, model/checkpoint, or artifact/checkpoint
  hash changed. Release metadata is outside the scientific manifest's input
  bindings, so no mechanical manifest refresh was performed.
- Confirmed distribution boundaries. Raw/processed patient images and all model
  binaries remain ignored and untracked; only the three expected `.gitkeep`
  placeholders occur under tracked raw/processed/checkpoint paths. The release
  will distribute source/configs/split metadata and Git-tracked frozen numerical
  evidence, not the external RSNA dataset. The ten audited checkpoints remain
  962,924,817 bytes of local ignored files; their manifest has a null public URL
  and supplies identity/provenance only. No checkpoint upload is authorized.
- Complete verification passed: `uv lock --check` (97 packages); locked
  CPU/dev sync; requested Ruff format (99 files) and lint scopes; CI-parity
  manifest-builder Ruff format/lint (one additional file); pytest (300 passed,
  one declared environment-conditional skip); scientific verifier (46
  artifacts, 110 present inputs, zero unavailable external/ignored inputs, 159
  referenced results); paper verifier (35 claims); package smoke; citation YAML
  and three-way 2.0.0 version consistency; `git diff --check`; and a separate
  trailing-whitespace check for the three untracked candidate files. The first
  pytest invocation encountered only an ACL conflict because ignored
  `.pytest-tmp`/`.pytest_cache` were owned by the managed sandbox identity; I
  preserved them under ignored
  `tmp/release-prep-pytest-sandbox-backup-20260902/`, then the exact pytest
  command passed. No user/source data was removed.
- Audited 582 tracked files for accidental distribution. The only files at or
  above 5 MiB are the expected scientific Grad-CAM panels (19.043 and 8.442
  MiB); no raw/processed image, DICOM, model/checkpoint binary, credential/
  environment file, private-key signature, or AWS access-key signature is
  tracked. The tracked `.json.gz` files are intentional frozen prediction
  bundles, not raw images or model binaries.

**What's still incomplete / next step:**
- The proposed release version is `2.0.0`, tag `v2.0.0`, and title `Medical
  Object Detector Benchmark v2.0.0 — Paper Artifact Release`. The exact
  candidate commit does not yet exist: current HEAD is still the starting/base
  SHA `226034c4f8b74b1bc3edfcd3da2dce7bd8c74d98`, and the nine intended public
  files are uncommitted worktree changes. Therefore CI run 57 proves the base
  boundary, while local verification proves the candidate worktree; it is not
  CI evidence for a release commit.
- After owner review, the next batch should commit exactly the nine intended
  public release-preparation files, push through the normal review path, and
  require Foundation CI success for that exact SHA. Only a later explicitly
  authorized publication batch may create the annotated `v2.0.0` tag and
  GitHub release.
- Unresolved citation/scholarly metadata: authoritative full paper author list,
  ORCIDs, affiliations, DOI, final venue/citation, acceptance/publication
  status, and actual release date. `CITATION.cff` intentionally does not guess
  them. Owner should also confirm whether the README copyright holder is the
  complete software author list before publication.
- Manuscript submission remains separately blocked by the human declaration
  register (funding, competing interests, ethics/data-use/consent,
  contributions, availability, and patient/public involvement). Checkpoint
  binaries remain undistributed; any future upload requires the explicit
  license/attribution/pretrained-weight/serialized-metadata review already
  documented. Neither issue was silently resolved by source-release metadata.

**Needs the user's review before proceeding:**
- Review the v2.0.0 rationale and D-009 single-version policy, citation author
  metadata/omissions, changelog classification, release notes, README landing
  text, and exact data/checkpoint exclusion language.
- Confirm the nine public candidate files are the intended commit scope. Do not
  include `AGENTS.md`, `CODEX.md`, `HANDOFF.md`, `PROJECT_SPEC.md`, any
  `BATCHES*.md`, local audits, historical failed/aborted/smoke/orchestration
  logs, or the ignored pytest backup.
- No release blocker is hidden: a reviewed candidate commit plus exact-sha
  green CI are still required before tagging, and final public tag/release
  creation remains explicitly outside this batch.

**Files touched:**
- Intended public release-candidate files: `CITATION.cff` (new),
  `CHANGELOG.md` (new), `RELEASE_NOTES_v2.0.0.md` (new), `README.md`,
  `pyproject.toml`, `uv.lock`, `src/meddet_benchmark/__init__.py`,
  `docs/REPRODUCIBILITY.md`, and `docs/DECISION_LOG.md`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  release assets).
- Ignored preserved test-temp state:
  `tmp/release-prep-pytest-sandbox-backup-20260902/`.
- Explicitly untouched: current/historical manuscripts, bibliography,
  scientific/claim/checkpoint manifests, all results/data artifacts, configs,
  source analysis code/tests, CI workflow, datasets, model weights/checkpoints,
  Git tags/releases, commits, branches, and remotes.

---

## Session 82 - 2026-09-02 - Foundation CI publication and remote confirmation

**What I did:**
- Began on local branch `fix/foundation-ci` at
  `470d958a4641a0f7f0725ebca71207625f5886d8`; `origin/main` was
  `b858772fe0d6abdcd68269b5163b4776a3acae0b`, the tracked tree was clean, and
  all established local-only orchestration files and historical logs were
  preserved. At the user's explicit direction, fast-forwarded `main` to the
  two reviewed CI commits and pushed them directly: `e5d44fd` fixes the
  clean-checkout threshold-sweep test and `470d958` modernizes the pinned
  Actions.
- Inspected `foundation-ci` run 55 (`33585534968`). Both Ubuntu and Windows
  passed checkout, setup, lock, sync, Ruff, and pytest (300 passed, one
  expected skip), proving the reported pytest regression was fixed. Both then
  failed the scientific artifact verifier because Batch 36 had recorded a
  Windows-worktree CRLF SHA-256 for
  `data/manifests/rsna-pneumonia-5000-audit.json`, whereas `.gitattributes`
  commits/checks out LF on GitHub runners. The expected stale hash was
  `75f08b27...`; the committed LF blob correctly hashed to `a39c847f...`.
- Confirmed the JSON was semantically identical and changed only its binding in
  `results/scientific_artifact_manifest.json`. Committed/pushed that repair as
  `de6d1f6` (`Fix clone-stable artifact manifest hash`). Run 56
  (`33585907662`) again passed pytest on both platforms and exposed the two
  remaining CRLF-derived bindings: `data/splits/rsna-pneumonia-5000/test.csv`
  (declared twice) and `results/tables/gradcam_localization_summary.csv`.
- Replaced all remaining worktree-derived values with hashes of the actual Git
  LF blobs (`6b589034...` and `65888889...`). The underlying CSV blobs matched
  the index after local line-ending normalization and had no Git diff. A clean
  `git archive` of HEAD plus only the proposed manifest passed both strict
  verifiers before publication. Committed/pushed the complete correction as
  `226034c4f8b74b1bc3edfcd3da2dce7bd8c74d98` (`Complete clone-stable
  artifact manifest hashes`).
- Confirmed `foundation-ci` run 57 (`33586066060`) completed successfully at
  that exact SHA. Both `ubuntu-latest` and `windows-latest` passed every step:
  immutable-SHA checkout/setup-uv, Python install, lock check, locked sync,
  Ruff format, Ruff lint, pytest, scientific artifact verification, manuscript
  claim verification, and package smoke. Local `main` and `origin/main` are
  identical at `226034c` with `0/0` divergence.
- Full local verification passed during the repair: `uv lock --check` (97
  packages), `uv sync --locked --group dev --extra cpu` (79 packages), exact CI
  Ruff format (100 files) and lint scopes, pytest (300 passed, one expected
  skip), scientific verifier (46 artifacts, 110 present inputs, zero
  unavailable inputs, 159 result references), paper verifier (35 claims),
  package smoke, and `git diff --check`. The final clean-archive check passed
  the scientific verifier with 106 present and four legitimately unavailable
  ignored/external inputs, plus all 35 paper claims.

**What's still incomplete / next step:**
- The software-integrity/release CI gate is restored. No CI warning or failure
  remains in run 57. Release preparation may proceed as a separate batch; no
  release was created here.
- The distinct Batch 38 submission blockers remain: authors must complete the
  human declarations and settle the data/model/checkpoint availability and
  release statement.

**Needs the user's review before proceeding:**
- Review the four published commits and successful run 57. Scientifically, the
  repair changes only byte-accurate manifest bindings to already committed LF
  blobs; it changes no artifact content, metric, threshold, seed, tolerance,
  numerical result, model/checkpoint, generated figure, or manuscript claim.

**Files touched:**
- Public commits: `tests/test_threshold_sweep.py`, `.github/workflows/ci.yml`,
  `results/scientific_artifact_manifest.json`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in any commit).
- Explicitly unchanged: `uv.lock`, `pyproject.toml`, source implementation,
  dataset/artifact contents, frozen scientific tables, paper numerical claims,
  checkpoint metadata/model weights, generated figures, manuscript content,
  and `docs/DECISION_LOG.md`.

---

## Session 81 - 2026-09-01 - Foundation CI software-integrity batch

**What I did:**
- Read the complete local operating state and relevant tracked CI,
  reproducibility, scientific-policy, verifier, test, configuration, lock, and
  source material. The batch began on `main` at
  `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`, aligned with the then-current
  `origin/main`, with no tracked/staged modifications and the established
  local-only orchestration files and historical logs untracked. During the
  investigation, the separately authored Session 80 publication advanced
  `main`/`origin/main` to `b858772fe0d6abdcd68269b5163b4776a3acae0b`;
  that concurrent user commit was preserved and used as the repair base.
- Inspected public GitHub Actions runs 51--54. The latest run 54
  (`33477906753`) failed only at pytest on both `ubuntu-latest` and
  `windows-latest`; checkout, setup-uv, Python install, lock check, sync, Ruff
  format, and Ruff lint all succeeded first. Both jobs reported the same test
  and exception: `tests/test_threshold_sweep.py::test_n5_sensitivity_config_retains_all_five_runs_and_seed_271`
  raised `FileNotFoundError` for
  `data/processed/rsna-pneumonia-5000/annotations/instances_test.json` from
  `_validate_upstream` -> `sha256_file`. Ubuntu finished 1 failed/299 passed/1
  skipped; Windows did the same. Run 51 at commit `fe304f7` is the earliest
  failing boundary; run 50 at `2f0dc6e` passed. A local `git archive` of the
  tracked `b858772` tree reproduced the same Windows failure (1 failed, 4
  passed), while the populated research workspace masked it because the
  ignored annotation file exists locally.
- Classified the regression as fixture construction/configuration, not
  implementation, ordering, serialization, tolerance, or platform behavior.
  Batch 35 added a metadata assertion that called production validation and
  thereby made a clean software test depend on an intentionally gitignored
  held-out dataset. Reworked only that test to validate the tracked Phase 5
  config/summary, complete ten-run detector/seed grid, all ten bundle SHA-256
  values, and the seed-271 bundle's identity, split, evaluator, annotation hash,
  and zero operating-point detections. Production `_validate_upstream`, frozen
  results, seed 271, metrics, thresholds, tolerances, seeds, claims, manifests,
  and provenance were not changed.
- Verified authoritative upstream releases and immutable targets. Updated
  `actions/checkout` from
  `11d5960a326750d5838078e36cf38b85af677262` (v4 line/package 4.3.0,
  `node20`) to `3d3c42e5aac5ba805825da76410c181273ba90b1`
  (v7.0.1, `node24`). Updated `astral-sh/setup-uv` from annotated v7 tag-object
  SHA `94527f2e458b27549849d47d273a16bec83a01e9` to v10.0.1 commit
  `20cfd1bf945f4377ade1205e4dbc17946fc9a30d` (`node24`). Kept the uv executable
  pinned to `0.11.27`, retained cache enablement and the Ubuntu/Windows matrix,
  and did not regenerate or modify `uv.lock`.
- Created dedicated local branch `fix/foundation-ci` from `b858772`. Commit
  `e5d44fd` contains only the test repair; commit `470d958` contains only the
  workflow modernization. Neither commit nor branch was pushed or merged.
- Full verification passed: `uv lock --check` (97 packages),
  `uv sync --locked --group dev --extra cpu` (79 packages checked), exact CI
  Ruff format scope (100 files), exact CI Ruff lint scope, pytest (300 passed,
  one expected metadata-only skip), scientific artifact verifier (46 artifacts,
  110 present inputs, zero unavailable inputs, 159 referenced result files),
  paper-claim verifier (35 claims), package/import smoke (`status: smoke`), and
  `git diff --check`. A writable temp `UV_CACHE_DIR` was required only to avoid
  the managed local Windows sandbox/elevated-identity ACL mismatch; it is not a
  repository or GitHub-runner configuration change.

**What's still incomplete / next step:**
- Push `fix/foundation-ci` when ready for review and require a fresh
  `foundation-ci` run on both matrix jobs before merging. The current remote
  run remains red because these local commits have not been pushed.
- After remote green status, merge through the repository's normal review path;
  no release was created in this batch.

**Needs the user's review before proceeding:**
- Review commits `e5d44fd` and `470d958`. Locally, `foundation-ci` is expected
  to pass and the software-integrity gate is ready for remote confirmation.
  Release preparation may proceed after that confirmation, but the separate
  Batch 38 human declaration/checkpoint-availability submission blockers remain.

**Files touched:**
- Public/review branch: `tests/test_threshold_sweep.py`,
  `.github/workflows/ci.yml`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in either commit).
- Explicitly unchanged: `uv.lock`, `pyproject.toml`, `results/`, `data/`, all
  frozen scientific tables/artifacts/manifests, checkpoint metadata/model
  weights, generated figures, manuscript claims/content, and
  `docs/DECISION_LOG.md`.

---

## Session 80 - 2026-09-01 - Batch 38 publication

**What I did:**
- Confirmed branch `main`, fetched `origin`, and verified the local and remote
  branches both started at `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`
  with `0/0` divergence.
- Staged exactly `docs/FINAL_SUBMISSION_AUDIT.md`. Kept `CODEX.md`,
  `HANDOFF.md`, standing rule/spec files, the prior review audit, and unrelated
  historical logs local/untracked, consistent with the established publication
  boundary.
- Inspected the staged file and fixed four Markdown hard-break whitespace
  findings. The final staged snapshot passed `git diff --cached --check` and
  contained one new file with 311 lines. The full audit gates had already passed
  in Session 79, and no code, result, claim binding, or manuscript changed.
- Created commit `b858772fe0d6abdcd68269b5163b4776a3acae0b`
  (`Add final submission audit`) directly on `main` and pushed it to
  `origin/main`. Post-push local and remote hashes match with zero divergence.

**What's still incomplete / next step:**
- Publication mechanics are complete. The audit disposition remains NOT READY
  FOR SUBMISSION until the authors complete the human-only declarations and
  settle the data/model availability and checkpoint-release statement.

**Needs the user's review before proceeding:**
- Review and resolve the blocking items in the published final audit. Do not
  characterize the project as submission ready before those items are closed.

**Files touched:**
- Public commit: `docs/FINAL_SUBMISSION_AUDIT.md`
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in the public commit)

---

## Session 79 - 2026-09-01 - Batch 38

**What I did:**
- Read the standing instructions, newest handoff, living project state, Batch
  38 final-gate instructions, current manuscript, canonical methodology/audit
  documents, manifests, result sources, README, and CI workflow. Recorded HEAD
  `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`, alignment with `origin/main`, and
  the pre-existing untracked/internal workspace state.
- Re-ran the full pytest suite (300 passed, one expected metadata-only skip),
  Ruff format/lint, package smoke, `uv lock --check`, the 46-artifact scientific
  verifier, and the 35-claim manuscript verifier. The initial `uv run` pytest
  wrapper could not read the sandbox-external uv cache, so the same locked local
  environment was run directly and passed.
- Traced 29 manuscript numbers from abstract, methods, results, discussion, and
  limitations to exact committed source cells/calculations; no mismatch or
  out-of-tolerance rounding was found. Audited every requested statistical,
  DCA/utility, threshold, XAI, radiography/DICOM, and calibration term in
  context.
- Built the cross-analysis dataset/patient/run/seed-271/estimand table and
  checked every manuscript caption/claim against it. Re-audited retrospective
  H1--H6, all 29 manuscript citation keys, CLAIM/STARD-AI/TRIPOD+AI scope, the
  README/document hierarchy and reproduction paths, manuscript/supplement
  links, named figures/tables/docs, and all ten local checkpoint hashes/sizes.
- Wrote `docs/FINAL_SUBMISSION_AUDIT.md`. Its disposition is **NOT READY FOR
  SUBMISSION**, while the software/evidence state is ready for author review.
  No scientific code, result, manuscript, commit, push, or remote state changed.

**What's still incomplete / next step:**
- Authors must complete the funding, competing-interest, ethics/data-use,
  consent, author-contribution, data/code/model-availability, and patient/public-
  involvement declarations.
- Authors must decide whether and where to publish the ten-checkpoint archive;
  if the manuscript changes, regenerate the claim/checklist bindings and rerun
  the final gates.

**Needs the user's review before proceeding:**
- Review the blocking and should-fix sections of
  `docs/FINAL_SUBMISSION_AUDIT.md`, complete the human-only declarations, and
  settle the release/availability statement. Do not describe the project as
  submission ready before those items are resolved.

**Files touched:**
- `docs/FINAL_SUBMISSION_AUDIT.md`
- `CODEX.md`
- `HANDOFF.md`

---

## Session 78 — 2026-09-01 — Batch 37 publication

**What I did:**
- Reloaded the completed Batch 37 state, confirmed the active branch was
  `main`, fetched `origin`, and verified local/remote both started at Batch 36
  commit `23b766dde47784764e14d29c76e0b23f107e33f6` with `0/0` divergence.
- Staged exactly four reviewed Batch 37 public files:
  `report/paper_draft.md`, `report/paper_claim_sources.yaml`,
  `docs/REPORTING_CHECKLIST.md`, and `docs/HYPOTHESIS_TRACEABILITY.md`. Kept
  internal state/spec files and all unrelated historical logs outside the
  commit; `report/report.md` remained unchanged.
- Revalidated the staged snapshot with `uv lock --check`, Ruff format/lint,
  300 passing tests and one expected metadata-only skip, the 35-claim paper
  verifier, the 46-artifact/110-input/159-reference scientific verifier, and
  staged whitespace checks.
- Created commit `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`
  (`Align manuscript with corrected evidence`) directly on `main` and pushed it
  to `origin/main`. A post-push fetch confirmed identical local/remote hashes
  and `0/0` divergence.

**What's still incomplete / next step:**
- Publication is complete. Await user direction before another batch,
  checkpoint release, or author-declaration completion.

**Needs the user's review before proceeding:**
- None for publication mechanics. The manuscript and declaration review points
  in Session 77 remain applicable to subsequent submission work.

**Files touched:**
- Public commit: the four reviewed Batch 37 files recorded in commit `9aaf414`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in the public commit).

---

## Session 77 — 2026-09-01 — Batch 37

**What I did:**
- Read the Batch 27--36 decisions, handoffs, current state, relevant spec/batch
  instructions, decision log, adjudication documents, result tables, scientific
  artifact manifest, claim manifest, citation audit, declaration register, and
  current manuscript before editing.
- Rewrote `report/paper_draft.md` as a bounded retrospective internal-testing
  comparison of the two disclosed lung-opacity pipelines. The structured
  abstract now reports the cohort/splits, five attempts, primary training-
  procedure estimand, absolute AP, operating-score behavior, seed 271, and the
  precision interval that crosses zero without a clinical-use or universal
  detector-family claim.
- Added/clarified canonical preprocessing, asymmetric training recipes,
  seed/run design, validation-only threshold selection, estimand separation,
  F-beta preference versus hypothetical loss, support-limited emitted-detection
  calibration, DCA exclusion, synthetic-shift classes, corrected Adebayo
  controls, and retrospective hypothesis status. Results visibly label n=3,
  n=5, n=5/n=4, checkpoint, and image scopes and retain the complete n=3-to-n=5
  change audit. Discussion, limitations, conclusion, and explicit author-action
  declarations now match the adjudicated evidence.
- Kept all citation-audit corrections, changed three paper-claim regex anchors
  to the final internal-testing wording, and refreshed the manuscript anchors
  in `docs/HYPOTHESIS_TRACEABILITY.md`. `report/report.md` was not edited.
- Regenerated `docs/REPORTING_CHECKLIST.md` against final manuscript SHA-256
  `0b4cbb4e9effb1c055f0da885ba23636e7adbfd1b885504d20b9a2715943f0f5`:
  44 CLAIM rows comprise 24 Yes, 17 No, and 3 Not Applicable, with STARD-AI and
  TRIPOD+AI retained only as analogical mappings.
- Validation passed `uv lock --check`, Ruff format/lint, Git whitespace, 300
  tests with one expected metadata-only skip, the paper verifier (35 claims),
  and the scientific verifier (46 artifacts, 110 present inputs, 159 result
  references).

**What's still incomplete / next step:**
- Batch 37 is complete and intentionally stopped for review. No commit, push,
  training, inference, result regeneration, or checkpoint/publication action
  occurred; published `main` remains at `23b766d`.
- Author-controlled declaration and release details remain explicit
  placeholders and must be completed by the authors before submission.

**Needs the user's review before proceeding:**
- Review the title/structured abstract, the n=5 principal versus n=3 historical
  result framing, the primary-versus-checkpoint-conditional statistical wording,
  and the scope-correct discussion/conclusion.
- Review the DCA removal, calibration/XAI/acquisition-shift boundaries, complete
  limitation set, and the 44-row regenerated reporting crosswalk.
- Confirm the author declaration placeholders and whether the manuscript should
  proceed to formatting/publication in a later batch.

**Files touched:**
- Manuscript/claim binding: `report/paper_draft.md`,
  `report/paper_claim_sources.yaml`.
- Reporting/state: `docs/REPORTING_CHECKLIST.md`,
  `docs/HYPOTHESIS_TRACEABILITY.md`, `CODEX.md`, and `HANDOFF.md`.

---

## Session 76 — 2026-08-31 — Batch 36 publication

**What I did:**
- Reloaded the completed Batch 36 state, confirmed the active branch was
  `main`, fetched `origin`, and verified local/remote both started at Batch 35
  commit `fe304f7c2bf26c59e204e7cd83c8cf6190f57857` with `0/0` divergence.
- Staged exactly 13 reviewed Batch 36 public files: CI, README and
  reproducibility documentation, the current-manuscript numeric wording change,
  the artifact/claim/checkpoint manifests, four scripts, and two test modules.
  Kept internal rule/spec/state files and all unrelated historical failed,
  aborted, smoke, and orchestration logs outside the commit.
- Verified that Git's staged bytes retained the checkpoint-release-manifest
  SHA-256 recorded by the scientific artifact manifest. Revalidated the staged
  snapshot with 300 passing tests and one expected metadata-only skip,
  `uv lock --check`, Ruff format/lint, package smoke, both scientific verifiers,
  and staged whitespace checks.
- Created commit `23b766dde47784764e14d29c76e0b23f107e33f6`
  (`Add scientific artifact verification`) directly on `main` and pushed it to
  `origin/main`. A post-push fetch confirmed `HEAD`, `origin/main`, and
  `origin/HEAD` all identify that commit with `0/0` divergence.

**What's still incomplete / next step:**
- Publication is complete. Await user direction before another batch or any
  checkpoint release/upload action.

**Needs the user's review before proceeding:**
- None for publication mechanics. The scientific/release review points in
  Session 75 remain applicable to later changes.

**Files touched:**
- Public commit: the 13 reviewed Batch 36 paths recorded in commit `23b766d`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in the public commit).

---

## Session 75 — 2026-08-31 — Batch 36

**What I did:**
- Read the standing/global rules, newest handoff, living state, Batch 36/spec,
  README, reproducibility contract, CI workflow, dependency locks, seed utility,
  current manuscript, checkpoint references, and all current phase/provenance
  summaries before changing the reproducibility boundary.
- Added `results/scientific_artifact_manifest.json`, a fixed reviewed inventory
  of 46 manuscript-critical artifacts. Every record contains the artifact,
  generator, config, and input SHA-256 bindings; exact CSV/JSON/PNG schema;
  study phase/reproduction tier; GPU/training flags; and result references. The
  maintenance builder is deliberately separate from the CI verifier so CI
  cannot bless stale evidence by refreshing hashes.
- Added `scripts/verify_scientific_artifacts.py`. It fails on missing/stale
  critical artifacts, generators, configs, or committed inputs; schema/row-count
  drift; malformed path/hash/flags; duplicate inventory; or missing declared
  result files. In the complete local workspace it checked 46 artifacts, 110
  present inputs, and 159 references; clean CI may omit only explicitly labeled
  external/ignored inputs.
- Added `report/paper_claim_sources.yaml` and
  `scripts/verify_paper_claims.py`. Thirty-five central cohort, performance,
  operating-point, FROC, efficiency, calibration, robustness, XAI, and
  inferential values are bound to exact cells/rows/JSON pointers or allow-listed
  calculations. Each has an explicit absolute rounding tolerance, each regex
  must match the manuscript exactly once, and every source must be listed in the
  scientific artifact manifest. Changed only the abstract's textual
  “threefold” to the equivalent machine-readable “3-fold.”
- Added eight regression tests covering valid manifests, absent allowed external
  inputs, stale hashes, schema drift, missing references, deterministic
  aggregations/calculations, manuscript mismatches, undeclared sources, and
  non-unique matches. Extended Ubuntu/Windows CI formatting/lint scope and added
  both lightweight verifiers after pytest; no training, dataset download,
  checkpoint loading, or GPU inference was added.
- Audited the ten exact Phase 5 best checkpoints. All ten local Git-ignored
  files exist, match Phase 5 SHA-256 values, and total 962,924,817 bytes. Added
  `results/checkpoint_release_manifest.json` with release filenames, sizes,
  hashes/configs, a conditional license/data-policy assessment, external release
  instructions, and `public_download_url: null`. No binary was committed,
  uploaded, or assigned a fabricated link.
- Rewrote the README/reproducibility wording to distinguish software tests,
  committed-analysis replay, exact inference, and exact retraining. Documented
  Python/package/uv-lock roles, CUDA 12.4/cuDNN 9.1/driver 610.47 hardware,
  RNG/DataLoader controls, `PYTHONHASHSEED` timing, warning-only ROI Align CUDA
  nondeterminism, Windows worker constraint, and the checkpoint audit/release
  procedure. Green CI is explicitly not end-to-end scientific regeneration.
- Final validation passed 300 tests with one expected metadata-only skip,
  `uv lock --check`, package smoke, both verifiers (46 artifacts/35 claims),
  Ruff format/lint, Git whitespace, and the independent ten-file checkpoint
  hash audit.

**What's still incomplete / next step:**
- Batch 36 is complete and intentionally stopped for review. No commit, push,
  checkpoint upload, public release, data download, model inference, or training
  occurred. Published `main` remains at Batch 35 commit `fe304f7`.
- A future release owner must complete the recorded attribution/license/privacy
  conditions before uploading checkpoint assets; until then the URL remains
  null and exact inference depends on the audited local files.

**Needs the user's review before proceeding:**
- Review the 46-artifact critical scope and whether any additional manuscript
  artifact should be promoted into the frozen inventory.
- Review the 35 numerical claim bindings, their unique regex anchors, and stated
  rounding tolerances.
- Review the four-level reproducibility wording and the conditional checkpoint
  release assessment/instructions. No publication or upload should proceed
  without approval.

**Files touched:**
- CI/code/tests: `.github/workflows/ci.yml`, `scripts/__init__.py`,
  `scripts/build_scientific_artifact_manifest.py`,
  `scripts/verify_scientific_artifacts.py`, `scripts/verify_paper_claims.py`,
  `tests/test_verify_scientific_artifacts.py`, and
  `tests/test_verify_paper_claims.py`.
- Manifests/artifacts: `results/scientific_artifact_manifest.json`,
  `results/checkpoint_release_manifest.json`, and
  `report/paper_claim_sources.yaml`.
- Docs/state: `README.md`, `docs/REPRODUCIBILITY.md`,
  `report/paper_draft.md`, `CODEX.md`, and `HANDOFF.md`.

---

## Session 74 — 2026-08-31 — Batch 35 publication

**What I did:**
- Reloaded the Session 73 state, fetched `origin`, and confirmed local `main` and
  `origin/main` started identical at `2f0dc6e7cb1ce0b232ef6fe5d1c1ca2d7f7a90a8`
  with zero divergence.
- Staged exactly 49 reviewed Batch 35 publication entries: configs, source,
  regression tests, canonical docs/current manuscript, four figures, 14 n=5 or
  comparison tables, and four Phase 35 provenance summaries. Kept internal
  rules/state/audits and unrelated failed, aborted, smoke, and orchestration
  logs out of the commit.
- Detected that Git's enforced LF normalization would change three CRLF-written
  provenance JSON blobs and break their recorded SHA-256 bindings in a clean
  checkout. Changed the three new atomic JSON writers to emit UTF-8 LF bytes,
  regenerated FROC/Pareto/final summaries, and verified all four JSONs are LF-
  only and every nested provenance hash resolves to the committed file bytes.
- Revalidated the corrected snapshot: 292 tests passed with one expected
  metadata-only skip; Ruff lint/format, Git whitespace, exact 49-path scope, all
  four offline preflights, artifact row/seed/classification/hash checks, local
  links, historical-result preservation, and visual review passed.
- Created commit `fe304f7c2bf26c59e204e7cd83c8cf6190f57857`
  (`Add five-run operating-regime sensitivity`) directly on `main` and pushed it
  to `origin/main`. A post-push fetch confirmed identical local/remote hashes and
  `0/0` divergence.

**What's still incomplete / next step:**
- Publication is complete. Await user direction before starting another batch
  or altering the reviewed operating-regime decision.

**Needs the user's review before proceeding:**
- None for publication mechanics. The scientific review points in Session 73
  remain the relevant considerations for any later manuscript revision.

**Files touched:**
- Public commit: the 49 reviewed Batch 35 Git entries recorded in commit
  `fe304f7c2bf26c59e204e7cd83c8cf6190f57857`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in the public release commit).

---

## Session 73 — 2026-08-31 — Batch 35

**What I did:**
- Read the global rules, newest handoff, living project state, decision log,
  relevant Batch 35/spec text, limitations, hypotheses, and current manuscript.
  Audited the frozen inputs and confirmed that PR, FROC, unchanged validation-
  selected-threshold performance, and Pareto can all use five test runs per
  detector from the ten existing hash-bound prediction bundles. No retraining,
  checkpoint/model loading, inference, threshold reselection, or test tuning ran.
- Preserved every original unsuffixed n=3 table/figure/log unchanged. Added four
  n=5 configs and an offline orchestration/audit module, extended threshold,
  FROC, and Pareto code, and added regression coverage that requires seeds
  17/42/137/271/314 and keeps YOLO11s seed 271 despite zero detections at 0.25
  and the frozen 0.05 threshold.
- Generated separately named n=5 per-run and aggregate PR/threshold, FROC,
  fixed-threshold, and Pareto artifacts plus a four-row recomputability inventory
  and a 19-row n=3-versus-n=5 conclusion audit. The audit contains 10
  strengthened, four weakened, five unchanged, and zero reversed conclusions;
  it reports adverse changes as well as favorable ones.
- Recorded D-008: the complete five-attempt sensitivity becomes the current
  manuscript's principal operating-regime display, while the n=3 analysis stays
  visible as historical/pre-specified provenance. The choice is based on design
  and provenance (complete predeclared attempt population), not aesthetics. The
  0.69/0.05 thresholds remain selected from n=3 validation evidence and are only
  applied unchanged to the five frozen test runs.
- Updated the requested threshold/FROC/Pareto/limitations docs, manuscript text
  and captions, README reproduction/provenance routing, and all related current
  hypothesis, comparison, plan, statistics, literature, YOLO, and supplementary
  docs. The preserved historical `report/report.md` was not changed.
- Reproduced all 16 n=5 core tables/figures byte-for-byte on a second offline
  run; visually reviewed the n=5 figures; passed all four no-training/no-inference
  preflights, 292 tests with one expected metadata-only skip, Ruff lint and
  formatting, exact row/seed/classification/hash checks, local-link resolution,
  and Git whitespace checks. Final Phase 35 summary SHA-256 is
  `297a395cb484c0e4628f3c68ff8dfb6a2c67a1b33fd7a1c3a24f21277de9a68a`.

**What's still incomplete / next step:**
- Batch 35 is complete and intentionally stopped for review. No commit or push
  occurred; published `main` remains at `2f0dc6e7cb1ce0b232ef6fe5d1c1ca2d7f7a90a8`.

**Needs the user's review before proceeding:**
- Review D-008 and confirm that the manuscript should keep the n=5 sensitivity
  as the principal operating-regime display with the original n=3 artifacts
  retained as provenance.
- Review the dedicated 19-row conclusion table, especially the four weakened
  findings (shared-threshold precision/recall margins and both mean AP gaps),
  alongside the strengthened and unchanged findings.
- Confirm the explicit boundary that test evaluation is n=5 but threshold
  selection remains n=3, and that seed 271's zero operating metrics coexist
  with nonzero threshold-free AP/low-threshold ranking.

**Files touched:**
- Config/code/tests: `configs/*_n5_sensitivity.yaml`,
  `src/analyze_operating_regime_sensitivity.py`,
  `src/evaluate_threshold_sweep.py`, `src/plot_froc_curves.py`,
  `src/plot_pareto_frontier.py`, and the four associated test modules.
- Artifacts: 16 `*_n5_sensitivity` CSV/PNG files,
  `results/tables/operating_regime_n5_inventory.csv`,
  `results/tables/operating_regime_n3_vs_n5_conclusions.csv`, and
  `results/logs/phase35_operating_regime_n5/`.
- Docs/state: `README.md`, the requested analysis/limitations/current-manuscript
  files, related canonical docs listed above, `docs/DECISION_LOG.md`, `CODEX.md`,
  and `HANDOFF.md`.

---

## Session 72 — 2026-08-31 — Batches 31--34 publication

**What I did:**
- Read the standing state and publication rules, fetched `origin`, and verified
  that local `main` and `origin/main` started identical at `fed66b2` with zero
  divergence.
- Staged exactly the 46 reviewed Batch 31--34 Git entries: implementation,
  configs, tests, canonical docs/manuscripts, figures, tables, and provenance.
  Preserved `report/report.md` as the historical/full report; its only staged
  change was the previously reviewed three-line Batch 32 acquisition-claim
  correction. Kept internal rules/state/audits and unrelated failed, aborted,
  smoke, and orchestration logs unstaged.
- Revalidated the release snapshot: 286 tests passed with one expected
  metadata-only skip; Ruff lint and format checks passed; `uv lock --check`
  passed; XAI, radiography-shift, and calibration preflights passed; all staged
  JSON/CSV artifacts parsed with expected row counts; all 29 manuscript citation
  keys and all staged local Markdown links resolved; Git whitespace, credential
  signature, and file-size screens passed. The environment has no `pip` module,
  so `pip check` was unavailable; the lockfile and executed test/preflight suite
  provide the dependency validation used for this release.
- Created commit `2f0dc6e7cb1ce0b232ef6fe5d1c1ca2d7f7a90a8`
  (`Correct evidence audits and reporting`) directly on `main` and pushed it to
  `origin/main`. A post-push fetch confirmed identical local/remote hashes and
  `0/0` divergence.

**What's still incomplete / next step:**
- Publication is complete. The author-controlled declaration fields, official
  submission checklist, remaining `No` CLAIM items, and scientific wording
  decisions documented in Session 71 still require human review before any
  submission-oriented revision.

**Needs the user's review before proceeding:**
- Review the published Batch 31--34 corrections and the Session 71 reporting,
  framework-applicability, declaration, sampling, and retrospective-hypothesis
  decisions. Do not begin a later batch or submission edit without direction.

**Files touched:**
- Public commit: the 46 reviewed Batch 31--34 Git entries recorded in commit
  `2f0dc6e7cb1ce0b232ef6fe5d1c1ca2d7f7a90a8`.
- Local session state: `CODEX.md`, `HANDOFF.md` (intentionally untracked and not
  included in the public release commit).

---

## Session 71 — 2026-08-31 — Batch 34

**What I did:**
- Read the complete standing project rules/state, Batches 28--33 handoffs,
  decision/limitation/hypothesis records, current paper, historical report
  routing, bibliography, and exact supporting artifacts before editing.
- Made the canonical hierarchy explicit in `README.md`:
  `report/paper_draft.md` is the current manuscript, `report/report.md` is the
  preserved historical/full technical report, `docs/` is the analysis/
  methodology/audit record, and `results/` is the numerical source of truth.
  The historical report was not rewritten or touched in this batch.
- Regenerated `docs/REPORTING_CHECKLIST.md` against the current paper. CLAIM
  2024 is primary and all 44 items use Yes/No/Not Applicable with exact
  manuscript/artifact evidence or a reason. The crosswalk uses internal
  testing/external testing terminology and explicitly is not an official
  compliance claim. Final STARD-AI 2025 was assessed as not a clean primary fit
  for this object-localization benchmark, and TRIPOD+AI as outside the primary
  individualized prediction-model scope; both are selected mappings by analogy.
- Added `docs/HYPOTHESIS_TRACEABILITY.md` with exact H1--H6 wording,
  retrospective/not-preregistered status, endpoint, split, seed count,
  artifacts, manuscript locations, result status, multiplicity, and limitations.
- Added `docs/CITATION_AUDIT.md`. Audited every one of the 29 manuscript citation
  keys plus CLAIM/STARD-AI/TRIPOD+AI. Added canonical COCO, SSIM, bootstrap,
  plus-one permutation, Holm, and DICOM PS3.4 sources; added the exact
  Ultralytics 8.4.110 release citation; and corrected the Ultralytics software
  author record. No unsupported or wrong-attribution row remains; limited
  mutable/triangulated implementation sources are marked `PARTIAL`.
- Added `docs/AUTHOR_DECLARATIONS_TODO.md` with explicit human-only actions for
  funding, competing interests, ethics/data use, consent, contributions,
  data/code availability, and patient/public involvement. No facts were
  invented.
- Updated the paper and `docs/LIMITATIONS.md` to disclose the deterministic
  seed-17/SHA-256/grouped/stratified 5,000-study selection, the absence of a
  formal sample-size or power calculation, resulting generalizability limits,
  retrospective hypothesis/multiplicity boundaries, and reporting-framework/
  declaration gaps. Renamed the held-out protocol as internal testing while
  retaining `validation` only for repository-named model-optimization artifacts.
- Verified all local Markdown targets, 44 CLAIM rows and official response
  values, every required field for H1--H6, all 29 citation-audit mappings, 32
  unique bibliography keys, zero unresolved paper citations, and Git whitespace.
  No code/config/numerical artifacts changed, so the 286-test Batch 33 result was
  not rerun. No commit or push occurred.

**What's still incomplete / next step:**
- Batch 34 is complete. The author-controlled declaration fields, official
  submission checklist, and remaining `No` CLAIM items cannot be resolved
  without human facts, unavailable source metadata, new analyses, or a later
  manuscript-revision decision. Stop for review.

**Needs the user's review before proceeding:**
- Review the strict CLAIM responses, especially the `No` decisions for the
  title/abstract, study-design label, reference-standard detail, prior-model
  comparison, participant flow/demographics/subgroups, availability, and funding.
- Confirm the applicability decisions: CLAIM primary; STARD-AI 2025 and
  TRIPOD+AI 2024 by analogy only.
- Supply or explicitly defer every `AUTHOR ACTION REQUIRED` declaration.
- Confirm the new sample-size/generalizability wording and the retrospective,
  nonconfirmatory H1--H6 traceability language before any submission edits.

**Files touched:**
- `README.md`, `report/paper_draft.md`, `report/references.bib`.
- `docs/REPORTING_CHECKLIST.md`, `docs/HYPOTHESIS_TRACEABILITY.md`,
  `docs/CITATION_AUDIT.md`, `docs/AUTHOR_DECLARATIONS_TODO.md`,
  `docs/LIMITATIONS.md`.
- `CODEX.md`, `HANDOFF.md`.

---

## Session 70 — 2026-08-31 — Batch 33 continuation

**What I did:**
- Resumed after the quota interruption by reading the top HANDOFF/CODEX state
  and inspecting the workspace rather than repeating completed work.
- Finished the last reference/provenance refinement: the record now separates
  the D-ECE definition from Küppers et al.'s detector demonstration protocol
  (0.3 probability threshold, NMS IoU 0.6, and correctness IoU 0.6/0.75). The
  project's 0.001 floor and IoU-0.50 matcher remain clearly identified as
  frozen project choices rather than a replication of those demonstration
  thresholds.
- Regenerated the Phase 33 provenance, verified all seven recorded artifact
  hashes plus source/config hashes, and confirmed 10 primary rows, 10 support
  rows, 160 sensitivity rows, and one correctly undefined high-floor row.
- Re-ran 11 focused calibration tests, the complete suite (286 passed, one
  expected metadata-only skip), Ruff format/lint, calibration preflight/run,
  and Git whitespace checks. No training, inference, calibrator fit, commit, or
  push occurred.
- After the user requested an explicit completeness check, re-audited all ten
  requested items against code, CSVs, figures, provenance, and canonical prose.
  Confirmed 10 unique support rows with every required field, 120 bin/support
  and 40 floor-sensitivity rows, ten flags for each original setting, exact
  preservation of all ten historical primary D-ECE values, one correctly
  undefined high-floor row, no seed-value logic or prospective outlier wording,
  no stale canonical artifact routes, and no D-ECE/DCA import or call coupling.
  A broad text scan initially matched the provenance phrase about Batch 30 DCA;
  the narrowed executable dependency check passed in both directions.

**What's still incomplete / next step:**
- Nothing remains to implement in Batch 33. Stop for user review.

**Needs the user's review before proceeding:**
- Same scientific review points as Session 69: definition/weighting,
  sparsity/floor sensitivity, descriptive claim boundary, and v2 supersession
  of historical Phase 18 reporting artifacts.

**Files touched:**
- `src/stats/calibration.py`, `docs/CALIBRATION_ANALYSIS.md`, `HANDOFF.md`.
- Regenerated `results/logs/phase33_calibration_support_v2/summary.json`
  (`a4461b3c...`); the seven table/figure hashes remain exactly as listed in
  Session 69.

---

## Session 69 — 2026-08-30 — Batch 33

**What I did:**
- Verified the D-ECE implementation against Küppers et al. The canonical
  population is post-NMS detections with score `>=0.001`; confidence plus
  relative center x/y and width/height are the five numeric axes, and predicted
  class is a categorical stratum. Matching is stable descending-score,
  same-class highest-IoU assignment to an unmatched target, no target reuse,
  IoU `>=0.50`, and 100 detections/image. Equal-width bins assign exact internal
  edges upward and 1.0 to the last bin. Supported cells are weighted by their
  share of all emitted detections; subminimum cells contribute zero without
  changing the denominator.
- Removed every seed-value special case from calibration code and presentation:
  the exact-grid assertion, seed-271 plot styling/annotation, and dedicated
  pathology provenance block. The Phase 5 run grid is now discovered
  dynamically. Replaced the retrospective H6 expectation that seed 271 should
  be worst with a neutral descriptive audit question; the observed largest
  YOLO D-ECE remains reported after analysis.
- Added versioned primary support diagnostics per run: total detections, total
  possible/occupied/supported cells, supported-detection fraction, and
  median/min/max supported-cell size. Of 3,125 possible cells, 68--354 are
  occupied and 15--169 supported; supported-detection fractions range
  0.543--0.983.
- Added the predeclared descriptive 3/5/7-bin by 1/4/8/16-minimum grid with the
  original 5-bin/minimum-8 setting flagged. Faster/YOLO equal-run mean D-ECE
  ranges were 0.0131--0.0531 and 0.0500--0.1551; absolute values decrease under
  stricter support because more detections occupy cells contributing zero.
- Added 0.001/0.005/0.01/0.05 floor sensitivity. At 0.005, equal-run mean
  retained population is only 46.6%/53.1% for Faster/YOLO. At 0.05, one YOLO
  run emits no detection and is explicitly undefined, not removed, imputed, or
  zero.
- Regenerated four versioned figures. Reliability diagrams are explicitly
  confidence-only marginal and every run is styled uniformly; separate figures
  show support/occupancy, bin/minimum sensitivity with the original cell
  outlined, and floor sensitivity with the original floor marked.
- Updated calibration documentation, README routing, supplement, reporting
  checklist, remediation audit, hypotheses, limitations, and paper. D-ECE is
  descriptive with no detector-level inference; missed ground-truth objects
  have no emitted confidence and lie outside its population; no clinical-risk
  or exam-level calibration follows. Detection D-ECE is programmatically and
  conceptually separate from the validation-frozen exam-outcome probability
  calibration required by valid DCA; Batch 30 fitted no such calibrator.
- Added 11 focused tests for arithmetic/weighting, boundary assignment,
  categorical classes, empty and sparse cells, score-floor inclusion/empty
  populations, confidence-only reliability, config grid inclusion, canonical
  matching, and absence of seed-value conditionals. Validation passed 286 tests
  with one expected metadata-only skip, Ruff format/lint, preflight, output and
  provenance hashes, visual review of all figures, and Git whitespace. No
  training, inference, calibrator fitting, commit, or push occurred.

**What's still incomplete / next step:**
- Batch 33 implementation and reporting are complete. Await review; do not
  commit, push, or begin a later batch without explicit user authorization.

**Needs the user's review before proceeding:**
- Review the class-conditioned D-ECE definition, the weighting choice that
  retains all emitted detections in (N), and the original minimum-8 rule.
- Review whether the substantial cell sparsity and score-floor population
  changes are communicated conservatively enough, especially the undefined
  high-floor run and the descriptive-only detector comparison.
- Confirm the v2 artifacts supersede the historical Phase 18 table/figure for
  reporting while preserving those historical files unchanged.

**Files touched:**
- Implementation/config/tests: `configs/calibration.yaml`,
  `src/stats/calibration.py`, `tests/test_calibration.py`.
- Documentation/manuscript: `README.md`, `docs/CALIBRATION_ANALYSIS.md`,
  `docs/HYPOTHESES.md`, `docs/LIMITATIONS.md`, `docs/SUPPLEMENTARY.md`,
  `docs/REPORTING_CHECKLIST.md`, `docs/REVIEW_REMEDIATION_AUDIT.md`,
  `report/paper_draft.md`, `CODEX.md`, `HANDOFF.md`.
- Versioned tables/provenance: `results/tables/calibration_summary_v2.csv`
  (`69a7c145...`), `calibration_support_v2.csv` (`8e1b842c...`),
  `calibration_sensitivity_v2.csv` (`b7461af1...`), and
  `results/logs/phase33_calibration_support_v2/summary.json` (`a4461b3c...`).
- Versioned figures: `reliability_diagrams_confidence_marginal_v2.png`
  (`9cc56b16...`), `calibration_support_occupancy_v2.png` (`659ddbaa...`),
  `calibration_binning_sensitivity_v2.png` (`12cea0d7...`), and
  `calibration_confidence_floor_sensitivity_v2.png` (`c43ae2a...`).

---

## Session 68 — 2026-08-30 — Batch 32

**What I did:**
- Audited every raw DICOM in the frozen 300-image acquisition-shift subset
  against current DICOM PS3.3/PS3.4 2026c. All are Secondary Capture Image
  Storage, `Modality=CR`, `MONOCHROME2`, workstation-converted (`WSD`), and
  marked as previously lossily JPEG-compressed. Pixel Intensity Relationship/
  Sign, Modality LUT/rescale, VOI LUT/Window Center/Width/function,
  Presentation Intent Type, and processing-description fields are absent in all
  300. The stored values therefore cannot be classified as approximately
  linear or logarithmic in incident X-ray signal or inverted reliably to such
  a scale.
- Added a 300-row field/missingness audit, 3,000 per-image pre/post-min-max
  diagnostics, a ten-condition preprocessing summary, and a corrected 20-row
  results table. The four DICOM windows are class A; the Poisson-like series is
  class D for physical validity and retained only as class-B generic intensity
  sensitivity; Gaussian blur is class C. The legacy `poisson_dose_*` IDs remain
  only to preserve exact RNG and bundle traceability.
- Confirmed the default DICOM `LINEAR` implementation, including half-unit and
  width-minus-one behavior, and added standard numeric regressions for
  center/width 2048/4096, 2048/1, and 0/100. MONOCHROME1/2 polarity and a
  min-max cancellation example are also regression-tested.
- Quantified canonical preprocessing: median NMAE residual ratios were 0.518
  and 0.593 for the center shifts, 1.004 for the narrow window, and 0.000 for
  the wide window, with 264/300 wide-window inputs exactly identical after
  scaling. Poisson-like medians were approximately unchanged; blur residual
  ratios were 2.455, 2.488, and 3.018 because min-max re-stretched their
  reduced dynamic ranges.
- Reclassified the analysis as radiography-motivated synthetic acquisition/
  display sensitivity throughout docs, README, manuscript, supplement,
  reporting checklist, limitations, and bibliography. DSI is explicitly
  descriptive and not an estimator of inter-site transportability. No dose,
  validated low-dose, detector-MTF, reconstruction-kernel, scanner, or clinical
  robustness interpretation remains.
- Did not rerun GPU inference: the implementation remains valid as a synthetic
  stress test, while only its former physical interpretation was unsupported.
  The Phase 22 table (`2cd14283...`), summary (`4782ac7c...`), and all 20
  prediction bundles remain unchanged and hash-bound. Phase 32 artifact hashes
  are metadata `013adc8b...`, detail `a31bb05c...`, preprocessing summary
  `db29cce7...`, corrected results `ddf85db4...`, and provenance `4c603e84...`.
- Validated 279 passing tests with one expected metadata-only skip, 15 focused
  acquisition/preprocessing tests, Ruff format/lint, `uv lock --check`,
  300-image preflight with zero PNG mismatches, exact source/config/artifact
  hashes, local links, manuscript citations, and Git whitespace. Batch 31's
  uncommitted work was preserved. No commit or push occurred.

**What's still incomplete / next step:**
- Batch 32 implementation is complete. Await review; do not commit, push, or
  begin a later batch without explicit user authorization.

**Needs the user's review before proceeding:**
- Review the conclusion that the 300 Secondary Capture objects do not support
  a dose/quantum-noise model, the per-transform A/B/C/D classification, and the
  corrected paper terminology.
- Review the preprocessing diagnostic, especially the almost-complete
  cancellation of the wide-window condition and re-stretching of blur
  differences, and confirm that preserving the old inference as superseded
  evidence was preferable to a scientifically unnecessary GPU rerun.

**Files touched:**
- Implementation/config/tests: `configs/acquisition_shifts.yaml`,
  `src/robustness/radiography_shifts.py`, `tests/test_radiography_shifts.py`.
- Documentation/manuscripts: `README.md`, `docs/ACQUISITION_SHIFTS.md`,
  `docs/LIMITATIONS.md`, `docs/SUPPLEMENTARY.md`,
  `docs/REPORTING_CHECKLIST.md`, `report/paper_draft.md`,
  `report/report.md`, `report/references.bib`.
- Public audit artifacts: `results/tables/acquisition_shift_dicom_metadata_audit.csv`,
  `results/tables/acquisition_shift_preprocessing_per_image.csv`,
  `results/tables/acquisition_shift_preprocessing_summary.csv`,
  `results/tables/radiography_synthetic_shift_results.csv`, and
  `results/logs/phase32_acquisition_shift_audit/summary.json`.
- Local-only state/provenance: `CODEX.md`, `HANDOFF.md`, plus Phase 32
  environment snapshots excluded by repository policy.

---

## Session 67 — 2026-08-30 — Batch 31

**What I did:**
- Audited the XAI implementation, config, historical Grad-CAM sanity artifacts,
  Phase 7 localization results, explainability docs, manuscript XAI section,
  and exact Adebayo et al. reference. Confirmed that the historical so-called
  data randomization was an inference-time RGB pixel-vector permutation and
  that the historical parameter control was one all-weights reinitialization.
- Renamed the former to an input-pixel randomization control everywhere outside
  the frozen historical CSVs and stated explicitly that it is not Adebayo et
  al.'s randomized-training test. Reclassified the latter as a full model-
  parameter randomization control. The canonical training-label test is marked
  `not performed`; no retraining was authorized or run.
- Added six transparent cumulative output-to-backbone parameter groups for
  Faster R-CNN and YOLO11s, strict exact-partition audits, a fresh deep copy at
  every stage, fixed-RNG deterministic reinitialization, and before/after
  checkpoint SHA-256 verification. The original checkpoint files were
  unchanged.
- Added independently normalized 40-by-40 Pearson, tie-aware Spearman, and
  Gaussian-window SSIM with one shared finite/constant-map rule. Added tests for
  checkpoint immutability, deterministic group randomization, metric and
  degenerate handling, partition completeness, and the input-versus-training-
  data terminology boundary.
- Reused the exact frozen 50-image subset and generated 700 per-image rows, 14
  summaries, and a versioned panel. Full-model Pearson/Spearman/SSIM means were
  `0.0025/0.0287/0.0381` for Faster R-CNN and
  `0.0242/0.0795/0.0476` for YOLO11s. Input-control means were
  `-0.0206/-0.0188/0.2468` and `-0.0021/0.0133/0.0067`. Intermediate stages
  were non-monotonic, so all cascade values remain descriptive.
- Reinterpreted Phase 7 localization conservatively: energy-in-box, pointing
  game, and box-area comparisons are descriptive, and small absolute overlap
  or lift is not strong-localization evidence. Updated XAI docs, limitations,
  supplementary/reporting routes, README, manuscript, and the complete
  Adebayo et al. NeurIPS 2018 BibTeX record. The old Batch 21 table/figure and
  provenance remained byte-identical historical artifacts.
- Validated 274 passing tests with one expected metadata-only skip, targeted
  XAI tests, Ruff format/lint, `uv lock --check`, preflight, exact hashes,
  partition coverage, checkpoint immutability, local links, manuscript
  citations, LF/CSV/JSON structure, and Git whitespace. No commit or push
  occurred.

**What's still incomplete / next step:**
- Batch 31 implementation is complete. Await review; do not commit, push, or
  begin a later batch without explicit user authorization.

**Needs the user's review before proceeding:**
- Review the detector-specific cumulative group definitions, non-monotonic
  stage curves, multi-metric findings (especially Faster R-CNN input-control
  SSIM), panel, and conservative manuscript wording.
- Confirm acceptance that the canonical Adebayo training-label data-
  randomization test is explicitly `not performed` because no new randomized-
  annotation retraining was authorized.

**Files touched:**
- Implementation/tests/config: `configs/xai_sanity.yaml`,
  `src/explainability/sanity_checks.py`, `tests/test_xai_sanity.py`.
- Documentation/manuscript: `README.md`, `docs/XAI_SANITY.md`,
  `docs/EXPLAINABILITY.md`, `docs/LIMITATIONS.md`, `docs/HYPOTHESES.md`,
  `docs/SUPPLEMENTARY.md`, `docs/REPORTING_CHECKLIST.md`,
  `report/paper_draft.md`, `report/references.bib`.
- Versioned public artifacts: `results/tables/gradcam_sanity_v2_summary.csv`,
  `results/tables/gradcam_sanity_v2_per_image.csv`,
  `results/figures/gradcam_sanity_v2_panel.png`,
  `results/logs/phase31_xai_sanity_v2/subset_manifest.csv`, and
  `results/logs/phase31_xai_sanity_v2/summary.json`.
- Local-only state/provenance: `CODEX.md`, `HANDOFF.md`, plus the Phase 31
  environment snapshots excluded by repository policy.

---

## Session 66 — 2026-08-29 — Batch 30 publication

**What I did:**
- Reloaded the Batch 30 handoff/state, fetched `origin/main`, and confirmed
  local and remote `main` both started at
  `22a814007a062347fd683645e4e9eb71367b14bc` with zero divergence.
- Staged exactly 27 reviewed Batch 30 public filesystem paths, represented as
  23 Git entries because the historical config, table, figure, and Phase 20
  provenance became 100% archive renames. The original DCA source was also
  preserved as a byte-identical archive copy. Internal protocol/state/audit
  files and unrelated failed/aborted/orchestration/environment diagnostics
  stayed unstaged.
- Audited the release snapshot: exact staged scope, no unstaged tracked
  changes, clean Git whitespace, all five historical archives identical to
  their former `HEAD` blobs, stable canonical artifact hashes, 198 rows with
  zero historical numerical difference, six-of-ten validation coverage,
  standard-DCA status false, 322 resolving local links, 21 resolving paper
  citation keys including both DCA method references, and no sensitive or
  stale canonical-path additions.
- Re-ran the release gates: 270 tests passed with one expected metadata-only
  skip; Ruff format/lint, `uv lock --check`, and raw-score preflight passed.
- Committed the exact snapshot on `main` as
  `fed66b22b2bfda37a665260c276402a18e8d83c7` (`Correct decision-curve
  interpretation`) and pushed it directly to `origin/main`. A post-push fetch
  confirmed identical local/remote hashes and zero divergence.

**What's still incomplete / next step:**
- Batch 30 publication is complete. Do not begin a later batch without
  explicit user authorization.

**Needs the user's review before proceeding:**
- None for publication.

**Files touched:**
- Public commit: 23 Git entries / 27 Batch 30 filesystem paths in commit
  `fed66b2`.
- Local-only session state: `CODEX.md`, `HANDOFF.md`.

---

## Session 65 — 2026-08-29 — Batch 30

**What I did:**
- Audited the complete Batch 20 DCA implementation, config, tests, 198-row
  table, figure, provenance, references, related documentation, and every DCA
  sentence in the paper. Confirmed that action was maximum emitted detector
  confidence `>= tau` while the same raw `tau` supplied the
  `tau/(1-tau)` false-positive weight. Classified the analysis as non-standard
  for conventional DCA under Vickers/Elkin 2006 and Vickers et al. 2008.
- Assessed the probability-calibration salvage gate. Ten detector/test runs are
  retained, but frozen validation predictions exist for only six (both
  detectors at seeds 17, 42, and 137); seeds 271 and 314 are missing for both
  detectors. Took the required removal/demotion path without choosing or
  fitting a calibrator and without test-driven calibration decisions.
- Removed raw-score DCA from the main paper Abstract, Methods, Results,
  Discussion synthesis, scenario claims, and Conclusion. Renumbered later
  sections/figures, added the corrective Supplementary/Limitations statement,
  added the verified 2008 extension reference, and synchronized README,
  reporting checklist, decision log D-007, and the DCA analysis note.
- Preserved the exact original Batch 20 source, config, table, figure, and
  provenance under explicit `pre_batch30_nonstandard` archive paths with
  verified hashes. Added a relabeled raw-score utility pipeline and generated
  its table, figure, and Phase 30 summary. The new run verified zero numerical
  difference across all 198 rows and 39 historical fields. Canonical new
  hashes: table `0bad365d...`, figure `8d91b2ed...`, summary `c7ae8123...`.
- Replaced the canonical DCA module with a strict probability-semantic guard:
  standard DCA accepts only a typed validation-frozen outcome-probability
  vector and a separately typed elicited decision threshold. New regression
  tests fail for raw detector-confidence predictors, raw confidence cutoffs
  passed as `p_t`, and plain numeric values. `docs/HYPOTHESES.md` required no
  edit because it contained no DCA claim.
- Verified the relabeled figure visually. Validation passed 270 tests with one
  expected metadata-only skip, focused tests, Ruff format/lint, raw-score
  preflight/run, archive/hash/numeric/provenance audits, `uv lock --check`, and
  Git whitespace checks. No training, inference, probability calibration,
  commit, or push occurred.

**What's still incomplete / next step:**
- Batch 30 implementation is complete locally. Await review; do not publish or
  begin a later batch without explicit authorization.

**Needs the user's review before proceeding:**
- Review D-007, the removal of the former paper Results section, the
  Supplementary/Limitations wording, and the relabeled non-standard figure.

**Files touched:**
- Code/config/tests: `src/clinical/decision_curve.py`,
  `src/clinical/raw_score_utility.py`, `src/clinical/archive/`,
  `configs/raw_score_utility.yaml`, archived Batch 20 config, and
  `tests/test_{decision_curve,raw_score_utility}.py`.
- Documentation/manuscript: `README.md`, `docs/DCA_ANALYSIS.md`,
  `docs/DECISION_LOG.md`, `docs/LIMITATIONS.md`,
  `docs/REPORTING_CHECKLIST.md`, `docs/SUPPLEMENTARY.md`,
  `report/paper_draft.md`, and `report/references.bib`.
- Artifacts: relabeled Phase 30 table/figure/summary plus exact archived Batch
  20 table/figure/provenance. Local state: `CODEX.md`, `HANDOFF.md`.

---

## Session 64 — 2026-08-29 — Batch 29 publication

**What I did:**
- Reloaded the Batch 29 handoff/state, fetched `origin/main`, and confirmed
  local and remote `main` both started at
  `fb4709b4995d85a05b6acfb20d3d1946d7cefdfa` with zero divergence.
- Staged exactly 24 reviewed public filesystem paths, represented as 21 Git
  entries because the old table, figure, and provenance became 100% archive
  renames. Internal `AGENTS`/batch/spec/state/audit files and unrelated
  failed/aborted/orchestration/environment diagnostics stayed unstaged.
- Audited the release snapshot: exact staged scope and working-byte identity,
  no unstaged tracked changes, clean Git whitespace, 352 resolving local
  links, valid provenance and CSV row counts, all eight bootstrap-frequency
  groups summing to one, canonical output hashes matching provenance, archive
  blobs matching their former `HEAD` objects, and no sensitive or stale
  canonical-path patterns.
- Re-ran the release gates: 266 tests passed with one expected metadata-only
  skip; Ruff format/lint, `uv lock --check`, and threshold preflight passed.
  Preflight confirmed six validation bundles, 750 validation images/321
  patient groups, no test access, no inference, and no training.
- Committed the exact snapshot on `main` as
  `22a814007a062347fd683645e4e9eb71367b14bc` (`Correct F-beta threshold
  sensitivity`) and pushed it directly to `origin/main`. A post-push fetch
  confirmed identical local/remote hashes and zero divergence. GitHub Actions
  `foundation-ci` run `33242465877` passed both Ubuntu and Windows jobs.

**What's still incomplete / next step:**
- Batch 29 publication is complete. Do not begin another batch without
  explicit user authorization.

**Needs the user's review before proceeding:**
- None for publication.

**Files touched:**
- Public commit: 21 Git entries / 24 Batch 29 filesystem paths in commit
  `22a8140`.
- Local-only session state: `CODEX.md`, `HANDOFF.md`.

---

## Session 63 — 2026-08-29 — Batch 29

**What I did:**
- Re-derived the implemented objective as
  `F_beta = (1 + beta^2) TP / ((1 + beta^2) TP + beta^2 FN + FP)` and
  replaced the unsupported clinical-cost interpretation throughout the
  canonical code, config, docs, decision log, README, table/figure labels, and
  manuscripts. D-006 now defines beta as a recall-versus-precision preference
  and beta squared as the relative recall weight in the harmonic mean. D-004's
  threshold-precedence rule remains, but its former terminology is superseded.
- Preserved the pre-test beta sweep `{1, 3, 5, 10}` and its original bootstrap
  random stream. All eight F-beta selections and estimates are unchanged:
  Faster R-CNN tau `0.69/0.33/0.12/0.03`; YOLO11s
  `0.02/0.01/0.01/0.01`. No test outcome selected, altered, or received one of
  these thresholds.
- Added validation-only stability diagnostics: a contiguous pointwise-LCB
  plateau within absolute 0.01 of the canonical maximum, the draw-specific
  bootstrap selected-tau distribution, and selection counts/frequencies for
  all 99 candidates per detector/beta. Faster R-CNN beta 3 and 5 have broad
  bootstrap 95% tau intervals (`0.13--0.51`, `0.04--0.29`); YOLO beta 3--10
  selects the 0.01 boundary in `99.7%/100%/100%` of draws.
- Added a separate, explicitly hypothetical validation loss
  `L(tau;r) = r * FN(tau) / N + FP(tau) / N` for assumed r
  `{1, 9, 25, 100}`. Selected Faster R-CNN thresholds are
  `0.87/0.61/0.23/0.04`; YOLO11s thresholds are
  `0.35/0.01/0.01/0.01`. The artifacts and prose state that these are linear
  detection-error assumptions, not measured patient harms or deployment
  utilities.
- Strengthened the input contract to require the model-development validation
  role, exact upstream validation split, an explicit false test-access flag,
  and exact annotation/split-manifest image identity. Tests cover test-label
  rejection, partition mismatch, the non-equivalence counterexample, stability
  frequencies, linear loss, and absence of a beta-to-cost conversion in the
  module.
- Regenerated five canonical Batch 29 artifacts twice with identical SHA-256
  hashes: F-beta summary `25d3d085...`, stability `e41c13e8...`, hypothetical
  loss `edd0564f...`, figure `830b829d...`, and provenance `9e0ae11a...`.
  The three old Batch 19 artifacts were moved to explicit pre-Batch-29 archive
  paths and verified content-identical to their tracked `HEAD` blobs. The new
  figure was visually inspected.
- `docs/HYPOTHESES.md` needed no edit because its operational wording did not
  claim a clinical cost. Validation passed: 266 tests plus one expected
  metadata-only skip, Ruff format/lint, `uv lock --check`, threshold preflight,
  Git whitespace, JSON/CSV row/schema/frequency/provenance checks, terminology
  audit, and deterministic artifact hashes. No training, inference, commit, or
  push occurred.

**What's still incomplete / next step:**
- Stop for user review. Do not commit, push, apply any sensitivity threshold to
  test, or begin another batch without explicit authorization.

**Needs the user's review before proceeding:**
- Review D-006 and the beta interpretation, the 0.01 plateau convention and
  bootstrap argmax diagnostic, the separate assumed-r loss analysis, the new
  artifact naming, and the manuscript wording.

**Files touched:**
- Implementation/config/tests: `src/stats/threshold_calibration.py`,
  `src/evaluate_threshold_selection.py`, `configs/threshold_calibration.yaml`,
  `tests/test_threshold_calibration.py`.
- Decisions/docs/manuscripts: `docs/DECISION_LOG.md`,
  `docs/THRESHOLD_CALIBRATION.md`, `docs/THRESHOLD_ANALYSIS.md`,
  `docs/LIMITATIONS.md`, `docs/SUPPLEMENTARY.md`,
  `docs/REPORTING_CHECKLIST.md`, `README.md`, `report/paper_draft.md`, and
  `report/report.md`.
- Canonical artifacts: `results/tables/recall_weighted_fbeta_threshold_summary.csv`,
  `results/tables/recall_weighted_fbeta_threshold_stability.csv`,
  `results/tables/hypothetical_detection_error_loss_summary.csv`,
  `results/figures/recall_weighted_fbeta_threshold_sensitivity.png`, and
  `results/logs/phase29_threshold_sensitivity/summary.json`.
- Historical archives:
  `results/tables/threshold_calibration_pre_batch29_archive.csv`,
  `results/figures/threshold_calibration_pre_batch29_archive.png`, and
  `results/logs/phase19_threshold_calibration/archive/summary_pre_batch29.json`.
- Session state: `CODEX.md`, `HANDOFF.md`.

---

## Session 62 — 2026-08-29 — Batch 28 publication

**What I did:**
- Reloaded the required project state, confirmed `main` tracked
  `origin/main`, fetched GitHub, and verified both started at
  `df2f35f43170699cca32540a283a50caa1764aee` with zero divergence.
- Staged exactly 22 public Batch 28 files: implementation/config/tests,
  decisions/docs/manuscripts, the regenerated clean table and Phase 8
  provenance, the historical paired-seed summary/table, and three seed-
  influence diagnostic tables. Internal `AGENTS`/batch/spec/state files,
  `docs/REVIEW_REMEDIATION_AUDIT.md`, unrelated orchestration/environment
  records, and historical failed/aborted runs remained unstaged.
- Audited the staged snapshot: exact scope, no unstaged tracked edits,
  source/artifact hash agreement, valid JSON, expected CSV row counts, no
  credential/private-key patterns, LF-stable hashed artifacts, 152 resolving
  local links, and clean Git whitespace.
- Re-ran the release gates: 262 tests passed with one expected metadata-only
  skip; Ruff format/lint, `uv lock --check`, and the statistics preflight all
  passed. Preflight confirmed 10 Phase 5 bundles, 750 clean images/323 patients,
  72 Phase 6 bundles, and 300 robustness images/183 patients.
- Committed the exact snapshot as
  `fb4709b4995d85a05b6acfb20d3d1946d7cefdfa` (`Correct statistical
  inferential targets`) and pushed it directly to GitHub `origin/main`.
  A post-push fetch confirmed local and remote hashes match with zero
  divergence. GitHub Actions `foundation-ci` run `33224313738` passed both the
  Ubuntu and Windows jobs.

**What's still incomplete / next step:**
- Batch 28 publication is complete. Do not begin Batch 29 or add a new
  statistical hypothesis test without explicit user authorization.

**Needs the user's review before proceeding:**
- None for publication. The next substantive remediation batch remains a
  separate decision.

**Files touched:**
- Public commit: 22 Batch 28 files recorded in commit `fb4709b4`.
- Local-only session state: `CODEX.md`, `HANDOFF.md`.

---

## Session 61 — 2026-08-29 — Batch 28

**What I did:**
- Added D-005 to `docs/DECISION_LOG.md`: the paper's primary
  training-procedure estimand includes held-out patient and stochastic
  retraining/run uncertainty; the secondary checkpoint-conditional estimand
  fixes the observed checkpoints and carries the patient-cluster permutation
  p-values.
- Audited both training implementations/configs. Equal seed labels do not form
  scientifically matched stochastic blocks or a common-random-number design:
  augmentation draws are absent in both arms, but loaders, batch sizes,
  initialization paths, frameworks, RNG-consumption sequences, and stopping
  trajectories are not coupled. The five runs are independent realizations
  that only share numeric labels.
- Implemented a patient-cluster plus independent within-detector trained-run
  bootstrap from the ten frozen Phase 5 bundles. Each draw reconstructs the
  nonlinear endpoints from sampled prediction evidence. It uses 5/5 runs for
  unconditional endpoints and 5 Faster R-CNN/4 YOLO11s defined runs for
  matched-detection IoU/Dice; no retraining or test-set change occurred.
- Preserved the previous paired-seed summary/table under explicitly named
  sensitivity archive paths. Added per-run metrics, leave-one-training-run-out,
  and descriptive leave-one-seed-label-out tables. Seed 271 remains included
  wherever its endpoint is defined; deleting its label is not presented as a
  corrected result. No new seed-aware p-value or hierarchical test was added.
- Regenerated the seven-row clean table and provenance, preserving the 497-row
  robustness table byte-for-byte. Primary Faster R-CNN-minus-YOLO11s intervals
  are wholly positive for recall, F1, mAP@0.5, and mAP@0.5:0.95. Precision is
  `-0.1024 [-0.2423, 0.0553]`, so it crosses zero under the primary estimand,
  although the separate checkpoint-conditional Holm p-value is `0.0020`.
  IoU/Dice also cross zero. Independent versus historical pairing changes no
  endpoint's zero-exclusion conclusion.
- Updated README, the canonical statistics write-up, hypotheses, limitations,
  supplement, project plan, and both manuscripts so every inference names its
  target. Removed the generic “five of seven endpoints” significance claim and
  added the plain-language CI/p-value distinction.
- Added regression coverage for patient rather than image resampling, seed 271
  retention, independent detector-run draws, and collision-safe repeated
  patient bootstrap identities. Validation passed: 262 tests plus one expected
  skip, Ruff format/lint, `uv lock --check`, statistics preflight, Git
  whitespace, required table columns, and artifact hashes. The clean table hash
  is `cf17182a5330f9f51d702e2f283dcf92accf3f78ef24e2030f86e1ee90cb12bb`;
  the frozen robustness hash remains
  `59c9ec17abacef102f854e463039e569fa22bac97e6fbc55c99af7f20f2dcb02`.

**What's still incomplete / next step:**
- Stop for user review. Do not commit, push, retrain, add a new hypothesis test,
  or start another remediation batch without explicit authorization.

**Needs the user's review before proceeding:**
- Approve the primary/secondary estimand choice, the independent-run bootstrap,
  the revised precision interpretation, and the updated manuscript wording.

**Files touched:**
- Implementation/config/tests: `src/stats/paired.py`,
  `src/stats/run_statistics.py`, `configs/statistics.yaml`,
  `tests/test_statistics.py`.
- Decisions/docs/manuscripts: `docs/DECISION_LOG.md`,
  `docs/STATISTICAL_ANALYSIS.md`, `docs/HYPOTHESES.md`,
  `docs/LIMITATIONS.md`, `docs/SUPPLEMENTARY.md`, `docs/PROJECT_PLAN.md`,
  `README.md`, `report/paper_draft.md`, `report/report.md`.
- Primary/provenance outputs: `results/tables/statistical_clean_comparison.csv`,
  `results/logs/phase8_statistics/summary.json`, `pip_freeze.txt`, and
  `run_environment.json`.
- New sensitivity/diagnostic outputs:
  `summary_paired_seed_sensitivity_archive.json`,
  `statistical_clean_comparison_paired_seed_sensitivity_archive.csv`,
  `statistical_clean_per_run_metrics.csv`,
  `statistical_clean_leave_one_run_out.csv`, and
  `statistical_clean_leave_one_seed_label_out.csv`.
- Session state: `CODEX.md`, `HANDOFF.md`.

---

## Session 60 — 2026-08-28 — Review remediation audit

**What I did:**
- Performed the requested audit from repository evidence rather than accepting
  the previous external review as ground truth. Recorded branch/HEAD, baseline
  worktree state, CI scope, latest local verification, public release assets,
  ignored checkpoint availability, and the documented canonical-file
  hierarchy.
- Created `docs/REVIEW_REMEDIATION_AUDIT.md` with one adjudication row for each
  of the 17 requested issues. Added detailed H1--H6 coverage and a claim-level
  verification of all 24 BibTeX citations used across repository Markdown plus
  CLAIM 2024, TRIPOD+AI 2024, STARD-AI 2025, and DICOM PS3.3 sources.
- Checked 18 manuscript quantities directly against committed CSV/JSON source
  artifacts, including cohort counts, clean AP/P/R/F1, seed 271, thresholds,
  FROC, D-ECE, CI/p-values, corruption retention, acquisition DSI, Grad-CAM,
  and XAI sanity. All values match at manuscript precision.
- Confirmed three submission blockers for later remediation: paired seed-index
  resampling is not supported by genuine cross-detector stochastic blocks;
  `beta^2` is not automatically a calibrated clinical FN:FP cost ratio; and
  the XAI pixel shuffle is not Adebayo's training-label data-randomization
  sanity test. A 300-file DICOM-header audit could not establish stored values
  proportional to beam intensity, so only the existing synthetic-proxy claim
  is supported.
- Made no scientific code, result, figure, checkpoint, or manuscript changes.
  Local software checks passed: `uv lock --check`, Ruff format/lint, 257 tests
  with one expected skip, and deterministic smoke. The locked CPU environment
  was not re-synchronized because the restricted sandbox could not download a
  missing seaborn wheel; checks used the existing environment with `--no-sync`.

**What's still incomplete / next step:**
- Stop for user review. No remediation has been implemented. After approval,
  the first scientific task is to choose and rerun a defensible independent or
  crossed seed-resampling method, then update affected claims from regenerated
  artifacts. Threshold-scenario relabeling and XAI-test renaming can follow in
  a separate explicitly authorized remediation batch.

**Needs the user's review before proceeding:**
- Review the audit's `BLOCKING BEFORE SUBMISSION`, `SHOULD FIX`, `OPTIONAL
  STRENGTHENING`, and `CLAIMS THAT SURVIVE UNCHANGED` sections, especially the
  paired-seed inference adjudication.

**Files touched:**
- New audit: `docs/REVIEW_REMEDIATION_AUDIT.md`
- Local-only session state: `CODEX.md`, `HANDOFF.md`
- No scientific result or manuscript file was modified.

---

## Session 59 — 2026-08-28 — Batch 26

**What I did:**
- Completed the Batch 18--26 iteration. Batch 18 added five-seed,
  detection-specific multivariate D-ECE and documented seed 271 as YOLO11s's
  worst calibration result. Batch 19 added D-004-governed, validation-only
  cost-sensitive threshold sensitivity without replacing the primary Batch 14
  thresholds. Batch 20 added full-test exam-level DCA with empirical 22.533%
  stratified-test prevalence and nominal-score caveats.
- Batch 21 added nested 50-image Grad-CAM parameter/input sanity checks and
  preserved the weak-localization, non-causal interpretation. Batch 22 added
  the 300-image, seed-17 raw-radiography VOI/Poisson/blur sensitivity study and
  kept it distinct from digital and clinical robustness. Batch 23 added the
  CLAIM/TRIPOD+AI/STARD-AI evidence crosswalk, supplementary index, and
  finite-n raincloud figure. Batch 24 published the first full IMRaD paper
  draft and the Vickers/Elkin DCA citation. Optional Batch 25 was not run.
- For Batch 26, audited manuscript quantities across 12 quantitative domains
  against tracked artifacts; no paper value required correction. Audited all
  tracked Markdown against `report/references.bib`: 24 unique citation keys and
  24 unique entries resolve, with no unused or duplicate key/DOI/title/URL;
  Küppers et al. and Vickers/Elkin are present.
- Added the missing DCA prevalence, same-test, nominal-score, localization,
  pointwise-interval, and clinical-utility limits plus Batch 23 reporting and
  governance gaps to `docs/LIMITATIONS.md`. Regenerated README's paper/report
  artifact-command index so all six executable modules added in Batches 18--23
  are explicit, including Batch 22 acquisition shifts.
- Validation passed: 257 tests with one expected metadata-only skip, Ruff
  format and lint, `uv lock --check`, deterministic smoke, and Git whitespace.
  Committed exactly `README.md` and `docs/LIMITATIONS.md` as
  `df2f35f43170699cca32540a283a50caa1764aee` (`Complete final consistency
  pass`) and pushed to `origin/main`. GitHub Actions run `33173026686` passed
  both Ubuntu and Windows jobs.

**What's still incomplete / next step:**
- Nothing remains for the Batch 18--26 iteration. Any stylistic/substantive
  manuscript revision is a separate editing pass. The optional Batch 25
  native-defaults training remains unrun and requires explicit authorization.

**Needs the user's review before proceeding:**
- No implementation checkpoint remains. Review the completed paper and choose
  whether to request an editing pass or a separately approved optional
  native-defaults experiment.

**Files touched:**
- Public Batch 26 commit: `README.md`, `docs/LIMITATIONS.md`
- Local-only session state: `CODEX.md`, `HANDOFF.md`
- Batch 18--24 public milestones: `cdfcdc0`, `cd155b2`, `5cc7d08`, `af24c10`,
  `d992cf8`, and `ce166a5`; Batch 26 closes the iteration at `df2f35f`.

---

## Session 58 — 2026-08-28 — Batch 24 publication

**What I did:**
- Fetched live GitHub `origin/main` and confirmed local `main` and remote
  started at `d992cf8` with zero divergence.
- Staged exactly `report/paper_draft.md` and `report/references.bib`. Excluded
  `CODEX.md`, `HANDOFF.md`, all protocol/batch files, host environment captures,
  smoke data, orchestration output, and unrelated historical diagnostics.
- Audited the staged snapshot: the manuscript remains 7,855 words with a
  257-word abstract; all 14 local links and 20 citation keys resolve; no
  sensitive path, credential, or private-key pattern matches; staged blobs are
  identical to the audited working bytes; and Git's whitespace gate passes.
- Committed the exact two-file snapshot as
  `ce166a59aeec69f6585ad5071286f4eb16fbee8d` (`Add controlled comparison
  paper draft`) and pushed it directly to GitHub `origin/main` as requested. A
  post-push fetch confirms identical local and remote hashes with zero
  divergence.

**What's still incomplete / next step:**
- Publication is complete. The manuscript remains a first full draft awaiting
  the user's substantive and stylistic editing review.
- Do not begin optional Batch 25 unless the user explicitly authorizes the
  extra training work after review.

**Needs the user's review before proceeding:**
- Review the title and abstract emphasis, evidence density and figure
  selection, scenario framing, pipeline-level attribution language, and the
  full unsoftened Limitations section.

**Files touched:**
- Public commit: `report/paper_draft.md`, `report/references.bib`
- Local-only session state: `CODEX.md`, `HANDOFF.md`

---

## Session 57 — 2026-08-28 — Batch 24

**What I did:**
- Created `report/paper_draft.md` without modifying `report/report.md`. The
  first full publication-style manuscript is 7,855 words with a 257-word
  abstract and follows the requested Title, Abstract, Introduction, Related
  Work, Materials and Methods, Results, Discussion, Limitations, and Conclusion
  structure.
- Integrated every Batch 9--23 evidence domain while preserving its actual
  scope: all-attempt n=5/n=4 clean results, frozen n=3 threshold/PR/FROC/Pareto
  work, patient-cluster inference, calibration, D-004 cost-sensitive threshold
  analysis, DCA, digital/acquisition-motivated robustness, Grad-CAM and sanity
  checks, the raincloud evidence, and reporting gaps. Exhaustive seed records,
  full grids, archives, and implementation details route to
  `docs/SUPPLEMENTARY.md` rather than being duplicated in the narrative.
- Framed the result as a trade-off between two disclosed pipelines, not a
  winner or architecture-family claim, and carried the training-recipe
  asymmetry, seed-271 pathology, nominal-score DCA limits, and consolidated
  clinical/external-validity hedging through the abstract, methods, results,
  discussion, and limitations. Added the Vickers/Elkin DCA entry to
  `report/references.bib`.
- Audited the draft: all 14 local Markdown links exist; all actual citation
  keys resolve; selected operating-point, calibration, and DCA spot checks
  match the committed CSV artifacts; no trailing whitespace was found; and
  `report/report.md` remains byte-unchanged with no Git diff.

**What's still incomplete / next step:**
- The manuscript is intentionally a first full draft. Apply the user's
  substantive and stylistic edits in a separate pass after review.
- Do not begin optional Batch 25 unless the user explicitly chooses to spend
  the additional training time.

**Needs the user's review before proceeding:**
- Review the title/abstract emphasis, evidence density and figure selection,
  pipeline-level attribution language, scenario framing, and whether the
  unsoftened Limitations section is appropriately complete for the intended
  venue.

**Files touched:**
- `report/paper_draft.md`, `report/references.bib`
- `CODEX.md`, `HANDOFF.md`

---

## Session 56 — 2026-08-28 — Batch 23 publication

**What I did:**
- Fetched live GitHub `origin/main` and confirmed local `main` started at zero
  divergence. Staged exactly the 11 audited Batch 23 public files; excluded
  `AGENTS.md`, batch/spec/state files, host environment captures, orchestration
  output, smoke data, and unrelated historical diagnostics.
- The staging whitespace/normalization gate caught that the new provenance JSON
  used Windows CRLF in the working tree and would have different bytes after
  Git normalization. Updated the atomic writer to force LF, added a regression
  test, regenerated the summary, and verified a second run is byte-identical.
  The figure hash remains `0ffdfcf...`; the clone-stable summary hash is now
  `e50ebf7...`.
- Re-ran validation after the fix: 257 tests passed with one expected
  metadata-only skip; Ruff lint/format, Git whitespace, uv lock, exact staged
  scope, aggregate/seed, local-link, sensitive-pattern, and staged-artifact-byte
  checks all passed.
- Committed the exact 11-file snapshot as
  `d992cf852387cd4ecd95b27df5067b3fdb4646bd` (`Add standards reporting and
  raincloud metrics`) and pushed it directly to GitHub `origin/main` as
  requested. A post-push fetch confirms identical local/remote hashes and zero
  divergence.

**What's still incomplete / next step:**
- Publication is complete. Await Batch 23 review; do not begin Batch 24.

**Needs the user's review before proceeding:**
- Review the conservative checklist statuses/reporting gaps, supplementary
  pointer organization, and raincloud layout/sample-size labels.

**Files touched:**
- Public commit: the 11 Batch 23 paths recorded above and in Session 55.
- Local-only session state: `CODEX.md`, `HANDOFF.md`.

## Session 55 — 2026-08-28 — Batch 23

**What I did:**
- Audited the current repository against all 44 CLAIM 2024 items, all 52
  TRIPOD+AI 2024 subitems, and all 40 STARD-AI 2025 main items (48 rows after
  subitems). Created a strict evidence crosswalk: every checked row has a
  resolving local artifact/code link, while absent abstracts, accrual dates,
  demographics/fairness evidence, ethics/consent, registration,
  funding/conflicts, patient/public involvement, flow diagram, and external
  evaluation remain unchecked or partial as appropriate.
- Added a pointer-style supplementary index to the full n=5 clean seed data,
  separately labeled frozen n=3 archives, threshold/calibration/DCA evidence,
  all 72 digital-corruption and 20 acquisition-shift rows, XAI/statistical
  records, the decision log, and exact reproduction commands.
- Added a strict config-driven Seaborn raincloud workflow for the seven clean
  predictive and seven compute/hardware endpoints. It verifies each plotted
  seed reduction against `detector_comparison.csv` before rendering. All panels
  state actual finite n; conditional IoU/Dice explicitly show Faster R-CNN n=5
  versus YOLO11s n=4 and identify seed 271 as undefined. Historical n=3 values
  are not mixed into the plot.
- Regenerated `raincloud_metrics.png` plus deterministic provenance from 138
  finite observations. A second run reproduced both files byte-for-byte
  (figure SHA-256 `0ffdfcf...`; summary `990624f...`). Added exact pandas and
  seaborn pins, updated the lockfile and README commands/index, and validated
  every local documentation link and source-table row count.
- Final validation: 256 tests passed with one expected metadata-only skip;
  Ruff lint/format, Git whitespace, uv lock, aggregate/seed, link/evidence, and
  source-count checks all passed.

**What's still incomplete / next step:**
- No Batch 23 implementation work remains. Batch 23 is local and uncommitted;
  await review and do not begin Batch 24.

**Needs the user's review before proceeding:**
- Review the deliberately conservative checklist statuses and named reporting
  gaps, the pointer-only supplementary organization, and the raincloud panel
  layout/sample-size labels—especially the conditional YOLO IoU/Dice n=4
  treatment.

**Files touched:**
- `docs/REPORTING_CHECKLIST.md`, `docs/SUPPLEMENTARY.md`
- `configs/raincloud_metrics.yaml`, `src/plot_raincloud_metrics.py`,
  `tests/test_raincloud_metrics.py`
- `results/figures/raincloud_metrics.png`,
  `results/logs/phase23_reporting/raincloud_metrics_summary.json`
- `README.md`, `requirements.txt`, `pyproject.toml`, `uv.lock`
- `CODEX.md`, `HANDOFF.md`

## Session 54 - 2026-08-27 - Batch 22 publication

**What I did:**
- Audited the Batch 22 publication scope before export. Replaced host-absolute
  artifact paths with repository-relative paths, preserved the pre-fix outputs
  under ignored `tmp/`, regenerated the complete 6,000-inference GPU analysis,
  and confirmed all scientific fields were unchanged.
- Verified the 20 compressed prediction bundles contain 300 records each, all
  CSV-recorded SHA-256 hashes match, all artifact paths remain inside the
  repository, and no host path, username, credential pattern, or private-key
  marker appears in their values. Excluded smoke output, host environment
  captures, internal state/protocol files, and unrelated historical logs.
- Re-ran the full validation suite: 254 tests passed with one expected
  metadata-only skip; Ruff lint/format and Git whitespace checks passed. The
  clone-stable result CSV and summary hashes remain `2cd1428...` and
  `4782ac7...`.
- Fetched live `origin/main`, confirmed zero initial divergence, committed the
  exact audited 29-file snapshot as
  `af24c100d9f8e3bc1a3735035948b4fc89d704fb` (`Add radiography
  acquisition-shift analysis`), and pushed it directly to `main` as requested.
  A post-push fetch confirmed identical local/remote hashes and zero
  divergence.

**What's still incomplete / next step:**
- Publication is complete. Await Batch 22 review; do not begin another batch.

**Needs the user's review before proceeding:**
- Review the absence-driven VOI baseline, the synthetic/non-calibrated Poisson
  and Gaussian parameter scope, the DSI results, and the explicit distinction
  between acquisition-motivated sensitivity, digital corruption robustness,
  and clinical robustness.

**Files touched:**
- Public commit: the 29 Batch 22 code/config/test/document/result/provenance
  paths recorded in Session 53.
- Local-only session state: `CODEX.md`, `HANDOFF.md`.

## Session 53 - 2026-08-27 - Batch 22

**What I did:**
- Passed the prerequisite before code changes: the ignored raw archive still
  contains 26,684 training and 3,000 competition-test DICOMs, and all 300
  frozen robustness studies map to distinct local training DICOMs.
- Audited the 300 source files directly. All are 8-bit unsigned
  `CR`/`MONOCHROME2` images with no Window Center/Width, VOI LUT Sequence,
  Modality LUT, Rescale Slope/Intercept, or pixel padding. Refactored the
  original per-image min-max logic into one shared function and proved clean
  reconversion is pixel-identical to every canonical sample PNG.
- Added a strict config-driven raw-radiography runner. It applies four exact
  DICOM PS3.3 default-`LINEAR` VOI center/width alternatives, three
  deterministic signal-dependent Poisson count conditions, and three finite
  Gaussian blur kernels before the shared 8-bit scaler. Both seed-17
  checkpoints use identical shifted pixels and the frozen unified evaluator;
  no training or checkpoint mutation occurs.
- Completed GPU smoke inference and the full 6,000-inference run. At the
  strongest Poisson condition, primary mAP@0.5:0.95 DSI is Faster R-CNN
  `0.314529` and YOLO11s `0.436641`; at 9x9 blur it is `0.248532` and
  `0.282869`. The VOI conditions are smaller/mixed, with the wide-window result
  near neutral (`-0.000119` and `0.000906`).
- Wrote the 20-row acquisition table, 20 resumable prediction bundles,
  provenance summary, README commands, and a dedicated method/results/scope
  document. The documentation explicitly separates raw-stage
  acquisition-physics motivation from Phase 6's generic post-conversion
  digital corruptions and rejects calibrated-dose, vendor-preset, scanner,
  clinical-robustness, and external-transportability claims.
- Audited all DSI formulas and bundle counts/hashes. A cache-only full rerun
  reproduced the clone-stable CSV and summary byte-for-byte (SHA-256
  `2cd1428...` and `4782ac7...`). Phase 6's summary and sample hashes remain
  unchanged. Final
  validation is 254 tests passed with one expected metadata-only skip; Ruff
  lint/format and whitespace checks pass.

**What's still incomplete / next step:**
- No Batch 22 implementation work remains. Await review; do not begin another
  batch.
- Batch 22 is local and has not been committed or published.

**Needs the user's review before proceeding:**
- Review the absence-driven VOI baseline, the synthetic/non-calibrated Poisson
  and Gaussian parameter scope, the DSI results, and the explicit distinction
  between acquisition-motivated sensitivity, digital corruption robustness,
  and clinical robustness.

**Files touched:**
- `configs/acquisition_shifts.yaml`, `src/robustness/radiography_shifts.py`,
  `src/data/prepare.py`, `tests/test_radiography_shifts.py`
- `results/tables/acquisition_shift_results.csv`,
  `results/logs/phase22_acquisition_shifts/`
- `docs/ACQUISITION_SHIFTS.md`, `docs/LIMITATIONS.md`, `README.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 52 — 2026-08-27 — Batch 21 publication

**What I did:**
- Re-read the publication-attempt handoff/current state and confirmed local
  `main` still contained only audited commit `5cc7d08` ahead of `origin/main`.
- After the user approved the disclosed export of the de-identified 50-study
  manifest and derived medical-radiograph panel, fetched live `origin/main` and
  confirmed a safe one-ahead/zero-behind fast-forward.
- Pushed `5cc7d081f66c77698a6c29ca4b8cc3131ef22ab7` (`Add Grad-CAM sanity
  checks`) directly to `origin/main`. A post-push fetch confirmed identical
  local/remote hashes and zero divergence.
- Preserved all protocol/state files, host-specific environment captures,
  orchestration output, and historical diagnostics as local-only files.

**What's still incomplete / next step:**
- Publication is complete. Await Batch 21 review; do not begin Batch 22.

**Needs the user's review before proceeding:**
- Review the Batch 21 fixed-reference target, failure denominators, and explicit
  conclusion that passing sanity checks reinforces failure-analysis use without
  supporting clinical reasoning or causal interpretation.

**Files touched:**
- Public commit: 12 Batch 21 files at `5cc7d08`
- `CODEX.md`, `HANDOFF.md` (local internal state; intentionally excluded from
  the public repository)

---

## Session 51 — 2026-08-27 — Batch 21 publication attempt

**What I did:**
- Re-read the Batch 21 handoff/current state, fetched live `origin/main`, and
  confirmed local and remote started at `cd155b2` with zero divergence.
- Audited and staged the exact 12-file Batch 21 public allowlist. Internal
  protocol/state files, host-specific `pip_freeze.txt`/`run_environment.json`,
  orchestration output, and historical failed/aborted/rejected/smoke folders
  remained excluded.
- Caught a pre-publication Windows newline issue: generated CSVs used CRLF, so
  Git's LF normalization invalidated their recorded SHA-256 values. Updated the
  atomic CSV writer to emit LF explicitly, added a regression test, and reran
  the checkpoint-only GPU analysis. The numerical results and panel remained
  unchanged, while all committed artifact/source/config hashes now verify.
- Revalidated the snapshot: 248 tests passed with one expected metadata-only
  skip; Ruff lint/format, whitespace, scope, sensitive-pattern, row-count, PNG,
  LF, and artifact-hash checks passed.
- Created local `main` commit
  `5cc7d081f66c77698a6c29ca4b8cc3131ef22ab7` (`Add Grad-CAM sanity checks`).
  The push did not execute: the safety gate requires explicit user approval
  after disclosure that the commit exports a 50-study de-identified manifest
  and a qualitative panel derived from medical radiographs to the configured
  GitHub repository.

**What's still incomplete / next step:**
- With explicit approval after the disclosure above, push local commit
  `5cc7d08` to `origin/main`, fetch, and verify zero divergence. Do not amend or
  add the local-only files.
- Do not begin Batch 22.

**Needs the user's review before proceeding:**
- Confirm whether to export the de-identified study manifest and derived
  radiograph panel already contained in `5cc7d08` to the configured public
  GitHub destination.

**Files touched:**
- Local public commit: the 12 Batch 21 files at `5cc7d08`
- `CODEX.md`, `HANDOFF.md` (local internal state; intentionally excluded from
  the public repository)

---

## Session 50 — 2026-08-27 — Batch 21

**What I did:**
- Implemented a strict, config-driven Grad-CAM sanity runner over a 50-image
  nested subset of the frozen Phase 6 robustness pool. Proportional largest-
  remainder allocation and seed-17 within-pool sampling yield 11 Lung Opacity,
  22 No Lung Opacity / Not Normal, and 17 Normal images from 41 NIH patients,
  with 18 boxes; the 300-image source manifest remains hash-bound.
- Defined one original trained highest-score candidate per image as a fixed
  reference region, then used a pre-activation foreground target at the same
  Phase 7 stride-16 layers for every condition. Faster R-CNN uses its fixed ROI
  classifier; YOLO11s uses the raw anchor closest to the reference center. This
  avoids conditioning the parameter test on a fully randomized proposal/box
  generator emitting valid post-NMS geometry and is explicitly distinguished
  from Phase 7's ground-truth-associated post-activation target.
- Deep-copied each trained checkpoint model and applied seeded
  `torch.nn.init.xavier_normal_` to every module weight tensor, including
  one-dimensional row views, while zeroing biases and preserving other buffers.
  Independently permuted RGB pixel vectors within each image through a stable
  SHA-256-derived seed, preserving exact pixel multisets and sending identical
  shuffled inputs to both detectors. No training or checkpoint mutation ran.
- Generated all four correlation endpoints. Faster R-CNN C_sanity is
  `0.001360` for parameter randomization (K=50) and `-0.015859` for data
  randomization (K=43); YOLO11s is `0.023442` (K=46) and `-0.003357` (K=50).
  Seven Faster shuffled-input and four YOLO randomized-weight maps are
  zero/constant and excluded rather than imputed. No valid pair reaches the
  predeclared `r >= 0.50` descriptive sanity-failure threshold.
- Wrote the method, findings, limitations, reproduction, and explicit claim
  impact. The result reinforces the existing cautious use of Grad-CAM as a
  model-specific failure-analysis tool but does not change the weak-localization
  finding or support clinical reasoning/causality. Added README routing and
  reconciled the explainability/consolidated-limitations documents.
- Visually inspected the 2984x3290 panel. The subset manifest, two CSVs, PNG,
  and machine summary are byte-identical across repeated GPU runs. Artifact,
  hash, formula, count, link, and PNG audits pass; 247 tests pass with one
  expected metadata-only skip; Ruff lint/format and whitespace checks are clean.

**What's still incomplete / next step:**
- No Batch 21 implementation work remains. Await review; do not begin Batch 22.
- Batch 21 is local and has not been committed or published.

**Needs the user's review before proceeding:**
- Review the fixed-reference/pre-activation target distinction from Phase 7,
  the explicit K/failure handling and `r >= 0.50` descriptive threshold, and the
  conclusion that passing sensitivity checks reinforces failure-analysis use
  without validating clinical reasoning.

**Files touched:**
- `configs/xai_sanity.yaml`, `src/explainability/sanity_checks.py`,
  `tests/test_xai_sanity.py`
- `results/tables/gradcam_sanity_summary.csv`,
  `results/tables/gradcam_sanity_per_image.csv`,
  `results/figures/gradcam_sanity_panel.png`,
  `results/logs/phase21_xai_sanity/`
- `docs/XAI_SANITY.md`, `docs/EXPLAINABILITY.md`, `docs/LIMITATIONS.md`,
  `README.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 49 — 2026-08-27 — Batch 20 publication

**What I did:**
- Fetched live `origin/main` and confirmed local `main` had zero starting
  divergence at `cdfcdc0`.
- Staged the exact nine-file Batch 20 public allowlist. Kept `CODEX.md`,
  `HANDOFF.md`, all batch/spec/protocol files, orchestration output, and
  historical failed/aborted/rejected/smoke diagnostics out of the commit.
- Validated the staged snapshot: 240 tests passed with one expected skip; Ruff
  lint and format checks passed; staged scope, whitespace, file-size,
  sensitive-pattern, artifact/JSON/CSV, and local Markdown-link audits passed.
- Created commit `cd155b2ac4b434f664397674bf4ac6528364dd5e` (`Add
  decision curve analysis`) directly on `main` and pushed it to
  `origin/main`. A post-push fetch confirmed identical local and remote hashes
  with zero divergence.

**What's still incomplete / next step:**
- Publication is complete. Await Batch 20 review; do not begin Batch 21.

**Needs the user's review before proceeding:**
- Review the Batch 20 DCA estimand, prevalence source, and scenario-linked
  interpretation before authorizing Batch 21.

**Files touched:**
- Public commit: nine Batch 20 files at `cd155b2`
- `CODEX.md`, `HANDOFF.md` (local internal state; intentionally excluded from
  the public repository)

---

## Session 48 — 2026-08-27 — Batch 20

**What I did:**
- Implemented a strict, config-driven, CPU-only decision-curve runner over all
  ten hash-bound Phase 5 test bundles. It reduces each detector to an exam flag
  when the maximum emitted box confidence reaches tau; it performs no training,
  inference, checkpoint loading, threshold fitting, or robustness-subsample
  reuse.
- Derived the disease-prevalence input empirically from the complete held-out
  test population and cross-checked manifest positivity against COCO annotation
  presence: 169 positive and 581 negative radiographs among 750 images from 323
  NIH patient groups, prevalence `0.225333` (`22.533%`).
- Reused the established patient-cluster/seed bootstrap plan for 2,000 common
  draws at all 99 thresholds. The CSV reports both detector net benefits and
  pointwise 95% intervals, treat-all/treat-none references, paired detector
  differences, population counts, prevalence, and strategy labels.
- Generated and visually inspected the required figure. Treat-all is the
  largest point-estimate strategy at thresholds `0.01--0.03`, Faster R-CNN at
  `0.04--0.41`, YOLO11s at `0.42--0.62`, and treat-none at `0.63--0.84`.
  Paired intervals favor Faster R-CNN at `0.01--0.27` and YOLO11s at
  `0.60--0.62` plus `0.64--0.88`; the write-up explicitly explains that much
  of the latter range is negative versus treat-none rather than positive
  clinical utility.
- Documented the established retrospective-screening and human-reviewed
  point-of-care scenarios, the exam-level/non-localization estimand, and the
  limitation that raw detector scores are nominal rather than validated
  clinical-risk probabilities. Added exact README reproduction commands.
- Confirmed the CSV, PNG, and provenance JSON are byte-identical across reruns.
  Artifact/formula/count audits pass; 240 tests pass with one expected skip;
  Ruff lint/format and whitespace checks are clean.

**What's still incomplete / next step:**
- No Batch 20 implementation work remains. Await review; do not begin Batch 21.
- Batch 20 is local and has not been committed or published.

**Needs the user's review before proceeding:**
- Review the exam-level maximum-score action rule, the full-test empirical
  prevalence, the distinction between pairwise detector advantage and benefit
  over treat-none, and the scenario-linked interpretation before Batch 21.

**Files touched:**
- `configs/decision_curve.yaml`, `src/clinical/__init__.py`,
  `src/clinical/decision_curve.py`, `tests/test_decision_curve.py`
- `results/tables/dca_summary.csv`, `results/figures/dca_curves.png`,
  `results/logs/phase20_decision_curve/summary.json`
- `docs/DCA_ANALYSIS.md`, `README.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 47 — 2026-08-25 — Batches 18--19 published to main

**What I did:**
- Fetched live `origin/main` and confirmed zero starting divergence at
  `5defa77` before publication.
- Staged the exact 21-file public Batch 18--19 allowlist. Excluded `AGENTS.md`,
  all batch/spec/state files including `CODEX.md`/`HANDOFF.md`, orchestration
  output, and historical failed/aborted/rejected/smoke diagnostics.
- Validated the staged snapshot: 235 tests passed with one expected skip;
  Ruff lint and format checks passed; staged whitespace, local Markdown links,
  JSON parsing, file-size, and sensitive-path checks passed.
- Created commit `cdfcdc025d495dc7db144aad6a0fd0471b468878` (`Add
  calibration and cost-sensitive threshold analysis`) directly on `main` and
  pushed it to `origin/main`. A post-push fetch confirmed identical local and
  remote hashes with zero divergence.

**What's still incomplete / next step:**
- Publication is complete. Await Batch 19 review; do not begin Batch 20.

**Needs the user's review before proceeding:**
- Review D-004's primary-versus-sensitivity relationship and the Batch 19
  threshold findings before authorizing Batch 20.

**Files touched:**
- Public commit: 21 Batch 18--19 files at `cdfcdc0`
- `CODEX.md`, `HANDOFF.md` (local internal state; intentionally excluded from
  the public repository)

---

## Session 46 — 2026-08-25 — Batch 19

**What I did:**
- Reconciled the new selector before implementation. D-004 keeps Batch 14's
  validation-selected thresholds (Faster R-CNN 0.69, YOLO11s 0.05) as the
  authoritative primary operating points and defines Batch 19 as a separate
  cost-sensitivity extension. No Batch 19 threshold replaces or feeds the
  existing test, FROC, or Pareto artifacts.
- Implemented a strict, config-driven, CPU-only runner over the six hash-bound
  Phase 14 validation bundles. It evaluates the canonical IoU-0.50 operating
  point at 99 thresholds, computes F1-beta for beta 1/3/5/10 (assumed FN/FP
  cost ratios 1/9/25/100), and selects the maximum lower 95% percentile bound.
  The run accesses no test prediction, checkpoint, model inference, or training.
- Extracted the existing Phase 8 hierarchical patient-cluster/seed draw into a
  shared `paired.py` helper and reused it unchanged. Every one of 2,000 common
  draws resamples 321 NIH validation-patient groups and the three frozen seeds;
  repeated exams move together, and the same draws are used across thresholds
  and beta values.
- Generated the required table and sensitivity figure. Faster R-CNN thresholds
  for beta 1/3/5/10 are `0.69/0.33/0.12/0.03`; YOLO11s thresholds are
  `0.02/0.01/0.01/0.01`. YOLO beta 3--10 is explicitly lower-boundary-limited,
  not claimed to be an interior/global optimum.
- Wrote the mathematical method, D-004 relationship, results, limitations, and
  exact reproduction commands. Added boundary status and full threshold curves
  to the machine-readable artifacts. The PNG was visually inspected, and the
  CSV/PNG/provenance JSON are byte-identical across repeated runs.
- Passed preflight over all frozen inputs; four focused new tests plus the
  existing suite pass. Final validation is 235 tests passed with one expected
  skip, Ruff lint/format clean, and `git diff --check` clean.

**What's still incomplete / next step:**
- No Batch 19 implementation work remains. Await review; do not begin Batch 20.
- Batch 18 and Batch 19 remain local and have not been committed or published.

**Needs the user's review before proceeding:**
- Review D-004's decision to preserve Batch 14 as primary, the hierarchical
  bootstrap/maximum-lower-bound selector, the Faster R-CNN threshold shifts,
  and the interpretation of YOLO's beta 3--10 lower-boundary result.

**Files touched:**
- `docs/DECISION_LOG.md`, `configs/threshold_calibration.yaml`,
  `src/stats/threshold_calibration.py`, `src/stats/paired.py`
- `tests/test_threshold_calibration.py`, `tests/test_statistics.py`
- `results/tables/threshold_calibration_summary.csv`,
  `results/figures/threshold_calibration_sensitivity.png`,
  `results/logs/phase19_threshold_calibration/summary.json`
- `docs/THRESHOLD_CALIBRATION.md`, `docs/LIMITATIONS.md`, `README.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 45 — 2026-08-25 — Batch 18

**What I did:**
- Implemented a strict, config-driven, CPU-only calibration runner over all ten
  frozen Phase 5 bundles. It independently implements Küppers et al.'s full
  five-dimensional D-ECE using confidence plus relative box center/scale, five
  equal-width bins per dimension, and the paper's eight-sample cell minimum.
  Matching reuses the canonical score-ordered same-class IoU-0.50 rule for all
  predictions retained at the frozen 0.001 floor; no checkpoint or model is
  loaded.
- Checked the public calibration framework's Apache-2.0 license against this
  repository's AGPL-3.0-only license. They are compatible under GNU's v3
  guidance, but the method was independently reimplemented to avoid adding the
  framework's broad dependency stack or copying its source. Added the paper to
  `report/references.bib`.
- Generated the required table and reliability figure. Five-seed mean D-ECE is
  Faster R-CNN `0.032043 +/- 0.005761` versus YOLO11s
  `0.099027 +/- 0.023160`. YOLO11s seed 271 is retained as the worst YOLO seed:
  D-ECE `0.131298`, mean confidence `0.005382`, TP fraction `0.150728`, global
  gap `0.145346`, maximum confidence `0.041274`, and 962 predictions.
- Added H6 and `docs/CALIBRATION_ANALYSIS.md`, explicitly separating measured
  probability calibration from Batch 10/14's score-scale/selectivity result.
  Added calibration scope to limitations and exact reproduction commands to
  the README.
- Added four focused regression tests, passed preflight over all bundle/config/
  annotation hashes, visually inspected the rendered 2380x1067 PNG, and
  confirmed the CSV, PNG, and provenance summary are byte-identical across
  repeated runs. Full validation is 231 tests passed with one expected skip,
  Ruff lint/format clean, local-link/BibTeX/artifact audits passed, and
  `git diff --check` clean.

**What's still incomplete / next step:**
- No Batch 18 implementation work remains. Await review; do not begin Batch 19.
- Batch 18 is local and has not been committed or published.

**Needs the user's review before proceeding:**
- Review the full-D-ECE protocol and conditional emitted-detection scope, the
  approximately threefold detector mean difference, the explicit seed-271
  underconfidence outlier, and the distinction from threshold selectivity.

**Files touched:**
- `configs/calibration.yaml`, `src/stats/calibration.py`,
  `tests/test_calibration.py`
- `results/tables/calibration_summary.csv`,
  `results/figures/reliability_diagrams.png`,
  `results/logs/phase18_calibration/summary.json`
- `docs/HYPOTHESES.md`, `docs/CALIBRATION_ANALYSIS.md`,
  `docs/LIMITATIONS.md`, `README.md`, `report/references.bib`
- `CODEX.md`, `HANDOFF.md`

---

## Session 44 — 2026-08-20 — Repository licensing follow-up

**What I did:**
- Verified live `origin/main` matched local `main` at `0324366` before making
  the licensing change.
- Reviewed the project boundary and current primary-source licensing guidance.
  Because the benchmark integrates Ultralytics YOLO under AGPL-3.0, selected
  GNU AGPL v3 only rather than a permissive license.
- Added the canonical GNU AGPL v3 text as `LICENSE`, declared
  `AGPL-3.0-only` plus the license-file metadata in `pyproject.toml`, and added
  a README copyright, license, attribution, and third-party-scope section.
  RSNA/NIH data and dataset-derived image content, pretrained weights, and
  dependencies remain governed by their own terms.
- Confirmed the local license matches GNU's canonical text after newline
  normalization, parsed `pyproject.toml`, checked the exact three-file staged
  scope, and passed whitespace validation.
- Committed as `5defa77d50e3988b3045271e4b1c2ef1588ffc85` (`Add AGPL-3.0
  repository license`) and pushed directly to `origin/main`. The post-push
  fetch confirmed identical local/remote hashes and zero divergence.

**What's still incomplete / next step:**
- No licensing or publication work remains. Continue to await the user's
  Batch 17 writing review; do not begin Batch 18 yet.

**Needs the user's review before proceeding:**
- Review the published Batch 17 writing before authorizing Batch 18.

**Files touched:**
- Public commit: `LICENSE`, `README.md`, `pyproject.toml`
- Local state only, intentionally excluded from the public commit: `CODEX.md`,
  `HANDOFF.md`

---

## Session 43 — 2026-08-20 — Batch 17 published to main

**What I did:**
- Fetched live `origin/main` and verified zero divergence at `1876044` before
  publication.
- Staged exactly the ten public Batch 17 writing files and excluded local
  project-state/protocol files and diagnostic logs.
- Validated the staged snapshot for exact scope, whitespace, sensitive paths,
  citation keys, local Markdown links, and explicit H1--H5 evidence scopes.
- Committed the writing revision as
  `032436632f674138162d2038b698fc2caba14db7` (`Reframe literature review and
  hypotheses`) and pushed it directly to `origin/main`. A post-push fetch
  confirmed identical local and remote hashes with zero divergence.

**What's still incomplete / next step:**
- No publication work remains. Await the user's Batch 17 review; do not begin
  Batch 18 or create `report/paper_draft.md` yet.

**Needs the user's review before proceeding:**
- Review the published Batch 17 writing before authorizing Batch 18.

**Files touched:**
- Public commit: `docs/DATASET_CHOICE.md`, `docs/DATASHEET.md`,
  `docs/EXPLAINABILITY.md`, `docs/HYPOTHESES.md`, `docs/LIMITATIONS.md`,
  `docs/LITERATURE_REVIEW.md`, `docs/PARETO_ANALYSIS.md`,
  `docs/PROJECT_PLAN.md`, `docs/YOLO_BASELINE.md`, `report/report.md`
- Local state only, intentionally excluded from the public commit: `CODEX.md`,
  `HANDOFF.md`

---

## Session 42 — 2026-08-20 — Batch 17

**What I did:**
- Reframed `docs/LITERATURE_REVIEW.md` around the specific controlled-comparison
  gap supported by the existing bibliography. The claim is deliberately scoped
  to the reviewed medical-detector studies: architecture is co-varied with
  splits/data, preprocessing, augmentation, operating thresholds, and metric
  definitions or aggregation. The project contribution is one patient-disjoint,
  unified-evaluation protocol plus the observed shared-threshold/different-
  operating-regime result.
- Added `docs/HYPOTHESES.md` with one primary research question and five
  retrospective, result-linked hypotheses covering coverage/AP, corrected
  precision-recall/FROC behavior, the accuracy-compute Pareto trade-off,
  corruption-specific robustness, and Grad-CAM failure analysis. Every
  hypothesis names an operational check, exact frozen tables/figures, and its
  n=5, paired n=4, frozen n=3, or seed-17 scope.
- Completed the requested terminology pass across project claims in `docs/`
  and `report/report.md`: the task is lung-opacity detection and boxes are
  opacity annotations, not diagnostic pneumonia labels or lesion masks.
  Official challenge/publication titles, BibTeX keys, URLs, file paths, and
  explicit pneumonia-diagnosis disclaimers remain unchanged.
- Ran writing-only static validation. All 22 citation keys used by the revised
  literature/report resolve in `report/references.bib`, every local Markdown
  link in the changed content resolves, no trailing whitespace was introduced,
  and `git diff --check` passes. No experiment, inference, resampling, result
  generation, or code test ran.

**What's still incomplete / next step:**
- Await the user's Batch 17 writing review. Do not start Batch 18 or create
  `report/paper_draft.md` until the user approves this framing.

**Needs the user's review before proceeding:**
- Review the scoped comparability-gap argument, the explicit post-hoc status
  and evidence mapping of H1--H5, and the lung-opacity terminology choices.

**Files touched:**
- `docs/LITERATURE_REVIEW.md`, `docs/HYPOTHESES.md`
- `docs/DATASET_CHOICE.md`, `docs/DATASHEET.md`, `docs/EXPLAINABILITY.md`,
  `docs/LIMITATIONS.md`, `docs/PARETO_ANALYSIS.md`, `docs/PROJECT_PLAN.md`,
  `docs/YOLO_BASELINE.md`
- `report/report.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 41 — 2026-08-20 — Batch 16 published to main

**What I did:**
- Confirmed restored GitHub CLI authentication from the host keyring, fetched
  live `origin/main`, and verified it had not advanced beyond `802cf75`.
- Generalized the avoidable user-specific Python path in the public Batch 16
  reproduction documentation while preserving exact generated provenance
  records.
- Staged the audited 97-file Batch 16 allowlist only. Local protocol/state
  files, orchestration logs, failed/aborted/rejected/smoke diagnostics, raw
  data, environments, and all model checkpoints remained excluded.
- Revalidated the publication snapshot: 227 tests passed with one expected
  metadata-only skip; Ruff lint and format checks passed; uv confirmed all 76
  installed packages are compatible; the staged whitespace and forbidden-path
  checks passed.
- Created commit `1876044e1333a842e8196bd7ad0a8617b0abc230`
  (`Complete five-seed clean analysis`) directly on `main` and pushed it to
  `origin/main`. A post-push fetch confirms local and remote hashes are equal
  with zero divergence.

**What's still incomplete / next step:**
- No Batch 16 publication work remains. Batch 17 may begin when requested.

**Needs the user's review before proceeding:**
- None for publication; report the successful main-branch push.

**Files touched:**
- Public Batch 16 commit: 97 files at `1876044`
- `CODEX.md`, `HANDOFF.md` (internal state; intentionally excluded from the
  public repository)

---

## Session 40 — 2026-08-20 — Batch 16 publication blocked on GitHub authentication

**What I did:**
- Audited the dirty worktree before the user-requested direct-to-`main`
  publication. The intended Batch 16 publication is one coherent 97-file
  change set: 31 tracked modifications plus 66 new code/config/result/archive
  artifacts. `git diff --check` passes, no secrets were detected, all large
  checkpoints remain ignored, and no candidate approaches GitHub's file-size
  limit.
- Identified and excluded 173 unrelated/local files: repository protocol/state
  files, Batch 5/16 orchestration logs, and historical failed, aborted,
  rejected, or smoke-run diagnostics. Nothing was staged.
- Verified local `main` and the cached `origin/main` both point to `802cf75`,
  but did not fetch because GitHub CLI authentication is invalid for the active
  `Alpha-lacrim` account. No commit or push was attempted.

**What's still incomplete / next step:**
- The user must reauthenticate with `gh auth login -h github.com`. Then fetch
  live `origin/main`, recheck divergence, generalize the avoidable local Python
  path in `docs/QUANTITATIVE_COMPARISON.md`, stage the explicit Batch 16
  allowlist, validate the staged snapshot, create one atomic commit, and push
  directly to `main` as authorized.

**Needs the user's review before proceeding:**
- Restore GitHub CLI authentication and confirm when it succeeds.

**Files touched:**
- `HANDOFF.md` only (internal state; intentionally excluded from publication)

---

## Session 39 — 2026-08-19 — Batch 16 completed n=5 clean analysis

**What I did:**
- Executed the user-approved all-attempt recovery after a fail-closed design
  audit. The evaluator now permits null IoU/Dice only when a run has zero
  fixed-threshold true positives, never coerces those values to zero, keeps all
  five attempts for AP/precision/recall/F1, and exposes defined/attempted n,
  excluded seed IDs, and reasons in every comparison layer. Regression tests
  reject every other missing/nonfinite combination.
- Kept `src/stats/paired.py` unchanged. The statistics runner builds the 323
  NIH patient clusters once, analyzes all five pairs for precision, recall,
  F1, and both AP endpoints, analyzes complete pairs `17/42/137/314` for
  conditional IoU/Dice, merges the seven predeclared rows, and applies one
  seven-endpoint Holm correction. Eligibility is config-declared and checked
  fail-closed against YOLO11s seed 271's exact TP/prediction counts. The two
  endpoint groups use separate deterministic random streams; patient-cluster
  construction/resampling and patient-label-swap algorithms are unchanged.
- Ran the unified evaluator on the locked Python 3.11 / Torch 2.6.0+cu124 GPU
  environment after one safe pre-inference stop exposed that the project
  `.venv` contains CPU-only Torch. The correct RTX 4060 run completed all ten
  checkpoints in 569.6 seconds. The Phase 5 summary is complete (SHA-256
  `09fedc40...`), all six legacy bundles remain byte-identical to their n=3
  records, and the numerical results match the independent oracle.
- Final descriptive clean values at score 0.25 include Faster R-CNN versus
  YOLO11s precision `0.1959 +/- 0.0552` versus `0.2983 +/- 0.1691`, recall
  `0.5799 +/- 0.0911` versus `0.0955 +/- 0.0607`, F1
  `0.2845 +/- 0.0528` versus `0.1427 +/- 0.0868`, AP50
  `0.3042 +/- 0.0189` versus `0.1626 +/- 0.0162`, and AP50:95
  `0.0995 +/- 0.0067` versus `0.0542 +/- 0.0060`, all n=5. Conditional
  IoU/Dice are Faster R-CNN n=5 (`0.6749/0.8010`) versus YOLO11s n=4
  (`0.6985/0.8181`); seed 271 is explicitly undefined only for those metrics.
- Ran the clean-only patient-cluster statistics in 170.6 seconds. The corrected
  Holm pattern **did not remain identical**: it strengthened from 4/7 at n=3
  to 5/7. F1 became significant (`p_Holm 0.0971805639 -> 0.0013997201`).
  Precision (`0.0019996001`), recall (`0.0013997201`), AP50
  (`0.0207958408`), and AP50:95 (`0.0341931614`) remain significant;
  conditional IoU and Dice remain non-significant (`0.1063787243` each).
- Added a hash-bound offline YOLO stability diagnostic. It confirms seed 271
  completed normally with no nonfinite or all-zero-loss epoch and best
  validation AP50:95 `0.08958`, yet its maximum test score is `0.0412735`, so
  it emits zero detections at both 0.25 and the historical n=3-selected 0.05.
  At the 0.001 COCO floor it has 962 predictions, 145 IoU-qualified matches,
  and mean matched IoU `0.673643`. It is therefore documented as operational
  confidence-score degeneracy, not classic loss/head collapse or unlucky IoU.
- Preserved the n=3 Phase 5 config, summary, three tables, corrected clean
  statistics, and exact historical threshold configs. Threshold/PR/FROC/Pareto
  remain explicitly n=3; archive-aware configs prevent silent n=5 mixing.
  Robustness and explainability were not rerun. The robustness table and Phase
  6 summary remain byte-identical at `59c9ec17...` and `4fe09e19...`.
- Reconciled README, the report, core/ancillary methodology documents, and
  limitations. Full validation passes: Ruff check, Ruff format check, 227
  tests passed with one expected metadata-only skip, all archive/protected
  hashes rechecked, and `git diff --check` passes.

**What's still incomplete / next step:**
- No Batch 16 work remains. Stop here for the user's review; do not begin Batch
  17 until the user accepts the changed Holm pattern and asymmetric n result.
- `report/paper_draft.md` does not yet exist. Batch 18 must prominently carry
  the all-attempt n=5/n=4 policy, seed-271 confidence instability, and frozen
  n=3 threshold/PR/FROC/Pareto scope into the abstract, Methods, Results,
  Discussion, and Limitations.

**Needs the user's review before proceeding:**
- Review and accept that the Holm-corrected pattern strengthened rather than
  merely holding: F1 is newly significant, producing 5/7 significant clean
  endpoints. Also review retention of seed 271 in all attempt-defined metrics,
  paired n=4 conditional localization, and the confidence-degeneracy finding.

**Files touched:**
- `src/evaluate.py`, `src/stats/run_statistics.py`,
  `src/analyze_yolo_seed_stability.py`
- `tests/test_evaluate.py`, `tests/test_statistics.py`, `tests/test_froc.py`,
  `tests/test_yolo_seed_stability.py`
- `configs/evaluation.yaml`, `configs/statistics.yaml`,
  `configs/yolo_seed_stability.yaml`, n=3 evaluation/threshold/FROC archive
  configs, and archive-routed threshold/Pareto configs
- Phase 5 n=5 comparison tables/summary/bundles, Phase 8 clean table/summary,
  YOLO stability table/summary, and explicit n=3 archives
- `README.md`, `report/report.md`, `docs/LIMITATIONS.md`,
  `docs/QUANTITATIVE_COMPARISON.md`, `docs/STATISTICAL_ANALYSIS.md`,
  `docs/ROBUSTNESS.md`, `docs/THRESHOLD_ANALYSIS.md`,
  `docs/PARETO_ANALYSIS.md`, `docs/FROC_ANALYSIS.md`,
  `docs/YOLO_BASELINE.md`, `docs/PROJECT_PLAN.md`, `CODEX.md`, `HANDOFF.md`

---

## Session 38 — 2026-08-19 — Batch 16 path recommendation

**What I did:**
- Evaluated replacement versus retention as a research-design decision. The
  seed was predeclared, completed without an objective numerical-failure
  marker, and learned normal AP/localization; replacing it solely because its
  confidence scale produced a bad endpoint would be post-selection on outcome
  and would understate the recipe's seed instability.
- Quantified the effect of retaining every attempted seed. At score 0.25,
  YOLO11s n=5 precision/recall/F1 would be
  `0.29832 +/- 0.16908`, `0.09552 +/- 0.06066`, and
  `0.14266 +/- 0.08683`, versus the n=3 values
  `0.3730/0.1356/0.1981`. AP50:95 remains stable at
  `0.05417 +/- 0.00603`. At the previously validation-frozen YOLO threshold
  0.05, seed 271 also emits nothing and the n=5 precision/recall/F1 becomes
  `0.25240 +/- 0.14179`, `0.19478 +/- 0.11095`, and
  `0.21921 +/- 0.12327`. This demonstrates calibration/operating-point
  instability rather than absent rank discrimination.
- Recommended retaining seed 271 as a valid recipe-level stochastic outcome:
  n=5 for AP and precision/recall/F1, detector-specific descriptive n=5/n=4
  for conditional localization, and four complete seed pairs for paired
  conditional IoU/Dice inference. Keep all seven clean endpoints in the Holm
  family and disclose the endpoint-specific n prominently.
- Confirmed this inferential plan can leave Batch 13's patient IDs, clustering,
  cluster bootstrap resampling, and patient-level label swaps unchanged. It
  does require an explicit endpoint-specific cross-seed eligibility/reduction
  rule in the statistics path; no such code change has yet been made.

**What's still incomplete / next step:**
- Await approval of the recommended retain-and-report path. If approved, patch
  and test only cross-seed aggregation/eligibility, rerun clean evaluation and
  patient-cluster statistics, preserve robustness/explainability, add prominent
  limitations/tables/paper-draft disclosure, and report the n=5 Holm pattern.
- If an extra seed is desired, treat the additional detector pair as a
  supplementary sixth-seed sensitivity analysis and retain seed 271 in every
  attempt-level result; do not train-until-success or silently replace it.

**Needs the user's review before proceeding:**
- Approve the recommended all-attempt analysis, or explicitly choose a
  different estimand. Nothing has been finalized and no replacement has been
  launched.

**Files touched:**
- Read-only threshold/summary diagnostics
- `CODEX.md`, `HANDOFF.md`

---

## Session 37 — 2026-08-19 — Batch 16 seed-271 validity gate

**What I did:**
- Performed the requested read-only validity diagnostic before authorizing any
  recovery. YOLO11s seed 271 produced no test detections at the frozen score
  threshold 0.25: prediction/TP/FP/FN counts were `0/0/0/268`, and fixed-point
  precision/recall/F1 were all zero. It did not produce thresholded boxes that
  merely failed the IoU>=0.5 match rule.
- Verified this is a confidence-scale pathology rather than absent ranking or
  localization. The test bundle contains 962 predictions at the COCO score
  floor 0.001, but their maximum score is only `0.0412735`. Test AP50 is
  `0.1587217` and AP50:95 is `0.0555799`, both within the other YOLO seeds'
  ranges; at the low score floor, 145 boxes match at IoU>=0.5 with mean matched
  IoU `0.673643`. The same fixed-threshold zero-output behavior occurs on
  validation, so it is not a test-only fluctuation.
- Compared all accepted YOLO training histories. Seed 271's box/class/DFL
  losses declined normally, validation mAP50:95 was nonzero from epoch 2 and
  peaked at `0.08958` at epoch 7 (highest best value among the five seeds), and
  validation-map early stopping completed cleanly at 12 epochs. This differs
  decisively from archived collapse diagnostics with zero losses/metrics or an
  explicit zero-classification-loss abort.
- Classified the result conservatively as a seed-specific confidence/output-
  score degeneracy despite normal convergence. Because it is not the user's
  authorized "normal detections but unlucky zero IoU matches" case, stopped
  without including seed 271 or implementing the n=4/n=5 recovery.
- Audited the requested recovery scope. Phase 5 aggregation can preserve null
  IoU/Dice over four finite seeds, but Batch 13's current seed-reduction loop
  would propagate the seed-271 NaN. A complete n=4 conditional inferential
  result would need metricwise finite-seed reduction there; patient clustering,
  bootstrap cluster draws, and patient-level permutation swaps would remain
  unchanged. No evaluator or statistics source was changed.

**What's still incomplete / next step:**
- Await the user's decision: train a predeclared replacement YOLO seed under
  the identical recipe, or retain seed 271 as a documented training/output-
  stability finding and decide how the clean headline comparison should be
  reported. Do not silently exclude it or finalize n=5 meanwhile.

**Needs the user's review before proceeding:**
- Choose replacement-seed training versus documented stability finding. Also
  acknowledge that a future n=4 conditional statistical result changes only
  cross-seed reduction, not patient clustering/resampling, but is not an
  evaluator-only code change.

**Files touched:**
- Diagnostic reads of YOLO summaries, epoch CSVs, shared-validation records,
  and all Phase 5 prediction bundles
- `CODEX.md`, `HANDOFF.md`

---

## Session 36 — 2026-08-19 — Batch 16 downstream failure diagnosis

**What I did:**
- Confirmed all four approved trainings completed successfully and serially
  with exit code 0: Faster R-CNN seeds 271/314 and YOLO11s seeds 271/314.
  Evaluation preflight subsequently validated all ten detector/seed contracts.
- Diagnosed the guarded downstream failure. Inference completed for all ten
  checkpoints and wrote all ten clean prediction bundles, but mean/SD
  aggregation raised `TypeError` because YOLO11s seed 271 produced zero true
  positives at the frozen fixed threshold. Its precision/recall/F1 are valid
  zeros, while matched-only IoU and Dice are correctly undefined (`null`);
  `aggregate_rows` incorrectly assumes every metric is numeric.
- Verified the documented project methodology explicitly treats conditional
  localization as undefined with no matched true positive. Also verified the
  primary clean comparison/statistics tables remain byte-identical to their
  n=3 archives, statistics never started, and the protected robustness table
  and Phase 6 summary retain hashes `59c9ec17...` and `4fe09e19...`.

**What's still incomplete / next step:**
- No retraining is required. Implement and test a methodology-preserving n=5
  aggregation policy for undefined conditional IoU/Dice, recover evaluation
  from the completed evidence where safe, then run clean-only patient-cluster
  statistics and validate all final artifacts. Do not rerun robustness or
  explainability.

**Needs the user's review before proceeding:**
- The batch is not complete. Report the stopped state and obtain authorization
  to implement the evaluator recovery; after completion, stop again for review
  of whether the corrected Holm significance pattern held at n=5.

**Files touched:**
- Diagnostic reads only under `results/logs/phase16_orchestration/` and
  `results/logs/phase5_evaluation/predictions/`
- `CODEX.md`, `HANDOFF.md`

---

## Session 35 — 2026-08-19 — Batch 16 monitored active run

**What I did:**
- Carefully audited the approved queue after the user's monitoring request.
  Faster R-CNN seed 271 completed epochs 1 and 2 with validation mAP50:95
  `0.1071597631` and `0.0984534389`, then advanced normally into epoch 3.
- Confirmed one main Python trainer plus six epoch-local DataLoader workers,
  one active training log, 100% GPU utilization, 3,128 MiB of 8,188 MiB VRAM,
  62 C, P0, and no unexpected stderr beyond the documented warn-only ROI Align
  deterministic-kernel notice. No duplicate GPU training was launched.
- Added and launched a read-only 60-second health watchdog. It records log
  age, Python process count, GPU utilization/VRAM/temperature, and both
  orchestration states; it alerts on a missing trainer, a log stale for more
  than 20 minutes, temperature at least 85 C, or training/downstream failure.
  Its first recorded state is `healthy` with no alerts. The downstream watcher
  remains correctly gated in `waiting_for_training`.

**What's still incomplete / next step:**
- The serial queue remains active on Faster R-CNN seed 271. Inspect
  `results/logs/phase16_orchestration/{health,training,downstream}_state.json`
  and the active per-job stdout/stderr before intervening. Do not launch a
  duplicate. On all four successes, validate the guarded n=5 clean evaluation
  and corrected patient-cluster statistics; on any failure, preserve partial
  evidence and diagnose the exact failed job/step.

**Needs the user's review before proceeding:**
- No decision is needed while monitoring remains healthy. Stop for review once
  the n=5 clean result is complete and explicitly report whether the corrected
  Holm-corrected n=3 significance pattern held.

**Files touched:**
- `results/logs/phase16_orchestration/watchdog.ps1`, `health_state.json`, and
  `health_samples.jsonl`
- Active seed-271 logs/checkpoints/epoch records
- `CODEX.md`, `HANDOFF.md`

---

## Session 34 — 2026-08-19 — Batch 16 approved run launch

**What I did:**
- Received explicit approval and launched one serial, fail-fast GPU queue:
  Faster R-CNN seed 271, Faster R-CNN seed 314, YOLO11s seed 271, then YOLO11s
  seed 314. Each command uses its exact approved config/gate and writes
  separate orchestration stdout/stderr plus atomic state. No GPU jobs overlap.
- Confirmed the first run is active with normal GPU utilization and progress
  through epoch-1 batch 1,100/1,750. Its only stderr content is the disclosed
  warn-only CUDA ROI Align nondeterminism notice.
- Added the n=5-compatible historical Faster R-CNN checkpoint hash check and a
  clean-only statistics mode. The latter updates only clean patient-cluster
  inference, compares n=5 Holm decisions against the corrected n=3 archive,
  and refuses to proceed if the existing robustness evidence is inconsistent.
  Ruff passes and 14 focused evaluator/statistics tests pass.
- Launched a second guarded watcher. After all four training jobs succeed, it
  will run evaluation preflight, the unified ten-checkpoint evaluator,
  statistics preflight, and `--scope clean`. It recorded the corrected
  robustness-table hash `59c9ec17...` and Phase 6 summary hash `4fe09e19...`
  and will require both to remain byte-identical.

**What's still incomplete / next step:**
- The long-running queue is active; do not launch duplicates. First inspect
  `results/logs/phase16_orchestration/training_state.json` and
  `downstream_state.json`. If either reports failure, diagnose that exact step
  without deleting partial evidence. If both report complete, validate all new
  artifacts, reconcile n=5 documentation, update CODEX/HANDOFF, and report
  whether the four-endpoint n=3 Holm pattern held.

**Needs the user's review before proceeding:**
- No decision is needed while the approved queue runs. The next review is the
  completed n=5 clean comparison and Holm-corrected significance pattern.

**Files touched:**
- `src/evaluate.py`, `src/stats/run_statistics.py`,
  `tests/test_statistics.py`, and `configs/statistics.yaml`
- `results/logs/phase16_orchestration/run_training.ps1`,
  `run_downstream.ps1`, state files, and per-step stdout/stderr
- Active/generated seed-271 training artifacts under `results/logs/` and
  `results/checkpoints/`
- `CODEX.md`, `HANDOFF.md`

---

## Session 33 — 2026-08-19 — Batch 16 preflight and timing sign-off

**What I did:**
- Selected the predeclared new seeds 271 and 314 and created exact
  seed/output-identity copies of both accepted detector configs. Extended the
  unified evaluation contract and its focused regression test from three to
  five seeds; it validates all ten detector/seed configurations as identical
  within detector except for seed and artifact identity.
- Audited the conservative timing-gate rejection caused by post-training
  source hashes. Bound reuse to the exact accepted timing-source hashes, the
  exact current source-manifest hash, and exact reviewed change/addition sets
  covering Ruff-only rewrites, downstream-only helpers, the recorded YOLO
  reporting recovery, and the valid-input-neutral cross-platform COCO path
  check. Generated four new provenance-bearing timing approvals and preserved
  the historical seed-42/137 gates byte-for-byte.
- Passed Ruff formatting/lint and three focused evaluator tests. The trainers'
  own full dataset/runtime/source/config approval checks accept all four new
  gates. Preflight reports exactly the 12 expected missing artifacts for the
  four not-yet-run trainings. GPU check: RTX 4060 Laptop, 8,188 MiB total and
  6,883 MiB free; disk check: 18.81 GiB free. No training, inference, optimizer
  step, checkpoint write, or statistics run occurred.
- Froze byte-identical n=3 archives for the comparison, mean/std, per-seed, and
  corrected patient-cluster clean-statistics tables before any primary output
  can be replaced. Archive hashes exactly match the current primary hashes.

**What's still incomplete / next step:**
- Await explicit approval, then run Faster R-CNN and YOLO11s seeds 271 and 314
  serially. Historical completed runs project 4.47 h of epoch-loop training;
  budget approximately 4.75--5 h including finalization, n=5 evaluation, and
  clean-only statistics. Historical early-stopping variation suggests roughly
  3--6 h total; the configured 30-epoch hard ceiling is about 10.94 h before
  finalization.
- After training, replace only the primary clean comparison/statistics outputs
  with n=5 results. Do not rerun robustness or explainability. Validate and
  reconcile documentation, then stop for review of the Holm pattern.

**Needs the user's review before proceeding:**
- Approve or decline the four full runs under the frozen seeds/configs and the
  approximately five-hour expected wall-clock budget. No run will start
  without explicit approval.

**Files touched:**
- `configs/evaluation.yaml`, `configs/faster_rcnn_seed271.yaml`,
  `configs/faster_rcnn_seed314.yaml`, `configs/yolo_seed271.yaml`, and
  `configs/yolo_seed314.yaml`
- `src/evaluate.py`, `tests/test_evaluate.py`
- Four new `results/logs/*_seed{271,314}_benchmark/benchmark_estimate.json`
  timing gates
- Four new `results/tables/*_n3_archive.csv` clean-table archives
- `CODEX.md`, `HANDOFF.md`

---

## Session 32 — 2026-08-19 — Batches 13--15 publication

**What I did:**
- Verified GitHub CLI authentication through the configured keyring, fetched
  `origin/main`, and confirmed local and remote `main` were synchronized at
  `48a5704` before publication.
- Staged exactly 50 Batch 13--15 paths covering patient-cluster statistics and
  archives, validation-selected thresholds and prediction bundles, FROC and
  corrected Pareto artifacts, implementation/tests/configs, and reconciled
  public documentation. Excluded `CODEX.md`, `HANDOFF.md`, static local
  protocol/spec files, orchestration output, and every failed, aborted, or
  rejected diagnostic directory.
- Audited the staged path list exactly, confirmed no unstaged tracked changes,
  checked that the largest blob was below 1 MB, scanned newly added lines for
  credential patterns, and passed `git diff --cached --check`. The publication
  retained Batch 15's successful validation: Ruff format/lint clean and 212
  tests passed with one expected metadata-only skip.
- Committed directly on `main` as `802cf75` (`Correct statistical and threshold
  methodology`) and pushed it to GitHub as requested. The push wrapper timed
  out while its Git process continued; no duplicate push was started. The
  original transfer completed, and local `HEAD`, `origin/main`, and GitHub's
  live `main` ref were then independently verified at full SHA
  `802cf75995680284ce5f34c2430155a84e6d0d7c` with zero divergence. No branch,
  PR, or tag was created.

**What's still incomplete / next step:**
- Publication is complete. Stop for review before Batch 16.

**Needs the user's review before proceeding:**
- Review commit `802cf75` on `main`, especially the corrected headline,
  validation-selected operating points, primary patient-cluster inference, and
  raw-versus-retention severe-darkness conclusion.

**Files touched:**
- Published commit `802cf75`: the audited 50-file Batch 13--15 scope described
  above.
- Local publication bookkeeping only: `CODEX.md`, `HANDOFF.md`.

---

## Session 31 — 2026-08-19 — Batch 15

**What I did:**
- Reconciled the report's methodology, quantitative results, statistics,
  Discussion, limitations, conclusion, and future-work language to the Batch
  13 patient-cluster and Batch 14 validation-threshold/FROC evidence. The
  report now treats the original score-0.25 YOLO11s precision advantage as a
  score-scale/selectivity artifact, adds the authoritative 0.69/0.05
  validation-selected test operating points, and frames the defensible
  trade-off as detection quality versus implementation-specific compute.
- Replaced the report's superseded clean CIs, Holm p-values, and jackknife
  Cohen's d with the primary patient-cluster table. Corrected the robustness
  inference: no raw AP comparison survives the 35-condition family, while the
  severe-darkness AP-retention advantage remains Holm-significant.
- Reconciled `docs/QUANTITATIVE_COMPARISON.md` and made the fixed-threshold
  caveat explicit in `docs/STATISTICAL_ANALYSIS.md`. Confirmed
  `docs/PARETO_ANALYSIS.md` uses validation-selected thresholds 0.69/0.05 and
  that `docs/FROC_ANALYSIS.md` and `docs/THRESHOLD_ANALYSIS.md` agree with the
  same frontier finding.
- Updated `docs/LIMITATIONS.md` to record the resolved Batch 13 clustering
  error, digital-not-clinical robustness, the validation-threshold limitation,
  and implementation-specific-not-architecture-general compute scope. A
  repository-wide stale-claim pass also reconciled the README headline,
  `docs/PROJECT_PLAN.md`, and `docs/ROBUSTNESS.md`.
- Read the six requested finding documents together after editing. Automated
  cross-document/artifact assertions and local-link checks pass; Ruff reports
  74 files formatted and lint-clean; 212 tests pass with one expected skip;
  `git diff --check` passes. The first test invocation encountered only a stale
  sandbox-owned pytest temp-directory cleanup error; the isolated fresh-temp
  rerun passed in full and its generated temp directory was removed.

**What's still incomplete / next step:**
- Batch 15 is complete locally and remains uncommitted together with the
  reviewed Batch 13--14 work. Stop for review before Batch 16.

**Needs the user's review before proceeding:**
- Review the new report Table 4c and revised Sections 10--12, especially the
  distinction between the statistically significant original score-0.25
  precision difference and the absence of a YOLO11s frontier advantage.
- Confirm the final headline: Faster R-CNN has stronger PR/FROC detection
  quality; YOLO11s's defensible advantage is implementation-specific compute;
  severe-darkness raw AP is descriptive while AP retention remains
  Holm-significant.

**Files touched:**
- `report/report.md`, `docs/QUANTITATIVE_COMPARISON.md`,
  `docs/STATISTICAL_ANALYSIS.md`, `docs/LIMITATIONS.md`, and a line-wrap-only
  cleanup in `docs/PARETO_ANALYSIS.md`
- `README.md`, `docs/PROJECT_PLAN.md`, and `docs/ROBUSTNESS.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 30 — 2026-08-19 — Batch 14

**What I did:**
- Corrected the test-set threshold-selection leak. The historical training
  artifacts contained validation aggregates but no raw scored detections, so I
  ran one inference-only pass of the six immutable best checkpoints over the
  750-image validation split and froze hash-bound bundles. Every bundle exactly
  reproduces its archived threshold-0.25 validation precision/recall/F1; no
  training, resume, optimizer step, checkpoint write, or test inference ran.
- Applied the unchanged 99-point Batch 10 grid to validation and predeclared
  maximum arithmetic mean validation F1 as the rule, with exact ties resolved
  toward the higher threshold. This selected 0.69 for Faster R-CNN and 0.05 for
  YOLO11s. Each was then applied exactly once to each frozen test bundle. Final
  mean test precision/recall/F1 is 0.3543/0.3607/0.3492 versus
  0.3096/0.2438/0.2718; test data did not feed back into selection.
- Rewired the Pareto recall panels to those validation-selected, test-evaluated
  rows. Faster R-CNN recall spans 0.2910--0.4030 and YOLO11s 0.2090--0.2612;
  neither detector strictly dominates any panel. The AP inputs are unchanged,
  and the original PR/F1 PNG hashes still exactly match the frozen Batch 10
  summary.
- Reparameterized the unchanged 594-row test sweep as FROC. At
  0.125/0.25/0.5/1/2 FP/image, Faster R-CNN sensitivity is
  0.2699/0.3607/0.4801/0.6032/0.6928 versus YOLO11s
  0.1803/0.2749/0.3321/0.3321/0.3321. The YOLO11s plateau is explicitly labeled
  as the threshold-0.01 sweep boundary rather than a global asymptote.
- Replaced score-calibration claims about the threshold finding with
  score-scale/selectivity language and explicitly distinguished it from
  probabilistic calibration. Added exact README reproduction commands and
  provenance summaries. Visually inspected both changed figures.
- Validation passes: 74 files Ruff-formatted, Ruff lint clean, 212 tests passed
  with one expected metadata-only skip, `git diff --check` clean, all source and
  artifact hashes/counts valid, selected test rows exactly match the existing
  frozen sweep at 0.69/0.05, and no unrelated Batch 13 work was overwritten.

**What's still incomplete / next step:**
- Batch 14 is complete locally and uncommitted. Stop for review before Batch
  15. Batch 15 must reconcile the corrected Batch 13 statistics and Batch 14
  threshold/FROC narrative throughout `report/report.md`,
  `docs/QUANTITATIVE_COMPARISON.md`, and the remaining cross-document claims.

**Needs the user's review before proceeding:**
- Review the maximum-mean-validation-F1 rule, frozen thresholds 0.69/0.05, the
  final test precision/recall/F1 table, the corrected Pareto recall panels, and
  the five FROC budgets. In particular, confirm the conservative description of
  YOLO11s's 0.01 boundary and the explicit statement that these budgets are
  clinically interpretable but not clinically validated recommendations.

**Files touched:**
- `configs/threshold_selection.yaml`, `src/evaluate_threshold_selection.py`,
  `tests/test_threshold_selection.py`, and the small optional historical-config
  guard extension in `src/evaluate.py`
- `results/tables/validation_threshold_sweep*.csv`,
  `results/tables/selected_operating_points*.csv`, and
  `results/logs/phase14_threshold_selection/`
- `configs/froc.yaml`, `src/plot_froc_curves.py`, `tests/test_froc.py`,
  `results/tables/froc_operating_points.csv`,
  `results/figures/froc_curves.png`, `results/logs/phase14_froc/`, and
  `docs/FROC_ANALYSIS.md`
- `configs/pareto.yaml`, `src/plot_pareto_frontier.py`,
  `tests/test_pareto_frontier.py`, `results/figures/pareto_frontier.png`, and
  `docs/PARETO_ANALYSIS.md`
- `docs/THRESHOLD_ANALYSIS.md`, `docs/LIMITATIONS.md`,
  `docs/PROJECT_PLAN.md`, `report/report.md`, `README.md`, `CODEX.md`, and
  `HANDOFF.md`

---

## Session 29 — 2026-08-18 — Batch 13

**What I did:**
- Replaced Phase 8 image-level resampling with the committed Batch 1 NIH
  patient groups from `data/splits/rsna-pneumonia-5000/test.csv`. The clean
  750-image set has 323 patient groups; the frozen 300-image robustness subset
  has 183. Each of the unchanged 2,000 bootstrap draws samples groups with
  replacement and includes every observed image in each selected group; each
  of the unchanged 5,000 permutations swaps detector labels once per group,
  consistently across its images, paired seeds, and clean/corrupted retention
  evidence.
- Reprocessed all six frozen Phase 5 clean bundles and all 72 Phase 6 bundles
  offline in 23m53s. No training, checkpoint loading, model inference, NMS, or
  threshold selection ran. All seven clean and 497 robustness rows now contain
  patient-cluster CIs, permutation p-values, and Holm corrections; point
  estimates remain exactly equal to the archived rows.
- Did not guess a cluster version of the old image-jackknife Cohen's d. The
  primary effect is now the paired raw aggregate difference with its
  patient-cluster bootstrap CI. The old d values remain audit-only in the exact
  image-level archives.
- Preserved the superseded clean CSV, robustness CSV, and summary under
  explicit `*_image_level_archive.*` paths. Their hashes exactly reproduce the
  former primary hashes `9eb05cf8...`, `8b766f59...`, and `064bba13...`; the
  corrected primary CSV hashes are `cdd67407...` and `59c9ec17...`.
- Updated `docs/STATISTICAL_ANALYSIS.md` with all corrected clean intervals and
  p-values, corrected selected robustness intervals and p-values, the archive
  rationale, and a complete 11-row before/after Holm-decision table. Updated
  the reproducibility/Definition-of-Done wording in README. Per instruction,
  did not touch `report/report.md` or `docs/QUANTITATIVE_COMPARISON.md`.
- Validation passes: preflight confirms 6 Phase 5 / 72 Phase 6 bundles and the
  323/183 group counts; Ruff format and lint are clean; focused statistics
  tests pass; the full CI-matching CPU-extra suite passes 208 tests with one
  expected metadata-only skip. Artifact/source/archive hashes, 7/497 row
  counts, identical point estimates, and the significance delta all pass a
  separate audit.

**What's still incomplete / next step:**
- Batch 13 is complete locally and uncommitted. Stop for review before Batch
  14. Batch 15 must later reconcile the corrected methodology/results into the
  untouched report and quantitative-comparison prose after Batch 14 lands.

**Needs the user's review before proceeding:**
- The seven clean conclusions do **not** change: four of seven endpoints remain
  Holm-significant (Faster R-CNN recall and both AP metrics; YOLO11s precision).
- The secondary robustness conclusions **do** change in both directions: six
  formerly significant rows lose significance and five gain it (88 to 87
  significant rows overall). Most importantly, darkness severity 5 raw
  mAP@0.5 changes from Holm `0.0070` to `0.1750`, and raw mAP@0.5:0.95 from
  `0.0070` to `0.0770`; neither raw AP result remains significant. Their
  retention differences remain significant at Holm `0.0070`. Review the full
  11-row delta in `docs/STATISTICAL_ANALYSIS.md` before Batch 14.

**Files touched:**
- `configs/statistics.yaml`, `src/stats/paired.py`,
  `src/stats/run_statistics.py`, `tests/test_statistics.py`
- `results/tables/statistical_clean_comparison.csv`,
  `results/tables/statistical_robustness_comparison.csv`, and both new
  `*_image_level_archive.csv` files
- `results/logs/phase8_statistics/summary.json`, `pip_freeze.txt`,
  `run_environment.json`, and `summary_image_level_archive.json`
- `docs/STATISTICAL_ANALYSIS.md`, `README.md`, `CODEX.md`, `HANDOFF.md`

---

## Session 28 — 2026-08-18 — Pre-Batch 13 housekeeping

**What I did:**
- Confirmed the red `foundation-ci` history and inspected run `32072457140`:
  both Ubuntu and Windows stopped at `ruff format --check`, which reported the
  same 36 unformatted files; lint, pytest, and smoke never ran.
- Ran the requested `uv run --locked ruff format src tests`, retained exactly
  Ruff's 36-file output, and appended D-003 to `docs/DECISION_LOG.md` with the
  accepted decision not to revive a formal Track-B-style native/best-practice
  comparison.
- Preflight uncovered a latent CI dependency/invocation issue masked by the
  formatter gate: tests added after the last green run import optional Torch,
  but CI did not install the existing `cpu` extra, and bare `pytest` did not
  expose the repository root reliably on Windows. Updated the workflow to use
  the locked CPU extra and `python -m pytest`; no gate, rule, or test was
  removed or weakened.
- Locally verified 70 files formatted, Ruff lint clean, 207 tests passed with
  one expected metadata-only skip, deterministic smoke passed, and staged diff
  checks passed. Published the 38-path scope as `9fee539` (`Restore foundation
  CI and record D-003`).
- The first post-fix run `32076606440` passed every Windows step and Ubuntu
  formatting/lint, then exposed one genuine platform-specific safety bug:
  `\\outside.png` was rooted on Windows but not rejected on POSIX. Changed the
  COCO filename validator to evaluate both `PurePosixPath` and
  `PureWindowsPath` semantics, re-ran the full local suite, and published the
  one-file follow-up as `48a5704` (`Validate COCO paths across platforms`).
- Confirmed foundation CI run `32076898917` completed with overall `success`:
  lock validation, locked CPU dependency sync, Ruff formatting, Ruff lint,
  pytest, and deterministic smoke all passed on both `ubuntu-latest` and
  `windows-latest`. No experiment, dataset, checkpoint, result, or report
  artifact was run or changed.

**What's still incomplete / next step:**
- Housekeeping is complete. Stop for the user's review before Batch 13.

**Needs the user's review before proceeding:**
- Review D-003, the explicit CPU-extra/module-form CI correction, and the
  portable COCO path-safety fix. Passing run:
  `https://github.com/Alpha-lacrim/medical-object-detector-benchmark/actions/runs/32076898917`.

**Files touched:**
- Published in `9fee539`: `.github/workflows/ci.yml`,
  `docs/DECISION_LOG.md`, and the 36 Ruff-formatted files under `src/` and
  `tests/`.
- Published in `48a5704`: `src/models/faster_rcnn_data.py`.
- Local bookkeeping only: `CODEX.md` and `HANDOFF.md`.

---

## Session 27 — 2026-08-18 — Batch 11 publication

**What I did:**
- Verified GitHub CLI authentication through the configured system keyring,
  fetched `origin/main`, and confirmed local and remote `main` were synchronized
  at `cddcaa8` before publication.
- Staged exactly the six Batch 11 publication paths: `README.md`, the Pareto
  config/renderer/test, `docs/PARETO_ANALYSIS.md`, and
  `results/figures/pareto_frontier.png`. Excluded `CODEX.md`, `HANDOFF.md`, all
  static local protocol/spec files, and every preserved diagnostic directory.
- Re-ran the publication gate on the staged snapshot: 207 tests passed, one
  expected metadata-only test skipped, Ruff passed, Pareto preflight validated
  all six seed rows and four non-dominance classifications, and the staged diff
  check passed.
- Committed the scope directly on `main` as `e33fbcb` (`Add
  accuracy-efficiency Pareto analysis`) and pushed it to `origin/main`. Local
  HEAD, `origin/main`, and the live remote ref all resolve to full SHA
  `e33fbcbd9d97e91fee75b4ffc5855ff83d0bb6f2` with zero divergence; no PR,
  temporary branch, or tag was created.

**What's still incomplete / next step:**
- Publication is complete. Stop for review before Batch 13, or Batch 12 if the
  additional-seed path is chosen first.

**Needs the user's review before proceeding:**
- Review the published Pareto figure, detector-specific recall operating points,
  and strict all-seeds dominance wording.

**Files touched:**
- Published commit `e33fbcb`: `README.md`, `configs/pareto.yaml`,
  `docs/PARETO_ANALYSIS.md`, `results/figures/pareto_frontier.png`,
  `src/plot_pareto_frontier.py`, and `tests/test_pareto_frontier.py`.
- Local publication bookkeeping only: `CODEX.md` and `HANDOFF.md`.

---

## Session 26 — 2026-08-15 — Batch 11

**What I did:**
- Added a config-driven, offline Pareto renderer that joins all six frozen
  Phase 5 seed rows to their exact compute CSVs, cross-checks the publication
  means, and consumes the completed Batch 10 threshold tables. No training,
  checkpoint loading, prediction processing, or inference ran.
- Produced the requested 2x2 `pareto_frontier.png` with every seed plotted and
  labeled. The mAP panels are threshold-independent; recall uses the best
  observed detector-level mean-F1 sweep thresholds (0.63 for Faster R-CNN and
  0.01 for YOLO11s) so the Batch 10 calibration finding is carried forward.
- Applied a conservative detector-level dominance rule requiring every seed to
  be strictly better than every alternative seed on both axes. Neither detector
  dominates in any panel: Faster R-CNN occupies the higher-AP/higher-recall
  region, while YOLO11s occupies the more efficient region on every resource
  axis.
- Wrote four scenario-conditional panel interpretations in
  `docs/PARETO_ANALYSIS.md`, added exact offline commands to `README.md`, and
  visually verified the final figure. Full validation passes: 207 tests, one
  expected metadata-only skip, Ruff clean, and `git diff --check` clean for the
  Batch 11 implementation/documentation paths.

**What's still incomplete / next step:**
- Batch 11 implementation is complete and remains local/uncommitted. Do not
  start Batch 13, or Batch 12 if the extra seeds are chosen first, until the
  user reviews the Pareto figure and threshold-aware recall choice.

**Needs the user's review before proceeding:**
- Review whether the detector-specific peak mean-F1 thresholds are the desired
  recall operating points and whether the strict all-seeds dominance wording is
  suitable for the paper-track framing.

**Files touched:**
- `configs/pareto.yaml`
- `src/plot_pareto_frontier.py`
- `tests/test_pareto_frontier.py`
- `results/figures/pareto_frontier.png`
- `docs/PARETO_ANALYSIS.md`
- `README.md`
- `CODEX.md`
- `HANDOFF.md`

---

## Session 25 — 2026-08-14 — Batch 10 publication

**What I did:**
- Confirmed `main` and `origin/main` were synchronized at `caeee58`, staged
  exactly the 15 Batch 10 code/config/test/doc/result paths, and excluded
  `CODEX.md`, `HANDOFF.md`, all other local protocol files, and every preserved
  diagnostic directory under the established publication convention.
- Verified the staged file list matched the intended list exactly and the
  staged diff check passed, then committed the scope as `cddcaa8` (`Add
  threshold sweep analysis`).
- Published the commit directly to `origin/main` as requested. A temporary
  branch and draft PR #3 had completed just before the user's direct-main
  correction arrived; PR #3 was immediately closed, and both the local and
  remote temporary branch refs were deleted after `main` was fast-forwarded.
- Final remote verification shows local `main` and `origin/main` both at full
  SHA `cddcaa893262119fdeb3c550c02758322fe1da82`, no temporary remote branch,
  and PR #3 closed. No tag or unrelated worktree content changed.

**What's still incomplete / next step:**
- Publication is complete. Stop for review of the Batch 10 PR/F1 curves before
  Batch 11.

**Needs the user's review before proceeding:**
- Confirm the direct-main publication and review the Batch 10 curve conclusions
  recorded in Session 24.

**Files touched:**
- Published commit `cddcaa8`: the 15 Batch 10 paths listed in Session 24,
  excluding local-only `CODEX.md` and `HANDOFF.md`.
- Local publication bookkeeping only: `CODEX.md` and `HANDOFF.md`.
- Remote bookkeeping: closed PR #3; deleted temporary local/remote branch
  `agent/batch-10-threshold-analysis`.

---

## Session 24 — 2026-08-14 — Batch 10

**What I did:**
- Added a strict, config-driven offline threshold pipeline over all six frozen
  Phase 5 clean prediction bundles. Preflight verifies the Phase 5 config,
  annotation, bundle hashes, detector/seed grid, and evaluator identities. The
  run performs no training, checkpoint loading, NMS, or inference.
- Swept 99 thresholds from 0.01 through 0.99 using the existing canonical
  `evaluate_operating_point` matcher at IoU 0.50/max 100 detections. Threshold
  0.25 precision/recall/F1 and pycocotools AP@0.5/AP@0.5:0.95 reproduce all six
  frozen bundle values within absolute tolerance `5e-12`.
- Exposed pycocotools' official 101-point precision tensor for IoU 0.50 and the
  mean across IoU 0.50:0.95, then produced aggregate mean ± sample-SD and
  per-seed CSVs, fixed-target results, a two-panel PR figure, an F1-threshold
  figure, and a hash/provenance summary.
- Found that threshold 0.25 materially exaggerates YOLO11s's low recall: its
  mean recall rises from 0.1356 to 0.3321 and F1 from 0.1981 to 0.2840 at
  threshold 0.01. Nevertheless, the official AP@0.5 curve has higher Faster
  R-CNN precision at 96/101 recall positions, five ties, and no YOLO11s-higher
  positions. At fixed mean precision 0.50, recall is 0.2189 versus 0.0759; at
  fixed mean recall 0.30, precision is 0.4208 versus 0.2547. YOLO11s cannot
  reach mean recall 0.50 in the sweep.
- Documented the mixed result plainly in `docs/THRESHOLD_ANALYSIS.md`: the
  same-threshold precision gap is primarily score calibration/selectivity, not
  a superior YOLO11s PR regime, while its lower attainable recall persists.
  Added the exact offline reproduction command to `README.md` and updated
  `CODEX.md`.
- Verified both figures visually. Full repository validation passes: 203 tests,
  one expected metadata-only skip, Ruff clean, `git diff --check` clean, all
  seven output hashes match the summary, and CSV row counts are exactly
  198 aggregate/594 per-seed threshold rows, 404 aggregate/1,212 per-seed PR
  rows, and eight fixed-target rows.

**What's still incomplete / next step:**
- Batch 10 implementation is complete. Do not start Batch 11 Pareto analysis
  until the user reviews the PR curves and accepts the calibrated wording of
  the operating-regime claim.

**Needs the user's review before proceeding:**
- Review `results/figures/precision_recall_curves.png`,
  `results/figures/f1_vs_threshold.png`, the non-interpolated fixed-target
  definitions/results in `results/tables/threshold_operating_targets.csv`, and
  the conclusion in `docs/THRESHOLD_ANALYSIS.md`.

**Files touched:**
- `configs/threshold_sweep.yaml`
- `src/evaluate_threshold_sweep.py`
- `src/meddet_benchmark/coco_evaluation.py`
- `tests/test_threshold_sweep.py`, `tests/test_coco_evaluation.py`
- `results/tables/threshold_sweep.csv`
- `results/tables/threshold_sweep_per_seed.csv`
- `results/tables/precision_recall_curves.csv`
- `results/tables/precision_recall_curves_per_seed.csv`
- `results/tables/threshold_operating_targets.csv`
- `results/figures/precision_recall_curves.png`
- `results/figures/f1_vs_threshold.png`
- `results/logs/phase10_threshold_sweep/summary.json`
- `docs/THRESHOLD_ANALYSIS.md`, `README.md`, `CODEX.md`, `HANDOFF.md`

---

## Session 23 — 2026-08-14 — Batch 9 publication

**What I did:**
- Verified GitHub CLI authentication through the configured system keyring and
  fetched `origin/main`. The remote was exactly two commits ahead, both changing
  only `docs/PROJECT_PLAN.md`.
- Fast-forwarded local `main` from `3a38088` to remote tip `d07f193`, thereby
  accepting GitHub's `docs/PROJECT_PLAN.md` byte-for-byte. No local Batch 9 file
  overlapped that remote change.
- Staged exactly `docs/DECISION_LOG.md` and
  `docs/QUANTITATIVE_COMPARISON.md`. Excluded `CODEX.md`, `HANDOFF.md`, all
  pre-existing untracked protocol files, and every diagnostic run directory in
  accordance with the established publication convention.
- Verified the staged diff had only the two intended files, `git diff
  --cached --check` passed, `docs/PROJECT_PLAN.md` matched `origin/main`, and
  no `results/` or `report/` file differed.
- Committed the Batch 9 deliverables as `caeee58` (`Document course submission
  freeze`) and pushed `main` directly to GitHub. Also pushed the annotated
  `v1-course-submission` tag, which remains anchored to course-deliverable HEAD
  `3a3808841795938a296d48ae3b379b0d10ef3d48`.

**What's still incomplete / next step:**
- Publication is complete. Do not start Batch 10 until the user reviews Batch
  9 and explicitly approves proceeding.

**Needs the user's review before proceeding:**
- Confirm the GitHub commit/tag and the cloud-plan/local-document reconciliation
  are the intended frozen starting point for the research-paper track.

**Files touched:**
- Remote commit `caeee58`: `docs/DECISION_LOG.md` and
  `docs/QUANTITATIVE_COMPARISON.md`
- Accepted from existing remote commits: `docs/PROJECT_PLAN.md`
- Local publication bookkeeping only: `CODEX.md` and `HANDOFF.md`
- Remote Git metadata: branch `main` and tag `v1-course-submission`

---

## Session 22 — 2026-08-14 — Batch 9

**What I did:**
- Created annotated tag `v1-course-submission` on the explicitly requested
  current HEAD `3a3808841795938a296d48ae3b379b0d10ef3d48`. The annotation
  names the two detectors, unified evaluator, robustness, Grad-CAM,
  statistics, and 12-section report. No history, branch, existing tag, result,
  or worktree content was rewritten.
- Created `docs/DECISION_LOG.md`. D-001 transparently reconstructs the original
  two-track proposal from the reconciled delivered architecture and states
  that only Track A was implemented. Append-only D-002 supersedes only Track B,
  descoping it for the RTX 4060 Laptop/16 GB RAM hardware and time budget and
  reframing the paper around controlled comparison plus multi-axis trade-off
  analysis rather than best-effort leaderboard performance.
- Audited the frozen 450.7637248 versus 21.4198784 GFLOP values against the
  exact PyTorch 2.6 profiler and official Torchvision/Ultralytics reference
  magnitudes. A read-only replay reproduced both totals. Faster R-CNN used
  exactly 1,000 post-NMS proposals for the profiled validation image and split
  into 113.1282432 G backbone/FPN, 80.7138816 G RPN, and 256.9216000 G RoI
  heads; YOLO11s reproduced 21.4198784 G. The counting registry, executed ops,
  exclusions, proposal dependence, and cross-source comparability limits are
  now recorded in `docs/QUANTITATIVE_COMPARISON.md` without changing either
  result CSV.
- Updated `CODEX.md` with the tag and D-002 decisions and the Batch 9 phase
  state. Documentation/diff checks pass, and the tag resolves to the intended
  commit as an annotated tag object.

**What's still incomplete / next step:**
- Batch 9 has no remaining implementation work. Do not start Batch 10 until the
  user reviews this housekeeping checkpoint.

**Needs the user's review before proceeding:**
- Review the retrospective wording of D-001, the formal D-002 scope resolution,
  and the GFLOP methodology/sanity-check interpretation.

**Files touched:**
- `docs/DECISION_LOG.md`
- `docs/QUANTITATIVE_COMPARISON.md`
- `CODEX.md`
- `HANDOFF.md`
- Git metadata only: annotated tag `v1-course-submission`

---

## Session 21 — 2026-08-12 — Publish reconciled V1 planning record

**What I did:**
- Revalidated GitHub CLI/keyring authentication outside the sandbox and
  confirmed remote `main` matched local HEAD `df81f71` before publication.
- Staged exactly `docs/PROJECT_PLAN.md` and the existing deletion of
  `Final project 1405.v1.pdf`; verified that no agent-related or diagnostic
  untracked files entered the index and that `git diff --cached --check` passed.
- Committed the reviewed snapshot as `3a38088` (`Reconcile V1 project plan`)
  and pushed `main` to `origin`.
- Verified local HEAD and remote `main` both resolve to
  `3a3808841795938a296d48ae3b379b0d10ef3d48` after the push.
- Did not create a GitHub release or change any tag. Existing `v1.0.0` still
  resolves to `df81f7139a8f98e56fe989c7e85dc56e08b238c2`.

**What's still incomplete / next step:**
- Wait for the user's explicit instruction before creating the V1 GitHub
  release or taking any tag-related action.

**Needs the user's review before proceeding:**
- Confirm the intended release/tag treatment now that `main` is one commit
  ahead of the existing immutable `v1.0.0` tag.

**Files touched:**
- `docs/PROJECT_PLAN.md` and `Final project 1405.v1.pdf` in published commit
  `3a38088`.
- `CODEX.md` and `HANDOFF.md` locally only; both remain untracked.

---

## Session 20 — 2026-08-12 — PROJECT_PLAN reconciliation (Steps 1–2)

**What I did:**
- Step 1 audited every section (§1–§15) of the pre-implementation
  `docs/PROJECT_PLAN.md` against the selected-dataset evidence, dataset audit,
  delivered configs and repository layout, limitations, final report, and
  generated results. The audit found that all sections required at least scoped
  reconciliation: §4 and §13 contradicted the delivered project throughout,
  while the other sections retained useful methodology but contained stale
  dataset, model, two-track, pending-work, metric, or artifact-path claims.
- Step 2 reconciled `docs/PROJECT_PLAN.md` without changing any experiment,
  config, result, report, or other research artifact. A Reconciliation Note now
  preserves the brain-tumor candidate history, records RSNA as the selected
  dataset, replaces the YOLO26-primary plan with delivered YOLO11s, and states
  that Track B was descoped/not run.
- Verified that the current repository has no `docs/DECISION_LOG.md` and no
  D-002 entry. The plan therefore says that the YOLO switch and Track B
  descoping are not separately logged rather than fabricating a rationale or a
  link. The only model-selection rationale quoted is the one present in
  `configs/yolo.yaml`.
- Reconciled the actual RSNA cohort/split, training differences, evaluation,
  compute, robustness, Grad-CAM, and paired-statistics protocols; converted §12
  to a milestone status table; replaced §13 with a fresh repository inventory;
  and matched §14 to the report's actual 12 top-level sections.
- Read the reconciled plan end to end, checked its claims against
  `docs/DATASET_CHOICE.md`, `docs/DATASHEET.md`, `docs/LIMITATIONS.md`, the
  frozen configs, analysis documents, and `report/report.md`; all 28 local
  Markdown links resolve and `git diff --check` passes for the plan.
- Updated the `CODEX.md` File Map to identify the reconciliation and its
  explicit missing-decision-log/Track-B scope.

**What's still incomplete / next step:**
- No implementation work is pending from this task. Await the user's review of
  the reconciled planning record before any further documentation change.

**Needs the user's review before proceeding:**
- Review the transparent treatment of the absent D-002/decision log, the
  milestone status classifications, and the preserved superseded material in
  §§4–5.

**Files touched:**
- `docs/PROJECT_PLAN.md`
- `CODEX.md`
- `HANDOFF.md`

---

## Session 19 — 2026-08-11 — Batch 8 post-completion evaluation

**What I did:**
- Evaluated the completed repository against `PROJECT_SPEC.md`, the current
  project state, the report, limitations, reproduction guide, committed
  evidence, source layout, tests, and Git state without advancing into a new
  implementation batch.
- Re-ran the documented verification with the recorded Python 3.11 benchmark
  environment: 198 tests passed, the one metadata-only test skipped as
  intended, Ruff passed, and `git diff --check` passed.
- Assessed the work as a strong, unusually auditable course research project:
  all 12 report sections and Definition of Done categories are present, the
  patient-safe split and unified evaluator are well defended, 78 prediction
  bundles and the statistical/robustness/explainability artifacts are
  committed, and the report states weak absolute performance and clinical
  limitations honestly.
- Identified the principal scientific limitations: one uncalibrated common
  score threshold drives the precision/recall operating-point comparison;
  image-level rather than patient-cluster resampling can understate uncertainty;
  only three training seeds and one primary checkpoint for robustness/Grad-CAM
  limit generality; detector-specific optimization and timing asymmetries weaken
  architecture-only attribution; and both models' absolute accuracy and
  Grad-CAM localization remain low.
- Identified optional repository-readiness improvements: the README installs
  15 pinned direct requirements without consuming the committed `uv.lock`,
  while the recorded environment contains 153 packages; the current local
  `.venv` is stale Python 3.13 rather than the validated Python 3.11; the local
  `main` ref is behind the already-updated remote; and 22 preserved user or
  diagnostic worktree entries make the current checkout non-clean.

**What's still incomplete / next step:**
- No required project work remains. Optional highest-value follow-ups are a
  validation-only threshold-calibration/sensitivity analysis, patient-cluster
  bootstrap, a locked clean-environment reproduction path, and external-site or
  full-cohort validation.

**Needs the user's review before proceeding:**
- Review the evaluation and choose whether any optional strengthening work
  should become a new explicitly scoped batch; no such work was started.

**Files touched:**
- `HANDOFF.md` only.

---

## Session 18 — 2026-08-11 — Batch 8 report assembly and wrap-up

**What I did:**
- Assembled `report/report.md` from the frozen Phase 1–8 evidence without
  running training, inference, robustness, Grad-CAM, or statistical
  recomputation. The report follows all 12 required sections and links every
  displayed result table and figure to its committed source artifact.
- Grounded the Discussion in the measured operating-point, AP, corruption,
  Grad-CAM, latency, model-size, parameter, and FLOP trade-offs. It recommends
  Faster R-CNN for accuracy-/recall-sensitive GPU-backed screening, treats
  YOLO11s as a conditional option for compute-constrained human-in-the-loop
  assistance, rejects autonomous use for both, and includes the prospective
  clinical/regulatory validation scope boundary.
- Reorganized all accumulated entries in `docs/LIMITATIONS.md` into one
  coherent statement covering dataset/annotation scope, hardware and sampling,
  detector asymmetries, metric/compute measurement, robustness,
  explainability, statistics, deployment, and regulation.
- Rewrote `README.md` as an ordered clean-checkout reproduction guide. It maps
  Tables 1–7 and Figures 1–9 to the exact generating command and source
  artifact, and records an explicit eight-item Definition of Done audit with
  every item satisfied.
- Independently checked the completion evidence: exactly 12 report sections;
  two detector arms and 14 comparison metrics; four corruption families,
  seven types, five severities each, and 72 result rows; 111 Grad-CAM targets
  per detector and three qualitative figure categories; seven clean and 497
  robustness statistical rows with the expected four non-estimable rows.
  All local Markdown links resolve, all 17 citation keys exist in
  `report/references.bib`, and all nine documented CLI entry points accept the
  stated arguments. Final validation is 198 tests passed, one expected
  metadata-only skip, Ruff clean, and `git diff --check` clean.
- Updated `CODEX.md` to mark Batch 8, Phases 9–12, all eight Definition of Done
  requirements, and the overall project complete. The existing user-owned PDF
  deletion and excluded failed/aborted/rejected diagnostic directories were
  preserved untouched.

**What's still incomplete / next step:**
- No required experiment, report, documentation, or validation work remains.
  The Batch 8 changes are local and have not been committed or pushed; review
  and publication are the only optional next steps.

**Needs the user's review before proceeding:**
- Review the scenario-specific recommendations and scope-of-claims language in
  `report/report.md`, the consolidated `docs/LIMITATIONS.md`, and the
  clean-checkout command sequence/Definition of Done audit in `README.md`.

**Files touched:**
- `report/report.md`, `docs/LIMITATIONS.md`, `README.md`, `CODEX.md`, and
  `HANDOFF.md`.

---

## Session 17 — 2026-08-11 — Main-branch promotion

**What I did:**
- Confirmed that GitHub's default branch is `main` and fetched the current
  `main` and `agent/implementation-foundation` refs over authenticated HTTPS.
- Verified that `main` at `317dca0` was an ancestor of the agent branch and was
  exactly 14 commits behind with no divergent commits.
- Fast-forwarded `main` to the accepted Batch 4–7 publication tip `f0e0f00`
  without a force-push, history rewrite, merge commit, or working-tree change.
  This bookkeeping record is synchronized to both branch refs so `main` and
  `agent/implementation-foundation` remain aligned.

**What's still incomplete / next step:**
- Branch promotion is complete. Stop before Batch 8; report assembly remains
  gated on the user's Batch 7 statistical-results review.

**Needs the user's review before proceeding:**
- Review the Batch 7 statistical outputs and conclusions recorded in Session 15.

**Files touched:**
- Publication bookkeeping only: `CODEX.md` and `HANDOFF.md`.

---

## Session 16 — 2026-08-11 — Batches 4–7 publication

**What I did:**
- Revalidated the GitHub account and the repository's HTTPS remote, then audited
  the accepted Batch 4–7 publication scope before staging. The user-owned
  deletion of `Final project 1405.v1.pdf`, all failed/aborted/rejected diagnostic
  run directories, and `results/logs/phase5_orchestration/` remained unstaged.
- Removed seven trailing blank lines caught by the staged diff check, then
  validated the exact publication source with the documented benchmark
  interpreter: 198 tests passed, one metadata-only test skipped as expected,
  Ruff passed, and `git diff --cached --check` passed.
- Committed the 192-file accepted scope as `99ee5e3` (`Complete evaluation
  robustness explainability and statistics`) and pushed it over HTTPS to
  `origin/agent/implementation-foundation`.

**What's still incomplete / next step:**
- Publication of the completed Batch 4–7 work is complete. Stop here; Batch 8
  report assembly remains gated on the user's Batch 7 statistical-results
  review.

**Needs the user's review before proceeding:**
- Review the Batch 7 statistical tables and interpretation listed in Session 15.
  Do not begin Batch 8 until the user explicitly approves proceeding.

**Files touched:**
- Publication bookkeeping: `CODEX.md` and `HANDOFF.md`.
- Published scope: the 192 files recorded by commit `99ee5e3`.

---

## Session 15 — 2026-08-11 — Batch 7 statistical analysis

**What I did:**
- Implemented a strict, config-driven Phase 8 statistical pipeline over the six
  frozen Phase 5 and 72 frozen Phase 6 prediction bundles. It reconstructs the
  exact operating-point metrics and COCO AP from per-image sufficient evidence,
  including weighted score-ordered matches at all ten COCO IoU thresholds; all
  78 original bundle metrics reproduce within absolute tolerance `5e-12`.
- Ran 2,000 paired percentile-bootstrap draws and 5,000 two-sided paired
  image-label permutations for every comparison. The 750-image clean analysis
  also resamples the three paired training seeds. The 300-image corruption
  analysis jointly resamples clean/corrupted evidence and reports both raw and
  clean-relative estimands. Every estimable row includes both detector CIs, a
  Faster-minus-YOLO difference CI, raw and Holm-adjusted p-values, and paired
  leave-one-image-out jackknife Cohen's d.
- Applied Holm correction across the seven clean endpoints and separately for
  every metric/estimand family across the 35 corruption conditions. Conditional
  IoU/Dice have 34 estimable grid tests because YOLO has no true positive under
  darkness severity 5; the four affected raw/retention rows are explicitly
  marked not estimable. McNemar is not forced onto the detector outputs because
  they do not provide one independent binary decision per image.
- On the clean three-seed comparison, Faster R-CNN retains corrected evidence
  for higher recall (difference 0.50249, 95% CI 0.43005 to 0.57596, Holm p
  0.00140), mAP@0.5 (0.14413, 0.09673 to 0.19488, Holm p 0.00140), and
  mAP@0.5:0.95 (0.04737, 0.03105 to 0.06833, Holm p 0.00160); YOLO retains
  higher precision (-0.21045, -0.30500 to -0.11773, Holm p 0.00140). F1 and
  conditional IoU/Dice do not cross the corrected 0.05 threshold.
- Across the corruption grid, darkness severity 5 is the only AP comparison to
  survive family-wise correction. Faster-minus-YOLO mAP@0.5:0.95 is 0.11565
  (0.07122 to 0.17024, Holm p 0.00700, d 0.241) in raw performance and 0.70287
  (0.39816 to 0.84242, Holm p 0.00700, d 0.433) in clean-relative retention.
  All exact condition/metric rows remain in the publication CSV.
- Independently audited all seven clean and 497 robustness rows against the
  Phase 5/6 point estimates, recomputed every Holm family, checked finite CIs,
  p-values/effects and valid-resample counts for all 493 estimable robustness
  rows, verified the exact four null rows, and validated all source/input/output
  hashes. Final clean/robustness/summary SHA-256 values are
  `9eb05cf8df7e26e237d8bf7b0c5eb85cdc1130477db46213e125217b692d819f`,
  `8b766f59b3e69aa7d011f2a8ac5499636cbf30ed7eb943b3c1ad8c753a49940b`,
  and `064bba1317195e749562f29a8fa089ac59a40957b6205f51d27c65babc3c3937`.
  Repository checks pass: 198 tests, one expected metadata-only skip, Ruff
  clean, and `git diff --check` clean.

**What's still incomplete / next step:**
- Batch 7 is complete. Do not start Batch 8 report assembly until the user
  reviews the statistical estimands, correction families, results, effect-size
  definition, and McNemar non-applicability decision.

**Needs the user's review before proceeding:**
- Review `results/tables/statistical_clean_comparison.csv`,
  `results/tables/statistical_robustness_comparison.csv`, and
  `docs/STATISTICAL_ANALYSIS.md`. In particular, confirm the pointwise-versus-
  family-wise interpretation, the conclusion that clean AP/recall and precision
  differences survive correction while F1/localization do not, and the result
  that only severe darkness survives the grid-wide AP correction.

**Files touched:**
- Config/code/tests: `configs/statistics.yaml`, `src/stats/__init__.py`,
  `src/stats/paired.py`, `src/stats/run_statistics.py`, and
  `tests/test_statistics.py`.
- Documentation/state: `README.md`, `docs/STATISTICAL_ANALYSIS.md`,
  `docs/QUANTITATIVE_COMPARISON.md`, `docs/ROBUSTNESS.md`,
  `docs/LIMITATIONS.md`, `CODEX.md`, and `HANDOFF.md`.
- Generated artifacts: `results/logs/phase8_statistics/` and
  `results/tables/statistical_{clean,robustness}_comparison.csv`.
- All pre-existing Batch 4–6 changes, diagnostic directories, ignored
  checkpoints, and the user-owned deletion of `Final project 1405.v1.pdf` were
  preserved.

---

## Session 14 — 2026-08-11 — Batch 6 explainability analysis

**What I did:**
- Implemented a strict, config-driven Phase 7 Grad-CAM pipeline for the two
  frozen seed-17 checkpoints. The comparable hooks are the 40 by 40 stride-16
  ResNet-50 `backbone.body.layer3` output before FPN and YOLO11s `model.6`
  output before its stride-32 stage/PAN neck. Both target the differentiable
  post-activation foreground probability of a low-threshold, post-NMS retained
  candidate; false negatives are explicitly labeled annotation-guided proxy
  targets rather than ordinary emitted detections.
- Reused the exact committed Phase 6 sample manifest, SHA-256
  `63b4dd706dc2fcd8a528a935957ccb318ed2cde51a6fd87d20feca348d00fc5e`.
  Quantitative energy-in-box and pointing-game metrics cover every one of its
  111 boxes across 68 positive images for both detectors (222 target records).
  The 232 box-negative images are correctly excluded from box metrics but
  remain available for qualitative false-positive cases.
- Completed the focused nine-test suite and two-positive-image-per-model GPU
  smoke. The full pass produced 110/111 valid Faster R-CNN maps with one
  explicit zero-energy map, and 111/111 valid YOLO maps. Faster R-CNN versus
  YOLO11s mean energy-in-box is 0.08689 versus 0.09749, compared with mean box-
  area references 0.07129 versus 0.07178; pointing accuracy is 0.10909 versus
  0.12613. On 110 paired valid targets, YOLO has higher energy in 76 and Faster
  R-CNN in 34; mean Faster-minus-YOLO energy is -0.00910.
- Selected heatmap cases from frozen predictions before consulting CAM values:
  three unique shared high-IoU true positives, three shared false positives on
  box-negative `No Lung Opacity / Not Normal` images, and three unique shared
  false negatives at proxy-IoU quantiles 0.2/0.5/0.8. Generated and visually
  inspected all three side-by-side figures. Faster R-CNN is generally more
  clustered but not reliably lesion-centered; YOLO is more diffuse/punctate
  with a small energy advantage. Both often emphasize non-box anatomy, borders,
  markers, and devices, and neither supports a clinical-reasoning claim.
- Documented the exact target/layer definitions, quantitative results, explicit
  where-is-it-looking answer, objective case rubric, box/proxy/seed caveats,
  and the CUDA ROI Align backward deterministic-warn-only limitation. The
  independent audit verified 222 target rows, 18 qualitative rows, all pairings,
  all recomputed means/pointing rates, six artifact hashes, and six source
  hashes. Final summary SHA-256:
  `2b8d2d5835c113e8dc24af9eecbece62571cc5bcc689bcb245c1f74e1c23a848`.
  Repository checks pass: 190 tests, one expected metadata-only skip, Ruff
  clean, and `git diff --check` clean.

**What's still incomplete / next step:**
- Batch 6 is complete. Do not start Batch 7 statistical analysis until the user
  reviews and accepts the Grad-CAM target/layer choice, proxy-target caveat,
  energy/pointing results, and paired qualitative interpretation.

**Needs the user's review before proceeding:**
- Review `results/figures/gradcam_good_predictions.png`,
  `results/figures/gradcam_bad_predictions.png`,
  `results/figures/gradcam_failure_cases.png`,
  `results/tables/gradcam_localization_summary.csv`, and
  `docs/EXPLAINABILITY.md`. In particular, confirm the conclusion that both
  models are weakly lesion-focused, YOLO's modest energy advantage coexists
  with more diffuse maps, and false-negative proxies are conditional diagnostics
  rather than explanations of emitted detections.

**Files touched:**
- Config/code/tests: `configs/explainability.yaml`, `src/explainability/`,
  `tests/test_gradcam.py`, `tests/test_pointing_game.py`, and
  `tests/test_explainability.py`.
- Documentation/state: `README.md`, `docs/EXPLAINABILITY.md`,
  `docs/LIMITATIONS.md`, `CODEX.md`, and `HANDOFF.md`.
- Generated artifacts: `results/logs/phase7_explainability/`,
  `results/tables/gradcam*.csv`, and `results/figures/gradcam*.png`; checkpoints
  remain ignored.
- All pre-existing Batch 4/5 changes, diagnostic directories, and the user-owned
  deletion of `Final project 1405.v1.pdf` were preserved.

---

## Session 13 — 2026-08-11 — Batch 5 robustness evaluation

**What I did:**
- Drew the fixed seed-17 proportional stratified robustness sample from the
  750-image held-out test manifest using largest-remainder allocation and one
  NumPy PCG64 generator. The 300-image result is 68 Lung Opacity, 132 No Lung
  Opacity / Not Normal, and 100 Normal images, with 68 positives, 232 negatives,
  111 boxes, and 183 patients. Committed manifest SHA-256:
  `63b4dd706dc2fcd8a528a935957ccb318ed2cde51a6fd87d20feca348d00fc5e`.
- Replaced the deferred corruption placeholder with a strict, config-driven
  Albumentations 2.0.8 pipeline. Darker/brighter lighting, Gaussian and salt-
  pepper noise, Gaussian and motion blur, and JPEG compression each have five
  ordered severities; JPEG spans qualities 90/70/50/35/20. Geometry is
  preserved, stochastic transforms derive a per-image/condition seed, and both
  detectors receive identical corrupted pixels.
- Implemented the resumable Phase 6 runner. It validates the frozen Phase 5
  thresholds, primary seed-17 configs/checkpoints, clean prediction provenance,
  canonical test annotations, and sample identities before CUDA. Each condition
  writes an atomic hashed prediction bundle immediately. Clean subset metrics
  reuse exact filtered Phase 5 predictions; all 70 corrupted detector
  conditions were inferred afresh.
- Completed the full 21,000-corrupted-image grid in 2,073.4 seconds end to end.
  Faster R-CNN and YOLO11s each have 36 clean/corrupted bundles with 300 records
  apiece. Across the 35 corruptions, Faster R-CNN versus YOLO11s mean raw
  mAP@0.5:0.95 is 0.11290 versus 0.05410 and mean clean-relative retention is
  0.76385 versus 0.70908 (23.62% versus 29.09% degradation). Faster R-CNN has
  higher raw mAP in all 36 matched conditions. Salt-and-pepper severity 5 is
  worst for both; YOLO11s also collapses at the darkest operating point.
- Generated raw/relative per-type figures, wide and tidy metric tables, and
  family-mean severity curves. Documented the exact method, results, null
  conditional IoU/Dice edge case, digital-corruption limits, primary-seed
  scope, image-level sampling, repeated patients, and 111-box sample size.
- Independently audited all 72 table rows/bundles, hashes, 300-prediction
  counts, matched condition sets, severity counts, sample allocation, and every
  defined corrupted/clean ratio. Final summary SHA-256:
  `4fe09e19bc7b7d620ab9e6a3785ecae0bb2ef16cb517fce1bc287b2de2fafb2b`.
  Repository checks pass: 181 tests, one expected metadata-only skip, Ruff
  clean, and `git diff --check` clean. Both figures were visually inspected.

**What's still incomplete / next step:**
- Batch 5 is complete. Do not start Batch 6 until the user reviews and accepts
  the robustness sample, severity grid, raw/relative curves, and limitations.
- After approval, Batch 6 should reuse
  `data/splits/rsna-pneumonia-5000/test_robustness_seed17_n300.csv` for the
  quantitative pointing-game metric, as required by the existing scope.

**Needs the user's review before proceeding:**
- Review `results/figures/robustness_map_50_95_raw.png`,
  `results/figures/robustness_map_50_95_relative.png`,
  `results/tables/robustness_results.csv`, and `docs/ROBUSTNESS.md`. In
  particular, confirm the 300-image image-stratified sample, five severity
  values, clean-relative definition, primary-seed scope, and interpretation of
  the severe salt-and-pepper/darkness failures before Batch 6.

**Files touched:**
- Config/code/tests: `configs/corruptions.yaml`,
  `src/meddet_benchmark/corruptions.py`, `src/robustness/`,
  `tests/test_corruptions.py`, and `tests/test_robustness.py`.
- Documentation/state: `README.md`, `docs/ROBUSTNESS.md`,
  `docs/LIMITATIONS.md`, `CODEX.md`, and `HANDOFF.md`.
- Generated accepted artifacts: the committed 300-image sample manifest,
  `results/logs/phase6_robustness/`, `results/tables/robustness*.csv`, and
  `results/figures/robustness*.png`; checkpoints remain ignored.
- All pre-existing Batch 4 changes, diagnostic directories, and the user-owned
  deletion of `Final project 1405.v1.pdf` were preserved.

---

## Session 12 — 2026-08-10 — Batch 4 unified comparison

**What I did:**
- Fixed the Phase 5 seed grid at 17, 42, and 137 and added seed-only Faster
  R-CNN/YOLO11s configs plus `configs/evaluation.yaml`. Contract validation
  confirms that model/data/optimizer/runtime/evaluation hyperparameters are
  identical within each detector except for seed and artifact identity.
- Derived provenance-bearing seed-42/137 timing approvals from the accepted
  seed-17 gates. A redundant Faster R-CNN seed-42 benchmark was briefly started
  and stopped before any epoch/checkpoint after the unchanged shape/memory
  contract was confirmed; its incomplete log is preserved under
  `faster_rcnn_rsna_seed42_benchmark_aborted-redundant-timing` and excluded.
- Completed four additional full trainings without test access. Faster R-CNN
  seed 42 stopped after 14 epochs (best 9, 8,278.03 s, 1,557.08 MiB) and seed
  137 after 8 epochs (best 2, 3,338.41 s, 1,557.08 MiB). YOLO11s seed 42
  stopped after 14 epochs (best 10, 1,586.39 s, 1,148.16 MiB) and seed 137
  after 19 epochs (best 14, 1,937.62 s, 1,148.16 MiB). All checkpoint, table,
  curve, summary, config, and source identities validate.
- Implemented `src/evaluate.py`, the single held-out adapter-to-metric harness.
  Both detectors emit canonical original-image boxes/scores/category IDs and
  are evaluated by the same score-ordered matcher and official pycocotools
  COCO evaluator. It reports precision, recall, F1, conditional matched-box
  IoU/Dice, mAP@0.5, mAP@0.5:0.95, FPS/latency, parameters, GFLOPs, peak GPU
  memory, and training time. Framework-native mAP remains checkpoint-selection
  evidence only.
- Evaluated all six frozen checkpoints on the 750-image/268-box held-out test
  split. Faster R-CNN versus YOLO11s mean ± sample SD: precision 0.1626 ±
  0.0439 versus 0.3730 ± 0.0395; recall 0.6381 ± 0.0526 versus 0.1356 ±
  0.0094; F1 0.2558 ± 0.0493 versus 0.1981 ± 0.0048; mAP@0.5 0.3084 ±
  0.0123 versus 0.1643 ± 0.0226; and mAP@0.5:0.95 0.1023 ± 0.0036 versus
  0.0549 ± 0.0080. YOLO11s is faster at 52.94 ± 10.65 FPS versus 17.42 ±
  5.69, with 9.43M versus 43.26M parameters and 21.42 versus 450.76 GFLOPs.
- Audited the final six-row grid independently: all requested values are
  finite, all means/sample SDs reproduce, all six checkpoint hashes match,
  and all six gzip bundles contain 750 prediction records and match their
  recorded hashes. Final publication/per-seed/long-form table SHA-256 values:
  `6b467c706dd39a9a240d99a552eb0218734c8b9eaf38b0bfbc70d347f921449c`,
  `ab4574589da9c63f4463e6ef13e4fef26dc565cd514cbd19118491ac0e7c09a8`,
  and `91affca6abe7fadcc70e0b5ca5836e74394d99df1b1af982ea7240dbcab9d482`.
  Summary SHA-256:
  `e6018a9fc2117ac41cc51ab395c22316e61ba40c030b8f54ce6e64c641ea8245`.
- Corrected the YOLO curve title to use each run ID. A reporting-only refresh
  initially remeasured speed; the comparison was then restored to each run's
  original immutable completion profile and the associated artifact hashes
  were revalidated. Final repository checks: 175 passed, one expected
  metadata-only skip, Ruff clean, and `git diff --check` clean.

**What's still incomplete / next step:**
- Batch 4 is complete. Do not start Batch 5 until the user reviews and accepts
  the comparison tables; these prediction bundles/metrics are the frozen basis
  for the later paired statistical tests.

**Needs the user's review before proceeding:**
- Review `results/tables/detector_comparison.csv` and the six detailed rows in
  `results/tables/detector_comparison_per_seed.csv`, with the definitions and
  timing caveat in `docs/QUANTITATIVE_COMPARISON.md`. In particular, confirm
  the accuracy/efficiency trade-off and the conditional interpretation of IoU
  and Dice before Batch 5.

**Files touched:**
- Config/code/tests: `configs/evaluation.yaml`, four seed configs,
  `src/evaluate.py`, `tests/test_evaluate.py`, and the YOLO curve-title code/test.
- Documentation/state: `README.md`, `docs/LIMITATIONS.md`,
  `docs/QUANTITATIVE_COMPARISON.md`, `CODEX.md`, and `HANDOFF.md`.
- Generated accepted artifacts: four seed training/derived-gate log trees and
  tables/curves, `results/logs/phase5_evaluation/`, and
  `results/tables/detector_comparison*.csv`; checkpoints remain ignored.
- The user-owned deletion of `Final project 1405.v1.pdf` and pre-existing
  failed/aborted diagnostic directories were left untouched.

---

## Session 11 — 2026-08-10 — Publish Batches 2–3

**What I did:**
- Reverified the confirmed publication scope, GitHub repository, current
  `agent/implementation-foundation` branch, and accepted Batch 2/3 artifacts.
- Staged 94 intended files while explicitly excluding the user-owned deletion
  of `Final project 1405.v1.pdf`, all failed/aborted diagnostic run directories,
  and every ignored model checkpoint. The staged whitespace and size audits
  passed.
- Reran repository validation immediately before publication: 172 tests passed,
  one expected metadata-only test skipped, and Ruff passed.
- Created commit `5dfee2d` (`Complete Faster R-CNN and YOLO baselines`) and
  pushed the branch, including its two earlier unpublished commits, to
  `origin/agent/implementation-foundation`.

**What's still incomplete / next step:**
- Batch 4 remains unstarted and requires the user's review/approval of the
  Batch 3 results and documented asymmetries.

**Needs the user's review before proceeding:**
- Confirm Batch 3 acceptance before requesting Batch 4.

**Files touched:**
- `HANDOFF.md`
- Git history/remote branch only; excluded local files remain untouched.

---

## Session 10 — 2026-08-10 — Batch 3 YOLO11s baseline

**What I did:**
- Confirmed the Batch 1 choice of YOLO11s at small scale and pinned
  `ultralytics==8.4.110`. Before any accepted training, resolved the required
  augmentation asymmetry by explicitly disabling every Ultralytics stochastic
  extra so the primary YOLO and Faster R-CNN arms both use deterministic
  resize-only inputs. Pinned the official pretrained checkpoint at SHA-256
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
- Implemented the strict config-driven YOLO data view, trainer, timing gate,
  checkpointing, early stopping, shared validation, curves, model/FLOP/latency
  profiling, recovery finalizer, and regression tests. The materialized views
  contain exactly 3,500 train images/1,267 boxes and 750 validation images/277
  boxes, including 2,702/581 negative images; the held-out test split was not
  accessed.
- Diagnosed and excluded the pre-benchmark numerical failures. The accepted
  policy uses batch/effective batch 4, 640 pixels, SGD LR 0.001 with one-epoch
  warmup, momentum 0.9, weight decay 0.0005, no Nesterov, bfloat16 AMP with
  float32 assignment/loss, native YOLO BatchNorm updates, ordinary seeded
  shuffle, two workers, and minimum-8/patience-5/maximum-30 validation-mAP
  stopping. All precision, normalization, LR, warmup, and scheduler asymmetries
  are disclosed in `docs/LIMITATIONS.md` and `docs/YOLO_BASELINE.md`.
- Completed the valid three-epoch benchmark in 141.79/134.99/136.20 seconds.
  Its 135.59-second steady epoch projected 18.18 minutes for eight epochs and
  67.90 minutes at the 30-epoch ceiling. Gate artifact SHA-256:
  `c339db91c05b1c8a1398dbbdcc7470ef1fd1932ddf1c374a87529faca45e1587`;
  config SHA-256:
  `5a9bd54c730a42db166d8e5c7075f863f914b5be7c66567f5bc91a70b50ef8d2`.
  The full-run training source identity matches the gate exactly.
- Completed the clean full seed-17 run from pretrained weights. It stopped at
  epoch 15 after patience reached 5/5; epoch 10 is best. Epoch-loop training
  took 1,975.64 seconds (32.93 minutes) and peak allocated VRAM was 1,148.16
  MiB. The native epoch-10 mAP50:95 used for selection was 0.07335.
- Final shared evaluation on all 750 validation images/277 boxes produced AP50
  0.26464 and AP50:95 0.08692. At score 0.25/match IoU 0.50 it produced
  precision 0.57143, recall 0.20217, and F1 0.29867 (56 TP, 42 FP, 221 FN).
  Batch-1 bfloat16 profiling over 100 synchronized images measured 65.24 FPS
  and mean/p50/p95 latency 15.33/14.49/19.82 ms. The model has 9,428,179 total
  and 9,428,163 training-time trainable parameters, 21.42 estimated GFLOPs,
  and a 19,172,819-byte/18.28-MiB checkpoint. Best checkpoint SHA-256:
  `65909164e82c1ef53c0d38e0d898d37bbbec5f46cb9f5cd029e76ba486c0371c`.
- Recovered reporting without retraining after Ultralytics materialized a
  750-path list as one inference batch and exhausted VRAM. The finalizer now
  streams only the audited validation directory, verifies all filenames,
  derives best epoch from immutable `results.csv` because stripped checkpoints
  store epoch `-1`, and records training-source and later reporting-source
  identities separately. Checkpoint copies/hashes, tables, summary, 15-row
  timing log, and the visually inspected best-epoch-10 curve all validate.
  Repository verification is 172 passed/1 expected metadata-only skip; Ruff
  and `git diff --check` pass.

**What's still incomplete / next step:**
- Batch 3 is complete. Do not start Batch 4 until the user reviews this
  one-seed result, curve, shared metrics, compute profile, and disclosed
  asymmetries.
- After approval, Batch 4 may implement the formal unified evaluator and the
  additional-seed headline comparison specified in Phase 5; no Batch 4 work
  has started.

**Needs the user's review before proceeding:**
- Review `results/figures/yolo_training_curves.png`,
  `results/tables/yolo_baseline_validation.csv`, and
  `results/tables/yolo_compute.csv`. In particular, confirm acceptance of the
  epoch-10 checkpoint, the shared AP50:95 0.08692 result, and the documented
  bfloat16/loss/BatchNorm/LR/scheduler asymmetries before Batch 4.

**Files touched:**
- Batch 3 config/code/tests: `configs/yolo.yaml`, `src/models/yolo_*.py`,
  `src/models/train_yolo.py`, and `tests/test_yolo_*.py`.
- Documentation/state: `README.md`, `docs/YOLO_BASELINE.md`,
  `docs/LIMITATIONS.md`, `CODEX.md`, and `HANDOFF.md`.
- Generated artifacts: `results/logs/yolo11s_rsna_seed17_{smoke,benchmark,full}/`,
  `results/checkpoints/yolo11s_rsna_seed17_full/`,
  `results/tables/yolo_{baseline_validation,compute}.csv`, and
  `results/figures/yolo_training_curves.png`; rejected numerical diagnostics
  and the preserved wrapper-collision attempt remain clearly named under
  `results/logs/` and are excluded from results.
- The pre-existing deletion of `Final project 1405.v1.pdf` was left untouched.

---

## Session 9 — 2026-08-09 — Batch 4 (blocked before start)

**What I did:**
- Read the newest `HANDOFF.md` entry, `CODEX.md`, `PROJECT_SPEC.md` §3 and §5 Phases 4–5, and the Batch 3–4 checkpoint language in `BATCHES.md` before taking any Batch 4 action.
- Verified that Batch 3 has not been run: `configs/yolo.yaml` contains only the Batch 1 model/version decision, the augmentation-asymmetry choice remains `TBD`, and no YOLO training implementation, checkpoint, log, curve, or baseline/compute table exists.
- Stopped without creating the unified evaluator or launching extra-seed runs because Batch 4 explicitly depends on the reviewed Batch 3 YOLO config and one-seed baseline; silently implementing Batch 3 inside Batch 4 would violate the repository's phase scope and review gate.

**What's still incomplete / next step:**
- Run Batch 3 first: make and document the augmentation-parity decision, implement YOLO11s training under the matched hardware protocol, complete the one-seed run, and stop for review.
- After Batch 3 is reviewed, rerun Batch 4 to build `src/evaluate.py`, evaluate both models identically, train seeds 2–3 for both detectors, and produce the mean ± std comparison tables.

**Needs the user's review before proceeding:**
- Confirm that Batch 3 should be run next. If the intended order has changed, explicitly resolve the conflict with the recorded sequential review gates before requesting Batch 4 again.

**Files touched:**
- `HANDOFF.md`

---

## Session 8 — 2026-08-09 — Batch 2 full Faster R-CNN baseline

**What I did:**
- Received explicit approval of Session 7's timing estimate and launched the documented full seed-17 command with the exact approved `benchmark_estimate.json`. Train mode independently revalidated the config, data, source, Python/Torch/CUDA/driver/GPU, AMP, batch, BatchNorm, and resolution identities, then restarted from COCO weights rather than timing-run weights.
- Completed 11 full train-plus-validation epochs and stopped by the configured validation-AP50:95 rule (minimum 8, patience 5, maximum 30). The scheduler reduced LR from 0.005 to 0.0005 at epoch 10. Epoch 6 is best: AP50:95 0.12764, AP50 0.33144, precision 0.14138, recall 0.68953, F1 0.23464 on all 750 validation images/277 boxes. The held-out test split was not accessed.
- Recorded 7,017.8 seconds (1.95 hours) of epoch-loop training and 1,556.6 MiB peak allocated GPU memory. Best-checkpoint batch-1 float16 AMP profiling over 100 synchronized batches measured 11.00 FPS and mean/p50/p95 latency 90.92/90.78/92.18 ms. The model has 43,256,153 total parameters, 43,030,809 trainable parameters, 450.76 estimated GFLOPs under the documented registered-operation convention, and a 173,412,170-byte/165.38-MiB checkpoint.
- Produced and validated the full CSV/JSONL history, summary, validation table, compute table, four-panel training curve, exact best checkpoint, and last restart state. All recorded sizes/hashes match the files; the visually inspected curve correctly marks epoch 6 and the epoch-10 LR reduction. Best checkpoint SHA-256: `9ec35c5d761f8e4bf7a43f7999f388ac1ffc0d533f62746409db280706dffab4`.

**What's still incomplete / next step:**
- Batch 2 is complete. Do not start Batch 3 until the user reviews the Faster R-CNN validation curves, final benchmark metrics, compute measurements, and one-seed limitation.
- Once approved, Batch 3 must implement only the pinned YOLO11s arm and make the required augmentation-asymmetry decision before training.

**Needs the user's review before proceeding:**
- Review `results/figures/faster_rcnn_training_curves.png`, `results/tables/faster_rcnn_baseline_validation.csv`, and `results/tables/faster_rcnn_compute.csv`, including the low-precision/high-recall operating point and validation variability. Approve or request changes before Batch 3.

**Files touched:**
- Generated artifacts: `results/logs/faster_rcnn_rsna_seed17_full/`, `results/checkpoints/faster_rcnn_rsna_seed17_full/{best_model,last_state}.pt`, `results/tables/faster_rcnn_{baseline_validation,compute}.csv`, `results/figures/faster_rcnn_training_curves.png`.
- Documentation/state: `docs/FASTER_RCNN_BASELINE.md`, `CODEX.md`, `HANDOFF.md`.

---

## Session 7 — 2026-08-09 — Batch 2 recovery and timing gate

**What I did:**
- Verified the official Kaggle aggregate ZIP (3,932,287,530 bytes; SHA-256 `133acacf95aa68c4d219124b17937f31cec073052096b9f9b122180df9d9af18`) by full CRC/path audit: 26,684 train DICOMs, 3,000 competition-test DICOMs, and exact official CSV hashes. Regenerated all 5,000 selected PNGs with zero missing/errors; the 12 earlier review conversions were byte-identical.
- Adopted the user-authorized local `torch-gpu` runtime before timed work: Python 3.11.15, Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, CUDA 12.4, and driver 610.47. Aligned dependency manifests/docs, verified CUDA NMS/AMP/FLOP support, passed preflight on all 3,500 train and 750 validation images, passed 160 tests with one expected metadata-only skip, passed Ruff, and completed the bounded smoke test.
- Diagnosed a first official-data attempt that finished all epoch-1 training batches but failed before validation with Windows `WinError 1455`: persistent training workers overlapped the new validation pool and exceeded the 16 GB host commit limit. It produced no epoch record/checkpoint. Preserved the failed metadata, kept six workers, changed train/validation pools to non-persistent, documented the decision, updated the pinned config hash, reran tests/smoke, and restarted cleanly from COCO weights.
- Completed exactly three clean benchmark epochs in 770.3/551.5/473.0 seconds (29.91 minutes total). Steady-state time is 512.2 seconds/epoch; eight epochs project to 1.21 hours and the 30-epoch upper bound to 4.34 hours (conservative 4.02--4.66-hour range). Peak allocated GPU memory was 1,556.6 MiB. Epoch 3 diagnostic validation AP50:95 was 0.10993; it is not a final result.
- Validated all CSV/JSONL/summary/projection invariants, both timing checkpoints, dataset/execution/implementation identities, and the same approval function used by train mode. The competition test split was not accessed. Approval artifact: `results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json`, SHA-256 `232460ae09827dfb780b0f5c6506bf9f545bbdc0e1483082c2c440035e8e8e8b`; config SHA-256 `ef1e3ebe1fbe3cf1a6e27bf8b9c12f61719c2ea8771c9758f64dc278dd0e2633`.

**What's still incomplete / next step:**
- Full one-seed Faster R-CNN training has not started. After explicit approval, run the documented `--mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json` command; early stopping uses validation AP50:95 with minimum 8, patience 5, maximum 30 epochs.
- After the full run, verify final metrics/curves/FPS/parameters/GFLOPs/model size/checkpoint/training time and stop again for Batch 2 result review. Do not start Batch 3 yet.

**Needs the user's review before proceeding:**
- Approve or reject the measured 4.34-hour maximum estimate (4.02--4.66-hour conservative range), bound to the artifact/hash above. No full training may start without explicit approval.

**Files touched:**
- Runtime/dependency state: `.python-version`, `requirements.txt`, `pyproject.toml`, `uv.lock`, `README.md`.
- Official-data provenance/docs: `data/README.md`, `data/manifests/rsna-pneumonia-5000-audit.json`, `docs/DATASHEET.md`.
- Batch 2 config/code/tests/docs: `configs/faster_rcnn.yaml`, `src/models/`, `src/meddet_benchmark/`, `src/utils/seed.py`, corresponding tests, `docs/FASTER_RCNN_BASELINE.md`, `docs/LIMITATIONS.md`, `docs/REPRODUCIBILITY.md`, `CODEX.md`, `HANDOFF.md`.
- Local ignored artifacts: official raw/extracted DICOMs, 5,000 processed PNGs/COCO files, smoke/benchmark log directories, timing checkpoints, and preserved aborted-attempt metadata directories. The pre-existing deletion of `Final project 1405.v1.pdf` was left untouched.

---

## Session 6 — 2026-08-04 — Batch 2 Kaggle OAuth retry

**What I did:**
- Reopened Kaggle OAuth after the user completed browser authentication and kept the local callback alive through authorization.
- Received the browser authorization code successfully, but Kaggle denied the subsequent OAuth token exchange with HTTP 403. Verified that the CLI remains on `LEGACY_API_KEY`, the official archive is absent, and the checkout still contains only 12 DICOMs and 12 processed PNGs.
- Made no Torch-environment substitution and did not start conversion, smoke testing, or training.

**What's still incomplete / next step:**
- From an authorized available network/location, manually download the official `stage_2_train_images.zip`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` through the authenticated Kaggle website and place them in `data/raw/rsna-pneumonia/`.
- Explicitly authorize either repinning to the verified existing Anaconda Python 3.11 / Torch 2.6.0+cu124 / Torchvision 0.21.0+cu124 environment, or retaining and redownloading the current Python 3.13 / Torch 2.13.0+cu130 pins.
- Then verify sources, prepare 5,000 PNGs, run tests/preflight/smoke, and run exactly three benchmark epochs.

**Needs the user's review before proceeding:**
- Confirm when the three official files are present and state the environment choice. Kaggle API retries are not useful while OAuth token exchange itself is denied.

**Files touched:**
- `CODEX.md`, `HANDOFF.md`

---

## Session 5 — 2026-08-04 — Batch 2 recovery

**What I did:**
- Resumed the interrupted setup, verified no installer remained active, and confirmed that the project `.venv` still lacks Torch/Torchvision while the interrupted `uv` cache contains 3.81 GB that cannot resolve the pinned wheels offline.
- Audited the user's Anaconda `torch-gpu` environment read-only: Python 3.11.15, Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, working RTX 4060 CUDA, Torchvision CUDA NMS, and AMP. Did not adopt it because the recorded project identity is Python 3.13 / Torch 2.13.0+cu130 / Torchvision 0.28.0+cu130.
- Installed only `kaggle==2.2.3` into `.venv`, validated the credential structure, and preserved the review-source metadata before attempting the official download. Kaggle returned HTTP 403 before downloading any bytes; public dataset lists and competition lists/files also return 403, so the failure is API-wide rather than an absent archive or downloader bug.
- Attempted Kaggle's forced browser OAuth flow; it remained pending without completing and was cancelled cleanly. Restored the two review-source CSVs and verified their exact recorded SHA-256 hashes. No official archive, new DICOM, processed PNG, smoke artifact, or benchmark artifact was produced.

**What's still incomplete / next step:**
- Complete a fresh Kaggle OAuth/login for account `alphalacrim`, or manually download the official `stage_2_train_images.zip`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` into `data/raw/rsna-pneumonia/`.
- Explicitly choose whether to retain the pinned Python 3.13 / Torch 2.13.0+cu130 stack and redownload it, or authorize repinning the experiment to the verified existing Python 3.11 / Torch 2.6.0+cu124 / Torchvision 0.21.0+cu124 Anaconda stack.
- After those two prerequisites, convert and verify exactly 5,000 PNGs, run tests/preflight/smoke, then run exactly three complete benchmark epochs and stop for timing approval.

**Needs the user's review before proceeding:**
- Resolve Kaggle authentication/manual official-file placement and make the explicit environment-identity choice above. The benchmark cannot start with mirror pixels or a silently substituted Torch stack.

**Files touched:**
- `CODEX.md`, `HANDOFF.md`
- Local ignored environment/data state: `.venv` gained `kaggle==2.2.3`; review CSVs were temporarily moved and then restored byte-identically.

---

## Session 4 — 2026-08-02 — Batch 3 (blocked before start)

**What I did:**
- Read `HANDOFF.md`, `CODEX.md`, `PROJECT_SPEC.md` §3 and §5 Phase 4, and the Batch 2–4 checkpoint language in `BATCHES.md` before taking any Batch 3 action.
- Rechecked the recorded prerequisites: only 12 of the fixed 5,000 processed PNGs are present, and `.venv` has no Torch, Torchvision, or Ultralytics installation.

**What's still incomplete / next step:**
- Batch 2 still needs the official selected pixels, CUDA dependencies, its complete three-epoch timing gate, explicit approval, and the one-seed full Faster R-CNN run.
- Batch 3 must remain unstarted until the user resolves or explicitly overrides that phase-order conflict; no YOLO config, augmentation decision, implementation, or training artifact was created in this session.

**Needs the user's review before proceeding:**
- Provide/authorize the missing official RSNA pixels and dependency installation, and direct completion of Batch 2 first; or explicitly override the recorded Batch 2 review gate and accept that Batch 3 still cannot complete a real-data training run until the same data/dependency blockers are resolved.

**Files touched:**
- `HANDOFF.md`

---

## Session 3 — 2026-08-02 — Batch 2

**What I did:**
- Implemented the strict `fasterrcnn_resnet50_fpn_v2` Batch 2 configuration and pipeline: canonical COCO train/validation adapter, derived class mapping, negative-image support, RGB tensor conversion, transfer-learning head replacement, float16 AMP, physical batch 2 with accumulation 2, frozen BatchNorm statistics, SGD/plateau scheduling, exact-best checkpoints, and validation-AP early stopping.
- Reused the shared COCO/operating-point evaluator so every epoch records loss components, precision, recall, F1, AP50, and AP50:95. Added configured atomic CSV/JSONL logs, best/last state checkpoints, validation/compute tables, four-panel curves, synchronized FPS/latency, parameter counts, mandatory GFLOPs, checkpoint size/hash, peak GPU memory, and training time.
- Made the three-epoch gate representative and hard to reuse accidentally: its epoch durations include equivalent checkpoint I/O, and full-run approval compares exact YAML, train/validation annotation and pixel manifests, implementation sources, dependency/CUDA/driver/GPU identity, AMP, batch, BatchNorm policy, and resolution. The full run always restarts from COCO weights.
- Added an idempotent `--mode finalize` path so profiling/table/plot failures after optimization can be recovered from the saved best checkpoint. Corrected exact-best versus early-stopping patience semantics, learning-rate logging, operating-point prediction counts, DataLoader worker-pool cleanup, Windows path containment, full image decoding, crowd rejection, and clean package-relative imports.
- Audited the real local data without touching test annotations: train is 7/3,500 images with 1,267 boxes (3,493 missing), validation is 5/750 with 277 boxes (745 missing). The fixed 5,000-study set is still missing 4,988 images overall. Benchmark/full modes now fail before Torch/CUDA with all exact missing train/validation paths.
- Added Batch 2 protocol/reproduction documentation and exact Pydantic/Matplotlib pins. Verified 160 tests pass, one Torch tensor-contract test is skipped because Torch is absent, Ruff passes, `git diff --check` is clean, and the RTX 4060 Laptop GPU is visible with 8,188 MiB VRAM and driver 610.47. No model training was started.

**What's still incomplete / next step:**
- Provide authorized Kaggle access (accepted competition rules plus `KAGGLE_USERNAME`/`KAGGLE_KEY` or `C:\Users\Pouyan\.kaggle\kaggle.json`), or place the official Stage 2 archive/DICOMs locally; then run configured conversion and confirm all 5,000 PNGs.
- Complete the pinned CUDA Torch 2.13.0/Torchvision 0.28.0 installation. Two attempts made slow partial-download progress but did not install either package; then run the bounded CUDA/AMP/spawn-DataLoader smoke check and the currently skipped tensor test.
- Run exactly three complete benchmark epochs, report the measured time projection, and stop for user sign-off. Only after approval may the one-seed full run, final profiling/tables/curves, final CODEX metrics, and checkpoint/training-time record be produced.

**Needs the user's review before proceeding:**
- No training-time estimate exists yet: a 12-image estimate would be misleading and violates the complete-real-data gate. The user must supply/authorize the official pixels, then review and explicitly approve the real three-epoch estimate before full training.
- No Batch 3 work may start; Batch 2 still requires the approved full run and later review of its curves and compute numbers.

**Files touched:**
- `configs/faster_rcnn.yaml`, `requirements.txt`, `README.md`
- `src/models/{__init__,faster_rcnn_config,faster_rcnn_data,faster_rcnn_model,faster_rcnn_training,faster_rcnn_evaluation,faster_rcnn_reporting,train_faster_rcnn}.py`
- `tests/test_faster_rcnn_{config,data,model,training,reporting}.py`, `tests/test_train_faster_rcnn.py`
- `src/meddet_benchmark/{__init__,__main__,coco_evaluation}.py`
- `docs/FASTER_RCNN_BASELINE.md`, `docs/LIMITATIONS.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 2 — 2026-08-02 — Batch 1

**What I did:**
- Inspected all three linked dataset candidates for advertised/verified image inventory, class semantics, annotation format/integrity, balance, license terms, and recoverable patient/study identity. Selected RSNA Stage 2 and documented why the two MRI exports cannot establish patient-disjoint splits.
- Selected YOLO11s with `ultralytics==8.4.110`; wrote the detector/Grad-CAM/medical-detection/corruption literature review and a 22-entry BibTeX file.
- Added a secret-safe Kaggle downloader that reads a complete environment-variable pair or `~/.kaggle/kaggle.json`/`KAGGLE_CONFIG_DIR`, redacts secrets, and distinguishes missing credentials, unaccepted rules, and geographic storage failures.
- Audited all 30,227 RSNA annotation rows: 26,684 valid studies, 9,555 valid boxes, zero malformed/non-positive/off-image/duplicate boxes, and zero target/class/mapping inconsistencies. Verified and recorded input SHA-256 values.
- Recovered true NIH patient keys from the official RSNA mapping. Deterministically selected 5,000 studies from 2,136 patient groups and created exact 3,500/750/750 splits with zero patient overlap. Wrote committed CSV manifests and generated per-split canonical COCO JSON.
- Converted 12 authentic review DICOMs, generated the split-distribution and labeled bounding-box sample figures, visually checked box placement, and wrote the dataset-choice report, datasheet, and limitations.
- Verified 80 repository tests pass, Ruff passes over `src`/`tests`, `git diff --check` is clean, and a repeated real-data preparation produces identical hashes for all three manifests and all three COCO files. No model training or Batch 2 implementation was started.

**What's still incomplete / next step:**
- Stop here until the user approves the RSNA choice, one-class task definition, patient-safe 5,000-study split, and EDA.
- Before Batch 2 can train, use authorized Kaggle credentials to acquire the other 4,988 selected DICOMs, rerun conversion, and compare the official-download CSV hashes with the committed audit.
- Batch 2 is Faster R-CNN only after this review gate. YOLO augmentation-parity handling remains deferred to Batch 3.

**Needs the user's review before proceeding:**
- Approve or reject the RSNA selection and fixed 5,000-study hardware scope.
- Review `results/figures/rsna_class_distribution.png` and `results/figures/rsna_annotation_samples.png`, plus the disclosed interim mirror provenance/full-image-download requirement in `docs/DATASHEET.md`.
- Confirm acceptance of the actual class map (`Lung Opacity` only) and the NIH-patient-group split strategy.

**Files touched:**
- `README.md`, `requirements.txt`, `configs/{dataset,yolo}.yaml`, `data/README.md`
- `src/data/{__init__,download,prepare,visualize}.py`
- `tests/test_{download,prepare,visualize}.py`
- `docs/{DATASET_CHOICE,DATASHEET,LITERATURE_REVIEW,LIMITATIONS}.md`, `report/references.bib`
- `data/manifests/rsna-pneumonia-5000-audit.json`, `data/splits/rsna-pneumonia-5000/*.csv`
- `results/figures/rsna_{class_distribution,annotation_samples}.png`, `results/figures/rsna_eda_summary.json`
- `CODEX.md`, `HANDOFF.md`

---

## Session 1 — 2026-08-02 — Batch 0

**What I did:**
- Created the §4 repository skeleton with decision-gated config/document placeholders and tracked empty directories; raw/processed data and checkpoints remain gitignored.
- Added exact dependency pins in `requirements.txt`, leaving Ultralytics as a commented Batch 1 placeholder.
- Implemented `src/utils/seed.py` to seed Python, NumPy, and available PyTorch/CUDA RNGs; enable deterministic behavior; and write package, platform, GPU, and driver metadata into every supplied run directory.
- Added `tests/test_seed.py` and `docs/REPRODUCIBILITY.md`. Verified 50 repository tests pass, Ruff passes on all new Python files, and a local smoke run detects the RTX 4060 Laptop GPU with driver 610.47.
- Preserved all pre-existing dataset/evaluator/corruption/model-adjacent files without modification. The repo was already Git-initialized with 12 commits, so no reinitialization or history rewrite was attempted.
- Treated the user-supplied `AGENTS.md`, `CODEX.md`, and `HANDOFF.md` replacements as the authoritative workflow reset. Their legacy content remains recoverable from earlier commits. Normalized the two memory filenames to uppercase for case-sensitive checkouts and replaced the stale README with Batch 0 instructions.

**What's still incomplete / next step:**
- Batch 1 must perform the literature review, select and pin the Ultralytics/YOLO version, investigate the three datasets, and stop again for dataset/EDA review.
- Reconcile the pre-existing `pyproject.toml`/`uv.lock` and `src/meddet_benchmark/` implementation before using them; they predate this workflow and include an Ultralytics pin that is not an approved project decision.

**Needs the user's review before proceeding:**
- Approve the skeleton, dependency versions, and `initialize_reproducibility(seed, output_dir)` artifact contract before Batch 1.
- Note that this checkout already had Git history; the new Batch 0 commit is not a literal root commit.

**Files touched:**
- `.gitignore`, `README.md`, `requirements.txt`, `configs/{dataset,faster_rcnn,yolo}.yaml`
- `data/{raw,processed,splits}/`, `src/{data,models,robustness,explainability,stats}/`, `notebooks/`, `results/{tables,figures,logs,checkpoints}/`
- `src/utils/{__init__,seed}.py`, `tests/test_seed.py`
- `docs/{DATASET_CHOICE,LITERATURE_REVIEW,LIMITATIONS,REPRODUCIBILITY}.md`, `report/report.md`
- `CODEX.md`, `HANDOFF.md`

---

## Session 0 — (not yet run)

No sessions completed yet. First session starts with **Batch 0** from `BATCHES.md`.
