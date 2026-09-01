# Final Submission Audit

**Audit date:** 2026-09-01

**Audited commit:** `9aaf414bdbab337a7c45283d4b32fe58b1e1700d` (`main`, aligned with `origin/main`)

**Current manuscript:** `report/paper_draft.md`

**Historical technical report:** `report/report.md`
**Audit disposition:** **NOT READY FOR SUBMISSION**

This is an adversarial audit of the committed scientific and manuscript state. A
completed prior batch was not accepted as evidence of success: the software
gates, claim bindings, source cells, captions, language, references, reporting
crosswalk, and named artifacts were rechecked from the current tree.

## Recorded repository and CI state

| Item | Audited state |
|---|---|
| HEAD | `9aaf414bdbab337a7c45283d4b32fe58b1e1700d`, `Align manuscript with corrected evidence`, 2026-09-01 04:14:47 +0330 |
| Branch/tracking | `main...origin/main`; zero ahead/behind at audit start |
| Tracked worktree | Clean at audit start and finish; Batch 38 changed no tracked scientific/code/manuscript file |
| Untracked worktree | Pre-existing internal rule/spec/state files (`AGENTS.md`, `BATCHES*.md`, `PROJECT_SPEC.md`, `CODEX.md`, `HANDOFF.md`, the prior review audit, and review notes) plus documented aborted/smoke/orchestration logs; Batch 38 adds only this audit and updates the intentionally untracked state logs |
| Test status | 300 passed, 1 expected metadata-only skip |
| CI workflow | Sole workflow `.github/workflows/ci.yml`, name `foundation-ci`; triggers on push and pull request; `contents: read`; cancel-in-progress concurrency |
| CI matrix | `ubuntu-latest` and `windows-latest`, `fail-fast: false` |
| CI action/tool pins | `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`; `astral-sh/setup-uv@94527f2e458b27549849d47d273a16bec83a01e9`; uv `0.11.27` |
| CI gates | `uv python install`; lock check; locked dev/CPU sync; Ruff format and lint; full pytest; scientific artifact verifier; manuscript claim verifier; deterministic package smoke |
| CI boundary | No data download, checkpoint load, model training, or GPU inference; green CI is a software/frozen-evidence check, not end-to-end scientific regeneration |

## BLOCKING FAILURES

1. **The human-only declarations remain unresolved.** Funding, competing
   interests, ethics/data-use, consent, author contributions, data/code/model
   availability, and patient/public involvement still require author decisions
   in `docs/AUTHOR_DECLARATIONS_TODO.md` and manuscript section 8. They cannot be
   inferred from repository evidence.
2. **The final availability statement is not settled.** All ten exact Phase 5
   checkpoints exist locally and match their recorded sizes and SHA-256 hashes,
   but they are Git-ignored, not committed, and not publicly uploaded. The
   checkpoint release manifest remains
   `manifest_ready_binaries_not_published`, with a null public URL. The authors
   must decide what will be released and make the manuscript declaration match
   that decision.

No software, numerical, statistical-language, or phantom-artifact blocker was
found. The unresolved author declarations nevertheless prohibit a
submission-ready conclusion.

## SHOULD-FIX

- Complete every item in `docs/AUTHOR_DECLARATIONS_TODO.md`, then replace the
  manuscript placeholders and regenerate `docs/REPORTING_CHECKLIST.md` against
  the changed manuscript hash.
- Decide whether to publish the ten-checkpoint archive. If publishing, populate
  the release manifest URL and verify the uploaded bytes; if not publishing,
  state the exact access boundary without implying clean-checkout exact
  inference.
- Author-review the CLAIM 2024 rows currently answered No, especially source
  accrual dates, demographic/subgroup reporting, participant flow, external
  testing, formal sample-size rationale, and availability. Several are genuine
  study limitations rather than facts recoverable from this repository.
- The bibliography contains three intentionally disclosed historical
  reading-list entries not cited by the current manuscript
  (`chattopadhay2018gradcampp`, `muhammad2020eigencam`, and
  `petsiuk2021drise`). Remove them if the target journal requires a cited-only
  bibliography.
- The README's archive-reproduction command names two outputs that are correctly
  described as generated on replay and are not currently committed:
  `results/figures/froc_curves_n3_archive_reproduction.png` and
  `results/tables/froc_operating_points_n3_archive_reproduction.csv`. This is
  not a phantom claim, but an author may run that optional replay before final
  packaging.

## VERIFIED CLEAN AREAS

- **Repository state and CI:** HEAD is aligned with `origin/main`; there were no
  tracked modifications at audit start. The only workflow is
  `.github/workflows/ci.yml`, running on Ubuntu and Windows with pinned action
  SHAs, Python/lock setup, Ruff format/lint, full pytest, both artifact
  verifiers, and the package smoke test. CI intentionally does not claim to
  reproduce GPU inference or training.
- **Software and evidence gates:** pytest, Ruff formatting, Ruff lint, package
  smoke, the 46-artifact scientific verifier, the 35-claim manuscript verifier,
  and `uv lock --check` all pass. The single skipped test is the expected
  metadata-only Faster R-CNN case.
- **Numerical traceability:** the manuscript values sampled below resolve to
  exact committed cells, JSON pointers, or deterministic calculations. No
  sampled mismatch or rounding error beyond the declared tolerance was found.
- **Statistical language:** the manuscript contains no use of “significant,”
  “significance,” or “superior.” Its two “robust” uses reject robust evidence;
  its “better” uses are bounded to coverage, a stated Pareto rule, retention, or
  an explicitly rejected universal detector-family claim. Uncertainty and all
  three p-value uses distinguish the primary population-level
  training-procedure estimand from the secondary checkpoint-conditional
  sensitivity. Seed and checkpoint scopes agree with the captions and source
  tables.
- **DCA/utility language:** no current figure, table, result, or citation implies
  that a raw detector score is a clinical threshold probability. The current
  raw-score utility artifact is explicitly non-standard and excluded from
  clinical interpretation. The frozen pre-Batch-30 archive retains its original
  labels only as a clearly named historical/nonstandard archive; it is not
  routed as current DCA evidence.
- **Threshold language:** every F-beta use defines beta as preference weighting,
  not a measured clinical cost. The separate linear-loss examples are explicitly
  hypothetical; the manuscript contains no claim that `C_FN` or `C_FP` was
  elicited from clinical practice.
- **XAI language:** input-pixel shuffling is named an input-pixel control, and
  the manuscript explicitly states that Adebayo-style training-label/data
  randomization was not performed. Sanity behavior is not presented as proof of
  clinical faithfulness, causal attention, or clinical reasoning. Localization
  is descriptive and weak in absolute magnitude.
- **Radiography/DICOM language:** the manuscript makes no CT, HU, or Hounsfield
  claim. It identifies the inspected objects as `Modality=CR` Secondary Capture,
  workstation-converted, 8-bit MONOCHROME2, lossy JPEG, and discloses missing
  rescale/Modality-LUT/VOI intent. Poisson-like and blur perturbations are
  synthetic sensitivity analyses, not reconstructed dose, quantum noise, or
  acquisition physics.
- **Calibration language:** D-ECE is consistently a descriptive,
  class-conditioned emitted-detection calibration endpoint, not exam-level
  clinical-risk calibration. Sparse support is visible: only 68–354 of 3,125
  possible cells were occupied and 15–169 met the minimum support per run.
- **Seed/scope consistency:** seed 271 is retained in every n=5 analysis. It is
  absent only from explicitly n=3 selection analyses and single-checkpoint
  seed-17 analyses. YOLO seed 271 is conditionally undefined for IoU/Dice, so
  those paired secondary comparisons are n=4; it is not silently deleted from
  unconditional endpoints. No caption/claim contradicts the scope table below.
- **Hypotheses:** H1–H6 are represented with the wording, endpoint, split, seed,
  and inference class recorded in `docs/HYPOTHESIS_TRACEABILITY.md`. H1 has the
  designated primary inferential evidence; H2–H6 are accurately bounded. All are
  retrospective and none is described as preregistered.
- **References:** all 29 manuscript citation keys resolve in
  `report/references.bib`, all 29 appear in `docs/CITATION_AUDIT.md`, and there
  are no duplicate BibTeX keys. The audit contains no unsupported or
  wrong-attribution verdict. A fresh primary-source scope check also confirmed
  the roles of CLAIM 2024, STARD-AI, TRIPOD+AI, and DICOM Secondary Capture.
- **Reporting hierarchy:** the reporting checklist binds to manuscript SHA-256
  `0b4cbb4e9effb1c055f0da885ba23636e7adbfd1b885504d20b9a2715943f0f5`, which
  exactly matches `report/paper_draft.md`; it does not point to the historical
  report. CLAIM 2024 is the primary crosswalk. STARD-AI and TRIPOD+AI are used
  only by analogy and no compliance claim is made.
- **README/document hierarchy:** README explicitly identifies
  `report/paper_draft.md` as the current manuscript and `report/report.md` as the
  historical technical report. All 30 referenced config files and all 21
  project Python modules exist. The only absent named output paths are the two
  accurately labeled generated-on-replay files listed under SHOULD-FIX.
- **Phantom-artifact check:** every manuscript-linked local figure (Figures
  1–7), table, supplement, bibliography, and documentation file exists. All 72
  supplement local links resolve. The ten locally referenced checkpoint files
  exist and match the release manifest. Externally unavailable items are
  explicitly labeled: the licensed/ignored raw and processed data boundary and
  the not-yet-published checkpoint archive.

## Numerical traceability (29 manuscript numbers)

All source artifacts in this table are committed at the audited HEAD. “Exact”
means integer/string equality or an exact deterministic calculation; other
tolerances are the manuscript claim-manifest tolerances.

| Manuscript location and value | Exact committed source cell/calculation | Source value | Tolerance | Result |
|---|---|---:|---:|---|
| Abstract: 5,000 studies | `data/manifests/rsna-pneumonia-5000-audit.json` → `subsample.selected_images` | 5,000 | exact | Match |
| Abstract: 2,136 patient groups | same → `subsample.selected_groups` | 2,136 | exact | Match |
| Abstract: 3,500/750/750 split | same → `splits.{train,val,test}.images` | 3,500/750/750 | exact | Match |
| Abstract/Results: Faster R-CNN mAP50:95 0.0995 | `results/tables/pareto_summary_n5_sensitivity.csv`, Faster R-CNN, `map_50_95_mean` | 0.0995015708 | 0.00005 | Match |
| Abstract/Results: YOLO11s mAP50:95 0.0542 | same, YOLO11s, `map_50_95_mean` | 0.0541683534 | 0.00005 | Match |
| Abstract/Discussion: approximately 3-fold throughput | YOLO `throughput_mean` / Faster `throughput_mean` in same table | 2.9735986251-fold | 0.1 | Match |
| Abstract/Discussion: 78% fewer parameters | `(43,256,153 - 9,428,179) / 43,256,153 × 100` from same table | 78.203843% | 0.5 pp | Match |
| Abstract/Discussion: 21-fold fewer operations | `450.7637248 / 21.4198784` GFLOPs from same table | 21.0441776-fold | 0.5 | Match |
| Abstract/Results: seed-271 mAP50:95 0.05558 | `results/tables/detector_comparison_per_seed.csv`, YOLO11s seed 271, `map_50_95` | 0.0555798883 | 0.000005 | Match |
| Abstract/Results: seed-271 maximum score 0.0412735 | `results/tables/yolo_seed_stability.csv`, seed 271, `maximum_prediction_score` | 0.0412735231 | 0.00000005 | Match |
| Abstract/Results: D-ECE 0.0990 vs 0.0320 | `results/tables/calibration_summary_v2.csv`, detector means | 0.0990267124 / 0.0320433064 | 0.00005 | Match |
| Methods: 26,684 source exams | dataset audit → `input_counts.annotation_exam_ids` | 26,684 | exact | Match |
| Methods: 9,555 valid boxes | dataset audit → `input_counts.valid_boxes` | 9,555 | exact | Match |
| Methods/Limitations: 21,684 excluded source studies | `26,684 - 5,000` from dataset audit | 21,684 | exact | Match |
| Methods: 2,000 bootstrap replicates | `configs/statistics.yaml` → bootstrap replicate count | 2,000 | exact | Match |
| Methods: 5,000 permutation replicates | `configs/statistics.yaml` → permutation replicate count | 5,000 | exact | Match |
| Methods/Limitations: 3,125 possible D-ECE cells | `configs/calibration.yaml`: five bins on each of five dimensions, `5^5` | 3,125 | exact | Match |
| Methods: robustness sample 300 images/183 patients/111 boxes | `data/splits/rsna-pneumonia-5000/test_robustness_seed17_n300.csv`: row count, distinct `nih_patient_id`, and sum of `box_count` | 300/183/111 | exact | Match |
| Results: Faster leads at 97 of 101 AP50 recall positions | `results/tables/operating_regime_n3_vs_n5_conclusions.csv`, official-PR n=5 conclusion row | 97/101 | exact | Match |
| Results: selected thresholds 0.69/0.05 | `results/tables/selected_operating_points_n5_sensitivity.csv`, Faster/YOLO threshold | 0.69/0.05 | exact | Match |
| Results: FROC sensitivity at 2 FP/image 0.6873/0.2925 | `results/tables/froc_operating_points_n5_sensitivity.csv`, detector rows at budget 2 | 0.6873134328 / 0.2925373134 | 0.00005 | Match |
| Results/Limitations: 68–354 occupied calibration cells | `results/tables/calibration_support_v2.csv`, min/max `occupied_cells` | 68–354 | exact | Match |
| Results/Limitations: 15–169 supported calibration cells | same, min/max `supported_cells` | 15–169 | exact | Match |
| Results: corruption mean retention 0.763846/0.709083 | `results/tables/robustness_results.csv`, mean `map_50_95_relative` over 35 non-clean rows per detector | 0.7638457116 / 0.7090831270 | 0.0000005 | Match |
| Results: strongest retained Poisson-like DSI 0.314529/0.436641 | `results/tables/radiography_synthetic_shift_results.csv`, `poisson_dose_12p5pct` | 0.3145287192 / 0.4366407684 | 0.0000005 | Match |
| Results: Grad-CAM energy 0.0869/0.0975 | `results/tables/gradcam_localization_summary.csv`, detector means | 0.0868948800 / 0.0974909561 | 0.00005 | Match |
| Results: primary precision difference −0.1024, CI [−0.2423, 0.0553] | `results/tables/statistical_clean_comparison.csv`, precision row | −0.1024262; [−0.24226168, 0.05531985] | 0.00005 | Match |
| Limitations: XAI nested subset 50 images/41 patients | `results/logs/phase31_xai_sanity_v2/subset_manifest.csv` row/distinct-patient counts, cross-checked against `summary.json` | 50/41 | exact | Match |
| Limitations: ten exact checkpoints | `results/checkpoint_release_manifest.json` → checkpoint records | 10 | exact | Match |

## Analysis-scope table

“Population-level” below means the Batch 28 training-procedure estimand over
independent trained runs and test-patient clusters. “Checkpoint-conditional” is
the explicitly secondary fixed-checkpoint sensitivity. Other rows are
descriptive even where their scripts produce interval summaries.

| Major analysis | Dataset/sample | Patient clusters | Detector runs/seeds | Seed 271? | Inference class |
|---|---:|---:|---|---|---|
| Cohort construction | 5,000 studies; train/val/test 3,500/750/750 | 2,136 total; 1,492/321/323 | N/A | N/A | Descriptive cohort audit |
| Clean score-0.25/AP summaries | 750 test images | 323 | 5 per detector: 17, 42, 137, 271, 314 | Yes | Descriptive absolute endpoints |
| Primary clean bootstrap | 750 test images | 323 | 5/5 unconditional; 5 Faster and 4 YOLO conditional-localization endpoints | Yes where defined | Population-level training-procedure |
| Secondary clean permutation sensitivity | 750 test images | 323 | Five observed checkpoints for unconditional endpoints; four pairs for IoU/Dice | Yes except undefined YOLO localization pair | Checkpoint-conditional |
| Test threshold/PR sensitivity | 750 test images | 323 | 5 per detector | Yes | Descriptive test sensitivity |
| Validation threshold selection | 750 validation images | 321 | n=3: 17, 42, 137 | No | Descriptive model selection |
| Frozen-threshold test operating point | 750 test images | 323 | n=5 test evaluation; thresholds selected on n=3 validation | Yes on test | Descriptive fixed-rule evaluation |
| FROC sensitivity | 750 test images | 323 | 5 per detector | Yes | Descriptive |
| F-beta/hypothetical loss sensitivity | 750 validation images | 321 | n=3: 17, 42, 137 | No | Descriptive validation sensitivity |
| Detection-level D-ECE | 750 test images | 323 | 5 per detector | Yes | Descriptive emitted-detection endpoint |
| Compute profile | Batch 1; 10 warm-up + 100 timed images/run | N/A | 5 per detector | Yes | Descriptive hardware-bound profile |
| Five-run Pareto sensitivity | 750 test images plus compute profile | 323 for accuracy | 5 per detector | Yes | Descriptive scenario/rule comparison |
| Digital corruption grid | 300 images, 111 boxes | 183 | One seed-17 checkpoint per detector | No | Descriptive grid |
| Corruption permutation analysis | Same fixed 300-image sample | 183 | One seed-17 checkpoint per detector | No | Checkpoint-conditional patient-cluster inference |
| Acquisition-shift audit | 300 images | 183 | One seed-17 checkpoint per detector | No | Descriptive synthetic sensitivity |
| Grad-CAM localization | 300 images; 111 positive boxes | 183 | One seed-17 checkpoint per detector | No | Descriptive localization |
| XAI sanity controls | 50 images | 41 | One seed-17 checkpoint per detector | No | Descriptive control |
| Archived raw-score utility | 750 test images | 323 | 5 per detector | Yes | Descriptive, nonstandard sensitivity |
| Conventional DCA | Not performed | N/A | N/A | N/A | No clinical-utility inference |

The manuscript captions were compared against this table. Figures 1–5 identify
their n=5, n=4, or n=3-selection boundary; Figure 6 identifies the seed-17
single-checkpoint scope; Figure 7 identifies the nested 50-image scope. No
caption or manuscript claim silently changes its estimand or run count.

## Language-search audit

The case-insensitive manuscript search produced these exact-word counts; hyphen
variants were inspected with their surrounding text as well.

| Family | Terms/counts | Audit result |
|---|---|---|
| Statistical | significant 0; significance 0; robust 2; superior 0; better 4; uncertainty 4; confidence interval 0; p-value 3; seed 41; checkpoint 22 | All bounded to the Batch 28 estimands and displayed run counts |
| DCA/utility | DCA 3; decision curve 0; net benefit 0; threshold probability 1; clinical utility 1 | All current uses reject standard DCA/clinical interpretation for raw scores |
| Threshold | cost 0; cost-weighted 0; `C_FN` 0; `C_FP` 0; beta 35 | Beta is preference weighting; costs are hypothetical only |
| XAI | data randomization 1; sanity 5; faithful 0; attention 2; reasoning 3; localization 18 | Correct control name; no faithfulness/clinical-reasoning inference |
| Radiography/DICOM | CT 0; HU 0; Hounsfield 0; dose 6; quantum 4; acquisition 11; windowing 0; VOI 5; Modality LUT 0; Rescale 1 | Planar-radiography and inspected-metadata boundaries are accurate |
| Calibration | calibrated 3; calibration 27; D-ECE 19; reliability 4; probability 8 | Detection-level only; sparse support and probability limitation disclosed |

## References and reporting audit

- 29 unique current-manuscript citation keys; 29/29 resolve; 29/29 have an
  explicit citation-audit row; zero duplicate BibTeX keys.
- 32 bibliography records total. The three records not used by the current
  manuscript are explicitly documented historical reading-list entries.
- No citation-audit row is classified unsupported or wrong attribution. The
  general GFL and mutable Ultralytics documentation rows remain transparently
  marked partial and are paired with the pinned local package/config/graph
  evidence used for the actual implementation claim.
- Live primary-source identity/scope checks were rerun for the reporting
  guidelines and DICOM standard: [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300),
  [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8),
  [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), and
  [DICOM PS3.3](https://dicom.nema.org/medical/dicom/current/output/html/part03.html).

## Exact commands/tests run and results

Commands were run from the repository root on 2026-09-01.

| Command | Result |
|---|---|
| `git rev-parse HEAD` | PASS — `9aaf414bdbab337a7c45283d4b32fe58b1e1700d` |
| `git status --short --branch` | `main...origin/main`; no tracked modification; pre-existing internal/untracked state and logs listed |
| `git log -1 --format='%H%n%ci%n%s'` | `Align manuscript with corrected evidence`, 2026-09-01 04:14:47 +0330 |
| `uv run --locked --extra cpu python -m pytest -q` | ENVIRONMENT FAILURE before collection — sandbox access denied to the external uv cache `.git`; not a test failure |
| `& .\.venv\Scripts\python.exe -m pytest -q` | PASS — 300 passed, 1 expected skip in 9.73 s |
| `& .\.venv\Scripts\python.exe -m ruff format --check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py scripts/build_scientific_artifact_manifest.py` | PASS — 100 files already formatted |
| `& .\.venv\Scripts\python.exe -m ruff check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py scripts/build_scientific_artifact_manifest.py` | PASS — all checks passed |
| `& .\.venv\Scripts\python.exe -m meddet_benchmark smoke configs/smoke.yaml` | PASS — CPU smoke status `smoke`, seed 17, config SHA-256 recorded |
| `& .\.venv\Scripts\python.exe scripts/verify_scientific_artifacts.py` | PASS — 46 artifacts, 110 present inputs, 0 unavailable external/ignored inputs in the committed critical set, 159 referenced result files |
| `& .\.venv\Scripts\python.exe scripts/verify_paper_claims.py` | PASS — 35 numerical claims |
| `$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.uv-cache-final-audit'); uv lock --check` | PASS — 97 packages resolved, lock current |
| Checkpoint rehash of `results/checkpoint_release_manifest.json` | PASS — 10/10 local files exist; size/hash/config bindings match; 962,924,817 bytes total; 0 Git-tracked; public URL null |
| Manuscript/supplement/README path and citation parsers | PASS — 13/13 manuscript links, 72/72 supplement links, 30/30 README configs, 21/21 README project modules, and 29/29 manuscript citation keys resolve |
| `git diff --check` | PASS — no whitespace error in tracked differences |

The six requested term-family searches were run literally as follows, followed
by contextual inspection of every match:

```powershell
rg -n -i 'significant|significance|robust|superior|better|uncertainty|confidence interval|p-value|seed|checkpoint' report/paper_draft.md
rg -n -i 'DCA|decision curve|net benefit|threshold probability|clinical utility' report/paper_draft.md
rg -n -i 'cost|cost-weighted|C_FN|C_FP|beta' report/paper_draft.md
rg -n -i 'data randomization|sanity|faithful|attention|reasoning|localization' report/paper_draft.md
rg -n -i 'CT|HU|Hounsfield|dose|quantum|acquisition|windowing|VOI|Modality LUT|Rescale' report/paper_draft.md
rg -n -i 'calibrated|calibration|D-ECE|reliability|probability' report/paper_draft.md
```

The checkpoint line in the command table denotes this exact read-only rehash
logic:

```powershell
$manifest = Get-Content results/checkpoint_release_manifest.json -Raw | ConvertFrom-Json
$bad = @(); $total = 0L
foreach ($c in $manifest.checkpoints) {
  $item = Get-Item -LiteralPath $c.source_path
  $hash = (Get-FileHash -LiteralPath $c.source_path -Algorithm SHA256).Hash.ToLowerInvariant()
  $tracked = git ls-files --error-unmatch -- $c.source_path 2>$null
  if ($item.Length -ne [int64]$c.size_bytes -or $hash -ne $c.sha256 -or $LASTEXITCODE -eq 0) { $bad += $c.source_path }
  $total += $item.Length
}
```

## Final statement

**NOT READY FOR SUBMISSION.** The committed software, evidence, manuscript
numbers, scope labels, and methodological language are clean enough for author
review, but the unresolved human-only declarations and final data/model
availability decision remain blocking. The appropriate handoff state is
**ready for author review**, not “submission ready.”
