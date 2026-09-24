# Final manuscript evidence and verification audit

Batch 51, 2026-09-24. Scientific draft for author review; not submission clearance.
The only editable journal manuscript is
[`report/paper_draft.md`](../report/paper_draft.md). Historical reports, PDFs and
the archived pre-external manuscript are provenance, not competing sources.
The starting scientific tree was `97eb9c476a848d11aac311d8da313e43a22489a0`.

## Scientific prerequisite gate

The gate used numerical artifacts and executable checks, not HANDOFF/CODEX
assertions. No training, new inference, threshold optimization, scientific
protocol amendment or regeneration of a settled result was performed.

| Evidence | Canonical record and verification | Finding |
|---|---|---|
| Internal benchmark and historical inference | [Phase 5 summary](../results/logs/phase5_evaluation/summary.json), [per-run metrics](../results/tables/detector_comparison_per_seed.csv), original scientific inventory and full local test suite | All five seeds, including 271, remain in each detector's unconditional endpoints. Historical conditional localization has its disclosed smaller support. |
| Exact-score FROC repair, Batch 42 | [v4 config](../configs/froc_exact_score_v4.yaml), its bound curves, budget tables and conservative bound; seven authorized-data FROC tests | Exact retained scores and unchanged matching; all ten runs, lower collection floor and residual support limit preserved. No full-frontier claim. |
| Five-run threshold selection, Batch 43 | [sensitivity config](../configs/threshold_selection_n5_validation_sensitivity.yaml), [summary](../results/logs/phase43_threshold_selection_n5_validation_sensitivity/summary.json), GPU preflight and artifact hashes | Validation-only n=5 selection remains secondary; historical n=3 thresholds remain primary. All five test runs remain represented. |
| Cohort characteristics, Batch 44 | [source audit](../data/manifests/rsna-pneumonia-5000-audit.json), [cohort table](../results/tables/rsna_cohort_characteristics.csv), bound header-extraction summary | Patient disjointness and all table counts checked. Nominal-year age assumption and missingness retained; no new fairness claim. |
| Matched timing, Batch 45 | [timing summary](../results/logs/phase45_inference_timing_v1/summary.json), tables/figure, `src.benchmark_inference --mode verify` | Saved experiment, 600 parity checks and provenance verified without re-timing. Hardware, boundary, AMP dtypes and technical repetitions remain explicit. |
| Internal paper focus, Batch 46 | [historical audit](INTERNAL_PAPER_FOCUS_AUDIT.md), exact archived paper and claim manifest | Baseline preserved; superseded by the current external-focused paper only for manuscript interpretation. |
| External protocol, Batch 47 | [protocol](VINDR_EXTERNAL_PROTOCOL.md), [v1 config](../configs/vindr_external_v1.yaml), [38-file freeze](VINDR_EXTERNAL_PROTOCOL_v1.sha256.json) | Original hashes and scientific settings preserved. Only exact local `Lung Opacity` is mapped; no outcome-guided ontology or endpoint change. |
| Adapter/preflight, Batch 48 | [adapter receipt](../results/vindr_external_v1/adapter_preflight.json) and full authorized source replay | All 3,000 images; 84 strict-positive images, 95 boxes, 2,916 strict negatives. Source integrity, decoding/round trips, loader/annotation equality and geometry pass. |
| Ten checkpoint runs, Batch 49 | [inference summary](../results/vindr_external_v1/inference_summary.json), full source/checkpoint/provenance and per-run metric replay | All ten checkpoint identities and all 3,000-image run outputs verified. Original numerical-correction evidence, floor/cap and inference precision remain bound. |
| Statistics/transportability, Batch 50 | [statistics summary](../results/vindr_external_v1/statistics/summary.json), full deterministic bootstrap replay | 130 input bindings, 20 internal/external point rows, separate 2,000-draw analyses and 44 interval rows verified against saved results. No new statistical test. |

The external inference summary SHA-256 remains
`d15070ebcafdfb2711b07db5090e67ef556b60d0042c87ac047b243bdf8fc656`;
the statistics summary remains
`a5267d223102c2312e0b568e06c7e4bf2f6157175e989d070afc943a23942f0b`.
The original scientific inventory, frozen configs, original sidecar, checkpoint
identities and numerical result files are unchanged.

## Manuscript traceability and claim boundaries

[`paper_claim_sources.yaml`](../report/paper_claim_sources.yaml) contains 417
exact numerical bindings and 33 semantic guards. CSV cells, JSON values and
allow-listed arithmetic use explicit rounding tolerances. This automated gate
is complemented by a whole-paper scientific reading: regex checks alone cannot
establish correct interpretation. Protocol constants and software identities
are traceable to the configs and provenance below.

| Manuscript material | Canonical sources / deterministic route |
|---|---|
| Abstract and §§4.6--4.8 | [transport comparisons](../results/vindr_external_v1/statistics/transport_comparison.csv), [intervals](../results/vindr_external_v1/statistics/intervals.csv), [policy sensitivity](../results/vindr_external_v1/statistics/threshold_policy_comparison.csv), [score support](../results/vindr_external_v1/statistics/score_summaries.csv) and the two external summaries |
| Table 1; §3.1 cohort, source totals, splits, annotation integrity | Source audit, split manifests, cohort table and Phase 44 summary; README dataset preparation/header commands |
| Table 2; §4.1 and seed-271 behavior | Phase 5 aggregate/per-run comparison and historical threshold-sweep evidence |
| Figure 1a/1b; §4.2 operating points | Five-run PR/F1 tables, historical n=3 selection and Batch 43 n=5 sensitivity; README §§5a--5b and Batch 43 commands |
| §4.3 internal exact-score FROC and no-reversal bound | Batch 42 v4 budget, comparison and floor-bound tables, source-bound lower-floor predictions; README §5c |
| Table 3; §4.4 historical uncertainty | [statistical comparison](../results/tables/statistical_clean_comparison.csv), [method](STATISTICAL_ANALYSIS.md), independently resampled runs and archived conditional tests; README statistics commands |
| Table 4; Figure 3; §3.7/4.5 timing, parameter counts and ratio | Phase 45 summary and [timing table](../results/tables/inference_timing_v1.csv), Phase 5 model counts; ratio computed from measured FPS |
| Table 5 external AP; Table 6 frozen policy | Batch 50 comparison/interval/policy tables; equal-run means, sample SDs and marginal training-procedure contrasts |
| Figure 2, Table 7; external support bounds | Batch 50 FROC plot/intervals and Batch 49 run support; replace only the single floor-limited contribution with sensitivity 1 to obtain each conservative aggregate upper bound |
| Figure 4, emissions and score population | Batch 50 threshold-transport plot, score/policy tables and Batch 49 run counts; zero emissions remain zero in the declared unconditional metric convention |
| §§3.2--3.6 protocol constants and numerical implementation | Detector resolved configs, [external scientific config](../configs/vindr_external_v1.yaml), [operation config](../configs/vindr_inference_v1.yaml), [statistical config](../configs/vindr_statistics_v1.yaml), adapter and inference provenance |
| Literature and reporting statements | [19-key source audit](CITATION_AUDIT.md), 40-entry shared bibliography, [44-item CLAIM crosswalk](REPORTING_CHECKLIST.md) |

Exact runnable commands, including original generating commands, remain in
[README](../README.md#final-manuscript-verification-batch-51); this table is
an index, not a second analysis implementation.

The final interpretation separates ranking from threshold behavior, observed
FROC support, training variability, matched implementation speed and external
transport. “Preserved ordering is not preserved performance” is qualified as
the observed equal-run AP direction. Four of five external FROC contrasts and
all four primary threshold contrasts include zero. The upper-budget external
missing-support bound permits a reversal; substantial Faster R-CNN cap
saturation remains a separate limitation. Low external FP/image is accompanied
by collapsed recall/emission. These results are not identical-task validation,
a dataset-factor causal attribution, or clinical validation.

Primary independent detector-run uncertainty is separate from fixed-checkpoint
sensitivity. Same-number seeds are not paired training replicates. Historical
n=3 threshold provenance and post-hoc n=5 sensitivity remain distinct. No
simultaneous interval coverage, new significance classification or unperformed
cross-dataset interaction test is implied. All adverse runs remain included.
The internal FP32/external bfloat16 YOLO inference difference qualifies Methods,
Results and Limitations. The one-ULP Faster R-CNN upper-bound correction is
summarized in Methods and documented in supplementary provenance.

## Portable CI and authorized scientific verification

An isolated export of the starting Git tree reproduced seven failures/errors
from full-data exact-FROC tests because the licensed RSNA annotation file is
absent. The repair marks exactly those seven tests `scientific_data` and
deselects them by default. Synthetic FROC tests and committed aggregate checks
stay portable. `--run-scientific` includes the original assertions; missing
authorized inputs still fail. No scientific check was weakened or deleted.

The public artifact gate also exposed Git's LF normalization of the original
training and validation split CSVs, whose frozen cohort bindings use CRLF bytes.
Both local files match their original recorded hashes, and replacing CRLF with
LF gives byte-for-byte equality to their starting Git blobs. Thus the rows,
values and partition membership are identical. Two narrow `.gitattributes`
exceptions now retain their original bytes in Git, matching the existing
timing/external artifact convention. Neither a scientific hash nor a split
definition was changed; these two intended byte-preservation diffs must be
included with the public CI repair.

The original external freeze also binds two editorial context files. Their
exact bytes are now archived under [report/provenance/batch46](../report/provenance/batch46/).
The read-only [`verify_frozen_external`](../scripts/verify_frozen_external.py)
wrapper validates the exact allowlist, original roles/hashes and archive
containment, then resolves reads/stat calls for only those two paths to those
bytes. It neither swaps files on disk nor redirects scientific paths. Four
regression tests cover byte/size identity, write refusal and restoration,
tamper rejection, scientific-path rejection and unchanged internal entries.
The existing 38-file freeze test still checks all 38 under this historical view.
Current manuscript claims are checked separately.

The [publication inventory](../results/publication_artifact_manifest.json)
extends the unchanged 72-entry internal inventory to 83 artifacts. All external
public summaries, tables and figures are required; private inputs are explicitly
absent-permitted in portable CI and hash-checked when present. The builder is a
deliberate maintenance command, never a CI hash-refresh step. Full external
source/metric/bootstrap verification requires authorized local inputs. The
historical absolute adapter-source binding additionally scopes full inference
verification to its recorded project location; it is not silently rewritten.

The isolated checkout was exported from the starting Git tree and overlaid with
the 30 intended public files, including the two split-byte preservation changes.
Only tracked `.gitkeep` placeholders existed in its raw/processed/checkpoint
directories; local orchestration, licensed images, annotations and checkpoints
were absent. Its own CPU environment was installed from the unchanged lock.
Every `run` command in `.github/workflows/ci.yml` then passed on Windows, with
offline cache use and no private-data environment overrides. The configured
Ubuntu runner was not executed locally.

| Final check | Outcome |
|---|---|
| Managed Python / lock / fresh CPU installation | Python 3.11.15; 97 packages resolved, 79 installed; unchanged lock passes |
| Portable default tests in isolated checkout | 438 passed, 1 expected metadata-only-environment skip, 7 scientific-data tests deselected |
| Full authorized local suite | 445 passed, the same 1 expected skip; fresh explicit temporary directory |
| Ruff formatting and lint | 124 files formatted; lint passes |
| Public artifact gate | 83 artifacts, 353 present input bindings, 227 explicitly unavailable external/ignored bindings, 201 result references |
| Authorized artifact gate | Same 83 artifacts and 201 references; all 580 input bindings present and verified |
| Current paper claims | 417 numerical bindings and all 33 semantic guards pass |
| Original external freeze | All 38 bindings pass with exact archived editorial context |
| External source/checkpoint/metric replay | All ten runs and all 3,000 images per run pass; unchanged summary hash |
| Full deterministic statistical replay | All 20 point rows and 2,000 draws per cohort reproduced; 44 interval rows and 130 input bindings pass |
| Bibliography / reporting | 40 unique entries; current 19 keys and preserved 19/27 keys resolve; all 44 CLAIM items reviewed without converting gaps to compliance claims |
| Links, figures and whitespace | 226 publication file/anchor links resolve; transport plots visually inspected; `git diff --check` passes |
| Package smoke | Synthetic CPU smoke passes in isolated environment |

An initial authorized test invocation encountered Windows permissions on an old
pytest temporary directory; the fresh-path rerun above passed. The first online
dependency install was stopped after slow downloads, then the clean environment
was installed successfully from the existing package cache. Access to uv's
existing cache/managed-Python lock required the tool's sandbox escalation;
no scientific check was bypassed. Local execution logs remain under ignored
`tmp/batch51/`, including `clean-ci.log`, `final-authorized-tests.log`,
`final-external-verify.log`, and `bootstrap-replay.log`.

The reviewed canonical manuscript SHA-256 is
`1f3b5fac3809ff287ddc952ca0432c405e7c891c1bfc85d7cb009cb6c40330d2`.

## Remaining author and submission work

The seven explicit declarations in §8 require author-controlled facts. No
ethics exemption, consent waiver, funding/conflict absence, authorship or
patient/public involvement is inferred. Data-access attestation is separate
from an institutional ethics determination. Checkpoint public URLs and final
archive/release details remain unresolved.

The crosswalk retains honest gaps in upstream acquisition/annotation reporting,
external demographics, subgroup/fairness analysis and participant-flow graphics.
No reader study, prospective evaluation or deployment validation was added.
The eventual journal format, official checklist and paginated references still
require submission preparation. This batch stops for user review; no commit,
push, rendered-PDF replacement or later batch is included.
