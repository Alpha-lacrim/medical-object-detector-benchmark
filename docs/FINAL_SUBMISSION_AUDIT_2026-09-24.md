# Final adversarial submission audit

Batch 52; 2026-09-24. **READY FOR AUTHOR REVIEW**, including the final PDF
inspection recorded below. This is not journal-submission clearance or an
assessment that a particular journal/quartile will accept the work.

The historical [September 1 audit](FINAL_SUBMISSION_AUDIT_2026-09-01.md) is
preserved byte-for-byte. The [Batch 51 audit](FINAL_MANUSCRIPT_AUDIT.md) remains
the prior manuscript record. This report supersedes their current-status claims.

## Repository state and canonical evidence

| Item | Audited state |
|---|---|
| Branch / HEAD | `main`, `ac6975df799a20c55d97204ddcdcbd0dab6675f9`; aligned with origin/main at startup |
| Index | Empty at startup; no staging, commit, push, tag, release or submission in this batch |
| Starting tracked change | Unrelated README Windows wget link; preserved |
| Starting untracked state | Local AGENTS/spec/batches/audits and historical aborted/smoke/orchestration logs; full list saved locally in `tmp/batch52/start-status.txt` |
| Current manuscript | `report/paper_draft.md` only; Batch 52 adds tied-score explanation, explicit support limits in captions and unresolved author metadata |
| Other manuscripts/PDF | `report/report.md`, long alternate Markdown and `A_Controlled_Comparative_Study___article.pdf` remain historical/noncanonical and unchanged |
| New article derivative | `output/pdf/rsna_vindr_article.pdf`, built from canonical Markdown/BibTeX; source/output identities in `report/paper_build_manifest.json` |
| Internal evidence | `results/scientific_artifact_manifest.json`; exact-score v4 FROC, post-hoc n=5 validation, cohort and matched timing artifacts, linked through `results/publication_artifact_manifest.json` |
| Batch 49 final experiment | `results/vindr_external_v1/adapter_preflight.json`, `inference_summary.json`; ten completed 3,000-image runs and all bound local inputs actually replayed |
| Batch 50 final statistics | `results/vindr_external_v1/statistics/summary.json`; `per_run.csv`, `intervals.csv`, `transport_comparison.csv`, `threshold_policy_comparison.csv`, `score_summaries.csv`; three figures; all exist and their 20 run rows/44 interval rows were replayed |
| Protected freeze | All 38 original bindings pass; only two exact historical editorial reads are redirected to the immutable Batch 46 archive |
| CI | `.github/workflows/ci.yml`, `foundation-ci`, push/PR, read-only permissions, Ubuntu/Windows matrix; pinned checkout/setup-uv actions and uv 0.11.27 |

External inference summary SHA-256:
`d15070ebcafdfb2711b07db5090e67ef556b60d0042c87ac047b243bdf8fc656`.
External statistics summary SHA-256:
`a5267d223102c2312e0b568e06c7e4bf2f6157175e989d070afc943a23942f0b`.
Both remained unchanged. No expected scientific hash was refreshed.

Public evidence is code/config/protocol, internal committed evidence and
nonidentifying external aggregates. Raw licensed images, external annotations,
image IDs, image-linked predictions, bootstrap working files and all ten
checkpoint binaries remain local/ignored. No public checkpoint URL exists.
The PDF's supporting-document links use the audited immutable public revision;
this is a technical link target, not a completed author availability declaration.

## BLOCKING FAILURES

No unresolved scientific or portable-repository failure was found in the
executed audit. No new experiment or methodological redesign was needed.
The adverse external outcome is a result, not a reason to relax verification.

Two narrow reporting issues were corrected: the FROC method now explicitly
states that tied-score groups enter together, and Figure 2/Table 7 captions
carry the upper-budget reversal and detection-cap limits. No scientific value,
candidate, threshold, inference path or statistical method changed.
The abstract's slash-separated partition wording was expanded to permit line
wrapping, and three exact text locators were updated without changing sources
or tolerances. A stale commented README command naming an unimplemented planned
module was replaced by a pointer to the completed statistical workflow.

## AUTHOR ACTION REQUIRED

- Confirm author names/order, affiliations, corresponding author and CRediT roles.
- Resolve funding/support and funder roles, conflicts, institutional ethics/data-use
  determination, consent applicability, and patient/public involvement. Access
  attestation does not establish an ethics exemption or consent waiver.
- Finalize source-access wording, exact repository/archive/release identifier,
  DOI if obtained, and checkpoint publication/availability decision. Do not
  describe private derivatives or unreleased weights as public.
- Select the journal, meet its article/abstract/figure/reference requirements,
  and complete its official reporting checklist against the final page numbers.

All seven declaration fields remain conspicuous in the source/PDF. See
[AUTHOR_DECLARATIONS_TODO.md](AUTHOR_DECLARATIONS_TODO.md). No author-controlled
fact was inferred from the code or dataset documentation.

## SHOULD-FIX

The CLAIM crosswalk retains gaps in upstream accrual/acquisition and annotation
detail, external demographics/clinical descriptors, subgroup reporting and a
cohort-flow graphic. A flow graphic can be prepared from known counts in a later
authorized editorial task. Unknown metadata and absent subgroup experiments must
remain limitations rather than invented answers. The current external uncertainty
cannot correct unknown patient dependence. Five training runs remain a small
empirical sample, and the usual augmentation-rich YOLO recipe was not evaluated.

The older alternate PDF is easy to mistake for current work; documentation now
identifies the new canonical derivative explicitly. The old file is preserved
for provenance. The full source verifier still binds the original absolute local
adapter location: relocation needs explicit provenance treatment, not hash updates.

## VERIFIED CLEAN AREAS

The full manuscript was read in context, including favorable and unfavorable
comparisons. Its task is opacity localization, not pneumonia diagnosis. It makes
no universal detector-family, causal-domain, clinical validation, deployment,
patient-benefit or state-of-the-art claim. AP is not presented as sufficient for
utility; numerical confidence scales are not interchangeable. The phrase
“Preserved ordering is not preserved performance” is explicitly restricted to
observed equal-run AP direction. The seed-17 external reversal at larger FROC
budgets and inconclusive checkpoint-conditional AP contrasts remain visible.

Historical internal intervals resample patient groups and independently selected
detector runs. Checkpoint-conditional p-values are separate, with the historical
four complete labels used only for conditional localization permutation; those
labels are not paired stochastic training replicates. New external intervals
resample images and independent runs; secondary seed-17 intervals hold checkpoints
fixed. Marginal intervals do not confer simultaneous/familywise coverage. No
cross-dataset interaction test was performed. Change labels remain descriptive.

D-ECE describes emitted detections under its floor/bin/support rules, excludes
missed targets and is not patient-risk calibration. No calibrator was fitted.
Grad-CAM/control results do not prove clinically meaningful reasoning. Synthetic
corruption and stored-array acquisition stress do not validate scanner or dose
models. Conventional DCA/net benefit was not performed.

## CLEAN-CHECKOUT CI STATUS

**PASS.** [GitHub run 35987188952](https://github.com/Alpha-lacrim/medical-object-detector-benchmark/actions/runs/35987188952)
is successful for this exact HEAD on **both Ubuntu and Windows**. Its job/step
receipt was fetched with `gh run view ... --json headSha,conclusion,status,url,jobs`.

A new `git archive HEAD` export under `tmp/batch52/clean-checkout` independently
ran every workflow `run` command on Windows in its own locked CPU environment.
No private-data environment overrides were inherited. Raw/processed/checkpoint
directories contained only committed placeholders. Results: **438 passed,
1 expected metadata-only-environment skip, 7 scientific-data tests deselected**.
Public evidence gate: 83 artifacts, 353 present input bindings, 227 explicitly
unavailable external/ignored bindings and 201 result references.

The seven marked tests retain their assertions and fail on missing data when
`--run-scientific` is requested. Tests were neither deleted nor weakened. The
public verifier requires every public artifact; absent-permitted bindings are
restricted inputs, not missing public results. Full authorized replay is separate.

Initial local setup encountered sandbox denial on uv's managed-Python lock,
then an incomplete optional offline cache. The approved retry used the normal
existing cache and completed all commands. These were environment setup failures;
they were not suppressed test failures. The final log is `tmp/batch52/clean-ci.log`.

After the final wording/build edits, the 21 intended public files were overlaid
onto that isolated checkout; the unrelated README wget hunk was excluded.
Private inputs remained absent. Using its own locked CPU interpreter, portable
pytest again passed **438/1 skip/7 deselected**, and all artifact/claim/freeze,
bibliography, package-smoke and full `src tests scripts` Ruff checks passed
(130 formatted files). The receipt is `tmp/batch52/final-candidate-ci.json` and
log `final-candidate-ci.log`. The initial rerun could not remove the earlier
elevated run's temporary directory; a fresh nested path then lacked its parent.
The successful command used `python -m pytest -q
--basetemp=.pytest-tmp-batch52-candidate -p no:cacheprovider`. Both setup-failure
logs were preserved; no assertion or dataset requirement changed. The final
local edits are not pushed, so the GitHub result above applies to the recorded
HEAD, while the isolated candidate check covers the final scientific wording.

## FULL-DATA SCIENTIFIC VERIFICATION STATUS

**PASS.** Authorized local pytest: **445 passed, 1 expected skip**, including all
seven scientific integration tests. The full external source/checkpoint/provenance
and metric replay verified ten runs with 3,000 images each. Full deterministic
statistical replay reproduced all 20 point rows and both separate 2,000-draw
analyses, 44 intervals and 130 source bindings with the original summary hash.
Local publication verification found all 580 input bindings present and correct.
Validation-only threshold preflight and saved timing verification passed.

GPU-dependent training/inference/timing were not repeated; this is an audit of
the saved experiment and its authorized replay. The GPU environment was used for
its environment-bound verifier/preflight, without new prediction collection.

## NUMERICAL TRACEABILITY

All **417** manuscript numerical bindings and **33** semantic guards pass. The
[complete fresh trace](SUBMISSION_NUMERICAL_TRACE.csv) records manuscript line,
value, full JSON pointer or CSV filter/column, every calculation operand,
unrounded source value, rounding, tolerance and PASS for each binding.
`python -m scripts.export_paper_trace --manifest report/paper_claim_sources.yaml
--output docs/SUBMISSION_NUMERICAL_TRACE.csv` recreates it after verification.

The representative adversarial selection below covers cohort, AP, operating
points, FROC, uncertainty, efficiency, external support, score behavior and bounds.
It is backed by the independent source/metric/bootstrap replay above, not only
agreement between manuscript and an aggregate. Decimal places (dp) use the
manifest's stated rounding/tolerance. A/B are Faster R-CNN/YOLO11s.

| # / claim | Manuscript value | Source / row / column or derivation | Unrounded source | Rounding | Result |
|---|---:|---|---:|---|---|
| 1. Cohort: full source (`source_radiograph_count`) | 26,684 | `data/manifests/rsna-pneumonia-5000-audit.json` / input_counts.annotation_exam_ids | 26684.0 | integer | PASS |
| 2. Cohort: selected patient groups (`subset_patient_group_count`) | 2,136 | `data/manifests/rsna-pneumonia-5000-audit.json` / subsample.selected_groups | 2136.0 | integer | PASS |
| 3. Cohort: usable nominal ages (`cohort_total_age_summary_n`) | 4,999 | `results/tables/rsna_cohort_characteristics.csv` [split=total] age_summary_n | 4999.0 | integer | PASS |
| 4. Internal test boxes (`test_reference_box_count`) | 268 | `results/tables/detector_comparison_per_seed.csv` [detector=faster_rcnn, seed=17] target_count | 268.0 | integer | PASS |
| 5. Internal AP50 A (`table_64`) | 0.304224 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=ap50] internal_a_mean | 0.3042238071036434 | displayed precision | PASS |
| 6. Internal AP50 B (`table_66`) | 0.162612 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=ap50] internal_b_mean | 0.16261194018585284 | displayed precision | PASS |
| 7. Internal shared-cutoff precision difference CI lower (`precision_difference_ci_low`) | -0.2423 | `results/tables/statistical_clean_comparison.csv` [metric=precision] difference_ci_low | -0.2422616796943514 | 4 decimal places | PASS |
| 8. Internal shared-cutoff precision difference CI upper (`precision_difference_ci_high`) | 0.0553 | `results/tables/statistical_clean_comparison.csv` [metric=precision] difference_ci_high | 0.05531985232606028 | 4 decimal places | PASS |
| 9. Internal checkpoint-conditional precision Holm p (`precision_holm_p_value`) | 0.0020 | `results/tables/statistical_clean_comparison.csv` [metric=precision] p_value_holm_conditional_on_observed_checkpoints | 0.001999600079984003 | 4 decimal places | PASS |
| 10. Historical threshold A (`faster_selected_threshold`) | 0.69 | `results/tables/selected_operating_points_n5_sensitivity.csv` [detector=faster_rcnn] selected_threshold | 0.69 | exact displayed threshold | PASS |
| 11. Historical threshold B (`yolo_selected_threshold`) | 0.05 | `results/tables/selected_operating_points_n5_sensitivity.csv` [detector=yolo11s] selected_threshold | 0.05 | exact displayed threshold | PASS |
| 12. Post-hoc threshold B (`posthoc_yolo_selected_threshold`) | 0.01 | `results/tables/threshold_selection_test_operating_points_n5_validation_sensitivity.csv` [detector=yolo11s, selection_scope=posthoc_n5_validation_sensitivity] selected_threshold | 0.01 | exact displayed threshold | PASS |
| 13. Internal historical recall A (`table_93`) | 0.350746 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=primary_recall] internal_a_mean | 0.35074626865671643 | displayed precision | PASS |
| 14. Internal historical recall B (`table_94`) | 0.194776 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=primary_recall] internal_b_mean | 0.19477611940298506 | displayed precision | PASS |
| 15. Internal FROC at 2 A (`faster_froc_sensitivity_at_2`) | 0.6978 | `results/tables/froc_operating_points_exact_score_v4.csv` [detector=faster_rcnn, fp_per_image_budget=2.0] sensitivity | 0.6977611940298508 | 4 decimal places | PASS |
| 16. Internal FROC at 2 B (`yolo_froc_sensitivity_at_2`) | 0.6090 | `results/tables/froc_operating_points_exact_score_v4.csv` [detector=yolo11s, fp_per_image_budget=2.0] sensitivity | 0.608955223880597 | 4 decimal places | PASS |
| 17. Internal floor endpoint seed137 (`yolo_seed137_froc_floor_endpoint_fp_per_image`) | 1.9907 | `results/tables/froc_operating_points_per_seed_exact_score_v4.csv` [detector=yolo11s, fp_per_image_budget=2.0, seed=137] candidate_floor_endpoint_fp_per_image | 1.9906666666666666 | 4 decimal places | PASS |
| 18. Internal conservative bound (`yolo_froc_conservative_upper_at_2`) | 0.6963 | `results/tables/froc_incomplete_frontier_bounds_v4.csv` [detector=yolo11s, fp_per_image_budget=2.0] conservative_upper_aggregate_sensitivity | 0.6962686567164178 | 4 decimal places | PASS |
| 19. Internal seed271 maximum score (`yolo_seed271_maximum_score`) | 0.0412735 | `results/tables/detector_comparison_per_seed.csv` [detector=yolo11s, seed=271] maximum_prediction_score | 0.04127352312207222 | 7 decimal places | PASS |
| 20. Matched FPS A (`timing_v1_faster_rcnn_fps`) | 20.91 | `results/tables/inference_timing_v1.csv` [detector=faster_rcnn] fps | 20.91092826987137 | 2 decimal places | PASS |
| 21. Matched FPS B (`timing_v1_yolo11s_fps`) | 57.56 | `results/tables/inference_timing_v1.csv` [detector=yolo11s] fps | 57.55529287767316 | 2 decimal places | PASS |
| 22. Matched latency A (`timing_v1_faster_rcnn_median_latency_ms`) | 47.20 | `results/tables/inference_timing_v1.csv` [detector=faster_rcnn] median_latency_ms | 47.203950001858175 | 2 decimal places | PASS |
| 23. Measured throughput ratio (`approximate_throughput_ratio`) | 2.75 | divide(`results/tables/inference_timing_v1.csv` [detector=yolo11s] fps, `results/tables/inference_timing_v1.csv` [detector=faster_rcnn] fps) | 2.7524025779668175 | 2 decimal places | PASS |
| 24. External images (`external_count_images_0`) | 3,000 | `results/vindr_external_v1/statistics/summary.json` / populations.external.images | 3000.0 | displayed precision | PASS |
| 25. External positive images (`external_count_positive_images_0`) | 84 | `results/vindr_external_v1/statistics/summary.json` / populations.external.positive_images | 84.0 | displayed precision | PASS |
| 26. External target boxes (`external_count_target_boxes_0`) | 95 | `results/vindr_external_v1/statistics/summary.json` / populations.external.target_boxes | 95.0 | displayed precision | PASS |
| 27. External strict negatives (`external_negative_count`) | 2,916 | `results/vindr_external_v1/adapter_preflight.json` / strict_negative_count | 2916.0 | displayed precision | PASS |
| 28. External AP50 A (`table_68`) | 0.002505 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=ap50] external_a_mean | 0.0025050980158512295 | displayed precision | PASS |
| 29. External AP50 B (`table_70`) | 0.000501 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=ap50] external_b_mean | 0.0005006936455752758 | displayed precision | PASS |
| 30. External AP50 difference CI lower (`table_73`) | 0.000331 | `results/vindr_external_v1/statistics/intervals.csv` [dataset=external, estimand=training_procedure, metric=ap50] difference_ci_low | 0.0003306033126865859 | displayed precision | PASS |
| 31. External AP50 difference CI upper (`table_74`) | 0.006689 | `results/vindr_external_v1/statistics/intervals.csv` [dataset=external, estimand=training_procedure, metric=ap50] difference_ci_high | 0.006689352372639503 | displayed precision | PASS |
| 32. External historical recall A (`table_95`) | 0.018947 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=primary_recall] external_a_mean | 0.018947368421052633 | displayed precision | PASS |
| 33. External historical recall B (`table_96`) | 0.004211 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=primary_recall] external_b_mean | 0.004210526315789474 | displayed precision | PASS |
| 34. External recall contrast CI lower (`table_98`) | -0.004880 | `results/vindr_external_v1/statistics/intervals.csv` [dataset=external, estimand=training_procedure, metric=primary_recall] difference_ci_low | -0.004879554351099064 | displayed precision | PASS |
| 35. External recall contrast CI upper (`table_99`) | 0.045485 | `results/vindr_external_v1/statistics/intervals.csv` [dataset=external, estimand=training_procedure, metric=primary_recall] difference_ci_high | 0.04548484848484832 | displayed precision | PASS |
| 36. External primary FP/image A (`table_109`) | 0.047667 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=primary_fp_per_image] external_a_mean | 0.04766666666666667 | displayed precision | PASS |
| 37. External primary FP/image B (`table_110`) | 0.027067 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=primary_fp_per_image] external_b_mean | 0.027066666666666666 | displayed precision | PASS |
| 38. External secondary FP/image B (`secondary_fp_b`) | 0.060067 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=secondary_fp_per_image] external_b_mean | 0.06006666666666667 | displayed precision | PASS |
| 39. External YOLO seed271 secondary detections (`external_empty_seed271`) | 54 | `results/vindr_external_v1/statistics/per_run.csv` [dataset=external, detector=yolo11s, seed=271] secondary_fp | 54.0 | displayed precision | PASS |
| 40. External AP-support detections/image B (`ap_emissions_external_b`) | 0.217200 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=ap_scores_detections_per_image] external_b_mean | 0.21719999999999998 | displayed precision | PASS |
| 41. External AP-support empty images B (%) (`ap_empty_external_b`) | 87.073333 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=ap_scores_percent_zero_detection_images] external_b_mean | 87.07333333333334 | displayed precision | PASS |
| 42. External FROC .25 CI lower (`table_130`) | 0.005345 | `results/vindr_external_v1/statistics/intervals.csv` [dataset=external, estimand=training_procedure, metric=froc_0.25] difference_ci_low | 0.005345165505226504 | displayed precision | PASS |
| 43. External FROC 1 CI lower (`table_140`) | -0.020939 | `results/vindr_external_v1/statistics/intervals.csv` [dataset=external, estimand=training_procedure, metric=froc_1] difference_ci_low | -0.020938891637803044 | displayed precision | PASS |
| 44. External FROC 2 A (`table_142`) | 0.164211 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=froc_2] external_a_mean | 0.16421052631578947 | displayed precision | PASS |
| 45. External FROC 2 B (`table_143`) | 0.098947 | `results/vindr_external_v1/statistics/transport_comparison.csv` [metric=froc_2] external_b_mean | 0.09894736842105264 | displayed precision | PASS |
| 46. External seed137 floor FP/image (`external_seed137_floor`) | 0.793333 | `results/vindr_external_v1/statistics/per_run.csv` [dataset=external, detector=yolo11s, seed=137] floor_fp_per_image | 0.7933333333333333 | displayed precision | PASS |
| 47. External conservative bound at 1 (`external_bound_1`) | 0.258947 | `results/vindr_external_v1/statistics/transport_comparison.csv` metric=froc_1, external_b_mean + (1 - `per_run.csv` [external,yolo11s,137].froc_1) / 5; full operands in trace CSV | 0.25894736842105265 | displayed precision | PASS |
| 48. External conservative bound at 2 (`external_bound_2`) | 0.280000 | `results/vindr_external_v1/statistics/transport_comparison.csv` metric=froc_2, external_b_mean + (1 - `per_run.csv` [external,yolo11s,137].froc_2) / 5; full operands in trace CSV | 0.28 | displayed precision | PASS |
| 49. External cap images A seed17 (`external_cap_17`) | 3,000 | `results/vindr_external_v1/statistics/per_run.csv` [dataset=external, detector=faster_rcnn, seed=17] images_at_collection_cap | 3000.0 | displayed precision | PASS |
| 50. External cap images A seed42 (`external_cap_42`) | 2,635 | `results/vindr_external_v1/statistics/per_run.csv` [dataset=external, detector=faster_rcnn, seed=42] images_at_collection_cap | 2635.0 | displayed precision | PASS |
| 51. External cap images A seed314 (`external_cap_314`) | 209 | `results/vindr_external_v1/statistics/per_run.csv` [dataset=external, detector=faster_rcnn, seed=314] images_at_collection_cap | 209.0 | displayed precision | PASS |

## ANALYSIS-SCOPE SUMMARY

All internal analyses are retrospective unless explicitly describing the later
external freeze. “Five runs” always means 17/42/137/271/314 per detector.

| Analysis | Dataset / unit / support | Runs and status | Inference / threshold / retained support |
|---|---|---|---|
| Cohort construction/headers | RSNA 5,000 exams, 2,136 NIH patients; 1,136 positive exams, 1,812 boxes; train/val/test 3,500/750/750 | No model scope; retrospective descriptive | Disjoint patients 1,492/321/323; age nominal-year assumption, one excluded outlier; no fairness inference |
| Internal AP/shared cutoff | 750 exams, 323 patients; 169 positive, 268 boxes | All five; historical primary evidence | Equal-run means/SD; AP floor .001, cap 100; micro P/R/F1 at .25 |
| Conditional IoU/Dice | Same cohort; matched detections only | Five A/four defined B; seed 271 retained as undefined | Misses excluded; no all-target localization claim |
| Historical clean uncertainty | Same 323 patient clusters | Five/five unconditional; five/four defined localization | Patient/run hierarchical bootstrap; distinct fixed-checkpoint permutation with Holm over seven endpoints |
| PR/test threshold sensitivity | Same internal 750 | All five; descriptive retrospective | Official 101-position PR; 99 test thresholds .01–.99; no test-side selection |
| Historical threshold selection | RSNA validation 750 exams, 321 patients; 169 positive, 277 boxes | Three seeds 17/42/137; historical primary policy | Max equal-run validation F1, exact ties higher threshold; .69/.05 |
| n=5 threshold selection | Same validation cohort | Five; post-hoc secondary | Identical rule; .70/.01; grid-boundary limitation visible |
| Transported operating points | Internal 750/323 patients and external 3,000 images | All five for both policies | Historical policy primary; n=5 descriptive; no external selection/calibration; emissions not a benefit endpoint |
| Internal exact-score FROC | Internal 750; sensitivity denominator 268 boxes | Five; descriptive observed means; aligned intervals separate | Floor .00001/cap 100; exact tied-score groups; budgets .125/.25/.5/1/2; no interpolation/extrapolation |
| External AP/FROC | VinDr 3,000 images; 84 strict positives, 95 boxes; patient grouping unknown | Ten frozen checkpoints, all five per detector; external frozen v1 | AP .001; FROC .00001, cap 100; lower-bound contributions and cap saturation explicit |
| Aligned transport uncertainty | Internal 323 patients / external 3,000 released images separately | Five independently resampled runs per detector | 2,000 marginal draws per dataset; historical thresholds/support fixed; no interaction test |
| Fixed seed-17 sensitivity | Same cohorts/units | One checkpoint per detector | Observation-only intervals, secondary; not a training-procedure result |
| Score/emission distributions | Same cohorts; emitted detection is score unit, images are count denominator | All five, four separately defined support populations | Per-run summaries then equal-run aggregates; no probability calibration |
| Matched timing | Same 100 RSNA test images, batch 1 | Seed 17 only, three 100-call technical repeats | Decoded host RGB through CPU output; excludes I/O/decoding/setup; explicit f16/bf16 AMP; 600 parity checks |
| Historical compute/Pareto | Internal run clouds, historical implementation timers | n=3 or n=5 exactly as labeled; supplementary | Descriptive; never relabeled as matched timing or statistical utility; incomplete GFLOPs not headline evidence |
| D-ECE/support sensitivity | Internal 750; emitted post-NMS detections | All five; supplementary descriptive | .001 primary floor; five bins per axis, min-cell eight; missed targets excluded; no patient-risk claim |
| F-beta/linear loss | Validation 750/321 patients | Historical n=3, secondary sensitivity | Preference beta / hypothetical exchangeable-box loss; not measured harm; not propagated to test |
| Corruption/acquisition stress | Internal fixed 300 images, 183 patients; 68 positive, 111 boxes | One seed-17 checkpoint each | Synthetic descriptive scope; corruption patient-cluster inference is checkpoint-conditional |
| Grad-CAM localization | Same 300; 111 boxes/68 positive images; 232 zero-box images excluded from box metrics | One seed-17 each | 40x40 maps, proxy targets when missed; excludes no-energy cases as disclosed; not clinical reasoning |
| XAI controls | Nested 50 images, 41 patients | Seed-17 checkpoints, one control draw per stage | Descriptive randomization/input controls; degenerate pairs disclosed; no randomized-training/data-label test |
| Archived raw-score utility | Internal 750/323 patients | Five; nonstandard historical sensitivity | Not conventional DCA, probability calibration or clinical net benefit |

## INTERNAL VS EXTERNAL PROTOCOL SUMMARY

| Dimension | RSNA internal | VinDr external |
|---|---|---|
| Target | Challenge pneumonia-like `Lung Opacity` boxes | Only exact local `Lung Opacity`; no consolidation/infiltration/atelectasis/ILD/global Pneumonia union; other findings not ignored |
| Cohort | Held-out 750 of selected 5,000; 169 positive, 268 boxes | Entire official test release: 3,000, 84 positive, 95 boxes, 2,916 strict negatives |
| Unit | Exams clustered within 323 NIH patients | Released study/image; no defensible patient key or patient-count claim |
| Checkpoints | Validation mAP-selected five per detector | Same exact ten SHA-bound checkpoints; no adaptation or selection |
| Thresholds | Historical validation n=3 primary; n=5 post-hoc secondary | Both frozen RSNA policies applied unchanged; no VinDr optimization/calibration |
| AP / FROC | .001 / .00001 floor; cap 100; IoU .50 FROC; official COCO AP | Same endpoints/cap; native filters strict > floor, common evaluator >= retained score |
| Numerical path | Historical YOLO accuracy FP32; Faster configured AMP | Verified YOLO bfloat16 / Faster float16 AMP; no dataset-only causal explanation |
| Resampling | Patient groups jointly across runs/detectors; runs independent within detector | Images jointly across runs/detectors; runs independent within detector |
| Interpretation | Conditional on fixed training dataset/recipe | Cross-dataset/cross-ontology transport, not identical-task or clinical validation |

## EXTERNAL FROC SUPPORT / CAP STATUS

Fresh replay and code/tests confirm exact retained-score events, stable matching,
whole tied-score entry, fixed budgets and no interpolation/extrapolation. Native
floor equality and cap are explicitly tested. YOLO seed 137 ends at **0.793333
FP/image**, below 1 and 2. Replacing only that missing-support contribution by
1 gives B aggregate upper bounds **0.258947 / 0.280000**, versus observed A
**0.117895 / 0.164211**. Reversal beyond retained candidates is not ruled out.
These are conservative mathematical support bounds, not CIs or attainable
performance estimates. They do not repair any cap limit. Internally, the analogous
2-FP bound **0.6963 < 0.6978** is a fixed-run floor-only result and does not transport.

Faster R-CNN hits cap 100 on **3,000 / 2,635 / 3,000 / 3,000 / 209** external
images for seeds 17/42/137/271/314; YOLO hits it on none. The canonical figure
and Table 7 captions now explicitly carry this additional limit. Only the
external .25-FP contrast CI excludes zero; four others include zero. Neither
observed mean ordering nor the marginal intervals establish an unrestricted
frontier or simultaneous detector ordering. No floor/cap change was made.

## FROZEN-THRESHOLD TRANSPORT STATUS

Historical thresholds .69/.05 were derived from RSNA validation seeds 17/42/137.
The post-hoc .70/.01 rule used all five RSNA validation runs; all five test runs
remain under each policy. No VinDr threshold search or calibration fit occurred.
External primary recall is **.018947/.004211**, precision **.010830/.004271**.
All four historical operating-metric contrast CIs include zero. Lower external
FP/image and emitted counts coexist with failed recall transport and are not
treated as improvement. Under the secondary policy, external FP/image order
reverses (.044200/.060067), but recall remains extremely low. YOLO seed 271's
zero detections at .05 and 54 false positives at .01 remain included.

## Implementation and efficiency audit

`restore_native_bounds` only corrects one-ULP float32 Faster R-CNN overshoots
above native upper bounds. Negative coordinates, larger excursions, wrong
detector/dtype and tampered raw repair records fail tests. It preserves interior
coordinates, scores, labels and counts. Full replay verifies preserved raw repair
records plus the initial failed-attempt archive, which preceded completed external
metrics. No frozen setting was changed; this is not outcome-guided protocol tuning.

Matched timing verification checks its saved raw intervals, environment, order,
warmups, preservation bindings and all 600 parity records. The reported
20.91/57.56 FPS and 47.20/17.29 ms medians describe decoded-host input to CPU
outputs, batch 1, one seed-17 checkpoint per detector, three technical repeats,
the specified RTX 4060 laptop/software state and different configured AMP dtypes.
No disk/decoding/clinical workflow cost is included; no universal speed law or
headline inference from incomplete profiler GFLOPs appears in the paper.

## References, reporting and availability

Offline bibliography checking resolves 19 current, 19 historical-report and 27
alternate-manuscript citation keys in 40 unique records. The current reference
set supports methods/context, not imported model-performance numbers.
Fresh primary-source checks read [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300),
the [ECCV calibration paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf),
the [CVF calibration paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.pdf),
the [official VinDr release](https://physionet.org/content/vindr-cxr/1.0.0/),
[Wu et al.](https://www.nature.com/articles/s41598-024-52156-7), and
the [coauthor-uploaded Chinnam paper](https://www.researchgate.net/publication/404406332_Automated_Pneumonia_Detection_in_Medical_X-Rays_Using_Modern_Object_Detectors).
IEEE's primary landing page remained bot-gated; the Nature VinDr methodology
route failed extraction in this session. Those access limits do not become fresh
full-text verification claims; the prior source audit and official release remain
explicit. The official release still requests the recorded PhysioNet 2026 citation.

All 44 CLAIM rows were reviewed for the current scope. The crosswalk is an
internal reporting aid, not certification or the journal's official checklist.
Its No/NA explanations and declaration/availability gaps remain honest. Public
link targets, cited figures/tables, config/module paths and private boundaries
were checked; no promised public external bundle or checkpoint download was invented.

## COMMANDS / TESTS EXECUTED

All commands ran from the root unless specified. `CPU` below means the existing
`.venv/Scripts/python.exe`; `GPUENV` means the documented `torch-gpu` interpreter.
Exact environment-specific commands are in README. Logs remain ignored locally
under `tmp/batch52/`; committed commands and immutable sources allow repetition.

| Command / execution | Outcome |
|---|---|
| `git status --porcelain=v1`, `git rev-parse HEAD`, `git diff --cached`, `git diff --binary` | Startup state recorded; no staged content |
| `gh run view 35987188952 --json headSha,conclusion,status,url,jobs` | Exact-HEAD Ubuntu and Windows success |
| `git archive --format=zip -o tmp/batch52/checkout.zip HEAD`; extract; all workflow run commands in fresh checkout | Locked Python 3.11.15 CPU setup, lock, lint/format, 438 tests, public artifacts, claims, freeze, bibliography and smoke PASS |
| `CPU -m pytest -q --run-scientific --basetemp=tmp/batch52/pytest-authorized -p no:cacheprovider` | 445 passed, 1 expected skip |
| `CPU -m ruff check src tests scripts`; `CPU -m ruff format --check src tests scripts` | PASS; includes audit/build additions in final check |
| `CPU -m meddet_benchmark smoke configs/smoke.yaml` | PASS, synthetic CPU |
| `CPU scripts/verify_scientific_artifacts.py --manifest results/publication_artifact_manifest.json` | 83 artifacts, all 580 local bindings, 201 result references PASS |
| `CPU scripts/verify_paper_claims.py` | 417 numerical bindings / 33 guards PASS after wording changes |
| `CPU -m scripts.verify_frozen_external --mode freeze` | Original 38 bindings PASS |
| `GPUENV -m scripts.verify_frozen_external --mode inference` with authorized `VINDR_CXR_ROOT` | All ten runs, 3,000 source images each; full checkpoint/source/repair/metric replay PASS |
| `CPU -m scripts.verify_frozen_external --mode replay` | 20 point rows; 2,000 draws per cohort; 44 intervals/130 inputs PASS |
| `GPUENV -m src.analyze_validation_threshold_sensitivity --config configs/threshold_selection_n5_validation_sensitivity.yaml --mode preflight` | Validation-only 750 images/277 boxes; hashes/environment PASS |
| `CPU -m src.benchmark_inference --config configs/inference_timing_v1.yaml --mode verify` | Saved timing, statistics, provenance, historical preservation PASS |
| `CPU scripts/check_bibliography.py --bibliography report/references.bib --manuscripts report/paper_draft.md report/report.md report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md` | 40 unique entries; all 19/19/27 cited keys PASS |
| `CPU -m scripts.export_paper_trace --manifest report/paper_claim_sources.yaml --output docs/SUBMISSION_NUMERICAL_TRACE.csv` | 417 exact trace rows PASS |
| Publication file/anchor, source checkpoint rehash/public-boundary, config/module path checks | 290 local links, 8 immutable PDF support links, 10 checkpoint hashes/configs, 44 README config paths and 28 modules/packages PASS |
| `python -m scripts.build_paper_pdf --config configs/paper_build.yaml` | Canonical build; final visual QA below |
| `python -m scripts.check_paper_pdf --pdf output/pdf/rsna_vindr_article.pdf --output report/paper_pdf_qa.json --text tmp/pdfs/batch52_article/extracted-final.txt` | 15 pages, zero automated errors; visual review below |
| `git diff --check` | PASS; final index remains empty |

The final support-path receipt is retained locally as
`tmp/batch52/support-qa.json`. Local Markdown targets/anchors were resolved from
the eleven current publication/README files; PDF support targets/anchors were
resolved from `git show ac6975d:<path>`. All 11 PDF input hashes and its output
hash match the build receipt. A fresh byte-size/SHA-256 comparison of every
`checkpoints` entry in `results/checkpoint_release_manifest.json` verified
962,924,817 checkpoint bytes and all ten config hashes. `git check-ignore -q`
confirms every binary remains ignored; the public-download field remains null.
The historical audit copy is byte-identical to its starting Git blob. The
unrelated pre-existing README wget link and local acquisition document remain
outside the Batch 52 publication changes.

## Final rendered PDF

**PASS for author review: 15 pages.** Built from the canonical Markdown and
BibTeX using Pandoc 3.9/citeproc and two pdfLaTeX passes (TeX Live 2021), with
versioned formatting-only Lua/TeX inputs and `configs/paper_build.yaml`. No
scientific content was edited in the PDF. The separate document environment
does not change the scientific dependency lock.

- Source SHA-256: `0f874aa80d3f90121a02553f41e6017d8b04178914dba70ce5cb54e6c1b24f45`.
- PDF SHA-256: `7143b8c532433dd38ee647ac43e414a1ca60e1537816e924a2d14c15f96c5319`.
- [Build/input receipt](../report/paper_build_manifest.json) and
  [automated PDF QA](../report/paper_pdf_qa.json) bind the reviewed file.

Every final page was rendered with Poppler and visually inspected. The title,
abstract and author-action line are legible (page 1); methods/cohort and Table 1
appear on pages 3–5; internal results, Tables 2–3 and Figures 1a/1b/2 on pages
6–9; timing Table 4/Figure 3 on page 10; external Tables 5–6 on page 11;
Figure 4/Table 7 on page 12; discussion/limitations/conclusion and all seven
declarations on pages 13–14; all 19 references fit on page 15. All seven tables
and five figure images are present. Figure 2 and Table 7 captions visibly retain
the YOLO upper-budget missing-support reversal and Faster R-CNN cap caveats.
No displayed equation is used; inline mathematical notation and confidence
intervals are readable.

Layout-only iterations corrected a table/float collision, inconsistent running
headers and an orphaned bibliography line before this final review. There is
no clipped text, overlapping material, missing glyph, unresolved citation or
accidental placeholder in the final output. The author metadata and seven
declaration placeholders are intentional unresolved author actions, not cleared
facts. Page 7 has whitespace because the next complete figure fits on page 8.
The final TeX pass has no overflow, missing-character, undefined-reference or
warning messages. Automated QA reports 15 nonempty pages, five embedded images,
27 resolved internal citation links, 27 distinct external URIs and zero errors.
Repository support links target files/anchors at the audited public revision;
publisher access limits remain as disclosed above. These checks establish link
targets and rendering, not perpetual external availability.

## FINAL STATUS

**READY FOR AUTHOR REVIEW.** Scientific/repository verification is complete;
the final rendered derivative and its visual inspection are recorded above.
Author declarations, release/availability decisions and journal-specific
requirements prevent **READY FOR JOURNAL SUBMISSION**. No training, new major
experiment, threshold tuning, ontology expansion, data release or publication
was performed. Stop for author review after Batch 52.
