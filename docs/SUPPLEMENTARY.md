# Supplementary Materials Index

**Primary runtime update (Batch 45):** [Matched v1 timing protocol](COMPUTE_TIMING.md),
[aggregate table](../results/tables/inference_timing_v1.csv),
[complete repetitions](../results/tables/inference_timing_v1_repetitions.csv),
[raw per-image times/agreement](../results/tables/inference_timing_v1_images.csv),
and [provenance summary](../results/logs/phase45_inference_timing_v1/summary.json)
replace the main asymmetric timing comparison. Three technical repeats are not
training replicates. Compute timing panels in the older seed-level/raincloud
and Pareto artifacts below are historical; profiler GFLOPs are incomplete
registered-operation counts only.


This index keeps exhaustive machine-readable evidence outside the narrative
report. It is pointer-style by design: values should be quoted from the linked
artifacts, not copied into another manually maintained table. Scope labels are
part of the evidence and must travel with any reused number.

## S1. Cohort, split, and provenance records

- [Dataset audit](../data/manifests/rsna-pneumonia-5000-audit.json) records the
  full-source counts, selected 5,000-study cohort, split counts, patient-group
  disjointness, and input hashes.
- [Train](../data/splits/rsna-pneumonia-5000/train.csv),
  [validation](../data/splits/rsna-pneumonia-5000/val.csv), and
  [test](../data/splits/rsna-pneumonia-5000/test.csv) manifests preserve every
  study assignment and NIH patient grouping key.
- [Aggregate cohort characteristics](../results/tables/rsna_cohort_characteristics.csv)
  and their [validation/provenance summary](../results/logs/phase44_cohort_characteristics/summary.json)
  report partition-wise age, sex, and AP/PA projection without exporting raw
  patient rows or identifiers. The summary records the nonconformant age-unit
  caveat and immutable split/mapping checks.
- [Datasheet](DATASHEET.md) documents collection, annotation provenance,
  preprocessing, access conditions, and known population limitations.

## S2. Full clean seed-level comparison

- [Per-seed detector comparison](../results/tables/detector_comparison_per_seed.csv)
  contains all ten detector/seed records, the frozen score-0.25 operating-point
  counts, seven predictive metrics, and compute measurements.
- [Publication comparison](../results/tables/detector_comparison.csv) and
  [long mean/SD table](../results/tables/detector_comparison_mean_std.csv)
  preserve detector-specific finite `n`, attempted `n`, undefined seeds, and
  reasons. IoU and Dice are conditional matched-box metrics with Faster R-CNN
  `n=5` and YOLO11s `n=4`; the other clean endpoints use `n=5` per detector.
- [Raincloud figure](../results/figures/raincloud_metrics.png) renders the same
  seed-level predictive and compute evidence. Its aggregate-to-seed audit,
  counts, input hashes, and figure hash are in the
  [provenance summary](../results/logs/phase23_reporting/raincloud_metrics_summary.json).
- Explicit three-seed historical artifacts remain separately named:
  [publication archive](../results/tables/detector_comparison_n3_archive.csv),
  [seed archive](../results/tables/detector_comparison_per_seed_n3_archive.csv),
  and [mean/SD archive](../results/tables/detector_comparison_mean_std_n3_archive.csv).

## S3. Operating-point and threshold evidence

- [Exploratory test sweep](../results/tables/threshold_sweep.csv),
  [seed-level sweep](../results/tables/threshold_sweep_per_seed.csv), and
  [operating targets](../results/tables/threshold_operating_targets.csv) are
  frozen `n=3` test descriptions, not threshold-selection evidence.
- [Validation sweep](../results/tables/validation_threshold_sweep.csv),
  [seed-level validation sweep](../results/tables/validation_threshold_sweep_per_seed.csv),
  [selected operating points](../results/tables/selected_operating_points.csv),
  and [seed-level applications](../results/tables/selected_operating_points_per_seed.csv)
  document the primary `n=3` validation-selected thresholds and their one-shot
  test application.
- Batch 35 retains every preceding n=3 file and adds the all-attempt test-side
  sensitivity: [aggregate threshold sweep](../results/tables/threshold_sweep_n5_sensitivity.csv),
  [per-run threshold sweep](../results/tables/threshold_sweep_per_seed_n5_sensitivity.csv),
  [official PR curves](../results/tables/precision_recall_curves_n5_sensitivity.csv),
  [per-run PR curves](../results/tables/precision_recall_curves_per_seed_n5_sensitivity.csv),
  and [fixed-threshold test application](../results/tables/selected_operating_points_n5_sensitivity.csv).
  Within Batch 35, threshold selection remains n=3; test sensitivity uses n=5
  and includes seed 271 exactly as observed.
- Batch 43 separately completes the four seed-271/314 validation bundles and
  reports the [five-run validation sweep](../results/tables/validation_threshold_sweep_n5_validation_sensitivity.csv),
  [aggregate test application](../results/tables/threshold_selection_test_operating_points_n5_validation_sensitivity.csv),
  [per-run test rows](../results/tables/threshold_selection_test_operating_points_per_seed_n5_validation_sensitivity.csv),
  and [classified conclusions](../results/tables/threshold_selection_n3_vs_n5_validation_conclusions.csv).
  The 0.70/0.01 thresholds are post-hoc validation sensitivity only; the
  historical n=3 thresholds and artifacts remain frozen provenance.
- [Recall-weighted F-beta threshold sensitivity](../results/tables/recall_weighted_fbeta_threshold_summary.csv),
  its [candidate-level stability frequencies](../results/tables/recall_weighted_fbeta_threshold_stability.csv),
  and the separate [hypothetical linear detection-error loss](../results/tables/hypothetical_detection_error_loss_summary.csv)
  are `n=3` validation analyses. Per [D-006](DECISION_LOG.md#d-006--treat-beta-as-an-f-beta-preference-parameter-and-separate-linear-loss),
  beta is a recall-preference parameter, the assumed loss ratios are not
  clinical valuations, and none replaces or feeds the primary thresholds,
  FROC, or Pareto artifacts.
- Historical [FROC operating points](../results/tables/froc_operating_points.csv)
  and the [Pareto figure](../results/figures/pareto_frontier.png) retain their
  frozen `n=3` scope. Separate n=5 grid-sensitivity artifacts retain the
  historical [FROC operating points](../results/tables/froc_operating_points_n5_sensitivity.csv)
  and [per-run FROC curves](../results/tables/froc_curves_per_seed_n5_sensitivity.csv).
  The current FROC evidence is the [approved lower-floor observed exact-score operating-point table](../results/tables/froc_operating_points_exact_score_v4.csv),
  [grid comparison](../results/tables/froc_grid_vs_exact_score_comparison_v4.csv),
  [conservative bound](../results/tables/froc_incomplete_frontier_bounds_v4.csv),
  and [exact-score figure](../results/figures/froc_exact_score_v4.png); it remains
  bounded by 0.00001 for YOLO11s seed 137 only at 2 FP/image, where even the
  mathematical-maximum bound cannot reverse detector ordering. The v2/v3
  exact-score outputs remain provenance. Separate n=5 sensitivity artifacts provide
  [Pareto points](../results/tables/pareto_points_n5_sensitivity.csv), and the
  [n=5 Pareto figure](../results/figures/pareto_frontier_n5_sensitivity.png).
  The [complete conclusion audit](../results/tables/operating_regime_n3_vs_n5_conclusions.csv)
  reports unchanged, strengthened, weakened, and reversed categories without
  selecting only favorable changes.

## S4. Calibration and exploratory raw-score utility

- The versioned [five-seed D-ECE table](../results/tables/calibration_summary_v2.csv),
  [cell-support table](../results/tables/calibration_support_v2.csv),
  [predeclared sensitivity grid](../results/tables/calibration_sensitivity_v2.csv),
  and [confidence-only marginal reliability diagrams](../results/figures/reliability_diagrams_confidence_marginal_v2.png)
  describe emitted detections at the 0.001 bundle floor. The 1-D diagram is not
  a visualization of all five D-ECE dimensions. These outputs do not fit a
  calibrator or estimate missed-target, exam-level, or clinical-risk
  calibration.
- The historical Batch 20 calculation is retained only as an explicitly
  **non-standard raw-score threshold utility/sensitivity** analysis. The
  [relabeled table](../results/tables/raw_score_threshold_utility_summary.csv),
  [relabeled figure](../results/figures/raw_score_threshold_utility_sensitivity.png),
  and [classification/provenance record](../results/logs/phase30_raw_score_utility/summary.json)
  use the complete 750-image internal test subset and five seeds. Maximum
  emitted confidence defines the exam flag, but that score is not a validated
  outcome probability; the results are not conventional DCA, clinical
  net-benefit evidence, or a deployment-threshold analysis. The exact original
  table, figure, code, config, and Phase 20 summary remain in explicitly named
  pre-Batch-30 archives listed in [the corrective analysis note](DCA_ANALYSIS.md).

## S5. Complete digital-corruption and acquisition-shift grids

- [Complete digital-corruption results](../results/tables/robustness_results.csv)
  contain both detectors' clean references and all 70 corrupted conditions on
  the fixed 300-image, seed-17 sample. The tidy
  [per-type curves](../results/tables/robustness_curves.csv) and
  [family means](../results/tables/robustness_family_mean_curves.csv) preserve
  raw, clean-relative, and degradation endpoints.
- [Patient-cluster corruption comparisons](../results/tables/statistical_robustness_comparison.csv)
  contain the full 497-row inferential grid. Superseded image-level results are
  retained only in the explicitly named
  [audit archive](../results/tables/statistical_robustness_comparison_image_level_archive.csv).
- [Reclassified synthetic-shift results](../results/tables/radiography_synthetic_shift_results.csv)
  contain 20 unchanged detector/condition results and all seven descriptive
  performance-retention/domain-sensitivity indices under corrected labels.
  The [per-image DICOM audit](../results/tables/acquisition_shift_dicom_metadata_audit.csv)
  and [preprocessing summary](../results/tables/acquisition_shift_preprocessing_summary.csv)
  show that all 300 inputs are lossy, workstation-converted Secondary Capture
  objects and quantify which perturbations canonical per-image min-max scaling
  cancels or re-stretches. [Method and scope](ACQUISITION_SHIFTS.md) classify
  each transform separately and exclude dose simulation, scanner-specific
  modeling, clinical robustness, and inter-site transportability claims. The
  original `acquisition_shift_results.csv` and Phase 22 bundles are retained as
  superseded historical evidence; the CPU-only correction is hash-bound in the
  [Phase 32 summary](../results/logs/phase32_acquisition_shift_audit/summary.json).

## S6. Explainability and sanity evidence

- [Per-target Grad-CAM data](../results/tables/gradcam_localization_per_target.csv),
  [aggregate localization](../results/tables/gradcam_localization_summary.csv),
  and [qualitative case manifest](../results/tables/gradcam_qualitative_cases.csv)
  preserve all target-level and selected-case evidence.
- [Per-image v2 control results](../results/tables/gradcam_sanity_v2_per_image.csv),
  [v2 summary](../results/tables/gradcam_sanity_v2_summary.csv), and
  [v2 panel](../results/figures/gradcam_sanity_v2_panel.png) document the nested
  50-image six-stage model-parameter cascade, input-pixel perturbation control,
  Pearson/Spearman/SSIM metrics, and invalid-map denominators. The unversioned
  Batch 21 files remain historical; their legacy `data_randomization` label
  denotes input-pixel shuffling, not randomized-label training.

## S7. Statistical evidence and archives

- [Primary clean patient-cluster comparison](../results/tables/statistical_clean_comparison.csv)
  reports seven primary training-procedure intervals with detector-specific run
  counts and separately labeled Holm p-values conditional on the observed
  checkpoints.
- Seed influence is explicit in the [per-run metrics](../results/tables/statistical_clean_per_run_metrics.csv),
  [leave-one-training-run-out](../results/tables/statistical_clean_leave_one_run_out.csv),
  and [descriptive leave-one-seed-label-out](../results/tables/statistical_clean_leave_one_seed_label_out.csv)
  tables. Seed 271 is not outcome-selected away.
- The [paired-seed sensitivity archive](../results/tables/statistical_clean_comparison_paired_seed_sensitivity_archive.csv)
  preserves the former common-index bootstrap as a nonprimary historical result.
- The [three-seed patient-cluster archive](../results/tables/statistical_clean_comparison_n3_archive.csv)
  and [superseded image-level archive](../results/tables/statistical_clean_comparison_image_level_archive.csv)
  remain available for audit and are not current inferential results.
- [Statistical method and interpretation](STATISTICAL_ANALYSIS.md) identify the
  correction families, patient-group units, endpoint-specific complete cases,
  and remaining uncertainty limits.

## S8. Decisions, limitations, and reproduction

- [Decision log](DECISION_LOG.md) is the authoritative append-only record for
  the controlled-comparison scope, Track B descoping, and primary-versus-cost-
  sensitive threshold precedence.
- [Consolidated limitations](LIMITATIONS.md) defines the dataset, compute,
  calibration, threshold, robustness, explainability, statistical, deployment,
  and regulatory claim boundaries.
- [Reporting checklist crosswalk](REPORTING_CHECKLIST.md) records which CLAIM
  2024, TRIPOD+AI 2024, and STARD-AI 2025 items are fully evidenced, partial,
  absent, or not applicable in the repository as it exists now.
- [README](../README.md) gives the exact commands that regenerate every linked
  table and figure; it is the command authority rather than this pointer index.

## S9. Frozen external testing and transportability

The final canonical manuscript integrates the completed external evidence.
The pre-external paper is preserved only as an immutable protocol baseline;
it is not a second maintained manuscript.

- [Frozen protocol](VINDR_EXTERNAL_PROTOCOL.md), [scientific config](../configs/vindr_external_v1.yaml),
  and [original hash sidecar](VINDR_EXTERNAL_PROTOCOL_v1.sha256.json): locally
  prespecified ontology, all-run inventory, endpoints, thresholds, floors/cap,
  numerical precision and inferential estimands. No external tuning occurred.
- [Adapter receipt](../results/vindr_external_v1/adapter_preflight.json) and
  [preflight review](VINDR_ADAPTER_PREFLIGHT.md): complete official test cohort,
  strict target support, source integrity, decoding and coordinate checks.
- [Inference summary](../results/vindr_external_v1/inference_summary.json) and
  [generated per-run report](VINDR_INFERENCE_RESULTS.md): every checkpoint,
  both threshold policies, AP, exact-score FROC, cap saturation and emissions.
- [Statistical methods](VINDR_STATISTICS.md), [bound summary](../results/vindr_external_v1/statistics/summary.json),
  [all twenty run rows](../results/vindr_external_v1/statistics/per_run.csv),
  [all marginal intervals](../results/vindr_external_v1/statistics/intervals.csv),
  [score/emission support](../results/vindr_external_v1/statistics/score_summaries.csv),
  [cross-dataset comparisons](../results/vindr_external_v1/statistics/transport_comparison.csv),
  and [threshold-policy sensitivity](../results/vindr_external_v1/statistics/threshold_policy_comparison.csv).
  Primary observation/run uncertainty, fixed seed-17 sensitivity, and descriptive
  comparisons remain separate. The aligned internal intervals do not replace
  historical Phase 8 artifacts in S7. The n=5 threshold policy remains post hoc.
- [Generated statistics report](VINDR_STATISTICS_RESULTS.md) gives all SDs,
  valid/undefined-draw counts and conservative missing-support bounds.
  [Per-run score-distribution figure](../results/vindr_external_v1/statistics/internal_external_scores.png)
  accompanies counts because confidence summaries depend on emission support.
- [Numerical implementation record](VINDR_INFERENCE.md) documents the initial
  Faster R-CNN float32 inverse-resize upper-bound overshoot of
  0.000244140625 pixels (one ULP), stopped before any complete result. Only that
  narrow upper-bound error is canonicalized; negative lower bounds and larger
  errors fail. Original failed-attempt evidence and raw corrected coordinates
  remain private and hash-bound. This was not a scientific protocol amendment
  or post-hoc optimization. Historical internal YOLO FP32 and verified external
  bfloat16 AMP remain a confound in cross-dataset comparisons.

External AP ordering is an observed equal-run result. Most FROC contrasts and
all historical-threshold contrasts include zero. At upper FROC budgets the
YOLO missing-support bound permits reversal beyond retained candidates;
substantial Faster R-CNN cap saturation remains independent of that bound.
Neither low FP/image with collapsed recall nor preserved relative AP ordering
establishes successful generalization. No patient grouping is invented for VinDr.

[Publication verification](FINAL_MANUSCRIPT_AUDIT.md) records the current paper's
claim bindings and clean-checkout checks. The [immutable editorial baseline
bindings](../report/provenance/batch46/baseline_bindings.json) allow read-only
legacy evidence replay while preserving every original freeze hash. Exact
commands are in [README](../README.md#final-manuscript-verification-batch-51).
Restricted images, IDs, annotations and image-linked derivatives remain local;
only nonidentifying aggregate external artifacts accompany the manuscript.
