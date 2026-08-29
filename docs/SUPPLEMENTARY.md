# Supplementary Materials Index

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
- [Recall-weighted F-beta threshold sensitivity](../results/tables/recall_weighted_fbeta_threshold_summary.csv),
  its [candidate-level stability frequencies](../results/tables/recall_weighted_fbeta_threshold_stability.csv),
  and the separate [hypothetical linear detection-error loss](../results/tables/hypothetical_detection_error_loss_summary.csv)
  are `n=3` validation analyses. Per [D-006](DECISION_LOG.md#d-006--treat-beta-as-an-f-beta-preference-parameter-and-separate-linear-loss),
  beta is a recall-preference parameter, the assumed loss ratios are not
  clinical valuations, and none replaces or feeds the primary thresholds,
  FROC, or Pareto artifacts.
- [FROC operating points](../results/tables/froc_operating_points.csv) and the
  [Pareto figure](../results/figures/pareto_frontier.png) retain their explicit
  frozen `n=3` scope.

## S4. Calibration and exploratory raw-score utility

- [Five-seed D-ECE table](../results/tables/calibration_summary.csv) and
  [reliability diagrams](../results/figures/reliability_diagrams.png) describe
  calibration of emitted detections at the 0.001 bundle floor; they do not fit
  a calibrator or estimate missed-target or clinical-risk calibration.
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
- [Acquisition-shift results](../results/tables/acquisition_shift_results.csv)
  contain 20 detector/condition rows and all seven metric-specific degradation
  sensitivity indices. [Method and scope](ACQUISITION_SHIFTS.md) distinguish
  this raw-array, acquisition-motivated sensitivity analysis from the generic
  post-conversion digital-corruption benchmark.

## S6. Explainability and sanity evidence

- [Per-target Grad-CAM data](../results/tables/gradcam_localization_per_target.csv),
  [aggregate localization](../results/tables/gradcam_localization_summary.csv),
  and [qualitative case manifest](../results/tables/gradcam_qualitative_cases.csv)
  preserve all target-level and selected-case evidence.
- [Per-image sanity results](../results/tables/gradcam_sanity_per_image.csv),
  [sanity summary](../results/tables/gradcam_sanity_summary.csv), and
  [panel](../results/figures/gradcam_sanity_panel.png) document the nested
  50-image parameter- and data-randomization checks and explicit invalid-map
  denominators.

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
