# Hypothesis Traceability

Audit date: 2026-08-31

This register preserves the exact wording in [`HYPOTHESES.md`](HYPOTHESES.md)
and maps it to the current manuscript, [`paper_draft.md`](../report/paper_draft.md).
H1--H5 were written in Batch 17 after most result artifacts were frozen. H6 was
added in Batch 18 for a subsequent frozen-bundle calibration audit. Therefore
**none is preregistered**. “Supported” below means that the named frozen
artifact satisfies the retrospectively declared operational check; it does not
convert the hypothesis into a prospective or confirmatory test.

## H1 — Detection coverage and average precision

> **Exact wording:** Under the unified held-out evaluator, Faster R-CNN will
> achieve higher recall, F1, AP@0.5, and AP@0.5:0.95 than YOLO11s.

| Field | Trace |
|---|---|
| Status | Retrospective, result-linked hypothesis; not preregistered. |
| Operational endpoint | Faster R-CNN must have the higher all-attempt point estimate for recall, F1, AP@0.5, and AP@0.5:0.95, and every primary Faster-R-CNN-minus-YOLO11s training-procedure interval for those endpoints must be wholly above zero. |
| Dataset/split | Patient-disjoint internal-testing split: 750 radiographs, 323 NIH patient groups, 169 opacity-positive images, and 268 boxes. |
| Seed count | Five attempted runs per detector: 17, 42, 137, 271, and 314. |
| Source artifacts | [`detector_comparison.csv`](../results/tables/detector_comparison.csv); [`statistical_clean_comparison.csv`](../results/tables/statistical_clean_comparison.csv); per-run endpoint inputs in [`statistical_clean_per_run_metrics.csv`](../results/tables/statistical_clean_per_run_metrics.csv). |
| Manuscript section | [`paper_draft.md` §§4.1 and 4.9](../report/paper_draft.md#41-clean-held-out-performance-and-seed-instability). |
| Result status | **Supported retrospectively** for all four named endpoints under the frozen operational rule. This is not a preregistered confirmatory finding. |
| Multiplicity | The primary 95% intervals are pointwise; no simultaneous multiplicity adjustment was applied to the four-part H1 claim. The separate checkpoint-conditional permutation p-values belong to one Holm family across seven clean endpoints and do not replace the primary training-procedure intervals. |
| Limitations | Compound retrospective hypothesis; five runs coarsely represent retraining variability; one internal dataset; no external testing; shared-threshold precision and conditional IoU/Dice are outside H1; no architecture-family causal claim. |

## H2 — A shared threshold does not define a shared operating regime

> **Exact wording:** Applying the same nominal score threshold to both
> detectors will not align their selectivity. YOLO11s can appear more precise
> at the shared score threshold 0.25 while Faster R-CNN retains the stronger
> common-evaluator precision-recall frontier and higher sensitivity at each
> reported false-positive-per-image budget.

| Field | Trace |
|---|---|
| Status | Retrospective, result-linked hypothesis; not preregistered. |
| Operational endpoint | Three signatures together: higher YOLO11s precision but lower recall at score 0.25; Faster R-CNN higher at more official AP@0.5 recall positions with none favoring YOLO11s; Faster R-CNN higher sensitivity at each FROC budget 0.125, 0.25, 0.5, 1, and 2 FP/image. |
| Dataset/split | Same 750-image/323-patient internal-testing split; frozen exploratory threshold/PR/FROC analyses. |
| Seed count | Three runs per detector: 17, 42, and 137. |
| Source artifacts | [`detector_comparison_n3_archive.csv`](../results/tables/detector_comparison_n3_archive.csv); [`precision_recall_curves.csv`](../results/tables/precision_recall_curves.csv); [`froc_operating_points.csv`](../results/tables/froc_operating_points.csv); figures [`precision_recall_curves.png`](../results/figures/precision_recall_curves.png) and [`froc_curves.png`](../results/figures/froc_curves.png). |
| Manuscript section | [`paper_draft.md` §§4.2, 4.3, and 5.1](../report/paper_draft.md#42-threshold-sensitivity-and-official-precision-recall). |
| Result status | **Supported retrospectively and descriptively:** 96/101 recall positions favor Faster R-CNN, five tie, and all five reported FROC budgets favor Faster R-CNN sensitivity. |
| Multiplicity | No inferential multiplicity procedure applies to the descriptive curve-position counts and predeclared FROC budgets. The score-0.25 clean endpoint p-values are part of the separate seven-endpoint Holm family, not a test of the whole H2 pattern. |
| Limitations | Frozen `n=3`; exploratory internal-testing threshold sweep; no probabilistic-calibration inference; YOLO11s reaches the 0.01 lower threshold boundary; no clinical operating point or external testing. |

## H3 — Accuracy-compute Pareto trade-off

> **Exact wording:** Neither implementation will strictly Pareto-dominate the
> other: Faster R-CNN will occupy the higher-AP and higher-validation-selected-
> recall region, while YOLO11s will occupy the higher-throughput, lower-latency,
> smaller-parameter, and lower-estimated-GFLOP region.

| Field | Trace |
|---|---|
| Status | Retrospective, result-linked hypothesis; not preregistered. |
| Operational endpoint | Across AP-or-model-optimization-selected-recall versus FPS, latency, parameters, and registered-operation GFLOPs, no detector may have every seed strictly better than every competing seed on both directed axes. |
| Dataset/split | Accuracy uses the 750-image patient-disjoint internal-testing split; thresholds were selected only on the 750-image/321-patient model-optimization split. Compute was measured on the stated RTX 4060 Laptop GPU protocol. |
| Seed count | Frozen three-run Pareto scope per detector: 17, 42, and 137. |
| Source artifacts | [`pareto_frontier.png`](../results/figures/pareto_frontier.png); [`detector_comparison_per_seed_n3_archive.csv`](../results/tables/detector_comparison_per_seed_n3_archive.csv); [`selected_operating_points_per_seed.csv`](../results/tables/selected_operating_points_per_seed.csv); [`faster_rcnn_compute.csv`](../results/tables/faster_rcnn_compute.csv) and [`yolo_compute.csv`](../results/tables/yolo_compute.csv). |
| Manuscript section | [`paper_draft.md` §§4.5 and 5.2](../report/paper_draft.md#45-compute-profile-and-pareto-structure). |
| Result status | **Supported retrospectively and descriptively** in all four frozen panels under the conservative all-seeds rule. |
| Multiplicity | No hypothesis-test family or multiplicity correction applies; this is a deterministic Pareto classification over the named runs and axes. |
| Limitations | One hardware/software stack; asymmetric preprocessing inside timing regions; unsupported operations omitted from GFLOPs; `n=3`; model-optimization-selected recall rather than clinical utility; pipeline, not architecture-family, comparison. |

## H4 — Corruption degradation is type- and severity-specific

> **Exact wording:** Clean-relative robustness will vary by corruption type and
> severity rather than follow one uniform degradation multiplier. The detector
> ordering in relative retention will also change across corruption types even
> if Faster R-CNN retains the higher raw AP baseline.

| Field | Trace |
|---|---|
| Status | Retrospective, result-linked hypothesis; not preregistered. |
| Operational endpoint | Curves must differ across the seven corruptions and five severities within detector; the between-detector severity-5 retention sign must reverse across corruption types; any multiplicity-controlled wording must agree with the patient-cluster robustness table. |
| Dataset/split | Fixed stratified sample of 300 internal-testing radiographs from 183 NIH patient groups, including 68 positive images and 111 boxes. |
| Seed count | One frozen checkpoint per detector, seed 17. |
| Source artifacts | [`robustness_curves.csv`](../results/tables/robustness_curves.csv); [`robustness_map_50_95_relative.png`](../results/figures/robustness_map_50_95_relative.png); [`statistical_robustness_comparison.csv`](../results/tables/statistical_robustness_comparison.csv). |
| Manuscript section | [`paper_draft.md` §§4.6 and 5.4](../report/paper_draft.md#46-common-corruption-robustness). |
| Result status | **Supported retrospectively** for the descriptive pattern; only selected individual contrasts receive multiplicity-controlled inferential support. |
| Multiplicity | Yes for formal corruption contrasts: Holm correction is applied separately within the 35-condition raw-performance and clean-relative-retention metric/estimand families. The broad qualitative curve pattern itself is not one adjusted omnibus test. |
| Limitations | Single checkpoint per detector; modest internal sample; deterministic digital perturbations; no acquisition physics, scanner/site/population shift, severity prevalence, or external testing; dependence among severity conditions remains. |

## H5 — Grad-CAM is a failure-analysis tool, not clinical-reasoning evidence

> **Exact wording:** Grad-CAM will show weak and heterogeneous spatial
> agreement with annotated lung-opacity boxes, making the maps useful for
> locating failure patterns and artifact sensitivity but insufficient as
> evidence that either detector reasons clinically.

| Field | Trace |
|---|---|
| Status | Retrospective, result-linked hypothesis; not preregistered. |
| Operational endpoint | Compare all-target mean energy-in-box with box-area reference, report pointing-game accuracy, inspect preselected TP/FP/FN-proxy panels, and reject any interpretation of plausible maps as causal or clinical-reasoning validation. Parameter and input controls must be interpreted only for their tested sensitivities. |
| Dataset/split | Primary localization: fixed 300-image/183-patient internal-testing sample with 111 boxes. Sanity controls: nested stratified 50-image/41-patient subset. |
| Seed count | One frozen checkpoint per detector, seed 17. |
| Source artifacts | [`gradcam_localization_summary.csv`](../results/tables/gradcam_localization_summary.csv); [`gradcam_good_predictions.png`](../results/figures/gradcam_good_predictions.png); [`gradcam_bad_predictions.png`](../results/figures/gradcam_bad_predictions.png); [`gradcam_failure_cases.png`](../results/figures/gradcam_failure_cases.png); [`gradcam_sanity_v2_summary.csv`](../results/tables/gradcam_sanity_v2_summary.csv). |
| Manuscript section | [`paper_draft.md` §§4.8 and 5.5](../report/paper_draft.md#48-grad-cam-localization-and-parameter-sensitivity). |
| Result status | **Supported retrospectively and descriptively:** localization is weak/heterogeneous, and low full-model-randomization similarity shows parameter sensitivity without validating causal or clinical reasoning. |
| Multiplicity | No multiplicity-controlled inferential hypothesis test was performed for localization or map-similarity measures. |
| Limitations | One seed and selected layers/targets; false negatives use annotation-guided proxy candidates; fixed-region pre-activation sanity target differs from the primary target; severe pixel permutation is out of distribution; no randomized-label retraining, causal test, expert explanation reference, or external testing. |

## H6 — Detection confidence calibration

> **Exact wording:** Under one common full multivariate Detection Expected
> Calibration Error (D-ECE) protocol, what values and support are observed for
> the ten frozen detector runs?

| Field | Trace |
|---|---|
| Status | Retrospective descriptive question added after the original detector results; not preregistered and not a directional hypothesis. |
| Operational endpoint | For every detector/seed, retain detections at score `>=0.001`; apply the canonical score-ordered same-class matcher at IoU 0.50; compute five-dimensional D-ECE; report possible/occupied/supported cells, supported fraction and sizes, the bin/minimum-support grid, and confidence-floor sensitivity without run-specific exclusion or imputation. |
| Dataset/split | All frozen post-NMS prediction bundles from the 750-image/323-patient internal-testing split. |
| Seed count | Ten runs total: five per detector (17, 42, 137, 271, and 314). |
| Source artifacts | [`calibration_summary_v2.csv`](../results/tables/calibration_summary_v2.csv); [`calibration_support_v2.csv`](../results/tables/calibration_support_v2.csv); [`calibration_sensitivity_v2.csv`](../results/tables/calibration_sensitivity_v2.csv); [`reliability_diagrams_confidence_marginal_v2.png`](../results/figures/reliability_diagrams_confidence_marginal_v2.png). |
| Manuscript section | [`paper_draft.md` §§3.6, 4.4, and 5.1](../report/paper_draft.md#36-detection-specific-confidence-calibration). |
| Result status | **Answered descriptively**, not supported/rejected: equal-run mean D-ECE is 0.0320 for Faster R-CNN and 0.0990 for YOLO11s under the primary specification, with support diagnostics reported for every run. |
| Multiplicity | Not applicable to the reported descriptive run summaries and predeclared sensitivity grid; no run-level inferential p-values or favorable-setting selection were used. |
| Limitations | Conditional on emitted detections; missed targets are outside the estimand; sparse multivariate cells and the support rule materially affect D-ECE; no fitted recalibration, exam-level outcome probability, decision-curve input, external testing, or run-level inference. |

## Interpretation rule

The manuscript may say that frozen evidence *retrospectively supports* H1--H5
or *descriptively answers* H6. It must not use “preregistered,” “prospectively
confirmed,” or equivalent language. Formal multiplicity control applies only
where listed above and does not propagate to untested compound narratives.
