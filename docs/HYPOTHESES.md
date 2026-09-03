# Research Question and Hypotheses

## Status and interpretation

H1--H5 were recorded in Batch 17 after the original experimental artifacts had been frozen;
H6 was added in Batch 18 for the subsequent frozen-bundle calibration analysis. They are
therefore **retrospective, result-linked hypotheses**, not a preregistration. Their purpose is
to make every research claim falsifiable against a named table or figure and to prevent the
paper narrative from selecting unsupported conclusions. Batch 18 computes a new calibration
summary but performs no training, checkpoint loading, inference, or prediction replacement.

The evidence has four distinct scopes that must not be merged:

- The clean unified comparison uses all five predeclared attempts for precision, recall, F1, AP@0.5, and AP@0.5:0.95. Conditional IoU and Dice use four complete seed pairs (`n=4`) for inference because YOLO11s seed 271 has no fixed-threshold true positive.
- The validation-selected thresholds remain the frozen three-seed choices. Precision-recall,
  FROC, fixed-threshold application, and Pareto now also have a five-run frozen-bundle
  sensitivity; the original unsuffixed n=3 artifacts remain unchanged as provenance.
- Robustness and Grad-CAM artifacts use only the primary seed-17 checkpoints on the fixed 300-image sample.
- Detection calibration uses all five seeds per detector and every frozen post-NMS test
  prediction retained at the 0.001 bundle floor. It remains distinct from both the historical
  n=3 threshold selection and the n=5 test-side operating-regime sensitivity.

## Primary research question

> Under one patient-safe (patient-disjoint) split, shared canonical preprocessing and augmentation policy, a common training-budget framework, and one model-independent evaluator, how do Faster R-CNN and YOLO11s trade off lung-opacity detection coverage and ranking accuracy, detector-specific operating regimes and confidence calibration, computational efficiency, corruption-specific robustness, and Grad-CAM-observed failure patterns on chest radiographs?

This question concerns two disclosed implementations under one controlled protocol. It does not identify a universal causal effect of detector family and does not ask whether either system can diagnose pneumonia or support clinical use.

## H1 - Detection coverage and average precision

**Hypothesis.** Under the unified held-out evaluator, Faster R-CNN will achieve higher recall, F1, AP@0.5, and AP@0.5:0.95 than YOLO11s.

**Operational check.** The hypothesis is supported only if Faster R-CNN has the higher all-attempt point estimate for all four endpoints and every primary training-procedure interval for the Faster-R-CNN-minus-YOLO11s difference remains wholly above zero.

**Existing evidence.** Check the clean means and sample sizes in [`detector_comparison.csv`](../results/tables/detector_comparison.csv) and the estimand-separated differences and intervals in [`statistical_clean_comparison.csv`](../results/tables/statistical_clean_comparison.csv). The frozen artifacts support H1 for all four endpoints with five independently resampled runs per detector.

**Boundary.** Fixed-threshold precision is not part of H1; its primary interval crosses zero even though the secondary checkpoint-conditional permutation favors YOLO11s at score 0.25. Conditional IoU and Dice are also outside H1: they exclude missed opacities, use five defined Faster R-CNN and four defined YOLO11s runs, and their primary intervals cross zero.

## H2 - A shared threshold does not define a shared operating regime

**Hypothesis.** Applying the same nominal score threshold to both detectors will not align their selectivity. YOLO11s can appear more precise at the shared score threshold 0.25 while Faster R-CNN retains the stronger common-evaluator precision-recall result over the retained predictions and higher observed sensitivity at each reported false-positive-per-image budget within the evaluated 0.01--0.99 score sweep.

**Operational check.** Three signatures must occur together: (1) the frozen score-0.25 comparison shows higher YOLO11s precision but lower YOLO11s recall; (2) the official AP@0.5 precision-recall curve favors Faster R-CNN at more recall positions, with no position favoring YOLO11s; and (3) within the evaluated score sweep, Faster R-CNN sensitivity is higher at every predeclared FROC budget of 0.125, 0.25, 0.5, 1, and 2 false positives per image.

**Existing evidence.** Check [`precision_recall_curves_n5_sensitivity.csv`](../results/tables/precision_recall_curves_n5_sensitivity.csv), [the n=5 precision-recall figure](../results/figures/precision_recall_curves_n5_sensitivity.png), [`froc_operating_points_n5_sensitivity.csv`](../results/tables/froc_operating_points_n5_sensitivity.csv), and [the n=5 FROC figure](../results/figures/froc_curves_n5_sensitivity.png), with the original unsuffixed n=3 files retained as provenance. The sensitivity supports H2: Faster R-CNN is higher at 97 of 101 AP@0.5 recall positions, four are tied, and within the evaluated 0.01--0.99 score sweep it has higher observed sensitivity at all five reported FROC budgets.

**Boundary.** This is a five-run test-side sensitivity of a historical n=3 analysis, not evidence of probabilistic miscalibration or n=5 validation selection. The YOLO11s test-sweep FROC curve reaches the lower threshold boundary at 0.01, so its plateau is not a claimed global asymptote. Seed 271 is retained exactly as observed.

## H3 - Accuracy-compute Pareto trade-off

**Hypothesis.** Neither implementation will strictly Pareto-dominate the other: Faster R-CNN will occupy the higher-AP and higher-validation-selected-recall region, while YOLO11s will occupy the higher-throughput, lower-latency, smaller-parameter, and lower-estimated-GFLOP region.

**Operational check.** In each AP-or-recall versus compute panel, the conservative all-seeds dominance rule must find no detector whose every seed is strictly better than every alternative seed on both axes. The underlying tables must preserve the expected opposite directions on accuracy and compute.

**Existing evidence.** Check [the n=5 four-panel Pareto sensitivity](../results/figures/pareto_frontier_n5_sensitivity.png), [`pareto_points_n5_sensitivity.csv`](../results/tables/pareto_points_n5_sensitivity.csv), and the frozen-threshold recall rows in [`selected_operating_points_per_seed_n5_sensitivity.csv`](../results/tables/selected_operating_points_per_seed_n5_sensitivity.csv). The historical n=3 figure remains unchanged; both scopes support H3 in all four panels.

**Boundary.** The result compares these measured pipelines on the stated laptop. It is not a universal ranking of detector families or a deployment-utility calculation. The Pareto sensitivity is n=5, but its recall thresholds were selected from n=3 validation evidence.

## H4 - Corruption degradation is type- and severity-specific

**Hypothesis.** Clean-relative robustness will vary by corruption type and severity rather than follow one uniform degradation multiplier. The detector ordering in relative retention will also change across corruption types even if Faster R-CNN retains the higher raw AP baseline.

**Operational check.** The multi-severity retention curves must differ across the seven corruptions within each detector, and the sign of the between-detector severity-5 retention difference must reverse across corruption types rather than favor one detector uniformly. Multiplicity-controlled claims must agree with the patient-cluster robustness table rather than treating every visual separation as significant.

**Existing evidence.** Check [`robustness_curves.csv`](../results/tables/robustness_curves.csv), [the clean-relative AP@0.5:0.95 curves](../results/figures/robustness_map_50_95_relative.png), and [`statistical_robustness_comparison.csv`](../results/tables/statistical_robustness_comparison.csv). The artifacts support H4: salt-and-pepper noise is most damaging for both detectors, the severity-5 relative ordering changes by corruption, and only selected contrasts such as severe-darkness AP@0.5:0.95 retention receive multiplicity-controlled support.

**Boundary.** This is a seed-17-only digital-corruption result on 300 images from 183 patient groups. It does not establish robustness to acquisition physics, new scanners, population shift, or clinical deployment.

## H5 - Grad-CAM is a failure-analysis tool, not clinical-reasoning evidence

**Hypothesis.** Grad-CAM will show weak and heterogeneous spatial agreement with annotated lung-opacity boxes, making the maps useful for locating failure patterns and artifact sensitivity but insufficient as evidence that either detector reasons clinically.

**Operational check.** Compare each detector's all-target mean energy-in-box with its mean box-area reference and inspect pointing-game accuracy in the summary table; then inspect the preselected true-positive, false-positive, and false-negative-proxy panels for extra-box activation. The interpretation fails if the text treats a plausible map as a causal explanation or as validation of clinical reasoning, regardless of localization score.

**Existing evidence.** Check [`gradcam_localization_summary.csv`](../results/tables/gradcam_localization_summary.csv), [the shared true-positive panels](../results/figures/gradcam_good_predictions.png), [the shared false-positive panels](../results/figures/gradcam_bad_predictions.png), [the false-negative proxy panels](../results/figures/gradcam_failure_cases.png), and the [v2 sensitivity-control summary](../results/tables/gradcam_sanity_v2_summary.csv). The frozen artifacts support H5 descriptively: energy-in-box exceeds the box-area reference only modestly, pointing accuracy is low for both detectors, the qualitative panels show frequent activation outside the annotated opacity, and low full-model-randomization similarity supports parameter sensitivity without validating localization.

**Boundary.** The analysis is seed-17-only and target- and layer-dependent. It contains no causal intervention or clinical-reasoning ground truth; false-negative maps use annotation-guided proxy candidates and do not explain an emitted detection. The pixel shuffle is an input-pixel perturbation control, not Adebayo training-label data randomization; randomized-annotation retraining was not performed, so no data-label-dependence claim follows.

## H6 - Detection confidence calibration

**Retrospective descriptive question.** Under one common full multivariate Detection
Expected Calibration Error (D-ECE) protocol, what values and support are observed for the
ten frozen detector runs? This audit was added after the original detector results and is
not a preregistered directional hypothesis. In particular, the method did not predeclare
that any named seed should be an outlier.

**Operational check.** For every detector and seed, retain all predictions at the frozen
0.001 bundle floor, apply the canonical score-ordered same-class matcher at IoU 0.50, and
compute the Küppers et al. five-dimensional D-ECE over confidence, relative box center, and
relative box width/height. Report every run uniformly, including total/possible/occupied/
supported cells, the detection fraction contributing through supported cells, supported-cell
sizes, the predeclared bin/minimum-support grid, and confidence-floor sensitivity. Do not
exclude or impute a run to stabilize the metric.

**Observed evidence.** Check [`calibration_summary_v2.csv`](../results/tables/calibration_summary_v2.csv),
the [support table](../results/tables/calibration_support_v2.csv), and the
[confidence-only marginal reliability figure](../results/figures/reliability_diagrams_confidence_marginal_v2.png).
The equal-run descriptive means are `0.0320 +/- 0.0058` for Faster R-CNN and
`0.0990 +/- 0.0232` for YOLO11s. The largest observed YOLO11s value is `0.1313` for seed
271; its mean score is `0.00538` while `0.15073` of its 962 retained detections are matched
true positives. D-ECE is an error measure, so the lower Faster R-CNN value
denotes lower descriptive calibration error under this fixed protocol. That
observation is reported after analysis, not encoded as a special case.

**Boundary.** D-ECE measures precision calibration conditional on the emitted prediction
population; missed targets have no confidence and therefore do not enter this black-box
calibration estimand. The sparse five-dimensional histogram and minimum-cell rule materially
affect support, so absolute values require the occupancy and sensitivity diagnostics. The
result is descriptive test-set evaluation of raw confidences, not a fitted recalibration map,
external validation, clinical-risk calibration, or run-level inference. It is also separate
from H2's score-scale/selectivity mismatch and from any separately calibrated exam-level
probability that could enter valid decision-curve analysis.
