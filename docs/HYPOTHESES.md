# Research Question and Hypotheses

## Status and interpretation

These hypotheses were recorded in Batch 17 after the experimental artifacts had been frozen. They are therefore **retrospective, result-linked hypotheses**, not a preregistration. Their purpose is to make every research claim falsifiable against a named existing table or figure and to prevent the paper narrative from selecting unsupported conclusions. No estimate is recomputed here.

The evidence has three distinct scopes that must not be merged:

- The clean unified comparison uses all five predeclared attempts for precision, recall, F1, AP@0.5, and AP@0.5:0.95. Conditional IoU and Dice use four complete seed pairs (`n=4`) for inference because YOLO11s seed 271 has no fixed-threshold true positive.
- Precision-recall, validation-threshold, FROC, and Pareto artifacts are frozen three-seed analyses.
- Robustness and Grad-CAM artifacts use only the primary seed-17 checkpoints on the fixed 300-image sample.

## Primary research question

> Under one patient-safe (patient-disjoint) split, shared canonical preprocessing and augmentation policy, a common training-budget framework, and one model-independent evaluator, how do Faster R-CNN and YOLO11s trade off lung-opacity detection coverage and ranking accuracy, detector-specific operating regimes, computational efficiency, corruption-specific robustness, and Grad-CAM-observed failure patterns on chest radiographs?

This question concerns two disclosed implementations under one controlled protocol. It does not identify a universal causal effect of detector family and does not ask whether either system can diagnose pneumonia or support clinical use.

## H1 - Detection coverage and average precision

**Hypothesis.** Under the unified held-out evaluator, Faster R-CNN will achieve higher recall, F1, AP@0.5, and AP@0.5:0.95 than YOLO11s.

**Operational check.** The hypothesis is supported only if Faster R-CNN has the higher all-attempt point estimate for all four endpoints and each paired Faster-R-CNN-minus-YOLO11s difference remains positive with a Holm-adjusted `p < 0.05` in the patient-cluster analysis.

**Existing evidence.** Check the clean means and sample sizes in [`detector_comparison.csv`](../results/tables/detector_comparison.csv) and the paired differences, confidence intervals, and corrected p-values in [`statistical_clean_comparison.csv`](../results/tables/statistical_clean_comparison.csv). The frozen artifacts support H1 for all four endpoints at `n=5` paired attempts.

**Boundary.** Fixed-threshold precision is not part of H1 and favors YOLO11s at score 0.25. Conditional IoU and Dice are also outside H1: they exclude missed opacities, use four complete pairs for inference, and are not Holm-significant.

## H2 - A shared threshold does not define a shared operating regime

**Hypothesis.** Applying the same nominal score threshold to both detectors will not align their selectivity. YOLO11s can appear more precise at the shared score threshold 0.25 while Faster R-CNN retains the stronger common-evaluator precision-recall frontier and higher sensitivity at each reported false-positive-per-image budget.

**Operational check.** Three signatures must occur together: (1) the frozen score-0.25 comparison shows higher YOLO11s precision but lower YOLO11s recall; (2) the official AP@0.5 precision-recall curve favors Faster R-CNN at more recall positions, with no position favoring YOLO11s; and (3) Faster R-CNN sensitivity is higher at every predeclared FROC budget of 0.125, 0.25, 0.5, 1, and 2 false positives per image.

**Existing evidence.** Check [`detector_comparison_n3_archive.csv`](../results/tables/detector_comparison_n3_archive.csv), [`precision_recall_curves.csv`](../results/tables/precision_recall_curves.csv), [the precision-recall figure](../results/figures/precision_recall_curves.png), [`froc_operating_points.csv`](../results/tables/froc_operating_points.csv), and [the FROC figure](../results/figures/froc_curves.png). The frozen artifacts support H2: Faster R-CNN is higher at 96 of 101 AP@0.5 recall positions, five are tied, and it has higher sensitivity at all five reported FROC budgets.

**Boundary.** This is a frozen `n=3` score-scale/selectivity finding, not an `n=5` curve analysis and not evidence of probabilistic miscalibration. The YOLO11s test-sweep FROC curve reaches the lower threshold boundary at 0.01, so its plateau is not a claimed global asymptote.

## H3 - Accuracy-compute Pareto trade-off

**Hypothesis.** Neither implementation will strictly Pareto-dominate the other: Faster R-CNN will occupy the higher-AP and higher-validation-selected-recall region, while YOLO11s will occupy the higher-throughput, lower-latency, smaller-parameter, and lower-estimated-GFLOP region.

**Operational check.** In each AP-or-recall versus compute panel, the conservative all-seeds dominance rule must find no detector whose every seed is strictly better than every alternative seed on both axes. The underlying tables must preserve the expected opposite directions on accuracy and compute.

**Existing evidence.** Check [the four-panel Pareto figure](../results/figures/pareto_frontier.png), the frozen clean seed rows in [`detector_comparison_per_seed_n3_archive.csv`](../results/tables/detector_comparison_per_seed_n3_archive.csv), and the validation-selected test recall rows in [`selected_operating_points_per_seed.csv`](../results/tables/selected_operating_points_per_seed.csv). The frozen `n=3` artifacts support H3 in all four panels.

**Boundary.** The result compares these measured pipelines on the stated laptop. It is not a universal ranking of detector families, a deployment-utility calculation, or a five-seed Pareto analysis.

## H4 - Corruption degradation is type- and severity-specific

**Hypothesis.** Clean-relative robustness will vary by corruption type and severity rather than follow one uniform degradation multiplier. The detector ordering in relative retention will also change across corruption types even if Faster R-CNN retains the higher raw AP baseline.

**Operational check.** The multi-severity retention curves must differ across the seven corruptions within each detector, and the sign of the between-detector severity-5 retention difference must reverse across corruption types rather than favor one detector uniformly. Multiplicity-controlled claims must agree with the patient-cluster robustness table rather than treating every visual separation as significant.

**Existing evidence.** Check [`robustness_curves.csv`](../results/tables/robustness_curves.csv), [the clean-relative AP@0.5:0.95 curves](../results/figures/robustness_map_50_95_relative.png), and [`statistical_robustness_comparison.csv`](../results/tables/statistical_robustness_comparison.csv). The artifacts support H4: salt-and-pepper noise is most damaging for both detectors, the severity-5 relative ordering changes by corruption, and only selected contrasts such as severe-darkness AP@0.5:0.95 retention receive multiplicity-controlled support.

**Boundary.** This is a seed-17-only digital-corruption result on 300 images from 183 patient groups. It does not establish robustness to acquisition physics, new scanners, population shift, or clinical deployment.

## H5 - Grad-CAM is a failure-analysis tool, not clinical-reasoning evidence

**Hypothesis.** Grad-CAM will show weak and heterogeneous spatial agreement with annotated lung-opacity boxes, making the maps useful for locating failure patterns and artifact sensitivity but insufficient as evidence that either detector reasons clinically.

**Operational check.** Compare each detector's all-target mean energy-in-box with its mean box-area reference and inspect pointing-game accuracy in the summary table; then inspect the preselected true-positive, false-positive, and false-negative-proxy panels for extra-box activation. The interpretation fails if the text treats a plausible map as a causal explanation or as validation of clinical reasoning, regardless of localization score.

**Existing evidence.** Check [`gradcam_localization_summary.csv`](../results/tables/gradcam_localization_summary.csv), [the shared true-positive panels](../results/figures/gradcam_good_predictions.png), [the shared false-positive panels](../results/figures/gradcam_bad_predictions.png), and [the false-negative proxy panels](../results/figures/gradcam_failure_cases.png). The frozen artifacts support H5 descriptively: energy-in-box exceeds the box-area reference only modestly, pointing accuracy is low for both detectors, and the qualitative panels show frequent activation on anatomy, borders, markers, and devices outside the annotated opacity.

**Boundary.** The analysis is seed-17-only and target- and layer-dependent. It contains no causal intervention or clinical-reasoning ground truth; false-negative maps use annotation-guided proxy candidates and do not explain an emitted detection.
