---
title: Comparative Analysis of Object Detectors for Medical Imaging
bibliography: references.bib
link-citations: true
---

All numerical results in this report are quoted or rounded from the committed
artifacts under [`results/`](../results/); no model inference, resampling, or
report-only metric recomputation was performed during report assembly. The
commands that regenerate every cited table and figure are indexed in the
repository [`README.md`](../README.md).

## 1. Introduction

Object detectors intended for medical images must be assessed on more than a
single clean-test accuracy score. A useful system must find clinically relevant
regions with an acceptable miss rate, remain stable under plausible image
degradation, operate within its deployment hardware budget, and expose enough
diagnostic evidence to support failure analysis. These requirements can pull in
different directions: a proposal-based detector may spend more computation on
each candidate, while a dense one-stage detector may favor throughput and model
compactness.

This study asks:

> Under identical data, a common evaluation protocol, and a matched training
> budget, how do a two-stage anchor-based Faster R-CNN and a modern one-stage
> anchor-free YOLO trade off detection accuracy, robustness to common image
> corruptions, and interpretability on medical images, and which trade-off is
> preferable for plausible deployment scenarios such as resource-constrained
> point-of-care assistance and retrospective batch screening?

The comparison uses `fasterrcnn_resnet50_fpn_v2` and Ultralytics YOLO11s on a
patient-grouped 5,000-study subset of the RSNA Pneumonia Detection Challenge.
Both models consume the same canonical annotations, use 640-pixel inputs, are
trained under the same seed grid and broadly matched optimization budget, and
are scored by one evaluator. The study then adds a seven-corruption,
five-severity stress test, matched-layer Grad-CAM analysis, and paired
image-level inference. The intended contribution is a controlled,
resource-scoped benchmark, not a claim that either detector is clinically
effective or safe.

## 2. Literature Review

### 2.1 Detector paradigms

Faster R-CNN introduced a Region Proposal Network that shares convolutional
features with an RoI-based second-stage classifier and box regressor
[@ren2015fasterrcnn]. The implementation studied here couples a ResNet-50
backbone to a Feature Pyramid Network (FPN), which supplies semantically useful
representations at several spatial scales [@lin2017fpn]. Anchors encode multiple
candidate shapes at each feature location; proposal filtering and the second
stage then refine a smaller candidate set.

YOLO began from the contrasting idea of predicting detections in a single
dense network pass [@redmon2016yolo]. This study uses YOLO11s rather than the
newer YOLO26 family because YOLO11 retains a conventional anchor-free,
NMS-based detection pipeline with greater continuity to the existing medical
detection literature. The implementation is pinned to
`ultralytics==8.4.110`. Its backbone and neck create stride-8, stride-16, and
stride-32 features; its decoupled dense head predicts classification scores and
distributed box offsets without predefined anchor shapes. Distributional box
regression follows the general approach developed for Generalized Focal Loss
[@li2020gfl]. YOLO11 has no standalone peer-reviewed architecture paper, so the
pinned source, official configuration, and instantiated model are treated as
the implementation authority [@ultralytics2024yolo11;
@ultralytics2026yoloarchitecture; @ultralytics2026yolo11config].

Architecture suggests hypotheses rather than a guaranteed ranking. A dense
one-stage head should reduce latency and model complexity, whereas proposal
refinement may help with subtle or variably sized findings. Published medical
detector results cannot resolve this comparison because their datasets,
preprocessing, augmentations, thresholds, and metric implementations differ.
Prior work has adapted Faster R-CNN and anchor-free detectors to the RSNA task
[@yao2021pneumonia; @wu2024pneumonia], but the simultaneous use of custom
backbones, anchors, losses, or preprocessing prevents attribution to detector
paradigm alone.

### 2.2 Robustness and explainability

Common-corruption benchmarks show why clean accuracy is insufficient:
performance can change sharply under lighting, noise, blur, and compression
without changing the semantic label [@hendrycks2019corruptions]. Detection
models are also vulnerable to such transformations [@michaelis2019detectionrobustness].
The present benchmark transfers the ordered, multi-severity stress-test
principle to chest radiographs while avoiding the claim that digital
corruptions faithfully reproduce scanner physics or clinical distribution
shift.

Grad-CAM uses gradients of a selected target score to weight convolutional
feature channels and produce a coarse spatial association map
[@selvaraju2017gradcam]. Detector explanations require additional decisions:
the exact candidate and score must be defined, discrete NMS and selection are
not explained, and layers must be matched by approximate spatial role rather
than name. Visually plausible saliency is not proof of causal or clinically
appropriate reasoning [@adebayo2018sanity; @arun2021saliency]. This study
therefore combines paired heatmaps with energy-in-box and pointing-game
localization measures [@zhang2016excitation].

## 3. Materials and Dataset

### 3.1 Dataset choice and task

The selected data are the 2018 RSNA Pneumonia Detection Challenge Stage 2
training set, whose expert bounding-box annotation process is documented by
Shih et al. [@shih2019rsna]. It was chosen over two smaller brain-MRI exports
because it has stronger annotation provenance, coherent negative-image
semantics, and an official mapping that permits patient-level grouping. The
canonical detector task has exactly one foreground category, `Lung Opacity`.
`Normal` and `No Lung Opacity / Not Normal` are study-level strata and remain
negative images with zero detector annotations; an opacity is a radiographic
finding, not a confirmed pneumonia diagnosis.

The full Stage 2 set contains 26,684 labeled studies and 9,555 boxes. To fit the
predeclared RTX 4060 Laptop GPU (8 GB VRAM), 16 GB RAM, and course-project time
budget, a deterministic seed-17, patient-grouped, stratified procedure selected
5,000 studies from 2,136 NIH patient groups. The Kaggle `patientId` is an exam
UUID; grouping instead uses the NIH patient prefix recovered from the official
RSNA mapping. No patient key crosses train, validation, or test.

**Table 1. Patient-grouped benchmark composition.**

| Split | Studies | Patient groups | Opacity | No opacity / not normal | Normal | Boxes |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,500 | 1,492 | 798 | 1,554 | 1,148 | 1,267 |
| Validation | 750 | 321 | 169 | 331 | 250 | 277 |
| Test | 750 | 323 | 169 | 331 | 250 | 268 |
| **Total** | **5,000** | **2,136** | **1,136** | **2,216** | **1,648** | **1,812** |

Table 1 is sourced from the generated
[`rsna-pneumonia-5000-audit.json`](../data/manifests/rsna-pneumonia-5000-audit.json)
and committed split manifests. Figure 1 confirms that the three study strata
retain the intended split proportions.

![Figure 1. Study-level distribution after patient-grouped splitting.](../results/figures/rsna_class_distribution.png)

### 3.2 Integrity, preprocessing, and data governance

The complete metadata audit found 26,684 valid studies, 9,555 valid positive
boxes, and no missing/non-numeric, non-positive-area, off-image, or exact
duplicate positive boxes. DICOM conversion streams one study at a time,
inverts `MONOCHROME1`, applies finite per-image min-max scaling, and writes
8-bit grayscale PNG. Per-split COCO JSON is the canonical annotation format
used by both model adapters. Figure 2 shows the generated 12-image EDA sample;
its role is illustrative and not a substitute for expert re-annotation.

![Figure 2. Deterministically selected radiographs from all splits and study strata; red rectangles are Lung Opacity annotations.](../results/figures/rsna_annotation_samples.png)

The challenge uses bespoke RSNA terms that permit research and education while
prohibiting re-identification and imposing attribution requirements on shared
data. Raw images and processed pixels are not redistributed in this repository.
The cohort is historical, single-institution, adult-heavy, and enriched rather
than prevalence representative; these properties constrain every downstream
claim.

## 4. Experimental Methodology

### 4.1 Controlled training protocol

Both arms use the same patient-safe data, one foreground class, 640 x 640 input,
COCO transfer initialization, SGD, a maximum of 30 epochs, validation
mAP@0.5:0.95 early stopping after a minimum of eight epochs, and seeds 17, 42,
and 137. The primary seed-17 runs were developed first; the two additional
seeds were spent only where they add the most inferential value—the final
held-out comparison. No stochastic training augmentation is used. In
particular, YOLO mosaic, mixup, cutmix, copy-paste, HSV changes, flips,
geometric transforms, erasing, auto-augmentation, and multi-scale training are
disabled to avoid changing both detector and training distribution.

Hardware and numerical constraints create disclosed asymmetries. Faster R-CNN
uses physical batch 2 with two-step gradient accumulation, float16 AMP, frozen
pretrained BatchNorm running statistics, learning rate 0.005, and a plateau
scheduler. YOLO11s uses physical/effective batch 4, bfloat16 AMP with float32
target assignment and loss, native BatchNorm updates, one-epoch warmup to
learning rate 0.001, and a constant post-warmup learning rate. Matched YOLO
settings caused reproducible numerical collapse; the accepted exceptions were
made before the valid benchmark runs and are treated as threats to a pure
architecture-only interpretation.

### 4.2 Unified evaluation

Training and checkpoint selection use only train and validation data. Once all
six validation-selected checkpoints are frozen, one adapter-to-metric harness
opens the held-out 750-image test set. Both frameworks emit original-image
`xyxy` boxes, canonical category IDs, and scores. The common evaluator applies:

- official pycocotools AP at IoU 0.50 and averaged over 0.50:0.95;
- global micro precision, recall, and F1 at score 0.25 and matching IoU 0.50;
- greedy, class-aware, score-ordered matching; and
- mean IoU and box Dice only over matched true-positive boxes.

Conditional IoU and Dice do not penalize missed findings and must be read with
recall. Framework-native validation mAP is used only to choose checkpoints.
All reported summary values are arithmetic mean ± sample standard deviation
over three seeds unless stated otherwise.

### 4.3 Compute, robustness, explainability, and inference

Compute profiles use batch-1 AMP inference, ten warm-up images, 100 timed images,
and CUDA synchronization. They record FPS, latency, parameter counts, estimated
registered-operation GFLOPs, training time, and peak allocated training memory.
FLOP counts exclude unsupported operations and are secondary to measured
latency. Faster R-CNN performs resizing inside its timed forward, while the YOLO
profile resizes before its timed forward-plus-NMS; the speed comparison is thus
deployment-oriented, not a pure kernel benchmark.

Robustness uses the seed-17 checkpoints and a fixed proportional 300-image test
sample containing 68 positive images, 232 negative images, 111 boxes, and 183
patients. Seven corruptions—darker, brighter, Gaussian noise, salt-and-pepper
noise, Gaussian blur, motion blur, and JPEG compression—are each evaluated at
five ordered severities. Both detectors receive identical corrupted pixels.

Explainability reuses the same 300 images and all 111 boxes. Ordinary ReLU
Grad-CAM hooks a stride-16, 40 x 40 pre-neck backbone tensor in each model:
ResNet-50 `backbone.body.layer3` for Faster R-CNN and YOLO11s `model.6`.
The target is the foreground probability of the retained low-threshold
candidate with highest IoU to each ground-truth box. Missed operating-point
detections use an explicitly labeled annotation-guided proxy candidate.

Statistical analysis reconstructs every metric from frozen prediction bundles.
The clean comparison uses 2,000 paired hierarchical image/seed bootstrap draws
and 5,000 paired image-label permutations. The corruption analysis resamples
clean and corrupted evidence jointly. It reports pointwise 95% percentile CIs,
raw and Holm-adjusted p-values, and paired jackknife Cohen's d. Holm correction
is applied across the seven clean endpoints and, for corruption, separately
within each metric/estimand family across conditions.

## 5. Baseline Faster R-CNN Implementation

The baseline is Torchvision `fasterrcnn_resnet50_fpn_v2` initialized from the
default COCO weights. The RPN and RoI heads are adapted to the config-derived
single foreground class while background remains implicit. Six DataLoader
workers are used in non-persistent train/validation pools so the two pools do
not coexist beyond the 16 GB Windows memory budget.

The seed-17 run stopped after epoch 11; epoch 6 was selected by validation
mAP@0.5:0.95. Table 2 is a direct excerpt of
[`faster_rcnn_baseline_validation.csv`](../results/tables/faster_rcnn_baseline_validation.csv)
and [`faster_rcnn_compute.csv`](../results/tables/faster_rcnn_compute.csv).

**Table 2. Seed-17 Faster R-CNN validation and compute summary.**

| Validation/compute measure | Faster R-CNN seed 17 |
|---|---:|
| Validation precision / recall / F1 | 0.1414 / 0.6895 / 0.2346 |
| Validation mAP@0.5 / mAP@0.5:0.95 | 0.3314 / 0.1276 |
| FPS / mean latency | 11.00 / 90.92 ms |
| Parameters / estimated GFLOPs | 43.26 M / 450.76 |
| Peak training GPU memory | 1,556.60 MiB |
| Training time | 7,017.80 s (1.95 h) |
| Checkpoint size | 165.38 MiB |

Training loss decreased through the accepted run, while validation AP
fluctuated after the epoch-6 maximum. The plateau scheduler reduced learning
rate at epoch 10, and the patience rule stopped the run at epoch 11 (Figure 3).

![Figure 3. Faster R-CNN seed-17 training and validation curves; the dashed line marks the selected epoch.](../results/figures/faster_rcnn_training_curves.png)

## 6. YOLO Implementation

The comparison model is YOLO11s initialized from the pinned official
`yolo11s.pt` checkpoint. It is a small, dense, anchor-free model with native
multi-scale prediction and NMS. A hardlinked YOLO data view is derived from the
same canonical COCO records; it does not define an independent split or label
source.

The seed-17 run stopped at epoch 15 and retained epoch 10. Table 3 is taken from
[`yolo_baseline_validation.csv`](../results/tables/yolo_baseline_validation.csv)
and [`yolo_compute.csv`](../results/tables/yolo_compute.csv).

**Table 3. Seed-17 YOLO11s validation and compute summary.**

| Validation/compute measure | YOLO11s seed 17 |
|---|---:|
| Validation precision / recall / F1 | 0.5714 / 0.2022 / 0.2987 |
| Validation mAP@0.5 / mAP@0.5:0.95 | 0.2646 / 0.0869 |
| FPS / mean latency | 65.24 / 15.33 ms |
| Parameters / estimated GFLOPs | 9.43 M / 21.42 |
| Peak training GPU memory | 1,148.16 MiB |
| Training time | 1,975.64 s (32.93 min) |
| Checkpoint size | 18.28 MiB |

Figure 4 shows decreasing training losses and the validation AP maximum at the
selected epoch. Ultralytics' native epoch-10 mAP@0.5:0.95 of 0.07335 drove
checkpoint selection only; the 0.0869 value in Table 3 comes from the common
final validation evaluator.

![Figure 4. YOLO11s seed-17 training and validation curves; the dashed line marks the selected epoch.](../results/figures/yolo_training_curves.png)

## 7. Quantitative Performance Comparison

The common evaluator processed 750 held-out images and 268 boxes for each of
the six frozen checkpoints. Tables 4a and 4b are rounded presentations of
[`detector_comparison.csv`](../results/tables/detector_comparison.csv); the
six run-level records remain in
[`detector_comparison_per_seed.csv`](../results/tables/detector_comparison_per_seed.csv).

**Table 4a. Held-out predictive metrics, mean ± sample SD over three seeds.**

| Predictive metric | Faster R-CNN | YOLO11s |
|---|---:|---:|
| Precision | 0.1626 ± 0.0439 | **0.3730 ± 0.0395** |
| Recall | **0.6381 ± 0.0526** | 0.1356 ± 0.0094 |
| F1 | **0.2558 ± 0.0493** | 0.1981 ± 0.0048 |
| Conditional matched-box IoU | 0.6732 ± 0.0084 | **0.6971 ± 0.0189** |
| Conditional matched-box Dice | 0.7997 ± 0.0065 | **0.8172 ± 0.0134** |
| mAP@0.5 | **0.3084 ± 0.0123** | 0.1643 ± 0.0226 |
| mAP@0.5:0.95 | **0.1023 ± 0.0036** | 0.0549 ± 0.0080 |

**Table 4b. Compute metrics, mean ± sample SD over three seeds.**

| Compute metric | Faster R-CNN | YOLO11s |
|---|---:|---:|
| FPS, batch 1 | 17.42 ± 5.69 | **52.94 ± 10.65** |
| Mean inference time | 62.72 ± 24.58 ms | **19.36 ± 3.49 ms** |
| Total parameters | 43.26 M | **9.43 M** |
| Estimated GFLOPs/image | 450.76 | **21.42** |
| Peak training memory | 1,556.92 ± 0.27 MiB | **1,148.16 ± 0.00 MiB** |
| Training time | 6,211.41 ± 2,566.64 s | **1,833.21 ± 214.60 s** |

Faster R-CNN has about 1.86 times YOLO11s' mean mAP@0.5:0.95 and a 0.5025
absolute recall advantage. YOLO11s is more selective, with a 0.2105 precision
advantage, and substantially cheaper: roughly three times the measured
throughput, 78% fewer parameters, and about 21 times fewer estimated GFLOPs.
YOLO's slightly higher matched-box IoU and Dice describe only the relatively
small set of findings it detects; they do not offset its much lower recall.

## 8. Robustness Evaluation

The clean 300-image subset mAP@0.5:0.95 is 0.147802 for Faster R-CNN and
0.076295 for YOLO11s. Averaged equally across 35 corrupted conditions, the raw
scores are 0.112898 and 0.054099, while clean-relative retention is 0.763846
and 0.709083. These correspond to mean degradations of 23.62% and 29.09%.
Faster R-CNN has higher raw mAP@0.5:0.95 in every clean and corrupted matched
condition. The full 72-row evidence is
[`robustness_results.csv`](../results/tables/robustness_results.csv), with tidy
per-type and family curves in
[`robustness_curves.csv`](../results/tables/robustness_curves.csv) and
[`robustness_family_mean_curves.csv`](../results/tables/robustness_family_mean_curves.csv).

**Table 5. Severity-5 clean-relative mAP@0.5:0.95 retention.**

| Severity-5 mAP@0.5:0.95 retention | Faster R-CNN | YOLO11s |
|---|---:|---:|
| Darker | **0.8674** | 0.1645 |
| Brighter | 0.6706 | **0.6882** |
| Gaussian noise | **0.4643** | 0.3692 |
| Salt and pepper | **0.2025** | 0.0570 |
| Gaussian blur | 0.6936 | **0.6949** |
| Motion blur | **0.6788** | 0.5780 |
| JPEG quality 20 | 0.6825 | **0.7662** |

Salt-and-pepper noise is the most damaging tested corruption for both
detectors. YOLO also collapses under the darkest condition: it emits no true
positive at the 0.25 operating threshold, leaving conditional IoU and Dice
undefined, though COCO AP remains defined from lower-score predictions. Mild
darkening slightly improves Faster R-CNN mAP on this finite sample, so the
empirical curves are not forced to be monotonic.

![Figure 5. Raw mAP@0.5:0.95 across all corruption types and severities.](../results/figures/robustness_map_50_95_raw.png)

![Figure 6. Clean-relative mAP@0.5:0.95 retention across the same grid.](../results/figures/robustness_map_50_95_relative.png)

These results describe deterministic corruptions applied after PNG conversion.
They do not simulate acquisition physics, reconstruction, DICOM windowing,
population shift, or a new institution.

## 9. Explainability Analysis

Table 6 is drawn directly from
[`gradcam_localization_summary.csv`](../results/tables/gradcam_localization_summary.csv).
The box-area fraction is a no-localization reference, not a chance-corrected
inferential baseline.

**Table 6. Grad-CAM localization on all ground-truth boxes in the fixed sample.**

| Detector | Valid / boxes | Mean energy in box | Box-area reference | Lift over area | Pointing accuracy |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 110 / 111 | 0.0869 | 0.0713 | +0.0156 | 0.1091 |
| YOLO11s | 111 / 111 | 0.0975 | 0.0718 | +0.0257 | 0.1261 |

On 110 targets with valid maps for both detectors, YOLO places more energy in
the ground-truth rectangle for 76 targets and Faster R-CNN for 34; the mean
Faster-minus-YOLO energy difference is -0.0091. The absolute values remain
weak. Faster R-CNN's valid true-positive maps average 0.1122 energy, but its
miss-proxy maps average only 0.0303. YOLO's corresponding descriptive values
are 0.1448 and 0.0877, over different status subsets.

The paired figures use cases selected from frozen prediction behavior before
CAM values were consulted: three shared high-IoU true positives, three shared
false positives on box-negative images, and three shared false negatives at
predeclared proxy-IoU quantiles. Green rectangles are ground truth and cyan
rectangles are the exact candidates whose foreground scores are differentiated.

![Figure 7. Paired Grad-CAM maps for shared high-IoU true-positive detections.](../results/figures/gradcam_good_predictions.png)

![Figure 8. Paired Grad-CAM maps for shared false positives on box-negative studies.](../results/figures/gradcam_bad_predictions.png)

![Figure 9. Paired annotation-guided proxy maps for shared false negatives.](../results/figures/gradcam_failure_cases.png)

Faster R-CNN tends to produce more clustered hotspots, but they are not
consistently centered on the boxed opacity. YOLO11s is more diffuse and
punctate while attaining a modest energy-in-box advantage. Both frequently
activate on the mediastinum, shoulders, chest wall, image borders, markers,
devices, and other anatomy. Neither set of maps supports a claim that the model
is reasoning clinically or reliably localizing the lesion. In particular,
false-negative proxy maps answer a conditional diagnostic question about a
latent candidate; they do not explain an emitted detection.

## 10. Statistical Analysis

Table 7 reports the clean three-seed paired analysis from
[`statistical_clean_comparison.csv`](../results/tables/statistical_clean_comparison.csv).
Differences are Faster R-CNN minus YOLO11s. Intervals are pointwise 95%
bootstrap CIs; family-wise claims use the Holm-adjusted p-value.

**Table 7. Clean paired statistical comparison.**

| Metric | Faster R-CNN | YOLO11s | Difference [95% CI] | Holm p | d |
|---|---:|---:|---:|---:|---:|
| Precision | 0.1626 | 0.3730 | -0.2105 [-0.3050, -0.1177] | 0.0014 | -0.182 |
| Recall | 0.6381 | 0.1356 | 0.5025 [0.4301, 0.5760] | 0.0014 | 0.624 |
| F1 | 0.2558 | 0.1981 | 0.0577 [-0.0130, 0.1355] | 0.0510 | 0.076 |
| Conditional IoU | 0.6732 | 0.6971 | -0.0239 [-0.0530, 0.0047] | 0.0510 | -0.075 |
| Conditional Dice | 0.7997 | 0.8172 | -0.0174 [-0.0377, 0.0026] | 0.0510 | -0.076 |
| mAP@0.5 | 0.3084 | 0.1643 | 0.1441 [0.0967, 0.1949] | 0.0014 | 0.095 |
| mAP@0.5:0.95 | 0.1023 | 0.0549 | 0.0474 [0.0311, 0.0683] | 0.0016 | 0.236 |

After correction, the data retain evidence for Faster R-CNN's recall and both
AP advantages, and for YOLO11s' precision advantage. F1 and conditional
localization do not cross the 0.05 Holm threshold. The effect-size magnitudes
differ from the raw aggregate gaps because paired jackknife Cohen's d
standardizes image-level pseudovalues rather than treating mAP as a mean of
per-image AP values.

The corruption table contains 497 rows in
[`statistical_robustness_comparison.csv`](../results/tables/statistical_robustness_comparison.csv).
Although Faster R-CNN has higher point-estimate raw mAP@0.5:0.95 for all 35
corruptions, only darkness severity 5 survives the grid-wide Holm correction
for that endpoint. Raw mAP is 0.1282 versus 0.0126, a difference of 0.1156
[0.0712, 0.1702] (`p_Holm=0.0070`, `d=0.241`). Clean-relative retention is
0.8674 versus 0.1645, a difference of 0.7029 [0.3982, 0.8424]
(`p_Holm=0.0070`, `d=0.433`). This distinction between point estimates and
multiplicity-controlled evidence prevents overclaiming across a correlated
grid.

McNemar's test is intentionally omitted. The task has multiple targets per
image plus false positives on negative images, not one independent binary
outcome per image. Collapsing detection into an image-level correct/incorrect
flag would discard the outcome structure, while target-level indicators would
remain nested within images.

## 11. Discussion

### 11.1 Accuracy and operating-point behavior

The two paradigms occupy meaningfully different operating regimes. Faster
R-CNN detects far more reference findings and achieves substantially higher AP;
these advantages survive paired correction for recall and both AP endpoints.
YOLO11s produces fewer detections and higher precision. Its slightly better
conditional IoU and Dice do not establish superior overall localization because
those statistics exclude missed findings, and their corrected comparisons are
inconclusive. Therefore, the result is not simply “Faster R-CNN has higher
mAP”: it is a recall-heavy proposal detector versus a selective, compact dense
detector at the fixed threshold.

The absolute results are modest for both models. Faster R-CNN's mean precision
is only 0.163 at the chosen operating point, while YOLO11s' mean recall is only
0.136. Neither frozen system is an acceptable autonomous clinical detector.
Threshold calibration could move each precision-recall trade-off, but it would
be a new experiment and is not inferred from these fixed results.

### 11.2 Robustness and explanation evidence

Faster R-CNN combines its clean accuracy advantage with higher raw AP in every
tested corruption condition and better mean retention over the full grid. Its
darkness robustness is especially stronger, and that difference survives the
35-condition correction. Yet both detectors are fragile to impulse noise, and
the single primary-seed, digitally corrupted sample cannot establish real-world
clinical robustness.

Explainability does not supply a reason to relax those concerns. YOLO11s has a
small descriptive energy-in-box advantage, while Faster R-CNN maps are more
clustered, but both pointing accuracies are near 0.1 and both models often
highlight extra-box anatomy and acquisition artifacts. The maps are useful for
failure investigation, not as evidence that either system uses a medically
valid causal mechanism. Consequently, neither detector earns a deployment
preference on interpretability grounds.

### 11.3 Scenario-specific recommendation

For **high-sensitivity retrospective screening or server-side case
prioritization**, Faster R-CNN is preferred. Its measured recall, AP, and
corruption-grid raw performance are stronger. Although 17.42 FPS is slower than
YOLO, it still processes images far faster than a human reading workflow on the
tested GPU. The cost is more false positives, parameters, FLOPs, and latency.

For **resource-constrained point-of-care assistance in which a human reviews
every image**, YOLO11s is conditionally preferred. Its 9.43 M parameters,
21.42 GFLOPs, 52.94 FPS, and higher precision make it easier to place on
constrained hardware and reduce nuisance alerts. Its 0.136 recall nevertheless
makes it unsuitable as the sole triage gate or a rule-out system. It is
defensible only as an auxiliary cue after threshold/model redesign and new
validation.

For **autonomous diagnosis, disease exclusion, or treatment guidance**, neither
model is suitable. Accuracy is low, Grad-CAM localization is weak, corruptions
are synthetic, and the study is retrospective and single-source. The benchmark
provides no evidence of prospective benefit or safety.

This assignment of paradigms is conditional rather than universal. If
point-of-care triage requires high sensitivity more than low latency, Faster
R-CNN remains the better of the measured models despite its compute cost. If a
retrospective service faces a severe throughput or edge-memory limit, YOLO's
efficiency may dominate, but accepting its miss rate would be a deployment
choice not supported by this benchmark. On the measured RTX 4060, raw detector
speed is unlikely to be the bottleneck for ordinary single-radiograph use; data
transfer, viewer integration, calibration, and human response also matter.

### 11.4 Limitations and scope of claims

The benchmark uses one historical, single-institution dataset, one foreground
category, a fixed 5,000-study subset, and only three training seeds. Robustness
and explainability use one seed-17 checkpoint per detector on a 300-image,
111-box subset. Repeated exams within the test set mean image-level bootstrap
units are not fully patient independent. Training precision, learning rate,
normalization, and scheduler differ where needed for stable YOLO learning; the
augmentation policy is controlled but may understate YOLO's conventional
augmentation-rich performance. Common corruptions act on converted PNGs and
are not scanner or site-shift models. Bounding boxes are coarse lesion
surrogates, Grad-CAM is layer- and target-dependent, and one Faster R-CNN CAM is
zero energy. The full consolidated limitations are maintained in
[`docs/LIMITATIONS.md`](../docs/LIMITATIONS.md).

Clinical deployment would require prospective clinical validation, external
multi-site validation, subgroup and calibration assessment, human-factors and
workflow testing, safety and cybersecurity controls, ongoing monitoring, and
the applicable medical-device regulatory process (for example FDA clearance or
CE marking). Those activities are beyond this retrospective benchmark. This is
a scope-of-claims statement, not an implementation requirement or evidence that
either model is ready to enter such a process.

## 12. Conclusions and Future Work

Under the frozen data and evaluation protocol, Faster R-CNN is the stronger
accuracy- and robustness-oriented detector: it has higher recall and AP with
multiplicity-corrected evidence, and higher raw mAP across every tested
corruption condition. YOLO11s is the stronger efficiency-oriented detector: it
is roughly three times faster, uses about one fifth of the parameters and one
twenty-first of the estimated FLOPs, and produces higher precision. Neither
model is reliably lesion-focused under the selected Grad-CAM protocol, and
neither supports clinical use.

The deployment implication is therefore conditional. Faster R-CNN is the more
defensible research choice for accuracy-sensitive, GPU-backed screening or
case prioritization. YOLO11s is attractive for constrained, human-in-the-loop
assistance where compute and alert burden dominate, but its measured miss rate
precludes use as an autonomous screen. Future work should evaluate the full
cohort and external sites, use patient-cluster resampling, run robustness and
explainability over multiple seeds, calibrate thresholds and probabilities,
test clinically motivated acquisition shifts, compare augmentation-rich and
matched-control recipes as separate ablations, add saliency sanity checks or a
detector-specific method such as D-RISE [@petsiuk2021drise], and perform
prospective workflow evaluation before making any clinical claim.
