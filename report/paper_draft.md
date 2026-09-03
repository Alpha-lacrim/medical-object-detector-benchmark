---
title: "A Controlled Comparative Study of Faster R-CNN and YOLO11s Pipelines for Lung-Opacity Localization on Chest Radiographs"
bibliography: references.bib
link-citations: true
reference-section-title: References
---

## Abstract

**Background:** Detector comparisons can confound the model with changes in
data, preprocessing, augmentation, score thresholds, and evaluation. We
conducted a controlled multi-axis characterization of two disclosed
lung-opacity detector pipelines, treating operating-score behavior as a
primary object of study.

**Methods:** This retrospective internal benchmark used a deterministic,
patient-disjoint subset of 5,000 RSNA radiographs. A two-stage Faster R-CNN
pipeline and a one-stage YOLO11s pipeline used common 640-pixel inputs,
disabled stochastic augmentation, validation-frozen model selection, and one
COCO-style evaluator. All five attempted training seeds per detector were
retained. Detector-specific thresholds were selected on validation using the
frozen n=3 procedure and applied unchanged across the n=5 internal-testing
sensitivity. The primary training-procedure estimand combined patient-cluster
and detector-run resampling; checkpoint-conditional permutation was secondary.

**Results:** The same raw score cutoff selected materially different operating
regimes. At 0.25, YOLO11s appeared more precise but had much lower recall; in
contrast, Faster R-CNN had higher mean precision at 97 of 101 AP@0.5 recall
positions, and mean mAP@0.5:0.95 was 0.0995 versus 0.0542. Within the evaluated
0.01--0.99 score sweep, Faster R-CNN had higher observed sensitivity at all five
reported false-positive budgets. Under the documented detector-specific
profiling procedure on the measured laptop, YOLO11s achieved approximately
3-fold higher measured throughput, 78% fewer parameters, and about 21-fold
fewer estimated registered operations. One retained YOLO11s run emitted no
detections at its frozen 0.05 threshold, giving zero precision, recall, and F1,
yet had mAP@0.5:0.95 of 0.05558 from lower-score retained predictions.
Detection-specific calibration error was descriptively higher for YOLO11s
(mean D-ECE 0.0990 versus 0.0320). Primary bootstrap intervals were wholly
positive for Faster R-CNN minus YOLO11s recall, F1, and both AP endpoints; the
fixed-threshold precision interval crossed zero. The separate
checkpoint-conditional permutation result favored YOLO11s precision at that
cutoff. Conditional matched-box localization was inconclusive. Controlled
stress tests and Grad-CAM analyses were descriptive and condition-specific,
with weak localization for both pipelines.

**Conclusions:** AP, raw score scale, fixed-threshold behavior, and
detection-level calibration described different properties of these two
pipelines. The findings support a bounded internal benchmark, not a causal
architecture ranking, pneumonia diagnostic validation, or clinical utility.

## 1. Introduction

Object detection on chest radiographs is not adequately characterized by one
internal-testing mean average precision (mAP) value. A model can preserve ranking
quality while producing an unstable score scale, appear precise only because a
shared threshold is unusually selective, localize successful detections while
missing most annotated findings, or lose performance under small changes in
image formation and display. Conversely, a computationally heavier system may
offer better coverage without being the appropriate choice for a constrained
device. Evaluation therefore has to connect ranking accuracy, explicit
operating points, calibration, compute, robustness, explanation behavior, and
uncertainty.

Published medical-detector comparisons rarely isolate all of these dimensions.
Changing a detector commonly changes its backbone, augmentation stack, input
handling, optimization recipe, post-processing, score threshold, and metric
implementation. Point estimates from such studies remain valuable within their
own protocols, but they cannot by themselves identify whether a difference is
caused by the detector organization, the training recipe, or the evaluation
path. A shared numerical confidence threshold creates an additional problem:
two pipelines can assign scores on different scales, so the threshold can
select different portions of their respective output distributions.

We study this problem using two deliberately disclosed pipelines: Torchvision
`fasterrcnn_resnet50_fpn_v2`, representing a two-stage anchor-based proposal
detector, and Ultralytics YOLO11s, representing a compact one-stage anchor-free
dense detector. The task is localization of the RSNA challenge category `Lung
Opacity` on chest radiographs. It is not pneumonia diagnosis: the target is a
non-specific radiographic finding, and the negative strata include both normal
studies and abnormal studies without an annotated opacity.

The primary research question was:

> Under one patient-disjoint split, shared canonical preprocessing and
> augmentation policy, a common training-budget framework, and one
> model-independent evaluator, how do Faster R-CNN and YOLO11s trade off
> lung-opacity detection coverage and ranking accuracy, operating regimes and
> confidence calibration, computational efficiency, corruption-specific
> robustness, and Grad-CAM-observed failure patterns?

The study makes three contributions. First, it constructs a controlled,
patient-disjoint comparison in which both pipelines use the same images,
annotations, input size, absence of stochastic training augmentation, seed
grid, internal-testing boundary, and metric path. Second, it treats threshold behavior
as a result rather than a hidden implementation choice by combining an
exploratory threshold sweep, official precision-recall curves,
validation-selected operating points, free-response ROC (FROC), and a separate
recall-weighted F-beta validation sensitivity analysis. Third, it broadens the evidence
base beyond clean mAP through detection-specific calibration, patient-cluster
inference, accuracy-efficiency Pareto analysis, digital and
acquisition-motivated robustness, and Grad-CAM localization plus
randomization sanity checks.

The hypotheses were recorded retrospectively after most experimental artifacts
had been frozen; they are result-linked checks, not preregistration. More
importantly, the design does not estimate a universal causal effect of
"one-stage" versus "two-stage" architecture. Numerical precision,
normalization, learning rate, scheduling, loss, assignment, and intrinsic
post-processing could not all be made identical. The appropriate unit of claim
is therefore these two working pipelines under the stated controls and hardware
budget.

## 2. Related Work

### 2.1 Two-stage and one-stage detector organization

Faster R-CNN couples a Region Proposal Network to an RoI-based second-stage
classifier and box regressor while sharing convolutional features
[@ren2015fasterrcnn]. The implementation used here adds a ResNet-50 Feature
Pyramid Network, which represents objects at multiple semantic scales
[@lin2017fpn]. Multiple anchor shapes generate spatial hypotheses; proposal
filtering and the second stage then refine a smaller candidate set.

YOLO originated from the contrasting formulation of dense detection in one
network pass [@redmon2016yolo]. YOLO11 retains that one-stage organization but
uses anchor-free grid references, multi-scale features, separate
classification and regression branches, distributed box offsets, and
non-maximum suppression. Distributed regression follows the general approach
introduced with Generalized Focal Loss [@li2020gfl]. Because YOLO11 has no
standalone peer-reviewed architecture paper, the pinned source release,
official configuration, and instantiated graph are the implementation
authority [@ultralytics2024yolo11; @ultralytics2026release;
@ultralytics2026yoloarchitecture; @ultralytics2026yolo11config]. YOLO11s was chosen over a larger model to fit the
8-GB GPU budget and over the newer NMS-free YOLO26 family to retain closer
continuity with the medical-detector literature.

Architecture motivates a computational hypothesis, not an accuracy ranking.
The dense head avoids per-proposal second-stage processing, while proposal
refinement could behave differently for subtle or variably sized opacity
annotations. The realized result also depends on optimization, score formation,
candidate filtering, and the dataset.

### 2.2 The comparability gap in medical detection studies

The RSNA resource paper documents a large chest-radiograph collection and an
expert annotation workflow for possible pneumonia-like pulmonary opacities
[@shih2019rsna]. Subsequent detector studies demonstrate the relevance of both
proposal-based and dense approaches, but do not form a controlled
architecture-family experiment. Modified Faster R-CNN work changes the
backbone, feature pyramid, anchors, and suppression strategy
[@yao2021pneumonia]. Anchor-free RSNA work combines a custom detector with
augmentation, focal loss, study-specific thresholds, NMS, and its own AP/AR
definitions [@wu2024pneumonia]. Related brain-MRI studies change the modality,
pretraining, loss, input organization, and YOLO variant
[@kang2023rcsyolo; @kang2025pkyolo].

| Prior work | Data and detector focus | Relevance to the present study | Comparability limit |
|---|---|---|---|
| Shih et al. [@shih2019rsna] | RSNA chest radiographs and expert opacity boxes | Defines the source resource and target | Resource paper, not a detector-family comparison |
| Yao et al. [@yao2021pneumonia] | Modified Faster R-CNN for chest radiographs | Shows two-stage adaptation to low-contrast opacity | Backbone, anchors, preprocessing, and post-processing change together |
| Wu et al. [@wu2024pneumonia] | Anchor-free RSNA detector | Supports dense localization on the same task family | Augmentation, losses, thresholds, and metrics differ |
| Kang et al. [@kang2023rcsyolo] | Efficiency-oriented YOLO for 2-D brain MRI | Treats medical detection as an accuracy-speed problem | Different modality and YOLO-to-YOLO emphasis |
| Kang et al. [@kang2025pkyolo] | Multiplanar MRI with domain pretraining | Highlights small-target and pretraining choices | Different data organization, pretraining, and model variants |

Across this reviewed set, split construction, preprocessing, augmentation,
thresholds, and metric aggregation move alongside the architecture. The gap is
therefore not an absence of high-performing detectors; it is the scarcity of
cross-paradigm evidence in which these major sources of variation are held
common and the remaining asymmetries are disclosed. Our contribution narrows
that gap for two specific pipelines. It does not close it for detector families
in the abstract.

### 2.3 Robustness, calibration, and explanation

Common-corruption studies show that clean accuracy can fail to predict behavior
under lighting, noise, blur, and compression shifts
[@hendrycks2019corruptions], including in object detection
[@michaelis2019detectionrobustness]. The ordered multi-severity principle is
useful for controlled stress testing, but digital corruptions applied to PNGs
must not be presented as scanner, acquisition, site, or population shift. We
therefore analyze post-conversion corruptions and radiography-motivated
synthetic acquisition/display transformations as separate experiments with
different claim boundaries.

Confidence calibration is likewise distinct from threshold selectivity.
Changing a threshold describes how a detector moves along a precision-recall
operating curve. Calibration asks whether the stated confidence agrees with the
empirical correctness rate. Because object detectors emit variable numbers of
located boxes, classifier-style expected calibration error is inadequate. We
use the multivariate Detection Expected Calibration Error (D-ECE) framework,
which conditions correctness on confidence and predicted box geometry
[@kuppers2020calibration].

Grad-CAM weights convolutional features using gradients of a defined target
score and yields a coarse spatial association map [@selvaraju2017gradcam]. For
detectors, the target candidate, score, and feature layer must be made explicit;
candidate selection and NMS are not themselves explained. Plausible heatmaps
can persist after model randomization and do not demonstrate causal or
clinically appropriate reasoning [@adebayo2018sanity; @arun2021saliency]. We
therefore pair qualitative maps with energy-in-box and pointing-game measures
[@zhang2016excitation], then assess parameter sensitivity with cumulative
head-to-backbone randomization and response to a separate inference-time
input-pixel perturbation. Adebayo et al.'s distinct data-randomization test
permutes training labels and retrains the model; we did not perform that test.

## 3. Materials and Methods

### 3.1 Dataset, target, and patient-disjoint split

We used the Stage 2 training set from the 2018 RSNA Pneumonia Detection
Challenge. The complete source contains 26,684 labeled radiographs and 9,555
positive boxes. The canonical detector task contains one foreground category,
`Lung Opacity`. The study-level categories `Normal` and `No Lung Opacity / Not
Normal` remain zero-box negative images. We avoid using "pneumonia detection"
as the task label because the bounding boxes represent a non-specific
radiographic finding rather than microbiologically or clinically confirmed
pneumonia.

Hardware and time constraints motivated a deterministic 5,000-study subset;
there was no formal statistical sample-size or power calculation. Selection
used seed 17, SHA-256 ordering, and the three study-level label strata while
keeping every examination from an NIH patient group together and tracking the
source-cohort stratum proportions. The remaining 21,684 labeled source studies
were excluded for compute scope rather than annotation-quality failure. The
Kaggle `patientId` identifies an examination rather than a unique person, so
the official RSNA mapping was used to recover NIH patient keys. The selected
studies came from 2,136 patient groups and were assigned as whole groups to
train, model-optimization (named `validation` in repository artifacts), or
internal-testing splits. Patient-key intersections among splits were empty.
The hardware-driven size improves reproducibility on the stated laptop but
widens sampling uncertainty and limits generalizability to the complete source
cohort.

| Split | Radiographs | NIH patient groups | Lung Opacity | No opacity / not normal | Normal | Boxes |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,500 | 1,492 | 798 | 1,554 | 1,148 | 1,267 |
| Validation | 750 | 321 | 169 | 331 | 250 | 277 |
| Internal testing | 750 | 323 | 169 | 331 | 250 | 268 |
| **Total** | **5,000** | **2,136** | **1,136** | **2,216** | **1,648** | **1,812** |

#### Canonical preprocessing

The metadata audit found no malformed, non-positive-area, off-image, or exact
duplicate positive boxes. One deterministic conversion path served both
pipelines: DICOM pixel arrays were checked for finite content, inverted for
`MONOCHROME1` polarity when needed, scaled per image to the finite minimum and
maximum, and written as 8-bit grayscale PNG. Canonical COCO JSON was the sole
annotation source; the YOLO view was derived from those same records rather
than from an independently prepared label set. Both pipelines received
640-pixel inputs, and neither received stochastic training augmentation. The
complete split manifests, source hashes, image inventory, and preprocessing
audit are indexed in the [Supplementary Materials
Index](../docs/SUPPLEMENTARY.md#s1-cohort-split-and-provenance-records).

### 3.2 Detector pipelines and controlled training factors

The two-stage arm was Torchvision
`fasterrcnn_resnet50_fpn_v2` with COCO initialization. The RPN and RoI heads
were adapted to the config-derived foreground class. The one-stage arm was
YOLO11s from Ultralytics `8.4.110`, also initialized from COCO weights. A
hardlinked YOLO view was generated from the canonical records; it did not
create a separate split or label source.

Shared factors were the patient-disjoint manifests, 640-pixel inputs, one
foreground class, COCO initialization, SGD, a maximum of 30 epochs,
validation-mAP@0.5:0.95 early stopping after at least eight epochs, and seeds
17, 42, 137, 271, and 314. Stochastic training augmentation was disabled in
both arms. In particular, the Ultralytics-only mosaic, mixup, cutmix,
copy-paste, HSV, flip, geometric, erasing, auto-augmentation, and multi-scale
options were off. This controlled the training distribution but could
disadvantage YOLO11s relative to its conventional augmentation-rich recipe.

The training recipe was not fully symmetric. Faster R-CNN used physical batch
2 with two-step gradient accumulation, float16 automatic mixed precision,
frozen pretrained BatchNorm running statistics, learning rate 0.005, and a
validation-plateau scheduler. YOLO11s used physical and effective batch 4,
bfloat16 forward/backward with float32 assignment and loss, native BatchNorm
updates, a one-epoch warmup to learning rate 0.001, and a constant post-warmup
rate. More closely matched YOLO settings reproducibly produced non-finite or
all-zero one-class heads during predeclared diagnostics. The accepted
exceptions preceded valid benchmark training. We therefore model training
recipe as an explicit comparison factor and interpret the experiment as two
controlled, disclosed pipelines rather than architecture alone. Full optimizer,
checkpoint, timing-gate, and environment details are delegated to the
[supplementary implementation record](../docs/SUPPLEMENTARY.md#s8-decisions-limitations-and-reproduction).

### 3.3 Seed/run design and unified internal testing

Checkpoint selection used model-optimization-split mAP only. After all
checkpoints were frozen, evaluation used the patient-disjoint internal-testing split.
Five attempts per detector were retained at seeds 17, 42, 137, 271, and 314;
none was replaced after its internal-testing behavior was observed. The numeric
seed labels provide within-program reproducibility metadata, not matched
stochastic blocks across the two training frameworks.
Both adapters emitted original-image `xyxy` boxes, canonical category IDs, and
scores to one evaluator. It computed official COCO AP at IoU 0.50 and averaged
from 0.50 to 0.95 [@lin2014coco]; global micro precision, recall, and F1 at
score 0.25 and match IoU 0.50; and mean box IoU and Dice over matched true
positives. COCO AP used predictions retained at score `>= 0.001` and did not
apply the selected single operating thresholds. Predictions were capped at 100
per image. Conditional IoU and Dice do not penalize missed targets and were
interpreted jointly with recall.

The evidence has deliberately different scopes:

- Clean precision, recall, F1, AP, and compute summaries use all five
  predeclared attempts per detector. Conditional IoU and Dice use Faster R-CNN
  `n=5` and YOLO11s `n=4` descriptively, and four complete seed pairs for
  inference, because YOLO11s seed 271 had no true positive at score 0.25.
- The original threshold sweep, validation-selected operating points, official
  precision-recall curves, FROC, and Pareto analysis remain frozen to seeds 17,
  42, and 137 (`n=3`) as historical/prespecified artifacts. A separately
  versioned all-attempt sensitivity recomputes internal-testing threshold/PR/FROC and
  Pareto evidence for all five runs. Threshold selection remains n=3; 0.69 and
  0.05 are applied unchanged to the five internal-testing bundles.
- Digital-corruption, raw-array acquisition-shift, and primary Grad-CAM analyses
  use the seed-17 checkpoint from each detector on one fixed 300-image sample.
- Detection calibration uses all five frozen internal-testing bundles. The XAI
  sanity extension uses a nested 50-image subset and the seed-17 checkpoints.

These sample-size labels travel with every result. Exhaustive seed rows and
explicit n=3 archives are linked in [Supplementary Sections S2 and
S3](../docs/SUPPLEMENTARY.md#s2-full-clean-seed-level-comparison).

### 3.4 Operating-point protocols and FROC

The original score-0.25 comparison was retained as a protocol-sensitivity
result, not a deployment choice. A frozen n=3 exploratory analysis evaluated
99 thresholds from 0.01 to 0.99 on the internal-testing predictions. It also read the
official pycocotools interpolated precision tensor at 101 recall positions,
rather than approximating the AP curve from the threshold grid.

Without changing the grid or matcher, a separately versioned sensitivity
repeated those internal-testing calculations over all five frozen runs per detector.
The same ten prediction bundles used by the clean comparison were hash-checked;
no checkpoint was loaded and no prediction was regenerated. The n=5 curves are
the principal operating-regime display under D-008 because they retain every
predeclared attempt, while the original n=3 curves remain unchanged provenance.

Primary single-threshold selection was performed independently on the six n=3
validation bundles. The same 99-point grid was evaluated with the common
matcher, and the threshold maximizing arithmetic mean validation F1 across the
three seeds was frozen for each detector; exact ties favored the higher
threshold. These thresholds were then applied once to the corresponding internal-testing
bundles. Batch 35 additionally applied the same thresholds to seeds 271 and
314. Internal-testing results did not feed back into selection in either scope.

FROC reparameterized the exploratory internal-testing sweep as sensitivity versus false
positives per image. At budgets of 0.125, 0.25, 0.5, 1, and 2 FP/image, each
seed contributed its highest observed sensitivity without exceeding the
budget. No interpolation or extrapolation was used. Historical n=3 and
sensitivity n=5 FROC curves use separate files and visible run-count labels;
neither selects a clinical threshold.

### 3.5 Recall-weighted F-beta and hypothetical error-loss sensitivity

The validation-only preference sensitivity used

$$
F_\beta(\tau)=
\frac{(1+\beta^2)P(\tau)R(\tau)}{\beta^2P(\tau)+R(\tau)},
$$

for $\beta\in\{1,3,5,10\}$. Beta is a recall-versus-precision preference
parameter; $\beta^2\in\{1,9,25,100\}$ is the relative recall weight in this
weighted harmonic mean, not an empirically measured clinical-harm ratio. In
count form, the objective is

$$
F_\beta(\tau)=\frac{(1+\beta^2)TP(\tau)}
{(1+\beta^2)TP(\tau)+\beta^2FN(\tau)+FP(\tau)},
$$

which is not the same objective as minimizing a linear weighted error. For each
detector and beta, the canonical sensitivity threshold maximized the lower
pointwise 95% bootstrap bound over the 0.01--0.99 grid. The original 2,000-draw
random stream was retained. A descriptive near-optimal plateau contained the
contiguous thresholds within 0.01 of the maximum lower bound. Each bootstrap
draw also supplied a mean-F-beta argmax and candidate selection frequency; this
diagnostic did not replace the canonical lower-bound rule.

A separate hypothetical analysis minimized
$L(\tau;r)=rFN(\tau)/N+FP(\tau)/N$ for assumed FN:FP loss ratios
$r\in\{1,9,25,100\}$, with $N$ the validation-image count. These are linear
box-error penalties, not measured patient harms or deployment utilities.
Decision D-006 keeps both analyses separate from the primary maximum-mean-F1
thresholds. Selection and diagnostics used validation only; no sensitivity
threshold was selected from or applied to internal testing, FROC, Pareto, or any
downstream outcome analysis.

### 3.6 Detection-specific confidence calibration

Calibration used all post-NMS detections retained by score `>= 0.001`. The
canonical matcher used stable descending-score order, a 100-detection image
cap, same-class pairing to the highest-IoU currently unmatched target, no
target reuse, and IoU `>= 0.50`. Following Küppers et al.
[@kuppers2020calibration], predicted class was a categorical stratum; the
one-class benchmark therefore had one stratum. Confidence, relative center
coordinates, width, and height formed the five numeric dimensions. Each was
partitioned into five equal-width bins. Internal edges entered the upper bin
and 1.0 entered the final bin. Cells with fewer than eight detections
contributed zero, while all emitted detections remained in the weighting
denominator. For supported cells $b$, D-ECE was

$$
\operatorname{D\text{-}ECE}=
\sum_b \frac{n_b}{N}\left|
\operatorname{precision}(b)-\operatorname{confidence}(b)
\right|.
$$

This is a calibration-error measure for black-box detection confidence
conditional on emitted detections; under a fixed protocol, lower D-ECE denotes
lower error. A missed ground-truth object has no emitted confidence and is
outside the estimand. Per run, we reported total detections, 3,125 possible
cells, occupied and supported cells, supported-detection fraction, and
supported-cell sizes. A predeclared descriptive grid crossed 3, 5, and 7 bins
per numeric dimension with minimum-cell sizes 1, 4, 8, and 16; the original
5-bin/minimum-8 setting remained primary. Confidence-floor sensitivity used
0.001, 0.005, 0.01, and 0.05. No setting selected a favorable result, no run
was specially handled, and no recalibration map was fitted. Reliability
diagrams were confidence-only marginal summaries, not visualizations of all
five D-ECE dimensions. D-ECE was programmatically separate from exam-level
outcome-probability calibration required for valid decision-curve analysis.
Conventional decision-curve analysis was not performed: frozen validation
predictions existed for only six of the ten retained detector runs, so a
complete set of validation-fitted, run-specific exam-outcome probability
mappings could not be frozen without dropping runs or using internal-testing
outcomes. The historical raw-score calculation was therefore excluded from the
main evidence rather than relabeled as probability-based DCA.

### 3.7 Compute and Pareto analysis

Compute profiles used batch-1 mixed-precision inference, 10 warm-up images, 100
timed images, and CUDA synchronization on an RTX 4060 Laptop GPU. We recorded
FPS, latency, parameters, training time, peak allocated memory, and estimated
registered-operation GFLOPs. The FLOP counter included registered convolution,
matrix, and batch-matrix operations but omitted unsupported work; measured
latency was therefore the primary runtime-efficiency measure. Faster R-CNN
resizing occurred inside its timed forward, whereas YOLO tensor resizing
occurred before timed forward-plus-NMS.

The frozen historical n=3 Pareto analysis and a separately labeled n=5
sensitivity paired run-specific AP or validation-selected internal-testing recall with
same-run FPS, latency, parameters, or estimated GFLOPs. Recall thresholds in
both scopes were selected from the original n=3 validation bundles. Strict
dominance required every seed of one detector to be better than every seed of
the other on both directed axes. Mean-only ordering was insufficient. The n=5
summary used equal-run means and sample SDs with five hardware rows per
detector; no n=3 and n=5 metric was mixed into one frontier.

### 3.8 Digital corruption and radiography-motivated synthetic sensitivity

The fixed robustness sample contained 300 internal-testing radiographs from 183 patients:
68 opacity-positive and 232 negative images with 111 boxes. Proportional
largest-remainder allocation preserved the three study strata and used no
detector result. Both detectors received identical corrupted pixels.

The post-conversion digital grid contained darker and brighter intensity,
Gaussian and salt-and-pepper noise, Gaussian and motion blur, and JPEG
compression, each at five ordered severities. Performance was reported both
raw and as clean-relative retention. These were deterministic transformations
of uint8 PNGs, not acquisition or site-shift simulations.

The separate stored-array study was re-audited against current DICOM PS3.3 and
PS3.4 [@dicom2026ps33; @dicom2026ps34]. All 300 objects record `Modality=CR`,
but their SOP Class is Secondary Capture Image Storage; every object is workstation-converted (`WSD`),
8-bit `MONOCHROME2`, and marked as previously lossily JPEG-compressed. All lack
Pixel Intensity Relationship/Sign, Presentation Intent Type, Modality
LUT/rescale, VOI LUT/Window Center/Width, and processing descriptions. The
stored values therefore cannot be classified as linear or logarithmic in
incident X-ray signal or reliably inverted to such a scale.

Before the shared 8-bit scaler, four exact DICOM default-`LINEAR` center/width
alternatives tested display-transform sensitivity (class A). Three
signal-dependent Poisson-like count conditions were retained only as generic
intensity perturbations (class B) while classified physically unsupported for
these values (class D). Finite 3x3, 5x5, and 9x9 Gaussian kernels tested a
generic blur/spatial-resolution proxy (class C), not a scanner-specific model.
The primary descriptive endpoint was

$$
\mathrm{DSI}=1-\frac{\mathrm{performance}_{shifted}}
{\mathrm{performance}_{clean}},
$$

using mAP@0.5:0.95. DSI was treated only as a descriptive
performance-retention/domain-sensitivity index, not an estimator of inter-site
transportability. Per-image normalized MAE and RMSE were also computed before
and after min-max scaling to quantify attenuation or amplification by canonical
preprocessing. These are internal synthetic sensitivities, not recovered vendor
presets, a dose/quantum-noise or validated low-dose simulation, measured scanner
transfer functions, or clinical robustness.

### 3.9 Grad-CAM localization and sanity checks

The primary explanation analysis used ordinary ReLU Grad-CAM at matched
stride-16, 40x40 pre-neck backbone tensors: ResNet-50
`backbone.body.layer3` and YOLO11s `model.6`. For each ground-truth box, the
target was the foreground probability of the low-threshold candidate with
highest IoU. Operating-point false negatives used an explicitly labeled,
annotation-guided proxy candidate. CAM energy within the ground-truth rectangle
and pointing-game accuracy quantified spatial agreement; box-area fraction was
a no-localization reference. These localization values are descriptive; no
inferential analysis tests their difference from the area reference.

The control analysis selected a nested, stratified 50-image/41-patient subset.
Each trained model's highest-score candidate defined a fixed reference region.
Cascading model-parameter randomization started with detector output heads and
cumulatively added neck and progressively earlier backbone groups in six
detector-specific stages. Each stage used a fresh in-memory model copy, seeded
Xavier-normal weights, zero biases, and preserved other buffers; the sixth
stage was a full model-parameter randomization control. The separate
input-pixel randomization control permuted RGB pixel vectors over spatial
positions while preserving their multiset and supplying identical shuffled
pixels to both trained detectors. Both controls used a fixed-region,
pre-activation foreground target to avoid requiring a fully randomized
proposal generator to emit valid post-NMS geometry. This target differs from
the primary Grad-CAM estimand. Maps were bilinearly reduced to 40x40,
independently min-max normalized, and compared with Pearson correlation,
tie-aware Spearman rank correlation, and Gaussian-window SSIM
[@wang2004ssim]. Non-finite or
degenerate pairs were excluded from every metric and counted, not imputed. The
on-disk checkpoint hashes were unchanged. The canonical Adebayo
training-label data-randomization test was not performed because randomized-
annotation retraining and fit verification were outside scope.

### 3.10 Inferential targets and patient-cluster protocol

The primary **training-procedure estimand** included internal-testing patient sampling
and stochastic retraining variability. Clean pointwise 95% bootstrap intervals
[@efron1993bootstrap] used
2,000 two-stage draws: patient groups were sampled with replacement, every exam
from a selected patient moved together, and trained runs were sampled
independently within detector. Same-number seeds were not treated as matched
blocks because the PyTorch and Ultralytics loaders, batch structures,
initialization paths, RNG-consumption sequences, and stopping trajectories were
not coupled. Precision, recall, F1, and AP used five runs per detector;
conditional IoU and Dice used five defined Faster R-CNN runs and four defined
YOLO11s runs. Dataset-level AP and all other nonlinear metrics were
reconstructed from the sampled predictions in every draw.

The secondary **checkpoint-conditional estimand** held the observed checkpoints
fixed. Two-sided permutation tests used 5,000 patient-group detector-label
swaps with a plus-one correction [@phipson2010permutation]; their p-values were
Holm-adjusted [@holm1979simple] across the seven clean endpoints and are labeled
conditional on observed checkpoints.
The unstandardized training-procedure difference, Faster R-CNN minus YOLO11s,
was the effect estimate. No seed-aware p-value was introduced. For the 35 corruption
conditions, raw performance and clean-relative retention were tested
separately, with Holm correction within each metric/estimand family. The same
patient draw was applied jointly to clean and corrupted evidence before a
retention ratio was calculated. Former image-level analyses are audit-only
archives. McNemar's test was omitted because multiple targets and false
positives do not reduce to one independent binary result per image without
discarding the detection structure.

### 3.11 Retrospective hypothesis and reporting status

H1--H5 were recorded after most experimental artifacts had been frozen, and H6
was added as a later descriptive calibration question. All six are therefore
retrospective, result-linked traceability devices rather than preregistered or
confirmatory hypotheses. Their operational endpoints, split and run scopes,
source artifacts, multiplicity boundaries, and limitations were mapped before
this manuscript rewrite.

## 4. Results

Sections 4.1--4.8 report absolute and descriptive secondary results with their
run, checkpoint, and image scopes. Section 4.9 is the inferential synthesis and
keeps the primary training-procedure intervals separate from the secondary
checkpoint-conditional p-values.

### 4.1 Clean internal-testing performance (n=5 per detector; conditional localization n=5/n=4)

The unified evaluator processed 750 internal-testing images and 268 reference boxes for
each of 10 frozen checkpoints. At the original score threshold of 0.25, Faster
R-CNN had higher recall, F1, and both AP endpoints, whereas YOLO11s had higher
precision and slightly higher conditional localization among successfully
matched detections. YOLO11s conditional IoU and Dice used only four defined
seeds.

| Endpoint | Faster R-CNN, mean +/- SD | YOLO11s, mean +/- SD |
|---|---:|---:|
| Precision at 0.25 | 0.1959 +/- 0.0552 (n=5) | **0.2983 +/- 0.1691 (n=5)** |
| Recall at 0.25 | **0.5799 +/- 0.0911 (n=5)** | 0.0955 +/- 0.0607 (n=5) |
| F1 at 0.25 | **0.2845 +/- 0.0528 (n=5)** | 0.1427 +/- 0.0868 (n=5) |
| Conditional matched-box IoU | 0.6749 +/- 0.0065 (n=5) | **0.6985 +/- 0.0157 (n=4)** |
| Conditional matched-box Dice | 0.8010 +/- 0.0049 (n=5) | **0.8181 +/- 0.0111 (n=4)** |
| mAP@0.5 | **0.3042 +/- 0.0189 (n=5)** | 0.1626 +/- 0.0162 (n=5) |
| mAP@0.5:0.95 | **0.0995 +/- 0.0067 (n=5)** | 0.0542 +/- 0.0060 (n=5) |

YOLO11s seed 271 was a critical all-attempt result. Its training losses
decreased and validation mAP converged normally, while internal-testing AP@0.5 and
AP@0.5:0.95 were 0.15872 and 0.05558, within the other YOLO seed range. Yet its
maximum internal-testing confidence was 0.0412735. It therefore emitted no detection at
0.25 and contributed observed zeros to precision, recall, and F1; matched-box
IoU and Dice were mathematically undefined. This was operational confidence-
score degeneracy, not classic loss/head collapse or a failure of otherwise
high-scoring boxes to meet the IoU rule. Retaining it prevented outcome-based
seed replacement.

The raincloud summary in Figure 1 visualizes the full clean seed distributions
and labels the actual finite n in each panel. The n=3 historical threshold data
are not mixed into it.

![Figure 1. Five-attempt clean predictive and compute distributions. Conditional YOLO11s IoU and Dice use four finite seeds; every other endpoint uses five seeds per detector.](../results/figures/raincloud_metrics.png)

### 4.2 Operating-regime sensitivity (internal-testing n=5; threshold selection n=3)

The n=5 all-attempt threshold sensitivity showed why the score-0.25 comparison
cannot be read as a general YOLO precision advantage. At 0.25, Faster R-CNN
precision/recall/F1 was 0.1959/0.5799/0.2845 and YOLO11s was
0.2983/0.0955/0.1427. At the lower grid boundary of 0.01, YOLO11s mean recall
rose to 0.2925 and its best observed F1 to 0.2657. Faster R-CNN's exploratory
peak mean F1 was 0.3562 at 0.63. These internal-testing optima were descriptive and did not
select either final threshold.

On the five-run official AP@0.5 curve, Faster R-CNN had higher mean
interpolated precision at 97 of 101 recall positions, with four ties and no
YOLO11s-higher position. At AP@0.5:0.95, Faster R-CNN was higher at 96
positions, YOLO11s at one near-zero-recall position, and four were tied. Mean
AP@0.5 was 0.3042 +/- 0.0189 versus 0.1626 +/- 0.0162; AP@0.5:0.95 was
0.0995 +/- 0.0067 versus 0.0542 +/- 0.0060. Thus the shared-threshold precision
difference reflected score scale and selectivity. COCO AP ranking, which does
not depend on the selected single operating threshold but remains conditional
on the retained prediction floor and common evaluation/post-processing
protocol, still favored Faster R-CNN (Figure 2). Seed 271 contributed its full
nonzero AP curve and was not filtered for its zero detections at 0.25.

![Figure 2a. Five-run all-attempt sensitivity of official precision-recall curves. Lines and bands are equal-run mean +/- sample SD; the original n=3 figure is retained as provenance.](../results/figures/precision_recall_curves_n5_sensitivity.png)

![Figure 2b. Five-run exploratory F1-versus-threshold sensitivity. Seed 271 is included exactly as observed; the internal-testing sweep is descriptive evidence, not threshold selection.](../results/figures/f1_vs_threshold_n5_sensitivity.png)

Validation selection remains n=3 and chose 0.69 for Faster R-CNN and 0.05 for
YOLO11s. Applying those unchanged thresholds to all five internal-testing bundles gave
precision/recall/F1 0.3624 +/- 0.0581, 0.3507 +/- 0.0463, and
0.3511 +/- 0.0184 for Faster R-CNN, versus 0.2524 +/- 0.1418,
0.1948 +/- 0.1110, and 0.2192 +/- 0.1233 for YOLO11s. Seed 271 contributes
defined zeros with no detection at 0.05. These are n=5 internal-testing sensitivities of an
n=3-selected rule, not thresholds reselected after internal-testing outcome inspection.

Within the evaluated 0.01--0.99 score sweep, five-run FROC sensitivity was
higher for Faster R-CNN at each predeclared budget: 0.2672 versus 0.1761 at
0.125 FP/image, 0.3642 versus 0.2455 at 0.25, 0.4836 versus 0.2925 at 0.5,
0.5970 versus 0.2925 at 1, and 0.6873 versus 0.2925 at 2 (Figure 3). YOLO11s
plateaued from the 0.5 budget because every run's best available point there
was the 0.01 lower sweep boundary. Seed 271 itself retained sensitivity 0.1493
at 0.01 and zero at 0.04 and above. The plateau is a grid boundary, not a
global asymptote.

![Figure 3. Five-run all-attempt FROC sensitivity within the evaluated 0.01--0.99 score sweep, with non-interpolated per-run budget summaries. The original n=3 figure remains unchanged as provenance.](../results/figures/froc_curves_n5_sensitivity.png)

The dedicated n=3-versus-n=5 audit reports unfavorable and favorable changes
under one prespecified directional-margin rule:

| Evidence statement | n=3 to n=5 classification |
|---|---|
| YOLO11s precision margin at shared score 0.25 | Weakened |
| Faster R-CNN recall margin at shared score 0.25 | Weakened |
| Faster R-CNN F1 margin at shared score 0.25 | Strengthened |
| Faster R-CNN mean AP@0.5 and AP@0.5:0.95 gaps | Weakened (both) |
| Faster R-CNN official-curve position lead | Strengthened at AP@0.5; unchanged at AP@0.5:0.95 |
| Faster R-CNN FROC sensitivity gap | Strengthened at all five budgets |
| Faster R-CNN precision/recall/F1 at frozen detector thresholds | Strengthened (all three) |
| Neither detector dominates the four Pareto panels | Unchanged (all four) |
| Reversed conclusions | None |

The complete 19-row table, including exact old/new margins and seed-271 roles,
is `results/tables/operating_regime_n3_vs_n5_conclusions.csv`.

### 4.3 Recall-preference and hypothetical-loss sensitivity (validation n=3)

When the lower pointwise bootstrap bound of $F_\beta$ was optimized on
validation, Faster R-CNN shifted monotonically toward less selective thresholds
as recall preference increased: 0.69, 0.33, 0.12, and 0.03 for
beta 1, 3, 5, and 10. Corresponding validation recall rose from 0.4404 to
0.6534, 0.7413, and 0.8207 while precision fell from 0.4164 to 0.1958, 0.1247,
and 0.0753.

YOLO11s selected 0.02 at beta 1, with precision 0.3492 and recall 0.3670, and
reached the 0.01 lower boundary at beta 3, 5, and 10, with precision 0.3025 and
recall 0.3971. Those three rows are best observed values within the grid, not
global optima. The beta settings are preference weights rather than empirical
clinical-harm measurements, the intervals are pointwise, and no selected F-beta threshold
was applied to internal testing. Per D-006, these results did not replace the primary
0.69/0.05 thresholds.

The 0.01-near-optimal plateaus were 0.64--0.70, 0.27--0.39, 0.07--0.16, and
0.03--0.04 for Faster R-CNN beta 1, 3, 5, and 10. Its bootstrap selected-tau
95% intervals were 0.63--0.75, 0.13--0.51, 0.04--0.29, and 0.01--0.09,
showing substantial selection instability at beta 3 and 5. YOLO11s beta 1 had
a 0.01--0.05 plateau and bootstrap interval 0.01--0.10; beta 3--10 selected
0.01 in 99.7%, 100%, and 100% of draws. That concentration reflected the lower
grid boundary rather than a demonstrated interior optimum.

The separate linear loss selected Faster R-CNN thresholds 0.87, 0.61, 0.23,
and 0.04 for assumed $r=1,9,25,100$; YOLO11s selected 0.35, 0.01, 0.01, and
0.01. The beta-1 and r-1 thresholds therefore differed sharply for both
detectors, directly demonstrating that F-beta optimization was not equivalent
to the declared linear error-loss minimization. These assumed ratios were not
clinical valuations, and none of the loss-selected thresholds was applied to
internal testing.

### 4.4 Detection calibration and reliability (descriptive n=5 per detector)

Across five seeds, mean D-ECE (calibration error; lower is better under the
fixed protocol) was 0.0320 +/- 0.0058 for Faster R-CNN and
0.0990 +/- 0.0232 for YOLO11s. The seed ranges did not overlap: 0.0266--0.0403
and 0.0714--0.1313. These are equal-run descriptive summaries; no D-ECE
inferential comparison was performed. The largest observed YOLO11s value was
0.13130 for seed 271. All 962 detections from that run at the 0.001 floor were
retained, with mean confidence 0.00538 and matched fraction 0.15073. This was
an observed result, not a method-level expectation or special-case branch.

The five-dimensional grid was sparse: 68--354 of 3,125 possible cells were
occupied per run and 15--169 met the eight-detection minimum. Supported-cell
detection fractions were 0.879--0.983 for Faster R-CNN and 0.543--0.906 for
YOLO11s. Across the predeclared 3/5/7-bin by 1/4/8/16-minimum grid, descriptive
mean D-ECE ranged from 0.0131 to 0.0531 and from 0.0500 to 0.1551,
respectively; Faster R-CNN remained lower at all 12 common settings, while
absolute values changed with support. Raising the score floor to 0.005 retained
only 46.6% and 53.1% of the two detectors' baseline emitted populations on
average. At 0.05 one YOLO11s run emitted no detection and its D-ECE was
undefined, not zero. Thus the floor materially defined the evaluated
population.

Figure 4 is a confidence-only marginal reliability summary. It does not
visualize the full class/location/scale-conditioned D-ECE. Calibration remained
conditional on emitted detections and was evaluated, not fitted, on the
internal-testing set; missed targets and clinical/exam risk were outside the
population.

![Figure 4. Confidence-only marginal reliability diagrams for all five runs per detector. This one-dimensional visual is not a visualization of all five D-ECE dimensions.](../results/figures/reliability_diagrams_confidence_marginal_v2.png)

### 4.5 Compute and Pareto sensitivity (n=5 per detector)

YOLO11s was the computationally lighter pipeline on the measured laptop. Mean
throughput was 60.29 +/- 12.62 FPS versus 20.28 +/- 5.62 for Faster R-CNN; mean
latency was 17.23 +/- 3.83 versus 53.93 +/- 21.15 ms/image. YOLO11s had 9.43
million parameters versus 43.26 million, 21.42 versus 450.76 estimated
registered-operation GFLOPs/image, 1,148.16 versus 1,556.89 MiB peak allocated
training memory, and 1,544.75 versus 6,661.01 seconds mean training time across
five seeds.

The n=5 Pareto sensitivity preserved the opposing directions. Every Faster
R-CNN run had higher AP while every YOLO11s run had higher throughput; the same
trade-off appeared for AP versus parameters and frozen-threshold recall versus
latency or registered operations. Recall was 0.3507 +/- 0.0463 versus
0.1948 +/- 0.1110; YOLO11s seed 271 supplied the observed zero at threshold
0.05 rather than being filtered. Under the conservative all-runs dominance
rule, neither detector strictly dominated in any panel (Figure 5), unchanged
from n=3. This is an accuracy-efficiency trade-off on one hardware/software
stack, not a universal property of model families.

![Figure 5. Five-run all-attempt accuracy-efficiency Pareto sensitivity. Each point joins same-run AP or internal-testing recall to same-run hardware metrics; recall uses thresholds selected from the original n=3 validation runs. The historical n=3 figure is retained separately.](../results/figures/pareto_frontier_n5_sensitivity.png)

### 4.6 Digital common-corruption sensitivity (single checkpoint per detector; 300 images)

On the 300-image clean robustness sample, mAP@0.5:0.95 was 0.147802 for Faster
R-CNN and 0.076295 for YOLO11s. Averaged equally across all 35 corrupted
conditions, raw mAP was 0.112898 and 0.054099, while clean-relative retention
was 0.763846 and 0.709083. These corresponded to mean degradation of 23.62%
and 29.09%. Faster R-CNN had higher raw mAP in all clean and corrupted matched
conditions, conditional on the two seed-17 checkpoints and the fixed sample.

| Severity-5 retention | Faster R-CNN | YOLO11s |
|---|---:|---:|
| Darker | **0.8674** | 0.1645 |
| Brighter | 0.6706 | **0.6882** |
| Gaussian noise | **0.4643** | 0.3692 |
| Salt and pepper | **0.2025** | 0.0570 |
| Gaussian blur | 0.6936 | **0.6949** |
| Motion blur | **0.6788** | 0.5780 |
| JPEG quality 20 | 0.6825 | **0.7662** |

Salt-and-pepper noise was the most damaging tested condition for both models.
The reversal of relative ordering across corruptions supported a type-specific
rather than uniform degradation model. The strongest inferential retention
contrast was darkness severity 5: 0.8674 versus 0.1645, difference 0.7029
(95% CI 0.2734--0.8158), Holm p=0.0070. The corresponding raw mAP difference
was 0.1156 (0.0609--0.1689) but did not survive the 35-condition Holm family
(p=0.0770). Pointwise separation and multiplicity-controlled evidence therefore
led to different conclusions.

![Figure 6. Seed-17 clean-relative mAP@0.5:0.95 under seven post-conversion digital corruptions and five severities.](../results/figures/robustness_map_50_95_relative.png)

### 4.7 Radiography-motivated synthetic acquisition/display sensitivity (single checkpoint per detector; 300 images)

The signal-dependent Poisson-like and Gaussian series produced ordered mAP
degradation. Under the strongest Poisson-like condition (relative count budget
0.125), Faster R-CNN shifted mAP was 0.101314 with DSI 0.314529; YOLO11s shifted
mAP was 0.042981 with DSI 0.436641. This condition is not a dose or quantum-noise
proxy. Under 9x9, sigma-2.0 generic blur, the corresponding DSI values were
0.248532 and 0.282869. YOLO11s had larger descriptive DSI at every tested
Poisson-like level and blur kernel.

The VOI results were smaller and mixed. Center shifts produced Faster R-CNN
DSI 0.0431 and 0.0862 versus YOLO11s 0.1177 and 0.0089. A 0.75-width window
gave DSI 0.0716 and -0.0132; the 1.25-width window was nearly neutral
(-0.0001 and 0.0009). The image-space audit showed why: min-max scaling reduced
the median NMAE of the two center shifts by 48.2% and 40.7%, and almost
completely cancelled the widened window (median post-scaling NMAE zero; 264/300
images exactly identical). It did not materially cancel the narrow window or
Poisson-like noise and re-stretched the normalized Gaussian-blur differences.
No display result establishes scanner-setting invariance, and DSI does not
estimate site transportability. The full seven-metric, 20-row table remains in
[Supplementary Section S5](../docs/SUPPLEMENTARY.md#s5-complete-digital-corruption-and-acquisition-shift-grids).

### 4.8 Explainability and XAI controls (single checkpoint; 300-image localization and 50-image sanity scopes)

Grad-CAM localization was weak for both detectors. Faster R-CNN had 110 valid
maps from 111 boxes, mean energy-in-box 0.0869 versus a 0.0713 box-area
reference, and pointing accuracy 0.1091. YOLO11s had 111 valid maps, mean
energy 0.0975 versus area reference 0.0718, and pointing accuracy 0.1261. On
110 jointly valid targets, YOLO had higher energy for 76 and Faster R-CNN for
34; mean Faster-minus-YOLO energy was -0.0091. The modest YOLO advantage did
not imply cleaner attention. Faster maps were generally more clustered but
often centered on the mediastinum, shoulders, chest wall, borders, markers, or
devices; YOLO maps were more diffuse and punctate across similarly irrelevant
or unannotated regions. False-negative proxy maps were conditional analyses of
latent candidates, not explanations of emitted detections. The absolute
energy-over-area lifts were only 0.0156 and 0.0257 and were not tested
inferentially; neither constitutes strong localization.

At the final full model-parameter stage, Faster R-CNN mean
Pearson/Spearman/SSIM values were 0.0025/0.0287/0.0381 (50 valid pairs), and
YOLO11s values were 0.0242/0.0795/0.0476 (46 pairs). Intermediate cumulative
stages were non-monotonic and remain descriptive. These low final-stage
similarities support parameter sensitivity under the stated audit but do not
prove anatomical correctness, causal faithfulness, or clinical reasoning. For
the input-pixel control, Faster R-CNN means were -0.0206/-0.0188/0.2468 (43
pairs) and YOLO11s means were -0.0021/0.0133/0.0067 (50 pairs). The non-zero
Faster R-CNN SSIM illustrates why one correlation was insufficient. This
severe perturbation control does not test dependence on the training
data-label relationship; randomized-label retraining was not performed
(Figure 7).

![Figure 7. Nested 50-image Grad-CAM input-pixel control and six-stage cascading model-parameter randomization panel.](../results/figures/gradcam_sanity_v2_panel.png)

### 4.9 Estimand-separated statistical synthesis (n=5/5 or n=5/4 runs)

The table reports the primary training-procedure differences and separately the
secondary checkpoint-conditional p-values. Differences are Faster R-CNN minus
YOLO11s; the corresponding absolute endpoint estimates are reported in Section
4.1. Every row contains 750 images from 323 patient clusters. `Runs A/B`
gives the eligible Faster R-CNN/YOLO11s trained-run counts.

| Endpoint | Runs A/B | Conditioning | Difference (95% training-procedure CI) | Seed 271 | Holm p, conditional on observed checkpoints |
|---|---:|---|---:|---|---:|
| Precision at 0.25 | 5/5 | Unconditional | -0.1024 (-0.2423, 0.0553) | Both runs; YOLO contributes zero | **0.0020** |
| Recall at 0.25 | 5/5 | Unconditional | 0.4843 (0.3830, 0.5830) | Both runs; YOLO contributes zero | **0.0014** |
| F1 at 0.25 | 5/5 | Unconditional | 0.1419 (0.0559, 0.2290) | Both runs; YOLO contributes zero | **0.0014** |
| Conditional IoU | 5/4 | Conditional on a matched detection | -0.0236 (-0.0585, 0.0089) | Faster defined; YOLO undefined | 0.1064 |
| Conditional Dice | 5/4 | Conditional on a matched detection | -0.0171 (-0.0420, 0.0066) | Faster defined; YOLO undefined | 0.1064 |
| mAP@0.5 | 5/5 | Unconditional | 0.1416 (0.1013, 0.1845) | Both ranked bundles contribute | **0.0208** |
| mAP@0.5:0.95 | 5/5 | Unconditional | 0.0453 (0.0313, 0.0599) | Both ranked bundles contribute | **0.0342** |

Under the primary estimand, the recall, F1, and both AP intervals remained
wholly above zero. Fixed-threshold precision did not: its interval crossed
zero, so it did not support a training-procedure difference. The
small checkpoint-conditional precision p-value instead says that the observed
checkpoints favored YOLO11s at the common numerical cutoff across patient
clusters. The CI and p-value differ because the former also resamples trained
runs; neither arithmetic result is wrong, and they do not test the same target.
Conditional localization excluded missed annotations and remained
inconclusive.

Independent detector-run resampling changed interval endpoints but did not
change zero exclusion for any endpoint relative to the historical paired-seed
sensitivity. Seed-label deletion was descriptive only: omitting label 271
changed the precision difference from -0.1024 to -0.1892, F1 from 0.1419 to
0.0938, and mAP@0.5:0.95 from 0.04533 to 0.04504. Seed 271 was not excluded
from the corrected analysis wherever its endpoint was defined.

The retrospective checks summarized higher Faster R-CNN coverage/AP; a
shared-threshold operating mismatch; no strict accuracy-compute Pareto
dominance; corruption-specific relative degradation; weak localization with
parameter-sensitive Grad-CAM and response to severe input-pixel perturbation;
and descriptively lower Faster R-CNN detection-specific calibration error
(D-ECE). The calibration audit did not predeclare that a named run should be an
outlier. These are structured summaries of the same frozen evidence, not
independent confirmatory tests or preregistered claims.

## 5. Discussion

### 5.1 The main result is an operating-regime mismatch within a controlled comparison

The most informative finding is not that one detector produced a larger mAP
value. It is that a shared numerical threshold failed to create a shared
operating regime. At score 0.25, YOLO11s appeared more precise because it was
far more selective, while Faster R-CNN retained much more of the target set.
The five-run official precision-recall analysis and FROC sensitivity within the
evaluated 0.01--0.99 score sweep showed that this was not a hidden YOLO
high-precision frontier: Faster R-CNN retained higher precision at almost every
official recall position and higher observed sensitivity at every reported
false-positive budget within that sweep. Detector-specific thresholds selected
on validation also differed sharply, 0.69 versus 0.05.

These measurements are related but not interchangeable. AP summarizes ranking
over predictions retained under the common evaluation and post-processing
protocol; it does not depend on the selected single operating threshold. Raw
score scale determines where a numerical cutoff falls. Fixed-threshold
precision and recall describe the resulting operating point. D-ECE asks
whether emitted scores agree with empirical correctness conditional on class,
location, and scale under a specified binning and support rule. A favorable
result for one object does not imply a favorable result for another.

The observed YOLO11s seed-271 run makes that distinction concrete: AP remained
within the sibling-seed range even as all scores compressed below 0.042 and
mean confidence was below the matched fraction. The run is evidence of
score-scale instability under this disclosed recipe, not a failed attempt to
hide or replace. The sparse-cell and floor sensitivity also showed that
absolute D-ECE depends materially on histogram support and which detections
enter the population. The evidence argues for reporting curves,
detector-specific validation selection, occupancy, sensitivity, and all
attempted runs rather than relying on one shared threshold or one D-ECE value.

### 5.2 Accuracy and efficiency remain opposed

Faster R-CNN provided the stronger coverage and ranking evidence. Its recall,
F1, and both AP training-procedure intervals remained wholly above zero, and
its FROC sensitivity was higher across the measured budgets within the
evaluated score sweep. Under the documented detector-specific profiling
procedure on the measured laptop, YOLO11s provided the stronger measured
implementation-efficiency evidence: approximately three times the throughput,
about one fifth the parameters, one twenty-first the estimated registered
operations, lower training memory, and shorter training time. The Pareto result
formalized this opposition; neither pipeline dominated when an accuracy
endpoint and a compute endpoint were optimized together.

Conditional matched-box IoU and Dice do not create a third conclusion in favor
of YOLO11s. They describe localization only after a true-positive match,
exclude the much larger missed-target burden, use five defined Faster R-CNN
runs and four defined YOLO11s runs for the primary interval, and were
inconclusive. They are useful diagnostics, but should not be compared directly
with unconditional coverage or AP.

### 5.3 Preference and hypothetical-loss sweeps do not establish clinical utility

The recall-weighted F-beta validation sweep illustrated how preference settings
can dominate a threshold. Increasing beta moved Faster R-CNN toward
progressively lower thresholds and higher recall. YOLO11s hit the lower grid
boundary by beta 3, exposing limited room within the declared sweep. The wide
Faster R-CNN bootstrap selected-tau distributions at beta 3 and 5 further show
that a single grid argmax can be unstable. Beta squared is a relative recall
weight in the harmonic mean, not a measured clinical-harm ratio.

The separate linear loss sweep made the distinction operational: assumed r-1
loss selected 0.87/0.35 rather than the beta-1 F-beta choices 0.69/0.02. Its
ratios assign hypothetical penalties to missed target boxes and false-positive
detections; they omit patient outcomes, actions, and downstream utility.
Because both analyses are validation-only, use pointwise intervals, and rely on
assumed preferences or penalties, they are sensitivity evidence only. D-006
appropriately preserves the equal-weight validation-F1 thresholds as the
primary historical operating points.

### 5.4 Robustness is condition-specific and remains internal

The two robustness studies answer different questions. Digital corruptions
showed that performance loss depended on corruption family and severity. Faster
R-CNN had the higher raw mAP throughout the grid and better average retention,
but relative ordering reversed for some brightness, blur, and JPEG conditions.
After patient clustering and multiplicity correction, the severe-darkness
retention advantage remained supported while the raw mAP difference did not.
This is precisely why raw and relative estimands, pointwise intervals, and
family-adjusted tests must be kept distinct.

The stored-array experiment adds radiography motivation without external
validity or a common level of physical support. DICOM `LINEAR` settings test
synthetic display transforms; Gaussian blur is only a generic
spatial-resolution proxy; and the count perturbation is Poisson-like intensity
noise, not a dose/quantum-noise simulation. Both checkpoints degraded as the
latter two series strengthened, with larger descriptive DSI for YOLO11s at each
level. Display-transform behavior was smaller and mixed because the source
objects lacked native VOI settings and per-image min-max scaling partly or
almost completely cancelled some changes. Neither experiment introduced a new
scanner, protocol, institution, or patient population. The correct claim is
sensitivity to declared transforms, not clinical robustness or site
transportability.

### 5.5 Parameter sensitivity does not rescue weak localization

Low similarity after cascading to full model-parameter randomization addresses
the narrow concern that the maps are invariant to model parameters. The
input-pixel control separately shows response when image spatial structure is
destroyed; it is not evidence of dependence on the learned data-label
relationship. Neither result makes the maps causal explanations. Energy-in-box
exceeded the box-area reference only modestly, pointing accuracy was near 0.1, and both
models frequently highlighted extra-box anatomy, markers, devices, and borders.
The maps therefore identify suspicious associations and failure patterns, not
clinically validated reasoning.

### 5.6 Scope-correct synthesis

The study supports a measured trade-off, not a recommendation for a clinical
scenario. Within this internal comparison, Faster R-CNN had higher coverage,
ranking accuracy, and FROC sensitivity within the evaluated score sweep, while
YOLO11s had lower latency, parameter count, registered-operation estimates,
training memory, and training time on the measured hardware/software stack.
The Pareto analysis preserves both directions and therefore does not identify
one dominant pipeline.

No result establishes suitability for screening, prioritization, diagnosis,
rule-out, treatment guidance, or point-of-care use. Such judgments would
require an intended-use specification, representative prevalence, calibrated
exam-level risk, external and prospective evaluation, subgroup assessment,
workflow evidence, and safety analysis that are absent here.

The claims apply only to these two disclosed pipelines. Decisions D-002 and
D-003 explicitly descoped a second, equal-opportunity architecture-specific
tuning track. YOLO's full native augmentation recipe and a correspondingly
tuned Faster R-CNN recipe were not compared. The present evidence therefore
cannot support "two-stage detectors are better," "all Faster R-CNN models are
more accurate," "all YOLO models are faster," or any other detector-family
law. It supports the narrower statement that these implementations occupied
different accuracy, efficiency, score-scale, fixed-threshold, and calibration
regimes under the stated controls.

## 6. Limitations

This is a controlled course-project comparison, not a clinical validation
study. It must not be used to diagnose pneumonia, exclude disease, prioritize
real patients, or guide care.

**Dataset, population, and target.** The experiment uses one non-specific
`Lung Opacity` category from one historical, single-institution, adult-heavy
RSNA/NIH source. The 5,000-study subset is stratified and enriched rather than
prevalence representative. Precision and false-positive behavior are therefore
benchmark characteristics, not deployment predictive values. Results may not
transport to pediatric patients, portable-care
settings, contemporary equipment, other institutions, modalities, or
multi-class tasks. No external testing dataset or demographic subgroup/fairness
analysis was available. Source accrual dates, full acquisition-device and
exposure details, participant demographics, and several reference-standard
details are absent. No prospective evaluation was performed.

**Reference standard and preprocessing.** Boxes are coarse rectangles rather
than pixel-accurate opacity masks; reader disagreement and ambiguous boundaries
remain. The technical annotation audit does not constitute re-reading by an
independent radiologist. Deterministic DICOM inversion and per-image min-max
scaling do not reproduce vendor VOI/display processing, and resizing to 640
pixels can remove small-opacity detail differently across pipelines.

**Sampling and seed scope.** The hardware-scoped cohort excludes 21,684 source
studies from model training/evaluation. Five clean training attempts per
detector remain a coarse sample of seed variation. The principal internal-testing
threshold, PR, FROC, and Pareto displays are five-run all-attempt sensitivities;
their original n=3 artifacts remain unchanged as provenance. Threshold
selection itself still uses only the original three validation runs, so the
n=5 fixed-threshold and recall-Pareto results are not n=5-selected operating
points.
Digital robustness, acquisition shifts, and primary Grad-CAM use one seed-17
checkpoint per detector and 300 images from 183 patients with 111 boxes. XAI
sanity uses 50 images from 41 patients. None of these secondary analyses
characterizes across-seed uncertainty.

**Pipeline attribution.** Disabling Ultralytics stochastic augmentations
controls the data distribution but may understate YOLO11s performance under its
conventional recipe. Precision type, batch behavior, BatchNorm handling,
learning rate, warmup, and scheduling differ because the more closely matched
YOLO diagnostics collapsed. Detector-intrinsic losses, assignment, proposal
handling, and NMS also differ. Only these two working pipelines were studied;
this is not a causal architecture-family effect or a survey of detector
variants. YOLO11s seed 271's operational
confidence-score degeneracy is retained as a recipe-level instability and not
silently removed.

**Metrics, thresholds, and calibration.** Score-0.25 precision, recall, and F1
are protocol-sensitivity endpoints. The primary validation-selected thresholds
use three seeds and equal-weight F1, which does not encode an elicited
clinical-harm function. Applying those thresholds to five internal-testing runs cannot add
missing validation evidence or justify reselection. The n=3-versus-n=5 margin
classifications are descriptive influence summaries rather than inferential
tests; several shared-threshold and AP gaps weakened even though no direction
reversed. The
F-beta extension uses beta 1, 3, 5, and 10 as recall-preference parameters, not
clinical-harm valuations. Its near-optimal plateaus and bootstrap argmax frequencies are
descriptive; its intervals are pointwise and its YOLO optima reach the lower
grid boundary. The separate linear loss assumes FN:FP penalties 1, 9, 25, and
100 without outcome-based valuation. None of these thresholds was tested
downstream. Conditional IoU and Dice exclude misses and
have asymmetric descriptive n. D-ECE is evaluated on the same internal-testing
bundles, conditions only on emitted predictions at the 0.001 floor, depends on
the chosen IoU/binning/minimum-cell protocol, and has only 68--354 occupied of
3,125 possible cells per run. Its comparison is descriptive; the reported
support and predeclared bin/floor sensitivities show that the absolute estimate
is population- and sparsity-dependent. D-ECE neither fits a calibrator nor
measures a missed ground-truth object (which has no emitted confidence),
exam-level risk, clinical-risk calibration, or patient harm.

**Decision analysis.** The historical exploratory calculation defined action
by maximum detector confidence `>= tau` and reused the same raw `tau` in the
`tau/(1-tau)` false-positive weight. A raw detector score is not automatically
the threshold probability required by conventional DCA
[@vickers2006decisioncurve; @vickers2008decisioncurveextensions]. Batch 30
therefore classified the calculation as non-standard, removed it from the main
Results, and retained its exact arithmetic only as a relabeled supplementary
raw-score utility/sensitivity artifact. Probability-based salvage was not
forced because frozen validation predictions were available for only six of
ten retained runs; no calibrator was selected or fitted. The enriched
internal-testing subset (169/750 positives; 323 patient groups) is not a
deployment-prevalence sample. The preserved calculation supplies no conventional
net-benefit, clinical-utility, beneficial-range, or deployment-readiness
evidence. Detection-level D-ECE was not used as an exam-level probability and
is programmatically separate from the validation-frozen outcome-probability
calibration required by any future valid DCA.

**Robustness.** Digital severity indices are not physically calibrated or
comparable across corruption types, and repeated transformations of the same
300 images are not independent deployment cohorts. The stored-array sample is
workstation-converted, lossily compressed Secondary Capture despite recording
`Modality=CR`; it lacks intensity-relationship/sign, presentation intent,
modality/VOI transforms, calibrated exposure, and detector-response metadata.
The four DICOM windows are synthetic display sensitivities. Poisson-like noise
is physically unsupported as a dose/quantum-noise proxy and is not a validated
low-dose acquisition simulation. Gaussian blur is not a measured scanner
transfer function or reconstruction model. Per-image scaling partly cancels
center shifts, almost completely cancels the widened window, and can re-stretch
blur differences. DSI is descriptive and does not estimate transportability.
Neither experiment establishes scanner safety, population robustness, or
clinical robustness.

**Explainability.** Grad-CAM uses coarse 40x40 maps and explains one selected
score, not proposal generation, NMS, box regression, candidate selection, or a
complete causal decision. False-negative maps are annotation-guided proxy
analyses unavailable in deployment. Box-based metrics treat coarse rectangles
as opacity masks; the small descriptive energy-over-area lifts are not
inferential evidence of strong localization. The v2 control extension uses one
deterministic random draw per cumulative stage, a severe pixel permutation, and
a different fixed-region/pre-activation target. It excludes seven Faster R-CNN
input-control maps and four YOLO11s full-randomization maps. Low final-stage
Pearson, Spearman, and SSIM values establish only parameter sensitivity under
this audit. The pixel shuffle is an input perturbation, not Adebayo
training-label data randomization; that retraining experiment was not
performed. Neither control establishes medical validity or causal attention.

**Statistical uncertainty.** Patient-cluster resampling and label swaps correct
the identified within-patient independence error for observed exams, but do not
create a representative population. Primary clean intervals resample five
trained runs independently within detector; conditional localization has five
defined Faster R-CNN runs and four defined YOLO11s runs. This is still coarse
training-variability evidence. Bootstrap intervals are pointwise; secondary
permutation tests condition on the observed checkpoints; corruption inference
is single-checkpoint; and Holm correction does not make corruption conditions
independent cohorts or guarantee transportability. H1--H5 and the H6
calibration question were recorded retrospectively; they organize frozen
evidence but do not provide preregistered, confirmatory hypothesis tests.

**Compute and reproducibility.** Timing and registered-operation estimates
describe one RTX 4060 Laptop GPU, pinned environment, and implementation path.
The FLOP counter omits unsupported operations, and the two frameworks place
resizing differently relative to timing. These values are pipeline-runtime
measurements on the documented stack, not hardware-independent architecture
constants. At the Batch 36 audit, all ten exact best-checkpoint files remained
available locally and matched the Phase 5 hashes, but the binaries were
Git-ignored, had not been publicly released, and had no public download URL.
A clean checkout can verify and replay committed frozen-prediction evidence but
cannot reproduce exact inference without separately supplied licensed data and
matching checkpoint assets. Exact retraining is also not promised to be
bitwise identical: the recorded CUDA ROI Align backward path is
nondeterministic under the pinned Torchvision stack, and hardware, scheduling,
and floating-point reductions can alter training trajectories.

**Reporting and governance gaps.** The reporting crosswalk is an evidence
audit, not certification of CLAIM, TRIPOD+AI, or STARD-AI compliance. The
repository does not contain a local ethics/consent determination, registration,
funding statement, conflict-of-interest statement, patient/public involvement
statement, participant-flow diagram, source accrual dates, or external and
demographic subgroup evaluation. These gaps cannot be repaired by narrative
wording without new traceable evidence.

Clinical deployment would require prospective and external multi-site
validation, subgroup and fairness assessment, validation-fitted risk
calibration, human-factors and workflow testing, safety and cybersecurity
controls, monitoring, and the applicable medical-device regulatory process.
Those activities are outside this study. Mentioning them defines the boundary
of the claims; it is not evidence that either pipeline is ready to enter such a
process.

## 7. Conclusion

Under one patient-disjoint internal protocol, common canonical inputs,
disabled stochastic augmentation, validation-frozen decisions, and one
evaluator, Faster R-CNN and YOLO11s occupied different operating and resource
regimes. A shared score threshold did not align selectivity. The five-run
precision-recall analysis and, within the evaluated 0.01--0.99 score sweep,
FROC sensitivity favored Faster R-CNN. The primary training-procedure intervals
were wholly positive for Faster R-CNN minus YOLO11s recall, F1, and both AP
endpoints. The primary interval did not support a fixed-threshold precision
difference at the training-procedure level. YOLO11s was substantially smaller
and had higher measured
throughput under the documented detector-specific profiling procedure on the
stated laptop, while seed 271 exposed score compression without corresponding
AP collapse.

The analyses also show why AP, raw score scale, fixed-threshold behavior, and
detection-level calibration must be reported separately. Detection D-ECE was
descriptive and support-sensitive; it was not exam-level risk calibration.
Digital corruptions and radiography-motivated synthetic transformations were
internal stress tests, and Grad-CAM maps localized opacities weakly despite
parameter sensitivity. The invalid raw-score decision-curve interpretation was
removed rather than used as evidence.

The contribution is a controlled, multi-axis characterization of two disclosed
pipelines and an empirical demonstration that nominally identical raw
confidence thresholds can select very different operating regimes. It is not a
universal one-stage-versus-two-stage law, a comparison of all Faster R-CNN or
YOLO implementations, or evidence about clinical populations or use. Complete
seed-level tables, stress-test grids, archives, provenance, and reproduction
routes are maintained in the [Supplementary Materials
Index](../docs/SUPPLEMENTARY.md).

## 8. Declarations

The following fields are intentionally unresolved placeholders. They are not
statements of absence, approval, exemption, or applicability.

**Funding and support — AUTHOR ACTION REQUIRED:** Identify every funding or
support source, grant and recipient where applicable, and the funder's role, or
insert an author-confirmed journal-appropriate no-funding statement.

**Competing interests — AUTHOR ACTION REQUIRED:** Provide an author-by-author
declaration under the target journal's policy, including an explicit
author-confirmed none statement where appropriate.

**Ethics and data use — AUTHOR ACTION REQUIRED:** Insert the responsible
institution's determination for this retrospective secondary analysis,
including the body, determination type, identifier, and date where applicable;
separately confirm compliance with the RSNA/Kaggle and NIH data-use terms.

**Consent — AUTHOR ACTION REQUIRED:** State whether consent was required,
waived, or not applicable under the documented ethics determination and give
the responsible rationale required by the journal.

**Author contributions — AUTHOR ACTION REQUIRED:** Insert the final author list
and author-approved contribution statement, preferably using CRediT roles where
accepted.

**Data, code, and model availability — AUTHOR ACTION REQUIRED:** Provide the
actual public repository/archive identifier, release or commit, license,
source-data access instructions, artifact scope, and exact checkpoint-release
status at submission. Do not insert a speculative checkpoint URL.

**Patient and public involvement — AUTHOR ACTION REQUIRED:** Confirm whether
patients or members of the public were involved and insert the journal-required
statement; do not infer non-involvement from the repository.
