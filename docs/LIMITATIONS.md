# Limitations

This benchmark is a controlled course-project comparison, not a clinical
validation study. It evaluates detector behavior on a fixed retrospective
dataset and must not be used to diagnose pneumonia, exclude disease, prioritize
real patients, or guide care.

## Dataset, population, and target scope

The experiment uses one foreground category from one source: `Lung Opacity` in
the historical RSNA challenge cohort derived from the NIH archive. The images
come from one institution and an adult-heavy population under historical
acquisition practices. Results may not transport to pediatric patients,
portable-care settings, newer equipment, other institutions, other modalities,
or multi-class localization tasks.

The challenge cohort was enriched using existing labels and is not a
prevalence-representative clinical sample. Reported precision and false-positive
behavior are therefore benchmark operating characteristics, not deployment
predictive values. The target is a non-specific radiographic opacity rather
than microbiologically or clinically confirmed pneumonia; other conditions can
cause an opacity, and pneumonia can exist without a boxed focal opacity.

The official mapping supports NIH patient-level grouping, and the train,
validation, and test patient-key intersections are empty. This controls known
repeat-exam leakage, but the released metadata may not expose every encounter,
device, or acquisition relationship. The held-out sets also contain repeated
exams within a split. The resulting within-patient dependence was identified
and corrected in Batch 13: primary confidence intervals resample NIH patient
groups, and permutation swaps move every observed exam from one patient
together. The remaining limitation is the finite number of observed patient
groups, not unaddressed image-level clustering.

## Annotation and preprocessing scope

Bounding boxes are coarse rectangular approximations, not pixel-accurate opacity
masks. Reader disagreement, ambiguous boundaries, and non-uniform annotation
certainty remain even though the metadata audit found no malformed,
non-positive-area, off-image, or exact duplicate positive boxes. Coordinate
validation and the 12-image EDA establish technical consistency, not clinical
correctness or a second expert reading.

DICOM conversion uses deterministic `MONOCHROME1` inversion and per-image
min-max scaling to 8-bit PNG. It does not reproduce vendor-specific window/VOI
processing. Later resizing to 640 pixels can remove small-opacity detail and
can affect detector families differently.

## Compute and sampling scope

The target machine is an RTX 4060 Laptop GPU with 8 GB VRAM, 16 GB system RAM,
and an i7-13650HX. Those constraints motivated a fixed stratified 5,000-study
subset rather than all 26,684 labeled studies, 640-pixel inputs, small-model
selection, mixed precision, bounded batches, and early stopping. These choices
make the study reproducible on the stated laptop but increase sampling
uncertainty and narrow generalization to the full challenge cohort.

The primary development pipeline uses seed 17. The held-out headline comparison
adds seeds 42, 137, 271, and 314 for five predeclared full trainings per
detector. All five attempted seeds are retained for AP and fixed-threshold
precision, recall, and F1, including YOLO11s seed 271's observed zeros. Five
runs remain a coarse estimate of training-seed variation. The additional-seed
timing approvals reuse the accepted seed-17 measurements because only RNG and
artifact identity change; every full run still records its own training time
and peak memory.

Robustness and explainability remain scoped to the selected seed-17 checkpoint
of each detector and the same fixed 300-image sample. That sample was drawn by
proportional largest-remainder allocation over the three test strata and
contains 68 positive and 232 negative images, 111 boxes, and 183 distinct NIH
patient IDs. Grad-CAM quantification covers all 111 boxes; nine paired images
are shown qualitatively under a predeclared case rubric. Neither phase
characterizes across-seed variation, and the modest box and patient counts make
relative rankings sample-sensitive.

## Detector-comparison validity

All extra Ultralytics stochastic augmentations are disabled to match Faster
R-CNN's resize-only pipeline: mosaic, mixup, cutmix, copy-paste, HSV jitter,
flips, geometric transforms, erasing, auto-augmentation, and multi-scale
training are off. This removes a known data-distribution confound but may
understate YOLO11s performance under its conventional augmentation-rich recipe.
It does not make the architectures, losses, or optimization dynamics identical.

Several numerical-stability differences remain:

- Faster R-CNN uses physical batch 2 with two-step gradient accumulation,
  float16 AMP, learning rate 0.005, frozen pretrained BatchNorm running
  statistics, and a validation-plateau scheduler.
- YOLO11s uses physical/effective batch 4, bfloat16 forward/backward with
  float32 target assignment and loss, one-epoch warmup to learning rate 0.001,
  native BatchNorm updates, and a constant post-warmup learning rate.

The accepted YOLO exceptions were introduced before valid training because the
more closely matched float16, higher-learning-rate, or frozen-normalization
diagnostics reproducibly collapsed the one-class head. Consequently, the
comparison is between two working, disclosed pipelines under a shared budget,
not a causal estimate of architecture alone.

Those safeguards prevented the previously observed non-finite/all-zero-loss
head collapse, but they did not make confidence scale stable across seeds.
YOLO11s seed 271 converged normally, with decreasing losses and nonzero
validation AP, yet its maximum held-out score was only `0.0412735`. It emitted
no detection at the frozen score-0.25 operating point, so its precision,
recall, and F1 are valid zeros even though AP@0.5 (`0.1587217`) and
AP@0.5:0.95 (`0.0555799`) remain in the range of the other YOLO seeds. This is
an **operational confidence-score degeneracy**, not the classic head/loss
collapse and not a case in which thresholded detections merely missed the IoU
matching criterion. Replacing the seed after observing this outcome would hide
recipe-level instability, so it remains part of the all-attempt analysis.

The Faster R-CNN physical microbatch remains two even with accumulation;
accumulation does not create batch-four BatchNorm statistics. Windows memory
constraints also require non-persistent Faster R-CNN train/validation worker
pools and fewer YOLO workers. These are resource-management differences rather
than experimental endpoints.

## Metric and compute-measurement scope

The original Phase 5 precision, recall, and F1 table uses one shared score
threshold of 0.25 and matching IoU of 0.50. Batch 14 separately selects
detector-specific thresholds on validation by maximum mean F1 (0.69 for Faster
R-CNN and 0.05 for YOLO11s) and applies them once to test; these are the primary
single-threshold operating points for the historical three-seed Batch 14
analysis. Equal-weight F1 is transparent but does not encode clinical
false-negative and false-positive costs, and only three validation seeds
informed that frozen selection. It was not reselected after the additional
seeds, and seed 271 also emits nothing at the selected YOLO threshold of 0.05.
The complete held-out threshold sweep is descriptive rather than a source of
deployment settings.

Batch 19 adds a separate validation-only cost-sensitivity analysis; D-004
explicitly prevents it from replacing the Batch 14 primary thresholds. It
assumes false-negative/false-positive cost ratios of 1, 9, 25, and 100 rather
than estimating clinical utility from outcomes. Its 2,000-draw hierarchical
bootstrap resamples 321 validation-patient groups and the three frozen seeds,
so seed uncertainty remains coarse. The confidence intervals are pointwise,
not simultaneous over the 99 candidate thresholds, and selecting the maximum
lower bound is not a 95% guarantee about that selected maximum. YOLO11s reaches
the 0.01 lower sweep boundary for beta 3, 5, and 10; those thresholds are best
observed within the grid rather than claimed global optima. None of the
cost-sensitive thresholds is applied to test or adopted as a deployment rule.

**The clean localization summaries have an intentionally asymmetric sample
size.** Conditional IoU and Dice average only matched true positives; they
describe box quality after a successful detection and must not be read without
recall. Faster R-CNN has defined clean localization for all five seeds
(`n=5`), whereas YOLO11s has it for four (`n=4`) because seed 271 has no
score-0.25 true positive. The descriptive table therefore reports detector-
specific `n=5` versus `n=4`; paired inference uses the four complete seed pairs
17, 42, 137, and 314. The undefined seed is not coerced to zero. Conditional
localization is also undefined for the seed-17 YOLO checkpoint under the
darkest corruption, which is a separate, corruption-specific event.

Batch 18 separately measures raw-score calibration with the full five-dimensional D-ECE
over confidence and relative box center/scale. It uses all five frozen seeds and every
post-NMS prediction retained at the 0.001 bundle floor, including YOLO11s seed 271. This is
test-set evaluation rather than validation-fitted recalibration: no calibrated mapping is
learned or evaluated on an independent second holdout. The estimand is conditional on emitted
detections, so missed targets without a score do not enter it. Absolute D-ECE also depends on
the fixed IoU-0.50 correctness rule, five-bin feature partition, and eight-sample cell
minimum; it is not directly comparable to D-ECE under a different protocol and does not
establish calibrated clinical risk.

Registered-operation GFLOPs omit unsupported operations and are estimates, not
direct hardware timings. Synchronized batch-1 speed profiles include each
framework's native detector forward and postprocessing, but Faster R-CNN
resizing occurs inside its timed model forward while YOLO tensor resizing occurs
before timing. FPS is therefore an implementation-specific deployment
comparison between these documented pipelines on the measured laptop, not an
architecture-general claim or a pure kernel benchmark. The external Anaconda
environment used for the measured runs is captured by run-level package and
hardware snapshots, but exact timing can still vary with GPU power state,
driver, and system load.

## Robustness scope

The corruption grid contains seven digital corruption types at five ordered
severities. The indices are not physically calibrated or comparable across
types. Equal-weighted mean retention over 35 conditions is a descriptive
summary, not an estimate under a known deployment distribution.

Albumentations applies brightness, synthetic noise/blur, and JPEG changes to
post-conversion uint8 PNGs. These transformations do not reproduce scanner
physics, DICOM window/VOI behavior, reconstruction failures, acquisition
protocol changes, population shift, site shift, or adversarial manipulation.
The grid measures digital robustness under the specified transformations, not
clinical robustness or safety. It repeatedly transforms the same 300 images,
so corruption conditions are not independent deployment cohorts.

## Explainability scope

Grad-CAM uses matched stride-16, 40 by 40 backbone maps interpolated to the
original 1024 by 1024 images. This limits spatial precision. The method explains
a selected foreground score, not proposal generation, NMS, candidate selection,
box regression, or the detector's complete causal decision path.

Operating-point false negatives have no emitted detection to explain. Their
maps therefore use the ground-truth-associated, low-threshold candidate with
highest IoU and are explicitly proxy failure-analysis maps. This
annotation-guided choice is unavailable in deployment, nearby boxes can reuse a candidate, and
the result is not an explanation of an actual detection.

Energy-in-box and pointing-game metrics treat rectangular boxes as pixel-level
opacity masks even though boxes contain normal tissue and larger boxes make
hits easier. The
analysis reports box-area reference values, excludes and reports one zero-energy
Faster R-CNN map, and does not assign box metrics to the 232 box-negative
images. The nine heatmap panels are objective examples, not prevalence
estimates. All false-positive panels come from `No Lung Opacity / Not Normal`
studies, so unannotated abnormalities can be meaningful context rather than a
simple confound.

CUDA ROI Align backward lacks a bitwise-deterministic implementation in the
pinned Torchvision build. The run is seeded under deterministic warn-only mode
and records the environment, but Faster R-CNN CAM bytes may vary slightly on a
different hardware/library rerun. More broadly, a plausible heatmap is a
failure-analysis association map and not evidence of clinical reasoning.

## Statistical scope

Batch 13 identified and corrected the original image-level clustering error.
The primary bootstrap now resamples all exams from each NIH patient together,
and the paired permutation swaps detector labels once per patient group. The
superseded image-level tables and summary remain explicitly archived for audit
but are not primary evidence. For precision, recall, F1, and both AP endpoints,
the clean hierarchical bootstrap resamples all five paired training seeds. For
conditional IoU and Dice it resamples the four complete pairs 17, 42, 137, and
314; seed 271 is ineligible only because that matched-only estimand is
undefined. Patient identifiers, patient-cluster construction, cluster
bootstrap expansion, and patient-level detector-label-swap algorithms are
unchanged. The two endpoint groups use separate deterministic random streams,
so their realized draw masks are not claimed to be identical. Corruption
inference remains conditional on the primary checkpoints.

Permutation p-values condition on the observed checkpoints, whereas clean
confidence intervals also resample the eligible seed pairs for each endpoint.
These answer related but different uncertainty questions. All seven clean
endpoints remain in one Holm family despite the endpoint-specific seed count;
the 95% percentile intervals are pointwise rather than simultaneous, and
Holm-adjusted p-values govern family-wise claims. Patient clustering handles
the observed within-patient dependence, but neither it nor Holm correction
makes severity conditions independent, calibrates their clinical likelihood,
guarantees transportability, or establishes external-site robustness.

McNemar's test is omitted because the benchmark does not produce one
independent binary outcome per image. Reducing multiple targets and
negative-image false positives to one flag would discard detection structure,
while target-level decisions would remain nested within images.

## Deployment and regulatory scope

The benchmark does not demonstrate prospective benefit, calibrated risk,
clinical safety, subgroup equity, workflow compatibility, or generalization to
another site. Clinical deployment would require prospective and external
validation, human-factors and safety engineering, monitoring, and the applicable
medical-device regulatory process, such as FDA clearance or CE marking. Those
activities are beyond this project; mentioning them defines the boundary of the
claims and is not an implementation deliverable.
