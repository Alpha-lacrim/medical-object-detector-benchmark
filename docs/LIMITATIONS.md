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
exams within a split, so image-level observations are not fully independent by
patient.

## Annotation and preprocessing scope

Bounding boxes are coarse rectangular approximations, not pixel-accurate lesion
masks. Reader disagreement, ambiguous boundaries, and non-uniform annotation
certainty remain even though the metadata audit found no malformed,
non-positive-area, off-image, or exact duplicate positive boxes. Coordinate
validation and the 12-image EDA establish technical consistency, not clinical
correctness or a second expert reading.

DICOM conversion uses deterministic `MONOCHROME1` inversion and per-image
min-max scaling to 8-bit PNG. It does not reproduce vendor-specific window/VOI
processing. Later resizing to 640 pixels can remove small-finding detail and
can affect detector families differently.

## Compute and sampling scope

The target machine is an RTX 4060 Laptop GPU with 8 GB VRAM, 16 GB system RAM,
and an i7-13650HX. Those constraints motivated a fixed stratified 5,000-study
subset rather than all 26,684 labeled studies, 640-pixel inputs, small-model
selection, mixed precision, bounded batches, and early stopping. These choices
make the study reproducible on the stated laptop but increase sampling
uncertainty and narrow generalization to the full challenge cohort.

The primary development pipeline uses seed 17. The held-out headline comparison
adds seeds 42 and 137 for three full trainings per detector, but three runs only
coarsely estimate training-seed variation. The seed-42/137 timing approvals
reuse the accepted seed-17 measurements because only RNG and artifact identity
change; every full run still records its own training time and peak memory.

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

The Faster R-CNN physical microbatch remains two even with accumulation;
accumulation does not create batch-four BatchNorm statistics. Windows memory
constraints also require non-persistent Faster R-CNN train/validation worker
pools and fewer YOLO workers. These are resource-management differences rather
than experimental endpoints.

## Metric and compute-measurement scope

Precision, recall, and F1 use one fixed score threshold of 0.25 and matching IoU
of 0.50. Another calibrated threshold would move the precision-recall trade-off
and requires a separate experiment. Conditional IoU and Dice average only
matched true positives; they describe box quality after a successful detection
and must not be read without recall. They are undefined when a condition has no
true positive, as occurs for YOLO11s under the darkest corruption.

Registered-operation GFLOPs omit unsupported operations and are estimates, not
direct hardware timings. Synchronized batch-1 speed profiles include each
framework's native detector forward and postprocessing, but Faster R-CNN
resizing occurs inside its timed model forward while YOLO tensor resizing occurs
before timing. FPS is therefore an implementation-level deployment comparison,
not a pure architecture kernel benchmark. The external Anaconda environment
used for the measured runs is captured by run-level package and hardware
snapshots, but exact timing can still vary with GPU power state, driver, and
system load.

## Robustness scope

The corruption grid contains seven digital corruption types at five ordered
severities. The indices are not physically calibrated or comparable across
types. Equal-weighted mean retention over 35 conditions is a descriptive
summary, not an estimate under a known deployment distribution.

Albumentations applies brightness, synthetic noise/blur, and JPEG changes to
post-conversion uint8 PNGs. These transformations do not reproduce scanner
physics, DICOM window/VOI behavior, reconstruction failures, acquisition
protocol changes, population shift, site shift, or adversarial manipulation.
The grid cannot establish clinical robustness or safety. It repeatedly
transforms the same 300 images, so corruption conditions are not independent
deployment cohorts.

## Explainability scope

Grad-CAM uses matched stride-16, 40 by 40 backbone maps interpolated to the
original 1024 by 1024 images. This limits spatial precision. The method explains
a selected foreground score, not proposal generation, NMS, candidate selection,
box regression, or the detector's complete causal decision path.

Operating-point false negatives have no emitted detection to explain. Their
maps therefore use the ground-truth-associated, low-threshold candidate with
highest IoU and are explicitly proxy diagnostics. This annotation-guided
choice is unavailable in deployment, nearby boxes can reuse a candidate, and
the result is not an explanation of an actual detection.

Energy-in-box and pointing-game metrics treat rectangular boxes as lesion masks
even though boxes contain normal tissue and larger boxes make hits easier. The
analysis reports box-area reference values, excludes and reports one zero-energy
Faster R-CNN map, and does not assign box metrics to the 232 box-negative
images. The nine heatmap panels are objective examples, not prevalence
estimates. All false-positive panels come from `No Lung Opacity / Not Normal`
studies, so unannotated abnormalities can be meaningful context rather than a
simple confound.

CUDA ROI Align backward lacks a bitwise-deterministic implementation in the
pinned Torchvision build. The run is seeded under deterministic warn-only mode
and records the environment, but Faster R-CNN CAM bytes may vary slightly on a
different hardware/library rerun. More broadly, a plausible heatmap is an
association diagnostic and not evidence of clinical reasoning.

## Statistical scope

The required bootstrap resamples images, but repeated exams reduce the effective
independent patient count. Its intervals can therefore be narrower than those
from a patient-cluster bootstrap. The clean hierarchical bootstrap resamples
only three paired training seeds, while corruption inference remains
conditional on the primary checkpoints.

Permutation p-values condition on the observed checkpoints, whereas clean
confidence intervals also resample the three seed pairs. These answer related
but different uncertainty questions. The 95% percentile intervals are
pointwise rather than simultaneous; Holm-adjusted p-values govern family-wise
claims. Holm correction does not remove within-patient dependence, make
severity conditions independent, calibrate their clinical likelihood, or
establish external-site robustness.

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
