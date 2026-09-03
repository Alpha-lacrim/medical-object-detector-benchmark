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
together. Batch 28 further separates primary training-procedure intervals from
secondary checkpoint-conditional p-values. The remaining limitation is the
finite number of observed patient groups, not unaddressed image-level clustering.

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

No formal statistical sample-size or power calculation determined the 5,000
studies. Seed 17 and deterministic SHA-256 ordering selected whole NIH patient
groups while tracking the three source label strata; the remaining 21,684
labeled studies were excluded for hardware scope, not annotation-audit failure.
Stratification and patient grouping reduce avoidable imbalance and leakage but
do not turn a hardware-constrained subset into a prevalence-representative or
statistically powered clinical sample. The resulting estimates may differ from
those for the full source cohort even before considering external populations.

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
analysis. Equal-weight F1 is transparent but does not encode an empirically
elicited clinical-harm function, and only three validation seeds
informed that frozen selection. It was not reselected after the additional
seeds, and seed 271 also emits nothing at the selected YOLO threshold of 0.05.
The complete held-out threshold sweep is descriptive rather than a source of
deployment settings.

Batch 35 recomputes test-side threshold, official PR, FROC, frozen-threshold,
and Pareto sensitivity over all five attempts without changing the n=3
validation selection. This improves consistency with the clean comparison but
does not create five-run validation evidence: threshold-selection n remains 3
while sensitivity test n is 5. The 0.01--0.99 test sweep remains exploratory,
its peak F1 and fixed-target rows are not selection rules, and pointwise
mean +/- sample-SD bands over five runs remain coarse. Seed 271 contributes its
observed nonzero AP and low-score FROC behavior but defined zeros at 0.25 and
0.05. The n=5 Pareto cloud uses five hardware rows per detector, but timing is
still specific to one laptop/software state, and strict cloud dominance is a
deterministic descriptive rule rather than uncertainty quantification.

The n=3-to-n=5 audit is not uniformly favorable. Shared-threshold YOLO11s
precision and Faster R-CNN recall margins weaken, as do the mean AP gaps; the
shared-threshold F1, all five FROC, and all three frozen-threshold gaps
strengthen; the AP@0.5 position count strengthens; the AP@0.5:0.95 position
count and all four Pareto labels are unchanged. No conclusion reverses. These
classifications compare observed margins and must not be read as inferential
evidence that adding two more training runs caused a true effect change.

Batch 29 corrects the separate validation-only F-beta analysis: beta is a
recall-versus-precision preference parameter, and beta squared is the relative
recall weight in the harmonic mean, not an empirically measured clinical-harm
ratio. The frozen beta values 1, 3, 5, and 10 and their original selected
thresholds are retained under D-006. A 0.01 absolute near-optimal-LCB plateau
and draw-specific bootstrap argmax frequencies describe threshold instability
without changing the canonical rule. Faster R-CNN's beta-3 and beta-5
bootstrap selected-threshold intervals are wide (0.13--0.51 and 0.04--0.29),
while YOLO11s beta 3--10 remains pinned to the 0.01 lower grid boundary.

The separate hypothetical linear detection-error loss
`r * FN / N + FP / N` uses assumed ratios 1, 9, 25, and 100. Those ratios assign
exchangeable penalties to unmatched target boxes and false-positive detections;
they are not patient-outcome valuations, deployment utilities, or evidence that
one kind of error causes a fixed amount of clinical harm. Both sensitivity
analyses use 2,000 patient-cluster/seed draws over only three validation seeds.
Their intervals are pointwise, not simultaneous over 99 candidates, and none
of their thresholds is applied to test or adopted as a deployment rule.

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

The versioned detection-calibration audit measures full five-dimensional D-ECE over
confidence and relative box center/scale for all five frozen seeds and every post-NMS
prediction retained at the 0.001 bundle floor. This is descriptive test-set evaluation,
not validation-fitted recalibration or a patient/run-level inferential comparison. Its
estimand is conditional on emitted detections: a missed ground-truth object has no emitted
confidence and therefore lies outside the D-ECE population. The (5^5=3,125)-cell grid is
sparse (68--354 occupied and 15--169 meeting the eight-detection minimum per run), and the
supported detection fraction ranges from 0.543 to 0.983. Absolute D-ECE changes with the
bin/minimum-cell grid and with the confidence-floor population; the versioned occupancy and
predeclared sensitivity outputs must accompany the original five-bin/minimum-8 values. D-ECE
under another protocol is not directly comparable and does not establish missed-target,
exam-level, or clinical-risk calibration. It is programmatically separate from the
validation-frozen exam-level outcome probabilities required by any valid DCA.

Batch 30 classified the historical Batch 20 calculation as **non-standard for
conventional DCA interpretation**. It used maximum emitted detector confidence
both to define an exam action and, through `tau/(1-tau)`, to weight false
positives. A bounded raw score is not automatically a predicted exam-outcome
probability or a decision-maker's harm/benefit threshold. The arithmetic is
preserved and relabeled as an exploratory raw-score threshold
utility/sensitivity curve, but it was removed from the main manuscript Results
and supplies no standard net-benefit, clinical-utility, beneficial-range, or
deployment evidence. Probability-based salvage was not attempted: only six of
the ten retained detector/runs have frozen validation predictions, leaving
both detectors' seeds 271 and 314 without the run-specific validation data
needed to fit and freeze mappings before test evaluation. No calibrator was
chosen or fitted. The historical population was also a deliberately
stratified, enriched 750-image internal test subset (169 positive, 581
negative, 323 patient groups; 22.533% image-level prevalence), not a deployment
prevalence sample. Full classification and archive provenance are in
[`DCA_ANALYSIS.md`](DCA_ANALYSIS.md).

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

Batch 32 re-audited the ten pre-conversion synthetic shifts against the current
DICOM standard. Although all 300 objects record `Modality=CR`, every object is
actually Secondary Capture Image Storage, workstation-converted (`WSD`), 8-bit
`MONOCHROME2`, and marked as previously lossily JPEG-compressed. All lack Pixel
Intensity Relationship/Sign, Presentation Intent Type, Modality LUT/rescale,
VOI LUT/Window Center/Width, processing descriptions, and calibrated exposure
or detector-response metadata. The stored values therefore cannot be shown to
be linear or logarithmic in incident X-ray signal, nor inverted reliably to
such a scale.

The DICOM `LINEAR` alternatives remain class-A display-transform sensitivity
settings, not recovered vendor presets. The former dose conditions are class D
for physical validity and retained only as class-B signal-dependent
Poisson-like intensity perturbations; they are not dose/quantum-noise proxies
and not validated low-dose acquisition simulations. Gaussian kernels are
class-C generic blur/spatial-resolution proxies, not a detector MTF,
reconstruction kernel, or scanner model. Per-image min-max scaling partly
cancels the two center shifts and almost completely cancels the widened window
(264/300 exact pixel identities), while it can re-stretch blur differences.
DSI is only a descriptive performance-retention/domain-sensitivity index and
does not estimate inter-site transportability. The historical detector outputs
remain numerically valid for these synthetic sensitivities, so no inference was
rerun. Neither stress test establishes clinical robustness, scanner safety, or
performance under a prospectively changed acquisition protocol.

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

Batch 31 corrects and extends the nested 50-image/41-patient control analysis.
The historical Batch 21 “data randomization” label referred to inference-time
within-image pixel-vector shuffling. It is retained only as an input-pixel
randomization stress control and cannot establish sensitivity to the learned
training data-label relationship. The canonical Adebayo training-label
data-randomization test was not performed because it would require retraining
identical detectors on randomized annotations and verifying that those tasks
were fit. The historical all-weights reinitialization is a full
model-parameter randomization control, not a cascading reproduction.

The v2 extension adds six transparent cumulative head-to-input layer groups for
each detector and compares independently normalized 40 by 40 maps with Pearson,
tie-aware Spearman, and Gaussian-window SSIM. Low similarity after full-model
randomization supports only parameter sensitivity under this audit. It does not
establish anatomical correctness, causal faithfulness, clinical reasoning, or
medical validity. The curve uses one deterministic random draw per stage, one
seed-17 checkpoint per detector, and no inferential interval; its non-monotonic
intermediate similarities should remain descriptive. Seven Faster R-CNN
input-control maps and four YOLO11s full-randomization maps are zero or constant
and excluded from all metrics rather than imputed.

The control analysis also uses a fixed trained reference region and a
pre-activation foreground target rather than Phase 7's ground-truth-associated
post-activation target. Pixel permutation is a severe out-of-distribution
perturbation. Faster R-CNN retains mean input-control SSIM 0.2468 despite
near-zero mean Pearson and Spearman correlations, underscoring that no single
similarity measure fully characterizes map persistence.

## Statistical scope

All H1--H5 statements were recorded retrospectively after most result artifacts
existed, and H6 was added as a retrospective descriptive calibration question.
They are traceability devices, not preregistered or confirmatory hypotheses.
H1 is compound, while its primary intervals are pointwise rather than a
multiplicity-adjusted simultaneous family; therefore its all-endpoint pattern
must not be presented as a prospectively controlled confirmatory test.

Batch 13 identified and corrected the original image-level clustering error.
The primary training-procedure bootstrap resamples all exams from each NIH
patient together and samples trained runs independently within each detector.
Same-number seeds are not matched stochastic blocks: the PyTorch and
Ultralytics loaders, batch structures, initialization/RNG paths, and stopping
trajectories are not coupled. Unconditional endpoints use five runs per
detector. Conditional IoU and Dice use five defined Faster R-CNN runs and four
defined YOLO11s runs; seed 271 remains included wherever its endpoint is
defined. Five runs are still a coarse empirical representation of retraining
variability.

The secondary permutation analysis swaps detector labels once per patient
group and conditions on the observed checkpoints. Its Holm-adjusted p-values
do not test training-procedure variability. In particular, fixed-threshold
precision has a primary interval crossing zero despite a small
checkpoint-conditional p-value; the latter cannot be used to support a
training-procedure difference. The historical paired-seed and superseded
image-level outputs remain explicit sensitivity/audit archives. All seven
checkpoint-conditional clean p-values remain in one Holm family; the primary
95% intervals are pointwise rather than simultaneous. Patient clustering handles
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

The Batch 34 checklist assesses the current `report/paper_draft.md`, not the
historical technical report. CLAIM 2024 is the primary medical-imaging AI
framework, but the internal crosswalk is not an official completed submission
checklist and does not establish compliance. Final STARD-AI 2025 is aimed at
diagnostic-accuracy studies using an AI index test; this object-localization
benchmark lacks a participant-level diagnostic index-test/reference-standard
analysis and is used only by analogy. TRIPOD+AI is likewise used only by
analogy because no individualized diagnostic or prognostic probability model
is developed. The evidence base still lacks source accrual dates, an
author-confirmed ethics/data-use and consent determination, funding and
competing-interest declarations, author contributions, a release-ready
data/code availability statement, patient/public involvement disclosure, a
participant-flow diagram, demographic subgroup/fairness evaluation, and
external testing. Narrative reporting cannot repair those gaps without new
traceable evidence; the unresolved declarations are isolated in
[`AUTHOR_DECLARATIONS_TODO.md`](AUTHOR_DECLARATIONS_TODO.md).
