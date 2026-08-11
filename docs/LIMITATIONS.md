# Limitations

This benchmark is a controlled course-project comparison, not a clinical
validation study. Its outputs must not be used to diagnose pneumonia or guide
patient care.

## Dataset scope

- The RSNA challenge data originate from a selected subset of the NIH clinical
  archive. A single source institution, historical acquisition practices, and
  an adult-heavy population limit transportability to children, portable-care
  settings, other hospitals, and newer imaging equipment.
- The challenge cohort was enriched using existing labels rather than sampled
  as a prevalence-representative clinical population. Reported precision and
  false-positive behavior therefore should not be interpreted as deployment
  prevalence estimates.
- The target is a radiographic **lung opacity**, not a microbiologically or
  clinically confirmed pneumonia diagnosis. Opacity can reflect several other
  conditions, and pneumonia can exist without a boxed focal opacity.
- The benchmark uses one foreground category. It cannot establish how the two
  detector families compare on fine-grained multi-class disease localization.
- Patient-level grouping prevents known repeat-exam leakage across splits, but
  the released metadata do not expose all possible acquisition or encounter
  relationships. Unrecognized correlations may remain.

## Annotation scope

- Bounding boxes are coarse rectangular approximations of findings rather than
  pixel-accurate segmentations. Reader disagreement and ambiguous boundaries
  are unavoidable.
- Only part of the challenge cohort received multiple independent reads and
  adjudication. Annotation certainty is therefore not uniform across studies.
- The coordinate audit can prove that boxes are syntactically valid and inside
  the declared image dimensions; it cannot prove every box is clinically
  correct. Visual EDA is a sample-based check, not a second expert reading.

## Compute-driven scope

- The main experiment uses a fixed, stratified 5,000-study subset rather than
  all 26,684 labeled studies. This fits iterative training and evaluation on an
  RTX 4060 Laptop GPU with 8 GB VRAM and 16 GB system RAM, but increases sampling
  uncertainty and narrows generalization to the full challenge cohort.
- The primary baseline-development pipeline used one seed; Phase 5 spends the
  reserved budget on two additional full trainings per detector (seeds 17, 42,
  and 137 total) and reports sample mean ± standard deviation for headline
  metrics. Three runs provide only a coarse estimate of training-seed variation.
  Robustness and explainability remain scoped to the selected primary
  checkpoint and will not represent full across-seed variability.
- The seed-42 and seed-137 timing approvals reuse the accepted seed-17 timing
  measurements because only RNG/output identity changes. The derived gate files
  record the original artifact hash, implementation compatibility, and target
  seed; actual training time and peak allocated memory are still measured for
  every run. This avoids repeated timing-only epochs but does not test whether
  seed choice itself creates small runtime variation.
- The corruption grid uses one fixed 300-image test subset, drawn with seed 17
  by proportional largest-remainder allocation over the three study strata.
  It contains 68 Lung Opacity, 132 No Lung Opacity / Not Normal, and 100 Normal
  images (68 positive, 232 negative), 111 boxes, and 183 distinct NIH patient
  identifiers. The exact deterministic procedure and manifest hash are in
  `docs/ROBUSTNESS.md`. Phase 7 reuses this exact sample: quantitative Grad-CAM
  covers all 111 ground-truth boxes, while its nine paired qualitative images
  are selected by a fixed high-IoU/shared-false-positive/failure-quantile
  rubric. These results still characterize only one primary checkpoint per
  detector and one small held-out sample.
- Robustness uses only each detector's selected primary seed-17 checkpoint, not
  all three training seeds. Its apparent between-model differences therefore
  combine architecture behavior with the unmeasured training-seed sensitivity
  of the corruption curves.
- The robustness sample is stratified and drawn without replacement at the
  image level. Its 300 images represent only 183 patients, so repeated exams
  mean image-level observations are not guaranteed independent. The patient-
  safe separation from training remains intact, but later paired inference
  must acknowledge this within-test clustering and the modest 111-box count.
- The five severity indices are ordered within each corruption but are not
  physically calibrated or equivalent across corruption types. Equal-weighted
  mean retention across the 35 conditions is a descriptive summary, not an
  estimate under a known deployment distribution.
- Albumentations applies the tested corruptions to post-conversion uint8 PNGs.
  Lighting shifts, synthetic noise/blur, and JPEG quality changes do not
  reproduce scanner physics, DICOM window/VOI behavior, reconstruction errors,
  institutional shift, or patient-population shift. Robustness on this grid
  cannot establish clinical robustness or safety.
- Conditional IoU/Dice are undefined if a condition has no true positives at
  the operating threshold. This occurs for YOLO11s at the darkest severity;
  the tables preserve blank ratios rather than replacing them with zero.
- Images are standardized to a configured 640-pixel training size later in the
  pipeline. Small findings may lose detail, and resizing can affect detector
  families differently.
- The Faster R-CNN baseline uses physical batch size 2 with gradient
  accumulation to an effective optimizer batch of 4. Accumulation does not
  enlarge per-forward BatchNorm samples, so pretrained BatchNorm running
  statistics are frozen while affine parameters remain trainable.
- Both primary detector arms use no stochastic training augmentation. For
  YOLO11s, all Ultralytics extras are explicitly disabled: mosaic, mixup,
  cutmix, copy-paste, HSV jitter, flips, geometric transforms, erasing,
  auto-augmentation, and multi-scale training. This removes augmentation as a
  known comparison confound, but it may understate YOLO's performance under its
  conventional default recipe and does not make the architectures' losses or
  optimization dynamics identical.
- YOLO11s uses a one-epoch linear learning-rate warmup from zero to 0.001,
  whereas Faster R-CNN starts at 0.005. No-warmup YOLO diagnostics reached a
  non-finite classification loss at epoch 1, batch 29 under AMP; warming up to
  0.005 avoided `NaN` but collapsed to all-zero losses and near-zero scores
  after epoch 1. The lower target is a disclosed numerical-stability exception
  and a remaining optimization asymmetry.
- YOLO uses mandatory bfloat16 AMP for its model forward/backward and runs
  task-aligned target assignment and detector losses in float32. With the
  pinned library's default float16 path, one-class head scores underflowed to
  zero before the loss and training collapsed; a positive-spread diagnostic
  showed batching was not the root cause and was discarded. Faster R-CNN uses
  float16 AMP, so exact autocast/loss-precision symmetry is not possible.
  Ultralytics' float16-specific AMP equivalence probe is replaced by a CUDA
  bfloat16 support/dtype gate plus smoke and per-batch numerical guards.
- Faster R-CNN retains pretrained BatchNorm statistics, while YOLO11s updates
  its native BatchNorm statistics. Forcing YOLO's statistics to remain frozen
  still collapsed the one-class head under otherwise stable bfloat16 AMP, so
  normalization behavior must remain an architecture-specific asymmetry.
- Phase 5 IoU and Dice are averaged only over matched true-positive boxes at
  score 0.25 and matching IoU 0.50. They characterize conditional localization
  quality, not missed findings, and must be interpreted together with recall.
- Synchronized batch-1 speed profiles include each framework's native detector
  forward and postprocessing, but Faster R-CNN performs configured resizing
  inside its model forward while the YOLO tensor resize occurs before timing.
  FPS is therefore an implementation-level deployment comparison rather than a
  pure architecture-only kernel benchmark.

- Grad-CAM uses a matched 40 by 40 stride-16 backbone feature map, so it is
  inherently coarse after interpolation to 1024 by 1024 images. It explains a
  selected foreground score, not proposal generation, NMS, candidate selection,
  box regression, or the full end-to-end decision process.
- Quantitative false-negative maps require a ground-truth-guided proxy: the
  low-threshold retained candidate with highest IoU to each missed box. This is
  useful for asking whether a latent candidate uses box-local evidence, but it
  conditions on annotation knowledge unavailable at deployment and is not an
  explanation of a detection the model actually emitted. Nearby ground-truth
  boxes can select the same proxy candidate.
- Energy-in-box and pointing-game values treat rectangular boxes as lesion
  masks even though boxes include normal tissue and make hits easier as their
  area grows. The report includes rasterized box area as a reference, excludes
  and reports one zero-energy Faster R-CNN map, and does not assign box metrics
  to the 232 images without a Lung Opacity box.
- CUDA ROI Align backward warns that it lacks a deterministic implementation in
  the pinned Torchvision build. Phase 7 remains seeded under deterministic
  warn-only mode and records the warning/environment, but Faster R-CNN CAM
  bytes may vary slightly across hardware or library reruns.
- The nine heatmap panels are objective examples, not a prevalence estimate.
  All selected false-positive panels come from `No Lung Opacity / Not Normal`
  studies; without boxes for alternative abnormalities, the maps cannot prove
  whether activation reflects a true non-opacity finding or a confounding
  marker, device, border, or background feature.
- Phase 8 uses the required image-level paired bootstrap. Repeated exams reduce
  the effective independent patient count, so its intervals can be narrower
  than a patient-cluster bootstrap would be. The patient-safe train/test split
  remains intact; this limitation concerns dependence within the held-out set.
- The clean hierarchical bootstrap resamples only three paired training seeds,
  which gives a coarse empirical account of seed variation. Corruption tests
  remain conditional on the two selected seed-17 checkpoints and cannot infer
  across-seed robustness variability.
- Permutation p-values condition on the observed trained checkpoints, whereas
  clean confidence intervals also resample the three seed pairs. These answer
  related but not identical uncertainty questions and can straddle a threshold
  differently. Holm-adjusted p-values determine family-wise claims; the 95%
  percentile intervals are pointwise rather than simultaneous.
- The grid has 35 repeated transformations of the same 300 images. Holm
  controls each metric/estimand p-value family, but it does not make corruption
  severities independent, calibrate their clinical likelihood, or establish
  robustness on a new site. Darkest-condition conditional IoU/Dice remain not
  estimable because YOLO has no operating-point true positive.
- McNemar's test is omitted because the benchmark does not produce one binary
  outcome per independent image. Reducing multiple targets and negative-image
  false positives to a single correct/incorrect flag would discard detection
  structure; target-level decisions would remain nested within images.

These constraints are fixed before model training. Any later adjustment will
be added to the decision log and reported with the affected results.
