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
- The primary training pipeline is scoped to one seed. If time permits, only
  the headline quantitative comparison will be repeated for two additional
  seeds. Robustness and explainability conclusions will not represent full
  across-seed variability.
- The later corruption grid and quantitative explanation metric will use the
  same documented, stratified 200–400-image test subset. Qualitative heatmaps
  will cover only a small set of representative successes and failures.
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

These constraints are fixed before model training. Any later adjustment will
be added to the decision log and reported with the affected results.
