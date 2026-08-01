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

These constraints are fixed before model training. Any later adjustment will
be added to the decision log and reported with the affected results.
