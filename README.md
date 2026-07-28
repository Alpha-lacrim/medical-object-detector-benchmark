# Medical Object Detector Benchmark

**Comparative analysis of Faster R-CNN and YOLO for medical-image object
detection**

This repository is the reproducible workspace for the final project described in
[`Final project 1405.v1.pdf`](./Final%20project%201405.v1.pdf). The study will
compare a two-stage detector (Faster R-CNN) with a one-stage detector (YOLOv8 or
newer) on one fixed medical-image dataset under controlled training conditions.

The benchmark is designed to answer more than “which selected implementation
has the highest mAP?” It will compare:

- clean-set detection quality;
- latency, throughput, memory, parameter count, FLOPs, training time, and model
  size;
- degradation under brightness, noise, blur, and JPEG corruptions;
- detection-aware Grad-CAM explanations for successes and failures;
- paired confidence intervals, significance tests, and effect sizes; and
- suitability for research or deployment under different constraints.

## Current status

Repository and protocol design are in progress. No dataset has been downloaded
and no model result has been produced yet. Any result directory, table, or
figure must remain clearly marked as generated evidence rather than an expected
outcome.

## Study strategy

The project separates two questions that should not be conflated:

- **Track A — assignment-aligned control:** both detectors share the strict
  training contract requested by the brief.
- **Track B — architecture-optimized:** each detector may use predeclared
  model-appropriate settings under the same tuning opportunity.

Both tracks use the same audited data, immutable test set, evaluator, metrics,
and final hardware, and their results are reported separately. Scientifically
necessary deviations from the brief are preregistered with evidence while the
compliant Track A is preserved whenever feasible.

## Recommended study

The current recommendation is:

- **Dataset:** [Medical Image DataSet: Brain Tumor Detection](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection),
  subject to an annotation and leakage audit.
- **Baseline:** torchvision Faster R-CNN with a ResNet-50-FPN-v2 backbone and
  pretrained weights.
- **Comparator:** Ultralytics YOLO26-small with pretrained weights, subject to
  an explainability and reproducibility preflight. YOLO11-small is the
  predeclared compatibility fallback if that preflight fails.
- **Runs:** one smoke-test seed, followed by at least three final seeds if the
  available compute budget permits.
- **Evaluation:** one shared, model-independent evaluator and one frozen test
  set for both detectors.

These choices remain provisional until the dataset audit and decision log are
frozen. Instructor clarification is preferred, but documented fallbacks prevent
silence from stalling the project. In particular, the PDF mentions five classes
while the likely source dataset describes four image categories.

Any conclusion will apply to the exact Faster R-CNN and YOLO variants tested,
not to every possible two-stage or one-stage detector.

## Work plan

1. Seek clarification on the source paper, class semantics, deadline, and
   accepted YOLO version; freeze documented fallbacks for unanswered items.
2. Download and audit the dataset; preserve source metadata, split manifests,
   checksums, class mappings, duplicate clusters, and sample visualizations.
3. Freeze the experiment contract before any final training.
4. Implement shared data, augmentation, evaluation, profiling, corruption,
   explainability, and statistics modules.
5. Train and validate both detectors under Track A, then run equal-opportunity
   tuning and final training under Track B.
6. Run clean, efficiency, robustness, Grad-CAM, and paired statistical analyses.
7. Generate all report tables and figures from machine-readable result files.
8. Reproduce the final results from a clean environment and complete the
   12-section report required by the brief.

The complete protocol and acceptance criteria are in
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). Preregistered interpretations
and deviations are in [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md).

## Repository memory

Future sessions must read these files before making changes:

- [`AGENTS.md`](AGENTS.md): stable operating rules and the mandatory
  start/end-of-session procedure.
- [`Codex.md`](Codex.md): durable project context, decisions, constraints, and
  the important-file map.
- [`Handoff.md`](Handoff.md): chronological changes, validation performed,
  incomplete work, and the exact next actions.

Large datasets, checkpoints, credentials, and generated experiment artifacts are
not committed to Git. This repository is for research and education only; it
does not establish clinical validity and must not be presented as a medical
device.
