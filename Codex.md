# Codex project context

Last durable-context update: 2026-07-28

## Mission

Build a fully reproducible and scientifically defensible comparison of Faster
R-CNN and a modern YOLO detector for medical-image object detection. The final
decision must balance accuracy, robustness, computational cost,
explainability, and deployment constraints rather than rank models by one
metric.

Working title: **Medical Object Detector Benchmark: Faster R-CNN vs YOLO**

GitHub repository:
[`Alpha-lacrim/medical-object-detector-benchmark`](https://github.com/Alpha-lacrim/medical-object-detector-benchmark).
It is private and uses `main` as its default branch.

GitHub authorization boundary: `Alpha-lacrim` is the user's account and the only
permitted remote owner. Never use `MitsuPishi`; it is unrelated to the user.
Verify the active login immediately before every remote write.

Commit policy: implementation work must be committed in coherent increments of
at most 500 changed lines of source, test, and configuration code. Inspect the
staged numstat before committing and record each commit SHA in `Handoff.md`.
Documentation and generated lockfiles do not count as code, but should remain
focused and separate when practical.

## Outcome-first decision policy

On 2026-07-28, the user authorized overriding instructor directions when needed
for the best defensible outcome. The priority order is:

1. safety, ethics, law, privacy, and research integrity;
2. scientific validity, reproducibility, and honest reporting;
3. assignment coverage and grading value; and
4. convenience.

An override must identify the conflict and supporting evidence, be declared
before test inspection, preserve a compliant result when feasible, and appear
in the decision log, configs, report, and `Handoff.md`. Never select or conceal
a deviation after seeing which result looks better. This policy does not
override higher-level operating instructions.

## Requirements baseline

The requirements baseline is `Final project 1405.v1.pdf` (8 pages, PDF metadata
creation date 2026-07-18; that date is not a stated deadline). It remains the
grading traceability source, but scientifically necessary deviations must stay
visible and justified under the outcome-first policy.

Recorded SHA-256:
`4E880C9E4F0F580E66CEF6C13F05543AC01CDF744D1A29969EF96C776FF66ABA`.

The PDF requires:

- one of its three linked medical-image datasets;
- Faster R-CNN with transfer learning as the baseline;
- YOLOv8 or a newer version as the comparator;
- an unchanged dataset, augmentation policy, and optimizer between detectors,
  with otherwise identical training conditions;
- precision, recall, F1, IoU, Dice, mAP@0.5, and mAP@0.5:0.95;
- FPS, parameters, GFLOPs, GPU memory, training time, inference time, and
  baseline model size;
- darker/brighter images, Gaussian and salt-and-pepper noise, Gaussian and
  motion blur, and JPEG quality 20/50 robustness tests;
- Grad-CAM for good predictions, bad predictions, and failure cases;
- statistical tests with p-values, confidence intervals, and effect sizes;
- deployment discussion and a 12-section final report.

No numeric rubric, deadline, report length, reference style, hardware budget, or
submission mechanism is stated.

## Likely source paper

The brief does not identify its “original paper.” The strongest match found is:

Abdulmajeed Alsufyani, “Performance comparison of deep learning models for
MRI-based brain tumor detection,” *AIMS Bioengineering* 12(1), 2025,
DOI [10.3934/bioeng.2025001](https://doi.org/10.3934/bioeng.2025001).

Why this is likely:

- it uses the PDF's second dataset link;
- it reports 3,903 MR images and a 70/20/10 train/validation/test split;
- it compares YOLOv8, YOLOv9, Faster R-CNN, and ResNet18;
- it discusses efficiency, augmentation, transfer learning, and Grad-CAM.

This is an inference, not a confirmed fact. The paper used different learning
rates, batch sizes, epoch budgets, and optimizers for its models. The assignment
appears to replace that design with a controlled two-detector comparison, but
the instructor should confirm before final experiments.

The paper is internally inconsistent and therefore cannot yet define an exact
reproduction target:

- its dataset section reports 3,903 images in four groups with a 70/20/10
  split; but
- its experimental setup and conclusion report 6,930 train, 1,980 validation,
  and 990 test images (9,900 total) in only three tumor groups.

The paper does not clearly reconcile whether 9,900 is an augmented corpus or a
different dataset/class definition. Confirm the intended counts, classes,
augmentation stage, and split manifests rather than choosing one silently.

## Dataset decision

Provisional recommendation:
[`pkdarabi/medical-image-dataset-brain-tumor-detection`](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection).

Reasons:

- it is the dataset used by the likely source paper;
- its data card describes 3,903 images and explicit train/validation/test
  partitions;
- it is smaller and more feasible than the RSNA challenge dataset;
- the third linked dataset describes only train/validation partitions, so using
  it would require creating a new test split.

Known class ambiguity:

- the likely source dataset describes four image categories: glioma,
  meningioma, pituitary, and no tumor;
- the assignment says to verify five classes;
- `no tumor` may be a negative image condition rather than a valid bounding-box
  class;
- Faster R-CNN also has an internal background label, but that does not prove
  the brief intended five semantic classes.

Do not freeze the class map until the dataset YAML and every annotation are
audited and the instructor confirms the interpretation. Do not manufacture a
box around a no-tumor image.

## Provisional model decision

- Baseline: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` with
  pretrained weights and a task-specific prediction head.
- Comparator: `YOLO26s` pretrained weights, because YOLO26 is the current
  Ultralytics generation as of this context update. The small variant is the
  feasible default for the detected 8 GB local GPU; do not switch scale after
  looking at test results.
- YOLO26 inference head: standard end-to-end one-to-one, NMS-free mode
  (`end2end=True`). Do not switch to the optional one-to-many/NMS path after
  benchmarking begins; it changes outputs, latency, FLOPs, and Grad-CAM
  targeting.
- Predeclared fallback: `YOLO11s` only if a preflight demonstrates that YOLO26
  cannot support the required detection-aware Grad-CAM or reproducible training
  controls. Record the evidence before any final run.

Authoritative implementation references:

- [Torchvision Faster R-CNN documentation](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)
- [Ultralytics model documentation](https://docs.ultralytics.com/models/)
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
- [Grad-CAM paper](https://doi.org/10.1109/ICCV.2017.74)

Pin exact package versions and model artifact hashes before training. Include
Ultralytics AGPL/enterprise licensing in the deployment discussion.

## Two-track experiment strategy

Both tracks share the audited raw data, immutable splits and test set, evaluator,
metric definitions, reporting schema, seeds where applicable, and final
profiling hardware.

- **Track A — assignment-aligned controlled comparison:** hold the dataset,
  preprocessing, shared augmentation corpus, exact optimizer and settings,
  schedule, update budget, effective batch size, checkpoint-selection rule, and
  seeds constant whenever both architectures permit. Record every unavoidable
  model-internal difference.
- **Track B — architecture-optimized comparison:** allow predeclared,
  model-specific optimizer settings, schedules, augmentations, input resolution,
  and training duration under the same tuning-trial or compute budget. Select
  configurations using training and validation data only.

Track A is the assignment-facing primary comparison while it remains valid.
Track B answers which detector performs best when each is used competently. If
Track A is impossible or materially invalid, document that conclusion before
test access and make Track B primary. Never pool, relabel, or selectively omit
the two tracks.

## Available local hardware

Detected on 2026-07-28:

- NVIDIA GeForce RTX 4060 Laptop GPU;
- 8,188 MiB reported VRAM;
- NVIDIA driver 610.47;
- CUDA compute capability reported as 8.9.

Re-detect hardware at run time and store the full environment snapshot with
every result. Effective batch size will likely require gradient accumulation
for Faster R-CNN on this GPU.

## Experiment invariants

After instructor clarification or a documented assumption sign-off, the final
protocol must freeze, version, and log:

- dataset source version, archive hash, file inventory, annotation schema, and
  class map;
- train/validation/test manifests and duplicate/patient grouping;
- deterministic preprocessing and shared/offline augmentations;
- input-size rule, interpolation, grayscale-to-three-channel handling, and
  normalization boundary;
- pretrained initialization, exact optimizer and its settings, scheduler,
  epoch/update budget, effective batch size, gradient accumulation, seeds
  `[17, 42, 2026]`, and checkpoint selection;
- model-native augmentations disabled unless exactly reproduced for both;
- validation-only operating-threshold selection;
- shared COCO-style matching and evaluation conventions;
- corruption parameters and seeds;
- Grad-CAM target score, target layer, normalization, and sample-selection rule;
- statistical unit, resampling plan, tests, effect sizes, confidence level, and
  multiplicity correction;
- device, precision, batch size, warm-up, repetitions, synchronization, and
  inclusion/exclusion of preprocessing and postprocessing for profiling.

The test set is touched only after the protocol, model variants, and thresholds
are frozen.

## Important files

| Path | Purpose | State |
|---|---|---|
| `Final project 1405.v1.pdf` | Authoritative assignment brief | Present |
| `README.md` | Public project overview and high-level roadmap | Present |
| `AGENTS.md` | Stable agent rules and mandatory session-memory workflow | Present |
| `Codex.md` | Durable context, decisions, invariants, and file map | Present |
| `Handoff.md` | Chronological changes, validation, incomplete work, and next action | Present |
| `docs/PROJECT_PLAN.md` | Detailed requirements traceability and execution protocol | Present |
| `docs/DECISION_LOG.md` | Preregistered interpretations, deviations, evidence, risks, and controls | Present |
| `.gitattributes` | Enforces LF text files and treats the assignment PDF as binary | Present |
| `.gitignore` | Prevents data, weights, credentials, and generated artifacts from entering Git | Present |
| `configs/experiment.yaml` | Frozen final experiment contract | Planned |
| `configs/corruptions.yaml` | Frozen corruption types, severities, and seeds | Planned |
| `data/README.md` | Download, license, expected hashes, and local layout | Planned |
| `data/manifests/` | Versioned split and annotation-audit manifests | Planned |
| `src/meddet_benchmark/` | Shared implementation package | Planned |
| `tests/` | Unit, integration, determinism, and evaluator parity tests | Planned |
| `reports/` | Report source plus generated table/figure manifests | Planned |

Update this table as important paths appear, move, or become obsolete.

## Open questions and fallback decisions

Seek instructor clarification when it is available, but do not let silence
stall the project. Freeze these fallbacks before final test access:

1. Treat the AIMS 2025 paper as contextual evidence, not exact authority, unless
   the instructor confirms it.
2. Use the audited 3,903-image linked Kaggle dataset unless its annotations or
   licensing fail the acceptance audit.
3. Derive detector classes only from valid annotated boxes. Keep background
   model-internal and treat `no tumor` as a negative image unless the labels
   prove it is a localized object.
4. Describe 3,903 as the apparent raw count. Treat 9,900 as unverified,
   potentially derived or augmented, and never call it raw without provenance.
5. Use YOLO26s in standard NMS-free mode; activate YOLO11s only after the
   predeclared compatibility preflight fails and the evidence is logged.
6. Resolve “identical training conditions” with the two-track strategy: Track A
   provides the strict controlled result and Track B the architecture-optimized
   result.
7. Until a deadline and compute budget are known, use checkpointed staged runs
   sized for the detected 8 GB GPU and preserve resumability.

## Memory maintenance

At every session start and end, follow `AGENTS.md`.

- Update this file only for durable facts, decisions, invariants, important
  paths, or commands.
- Record what changed, what was validated, and what remains in `Handoff.md`.
- Never treat an old handoff status as ground truth without checking the files
  and Git state.
