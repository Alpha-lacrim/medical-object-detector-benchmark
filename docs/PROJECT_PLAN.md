# Project plan: medical object detector benchmark

Document status: **protocol draft v0.1**

Prepared: **2026-07-28**

Authority: `Final project 1405.v1.pdf`

## 1. Aim and research questions

The project will compare one specified Faster R-CNN implementation with one
specified modern YOLO implementation on the medical-image dataset approved
after the interpretation gate. The recommended brain-MRI scope remains
provisional.

Primary research question:

> Under one controlled dataset and evaluation protocol, how do the selected
> Faster R-CNN and YOLO implementations differ in detection quality, efficiency,
> corruption robustness, explanation localization, and deployment suitability?

Secondary questions:

1. Are clean-test performance differences statistically and practically
   meaningful?
2. Which corruption families cause the largest absolute and relative
   degradation?
3. Do Grad-CAM maps concentrate on annotated tumor regions, and does that
   behavior change on failures or corrupted inputs?
4. Does the accuracy/robustness benefit, if any, justify each model's latency,
   memory, size, and licensing constraints?

## 2. Definition of success

The result is complete only when:

- a clean environment can reproduce data manifests, training, evaluation,
  figures, and tables from documented commands;
- both detectors use the same frozen comparison contract except for unavoidable
  architecture-specific components;
- every metric requested by the PDF is reported with an explicit definition;
- robustness includes every required corruption and JPEG quality 20 and 50;
- Grad-CAM covers paired successes and failures for both models;
- uncertainty, paired significance, effect sizes, and multiplicity handling are
  reported;
- the final conclusion uses accuracy, robustness, compute, explanations, and
  deployment evidence; and
- limitations explicitly cover public-dataset quality, leakage risk, missing
  external validation, lack of clinical validation, and the inability of two
  selected variants to establish a universal one-stage-versus-two-stage claim.

## 3. Requirements traceability

| Brief requirement | Planned evidence |
|---|---|
| Literature review | Curated bibliography on the source paper, Faster R-CNN, selected YOLO generation, medical detection, Grad-CAM, robustness, and paired inference |
| Exact dataset preparation | Archive hash, source metadata, file inventory, class map, split manifests, annotation audit, duplicate audit, and sample grid |
| Faster R-CNN transfer learning | Versioned config, pretrained-weight identifier/hash, training logs, curves, checkpoint, and evaluation JSON |
| YOLOv8 or newer | Versioned config, pretrained-weight identifier/hash, training logs, curves, checkpoint, and evaluation JSON |
| Identical conditions | Machine-readable experiment contract plus a generated difference report listing only model-specific fields |
| Detection metrics | Shared evaluator output for precision, recall, F1, IoU, Dice, AP50, and AP50:95, aggregate and per class |
| Compute metrics | Repeated model-only and end-to-end latency/FPS, parameters, FLOPs, peak GPU memory, training time, inference time, and serialized size |
| Robustness | Deterministic corruption manifests, per-severity results, retention ratios, absolute/relative drops, and plots |
| Grad-CAM | Detection-target definition, paired case-selection manifest, heatmaps, box-overlap attention scores, and sanity checks |
| Statistics | Paired intervals/tests, effect sizes, adjusted p-values, and a declared statistical unit |
| Scientific discussion | Evidence-weighted comparison, deployment scenarios, limitations, and future work |
| Final report | Twelve sections in the exact order given by the brief |

### Named acceptance artifacts

The repository must produce these explicit deliverables before submission:

- `reports/literature_review/`: reviewed sources on YOLOv8 or newer, XAI, and
  Grad-CAM, with a traceable bibliography;
- `reports/dataset_report/`: dataset-version report, class-distribution
  table/plot, annotation audit, and labeled random-sample figures;
- `reports/training/faster_rcnn/`: loss, precision, recall, and mAP curves plus
  baseline performance tables and figures;
- `reports/training/yolo/`: performance report, loss/precision/recall/mAP
  curves, and inference-speed evidence;
- `reports/tables/detector_comparison.*`: final detection and computational
  comparison tables;
- `reports/robustness/`: all condition tables, degradation/retention plots, and
  family-balanced summaries;
- `reports/explainability/`: good, bad, and failure-case heatmaps with the case
  manifest and quantitative attention results; and
- `reports/statistics/`: hypotheses, confidence intervals, p-values, effect
  sizes, multiplicity adjustments, and reproducible test output.

Exact file formats will be frozen with the implementation schema. Generated
artifacts must link back to result JSON, configs, seeds, and Git commits.

## 4. Interpretation gate

The PDF has no numeric rubric and contains several inconsistencies. Resolve
these before expensive final runs:

1. Confirm that the intended source is
   [Alsufyani 2025](https://doi.org/10.3934/bioeng.2025001).
2. Resolve the paper's contradiction between:
   - 3,903 images, four groups, and 70/20/10 in its dataset section; and
   - 6,930/1,980/990 images (9,900 total) and three groups in its
     experiment/conclusion.
3. Confirm the selected dataset, exact version, whether augmentation explains
   the 9,900 count, and the authoritative split manifests.
4. Resolve “five classes.” The likely dataset has four image categories; a
   background label is model-internal, and a no-tumor image may have no object.
5. Confirm that the page-8 phrase “three two detectors” means two detectors.
6. Confirm whether YOLO26 in standard end-to-end NMS-free mode is acceptable
   and whether “identical” includes the numeric learning rate, effective batch
   size, and epoch budget.
7. Obtain the deadline, submission format, and available GPU/time budget.

If answers cannot be obtained, preserve the assumptions in a signed-off
decision record and run the scientifically safest interpretation without
claiming that ambiguity was resolved.

## 5. Dataset selection

### Candidate comparison

| Candidate | Advantages | Conflicts/risks | Decision |
|---|---|---|---|
| RSNA Pneumonia Detection Challenge | Clinically recognized source, large X-ray localization task | Multi-gigabyte download, different anatomy/task, not five classes, heavier compute | Not preferred |
| Medical Image DataSet: Brain Tumor Detection | 3,903 images; reported 70/20/10 split; matches likely source paper; manageable | Data provenance and patient IDs unclear; four image categories; annotation semantics need audit | **Provisional choice** |
| MRI for Brain Tumor with Bounding Boxes | 5,249 images; bounding boxes; four reported groups | Data card reports only train/validation partitions; a new test split would violate exact reproduction | Backup only |

### Required audit

Do not trust data-card counts without checking the downloaded version.

1. Record source URL, owner, version/update timestamp, license, archive size, and
   SHA-256.
2. Inventory every image and label, then compare actual and documented counts.
3. Parse all boxes and reject/report:
   - missing image/label pairs;
   - non-finite or out-of-range coordinates;
   - zero/negative-area boxes;
   - unknown class IDs;
   - exact duplicate boxes;
   - unreadable or unexpected image formats.
4. Determine whether `no tumor` is encoded as an empty label, image-level
   category, or bounding box. Do not infer.
5. Compute image and box counts per class and split.
6. Detect exact duplicates using cryptographic hashes and near duplicates using
   perceptual hashes. Check whether clusters cross splits.
7. Use patient/study grouping if identifiers exist. If not, state that
   patient-level leakage cannot be excluded.
8. Create deterministic sample grids with images, boxes, labels, dimensions,
   and split names.
9. Reproduce the confirmed paper dataset/version and split as the primary
   analysis. Use the supplied dataset split only after verifying it is
   identical. If the paper cannot be reconstructed, declare the supplied split
   as a fallback benchmark and do not label it an exact reproduction. If
   leakage is found, retain the approved primary split for traceability and
   report a predeclared grouped sensitivity analysis.
10. Convert once to a canonical COCO-style annotation file; verify round-trip
    conversion against the source boxes.

Raw data remains outside Git. Commit only scripts, metadata, manifests, hashes,
and small license-compatible examples.

## 6. Models and fairness contract

### Models

- Faster R-CNN: Torchvision `fasterrcnn_resnet50_fpn_v2`, COCO-pretrained,
  prediction head replaced for the audited detector classes plus background.
- YOLO: Ultralytics `YOLO26s`, COCO-pretrained.
- YOLO26 head mode: standard end-to-end one-to-one NMS-free inference
  (`end2end=True`), frozen before preflight. The optional one-to-many/NMS path
  is a different configuration and must not be substituted mid-study.
- Compatibility fallback: `YOLO11s`, allowed only if the YOLO26 explainability
  or training-control preflight fails and the reason is recorded before final
  runs.

### Provisional controlled fields

The PDF unambiguously requires the same dataset, augmentation, and optimizer.
Whether every numeric training setting must also match is a blocking instructor
question. The strict, scientifically conservative default below becomes binding
if no answer is available. Once approved, both models must share:

- exact train/validation/test manifests;
- decoded pixels and grayscale-to-three-channel conversion;
- aspect-ratio-preserving resize/letterbox rule and interpolation;
- the same deterministic offline/shared augmentation corpus;
- no additional framework-native augmentation;
- the exact optimizer implementation and settings;
- learning-rate schedule and warm-up definition;
- fixed epoch budget and validation cadence;
- effective batch size, using gradient accumulation if required;
- random seeds;
- pretrained initialization policy;
- checkpoint selection rule based on validation AP50:95;
- stopping rule (prefer a fixed epoch budget; no model-specific early stopping);
- one shared evaluator and operating-point policy; and
- the same final hardware and precision protocol.

Unavoidable differences—loss functions, proposal generation, heads,
architecture-internal normalization, and postprocessing—are detector
components, not experimental controls. List them explicitly in the report.

### Input-size rule

Freeze one shared size after the dataset audit and before model training:

1. Determine the confirmed paper's exact resize/crop behavior. It reports
   approximately 139x132 native images resized to 128x128, but this must be
   reconciled with the downloaded files.
2. Using training annotations only, simulate candidates
   `{128, 256, 320, 512, 640}` with aspect-ratio-preserving letterboxing.
3. Discard a candidate when more than 5% of training boxes would have a
   shortest side below 4 pixels or the median shortest side would be below 16
   pixels after resizing.
4. Choose the smallest remaining candidate supported by both detectors. If no
   candidate passes, use 640 and state the unresolved small-lesion limitation.
5. Record interpolation, padding value, lesion-size distribution, and the
   reason for any deviation from the confirmed paper's 128x128 preprocessing.

This rule uses label geometry rather than test performance and prevents a
resolution that erases small lesions.

### Augmentation

Split first. Generate a deterministic, versioned training-only augmentation
manifest or corpus that both frameworks consume identically. Start with
conservative transforms whose box geometry is unambiguous:

- horizontal flip only if medically justified;
- small rotation/translation/scale with clipped, validated boxes;
- mild brightness/contrast variation; and
- no mosaic, mixup, copy-paste, or framework-specific HSV defaults.

Do not use robustness-test corruptions as uncontrolled training augmentation
unless the protocol explicitly defines a separate robustness-training study.

### Initial training budget

Use two stages:

1. **Preflight:** a tiny subset and 1–2 epochs to verify data parity, finite
   losses, checkpointing, evaluation, profiling, and Grad-CAM end to end.
2. **Final:** at least three fixed seeds, a fixed 100-epoch budget, validation
   each epoch, and the best validation AP50:95 checkpoint per seed.

If compute cannot support three runs, report a single-run comparison as
exploratory and use image-level paired uncertainty without implying
training-seed stability.

The proposed starting optimizer is AdamW with cosine decay and a shared
effective batch size. Freeze exact values only after a training-only/validation
pilot; do not use test results. Any hyperparameter search must use the same
budget and search space for both models.

## 7. Shared evaluation protocol

### Detection matching

- Convert both predictions to the same COCO schema.
- Use class-aware one-to-one matching, highest score first, at IoU >= 0.50 for
  operating-point metrics.
- Use standard COCO IoU thresholds 0.50:0.05:0.95 for AP50:95.
- State maximum detections per image and treatment of empty images.

### Required metrics

- AP50 and AP50:95: shared COCO-style evaluator, macro and per class.
- Precision, recall, F1: report at:
  1. a fixed documented threshold for comparability; and
  2. a model-specific threshold selected once on validation data for a realistic
     operating point.
- IoU:
  - localization-only mean IoU over class-aware matched true-positive box
    pairs, macro-averaged by class and clearly labeled as not penalizing misses;
  - detection-aware set IoU by rasterizing the union of all above-threshold
    predicted rectangles and ground-truth rectangles per image and class, then
    macro-averaging image-class units. False-positive area enlarges the union;
    false-negative area reduces the intersection. Define empty/empty as 1 and
    one-sided empty as 0.
- Dice:
  - detection-aware set Dice on the same per-image/per-class union masks,
    `2 * |P intersect G| / (|P| + |G|)`, with the same empty-set conventions;
  - optional matched-pair box Dice `2 * IoU / (1 + IoU)` as a
    localization-only secondary measure.
  State explicitly that box-mask Dice is not tumor-segmentation Dice.
- Confusion/error analysis: true positives, false positives, false negatives,
  localization errors, class confusions, and empty-image behavior.

Report mean and dispersion across seeds, plus per-class values. Keep raw,
per-image predictions so every statistic is reproducible.

## 8. Computational benchmark

Run on one otherwise idle device with versions and clocks/power mode recorded.

For each model and precision:

1. report parameter count and serialized checkpoint/export size;
2. compute FLOPs with the same input shape and the same counting library,
   acknowledging unsupported operations;
3. reset and record peak allocated GPU memory for training and inference;
4. record wall-clock training time and best-checkpoint epoch;
5. run at least 50 warm-up iterations and 500 measured iterations;
6. synchronize CUDA before and after every timed region;
7. report median, mean, standard deviation, p95 latency, and FPS;
8. measure batch size 1 as primary, with a secondary throughput batch if useful;
9. separate model-only latency from end-to-end decode/preprocess/model/NMS
   latency; and
10. report FP32 primary results and predeclared FP16 deployment results only if
    both models support the same precision path.

Do not mix TensorRT numbers for one detector with eager PyTorch numbers for the
other.

## 9. Robustness benchmark

Apply corruptions only to the frozen test images. Keep geometry and annotations
unchanged. Generate every corrupted image independently from the clean source,
never cumulatively. Use deterministic seeds and save a corruption manifest.

| Family | Required conditions | Prespecified severity proposal |
|---|---|---|
| Brightness | darker, brighter | factors 0.6/0.8 and 1.2/1.4 |
| Gaussian noise | Gaussian | normalized sigma 0.02/0.05/0.10 |
| Impulse noise | salt-and-pepper | pixel probability 0.01/0.03/0.05 |
| Gaussian blur | Gaussian blur | sigma 1/2/3 with recorded kernel |
| Motion blur | motion blur | kernel length 5/9/15, seeded angle |
| JPEG | quality 20%, 50% | exactly Q=20 and Q=50 |

Before freezing, verify on a small training-only sample that severities are
visible but not dominated by implementation artifacts.

For every model, seed, corruption, and severity, report:

- the condition-specific value of every PDF-listed detection metric;
- absolute change from clean;
- relative retention, for example `AP_corrupt / AP_clean`;
- worst-case drop;
- a family-balanced mean: average severities within each corruption type, types
  within each family, and then families equally so families with more
  severities do not dominate; and
- performance-versus-severity curves.

Use the same clean model checkpoints; do not fine-tune on corrupted test data.

## 10. Explainability plan

Grad-CAM for detectors needs a defined scalar target. For each analyzed
detection:

1. target the pre-postprocessing class score associated with a matched or
   selected detection;
2. choose and record the last spatial feature layer appropriate to each
   architecture;
3. upsample and normalize heatmaps using one shared visualization rule;
4. preserve the prediction, ground-truth box, class, score, IoU, checkpoint,
   image hash, and target-layer name.

Build a deterministic paired case manifest from evaluation outputs:

- high-confidence true positives;
- low-confidence or poor-localization true positives;
- false positives;
- false negatives using the highest relevant pre-NMS response when possible;
- corruption-induced failures; and
- the same images for both detectors wherever possible.

In addition to required panels, quantify:

- fraction of positive heatmap energy inside the ground-truth box;
- pointing-game success (heatmap maximum inside the box);
- change in these measures under corruption; and
- model-parameter randomization or target-label sanity checks on a subset.

Do not claim that Grad-CAM proves causal reasoning or clinical trustworthiness.

## 11. Statistical analysis

Declare the image (or patient, if identifiers exist) as the paired evaluation
unit. Preserve per-image predictions and metrics.

Primary estimand:

- the difference in clean-test COCO AP50:95 between the two selected
  implementations, averaged across final training seeds.

Primary inference:

- a stratified paired bootstrap with at least 10,000 image resamples, or patient
  resamples when IDs exist. Resample matched units, keep each unit's complete
  prediction bundle, recompute dataset-level AP for each model/seed, average
  across seeds, and then compute the paired difference and 95% interval;
- a paired permutation test that swaps complete matched prediction bundles and
  recomputes dataset-level AP; and
- a practical effect reported as the absolute AP-point difference and its
  relative change.

Secondary analysis:

- paired intervals/tests for AP50, frozen-threshold F1, robustness retention,
  attention-inside-box score, and other declared outcomes;
- McNemar's test only for a clearly defined paired binary outcome, such as
  whether each image contains at least one correctly localized target;
- Wilcoxon signed-rank for paired continuous per-image scores when its
  assumptions and zero handling are documented; and
- Holm correction within each declared family of secondary tests.

For multiple training seeds, report seed-level variation and use a hierarchical
bootstrap over seeds and images where feasible. AP remains a dataset-level
metric: never calculate or average “per-image AP,” and do not treat thousands of
boxes from one image as independent observations.

Publish the statistical-analysis script, frozen hypotheses, and
machine-readable test results before writing the conclusion.

## 12. Implementation sequence

### Milestone 0 — Clarify and freeze

- Get instructor answers.
- Record hardware/storage budget.
- Pin the repository scope and acceptance criteria.

Exit criterion: no unresolved interpretation can change dataset, classes, or
models.

### Milestone 1 — Reproducible foundation

- Add environment lock and package metadata.
- Add config schema and deterministic seed utilities.
- Add continuous integration for lint, unit tests, and a tiny CPU smoke test.

Exit criterion: a clean environment passes checks without real data.

### Milestone 2 — Dataset pipeline

- Add authenticated/manual download instructions.
- Audit, convert, hash, and visualize the selected dataset.
- Freeze split and augmentation manifests.

Exit criterion: counts and annotations reconcile, or discrepancies are
documented and approved.

### Milestone 3 — Shared evaluator first

- Implement prediction schema, matching, metrics, curves, and test fixtures.
- Validate with hand-calculated toy cases and, where possible, a trusted COCO
  implementation.

Exit criterion: both adapter formats yield identical metrics for identical toy
predictions.

### Milestone 4 — Faster R-CNN baseline

- Implement adapter, training, checkpoint selection, and profiling.
- Complete preflight, then final seeds.

Exit criterion: reproducible logs, curves, checkpoints, predictions, and
baseline tables.

### Milestone 5 — YOLO comparator

- Implement adapter with native augmentation disabled.
- Generate a config-difference report against the baseline contract.
- Complete preflight, then final seeds.

Exit criterion: only detector-specific fields differ.

### Milestone 6 — Full benchmark

- Clean evaluation and compute profiling.
- Corruption matrix.
- Grad-CAM panels and quantitative attention checks.
- Paired statistical analysis.

Exit criterion: every PDF-required metric and artifact exists with provenance.

### Milestone 7 — Report and reproduction

- Generate tables/figures from results.
- Write the 12 required sections.
- Re-run from a clean checkout or container.
- Check citations, limitations, artifact hashes, and result consistency.

Exit criterion: one documented command chain reproduces the submitted evidence.

## 13. Planned repository architecture

```text
.
|-- AGENTS.md
|-- Codex.md
|-- Handoff.md
|-- README.md
|-- Final project 1405.v1.pdf
|-- configs/
|   |-- experiment.yaml
|   `-- corruptions.yaml
|-- data/
|   |-- README.md
|   `-- manifests/
|-- docs/
|   `-- PROJECT_PLAN.md
|-- reports/
|   |-- figures/
|   |-- tables/
|   `-- report/
|-- src/
|   `-- meddet_benchmark/
|       |-- data/
|       |-- models/
|       |-- evaluation/
|       |-- profiling/
|       |-- robustness/
|       |-- explainability/
|       `-- statistics/
`-- tests/
```

Create directories when their first real file is needed; do not fill the
repository with unowned placeholders.

## 14. Final report mapping

Use the exact order required by the brief:

1. Introduction
2. Literature Review
3. Materials and Dataset
4. Experimental Methodology
5. Baseline Faster R-CNN Implementation
6. YOLO Implementation
7. Quantitative Performance Comparison
8. Robustness Evaluation
9. Explainability Analysis
10. Statistical Analysis
11. Discussion
12. Conclusions and Future Work

Each results section must point to the config, result file, seed(s), and commit
that generated its claims.

## 15. Major risks and mitigations

| Risk | Mitigation |
|---|---|
| Class-count ambiguity | Instructor confirmation plus annotation-derived class map |
| Patient or duplicate leakage | Patient grouping when possible; exact/near-duplicate audit; sensitivity analysis |
| “Identical” conditions hide framework defaults | Shared/offline augmentations, disabled native augmentation, generated config diff |
| Test-set tuning | Validation-only thresholds and model selection; freeze before test |
| Different metric implementations | One model-independent evaluator |
| Single-seed winner | At least three seeds or explicitly exploratory claims |
| Decorative Grad-CAM | Detection targets, paired manifest, quantitative localization and sanity checks |
| Timing bias | Same hardware/runtime/precision, warm-up, synchronization, repeated distributions |
| Multiple statistical tests | Predeclared primary outcome and Holm correction |
| Public data lacks external validity | Explicit limitation; no clinical claims |
| Large artifacts or credentials enter Git | `.gitignore`, pre-commit checks, manifests/hashes only |
| YOLO licensing affects deployment | Document AGPL/enterprise terms in deployment analysis |
