# Project plan: medical object detector benchmark

## Reconciliation Note

This document was originally written on 2026-07-28, before the dataset and
model decisions were finalized. It is retained as a reconciled planning record,
not as evidence that every original proposal was implemented unchanged.

The delivered benchmark differs from the draft in three concrete ways:

- **Dataset:** two brain-tumor candidates were investigated, but the benchmark
  ultimately selected the RSNA Pneumonia Detection Challenge 2018, Stage 2.
  The evidence and selection criteria are in
  [DATASET_CHOICE.md](DATASET_CHOICE.md), and the completed audit is in
  [DATASHEET.md](DATASHEET.md).
- **YOLO model:** the draft made YOLO26s primary and YOLO11s a compatibility
  fallback; V1 actually uses Ultralytics YOLO11s.
  The frozen [YOLO configuration](../configs/yolo.yaml) records the delivered
  choice and its stated rationale: YOLO11 is an established anchor-free baseline
  with more independent literature than YOLO26. No additional rationale is
  inferred here.
- **Comparison scope:** the proposed Track B architecture-optimized study was
  descoped and was not run.

Document status: **reconciled V1 implementation record (2026-08-12)**

Original protocol prepared: **2026-07-28 (draft v0.3)**

Requirements baseline: `Final project 1405.v1.pdf`

Implementation checkpoint (2026-08-12): **V1 is complete within its delivered
scope.** Dataset selection and audit, patient-grouped splits, both detector
implementations, three-seed training and clean evaluation, compute profiling,
robustness, explainability, paired statistics, the 12-section report, tests, and
reproduction documentation are present. Track B and the additional analyses
identified below as not attempted are outside the V1 evidence base.

## 1. Aim and research questions

The delivered project compares one Faster R-CNN implementation with one YOLO
implementation on a patient-grouped subset of the RSNA Pneumonia Detection
Challenge. The original recommended brain-MRI scope was provisional and is
superseded by the documented RSNA decision.

Primary research question:

> Under one controlled dataset and evaluation protocol, how do the selected
> Faster R-CNN and YOLO implementations differ in detection quality, efficiency,
> corruption robustness, explanation localization, and deployment suitability?

Secondary questions addressed by V1:

1. Are clean-test performance differences statistically and practically
   meaningful across the three training seeds?
2. Which corruption types and families cause the largest absolute and relative
   degradation for the primary seed checkpoints?
3. Do Grad-CAM maps concentrate on annotated lung-opacity boxes across selected
   true-positive, false-positive, and false-negative cases?
4. How should the measured accuracy, robustness, latency, memory, model size,
   and explanation evidence constrain deployment recommendations?

V1 does not study brain tumors, corrupted-input Grad-CAM changes, clinical
outcomes, or universal one-stage-versus-two-stage superiority. Those boundaries
are stated in [LIMITATIONS.md](LIMITATIONS.md) and the
[final report](../report/report.md).

## 2. Definition of success

For the delivered V1 scope, success means that:

- the fixed data manifests, configs, checkpoints, prediction bundles, tables,
  figures, and documented command chain make every reported number traceable;
- both detectors use the same patient-grouped split, 640 x 640 input, three
  seeds, no stochastic training augmentation, validation-based checkpoint
  selection, unified evaluator, and frozen operating point;
- all requested predictive and compute metrics are explicitly defined;
- robustness covers seven corruption types in four families at five severities;
- Grad-CAM covers paired good predictions, false positives, and shared false
  negatives and includes quantitative localization measures;
- clean and corruption comparisons include paired uncertainty, permutation
  tests, effect sizes, and Holm multiplicity correction;
- the conclusion integrates accuracy, robustness, compute, explainability, and
  deployment evidence; and
- limitations cover dataset quality, leakage/dependence risk, fixed-threshold
  sensitivity, missing external and clinical validation, restricted seed/sample
  scope, and the inability of two variants to establish a universal detector-
  family result.

The original definition also required a separately reported Track A and Track B,
a completed deviation matrix, corrupted-input Grad-CAM, and additional sanity
checks. Those items were not part of delivered V1 and are not silently counted
as complete.

### Outcome-first decision framework

The original priority order remains appropriate: safety, ethics, law, privacy,
and research integrity; scientific validity and reproducibility; assignment
coverage; then convenience. Decisions must not be selected, hidden, or relabeled
after test performance is known.

The draft required every deviation to be preregistered in a versioned decision
log. That record is not present in the current repository. Consequently, V1's
traceable sources of truth are the dataset documents, model and analysis
configs, resolved run records, result artifacts, limitations, and final report.
This reconciliation identifies the missing log rather than claiming that the
original logging requirement was met.

## 3. Requirements traceability

| Requirement | Delivered V1 evidence |
|---|---|
| Literature review | [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) and [references.bib](../report/references.bib) |
| Dataset selection and preparation | [DATASET_CHOICE.md](DATASET_CHOICE.md), [DATASHEET.md](DATASHEET.md), `configs/dataset.yaml`, `data/manifests/`, and `data/splits/` |
| Faster R-CNN transfer learning | `configs/faster_rcnn*.yaml`, `src/models/faster_rcnn_*`, training logs, checkpoints, curves, and [FASTER_RCNN_BASELINE.md](FASTER_RCNN_BASELINE.md) |
| YOLOv8 or newer | `configs/yolo*.yaml`, `src/models/yolo_*`, training logs, checkpoints, curves, and [YOLO_BASELINE.md](YOLO_BASELINE.md) |
| Controlled conditions | Resolved configs and matched-training contracts in `results/logs/`; the remaining architecture-specific differences are disclosed in the report |
| Detection metrics | `src/meddet_benchmark/evaluation.py`, `src/meddet_benchmark/coco_evaluation.py`, `configs/evaluation.yaml`, and `results/tables/detector_comparison*` |
| Compute metrics | Per-seed compute tables, benchmark estimates, and [QUANTITATIVE_COMPARISON.md](QUANTITATIVE_COMPARISON.md) |
| Robustness | `configs/corruptions.yaml`, `results/logs/phase6_robustness/`, robustness tables/figures, and [ROBUSTNESS.md](ROBUSTNESS.md) |
| Grad-CAM | `configs/explainability.yaml`, `results/logs/phase7_explainability/`, Grad-CAM tables/figures, and [EXPLAINABILITY.md](EXPLAINABILITY.md) |
| Statistics | `configs/statistics.yaml`, `src/stats/`, statistical tables, and [STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md) |
| Scientific discussion and limitations | Sections 11-12 of the [final report](../report/report.md) and [LIMITATIONS.md](LIMITATIONS.md) |
| Final report | [report/report.md](../report/report.md), with all 12 required sections |
| Reproducibility | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), `README.md`, configs, environment captures, hashes, and committed evidence |
| Instruction deviations | This reconciliation note records the delivered deviations; a separate decision/deviation log is not present |

Generated evidence lives under `results/`, not the draft's proposed `reports/`
directory. Result summaries link back to configs, seeds, input hashes,
checkpoint hashes, environment captures, and prediction bundles where relevant.

## 4. Interpretation gate as originally written (brain-tumor scope, superseded)

The original gate presupposed a brain-tumor reproduction problem and asked the
project to resolve:

1. whether the intended source was Alsufyani (2025);
2. the source paper's contradictory 3,903-image/four-group and
   9,900-image/three-group descriptions;
3. whether augmentation explained the larger count and which split was
   authoritative;
4. the phrase "five classes" when the likely data described four image
   categories, a model-internal background label, and possibly box-negative
   no-tumor images;
5. whether "three two detectors" meant two detectors;
6. whether YOLO26 end-to-end inference and numerically identical training
   settings were required; and
7. the deadline, submission format, and compute budget.

These questions are preserved because they explain why the brain-tumor
candidates required scrutiny. They do not describe the delivered dataset or
model protocol.

### Resolved

The completed evidence review selected RSNA Stage 2; see
[DATASET_CHOICE.md](DATASET_CHOICE.md). The brain-tumor source-paper count,
augmentation, and "five classes" puzzles therefore do not determine V1. The
delivered RSNA task has one foreground detection class, **Lung Opacity**;
background is model-internal, while Normal and No Lung Opacity / Not Normal are
study-level sampling strata rather than detection classes. The project uses
exactly two detectors. The hardware scope is the recorded RTX 4060 Laptop GPU
(8 GB VRAM), 16 GB system RAM, and i7-13650HX environment.

## 5. Dataset candidates as originally evaluated (superseded)

The following table preserves the draft's historical verdicts. Its final column
is explicitly superseded and must not be read as the delivered selection.

| Candidate | Advantages considered | Conflicts/risks considered | Original verdict (superseded) |
|---|---|---|---|
| RSNA Pneumonia Detection Challenge | Clinically recognized source; large radiograph localization task | Multi-gigabyte download; different anatomy from the provisional brain-MRI framing; not five classes; heavier compute | Not preferred |
| Medical Image DataSet: Brain Tumor Detection | Advertised 3,903 images and 70/20/10 split; appeared to match the likely source paper; manageable scale | Provenance and patient IDs unclear; four image categories; annotation semantics required audit | **Provisional choice** |
| MRI for Brain Tumor with Bounding Boxes | Advertised 5,249 images and four groups; bounding boxes available | Publisher supplied train/validation only; no authoritative test split; provenance and label integrity concerns | Backup only |

### Resolved

The actual selection is **RSNA Pneumonia Detection Challenge 2018, Stage 2**, as
documented in [DATASET_CHOICE.md](DATASET_CHOICE.md). RSNA was selected after
source inspection because it provides a verifiable 26,684-study inventory,
9,555 valid positive boxes, clear task semantics, official provenance and
terms, and an official mapping that enables NIH patient grouping. The two
brain-tumor alternatives could not provide the same combination of auditability,
patient-safe splitting, and authoritative partitioning.

V1 uses a deterministic, hardware-scoped 5,000-study subset from 2,136 NIH
patient groups. The fixed train/validation/test counts are 3,500/750/750, with
no patient-key overlap. The sole foreground class is Lung Opacity. Images are
converted from DICOM and annotations to canonical COCO and YOLO forms; raw data
remain outside Git. The exact audit, split composition, integrity checks,
provenance, and known risks are in [DATASHEET.md](DATASHEET.md), the committed
manifest, and the split CSV files.

The original audit principles remain applicable: verify downloaded counts and
boxes, preserve hashes and provenance, check pairing and duplicates, group by
patient where possible, visualize deterministic samples, validate conversions,
and keep raw data outside version control. V1 completed exact-duplicate and
annotation checks; it did not claim a perceptual near-duplicate sensitivity
analysis beyond the documented audit.

## 6. Models and fairness contract

### Delivered models

- **Faster R-CNN:** Torchvision
  `fasterrcnn_resnet50_fpn_v2`, COCO-pretrained, with its prediction head
  replaced for one foreground class plus background.
- **YOLO:** Ultralytics YOLO11s (`ultralytics==8.4.110`), COCO-pretrained.

The original YOLO26s primary / YOLO11s fallback hierarchy is superseded.
`configs/yolo.yaml` records YOLO11s as an established anchor-free baseline with
more independent literature than YOLO26. Because no current decision-log entry
records the switch, this document does not add a retrospective trigger or claim
that a specific YOLO26 preflight failed.

### Delivered controlled comparison

V1 is one controlled comparison and is not reported as separate Track A and
Track B results. Both detector arms share:

- the exact patient-grouped train, validation, and test manifests;
- one foreground class, grayscale-to-three-channel conversion, and 640 x 640
  inputs;
- COCO-pretrained initialization;
- no stochastic training augmentation or multi-scale training;
- seeds 17, 42, and 137;
- a maximum of 30 epochs, validation each epoch, minimum eight epochs,
  early-stopping patience five, and validation mAP@0.5:0.95 checkpoint
  selection;
- SGD, momentum 0.9, weight decay 0.0005;
- the shared canonical prediction schema, evaluator, score/matching thresholds,
  hardware, and batch-1 AMP profiling protocol.

Architecture/runtime-specific fields are disclosed rather than presented as
identical: Faster R-CNN uses physical batch 2 with two-step gradient
accumulation, float16 AMP, learning rate 0.005, frozen batch-normalization
statistics, gradient clipping, and a plateau scheduler. YOLO11s uses
physical/effective batch 4, bfloat16 forward/backward with float32 loss,
learning rate 0.001, native training batch-normalization statistics, a one-epoch
warm-up, and a constant post-warm-up learning rate. Losses, proposal/head logic,
and postprocessing remain detector components. These differences and the
accepted numerical-stability exceptions are documented in
[LIMITATIONS.md](LIMITATIONS.md) and report section 4.1.

### Track B (descoped from V1)

The draft proposed an architecture-optimized Track B with equal tuning
opportunity, model-specific schedules, resolutions, augmentations, and trial
ledgers. It was not attempted and has no Track B results. No current D-002 entry
exists, so its descoping rationale is not separately logged. Track B must not be
implied by the exploratory or failed run directories preserved outside the
frozen result set.

### Input, augmentation, and training as delivered

The input size is fixed at 640 x 640 in both model configs. The draft's
brain-MRI candidate simulation over `{128, 256, 320, 512, 640}` was not the
selection procedure used for RSNA. Both arms disable stochastic augmentation,
including YOLO HSV, geometric, mosaic, mixup, cutmix, copy-paste,
auto-augmentation, erasing, and multi-scale options. Robustness corruptions are
test-time stressors only and were never used to fine-tune the models.

Both adapters completed smoke/preflight checks and three final training seeds.
The original AdamW/cosine, fixed 100-epoch, equal-effective-batch proposal is
superseded by the frozen SGD-based configs summarized above.

## 7. Shared evaluation protocol

Both detectors write the same canonical per-image prediction representation and
are evaluated by the same implementation.

### Detection matching and operating point

- Official `pycocotools.COCOeval` supplies bounding-box AP at IoU 0.50 and the
  COCO 0.50:0.05:0.95 range.
- Global micro precision, recall, and F1 use a fixed score threshold of 0.25,
  class-aware one-to-one matching at IoU 0.50, and at most 100 detections per
  image.
- COCO AP retains predictions down to score 0.001 and is not computed from the
  0.25-filtered operating-point set.
- Conditional mean IoU is computed only over matched true-positive box pairs;
  conditional box Dice is `2 * IoU / (1 + IoU)` over the same pairs. These
  measures do not penalize missed or extra detections and are not segmentation
  Dice.
- Empty and box-negative images contribute to false-positive/false-negative
  counts as appropriate. Conditional localization is undefined when a detector
  has no matched true positive.

V1 did **not** add model-specific validation-selected thresholds, rasterized
set-IoU/set-Dice, or a separate per-class analysis beyond the sole foreground
class. The fixed threshold and conditional-metric limitations are stated in
[LIMITATIONS.md](LIMITATIONS.md). Raw prediction bundles are preserved for all
three clean-evaluation seeds.

## 8. Computational benchmark

The delivered compute protocol uses the recorded RTX 4060 Laptop GPU, batch-1
AMP inference, 10 warm-up images, 100 timed images, and CUDA synchronization.
It reports parameters, serialized checkpoint size, estimated GFLOPs for
registered operations, FPS, mean/p50/p95 latency, measured training time,
best-checkpoint epoch, and peak allocated training memory. Per-seed evidence is
retained in `results/tables/` and the corresponding run summaries.

The draft's 50 warm-up / 500 measured iteration target, FP32-primary comparison,
and fully separate model-only versus end-to-end timing were not implemented.
Faster R-CNN resizing occurs inside its timed forward, while YOLO resizing is
outside the timed forward-plus-NMS interval. The report therefore treats speed
as deployment-relevant under the measured wrappers, not as a pure architecture
kernel benchmark. No TensorRT/eager-framework mixture is reported.

## 9. Robustness benchmark

Robustness uses only the validation-selected seed-17 checkpoints and a fixed,
stratified 300-image sample from the held-out test split. Each corrupted image
is generated independently from clean input with unchanged geometry and labels.
The committed manifest and `configs/corruptions.yaml` make sampling and
generation deterministic.

The grid contains seven types across four families, each at five ordered
severities (35 corrupted conditions per detector):

| Family | Types | Frozen severity values |
|---|---|---|
| Lighting | darker, brighter | multipliers 0.90-0.50 and 1.10-1.50 |
| Noise | Gaussian, salt-and-pepper | sigma fractions 0.010-0.075; affected fractions 0.0025-0.0400 |
| Blur | Gaussian, motion | sigma 0.5-3.0 pixels; kernels 3-17 pixels |
| Compression | JPEG | quality 90, 70, 50, 35, 20 |

For each condition V1 preserves all seven predictive metrics, absolute clean
change, clean-relative retention, type/severity curves, and four-family
summaries. The same checkpoints, thresholds, NMS, and evaluator are used
throughout; no corrupted test image is used for training. Robustness is not
averaged across the three training seeds, and the 300-image/111-box scope limits
generality, as documented in [ROBUSTNESS.md](ROBUSTNESS.md).

## 10. Explainability plan as delivered

V1 computes ordinary ReLU Grad-CAM for all 111 ground-truth boxes in the same
seed-17, 300-image sample used for robustness. Faster R-CNN has 110 valid maps
because one target produced zero heatmap energy and was excluded under the
declared policy; YOLO11s has 111 valid maps. The matched stride-16, 40 x 40
layers are:

- Faster R-CNN: `backbone.body.layer3`;
- YOLO11s: `model.6`.

For each ground-truth target, the scalar is the post-activation foreground
probability of the retained low-threshold candidate with highest IoU to that
box, using score as a deterministic tie-breaker. An operating-point false
negative therefore uses the best available pre-threshold candidate as an
explicit proxy. The outputs retain target, layer, score, IoU, checkpoint, and
image provenance.

Quantitative measures are energy inside the ground-truth box and pointing-game
success, with the rasterized box-area fraction as a random baseline. Qualitative
panels contain three shared high-IoU true positives, three shared false
positives on box-negative images, and three shared false negatives selected at
proxy-IoU quantiles.

The draft's poor-localization true-positive category, corrupted-input CAM
comparison, parameter-randomization test, and target-label sanity test were not
implemented. Grad-CAM is treated as a coarse, target- and layer-dependent
association map, not evidence of causal reasoning or clinical trustworthiness;
see [EXPLAINABILITY.md](EXPLAINABILITY.md).

## 11. Statistical analysis

The clean analysis covers all 750 test images and paired training seeds 17, 42,
and 137. For each of the seven predictive metrics, the detector difference is
computed within seed and then averaged across paired seeds. Inference uses:

- 2,000 paired hierarchical percentile bootstrap draws over image indices and
  paired seed indices;
- 5,000 two-sided paired image-label permutation draws;
- pointwise 95% bootstrap intervals;
- paired jackknife Cohen's d; and
- Holm correction across the seven clean endpoints.

Dataset-level AP is recomputed from complete prediction bundles in every draw;
the analysis does not average a fictional per-image AP. Precision, recall, F1,
and conditional localization metrics are rebuilt from their sufficient
per-image contributions.

The corruption analysis uses the paired seed-17 predictions on the fixed
300-image sample. It evaluates both raw detector differences and differences in
clean-relative retention for each of 35 conditions, with Holm correction within
each metric-and-estimand family.

The draft's patient-cluster bootstrap, at least 10,000 resamples, formal Track A
primary/Track B secondary split, McNemar test, and Wilcoxon signed-rank tests
were not implemented. Image-level resampling leaves residual within-patient
dependence, and three seeds give only a coarse estimate of training variation.
The exact estimands and results are in
[STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md).

## 12. Implementation sequence and completion status

The original milestone exit criteria are reconciled below. "Partially met"
means that useful delivered work satisfies part of the original criterion but a
named draft-only requirement did not occur.

| Milestone | Exit criterion | Status |
|---|---|---|
| 0 - Clarify and freeze | Dataset, classes, models, hardware, scope, and acceptance criteria are frozen, with every material interpretation recorded before test access; delivered choices are frozen, but the brain-MRI gate was superseded and a complete decision/deviation log is not present | **Partially met** |
| 1 - Reproducible foundation | A clean environment can run lint, unit checks, and the tiny CPU smoke path from locked metadata | **Met** |
| 2 - Dataset pipeline | Source inventory, annotation audit, patient-grouped split, conversion, hashes, and visual checks reconcile or disclose discrepancies | **Met** |
| 3 - Shared evaluator first | Both adapters yield identical metrics for identical canonical toy predictions, including official COCO AP checks | **Met** |
| 4 - Detector adapters and preflight | Both adapters produce canonical predictions and pass data, evaluation, profiling, checkpoint, and Grad-CAM preflight; the delivered single-comparison paths passed, but the proposed two-track preflight was not attempted | **Partially met** |
| 5 - Controlled and optimized training | Both detectors complete three traceable seeds and both proposed tracks are complete; three full seeds per detector exist, but no separate Track B search/results exist | **Partially met** |
| 6 - Full benchmark | Clean evaluation, compute, corruption matrix, Grad-CAM outputs, and paired statistics exist with provenance for the delivered V1 scope | **Met** |
| 7 - Report and reproduction | Generated evidence feeds the 12-section report and a documented reproduction chain, with citations, links, limitations, and consistency checked; no Track A/Track B compliance matrix or recorded full fresh-GPU rerun is claimed | **Partially met** |

## 13. Actual repository architecture

The following tree reflects a fresh 2026-08-12 inventory of the delivered
research repository. Generated/raw data subtrees and repeated per-seed result
files are condensed with descriptive names, but directory names are exact.

```text
.
|-- .github/
|   `-- workflows/ci.yml
|-- .gitattributes
|-- .gitignore
|-- .python-version
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- uv.lock
|-- configs/
|   |-- dataset.yaml
|   |-- faster_rcnn.yaml (+ seed42/seed137 variants)
|   |-- yolo.yaml (+ seed42/seed137 variants)
|   |-- evaluation.yaml
|   |-- corruptions.yaml
|   |-- explainability.yaml
|   |-- statistics.yaml
|   `-- smoke.yaml
|-- data/
|   |-- README.md
|   |-- manifests/rsna-pneumonia-5000-audit.json
|   |-- raw/
|   |-- processed/
|   `-- splits/rsna-pneumonia-5000/
|       |-- train.csv
|       |-- val.csv
|       |-- test.csv
|       `-- test_robustness_seed17_n300.csv
|-- docs/
|   |-- DATASET_CHOICE.md
|   |-- DATASHEET.md
|   |-- FASTER_RCNN_BASELINE.md
|   |-- YOLO_BASELINE.md
|   |-- QUANTITATIVE_COMPARISON.md
|   |-- ROBUSTNESS.md
|   |-- EXPLAINABILITY.md
|   |-- STATISTICAL_ANALYSIS.md
|   |-- LIMITATIONS.md
|   |-- LITERATURE_REVIEW.md
|   |-- REPRODUCIBILITY.md
|   `-- PROJECT_PLAN.md
|-- notebooks/
|-- report/
|   |-- report.md
|   `-- references.bib
|-- results/
|   |-- checkpoints/
|   |-- figures/ (dataset, training, robustness, and Grad-CAM figures)
|   |-- logs/ (training plus phase5-phase8 provenance and predictions)
|   `-- tables/ (clean, compute, robustness, Grad-CAM, and statistics tables)
|-- src/
|   |-- data/
|   |-- models/
|   |-- robustness/
|   |-- explainability/
|   |-- stats/
|   |-- utils/
|   |-- meddet_benchmark/
|   `-- evaluate.py
`-- tests/ (26 test modules)
```

The draft's `reports/` tree was never used; the actual generated evidence root
is `results/`, while the assembled manuscript and bibliography live in
`report/`. A standalone `src/meddet_benchmark/evaluation/` package was not
created; shared evaluation modules live directly under `src/meddet_benchmark/`
with the evaluation entry point at `src/evaluate.py`.

## 14. Final report mapping

The delivered [report/report.md](../report/report.md) uses the exact required
top-level order:

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

The quantitative sections point to their source tables and figures; the
reproduction documentation maps those artifacts to commands, configs, and
stored inputs. The draft requirement to label separate Track A and Track B
results and include a Track compliance/deviation matrix is superseded because
V1 contains only one controlled comparison. The reconciliation note and
methodology disclosure provide the accurate scope record.

## 15. Major risks and mitigations

| Risk | V1 mitigation and remaining limitation |
|---|---|
| Class-count ambiguity | Resolved from the audited annotations: one foreground Lung Opacity class; background is model-internal and study strata are not detection classes |
| Undocumented protocol deviations | Dataset/model choices are traceable to dataset docs and configs, and this reconciliation exposes the missing decision log and Track B; no claim is made that the original preregistration mechanism was completed |
| Patient or duplicate leakage | Official NIH keys enforce patient-disjoint splits and exact-duplicate/annotation checks are recorded; repeated exams within a split remain dependent, and no near-duplicate sensitivity claim is made |
| Framework defaults obscure fairness | Stochastic/native augmentation is disabled, resolved configs are retained, and architecture/training differences are disclosed; no generated two-track config-difference report is claimed |
| Test-set tuning | Checkpoints are selected by validation mAP@0.5:0.95 and the test operating threshold is frozen at 0.25; threshold sensitivity/calibration was not performed |
| Different metric implementations | One model-independent evaluator and official pycocotools AP are used for both detectors |
| Single-seed winner | Clean results use three full training seeds; robustness and explainability remain primary-seed-only and are labeled accordingly |
| Decorative or overclaimed Grad-CAM | Detection-specific targets, matched layers, paired cases, energy-in-box, pointing game, and a random area baseline are retained; corruption and parameter-randomization sanity tests were not run |
| Timing bias | Same device, AMP, batch size, warm-up count, synchronization, and repeated timing summaries are used; wrapper-level resize asymmetry remains disclosed |
| Multiple statistical tests | A declared seven-metric clean family and per-metric/per-estimand corruption families use Holm correction |
| Public-data external validity | The report limits claims to this fixed RSNA subset and rejects clinical deployment without prospective, external-site validation |
| Large artifacts or credentials enter Git | Raw images remain outside Git; manifests, hashes, configs, compact prediction bundles, tables, figures, and documentation provide provenance |
| YOLO licensing affects deployment | No deployment-license clearance analysis was completed; V1 recommendations are research-scenario comparisons and must not be read as legal approval for deployment |
