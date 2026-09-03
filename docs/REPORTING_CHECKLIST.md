# Reporting Checklist Crosswalk

Audit date: 2026-09-03

Regenerated after the Batch 37 evidence-alignment rewrite of
[`report/paper_draft.md`](../report/paper_draft.md) and refreshed after the
2026-09-02 surgical correction and final submission-readiness wording audit
(SHA-256
`dbd7f97b959a436ecbbee543e9a45cc67cad75bd6479f9ed1e23a97e508c7432`).

This audit is against the **current manuscript**,
[`report/paper_draft.md`](../report/paper_draft.md). It is not an audit of the
historical/full technical report in `report/report.md`. Repository artifacts
are cited when the user-requested evidence rule permits an exact artifact to
support a `Yes` response.

This is an internal reporting crosswalk. It is not an official journal
submission checklist, a reporting-quality score, or a claim of compliance.
An official checklist must be completed in the publisher's required form at
submission and updated against the final paginated manuscript.

## Framework choice and applicability

| Framework | Applicability decision | Use here |
|---|---|---|
| CLAIM 2024 | Directly applicable. The work develops and internally tests two AI object detectors on medical images. | **Primary checklist.** All 44 items use the official `Yes` / `No` / `Not Applicable` response structure. |
| STARD-AI 2025 | Not a clean primary fit. Final STARD-AI covers diagnostic-accuracy studies of an AI-based index test. This benchmark estimates object-localization performance against boxes, allows multiple detections and false positives per image, and does not estimate participant-level diagnostic accuracy against a clinical reference standard. | Selected reporting concepts are mapped **by analogy only**. This is not the official STARD-AI submission checklist and does not establish STARD-AI compliance. |
| TRIPOD+AI 2024 | Outside primary scope. The study does not develop or evaluate an individualized diagnostic or prognostic prediction model that returns a person-level outcome probability. Detector boxes and confidence scores are not such a model. | Selected transparency concepts are mapped **by analogy only**. This is not a TRIPOD+AI compliance assessment. |

## CLAIM 2024 primary crosswalk

The official 2024 update defines 44 items and the three responses `Yes`, `No`,
and `Not Applicable`. For `Yes`, the evidence column names a manuscript section
or exact repository artifact. For `No` and `Not Applicable`, the unresolved or
out-of-scope reason is explicit. CLAIM terminology is followed: the
patient-disjoint held-out split is **internal testing**; this study has no
**external testing** dataset.

| Item | Reporting topic (paraphrased) | Response | Evidence or explanation |
|---:|---|---|---|
| 1 | Identify the work as AI methodology and name the technology category in the title or abstract | Yes | The [structured abstract](../report/paper_draft.md#abstract) explicitly identifies two deep-learning object-detection pipelines and names Faster R-CNN and YOLO11s. |
| 2 | Structured abstract with design, population, partitions, retrospective/prospective status, statistics, outcomes, implications, and availability | No | The [structured abstract](../report/paper_draft.md#abstract) now reports retrospective internal testing, cohort and partition counts, the primary estimand, absolute and comparative outcomes, and bounded implications. It still omits a release-ready software/data/model availability statement because those author-controlled release details remain unresolved. |
| 3 | Scientific/clinical background, intended use, current practice, and rationale | Yes | [`paper_draft.md` §1](../report/paper_draft.md#1-introduction) and [§5.6](../report/paper_draft.md#56-scope-correct-synthesis) define the scientific comparison, intended research use, and non-clinical boundary. |
| 4 | A priori aims, objectives, and hypotheses | No | The aim is explicit, but H1--H5 were recorded retrospectively and H6 is a retrospective descriptive question; see [`paper_draft.md` §1](../report/paper_draft.md#1-introduction) and [`HYPOTHESIS_TRACEABILITY.md`](HYPOTHESIS_TRACEABILITY.md). |
| 5 | State whether the study is prospective or retrospective | Yes | The [abstract](../report/paper_draft.md#abstract) plainly identifies a retrospective internal-testing design, and [§3.11](../report/paper_draft.md#311-retrospective-hypothesis-and-reporting-status) distinguishes retrospective hypotheses from preregistration. |
| 6 | State the study goal, modeling task, target, and intended role | Yes | [`paper_draft.md` §§1 and 3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) define a controlled, non-clinical lung-opacity object-localization benchmark. |
| 7 | Identify data sources and their match to the intended population | Yes | [`paper_draft.md` §§2--3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) identifies the RSNA/NIH source and limits the intended inference. |
| 8 | Eligibility, setting, location, dates, demographics, and sampling method | No | Source, subset, strata, and counts are reported, but source accrual dates, detailed eligibility, and demographics are unavailable; the cohort cannot be characterized as consecutive, random clinical sampling. |
| 9 | Preprocessing steps | Yes | The canonical-preprocessing subsection in [`paper_draft.md` §3.1](../report/paper_draft.md#canonical-preprocessing), [`configs/dataset.yaml`](../configs/dataset.yaml), and [`docs/DATASHEET.md`](DATASHEET.md) record conversion, polarity inversion, finite min-max scaling, canonical annotations, and image format. |
| 10 | Selection of data subsets | Yes | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split), [`configs/dataset.yaml`](../configs/dataset.yaml), and [`docs/DATASET_CHOICE.md`](DATASET_CHOICE.md) record seed-17 SHA-256 ordering, label-stratum tracking, patient grouping, and the hardware-scoped 5,000-study selection. |
| 11 | De-identification method | No | Public-source identifiers were used for grouping, but the upstream de-identification process and any local verification procedure are not documented in the manuscript or repository evidence. |
| 12 | Missing-data assessment and handling | No | Box-coordinate integrity and conversion completeness were audited, but the manuscript does not give a general missing-data assessment or handling rule for unavailable source clinical/acquisition variables. |
| 13 | Image-acquisition protocol and equipment detail | No | [`paper_draft.md` §§3.8 and 6](../report/paper_draft.md#38-digital-corruption-and-radiography-motivated-synthetic-sensitivity) explicitly say acquisition fields, scanner settings, and processing history needed for reproducibility are unavailable. |
| 14 | Reference-standard definition and labeling instructions | No | The RSNA challenge boxes and one-class mapping are identified, but the manuscript does not reproduce the original reader instructions or full adjudication procedure. |
| 15 | Rationale for the reference standard and assessment of possible errors | Yes | [`docs/DATASET_CHOICE.md`](DATASET_CHOICE.md) records why RSNA was selected, and [`paper_draft.md` §6](../report/paper_draft.md#6-limitations) describes coarse-box, disagreement, and absent re-reading limitations. |
| 16 | Annotation sources, number and qualifications of annotators, and instructions | No | [`docs/DATASHEET.md`](DATASHEET.md) records 18 board-certified radiologists from 16 institutions, including 12 thoracic specialists, but the full instructions and case-assignment/adjudication detail are not reported locally. |
| 17 | Annotation procedure and software for the internal-testing set | No | The source boxes were reused; local conversion is traceable, but original annotation software, display conditions, and internal-testing-case labeling procedure are unavailable. |
| 18 | Inter- and intrarater variability | No | No local rereading study or source-level inter/intrarater agreement statistic is available. |
| 19 | Partition assignment, sizes, proportions, differences, and class imbalance | Yes | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split), the exact [`data/splits/rsna-pneumonia-5000/`](../data/splits/rsna-pneumonia-5000/) CSV manifests, and [`data/manifests/rsna-pneumonia-5000-audit.json`](../data/manifests/rsna-pneumonia-5000-audit.json) give the patient-grouped 70/15/15 allocation and stratum counts. |
| 20 | Disjointness level between partitions | Yes | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) reports empty NIH patient-key intersections; [`data/manifests/rsna-pneumonia-5000-audit.json`](../data/manifests/rsna-pneumonia-5000-audit.json) is the exact audit artifact. |
| 21 | Internal-testing sample size and how it was determined | Yes | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) reports 750 internal-testing studies/323 patient groups and explicitly states that the 5,000-study cohort was hardware-scoped with no formal sample-size or power calculation. The dedicated row below gives the complete interpretation. |
| 22 | Sufficient model detail to reconstruct inputs, outputs, and architecture | Yes | [`paper_draft.md` §3.2](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), [`src/models/faster_rcnn_model.py`](../src/models/faster_rcnn_model.py), [`src/models/yolo_training.py`](../src/models/yolo_training.py), and [`configs/`](../configs/) identify the implemented pipelines and configuration-derived outputs. |
| 23 | Software versions and hardware | Yes | [`paper_draft.md` §§3.2 and 3.7](../report/paper_draft.md#37-compute-and-pareto-analysis), [`uv.lock`](../uv.lock), and [`results/logs/phase5_evaluation/summary.json`](../results/logs/phase5_evaluation/summary.json) record versions and the measured system. |
| 24 | Parameter initialization | Yes | [`paper_draft.md` §3.2](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), [`configs/faster_rcnn.yaml`](../configs/faster_rcnn.yaml), and [`configs/yolo.yaml`](../configs/yolo.yaml) record COCO initialization and the canonical seed; seed-specific configs and model-loader code record the remaining seeds and head adaptation. |
| 25 | Training, augmentation, stopping, hyperparameters, objectives, and frozen parameters | Yes | [`paper_draft.md` §3.2](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), [`configs/faster_rcnn.yaml`](../configs/faster_rcnn.yaml), [`configs/yolo.yaml`](../configs/yolo.yaml), and per-run `resolved_config.json` files indexed in [`SUPPLEMENTARY.md` S2](SUPPLEMENTARY.md#s2-full-clean-seed-level-comparison) provide the exact record. |
| 26 | Final-model selection | Yes | [`paper_draft.md` §§3.2--3.3](../report/paper_draft.md#33-seedrun-design-and-unified-internal-testing) reports model-optimization-split mAP selection before frozen internal testing; per-run checkpoint metadata are indexed in [`SUPPLEMENTARY.md` S2](SUPPLEMENTARY.md#s2-full-clean-seed-level-comparison). |
| 27 | Ensembling method | Not Applicable | Neither arm is an ensemble; each reported run is one Faster R-CNN or YOLO11s checkpoint. Seed summaries quantify repeated training and do not combine predictions into an ensemble. |
| 28 | Performance metrics, rationale, and comparison with prior models | No | [`paper_draft.md` §§3.3--3.5](../report/paper_draft.md#33-seedrun-design-and-unified-internal-testing) defines the metrics and scientific rationale, but prior publications are contextual only and are not directly reproduced under this study's split/evaluator for quantitative comparison. |
| 29 | Uncertainty/significance procedures and statistical software | Yes | [`paper_draft.md` §3.10](../report/paper_draft.md#310-inferential-targets-and-patient-cluster-protocol), [`docs/STATISTICAL_ANALYSIS.md`](STATISTICAL_ANALYSIS.md), [`results/tables/statistical_clean_comparison.csv`](../results/tables/statistical_clean_comparison.csv), and [`results/logs/phase8_statistics/summary.json`](../results/logs/phase8_statistics/summary.json) separate training-procedure intervals from checkpoint-conditional permutation tests and record seeds/draw counts. |
| 30 | Robustness or sensitivity analyses | Yes | [`paper_draft.md` §§3.8 and 4.6--4.7](../report/paper_draft.md#38-digital-corruption-and-radiography-motivated-synthetic-sensitivity) and [`docs/ACQUISITION_SHIFTS.md`](ACQUISITION_SHIFTS.md) report digital corruptions and bounded radiography-motivated synthetic sensitivities. |
| 31 | Explainability methods, parameters, and validation/sanity checks | Yes | [`paper_draft.md` §§3.9 and 4.8](../report/paper_draft.md#39-grad-cam-localization-and-sanity-checks), [`docs/EXPLAINABILITY.md`](EXPLAINABILITY.md), and [`docs/XAI_SANITY.md`](XAI_SANITY.md) record layers, targets, localization measures, and randomization controls. |
| 32 | Evaluation on internal data | Yes | [`paper_draft.md` §§3.3 and 4.1](../report/paper_draft.md#33-seedrun-design-and-unified-internal-testing) identifies the patient-disjoint held-out evaluation as internal testing and preserves its detector/seed-specific sample sizes. |
| 33 | External testing, or an explicit absence and rationale | No | No external testing dataset exists. [`paper_draft.md` §6](../report/paper_draft.md#6-limitations) discloses the gap, but disclosure of absence is not external testing. |
| 34 | Registration information for a clinical trial | Not Applicable | This is a retrospective computational benchmark with no participant enrollment or intervention and is not a clinical trial. The absence of preregistration remains separately disclosed. |
| 35 | Numbers included/excluded at each stage and a flow diagram | No | Source and split counts are available, but the manuscript has no participant/study flow diagram and does not present every source-to-analysis exclusion in a single flow. |
| 36 | Demographic and clinical characteristics by partition | No | The released/local cohort evidence lacks the demographic variables required for partition-wise descriptive reporting. |
| 37 | Final performance for all partitions and relevant subgroups | No | Final internal-testing performance is reported, but no demographic/clinical subgroup evaluation exists and model-optimization results are not presented as a full subgrouped performance table. |
| 38 | Diagnostic-classification performance, uncertainty, calibration, and subpopulations | Not Applicable | The primary task is object localization, not participant-level diagnostic classification. Detection AP/FROC/uncertainty and emitted-detection D-ECE are reported, but they must not be relabeled as patient-level diagnostic sensitivity/specificity or clinical-risk calibration. |
| 39 | Failure analysis | Yes | [`paper_draft.md` §§4.1--4.2, 4.6--4.8, and 5.5](../report/paper_draft.md#41-clean-internal-testing-performance-n5-per-detector-conditional-localization-n5n4) covers seed-271 score compression, operating-regime sensitivity, corruptions, DICOM evidence boundaries, and XAI failure patterns. |
| 40 | Study limitations | Yes | [`paper_draft.md` §6](../report/paper_draft.md#6-limitations) and [`docs/LIMITATIONS.md`](LIMITATIONS.md) give data, sampling, model, statistical, robustness, XAI, and governance limitations. |
| 41 | Implications and intended role | Yes | [`paper_draft.md` §§5.6--6](../report/paper_draft.md#56-scope-correct-synthesis) limits conclusions to controlled pipeline trade-offs and prohibits clinical use. |
| 42 | Protocol and additional technical detail availability | Yes | [`README.md`](../README.md), [`SUPPLEMENTARY.md`](SUPPLEMENTARY.md), configuration files, run summaries, and source commands provide the reproducibility record. |
| 43 | Software, model, and data availability statement | No | [`paper_draft.md` §8](../report/paper_draft.md#8-declarations) contains an explicit author-action placeholder, not a release-ready statement. A public/archive URL, version, data-access terms, and public checkpoint status remain unresolved; see [`AUTHOR_DECLARATIONS_TODO.md`](AUTHOR_DECLARATIONS_TODO.md). |
| 44 | Funding/support and the funder's role | No | These author-controlled facts are unknown and must not be inferred. [`paper_draft.md` §8](../report/paper_draft.md#8-declarations) and [`AUTHOR_DECLARATIONS_TODO.md`](AUTHOR_DECLARATIONS_TODO.md) preserve explicit author-action placeholders. |

## Required sample-size and subset-selection row

| Reporting point | Current disclosure | Consequence |
|---|---|---|
| Source, selected size, method, formal calculation, and generalizability | The source has 26,684 labeled studies. A deterministic, seed-17, SHA-256-ordered procedure selected 5,000 studies while tracking three label strata and keeping complete NIH patient groups together; it then assigned 3,500/750/750 studies to training/model optimization/internal testing. The other 21,684 studies were excluded for the stated 8-GB-VRAM laptop compute scope, not because they failed the annotation audit. **No formal statistical sample-size or power calculation existed.** Evidence: [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split), [`configs/dataset.yaml`](../configs/dataset.yaml), [`DATASET_CHOICE.md`](DATASET_CHOICE.md), and [`DATASHEET.md`](DATASHEET.md). | Determinism, stratification, and patient grouping support reproducibility and leakage control, but do not establish adequate power, prevalence representativeness, or equivalence to the complete challenge cohort. Sampling uncertainty is larger and generalizability is limited before any external-population shift is considered. |

## STARD-AI 2025 selected reporting crosswalk by analogy

The source is the **final version of record**, not the 2020 announcement or
2021 protocol: Sounderajah, Guni, Liu, et al., *The STARD-AI reporting guideline
for diagnostic accuracy studies using artificial intelligence*, *Nature
Medicine* 31, 3283--3289 (2025), DOI
[10.1038/s41591-025-03953-8](https://doi.org/10.1038/s41591-025-03953-8).
The final paper describes a minimum reporting set for AI-centered diagnostic
test-accuracy studies.

This manuscript is not sufficiently diagnostic-accuracy-centered for an
official STARD-AI completion: it does not define a participant-level index-test
result, clinical diagnostic target, diagnostic threshold, or 2-by-2 reference-
standard comparison. The following are selected useful concepts only.

| STARD-AI concept used by analogy | Current mapping | Boundary/gap |
|---|---|---|
| Data source, eligibility, sampling, and participant flow | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) reports source and split counts. | Accrual dates, detailed eligibility, demographics, and a flow diagram are unavailable. |
| Reference standard | Challenge bounding boxes are described in [`paper_draft.md` §§2--3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split). | Boxes are object-localization annotations, not a participant-level clinical diagnosis; original reader instructions and local agreement assessment are unavailable. |
| Internal versus external testing | Patient-disjoint held-out evaluation is explicitly called internal testing in [`paper_draft.md` §3.3](../report/paper_draft.md#33-seedrun-design-and-unified-internal-testing). | No external testing was performed. |
| Sample-size rationale | The dedicated row above reports the hardware-driven 5,000-study subset and absence of formal calculation. | This is not a diagnostic-accuracy power calculation. |
| Uncertainty, indeterminate outputs, and failures | [`paper_draft.md` §§3.10, 4.1, and 4.9](../report/paper_draft.md#310-inferential-targets-and-patient-cluster-protocol) reports intervals, the zero-detection YOLO run, and endpoint-specific undefinedness. | Multiple boxes/false positives and conditional localization do not reduce to one diagnostic result per participant. |
| Generalizability and clinical role | [`paper_draft.md` §§5.6--6](../report/paper_draft.md#56-scope-correct-synthesis) explicitly limits use. | No clinical accuracy, prospective benefit, fairness, or external-site evidence exists. |

## TRIPOD+AI 2024 selected reporting crosswalk by analogy

Source: Collins, Moons, Dhiman, et al., *TRIPOD+AI statement: updated guidance for
reporting clinical prediction models that use regression or machine learning
methods*, *BMJ* 385:e078378 (2024), DOI
[10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378).
TRIPOD+AI addresses development and evaluation of models that combine
predictors to estimate an individualized outcome. This study instead evaluates
object detections; it is not primarily a clinical prediction-model study.

| TRIPOD+AI concept used by analogy | Current mapping | Boundary/gap |
|---|---|---|
| Clear objectives and analysis populations | [`paper_draft.md` §§1 and 3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) states the comparison and partitions. | Hypotheses are retrospective, not preregistered. |
| Model specification and reproducibility | [`paper_draft.md` §§3.2--3.3](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), configs, source, and [`SUPPLEMENTARY.md`](SUPPLEMENTARY.md) record the pipelines. | Outputs are boxes/scores, not individualized diagnostic/prognostic probabilities. |
| Sample size and missing data | The dedicated row above reports the subset; annotation/conversion audits are traceable. | No formal sample-size calculation and no general clinical-variable missing-data analysis exist. |
| Model performance and calibration | Detection metrics, uncertainty, and emitted-detection D-ECE are in [`paper_draft.md` §§3.3, 3.6, and 4](../report/paper_draft.md#33-seedrun-design-and-unified-internal-testing). | D-ECE is not exam-level outcome-probability calibration; decision-curve analysis was correctly not performed. |
| Open science, declarations, and patient/public involvement | Code/config/artifact provenance is indexed in [`README.md`](../README.md) and [`SUPPLEMENTARY.md`](SUPPLEMENTARY.md); author-action placeholders are in [`paper_draft.md` §8](../report/paper_draft.md#8-declarations). | A release-ready availability statement and author-controlled declarations remain unresolved; no PPI fact may be assumed. |

## Submission blockers and linked audits

- Resolve all `AUTHOR ACTION REQUIRED` fields in [`paper_draft.md` §8](../report/paper_draft.md#8-declarations)
  and [`AUTHOR_DECLARATIONS_TODO.md`](AUTHOR_DECLARATIONS_TODO.md).
- Use [`HYPOTHESIS_TRACEABILITY.md`](HYPOTHESIS_TRACEABILITY.md) to keep every
  H1--H6 statement retrospective and tied to its exact endpoint/artifact.
- Use [`CITATION_AUDIT.md`](CITATION_AUDIT.md) for the exhaustive methodological,
  standards, dataset, DICOM, statistical, calibration, DCA, and XAI source audit.
- Re-complete the journal's official checklist after formatting the final
  submission. This crosswalk must not be submitted as proof of compliance.

## Authoritative guideline sources

- Ali S. Tejani, Michail E. Klontzas, Anthony A. Gatti, John T. Mongan,
  Linda Moy, Seong Ho Park, and Charles E. Kahn Jr., for the CLAIM 2024 Update
  Panel. *Checklist for Artificial Intelligence in Medical Imaging (CLAIM):
  2024 Update*. *Radiology: Artificial Intelligence* 6(4), e240300 (2024).
  DOI [10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300); PMID 38809149.
- Viknesh Sounderajah, Ahmad Guni, Xiaoxuan Liu, et al., for the STARD-AI
  Steering Committee. *The STARD-AI reporting guideline for diagnostic
  accuracy studies using artificial intelligence*. *Nature Medicine* 31,
  3283--3289 (2025). DOI
  [10.1038/s41591-025-03953-8](https://doi.org/10.1038/s41591-025-03953-8);
  PMID 40954311. The publisher page includes the July 2026 author correction.
- Gary S. Collins, Karel G. M. Moons, Paula Dhiman, et al. *TRIPOD+AI statement: updated
  guidance for reporting clinical prediction models that use regression or
  machine learning methods*. *BMJ* 385:e078378 (2024). DOI
  [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378).
