# Reporting Checklist Crosswalk

**Repository audit date:** 2026-08-28
**Current manuscript:** [12-section technical report](../report/report.md); the
Batch 24 IMRaD paper draft does not yet exist.

This is an evidence-location audit, not a reporting-quality score or a claim of
clinical readiness. The applicable sources are the
[CLAIM 2024 update](https://doi.org/10.1148/ryai.240300),
[TRIPOD+AI 2024 statement](https://doi.org/10.1136/bmj-2023-078378) and its
[official expanded checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf),
and the [STARD-AI 2025 statement](https://doi.org/10.1038/s41591-025-03953-8).
The 2026 STARD-AI author correction added an omitted steering-committee member;
it did not change the checklist.

Status is deliberately strict:

- `[x] Complete` means the material elements of the item are supported by a
  named repository file or generated table.
- `[ ] Partial` means some evidence exists, but at least one requested element
  is absent.
- `[ ] Not reported` means no traceable repository statement was found.
- `[ ] N/A` means the item does not apply to this retrospective two-model
  object-detection benchmark; it remains unchecked.

The crosswalk applies prediction-model and diagnostic-accuracy guidance by
analogy where useful. This project predicts boxes and scores for a radiographic
finding; it is not a patient-level clinical-risk model, diagnostic device
study, prospective accuracy study, or clinical trial.

## CLAIM 2024

| Item | Audited topic | Status | Traceable repository evidence or gap |
|---:|---|---|---|
| 1 | AI method identified in title or abstract | [ ] Partial | [Report title](../report/report.md) names Faster R-CNN, YOLO11s, and lung-opacity detection, but does not explicitly say AI/deep learning and there is no abstract. |
| 2 | Structured abstract | [ ] Not reported | [Current report](../report/report.md) begins with provenance text and Section 1; no abstract section exists. |
| 3 | Background, intended use, and role | [x] Complete | [Report Sections 1–2 and 11](../report/report.md), [literature review](LITERATURE_REVIEW.md), and [limitations](LIMITATIONS.md) define the scientific gap, scenarios, and non-clinical role. |
| 4 | Aims, objectives, and hypotheses | [ ] Partial | The research question is in [report Section 1](../report/report.md) and [HYPOTHESES.md](HYPOTHESES.md), but that file explicitly records H1–H6 as retrospective rather than preregistered. |
| 5 | Prospective or retrospective design | [x] Complete | [Report Sections 1 and 11.4](../report/report.md), [datasheet](DATASHEET.md), and the [raw-score utility correction](DCA_ANALYSIS.md) identify a retrospective benchmark. |
| 6 | Study goal and intended task | [x] Complete | [Report Section 1](../report/report.md), [datasheet summary](DATASHEET.md), and [deployment limits](LIMITATIONS.md) define controlled lung-opacity detection, not pneumonia diagnosis. |
| 7 | Data sources and fit to intended population | [x] Complete | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), and [dataset-choice record](DATASET_CHOICE.md) identify RSNA Stage 2, source links, rationale, and representativeness limits. |
| 8 | Inclusion and exclusion criteria | [ ] Partial | [Datasheet](DATASHEET.md) identifies Stage 2 eligibility, the deterministic 5,000-study selection, and 21,684 compute-scope exclusions, but participant dates and demographic restrictions are unavailable. |
| 9 | Data preprocessing | [x] Complete | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), and [`src/data/prepare.py`](../src/data/prepare.py) trace DICOM decoding, MONOCHROME1 inversion, finite min–max scaling, 8-bit PNG output, COCO conversion, and checks. |
| 10 | Selection of data subsets | [x] Complete | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), [robustness config](../configs/corruptions.yaml), and [XAI sanity config](../configs/xai_sanity.yaml) define the 5,000-, 300-, and nested 50-image selections. |
| 11 | De-identification method | [ ] Not reported | [Datasheet access terms](DATASHEET.md) prohibit re-identification and the repository omits pixels, but the source cohort's de-identification procedure is not described. |
| 12 | Missing-data handling | [x] Complete | [Dataset audit](../data/manifests/rsna-pneumonia-5000-audit.json) and [datasheet integrity checks](DATASHEET.md) report zero invalid selected studies/boxes, expected blank negative-box fields, and no imputation. |
| 13 | Image-acquisition protocol | [ ] Partial | [Acquisition-shift audit](ACQUISITION_SHIFTS.md) and [its config](../configs/acquisition_shifts.yaml) verify sampled files as 8-bit unsigned CR/MONOCHROME2 without native VOI/modality transforms, but manufacturer and original exposure protocol are unavailable. |
| 14 | Reference-standard definition | [ ] Partial | [Datasheet](DATASHEET.md), [dataset audit](../data/manifests/rsna-pneumonia-5000-audit.json), and [preparation code](../src/data/prepare.py) define the challenge boxes and negative studies, but do not reproduce the original annotator instructions. |
| 15 | Rationale for reference standard | [ ] Partial | [Dataset choice](DATASET_CHOICE.md) and [datasheet](DATASHEET.md) justify the expert-box source and describe coarse-box uncertainty, but do not compare alternative reference-standard procedures. |
| 16 | Annotation source and annotator qualifications | [x] Complete | [Datasheet collection section](DATASHEET.md) records 18 board-certified radiologists, 16 institutions, 12 thoracic specialists, triple reading/adjudication counts, and primary source links. |
| 17 | Test-set annotation procedure | [ ] Partial | [Datasheet](DATASHEET.md) traces reuse of existing Stage 2 boxes and label conversion, but original test-annotation instructions and annotation-software version are not present. |
| 18 | Inter- and intrarater variability | [ ] Not reported | [Datasheet](DATASHEET.md) gives triple-read/adjudication counts but no inter- or intrarater variability estimate or mitigation analysis. |
| 19 | Assignment to train, tuning, and test partitions | [x] Complete | [Dataset config](../configs/dataset.yaml), [audit](../data/manifests/rsna-pneumonia-5000-audit.json), and [train](../data/splits/rsna-pneumonia-5000/train.csv), [validation](../data/splits/rsna-pneumonia-5000/val.csv), and [test](../data/splits/rsna-pneumonia-5000/test.csv) manifests preserve the 70/15/15 assignment. |
| 20 | Disjointness level | [x] Complete | [Datasheet](DATASHEET.md) and [audit](../data/manifests/rsna-pneumonia-5000-audit.json) document NIH-patient-level grouping and zero overlap across all splits. |
| 21 | Intended testing sample size | [x] Complete | [PROJECT_SPEC §3](../PROJECT_SPEC.md), [dataset config](../configs/dataset.yaml), and [datasheet](DATASHEET.md) identify the hardware/time-scoped 5,000-study cohort and 750-study test partition; no formal power calculation is claimed. |
| 22 | Model architecture and modifications | [x] Complete | [Faster R-CNN protocol](FASTER_RCNN_BASELINE.md), [YOLO protocol](YOLO_BASELINE.md), their [`configs/`](../configs), and [`src/models/`](../src/models) identify architectures, adapters, heads, inputs, and modifications. |
| 23 | Software versions and hardware | [x] Complete | [Pinned requirements](../requirements.txt), [lockfile](../uv.lock), [reproducibility record](REPRODUCIBILITY.md), run-level environment logs, and [README](../README.md) specify versions and the RTX 4060 Laptop/i7/16-GB host. |
| 24 | Parameter initialization | [x] Complete | [Faster R-CNN protocol](FASTER_RCNN_BASELINE.md), [YOLO protocol](YOLO_BASELINE.md), configs, and checkpoint hashes identify COCO transfer initialization and the pinned starting weights. |
| 25 | Training procedures and hyperparameters | [x] Complete | [Model protocols](FASTER_RCNN_BASELINE.md), [YOLO protocol](YOLO_BASELINE.md), per-seed configs, and [`src/models/`](../src/models) trace augmentation, optimizer, precision, batch, schedules, early stopping, and asymmetries. |
| 26 | Final-model selection | [x] Complete | [Report Sections 4–6](../report/report.md), model protocols, training summaries, and validation tables record validation mAP@0.5:0.95 selection and selected epochs. |
| 27 | Ensembling | [ ] N/A | Each detector/seed is evaluated separately; [evaluation config](../configs/evaluation.yaml) contains no ensemble. |
| 28 | Performance metrics and rationale | [x] Complete | [Quantitative comparison](QUANTITATIVE_COMPARISON.md), [`src/evaluate.py`](../src/evaluate.py), calibration, FROC, robustness, and explainability documents define the endpoints and scope; the [historical raw-score calculation](DCA_ANALYSIS.md) is explicitly excluded from standard DCA interpretation. |
| 29 | Statistical uncertainty and significance | [x] Complete | [Statistics config](../configs/statistics.yaml), [method](STATISTICAL_ANALYSIS.md), [`src/stats/`](../src/stats), and [clean table](../results/tables/statistical_clean_comparison.csv) trace patient-cluster CIs, permutation p-values, effects, and Holm adjustment. |
| 30 | Robustness or sensitivity analysis | [x] Complete | [Digital-corruption grid](ROBUSTNESS.md), [complete table](../results/tables/robustness_results.csv), [acquisition shifts](ACQUISITION_SHIFTS.md), threshold sensitivity, and seed-stability artifacts are complete. |
| 31 | Explainability method and validation | [x] Complete | [Explainability protocol](EXPLAINABILITY.md), [`src/explainability/`](../src/explainability), [localization data](../results/tables/gradcam_localization_per_target.csv), and [sanity checks](XAI_SANITY.md) trace targets, layers, metrics, and randomization tests. |
| 32 | Internal-data evaluation | [x] Complete | [Evaluation config](../configs/evaluation.yaml), [per-seed test table](../results/tables/detector_comparison_per_seed.csv), and [report Section 4.2](../report/report.md) document held-out same-source testing after checkpoint freeze. |
| 33 | External-data testing | [x] Complete | No external test was performed; that fact and its consequence are explicit in [limitations](LIMITATIONS.md) and [report Sections 11.4–12](../report/report.md). |
| 34 | Clinical-trial registration | [ ] N/A | This is a retrospective software benchmark with no intervention or participant enrollment. |
| 35 | Included/excluded patient or examination flow | [ ] Partial | [Datasheet](DATASHEET.md) and [audit](../data/manifests/rsna-pneumonia-5000-audit.json) give all source, selected, split, and compute-scope exclusion counts, but no participant-flow diagram exists. |
| 36 | Demographic and clinical characteristics by partition | [ ] Partial | Split-level study strata, patient groups, and boxes are in [datasheet](DATASHEET.md); age, sex, race/ethnicity, and clinical characteristics are not available in repository artifacts. |
| 37 | Performance on all partitions and subgroups | [ ] Partial | Validation and held-out test results are reported in baseline and comparison tables, but no demographic subgroup performance analysis exists. |
| 38 | Accuracy estimates, precision, calibration, and weak subpopulations | [ ] Partial | [Clean inference](../results/tables/statistical_clean_comparison.csv), [calibration](../results/tables/calibration_summary.csv), and [reliability diagrams](../results/figures/reliability_diagrams.png) cover uncertainty and calibration; demographic subgroup estimates are absent. |
| 39 | Failure analysis | [x] Complete | [YOLO stability audit](../results/tables/yolo_seed_stability.csv), [Grad-CAM cases](../results/tables/gradcam_qualitative_cases.csv), [failure panels](../results/figures/gradcam_failure_cases.png), and [XAI sanity data](../results/tables/gradcam_sanity_per_image.csv) expose incorrect/unstable behavior. |
| 40 | Study limitations | [x] Complete | [Consolidated limitations](LIMITATIONS.md) cover data, annotation, compute, model asymmetry, thresholds, calibration, robustness, XAI, statistics, deployment, and generalizability. |
| 41 | Implications for practice and intended role | [x] Complete | [Report Section 11.3](../report/report.md) and [deployment limits](LIMITATIONS.md) separate retrospective screening, human-reviewed point-of-care assistance, and prohibited autonomous use. |
| 42 | Full protocol or technical details | [x] Complete | [PROJECT_SPEC](../PROJECT_SPEC.md), [README commands](../README.md), configs, source, and this [supplement index](SUPPLEMENTARY.md) provide the detailed record. |
| 43 | Software, model, and data availability | [ ] Partial | Code/license and data acquisition/restrictions are explicit in [README](../README.md), [LICENSE](../LICENSE), and [datasheet](DATASHEET.md); trained checkpoints are ignored/regenerable but no public trained-model release is identified. |
| 44 | Funding and funder role | [ ] Not reported | No funding or funder-role statement was found in the report, README, or project documents. |

## TRIPOD+AI 2024

The study both develops and evaluates two non-generative neural networks, so
development (`D`) and evaluation (`E`) items are considered unless the row
explains why the prediction-model formulation is not applicable.

| Item | Audited topic | Status | Traceable repository evidence or gap |
|---:|---|---|---|
| 1 | Title: study type, target population, outcome | [ ] Partial | [Report title](../report/report.md) names models and lung-opacity detection but not explicitly model development/evaluation or the RSNA target population. |
| 2 | Abstract checklist | [ ] Not reported | No abstract exists in the [current report](../report/report.md). |
| 3a | Healthcare context and rationale | [x] Complete | [Report Sections 1–2](../report/report.md) and [literature review](LITERATURE_REVIEW.md) identify the diagnostic-imaging context, comparability gap, and prior models. |
| 3b | Target population, purpose, pathway, users | [x] Complete | [Datasheet](DATASHEET.md), [report Section 11.3](../report/report.md), and [limitations](LIMITATIONS.md) define the historical cohort, non-diagnostic purpose, human-reviewed scenarios, and users. |
| 3c | Known health inequalities | [ ] Not reported | The repository flags absent demographic/subgroup evidence and potential dataset bias, but does not report known healthcare inequalities between sociodemographic groups. |
| 4 | Objectives and development/evaluation scope | [x] Complete | [Research question](HYPOTHESES.md), [report Section 1](../report/report.md), and [decision log](DECISION_LOG.md) define controlled model development and held-out comparative evaluation. |
| 5a | Development/evaluation data sources and representativeness | [x] Complete | [Datasheet](DATASHEET.md), [dataset config](../configs/dataset.yaml), and [limitations](LIMITATIONS.md) trace one RSNA/NIH source and non-representativeness. |
| 5b | Data-accrual dates | [ ] Not reported | Challenge year and historical source are stated, but participant/image accrual start and end dates are absent. |
| 6a | Healthcare setting and centres | [x] Complete | [Datasheet](DATASHEET.md) identifies a single NIH image cohort and a separate multi-institutional annotator network. |
| 6b | Participant eligibility | [ ] Partial | [Datasheet](DATASHEET.md) documents Stage 2 and compute-scope selection, but the source cohort's full participant eligibility criteria are not reproduced. |
| 6c | Treatments | [ ] N/A | The models analyze stored radiographs; no treatment occurs between prediction and a future outcome. |
| 7 | Preprocessing, quality checks, demographic consistency | [ ] Partial | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), and [`prepare.py`](../src/data/prepare.py) fully trace preprocessing/QA, but demographic consistency cannot be evaluated from available metadata. |
| 8a | Predicted outcome, timing, rationale, group consistency | [ ] Partial | Lung-opacity boxes and negative images are defined in [datasheet](DATASHEET.md), but this is not a time-horizon outcome and group-consistency assessment is absent. |
| 8b | Outcome-assessor qualifications and demographics | [ ] Partial | [Datasheet](DATASHEET.md) gives radiologist counts and qualifications, but not annotator demographics. |
| 8c | Blinding of outcome assessment | [ ] Not reported | The original challenge annotation blinding procedure is not present. |
| 9a | Initial predictor choice and preselection | [ ] Partial | Inputs are complete decoded radiographs with no handcrafted predictor screening, as traced by configs and model adapters; an explicit predictor-choice rationale is not reported in TRIPOD terms. |
| 9b | Predictor definitions, timing, and blinding | [ ] Partial | Pixel inputs and transformations are reproducible in [datasheet](DATASHEET.md) and [`src/models/`](../src/models), but acquisition timing and predictor-to-outcome blinding are not documented. |
| 9c | Subjective predictor assessors | [ ] N/A | Image pixels are decoded automatically; no human assessor derives model predictors. |
| 10 | Development/evaluation sample-size rationale | [ ] Partial | [PROJECT_SPEC §3](../PROJECT_SPEC.md), [dataset config](../configs/dataset.yaml), and [datasheet](DATASHEET.md) explain hardware-scoped sizes, but no statistical sufficiency or formal sample-size calculation is provided. |
| 11 | Missing data and omissions | [x] Complete | [Audit](../data/manifests/rsna-pneumonia-5000-audit.json) and [datasheet](DATASHEET.md) report expected negative-row blanks, zero invalid selected studies, no imputation, and compute-only exclusions. |
| 12a | Partitioning, purpose, size, and leakage | [x] Complete | [Dataset config](../configs/dataset.yaml), split manifests, [audit](../data/manifests/rsna-pneumonia-5000-audit.json), and [report Section 3](../report/report.md) trace patient-disjoint 70/15/15 use. |
| 12b | Predictor transformations | [x] Complete | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), and [`prepare.py`](../src/data/prepare.py) specify bit-depth conversion, inversion, scaling, resizing, and canonical inputs. |
| 12c | Model type, rationale, building, tuning, internal evaluation | [x] Complete | [Literature review](LITERATURE_REVIEW.md), model protocols, configs, source, and [decision log](DECISION_LOG.md) trace model choice, training steps, validation-driven selection, and no Track B search. |
| 12d | Clustering and heterogeneity | [ ] Partial | [Statistics method](STATISTICAL_ANALYSIS.md), [config](../configs/statistics.yaml), and [`paired.py`](../src/stats/paired.py) account for repeated exams within NIH patients, but performance heterogeneity across sites/countries is not estimable from one source. |
| 12e | Performance measures, plots, rationale, comparisons | [x] Complete | [Quantitative comparison](QUANTITATIVE_COMPARISON.md), calibration, FROC, robustness, XAI, and [README index](../README.md) enumerate measures, plots, and comparison rules. |
| 12f | Model updating after evaluation | [ ] N/A | [Calibration analysis](CALIBRATION_ANALYSIS.md) explicitly performs descriptive evaluation without recalibration/refitting; no updated model exists. |
| 12g | Calculation of model predictions | [x] Complete | [Evaluation config](../configs/evaluation.yaml), [`src/evaluate.py`](../src/evaluate.py), model adapters, and hashed prediction bundles trace inference and postprocessing. |
| 13 | Class-imbalance methods and recalibration | [ ] N/A | [Dataset config](../configs/dataset.yaml) preserves proportional strata; no oversampling, SMOTE, class reweighting, or imbalance-driven recalibration is reported. |
| 14 | Fairness methods | [ ] Not reported | No fairness mitigation or subgroup performance analysis was undertaken; [limitations](LIMITATIONS.md) identify this gap. |
| 15 | Model outputs and threshold rationale | [x] Complete | [Threshold analysis](THRESHOLD_ANALYSIS.md), [threshold config](../configs/threshold_selection.yaml), [selected operating points](../results/tables/selected_operating_points.csv), [D-004](DECISION_LOG.md), and [cost sensitivity](THRESHOLD_CALIBRATION.md) separate original, primary, exploratory, and sensitivity thresholds. |
| 16 | Development-versus-evaluation differences | [x] Complete | [Datasheet split table](DATASHEET.md), [audit](../data/manifests/rsna-pneumonia-5000-audit.json), and [report Section 4](../report/report.md) show same-source, patient-disjoint partitions with common definitions and near-proportional strata. |
| 17 | Ethics approval and consent/waiver | [ ] Not reported | No local ethics-board determination, approval, consent, or waiver statement was found. |
| 18a | Funding | [ ] Not reported | No funding statement exists. |
| 18b | Conflicts of interest | [ ] Not reported | No author conflict-of-interest or financial-disclosure statement exists. |
| 18c | Protocol access or no-protocol statement | [ ] Partial | [PROJECT_SPEC](../PROJECT_SPEC.md), batch files, and configs preserve the working protocol, but the report does not identify a formal public protocol or state that none was prepared. |
| 18d | Study registration | [ ] Not reported | No registry name/identifier or explicit non-registration statement exists. |
| 18e | Data availability and restrictions | [x] Complete | [README acquisition steps](../README.md), [dataset config](../configs/dataset.yaml), [datasheet access conditions](DATASHEET.md), and split manifests identify source, retrieval, competition-rule restrictions, and non-redistribution. |
| 18f | Analytical code availability | [x] Complete | Repository source, [README commands](../README.md), [pinned requirements](../requirements.txt), [lockfile](../uv.lock), [license](../LICENSE), and run environments provide code and execution conditions. |
| 19 | Patient and public involvement | [ ] Not reported | No involvement activity or explicit statement of no involvement was found. |
| 20a | Participant flow and outcome counts | [ ] Partial | [Datasheet](DATASHEET.md) and [audit](../data/manifests/rsna-pneumonia-5000-audit.json) give source/selected/split patient, study, box, positive, and negative counts, but there is no flow diagram. |
| 20b | Participant characteristics, dates, predictors, events, missingness | [ ] Partial | Split strata, patient groups, boxes, and audit failures are reported; demographics, accrual dates, and treatments are unavailable. |
| 20c | Evaluation-versus-development predictor distribution | [ ] Partial | [Datasheet](DATASHEET.md) compares study strata and boxes across splits, but not demographic or pixel-distribution characteristics. |
| 21 | Participants and outcomes in each analysis | [x] Complete | [Datasheet](DATASHEET.md), [quantitative comparison](QUANTITATIVE_COMPARISON.md), [robustness](ROBUSTNESS.md), [XAI sanity](XAI_SANITY.md), and the [raw-score utility audit](DCA_ANALYSIS.md) report analysis-specific images, patient groups, boxes/events, seeds, and exclusions. |
| 22 | Full model available for third-party predictions | [ ] Partial | Architecture/training/inference code and exact configs are available, but trained checkpoints are ignored/regenerable and no public model object/API is identified. |
| 23a | Performance with CIs and key subgroups | [ ] Partial | [Clean statistics](../results/tables/statistical_clean_comparison.csv), [calibration](../results/tables/calibration_summary.csv), and figures report overall uncertainty; key demographic subgroup estimates are absent. |
| 23b | Heterogeneity across clusters | [ ] N/A | Patient clusters are used to preserve dependence in inference, not to estimate site/country performance heterogeneity; only one image source is available. |
| 24 | Updated-model results | [ ] N/A | No recalibrated/refitted model was created after evaluation. |
| 25 | Interpretation, prior studies, and fairness | [ ] Partial | [Report Sections 2 and 11](../report/report.md) interpret results against prior work without architecture-family overclaiming; fairness implications cannot be evaluated without subgroup data. |
| 26 | Limitations, bias, uncertainty, generalizability | [x] Complete | [LIMITATIONS.md](LIMITATIONS.md) provides a consolidated, domain-by-domain account including representativeness, missing demographics, sample size, uncertainty, and external validity. |
| 27a | Handling poor-quality or unavailable implementation inputs | [ ] Partial | Input validation, corruption behavior, and acquisition sensitivity are tested in code and docs, but no deployment-time rejection/handling policy is specified. |
| 27b | User interaction and required expertise | [ ] Partial | Human review is required in the point-of-care scenario in [report Section 11.3](../report/report.md), but user training and expertise requirements are not specified. |
| 27c | Future research for applicability/generalizability | [x] Complete | [Report Section 12](../report/report.md) and [limitations](LIMITATIONS.md) call for external multi-site, subgroup, prospective, workflow, calibration, and regulatory work. |

## STARD-AI 2025

STARD-AI is used here as a diagnostic-accuracy transparency crosswalk. The
index test is each frozen detector pipeline and the reference standard is the
pre-existing RSNA lung-opacity annotation. Localization AP/box matching is not
the same estimand as patient-level disease diagnosis.

| Item | Audited topic | Status | Traceable repository evidence or gap |
|---:|---|---|---|
| 1 | AI diagnostic-accuracy identification and accuracy measure | [ ] Partial | [Report title](../report/report.md) identifies detector comparison and target finding, but not an accuracy measure and no abstract exists. |
| 2 | Structured abstract | [ ] Not reported | No abstract exists. |
| 3 | Background, intended use, novelty, workflow | [x] Complete | [Report Sections 1–2 and 11.3](../report/report.md) and the [literature review](LITERATURE_REVIEW.md) define context, novelty, and hypothetical workflows. |
| 4 | Objectives and hypotheses | [x] Complete | [Research question and H1–H6](HYPOTHESES.md) are explicit and labeled retrospective; operational checks point to artifacts. |
| 5 | Prospective versus retrospective data collection | [x] Complete | [Datasheet](DATASHEET.md), [report](../report/report.md), and [limitations](LIMITATIONS.md) identify retrospective reuse of historical data. |
| 6 | Ethics approval or justification | [ ] Not reported | No ethics-board approval, waiver, or non-requirement justification is present. |
| 7 | Participant- and data-level eligibility criteria | [ ] Partial | [Datasheet](DATASHEET.md) reports dataset and deterministic subset criteria, but source participant eligibility is incomplete and not ordered as a flow. |
| 8 | Basis for identifying potentially eligible participants | [ ] Partial | [Datasheet](DATASHEET.md) identifies the RSNA Stage 2/NIH source and challenge strata, but not the full upstream clinical identification process. |
| 9 | Setting, location, and dates | [ ] Partial | The single NIH image source is identified; participant/image accrual dates are absent. |
| 10 | Consecutive, random, or convenience series | [x] Complete | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), and [`prepare.py`](../src/data/prepare.py) identify deterministic patient-grouped stratified sampling rather than a consecutive clinical series. |
| 11 | Dataset source and collection purpose | [x] Complete | [Dataset config](../configs/dataset.yaml) and [datasheet](DATASHEET.md) identify an open challenge dataset derived from routinely acquired NIH radiographs and all access conditions. |
| 12 | Dataset annotators and annotation process | [x] Complete | [Datasheet](DATASHEET.md) records radiologist numbers/experience, triple reading, adjudication, and source references. |
| 13 | Acquisition devices and index-test software | [ ] Partial | Pinned software/framework versions are complete in requirements, configs, and run environments; source imaging manufacturer/model details are unavailable. |
| 14 | Acquisition protocol and preprocessing | [ ] Partial | [Datasheet](DATASHEET.md), [dataset config](../configs/dataset.yaml), and [acquisition audit](ACQUISITION_SHIFTS.md) fully trace available preprocessing and sampled DICOM attributes, but original exposure/device protocols are missing. |
| 15a | Replicable index test | [x] Complete | Exact model configs, [`src/models/`](../src/models), [`src/evaluate.py`](../src/evaluate.py), [README](../README.md), and pinned dependencies define both detector pipelines. |
| 15b | Index-test development and data partitions | [x] Complete | [Datasheet](DATASHEET.md), model protocols, evaluation config, and [per-seed table](../results/tables/detector_comparison_per_seed.csv) trace training, tuning, testing, seeds, and sample sizes. |
| 15c | Positivity cutoffs and prespecified/exploratory status | [x] Complete | [Threshold analysis](THRESHOLD_ANALYSIS.md), configs, [selected points](../results/tables/selected_operating_points.csv), and [D-006](DECISION_LOG.md) distinguish original, validation-selected, exploratory, recall-preference, and hypothetical-loss thresholds. |
| 15d | End user and required expertise | [ ] Partial | [Report Section 11.3](../report/report.md) requires human review and defines scenarios, but does not state formal expertise/training requirements. |
| 16a | Replicable reference standard | [ ] Partial | Challenge box/negative semantics and conversion are traceable in [datasheet](DATASHEET.md) and [`prepare.py`](../src/data/prepare.py); original annotation instructions/software are absent. |
| 16b | Rationale for reference standard | [ ] Partial | [Dataset choice](DATASET_CHOICE.md) justifies expert-box provenance and [limitations](LIMITATIONS.md) discuss coarse boxes, but alternative reference methods are not compared. |
| 16c | Reference-standard categories/cutoffs | [x] Complete | [Dataset config](../configs/dataset.yaml), [datasheet](DATASHEET.md), and audit define `Lung Opacity`, negative study strata, category ID, and box presence. |
| 17a | Information available to index-test performers | [x] Complete | Automated inference consumes images only; [report Section 4.2](../report/report.md), model adapters, and [`src/evaluate.py`](../src/evaluate.py) keep held-out annotations in the evaluator after predictions. |
| 17b | Information available to reference assessors | [ ] Partial | The RSNA reference annotations predate this detector study, but the original assessors' available clinical information is not documented locally. |
| 18 | Accuracy-estimation and comparison methods | [x] Complete | [Quantitative comparison](QUANTITATIVE_COMPARISON.md), [statistics](STATISTICAL_ANALYSIS.md), calibration, and FROC documents plus source define all canonical comparisons. |
| 19 | Indeterminate results | [x] Complete | [Per-seed table](../results/tables/detector_comparison_per_seed.csv), [publication table](../results/tables/detector_comparison.csv), and [statistics config](../configs/statistics.yaml) preserve undefined YOLO seed-271 IoU/Dice as null with reason rather than imputing zero. |
| 20 | Missing index/reference data | [x] Complete | [Dataset audit](../data/manifests/rsna-pneumonia-5000-audit.json) reports zero invalid selected studies/boxes and [datasheet](DATASHEET.md) explains expected blank negative rows and no imputation. |
| 21 | Variability analyses and their status | [x] Complete | Five-seed clean variability, frozen `n=3` threshold/FROC/Pareto scope, seed-17 robustness/XAI scope, and retrospective hypotheses are distinguished in [HYPOTHESES.md](HYPOTHESES.md) and [supplement](SUPPLEMENTARY.md). |
| 22 | Intended sample size and determination | [ ] Partial | Hardware/time-scoped sizes are predeclared in [PROJECT_SPEC §3](../PROJECT_SPEC.md) and configs, but no formal diagnostic-accuracy sample-size calculation is present. |
| 23 | Error analysis and fairness assessment | [x] Complete | [Seed-stability audit](../results/tables/yolo_seed_stability.csv), [explainability failures](EXPLAINABILITY.md), and [XAI sanity](XAI_SANITY.md) detail error analyses; no fairness analysis was undertaken and [limitations](LIMITATIONS.md) state the subgroup gap. |
| 24 | Participant flow diagram | [ ] Partial | Complete source, selected, excluded, and split counts exist in [datasheet](DATASHEET.md) and [audit](../data/manifests/rsna-pneumonia-5000-audit.json), but no flow diagram exists. |
| 25 | Baseline demographic, clinical, and technical characteristics | [ ] Partial | Split strata, patients, boxes, and available technical processing are reported; demographic and broader clinical characteristics are absent. |
| 26a | Disease-severity distribution in positives | [ ] Not reported | No severity scale for lung-opacity-positive studies is available in repository artifacts. |
| 26b | Alternative diagnoses in negatives | [ ] Partial | `Normal` and `No Lung Opacity / Not Normal` counts are reported, but alternative diagnoses within the latter group are not available. |
| 27 | Interval/interventions between index and reference tests | [ ] N/A | The detector and pre-existing annotations are evaluated on the same stored image; this is not a delayed-reference clinical pathway study. |
| 28 | Representativeness of target-condition distribution | [x] Complete | [Datasheet](DATASHEET.md), [raw-score audit population boundary](DCA_ANALYSIS.md), and [limitations](LIMITATIONS.md) explicitly state that the stratified/enriched test distribution is not natural clinical prevalence. |
| 29 | External-evaluation dataset differences | [ ] N/A | No independent external dataset is evaluated; [limitations](LIMITATIONS.md) make this boundary explicit. |
| 30 | Cross-tabulation/distribution of index versus reference results | [ ] Partial | [Per-seed comparison](../results/tables/detector_comparison_per_seed.csv) reports TP/FP/FN and prediction counts for object detection, but no patient-level diagnostic 2×2 table is appropriate or reported. |
| 31 | Accuracy estimates and precision | [x] Complete | [Clean statistical table](../results/tables/statistical_clean_comparison.csv) reports point estimates and patient-cluster 95% CIs; calibration adds endpoint-specific uncertainty. |
| 32 | Adverse events | [ ] N/A | Retrospective software evaluation on stored data involved no participant-facing index or reference test. |
| 33 | Limitations, bias, uncertainty, generalizability | [x] Complete | [LIMITATIONS.md](LIMITATIONS.md) and [report Section 11.4](../report/report.md) are comprehensive. |
| 34 | Practice implications and clinical role | [x] Complete | [Report Section 11.3](../report/report.md) and deployment limits state conditional research scenarios and reject autonomous use; the [raw-score correction](DCA_ANALYSIS.md) disclaims conventional DCA evidence. |
| 35 | Ethics and fairness considerations | [ ] Partial | Data terms, non-reidentification, prospective/regulatory limits, and missing subgroup evidence are discussed; no ethics determination or fairness assessment exists. |
| 36 | Registration | [ ] Not reported | No registration number/name or explicit non-registration statement exists. |
| 37 | Full protocol access | [ ] Partial | [PROJECT_SPEC](../PROJECT_SPEC.md), batches, configs, source, and README form an auditable working protocol, but no formal public protocol is identified as such. |
| 38 | Funding and support | [ ] Not reported | No funding/support statement exists. |
| 39 | Commercial interests | [ ] Not reported | No commercial-interest or conflict statement exists. |
| 40a | Dataset/code availability and reuse restrictions | [x] Complete | [README](../README.md), [datasheet](DATASHEET.md), [dataset config](../configs/dataset.yaml), [LICENSE](../LICENSE), and pinned dependencies state access, competition terms, non-redistribution, code license, and reproduction. |
| 40b | Stored, auditable outputs | [x] Complete | [Per-seed tables](../results/tables/detector_comparison_per_seed.csv), hashed prediction/provenance summaries, complete corruption grids, logs, archives, and [supplement index](SUPPLEMENTARY.md) retain auditable outputs and scope. |

## Highest-priority reporting gaps before a journal submission

The unchecked rows are not silently curable from existing data. Before a
submission, the authors would need to decide or obtain, at minimum:

1. a structured abstract and title that explicitly identify development and
   held-out evaluation;
2. an ethics/consent-waiver or non-requirement statement;
3. funding, conflict-of-interest, registration, protocol, and patient/public
   involvement statements;
4. available participant dates, demographics, acquisition-device/protocol
   details, and reference-standard annotation/blinding details—or explicit
   statements that the source dataset does not provide them;
5. a participant-flow diagram and an explicit lack-of-formal-sample-size-
   calculation statement; and
6. demographic subgroup/fairness evaluation and external testing before any
   generalizable clinical-performance claim.
