# Reporting Checklist Crosswalk

Audit date: 2026-09-24

Updated for the final internal/external manuscript (Batch 51). Current
manuscript SHA-256: `1f3b5fac3809ff287ddc952ca0432c405e7c891c1bfc85d7cb009cb6c40330d2`.
The paper separates ranking, operating rules, training variability, observed
FROC support, implementation timing and cross-dataset transportability. All five
runs per detector remain represented. External testing is complete; human
declarations and final submission details remain unresolved.

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
| CLAIM 2024 | Directly applicable. The work develops, internally tests and externally tests two AI object detectors on medical images. | **Primary checklist.** All 44 items use the official `Yes` / `No` / `Not Applicable` response structure. |
| STARD-AI 2025 | Not a clean primary fit. Final STARD-AI covers diagnostic-accuracy studies of an AI-based index test. This benchmark estimates object-localization performance against boxes, allows multiple detections and false positives per image, and does not estimate participant-level diagnostic accuracy against a clinical reference standard. | Selected reporting concepts are mapped **by analogy only**. This is not the official STARD-AI submission checklist and does not establish STARD-AI compliance. |
| TRIPOD+AI 2024 | Outside primary scope. The study does not develop or evaluate an individualized diagnostic or prognostic prediction model that returns a person-level outcome probability. Detector boxes and confidence scores are not such a model. | Selected transparency concepts are mapped **by analogy only**. This is not a TRIPOD+AI compliance assessment. |

## CLAIM 2024 primary crosswalk

The official 2024 update defines 44 items and the three responses `Yes`, `No`,
and `Not Applicable`. For `Yes`, the evidence column names a manuscript section
or exact repository artifact. For `No` and `Not Applicable`, the unresolved or
out-of-scope reason is explicit. CLAIM terminology is followed: the
patient-disjoint RSNA held-out split is **internal testing**; VinDr is
**external testing** under a non-identical opacity ontology, not identical-task
validation. `Validation` denotes the RSNA model/threshold-selection partition.

| Item | Reporting topic (paraphrased) | Response | Evidence or explanation |
|---:|---|---|---|
| 1 | Identify the work as AI methodology and name the technology category in the title or abstract | Yes | The [structured abstract](../report/paper_draft.md#abstract) explicitly identifies two deep-learning object-detection pipelines and names Faster R-CNN and YOLO11s. |
| 2 | Structured abstract with design, population, partitions, retrospective/prospective status, statistics, outcomes, implications, and availability | No | The structured abstract reports retrospective internal/external testing, both cohorts, patient-disjoint internal partitions, all runs, estimands, absolute collapse and bounded ordering. Release-ready software/data/model availability remains an unresolved author-controlled declaration. |
| 3 | Scientific/clinical background, intended use, current practice, and rationale | Yes | [`paper_draft.md` §1](../report/paper_draft.md#1-introduction) and [§5](../report/paper_draft.md#5-discussion) define the scientific comparison, intended research use, and non-clinical boundary. |
| 4 | A priori aims, objectives, and hypotheses | No | Internal aims/hypotheses are retrospective. External aims, ontology, endpoints and policies were locally frozen before full inference in [the protocol](VINDR_EXTERNAL_PROTOCOL.md); this is not public preregistration. The final narrative does not retrospectively convert internal H1--H6 into prospective hypotheses. |
| 5 | State whether the study is prospective or retrospective | Yes | The abstract and §1 identify retrospective secondary analysis and distinguish locally prespecified external testing from the retrospective internal analysis. |
| 6 | State the study goal, modeling task, target, and intended role | Yes | [`paper_draft.md` §§1 and 3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) define a controlled, non-clinical lung-opacity object-localization benchmark. |
| 7 | Identify data sources and their match to the intended population | Yes | Manuscript §§3.1 and 3.5 identify RSNA/NIH and official PhysioNet VinDr-CXR v1.0.0, with exact strict local-target mapping and research-only scope. |
| 8 | Eligibility, setting, location, dates, demographics, and sampling method | No | RSNA subset, sampling and limited header demographics are reported; VinDr includes its complete released test set and strict support. Detailed RSNA source accrual/eligibility, broader clinical variables and a locally verified external demographic table remain unavailable. No consecutive clinical-sampling claim follows. |
| 9 | Preprocessing steps | Yes | Manuscript §§3.1/3.5, [dataset config](../configs/dataset.yaml), [frozen external config](../configs/vindr_external_v1.yaml) and [adapter receipt](../results/vindr_external_v1/adapter_preflight.json) record conversion, inversion, native transforms and technical integrity checks. |
| 10 | Selection of data subsets | Yes | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split), [`configs/dataset.yaml`](../configs/dataset.yaml), and [`docs/DATASET_CHOICE.md`](DATASET_CHOICE.md) record seed-17 SHA-256 ordering, label-stratum tracking, patient grouping, and the hardware-scoped 5,000-study selection. |
| 11 | De-identification method | No | VinDr source de-identification is documented by the release and absent usable patient identity tags were verified locally; this does not provide defensible patient grouping. Complete RSNA upstream/local de-identification reporting remains unavailable. |
| 12 | Missing-data assessment and handling | Yes | Box-coordinate/conversion completeness and `PatientAge`/`PatientSex`/`ViewPosition` missingness are explicit. The sole out-of-range age is excluded from age summaries; numeric ages without encoded units are retained only as nominal years with a caveat. Uncollected clinical variables are named as unavailable rather than silently imputed. |
| 13 | Image-acquisition protocol and equipment detail | No | [`paper_draft.md` §§3.7 and 6](../report/paper_draft.md#6-limitations) explicitly say acquisition fields, scanner settings, and processing history needed for reproducibility are unavailable. |
| 14 | Reference-standard definition and labeling instructions | No | Both reference standards and the frozen exact-label external mapping are reported. VinDr consensus construction is cited, but complete source reader instructions and all case-level adjudication details are not reproduced locally. |
| 15 | Rationale for the reference standard and assessment of possible errors | Yes | [`docs/DATASET_CHOICE.md`](DATASET_CHOICE.md) records why RSNA was selected, and [`paper_draft.md` §6](../report/paper_draft.md#6-limitations) describes coarse-box, disagreement, and absent re-reading limitations. |
| 16 | Annotation sources, number and qualifications of annotators, and instructions | No | [`docs/DATASHEET.md`](DATASHEET.md) records 18 board-certified radiologists from 16 institutions, including 12 thoracic specialists, but the full instructions and case-assignment/adjudication detail are not reported locally. |
| 17 | Annotation procedure and software for the internal-testing set | No | The source boxes were reused; local conversion is traceable, and the official RSNA description names md.ai, but the complete display and case-assignment/adjudication procedure is not reported for this internal-testing subset. |
| 18 | Inter- and intrarater variability | No | No local rereading study or source-level inter/intrarater agreement statistic is available. |
| 19 | Partition assignment, sizes, proportions, differences, and class imbalance | Yes | RSNA patient-grouped partition sizes, counts and differences are in §3.1 and the committed manifests. The complete VinDr test cohort and strict support are reported separately in §§3.5/4.6; it was not split, sampled or adapted. |
| 20 | Disjointness level between partitions | Yes | Empty NIH patient-key intersections establish internal partition disjointness. The external release is independent as a dataset; no unverified patient linkage, within-VinDr independence or identical-task claim is made. |
| 21 | Internal-testing sample size and how it was determined | Yes | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) reports 750 internal-testing studies/323 patient groups and explicitly states that the 5,000-study cohort was hardware-scoped with no formal sample-size or power calculation. The dedicated row below gives the complete interpretation. |
| 22 | Sufficient model detail to reconstruct inputs, outputs, and architecture | Yes | [`paper_draft.md` §3.2](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), [`src/models/faster_rcnn_model.py`](../src/models/faster_rcnn_model.py), [`src/models/yolo_training.py`](../src/models/yolo_training.py), and [`configs/`](../configs/) identify the implemented pipelines and configuration-derived outputs. |
| 23 | Software versions and hardware | Yes | Manuscript §§3.2/3.5/3.7, pinned environments and [timing provenance](../results/logs/phase45_inference_timing_v1/summary.json) disclose hardware/software and the internal FP32 versus external bfloat16 YOLO accuracy paths. |
| 24 | Parameter initialization | Yes | [`paper_draft.md` §3.2](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), [`configs/faster_rcnn.yaml`](../configs/faster_rcnn.yaml), and [`configs/yolo.yaml`](../configs/yolo.yaml) record COCO initialization and the canonical seed; seed-specific configs and model-loader code record the remaining seeds and head adaptation. |
| 25 | Training, augmentation, stopping, hyperparameters, objectives, and frozen parameters | Yes | [`paper_draft.md` §3.2](../report/paper_draft.md#32-detector-pipelines-and-controlled-training-factors), [`configs/faster_rcnn.yaml`](../configs/faster_rcnn.yaml), [`configs/yolo.yaml`](../configs/yolo.yaml), and per-run `resolved_config.json` files indexed in [`SUPPLEMENTARY.md` S2](SUPPLEMENTARY.md#s2-full-clean-seed-level-comparison) provide the exact record. |
| 26 | Final-model selection | Yes | RSNA validation mAP selected each checkpoint. All ten checkpoint hashes/config identities are bound in [the release inventory](../results/checkpoint_release_manifest.json) and [external summary](../results/vindr_external_v1/inference_summary.json). No external checkpoint selection occurred. |
| 27 | Ensembling method | Not Applicable | Neither arm is an ensemble; each reported run is one Faster R-CNN or YOLO11s checkpoint. Seed summaries quantify repeated training and do not combine predictions into an ensemble. |
| 28 | Performance metrics, rationale, and comparison with prior models | No | [`paper_draft.md` §§3.3--3.5](../report/paper_draft.md#33-five-retained-runs-and-common-evaluation) defines the metrics and scientific rationale, but prior publications are contextual only and are not directly reproduced under this study's split/evaluator for quantitative comparison. |
| 29 | Uncertainty/significance procedures and statistical software | Yes | Manuscript §3.6 and [external statistical methods](VINDR_STATISTICS.md) distinguish observation/run uncertainty, historical checkpoint-conditional tests, seed-17 external sensitivity and descriptive cross-dataset contrasts. Intervals are marginal; no interaction test or same-number stochastic seed pairing was introduced. |
| 30 | Robustness or sensitivity analyses | Yes | Manuscript §§3.4/3.8/4.2/4.7/4.8 and [supplement](SUPPLEMENTARY.md) report post-hoc n=5 threshold sensitivity, finite-support bounds, conditional-checkpoint sensitivity and secondary synthetic stress analyses. |
| 31 | Explainability methods, parameters, and validation/sanity checks | Yes | Manuscript §3.8 and Supplementary S6 link the existing Grad-CAM layers, proxy targets, localization and randomization controls; none establishes clinical reasoning or causal faithfulness. |
| 32 | Evaluation on internal data | Yes | [`paper_draft.md` §§3.3 and 4.1](../report/paper_draft.md#33-five-retained-runs-and-common-evaluation) identifies the patient-disjoint held-out evaluation as internal testing and preserves its detector/seed-specific sample sizes. |
| 33 | External testing, or an explicit absence and rationale | Yes | Manuscript §§3.5/4.6--4.8 and [external artifacts](VINDR_STATISTICS_RESULTS.md) report frozen, adaptation-free testing of every run on all 3,000 official VinDr images, with 84 strict-target-positive images/95 boxes. Non-identical ontology, unknown patient dependence, floor/cap limits and precision-path differences are explicit. This is cross-dataset/cross-annotation-ontology transportability, not identical-task validation. |
| 34 | Registration information for a clinical trial | Not Applicable | Retrospective computational research without participant enrollment or intervention; no clinical trial. Internal analyses were not preregistered; external scientific settings were locally frozen before inference. |
| 35 | Numbers included/excluded at each stage and a flow diagram | No | Source/subset/split counts and complete external inclusion are explicit, but a single participant/study flow diagram is absent. |
| 36 | Demographic and clinical characteristics by partition | No | RSNA partition-wise study counts and limited header age/sex/projection are reported with missingness and unit caveats. VinDr strict-target support is complete, but a locally verified external demographic/clinical table and broader clinical descriptors are absent. Aggregate demographics do not establish fairness. |
| 37 | Final performance for all partitions and relevant subgroups | No | All-run internal and external performance and uncertainty are reported; no demographic/clinical subgroup performance evaluation exists. |
| 38 | Diagnostic-classification performance, uncertainty, calibration, and subpopulations | Not Applicable | The primary task is object localization, not participant-level diagnostic classification. Detection AP/FROC/uncertainty and emitted-detection D-ECE are reported, but they must not be relabeled as patient-level diagnostic sensitivity/specificity or clinical-risk calibration. |
| 39 | Failure analysis | Yes | Manuscript §§4.1--4.8 reports the retained seed-271 behavior, failed threshold transport, low emissions, support bounds and checkpoint-conditional disagreements. Existing corruption/XAI failure analysis remains supplementary. |
| 40 | Study limitations | Yes | Manuscript §6 and [limitations](LIMITATIONS.md) cover internal sampling/training, ontology/support/patient/precision confounds, coarse run variability, secondary scope and lack of prospective/reader/clinical/fairness evaluation. |
| 41 | Implications and intended role | Yes | [`paper_draft.md` §§5--6](../report/paper_draft.md#5-discussion) limits conclusions to controlled pipeline trade-offs and prohibits clinical use. |
| 42 | Protocol and additional technical detail availability | Yes | [README Batch 51](../README.md#final-manuscript-verification-batch-51), frozen v1 protocol, public aggregate manifests and [supplement](SUPPLEMENTARY.md) provide exact commands and separate portable evidence checks from authorized-data replay. Historical editorial baseline bytes are preserved without modifying the scientific freeze. |
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
| Data source, eligibility, sampling, and participant flow | [`paper_draft.md` §3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split) reports source/split counts and partition-wise DICOM-header age, sex, and AP/PA projection. | Accrual dates, detailed eligibility, broader clinical/demographic variables, and a flow diagram remain unavailable. |
| Reference standard | Challenge bounding boxes are described in [`paper_draft.md` §§2--3.1](../report/paper_draft.md#31-dataset-target-and-patient-disjoint-split). | Boxes are object-localization annotations, not a participant-level clinical diagnosis; original reader instructions and local agreement assessment are unavailable. |
| Internal versus external testing | Patient-disjoint held-out evaluation is explicitly called internal testing in [`paper_draft.md` §3.3](../report/paper_draft.md#33-five-retained-runs-and-common-evaluation). | Frozen VinDr external testing is now reported in §§3.5 and 4.6--4.8; non-identical ontology and absent patient linkage limit interpretation. |
| Sample-size rationale | The dedicated row above reports the hardware-driven 5,000-study subset and absence of formal calculation. | This is not a diagnostic-accuracy power calculation. |
| Uncertainty, indeterminate outputs, and failures | [`paper_draft.md` §§3.6, 4.1, 4.4, and 4.6--4.8](../report/paper_draft.md#36-estimands-and-uncertainty) report intervals, the zero-detection YOLO run, and endpoint-specific undefinedness. | Multiple boxes/false positives and conditional localization do not reduce to one diagnostic result per participant. |
| Generalizability and clinical role | [`paper_draft.md` §§5--6](../report/paper_draft.md#5-discussion) explicitly limit use. | External testing is complete but confounds dataset, ontology and numerical differences. No diagnostic utility, prospective benefit or complete fairness assessment is established. |

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
| Sample size and missing data | The dedicated row above reports the subset; annotation/conversion and age/sex/projection missingness audits are traceable. | No formal sample-size calculation exists, and uncollected broader clinical variables cannot be assessed or imputed. |
| Model performance and calibration | Detection metrics, uncertainty, and emitted-detection D-ECE are in [`paper_draft.md` §§3.3, 3.8, and 4](../report/paper_draft.md#33-five-retained-runs-and-common-evaluation). | D-ECE is not exam-level outcome-probability calibration; decision-curve analysis was correctly not performed. |
| Open science, declarations, and patient/public involvement | Code/config/artifact provenance is indexed in [`README.md`](../README.md) and [`SUPPLEMENTARY.md`](SUPPLEMENTARY.md); author-action placeholders are in [`paper_draft.md` §8](../report/paper_draft.md#8-declarations). | A release-ready availability statement and author-controlled declarations remain unresolved; no PPI fact may be assumed. |

## Submission blockers and linked audits

- Resolve all `AUTHOR ACTION REQUIRED` fields in [`paper_draft.md` §8](../report/paper_draft.md#8-declarations)
  and [`AUTHOR_DECLARATIONS_TODO.md`](AUTHOR_DECLARATIONS_TODO.md).
- Use [`HYPOTHESIS_TRACEABILITY.md`](HYPOTHESIS_TRACEABILITY.md) to retain the historical
  H1--H6 provenance and endpoint/artifact bindings. These labels are no longer
  journal prose; this does not make the focused research question prospective.
- Use [`CITATION_AUDIT.md`](CITATION_AUDIT.md) for the current 19-key
  manuscript source audit and the explicitly archived broader source review.
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
