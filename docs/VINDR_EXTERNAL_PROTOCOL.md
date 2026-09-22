# VinDr-CXR external testing protocol v1

Frozen 2026-09-22, Batch 47. Protocol ID: `vindr-cxr-external-testing-v1`.
Machine-readable contract: [vindr_external_v1.yaml](../configs/vindr_external_v1.yaml).
Byte hashes and internal dependencies: [freeze manifest](VINDR_EXTERNAL_PROTOCOL_v1.sha256.json).
Decision: [D-017](DECISION_LOG.md#d-017--freeze-vindr-cxr-external-testing-v1-before-performance).

**No external detector inference or external detector results were inspected in
this session.** This is a local prespecification before external performance,
not a public preregistration. The user must review this document before Batch 49.
Batch 48 must implement and test the adapter first; this document does not
authorize either later batch.

## 1. Scientific question and prerequisite gate

Do ranking performance, false-positive regimes, raw score distributions and
RSNA-selected operating points transport to the official VinDr-CXR test set?
Use **external testing**, **cross-dataset transportability**, and
**cross-annotation-ontology transportability**. Do not describe the experiment
as identical-task external validation or pneumonia diagnosis. Dataset,
annotation, population, acquisition and implementation differences are
confounded; none can independently explain a performance change here.

The Batch 46 canonical manuscript is `report/paper_draft.md`, at baseline
commit `1a7104d7a817b835f5a4d171b44d532fcbee9701` on `main`. Its current 66
numerical claims and semantic guards pass. The scientific verifier passes
72 artifacts, 346 present inputs, zero unavailable inputs and 201 result
references. Bibliography verification passes all three manuscript variants.
All ten local checkpoint files match release-manifest size/SHA-256 without
loading models; 38 focused COCO/matcher/FROC/threshold tests pass.
The Batch 42 v4 FROC, Batch 43 validation sensitivity, Batch 44 cohort and
Batch 45 timing summaries are complete and agree with the Batch 46 audit.
Their hashes, source definitions and manuscript hash are bound in the freeze
manifest. Pre-existing download-tooling edits and unrelated untracked files
are preserved. No manuscript, internal result or historical threshold changes.

## 2. Release, access, provenance and sharing

Use only [official PhysioNet VinDr-CXR v1.0.0](https://physionet.org/content/vindr-cxr/1.0.0/),
[version DOI](https://doi.org/10.13026/3akn-b287), released 2021-06-22.
The release specifies 15,000 training and 3,000 test images. Test annotations
are consensus labels: three initial readers followed by two reviewing
radiologists. Local boxes and global diagnoses are separate. Image IDs derive
from SOP instance identifiers; patient identifiers were removed or randomized.
The official test CSV has no radiologist identifier. These release facts do
not establish equivalence to the RSNA annotation task.

On 2026-09-22 the user personally confirmed approved PhysioNet access, DUA
acceptance and an official download **before restricted contents were read**.
This is a recorded user attestation, not an independent account inspection.
Codex neither accepted terms nor authenticated/downloaded data. Access requires
credentialing, required training and the project DUA on the release page.
The local `LICENSE.txt` and current
[license](https://physionet.org/content/vindr-cxr/view-license/1.0.0/) and
[DUA](https://physionet.org/content/vindr-cxr/view-dua/1.0.0/) specify version
1.5.0: protect access, do not reidentify or share restricted data, use it for
lawful research, and share code accompanying disseminated results.

The terms do not grant blanket permission to publish arbitrary derived data.
Project policy therefore keeps DICOMs, PNGs, annotations, row-level manifests,
identifiers, header values and image-linked predictions/scores inside the
ignored `data/processed/vindr-cxr-external-v1/` tree. Do not upload them to
external services or put them in tracked `results/`. Publish only code,
protocol/config/hashes and nonidentifying aggregate statistics or plots after
content review. Image examples or reusable row-level derivatives require an
explicit permission determination before release. No data are distributed by
this batch. The local license's 2021 copyright date differs from the website's
current 2026 date; both retain license version 1.5.0.

Resolve the dataset root solely from `VINDR_CXR_ROOT`; no personal path or
fallback search is allowed. It points to the directory containing `test/`,
`annotations_test.csv`, `image_labels_test.csv`, `LICENSE.txt` and
`SHA256SUMS.txt`. The supplied local layout is repository-relative
`data/raw/vindr-cxr/`, with annotation CSVs directly under that root. Different
layouts need an explicit configuration amendment, never inferred relocation.

Batch 47 metadata checks established:

| Check | Observation |
|---|---|
| DICOM filenames / image-label rows / distinct image IDs | 3,000 / 3,000 / 3,000; sets agree with official manifest test entries |
| Box CSV columns | `image_id,class_name,x_min,y_min,x_max,y_max` |
| Box CSV IDs | All belong to the test inventory |
| Source integrity | Both annotation CSVs, license and tag supplement match the supplied SHA256 manifest; config binds their hashes and the manifest hash |
| Image header shape | 3,000 single-frame, one-sample grayscale objects |
| Photometric interpretation | 2,459 MONOCHROME2; 541 MONOCHROME1 |
| Transfer syntax | 1,033 JPEG 2000 lossless; 1,966 implicit VR little endian; 1 explicit VR little endian |
| Nonempty patient/study/series/SOP identity tags | Zero for each of the four tags across all 3,000 objects |

Only headers and annotation schema/inventory were inspected, without pixel
decoding, image visualization or predictions. The supplied checksum manifest
is not independently authenticated by a new download. Full DICOM content
checksum verification, decoding, box integrity, target counts and conversion
remain Batch 48 gates. Missing/corrupt files stop the run; do not silently
reduce the cohort or substitute Kaggle competition data.

## 3. Population and strict target ontology

Include every official test image, without sampling or post-result exclusions.
Target concept: **`Lung opacity`**. The actual official `annotations_test.csv`
uses **`Lung Opacity`**, also the image-label column spelling. This explicit
capitalization mapping is fixed now: exact equality to that local box label,
without fuzzy matching, case folding, synonyms or ontology expansion.

Map only those boxes to canonical category 1, `Lung Opacity`, corresponding
to Faster R-CNN foreground 1 and YOLO foreground 0. Derive class count from
the configured category map. Do not merge Consolidation, Infiltration,
Atelectasis, ILD, global Pneumonia, or any other finding. Global labels cannot
create local targets. All images with no strict-target boxes are negatives
for this target, including images with other abnormalities. Other findings
do not create ignore regions: predictions not matching a strict-target box
are false positives under this ontology, not necessarily clinical errors.

Read original-coordinate `xyxy`; COCO conversion is `[xmin,ymin,xmax-xmin,
ymax-ymin]`, with no `+1`. Keep native dimensions. Validate finite coordinates,
positive areas, image membership, bounds and exact duplicates before inference.
Malformed/duplicate target annotations stop for a documented data-only review;
do not silently clip, merge or drop them. Keep official row order within an
image and deterministic sorted image order. No training images or labels may
enter fine-tuning, calibration, threshold/model selection or parameter tuning.

## 4. Frozen DICOM and model execution contract

Decode one image at a time with `pydicom.pixel_array` and a lossless decoder
supporting the observed transfer syntaxes. Record decoder/package versions.
Use the repository's `scale_radiograph_to_uint8` operation: squeeze a float32
array to two dimensions, obtain finite min/max, replace NaN/negative infinity
with the minimum and positive infinity with the maximum, invert MONOCHROME1
using `high + low - array`, min-max scale to 0--255, `numpy.rint`, clip and cast
to uint8. No finite pixels is an error; constant arrays become zero as in the
reference. Record these cases, without outcome-dependent exclusions.

Deliberately use decoded stored pixel values without Modality LUT,
RescaleSlope/Intercept, VOI/window transforms, padding masks or histogram
equalization, consistent with the internal canonical transform. This is a
research display normalization, not recovery of quantitative physical units.
Do not assume VinDr headers match RSNA. Batch 48 must audit dimensions, pixel
representation, rescale/window/LUT fields, padding and orientation/projection,
check polarity/coordinates on a fixed technical sample, and stop for a
versioned pre-inference amendment if semantics invalidate this contract.
Do not compare preprocessing alternatives by external performance.

Save lossless grayscale PNGs at native dimensions; replicate to RGB at model
input. No crop, rotation, flip or test-time augmentation. Use the checkpoint's
frozen aspect-preserving detector-native transforms: Faster R-CNN min/max 640
and existing ImageNet mean/std; YOLO11s native 640 letterbox and normalization.
Restore predictions to original-image coordinates before common evaluation.

Use all ten hash-verified RSNA best checkpoints for seeds 17, 42, 137, 271 and
314, with the run/config paths in the frozen inventory. No checkpoint changes,
ensembling, extra training or favorable-run selection. Run sequentially on the
RTX 4060 Laptop GPU with batch 1, bounded RAM, deterministic settings and AMP:
Faster R-CNN float16; YOLO11s bfloat16. Use the Batch 45 explicit autocast path
and verify actual convolution dtypes; `amp=True` alone is insufficient for
Ultralytics inference. Batch 45's `quantize=None` path is the reference.
Record the config, protocol, checkpoint, adapter-code, source inventory and
environment hashes with each future run.

**Precision limitation:** historical YOLO accuracy bundles used an FP32
inference path; Batch 45 documents this. AMP is mandatory for this new work.
Threshold values remain unchanged, but comparison to historical internal
results includes this disclosed implementation difference. Do not relabel
historical outputs as AMP or rerun them in this batch.

## 5. Endpoints and candidate support

Collect at score floor **0.00001** for both detectors, retaining at most **100**
detections/image after native class-aware NMS at **IoU 0.50**. Preserve native
proposal settings. No additional NMS, weighted-box fusion or calibration.
Keep every image in prediction bundles, including empty predictions. Scores
must be finite and in [0,1]; store full emitted precision without decimal
rounding. Evaluation uses `score >= threshold`; native candidate filtering
and any equality behavior must be documented by adapter tests.

Primary endpoints:

- **AP@0.50 and AP@0.50:0.95:** common COCO bbox evaluator, one class, all
  areas, maxDets [1,10,100], 101 recall points, IoUs 0.50:0.05:0.95. Apply the
  internal AP floor **0.001** to the lower-floor bundle. AP therefore remains
  comparable in score support to historical internal AP; no new lower-floor
  AP endpoint is primary. AP is independent of the transported single cutoff,
  but conditional on candidate support, cap and evaluator.
- **Exact-score FROC:** use all retained candidates down to 0.00001, every
  distinct score, an empty upper sentinel (`nextafter(max_score,+inf)`) and
  the floor endpoint. Tied scores enter together. Sensitivity is matched
  strict-target boxes / all strict-target boxes; FP/image uses **all 3,000
  images**, including strict-target negatives. Use existing score-descending,
  stable, one-to-one greedy same-class matching at IoU >=0.50. Match to the
  maximum-IoU unmatched target; equal IoUs retain source target order.
- **Sensitivity at FP/image budgets 0.125, 0.25, 0.5, 1 and 2:** per run take
  the maximum observed sensitivity with FP/image <= budget (tolerance 1e-12),
  with ties resolved by fewer FP/image then higher score; no interpolation.
  Aggregate these per-run sensitivities equally. Report achieved FP/image,
  candidate-floor endpoint and floor-limited status/count. FROC coordinates
  are descriptive curve evaluations, **not VinDr-selected deployment
  thresholds**; do not export their score coordinates as an operating rule.

Reuse the current internal exact-score v4 definitions. If a budget remains
beyond a run's floor endpoint, label its observed contribution as a lower
bound, not a completed frontier. Report detection-cap saturation as another
support limit. Do not lower the floor or raise the cap after seeing VinDr.
No claim of a global asymptote or exhaustive FROC is permitted.

## 6. Frozen threshold transport and score summaries

| Role | Faster R-CNN | YOLO11s | Selection provenance |
|---|---:|---:|---|
| Primary historical transport | 0.69 | 0.05 | RSNA validation seeds 17/42/137; historical maximum equal-run mean F1 |
| Secondary post-hoc sensitivity | 0.70 | 0.01 | Batch 43 RSNA validation seeds 17/42/137/271/314; same rule |

Apply each detector's threshold unchanged to **all five** external runs.
Historical selection count remains three, evaluation run count five. The
secondary thresholds exist and are verified in Batch 43's source summary and
table; they remain labeled **post-hoc n=5 validation sensitivity** even though
their external application is specified now. No VinDr-specific threshold,
calibration fit, F1 optimization, prevalence correction or threshold rounding.

For both policies report per-run TP/FP/FN, precision, box recall/sensitivity,
F1, FP/image, detections/image and percentage of images emitting at least one
detection, plus underlying counts and denominators. Use the shared evaluator's
zero-prediction convention: if targets exist, AP/precision/recall/F1 are zero.
An actually undefined quantity remains NA with its reason; no run is removed.

Describe scores separately for each detector/run at four fixed supports:
all candidates >=0.00001; AP candidates >=0.001; historical threshold;
post-hoc n=5 threshold. Report **detection count, min, 1/5/25/50/75/95/99th
percentiles, median, arithmetic mean and max**, using linear quantiles.
These are detection-level emitted-population descriptions, not calibrated
patient probabilities. Empty populations have count zero and other summaries
NA. Also report per-image detection counts and zero-detection image counts.
Give equal-run mean/sample SD of defined summaries with their contributing
run counts; do not silently pool detections or weight high-output runs more.
Use per-run empirical CDFs on the AP support for internal/external plots,
which prevents floor differences from masquerading as score-scale shifts.

## 7. Observational unit, aggregation and uncertainty

**No defensible patient grouping is available.** Neither annotation CSV has a
patient key. All 3,000 inspected headers lack nonempty `PatientID`,
`StudyInstanceUID`, `SeriesInstanceUID` and `SOPInstanceUID`. The public release
describes image-ID hashing rather than a patient linkage. Do not invent groups
from filenames, hashes, demographics or apparent image similarity. The
observational unit and resampling key are the **released study/image ID**;
this does not assert 3,000 independent patients or one image per patient.
Unknown repeated-patient dependence may make image-based intervals too narrow.

For every endpoint report all five run values, arithmetic mean and sample SD
(ddof=1). Seed 271 is retained, including zero detections or adverse results.
No ensembling or prediction pooling. No pairing of same-number training seeds.

Planned primary uncertainty follows the existing training-procedure estimand:
2,000 bootstrap replicates, RNG seed 20260811, paired image sampling with
replacement shared across detectors and runs, and independent sampling of
five runs with replacement within each detector. Recompute each sampled run's
cohort-level metric and then its equal-run mean. Repeated sampled images need
distinct replicate keys so COCO cannot deduplicate them. Report marginal 95%
percentile intervals for the two AP endpoints, five FROC budget sensitivities
and historical threshold-transport metrics, with Faster R-CNN minus YOLO11s
contrasts. Secondary seed-17 checkpoint-conditional intervals hold the models
fixed and resample only paired images. Secondary threshold-policy summaries
stay descriptive. Score quantiles are descriptive, without detection-level
bootstrap inference. Record undefined replicates rather than retrying until a
desired result; report valid counts and NA when the interval is undefined.

These are marginal intervals, not familywise confirmatory tests. No additional
null-hypothesis tests or multiplicity-sensitive significance claims are planned.
Keep RSNA and VinDr separate; do not pool AP, FROC or observations. Internal
resampling retains its valid NIH patient groups. Any later internal/external
comparison must show different ontology, candidate support and precision-path
limits rather than implying a causal explanation for the shift.

## 8. Planned outputs and prohibited tuning

| Output | Prespecified contents |
|---|---|
| T1: cohort/contract | Release/split, strict positives/boxes/negatives, preprocessing, observational unit, access and support limits |
| T2: ranking/FROC | All ten runs, AP endpoints, five FROC budgets, equal-run mean/SD, uncertainty and floor/cap flags |
| T3: threshold transport | Historical and separately post-hoc thresholds, all run metrics, counts and denominators; internal/external columns |
| T4: scores | Fixed-support counts, quantiles and range summaries, empty-support and contributing-run counts |
| F1: FROC | Separate internal/external panels, per-run curves, prespecified budgets and floor endpoints; no unsupported interpolation |
| F2: score distributions | Internal/external per-run empirical CDFs at the common AP support; detection-count annotations |
| F3: threshold transport | Historical-policy metrics with per-run points/intervals and clearly separate post-hoc sensitivity |

Keep figures nonidentifying; no radiograph examples are planned. Save private
row-level evidence locally and only reviewed nonidentifying aggregates under
the configured result root. Report rank reversals, collapse, adverse thresholds
and every retained run exactly as observed.

Prohibited after external results: ontology merges; filtering the population;
new thresholds; model/seed selection; fine-tuning; recalibration; preprocessing,
input-size, NMS, floor, cap or endpoint changes; cherry-picked runs/figures;
using VinDr training data to tune any choice. FROC assessment must not become
threshold selection. An implementation bug requires stopping, documenting the
bug, preserving superseded artifacts and fixing only the bug. Any scientific
contract change requires a separately versioned amendment with timing/reason
and prior-result exposure disclosed; never edit v1 and silently replace hashes.

## 9. Freeze verification and review gates

The JSON sidecar binds the exact UTF-8/LF protocol and YAML bytes plus internal
source/config/summary dependencies. It deliberately does not hash itself or
the evolving decision log/state files. It is a local integrity record, not a
trusted timestamp or signature. Each future run must preserve both digest
values and the sidecar digest in its provenance. README contains the executable
offline hash check and explicitly planned later-batch command interfaces.

Before Batch 49: user reviews this document and authorizes proceeding; Batch
48 passes official inventory and full checksums, decoding/coordinate/ontology
tests and data-only preflight; all ten checkpoints match the bound manifest;
protocol/config/dependencies still match; no external performance-driven
amendment has occurred. If any condition fails, stop. Batch 47 ends here.
