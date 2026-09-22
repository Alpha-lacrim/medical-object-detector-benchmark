# VinDr-CXR adapter preflight — Batch 48

Protocol: [frozen v1](VINDR_EXTERNAL_PROTOCOL.md).
Scientific configuration: [vindr_external_v1.yaml](../configs/vindr_external_v1.yaml).
Operational configuration: [vindr_adapter_v1.yaml](../configs/vindr_adapter_v1.yaml).
Implementation: [prepare_vindr.py](../src/data/prepare_vindr.py).
Reproduction: [README commands](../README.md#vindr-cxr-external-protocol-and-adapter-batches-4748-experiment-not-run).

Status: **passed; Batch 48 complete, pending user review** (2026-09-23).
The completed `prepare` receipt is
[adapter_preflight.json](../results/vindr_external_v1/adapter_preflight.json).
All 3,000 images passed decoding, PNG round-trip checks and integration with
the existing common loader. This is a data-only audit, with no detector
construction, checkpoint loading, predictions or external performance analysis.

## Prerequisite and publication boundaries

The initial checkout is `main` at
`1a7104d7a817b835f5a4d171b44d532fcbee9701`. Existing Batch 47/download-tooling
edits and unrelated untracked files are preserved. All 38 frozen dependencies
match the Batch 47 sidecar. The scientific config remains SHA-256
`9516d62b8164ca25794ad0be3ba3f13eec32e29b2377442194a3095b2671a19b`;
the protocol remains
`7b157a19550340d00d7a043ef4a380739d54faae5525a6ebba27ed1cd5ccff87`.
Neither file nor the freeze sidecar was amended.

Approved access, DUA acceptance and official download are supported by the
user's recorded Batch 47 attestation. The root is resolved exclusively through
the configured `VINDR_CXR_ROOT`; no personal path or directory search is built
into the adapter. Release metadata hashes and the source checksum-manifest hash
match v1.0.0. All 3,000 DICOM content hashes match its official-test inventory.
The supplied manifest's authenticity rests on the attested official download;
no authenticated fetch or terms acceptance was performed here.

At startup there was no configured private external prediction directory or
external aggregate-results directory, and the handoff records no preceding
external analysis. The adapter refuses existing prediction files or unexpected
external result artifacts. This gate is evidence about this repository's
recorded workflow, not proof about work outside the supplied workspace.

Restricted source data, PNGs, identifiers, row-level headers, manifest and
annotations remain in ignored `data/`. The adapter checks output containment
and `git check-ignore` before writing. The public summary contains aggregate
counts, dimensions, contract settings, software versions and hashes only.
No DICOMs, images or restricted annotation rows belong in a commit. No staging,
commit, push or publication was requested or performed.

## Cohort and annotation conversion

| Quantity | Count |
|---|---:|
| Official test study/images | 3,000 |
| Strict-target-positive images | 84 |
| Strict-target boxes | 95 |
| Strict-target-zero-box images | 2,916 |
| Invalid/out-of-bounds/zero-area target boxes | 0 |
| Exact duplicate target boxes | 0 |

“Study/image” means one released image ID, not a verified independent patient.
No defensible patient identifier or grouping is available.

The verified box schema is exactly
`image_id,class_name,x_min,y_min,x_max,y_max`. Image-label IDs uniquely match
the official test inventory; all box-row IDs belong to it. Only exact source
label `Lung Opacity` maps to target concept `Lung opacity` and canonical
category 1. No case folding, synonyms, global-label substitution or other
finding merges are performed. Every image without a strict target remains
negative, including images with other abnormalities. Those findings do not
create ignore regions. The 95 boxes are annotation targets, not 95 patients.

Images are ordered by source filename, with source annotation row order
retained within each image. Canonical COCO uses native width/height and
`[xmin, ymin, xmax-xmin, ymax-ymin]`, with no inclusive `+1`. The common
evaluator uses filename IDs including `.png` and `(height, width)` image sizes.
The private manifest also records original source IDs, target counts and
source/PNG SHA-256s. Empty negatives are present in both manifest and COCO.

## Actual DICOM semantics and preprocessing

| Header property | Observed official-test metadata |
|---|---|
| Polarity | 2,459 MONOCHROME2; 541 MONOCHROME1 |
| Dimensions | Heights 1,184–3,408; widths 1,120–3,320 pixels |
| Pixel storage | All unsigned, 16 bits allocated; valid HighBit = BitsStored − 1 |
| Bits stored | 10: 35 images; 12: 2,073; 14: 624; 16: 268 |
| Transfer syntax | 1,033 JPEG 2000 lossless; 1,966 implicit-VR little-endian; 1 explicit-VR little-endian |
| Rescale | 2,414 identity slope/intercept pairs (1/0); 586 absent pairs |
| Window center/width | Present on 2,941; absent on 59; values vary |
| Modality/VOI LUT sequences, VOI LUT function, presentation LUT shape, pixel padding | Absent |
| ViewPosition, PatientOrientation, ImageOrientationPatient | Absent on all 3,000 |
| Patient/study/series/SOP identity tags | No nonempty values established in Batch 47; no patient grouping inferred |

Decode the stored pixel array through pydicom, including lossless JPEG 2000.
Apply the existing shared `scale_radiograph_to_uint8`: float32, finite extrema,
specified nonfinite replacement, MONOCHROME1 inversion using `high+low-array`,
per-image min-max, `numpy.rint`, clip and uint8. Constant arrays map to zero;
an array with no finite pixels is an error. Save native-dimension lossless
grayscale PNG; the existing loader replicates channels to RGB.
PNG compression level 1 is operationally configured and checked by lossless
pixel round trip. An initial partial conversion using Pillow's default level
6 was interrupted for encoding speed; the complete run regenerates every PNG.
No preprocessing alternative or detector performance was compared.

Modality/rescale/window/LUT processing and padding masks remain deliberately
disabled as frozen. No rotation, flip, crop, histogram equalization or
test-time augmentation is introduced. Missing orientation tags do not justify
inventing an orientation or projection. Native pixel axes are preserved for
the annotations. Stored-pixel normalization is a research display transform,
not vendor display reproduction or quantitative physical-unit recovery.

Every image receives a checksum, header, decode, dimension and lossless PNG
round-trip check. The repeat/geometry sample is deterministic: first sorted
image in each transfer-syntax/polarity/bit-depth stratum plus all strict-target
positives. Repeat decoding must be pixel-identical. Torchvision's actual resize
and postprocess functions and Ultralytics' actual LetterBox instance mapping
and `scale_boxes` restoration check strict boxes plus asymmetric/corner probes.
Both square and automatic rectangular letterboxing are exercised. The configured
absolute coordinate tolerance is 0.001 native pixels. These are technical
transform checks without detector forward passes.

The completed technical sample contains 94 images. Repeat preprocessing was
pixel-identical; maximum inverse-coordinate error was
0.00019315886765980395 native pixels, below the 0.001-pixel tolerance. All
3,000 images decoded, with zero constant images and zero nonfinite pixels.
The decoder stack was pydicom 3.0.2, Pillow 12.3.0/OpenJPEG 2.5.4 and NumPy
2.4.4 on Python 3.11.15. Geometry checks used Torch 2.6.0+cpu, Torchvision
0.21.0+cpu and Ultralytics 8.4.110. No additional decoder package was needed.

## Frozen evaluator contract

| Setting | Value |
|---|---|
| Candidate/FROC collection floor | 0.00001 |
| AP score floor | 0.001 |
| Native NMS / common matching IoU | 0.50 / 0.50 |
| Detection cap | 100 per image |
| COCO AP | IoU 0.50:0.05:0.95; 101 recall points; all areas; maxDets [1,10,100] |
| FROC budgets | 0.125, 0.25, 0.5, 1, 2 FP/image |
| FROC denominators | 95 strict-target boxes; all 3,000 test images |
| Matching | Score-descending greedy, one-to-one, same class; stable tie handling |
| Threshold support | `score >= threshold`; every distinct retained score plus frozen endpoints |
| FROC budget selection | Maximum observed sensitivity at/below budget; no interpolation |
| Historical RSNA transport | Faster R-CNN 0.69; YOLO11s 0.05 |
| Post-hoc n=5 sensitivity | Faster R-CNN 0.70; YOLO11s 0.01 |

All ten runs and mandatory CUDA/AMP remain Batch 49 requirements. Preparation
uses the CPU environment only; no inference-precision claim is made here.
The historical FP32 versus planned AMP YOLO caveat remains unchanged.
Native candidate-floor equality behavior, actual convolution AMP dtypes,
checkpoint identities and prediction-bundle validation must be checked by the
separately authorized Batch 49 collection implementation. Batch 48 tests the
data and geometric interfaces and does not claim an inference smoke pass.

## Verification and review gate

The synthetic adapter suite covers exact ontology filtering, negatives with
other findings, malformed/duplicate boxes, source-order/COCO conversion,
common evaluator false positives, both polarities, signed and unsigned pixels,
constant/nonfinite behavior, determinism, native geometry, release/config
tampering, access/root gates and full synthetic prepare/preflight integration.
It contains no restricted source fixtures.

The combined focused check passes 71 tests with one declared metadata-only
environment skip. Ruff lint/format checks pass. The unchanged internal paper
verifier passes 66 numerical claims and its semantic guards; the scientific
verifier passes 72 artifacts, 346 present inputs, zero unavailable inputs and
201 result references. An initial pytest invocation encountered existing
temporary-directory permissions; the documented fresh workspace-local
`--basetemp` and disabled cache resolve that environment issue.

After the quota interruption, all 3,000 saved PNG hashes were independently
rechecked against the completed manifest. The manifest/COCO hashes, current
adapter/config hashes and all 38 frozen dependencies matched the receipt;
fresh source-to-COCO conversion agreed exactly with the saved annotations.
Private and aggregate summaries also agreed, and no prediction output existed.

Batch 48 stops for user review. A passing adapter demonstrates ingestion and
coordinate compatibility; it provides no evidence of external detector
performance. Do not proceed to Batch 49 automatically, and do not alter v1 in
response to future predictions.
