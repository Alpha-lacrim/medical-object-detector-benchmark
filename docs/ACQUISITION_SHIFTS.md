# Radiography-Motivated Synthetic Acquisition/Display Sensitivity

## Question, frozen scope, and Batch 32 correction

This checkpoint-conditional analysis asks how the primary seed-17 Faster R-CNN
and YOLO11s models respond to declared synthetic changes applied to the stored
DICOM arrays before the project's canonical per-image min-max conversion. It
uses the frozen 300-image Phase 6 subset (183 NIH patient groups and 111 opacity
boxes) and the existing 20 prediction bundles. It does not train, fine-tune,
select a threshold, or estimate performance at another site.

Batch 32 corrected the physical interpretation without changing the historical
pixels, predictions, or metrics. The earlier umbrella description
"acquisition-physics-motivated" was too strong. The conditions now have
separate classes:

- **A:** DICOM display-transform sensitivity;
- **B:** generic intensity perturbation;
- **C:** physics-motivated proxy; and
- **D:** physically unsupported for these stored values.

The old Phase 22 table and summary remain byte-identical historical artifacts.
The canonical, relabeled performance table is
[`radiography_synthetic_shift_results.csv`](../results/tables/radiography_synthetic_shift_results.csv),
and Batch 32 provenance is
[`phase32_acquisition_shift_audit/summary.json`](../results/logs/phase32_acquisition_shift_audit/summary.json).
No GPU inference was rerun because the transforms remain valid synthetic stress
tests; only the former physics/dose interpretation failed the metadata audit.

## Current DICOM-standard interpretation

The primary authority is the current NEMA DICOM Standard observed during this
audit as PS3.3/PS3.4 2026c:

- [PS3.3 C.7.6.3, Image Pixel Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html);
- [PS3.3 C.8.11.3, Pixel Intensity Relationship and grayscale transformations](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.11.3.html);
- [PS3.3 C.11.1, Modality LUT Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.1.html);
- [PS3.3 C.11.2, VOI LUT Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html);
- [PS3.3 A.8, Secondary Capture Image IODs](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html);
- [PS3.3 C.8.6, Secondary Capture Modules](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.6.html); and
- [PS3.4 B.5, Standard SOP Classes](https://dicom.nema.org/medical/dicom/current/output/chtml/part04/sect_B.5.html).

Photometric Interpretation specifies display polarity, not the relationship of
stored values to incident X-ray intensity. `MONOCHROME2` means the minimum
sample is intended to display as black after VOI processing; `MONOCHROME1`
means it is intended to display as white. Pixel Intensity Relationship and its
Sign, where defined by a modality-specific IOD such as DX, describe the
relationship to incident beam intensity but are not presentation transforms.
Modality LUT/rescale maps stored values to modality values; Window Center/Width
or a VOI LUT then maps those values toward display.

### What the 300 files actually contain

The complete per-image table is
[`acquisition_shift_dicom_metadata_audit.csv`](../results/tables/acquisition_shift_dicom_metadata_audit.csv).
It contains every requested value and an explicit `<field>_missing` flag for
each field. All 300 rows have the same relevant profile:

| Finding | Count |
|---|---:|
| SOP Class = Secondary Capture Image Storage (`1.2.840.10008.5.1.4.1.1.7`) | 300 |
| Modality = `CR` | 300 |
| Photometric Interpretation = `MONOCHROME2` | 300 |
| Conversion Type = `WSD` (workstation) | 300 |
| Lossy Image Compression = `01`; Method = `ISO_10918_1` | 300 |
| Pixel Intensity Relationship / Sign present | 0 / 0 |
| Rescale Slope / Intercept / Type present | 0 / 0 / 0 |
| Modality LUT / VOI LUT present | 0 / 0 |
| Window Center / Width / VOI function present | 0 / 0 / 0 |
| Presentation Intent Type / Presentation LUT Shape present | 0 / 0 |
| Acquisition processing description/code or Derivation Description present | 0 |
| Eligible for an X-ray-signal Poisson proxy | 0 |

`Modality=CR` identifies the source process recorded in these objects; it does
not change their SOP Class into the Computed Radiography Image IOD. The current
standard describes single-frame Secondary Capture as modality-independent and
relatively unconstrained, and `WSD` as workstation conversion. Together with
the lossy-compression history, these headers establish that the stored 8-bit
values have been converted/processed. They do not establish whether the values
are linear or logarithmic in incident X-ray intensity, nor provide a reliable
inverse to such a scale. They may be display-ready or otherwise processed, but
the missing intent/processing fields do not distinguish those possibilities.

## MONOCHROME polarity and DICOM `LINEAR` implementation

The canonical scaler computes a finite per-image min-max map and, when enabled,
inverts `MONOCHROME1` relative to `MONOCHROME2`. Performing that inversion
before a full-range linear min-max map is algebraically equivalent to applying
the polarity reversal afterward for non-constant arrays. All audited files are
`MONOCHROME2`, so no inversion occurred in this subset.

The `dicom_linear_window` implementation exactly follows the default `LINEAR`
function in PS3.3 C.11.2.1.2.1, including `center - 0.5`, the `width - 1`
denominator, the asymmetric upper boundary, floating-point calculation, and
the `width=1` threshold case. Regression tests reproduce the standard's
published examples for `(center,width)=(2048,4096)`, `(2048,1)`, and `(0,100)`,
in addition to comparison with pydicom's implementation.

Because no native Modality LUT/rescale or VOI attributes are present, the four
window settings are synthetic alternatives on stored values. The per-image
full-input-range window is the identity reference; it is not a recovered
vendor preset.

## Condition-by-condition scientific classification

| Conditions | Class | Defensible use | Excluded claim |
|---|---|---|---|
| Four DICOM `LINEAR` center/width alternatives | **A** | Sensitivity to exact standard-defined display transforms with synthetic parameters | Recovered vendor display settings or scanner protocol |
| Three signal-dependent Poisson-like count budgets | **D**, retained only as **B** | Generic signal-dependent intensity-noise sensitivity | Dose reduction, quantum-noise proxy, or validated low-dose acquisition simulation |
| Three finite Gaussian blur kernels | **C** | Generic blur/spatial-resolution sensitivity | A particular detector MTF, reconstruction kernel, or scanner process |

The legacy machine identifiers for the Poisson-like conditions contain
`poisson_dose_*` so the exact historical RNG streams and bundle filenames remain
traceable. They are not scientific dose labels. The displayed names and the
canonical table use *relative count budget* and *Poisson-like noise*.

## Effect of canonical per-image min-max preprocessing

For each image and condition, the audit computed MAE and RMSE before scaling
and after the shared 8-bit min-max scaler. NMAE before scaling is normalized by
the possible 8-bit stored range; NMAE afterward is normalized by 255. The full
3,000-row detail is
[`acquisition_shift_preprocessing_per_image.csv`](../results/tables/acquisition_shift_preprocessing_per_image.csv),
with the ten-condition summary in
[`acquisition_shift_preprocessing_summary.csv`](../results/tables/acquisition_shift_preprocessing_summary.csv).

| Condition | Median NMAE before | Median NMAE after | Median after/before | Min-max effect |
|---|---:|---:|---:|---|
| Center -15% | 0.150509 | 0.082441 | 0.518 | Partly cancelled (48.2%) |
| Center +15% | 0.129007 | 0.073023 | 0.593 | Partly cancelled (40.7%) |
| Width x0.75 | 0.062438 | 0.056985 | 1.004 | Not materially cancelled |
| Width x1.25 | 0.042890 | 0.000000 | 0.000 | Almost completely cancelled; 264/300 exactly identical |
| Poisson-like 0.50 | 0.011388 | 0.012487 | 0.998 | Not materially cancelled |
| Poisson-like 0.25 | 0.016111 | 0.017441 | 1.000 | Not materially cancelled |
| Poisson-like 0.125 | 0.022782 | 0.024329 | 1.001 | Not materially cancelled |
| Gaussian 3x3, sigma 0.5 | 0.001317 | 0.003474 | 2.455 | Re-stretched/amplified by min-max |
| Gaussian 5x5, sigma 1.0 | 0.003889 | 0.010352 | 2.488 | Re-stretched/amplified by min-max |
| Gaussian 9x9, sigma 2.0 | 0.006493 | 0.019983 | 3.018 | Re-stretched/amplified by min-max |

The configured labels call a median residual ratio at most 0.05 "almost
completely cancelled" and a median attenuation of at least 20% "partly
cancelled." A ratio above one is not extra scanner blur: it means per-image
min-max re-expanded the blurred image's reduced range, increasing normalized
pixel differences at model input.

## Descriptive performance sensitivity

The detector metrics are unchanged from Phase 22. For each metric and detector:

```text
performance ratio = performance_shifted / performance_clean
DSI = 1 - performance ratio
```

DSI is an unclipped descriptive performance-retention/domain-sensitivity index.
Positive values indicate degradation, zero indicates no change, and negative
values indicate a finite-sample improvement. DSI does **not** estimate real
inter-site transportability, scanner generalization, clinical robustness, or
safety.

| Synthetic condition | Faster R-CNN shifted mAP | Faster R-CNN DSI | YOLO11s shifted mAP | YOLO11s DSI |
|---|---:|---:|---:|---:|
| DICOM center -15% | 0.141434 | 0.043089 | 0.067317 | 0.117673 |
| DICOM center +15% | 0.135060 | 0.086212 | 0.075613 | 0.008936 |
| DICOM width x0.75 | 0.137225 | 0.071562 | 0.077304 | -0.013232 |
| DICOM width x1.25 | 0.147820 | -0.000119 | 0.076226 | 0.000906 |
| Poisson-like relative count 0.50 | 0.120486 | 0.184815 | 0.054714 | 0.282860 |
| Poisson-like relative count 0.25 | 0.115599 | 0.217877 | 0.046235 | 0.393989 |
| Poisson-like relative count 0.125 | 0.101314 | 0.314529 | 0.042981 | 0.436641 |
| Gaussian 3x3, sigma 0.5 | 0.139986 | 0.052880 | 0.070779 | 0.072296 |
| Gaussian 5x5, sigma 1.0 | 0.131398 | 0.110987 | 0.063844 | 0.163198 |
| Gaussian 9x9, sigma 2.0 | 0.111069 | 0.248532 | 0.054713 | 0.282869 |

Within the declared Poisson-like and Gaussian series, mAP DSI increased with
perturbation strength for both frozen checkpoints. This is an internal
sensitivity pattern, not evidence that the count budgets correspond to dose or
that the kernels represent a scanner.

## Distinction from post-conversion digital corruptions

| Analysis | Stage | Supports | Does not support |
|---|---|---|---|
| Radiography-motivated synthetic acquisition/display sensitivity | Stored Secondary Capture arrays before canonical min-max scaling | Classes A/B/C under the exact declared transforms, plus explicit class-D rejection of the dose interpretation | Calibrated dose, scanner physics, recovered presets, site transportability, or clinical robustness |
| Digital common-corruption sensitivity | Already converted uint8 PNGs | Sensitivity to declared brightness, noise, blur, motion blur, and JPEG changes | DICOM display behavior, acquisition protocol changes, or scanner physics |

Both analyses repeatedly transform the same internal sample. Neither introduces
a new scanner, institution, population, or prospective protocol.

## Reproduction and provenance

```powershell
& $benchmarkPython -m pytest tests/test_radiography_shifts.py tests/test_prepare.py -q
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode preflight
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode audit
```

`preflight` and `audit` are CPU-only. The audit verifies all raw/header/PNG
identities, the historical table (`2cd14283...`), historical summary
(`4782ac7c...`), and all 20 frozen prediction bundles (aggregate manifest hash
`28a2e10a...`). The `smoke` and `run` modes remain historical inference paths;
they were not invoked in Batch 32 because no transform implementation or model
input needed correction.
