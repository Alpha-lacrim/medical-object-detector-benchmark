# Acquisition-Shift Sensitivity for Planar Radiography

## Question and frozen scope

This checkpoint-only analysis asks how the primary seed-17 Faster R-CNN and
YOLO11s models respond when acquisition-physics-motivated transformations are
applied before the project's DICOM-to-PNG conversion. It uses the exact frozen
300-image Phase 6 robustness subset, its 111 opacity boxes and both existing
seed-17 checkpoints. It performs no training, fine-tuning, threshold selection,
or checkpoint mutation.

The raw-data prerequisite passed. All 300 manifest studies have a local source
DICOM. Every file is an 8-bit unsigned `CR` planar radiograph with
`MONOCHROME2` Photometric Interpretation. None contains Window Center/Width, a
VOI LUT Sequence, a Modality LUT, Rescale Slope/Intercept, or pixel-padding
metadata. Reapplying the project's canonical raw conversion produces pixels
identical to every corresponding processed PNG. The aggregate raw-file
fingerprint is
`7266ec165d4db6d7e7f855e4d1524861e91392827fa32640183261ce09997cbe`.

These observations matter for interpretation: the VOI settings below are
controlled sensitivity settings, not recovered vendor display presets. The
stored pixels also do not provide calibrated exposure or detector-response
metadata. No modality-specific quantitative attenuation scale is used or
implied; these are planar radiographs.

## Raw-array transformations

### DICOM VOI Window Center/Width

The implementation follows the default `LINEAR` function in
[DICOM PS3.3 C.11.2.1.2.1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html),
including the standard's `center - 0.5` boundary and `width - 1` denominator.
For each image with lowest and highest used input values `x1` and `x2`, the
standard defines the full-input-range window as:

```text
center = (x1 + x2 + 1) / 2
width  = x2 - x1 + 1
```

This is the clean identity reference because the files contain no native VOI
choice. The four declared alternatives shift the center by -15% or +15% of
that width, or multiply the width by 0.75 or 1.25. Values below/above the
window map to the minimum/maximum output as specified by DICOM. The code is
unit-tested against pydicom 3.0.2's standard-backed `apply_windowing` result.

### Signal-dependent Poisson noise

The raw unsigned stored-value range is normalized to a nonnegative signal. A
deterministic Poisson draw is then made with expected counts proportional to
that signal. The three synthetic dose fractions are 0.50, 0.25, and 0.125 of
a declared 4,096-count full-scale reference; the result is rescaled to the
stored-value range and clipped to the detector range. Thus noise variance is
signal-dependent and rises as the count budget falls. Seeds are derived from
the experiment seed, image ID, and condition ID, so both detectors receive the
same shifted pixels.

The count reference and fractions are stress-test parameters, not measured
patient exposure, entrance air kerma, or a calibrated dose conversion. The
model omits read noise, scatter, detector response, postprocessing, and other
acquisition-chain effects.

### Gaussian detector/processing blur

Gaussian filtering is applied to the raw array using explicitly finite 3x3,
5x5, and 9x9 kernels with sigma 0.5, 1.0, and 2.0 pixels, respectively. These
model ordered detector/processing-resolution variation. They are not measured
scanner modulation-transfer functions or vendor reconstruction kernels.

After every shift, the array is passed through the shared
`scale_radiograph_to_uint8` function used by the original conversion. This
per-image min-max scaling produces the valid 8-bit inputs consumed by both
models. It can cancel a purely affine VOI change that does not clip the image;
the near-neutral wide-window result is therefore an expected interaction with
the frozen preprocessing contract, not evidence that window choice is
generally irrelevant.

## Endpoint and results

For each metric and detector:

```text
performance ratio = performance_shifted / performance_clean
DSI = 1 - performance ratio
```

DSI is not clipped. Positive values indicate degradation, zero indicates no
change, and negative values indicate a finite-sample improvement. The primary
performance endpoint is mAP@0.5:0.95. Its clean values on this exact subset are
0.147802 for Faster R-CNN and 0.076295 for YOLO11s.

| Shift | Faster R-CNN shifted mAP | Faster R-CNN DSI | YOLO11s shifted mAP | YOLO11s DSI |
|---|---:|---:|---:|---:|
| VOI center -15% width | 0.141434 | 0.043089 | 0.067317 | 0.117673 |
| VOI center +15% width | 0.135060 | 0.086212 | 0.075613 | 0.008936 |
| VOI width x0.75 | 0.137225 | 0.071562 | 0.077304 | -0.013232 |
| VOI width x1.25 | 0.147820 | -0.000119 | 0.076226 | 0.000906 |
| Poisson dose fraction 0.50 | 0.120486 | 0.184815 | 0.054714 | 0.282860 |
| Poisson dose fraction 0.25 | 0.115599 | 0.217877 | 0.046235 | 0.393989 |
| Poisson dose fraction 0.125 | 0.101314 | 0.314529 | 0.042981 | 0.436641 |
| Gaussian 3x3, sigma 0.5 | 0.139986 | 0.052880 | 0.070779 | 0.072296 |
| Gaussian 5x5, sigma 1.0 | 0.131398 | 0.110987 | 0.063844 | 0.163198 |
| Gaussian 9x9, sigma 2.0 | 0.111069 | 0.248532 | 0.054713 | 0.282869 |

Within the declared Poisson and Gaussian series, mAP DSI increases as the
shift becomes stronger for both detectors. YOLO11s has the larger DSI at every
Poisson level and every blur kernel. The VOI conditions are smaller and not
ordered as a universal severity curve; they change which ends of each image's
stored-value range clip, and the mandatory downstream min-max scaling changes
their effective model input. These are descriptive results from one sample and
one checkpoint per detector, not inferential evidence of a detector-family
effect.

The complete 20-row table is
`results/tables/acquisition_shift_results.csv`. It includes all seven unified
metrics, clean and shifted performance, ratios, DSI values, operating-point
counts, transformation parameters, inference times, and hashes of all 20
300-image prediction bundles. Full provenance is in
`results/logs/phase22_acquisition_shifts/summary.json`.

## Distinction from digital common-corruption robustness

This analysis and Phase 6 answer different, deliberately limited questions:

| Analysis | Transformation stage | What it supports | What it does not support |
|---|---|---|---|
| Acquisition-shift sensitivity (`acquisition_shift_results.csv`) | Raw DICOM stored arrays before the canonical 8-bit conversion | Sensitivity to exact DICOM LINEAR VOI choices and acquisition-chain proxies for quantum noise and detector/processing blur | Calibrated dose response, recovered vendor presets, scanner validation, site transportability, or clinical robustness |
| Digital common-corruption robustness (`robustness_curves.csv`) | Already converted uint8 PNGs | Sensitivity to generic brightness, Gaussian/impulse noise, blur, motion blur, and JPEG corruptions | Scanner physics, DICOM VOI behavior, acquisition protocol change, or clinical robustness |

The acquisition-shift analysis is more closely motivated by radiography
formation and display processing than the generic digital-corruption grid, but
it remains a synthetic internal stress test over the same images. Neither
analysis introduces a new scanner, site, patient population, prospective
workflow, or clinically calibrated acquisition protocol. Consistent with
`docs/LIMITATIONS.md`, both measure scoped model sensitivity, not clinical
robustness, safety, or deployment readiness.

## Reproduction

```powershell
& $benchmarkPython -m pytest tests/test_radiography_shifts.py tests/test_prepare.py -q
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode preflight
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode smoke
& $benchmarkPython -m src.robustness.radiography_shifts --config configs/acquisition_shifts.yaml --mode run
```

Preflight reads all 300 ignored raw files, validates their aggregate metadata,
and proves clean raw-to-PNG pixel identity before CUDA initialization. Smoke
artifacts are isolated under `results/logs/phase22_acquisition_shifts/smoke/`.
The full run is condition-resumable and loads only the existing checkpoints.
