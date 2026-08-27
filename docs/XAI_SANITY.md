# Grad-CAM Sanity Checks

## Scope and provenance

Batch 21 tests whether the project's Grad-CAM maps respond to model parameters
and image spatial structure. It uses only the trained seed-17 Faster R-CNN and
YOLO11s checkpoints; it performs no training or checkpoint update.

The 50-image analysis set is nested inside the frozen 300-image Phase 6
robustness pool, whose manifest SHA-256 remains
`63b4dd706dc2fcd8a528a935957ccb318ed2cde51a6fd87d20feca348d00fc5e`.
Proportional largest-remainder allocation followed by seeded sampling without
replacement yields 11 `Lung Opacity`, 22 `No Lung Opacity / Not Normal`, and 17
`Normal` studies. The subset contains 41 NIH patient groups and 18 opacity
boxes. It is not a new draw from the 750-image test split. The exact nested
manifest and all input hashes are recorded in
`results/logs/phase21_xai_sanity/summary.json`.

## Method

One image-level reference region is defined for every study, including
box-negative studies: the trained detector's highest-score retained candidate
on the unmodified radiograph. This selection uses no ground-truth box and is
fixed before either randomization. Both sanity tests use ordinary ReLU
Grad-CAM at the same stride-16, 40 by 40 layers as Phase 7:

- Faster R-CNN `backbone.body.layer3`;
- YOLO11s `model.6`.

For a controlled comparison that does not depend on a fully randomized
proposal generator emitting a valid post-NMS box, the differentiated target is
the pre-activation foreground score at the fixed trained reference region.
Faster R-CNN rescores the region through its ROI classifier. YOLO11s uses the
raw prediction anchor whose decoded center is closest to the reference-region
center. This preserves one spatial target across conditions. It differs from
Phase 7's post-activation, highest-IoU-to-ground-truth target, so this experiment
tests the sensitivity of the same Grad-CAM layers and construction rather than
recomputing Phase 7's energy-in-box estimand.

The two randomizations are:

1. **Parameter randomization.** A deep copy of each trained model is made on
   CPU. Every module weight tensor is reinitialized with
   `torch.nn.init.xavier_normal_` (one-dimensional weights are treated as row
   vectors), every bias tensor is set to zero, and non-weight/non-bias buffers
   are preserved. The initialization seed is 17. Randomized-weight scoring runs
   in float32 to avoid AMP saturation from the deliberately untrained weights.
2. **Data randomization.** Each RGB pixel vector is permuted without replacement
   over spatial locations using an image-specific seed derived from SHA-256 of
   the configured seed and image ID. The exact pixel-vector multiset is
   preserved, spatial anatomy is destroyed, and both detectors receive the
   same shuffled image. The trained weights are unchanged.

For each detector and test, full-resolution Pearson correlation is computed
between the trained and randomized maps. Zero, constant, non-finite, or failed
maps are excluded and counted:

\[
C_{\mathrm{sanity}} = \frac{1}{K}\sum_{k=1}^{K}
\operatorname{Corr}(M_{\mathrm{trained}}^{(k)},
M_{\mathrm{randomized}}^{(k)}),
\]

where `K` is the number of images with two valid, nonconstant maps. A
correlation at least 0.50 is predeclared as a descriptive sanity failure
(substantial persistence after randomization); this threshold is an audit rule,
not a literature-wide clinical cutoff.

## Findings

| Detector | Test | Valid `K` / 50 | Randomized-map failures | `C_sanity` | Median `r` | Range | Sanity failures (`r >= 0.50`) |
|---|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | Parameter randomization | 50 / 50 | 0 (0%) | 0.0014 | 0.0093 | -0.1948 to 0.2081 | 0 / 50 (0%) |
| Faster R-CNN | Data randomization | 43 / 50 | 7 (14%) | -0.0159 | -0.0241 | -0.1130 to 0.1520 | 0 / 43 (0%) |
| YOLO11s | Parameter randomization | 46 / 50 | 4 (8%) | 0.0234 | 0.0216 | -0.3739 to 0.3193 | 0 / 46 (0%) |
| YOLO11s | Data randomization | 50 / 50 | 0 (0%) | -0.0034 | -0.0023 | -0.1938 to 0.1105 | 0 / 50 (0%) |

All trained maps are valid. The 11 excluded randomized maps are zero or
constant ReLU Grad-CAM outputs; none is imputed as zero correlation. Across the
remaining pairs, all four mean correlations are close to zero and no image
crosses the 0.50 sanity-failure threshold. The qualitative panel uses one
lexicographically first nested-subset case per stratum, selected before seeing
the correlations. It visibly agrees with the numerical result: randomized
weights and shuffled spatial structure produce markedly different maps.

## Meaning for the existing Grad-CAM claim

**On balance, this reinforces the project's existing framing of Grad-CAM as a
failure-analysis tool and does not upgrade it to evidence of clinical
reasoning.** The near-zero correlations show that these maps are sensitive to
learned weights and to the spatial arrangement of the input. That passes a
necessary basic sanity check and argues against the narrower concern that the
reported maps are largely invariant architecture templates or input-independent
edge patterns.

Sensitivity is not clinical validity. The test does not show that the learned
signal is causal, medically appropriate, or localized to the annotated
opacity. It does not explain proposal generation, NMS, or box regression. The
original Phase 7 finding—weak energy-in-box and pointing accuracy with frequent
activation on anatomy, borders, markers, and devices—therefore remains the
relevant evidence about localization quality. The sanity result makes those
maps more defensible for auditing model-specific failures, but it cannot make
them a clinical-reasoning trace.

The result also adds qualifications. It uses one random initialization, one
seed-17 checkpoint per detector, 50 images from 41 patient groups, a severe
out-of-distribution pixel permutation, and no inferential interval. The fixed
reference-region/pre-activation target is necessary for a complete detector
randomization grid but is not identical to Phase 7's ground-truth-associated
post-activation target. Seven Faster R-CNN shuffled-input maps and four
YOLO11s randomized-weight maps are non-estimable. Finally, correlations are
computed after interpolating coarse 40 by 40 CAMs to 1024 by 1024, so they
remain layer-, target-, and interpolation-dependent descriptive measurements.

## Reproduction

```powershell
$benchmarkPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $benchmarkPython -m pytest tests/test_xai_sanity.py tests/test_gradcam.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode preflight
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode run
```

The aggregate result is
`results/tables/gradcam_sanity_summary.csv`, the complete per-image audit is
`results/tables/gradcam_sanity_per_image.csv`, and the qualitative comparison is
`results/figures/gradcam_sanity_panel.png`.
