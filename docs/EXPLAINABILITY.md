# Explainability Analysis

## Scope and provenance

Phase 7 uses only the selected seed-17 Faster R-CNN and YOLO11s checkpoints and
the exact Phase 6 robustness manifest. The manifest contains 300 held-out test
images, including 68 images with 111 ground-truth Lung Opacity boxes and 232
images without a Lung Opacity box. Its SHA-256 is
`63b4dd706dc2fcd8a528a935957ccb318ed2cde51a6fd87d20feca348d00fc5e`.
The box-based quantitative analysis uses all 111 boxes. The 232 box-negative
images cannot contribute an energy-in-box or pointing-game value; they remain
eligible for the paired qualitative false-positive analysis.

The complete run is configured by `configs/explainability.yaml`. It validates
the finished Phase 6 config, summary, sample manifest, clean prediction bundles,
training configs, and both checkpoint hashes before initializing CUDA. The
machine-readable audit is `results/logs/phase7_explainability/summary.json`.

## Grad-CAM protocol

Both detectors use ordinary ReLU Grad-CAM at a stride-16, 40 by 40 backbone
feature tensor for the common 640-pixel detector input:

- Faster R-CNN: ResNet-50 `backbone.body.layer3`, the C4 output before the FPN;
- YOLO11s: `model.6`, the P4 backbone block before the stride-32 stage and PAN
  neck.

This matches spatial stride and pre-neck semantic depth rather than pretending
that unlike module names denote equivalent layers. For each ground-truth box,
the target is the foreground probability of the low-threshold retained
candidate with the highest IoU to that box, with score as a deterministic
tie-break. Both detectors retain at most 100 candidates after their configured
0.50 NMS. The candidate is selected with detached geometry, while the selected
post-softmax Faster R-CNN score or post-sigmoid YOLO score remains
differentiable. NMS and candidate selection are discrete and are not explained
by the resulting gradient.

An operating-point true positive targets an emitted detection. An operating-
point false negative instead targets the best available pre-threshold proxy and
is explicitly labeled `miss_proxy_candidate`; it is not presented as an
explanation of a detection that did not exist. Multiple boxes in one image are
explained separately. A proxy candidate can therefore be reused when nearby
boxes prefer the same retained candidate.

The nonnegative CAM is bilinearly restored to original image space. The primary
metric is the fraction of total CAM mass whose pixel centers fall inside the
ground-truth box. The secondary pointing game records whether the single hottest
pixel falls inside. Rasterized box-area fraction is the no-localization
reference. A zero-energy CAM is undefined rather than silently scored as zero;
one of 111 Faster R-CNN maps was zero and is reported but excluded from its
metric denominator. YOLO11s had no zero maps.

## Quantitative results

| Detector | Valid / boxes | Mean energy in box | Box-area reference | Mean lift over area | Pointing accuracy |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 110 / 111 | 0.0869 | 0.0713 | +0.0156 | 0.1091 |
| YOLO11s | 111 / 111 | 0.0975 | 0.0718 | +0.0257 | 0.1261 |

The image-macro mean energy scores are 0.0896 for Faster R-CNN and 0.0940 for
YOLO11s. On the 110 targets with valid maps from both models, YOLO has higher
energy in 76 and Faster R-CNN in 34; mean Faster-minus-YOLO energy is -0.0091
and the median difference is -0.0240. Paired pointing outcomes are: 1 both hit,
11 Faster-only, 12 YOLO-only, and 86 neither. These are descriptive Phase 7
results, not inferential tests; uncertainty and paired testing belong to Phase
8.

Operating status changes the interpretation substantially. Faster R-CNN's 76
valid true-positive maps average 0.1122 energy and 0.1447 pointing accuracy,
while its 34 miss-proxy maps average only 0.0303 and 0.0294. YOLO's 19 true-
positive maps average 0.1448 and 0.1579; its much larger set of 92 miss-proxy
maps averages 0.0877 and 0.1196. These status subsets are not the same paired
boxes, so their means must not be read as a controlled architecture effect.

## Where are the models looking?

Faster R-CNN usually produces coarser, more clustered hotspots. Some mass falls
within the boxed opacity, especially for confident true positives, but strong
peaks also occur over the neck, shoulders, mediastinum, image boundary, and
radiographic markers or support devices. Its miss-proxy maps lose most of even
that weak box concentration.

YOLO11s is systematically more diffuse and punctate at the matched layer. Its
maps spread across both lungs, the chest wall, shoulders, abdomen, edges, text
markers, and devices. It places slightly more total mass inside boxes on this
sample, including on many low-confidence miss proxies, but that does not mean
its attention is cleaner: image-wide hotspots make the single hottest point
miss almost as often as Faster R-CNN's.

Therefore, neither model is reliably focused on the actual boxed finding. Both
are often using broader thoracic context and are visibly sensitive to borders,
markers, devices, and unrelated anatomy. The systematic difference is modest:
Faster R-CNN is more spatially clustered but not consistently lesion-centered;
YOLO is more dispersed yet has a small descriptive advantage in energy-in-box.
The false-positive panels are all from the `No Lung Opacity / Not Normal`
stratum, so visible non-opacity abnormalities may be real. Without annotations
for those alternative findings, the heatmaps cannot distinguish clinically
meaningful context from a confounding artifact.

## Qualitative selection and caveats

The figures are selected from frozen predictions without consulting CAM values:
three unique shared true positives with the highest minimum detector IoU, three
unique box-negative images with the highest minimum false-positive score across
models, and three unique shared false negatives at the 0.2, 0.5, and 0.8
quantiles of mean proxy IoU. Green is ground truth and cyan is the exact
candidate whose score is differentiated.

Grad-CAM is a coarse association map, not a causal account or a clinical
reasoning trace. The ground-truth-guided proxy choice makes the quantitative
question conditional—"given the candidate most associated with this box, where
does its class-score gradient concentrate?"—and does not explain why the full
detector missed the finding. A rectangle is also a loose lesion surrogate and
makes pointing easier as its area grows. Finally, Torchvision warns that CUDA
ROI Align backward is not bitwise deterministic; the run uses the project's
seeded deterministic-warn-only policy and records its environment, but Faster
R-CNN CAM bytes are not guaranteed identical across hardware/library reruns.

## Reproduction

```powershell
$benchmarkPython = 'C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe'
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode preflight
& $benchmarkPython -m pytest tests/test_gradcam.py tests/test_pointing_game.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode smoke
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode run
```

The per-box evidence is `results/tables/gradcam_localization_per_target.csv`,
the aggregate table is `results/tables/gradcam_localization_summary.csv`, and
the exact paired case manifest is
`results/tables/gradcam_qualitative_cases.csv`.
