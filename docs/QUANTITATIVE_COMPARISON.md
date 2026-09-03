# Unified quantitative comparison

## Scope and freeze point

Phase 5 compares the selected validation-best checkpoint from each training seed
on the untouched 750-image RSNA test split. The seed grid was fixed before the
additional runs as 17, 42, 137, 271, and 314. The additional configs change
only RNG state and artifact identity; detector-specific hyperparameters remain
equal to the accepted Batch 2 and Batch 3 configs. Training and early stopping
continue to use only train and validation data. `src/evaluate.py` is the first
model entry point authorized to open `instances_test.json`.

## One prediction-to-metric path

Both checkpoint adapters emit the same immutable `ImagePrediction` records:
original-image `xyxy` boxes, canonical COCO category IDs, confidence scores,
image filename, and original height/width. Faster R-CNN and YOLO11s retain their
configured native heads and NMS, but neither framework's mAP implementation is
used for the comparison. The records from both are passed to the same functions
in `src/meddet_benchmark/`:

- official `pycocotools.COCOeval` bounding-box AP at IoU 0.50 and at IoU
  0.50:0.95;
- one score-ordered, class-aware greedy matcher for precision, recall, F1,
  localization IoU, and box Dice; and
- the same original Phase 5 score threshold (0.25), matching IoU (0.50), COCO
  minimum score (0.001), and maximum detections per image (100).

IoU and Dice are means over matched true-positive boxes at the frozen operating
point. They measure localization quality conditional on detection and must be
read with recall; missed targets do not receive an artificial zero-sized box.
Box Dice is derived from each matched box IoU as `2 × IoU / (1 + IoU)`.

Each seed also writes a deterministic gzip-compressed prediction bundle with the
checkpoint hash, test-annotation hash, thresholds, raw post-NMS prediction
records, per-image match results, and COCO results. Phase 8 uses these bundles
as frozen input evidence for the paired analysis in
`docs/STATISTICAL_ANALYSIS.md`.

## Compute measurements

The comparison table joins each test result to its completed training summary
and synchronized batch-1 compute profile. FPS and mean/p50/p95 inference time
use 10 warm-up and 100 timed images with CUDA synchronization. Parameter count,
trainable parameter count, registered-operation GFLOPs, checkpoint size,
epoch-loop training time, and peak allocated training GPU memory retain their
run-level provenance and checkpoint hash. The full-test prediction pass also
records synchronized inference seconds and FPS as an audit measure.

FLOPs are implementation estimates rather than hardware timings. Unsupported
operations differ by architecture, so latency/FPS are the runtime-relevant
efficiency measures. Faster R-CNN preprocessing/resizing remains inside its
model forward, while the YOLO profile performs resize/tensor conversion before
the timed forward-plus-NMS interval; this inherited framework asymmetry is
reported rather than treated as an architecture-only difference.

### GFLOP counting contract and sanity check

The frozen values are 450.7637248 GFLOPs/image for Faster R-CNN and
21.4198784 GFLOPs/image for YOLO11s, a 21.04-fold registered-operation gap.
They were produced under the locked PyTorch 2.6.0+cu124 environment by
`torch.utils.flop_counter.FlopCounterMode`, in evaluation and inference mode,
on batch 1 and the first validation image. Both model inputs become 640 x 640;
AMP changes dtype, not the dense shape-based operation formulas. The values in
the CSVs remain unchanged.

The [PyTorch 2.6 counter source](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/flop_counter.py)
registers formulas for convolution, `mm`, `addmm`, `bmm`, `baddbmm`, and several
scaled-dot-product/flash/efficient-attention forward and backward operators.
Its convolution and matrix formulas use two FLOPs per multiply-add and do not
add bias operations. Unregistered operations contribute no FLOPs unless they
decompose into a registered operator. This inference-only workload executes no
backward or attention kernels. A read-only replay of the exact profiling path
reproduced the CSV totals and observed only these counted operations:

| Profiled forward | Convolution | `addmm` | `bmm` | Total |
|---|---:|---:|---:|---:|
| Faster R-CNN | 425.0531328 G | 25.7105920 G | 0 G | 450.7637248 G |
| YOLO11s | 21.2969984 G | 0 G | 0.1228800 G | 21.4198784 G |

Consequently, these totals exclude work such as image decoding and transfer,
elementwise activations, normalization, ordinary pooling, box clipping and
decoding, thresholding, sorting/top-k, softmax/sigmoid, RoIAlign, and NMS except
where an operation happens to decompose into one of the registered operators.
The synchronized latency/FPS profiles do include much of that runtime work and
remain the primary deployment-efficiency evidence.

**Faster R-CNN scope and proposals.** The counter encloses the complete
Torchvision model call, so it includes registered convolution/linear work in
the ResNet-50 backbone and FPN, RPN head, v2 RoI box head, and final class/box
predictor. It does not count RPN or final NMS itself, anchor/proposal
bookkeeping, or MultiScaleRoIAlign. The configured model retains Torchvision's
evaluation defaults of the top 1,000 candidates per FPN level before RPN NMS
(up to 5,000 across its five levels) and at most 1,000 proposals per image
after RPN NMS, with RPN score threshold 0.0. Those candidate-selection steps
are uncounted; the registered RPN-head convolutions run over all anchors on the
feature maps. The configured model also retains the default
100-detection final cap and 0.5 box NMS threshold, while lowering the
model-internal box score threshold from 0.05 to 0.0; the shared evaluator later
applies its frozen 0.25 operating threshold. The profiled image
`0004cfab-14fd-4e49-80ba-63a80b6bddd6.png` reached the 1,000 post-NMS proposal
cap, and all 1,000 proposals entered the RoI head. The count is therefore an
observed, data-dependent full-forward value—not a fixed hypothetical-proposal
formula. Its module attribution is 113.1282432 G for backbone+FPN,
80.7138816 G for the RPN head, and 256.9216000 G for the RoI heads. The
proposal-heavy four-convolution v2 RoI head explains why resolution-only
backbone scaling is not a valid estimate for this detector.

**YOLO11s scope.** The model is fused, evaluated on one pre-resized
`[1, 3, 640, 640]` tensor, and its full network forward is inside the counter.
The detected registered work is convolution plus a small `bmm` contribution
from the head path. Native NMS is deliberately outside the FLOP context but
inside the latency interval; decode/selection operations without registered
formulas do not add to the FLOP total.

**External magnitude check.** The official
[Ultralytics YOLO11 model table](https://docs.ultralytics.com/models/yolo11#performance-metrics)
reports 21.5B FLOPs and 9.4M parameters for YOLO11s at 640 pixels. The local
21.4199 G/9.428M profile is effectively the same scale. Torchvision 0.21's
[official Faster R-CNN v2 weight metadata](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)
reports 280.37 GFLOPs and 43.7M parameters for the stock COCO model, versus the
local 450.76 G and 43.26M. The parameter agreement and order of magnitude are
sound; the FLOP totals are not directly interchangeable because the published
metadata does not establish the same image, proposal realization, class head,
or operator-counting convention. For reference, the local two-FLOPs-per-MAC
total is 225.3818624 GMAC-equivalent before accounting for those other
differences. The defensible claim is therefore the internally consistent
21.04-fold registered-op gap under this documented profiler, supported by the
separately measured latency/FPS gap—not that every paper using the label
"GFLOPs" should reproduce either absolute value.

## Across-seed summary

All requested metrics are retained per seed. The publication table reports the
arithmetic mean and sample standard deviation (`ddof=1`) across the five
predeclared attempts. AP, precision, recall, F1, and compute metrics use all
five seeds for both detectors. Conditional matched-box IoU and Dice use every
seed for which the metric is defined: Faster R-CNN `n=5`, YOLO11s `n=4`.
YOLO11s seed 271 is retained in the per-seed table, but its conditional values
are null because it emitted no score-0.25 detection; they are not coerced to
zero. These summaries describe training-seed variation, not confidence
intervals, and do not replace Phase 8's paired patient-cluster inference.

## Held-out results

The common evaluator processed all 750 test images and 268 reference boxes for
each of the ten frozen checkpoints. Values below are mean ± sample standard
deviation over seeds 17, 42, 137, 271, and 314, except for the explicitly
conditional YOLO11s localization cells.

| Predictive metric | Faster R-CNN mean ± SD (n) | YOLO11s mean ± SD (n) |
|---|---:|---:|
| Precision | 0.1959 ± 0.0552 (n=5) | **0.2983 ± 0.1691 (n=5)** |
| Recall | **0.5799 ± 0.0911 (n=5)** | 0.0955 ± 0.0607 (n=5) |
| F1 | **0.2845 ± 0.0528 (n=5)** | 0.1427 ± 0.0868 (n=5) |
| Matched-box IoU | 0.6749 ± 0.0065 (**n=5**) | **0.6985 ± 0.0157 (n=4)** |
| Matched-box Dice | 0.8010 ± 0.0049 (**n=5**) | **0.8181 ± 0.0111 (n=4)** |
| mAP@0.5 | **0.3042 ± 0.0189 (n=5)** | 0.1626 ± 0.0162 (n=5) |
| mAP@0.5:0.95 | **0.0995 ± 0.0067 (n=5)** | 0.0542 ± 0.0060 (n=5) |

**The `n=5` versus `n=4` localization asymmetry is substantive, not a
formatting footnote.** Faster R-CNN has defined IoU/Dice for all five seeds;
YOLO11s seed 271 has no matched true positive at score 0.25, so its IoU/Dice
are undefined and excluded only from those conditional summaries.

| Compute metric | Faster R-CNN mean ± SD (n=5) | YOLO11s mean ± SD (n=5) |
|---|---:|---:|
| FPS, batch 1 | 20.28 ± 5.62 | **60.29 ± 12.62** |
| Mean inference time (ms/image) | 53.93 ± 21.15 | **17.23 ± 3.83** |
| Total parameters | 43,256,153 ± 0 | **9,428,179 ± 0** |
| Trainable parameters | 43,030,809 ± 0 | **9,428,163 ± 0** |
| Estimated GFLOPs/image | 450.764 ± 0 | **21.420 ± 0** |
| Peak training GPU memory (MiB) | 1,556.89 ± 0.26 | **1,148.16 ± 0.00** |
| Training time (seconds) | 6,661.01 ± 2,127.72 | **1,544.75 ± 425.40** |

This table retains the original score-0.25 operating point. Across all five
attempts, Faster R-CNN has about 1.84 times YOLO11s' mAP@0.5:0.95 and much
higher recall and F1. YOLO11s' higher mean precision at the same nominal cutoff
is an operating-point score-scale effect, not evidence of a generally superior
precision-recall frontier. The current five-run frozen-bundle sensitivity finds
higher Faster R-CNN mean precision at 97 of 101 AP@0.5 recall positions, with
four ties. It retains the original 96-of-101 three-seed Batch 10 result as
historical provenance. The detector-specific thresholds remain the Batch 14
validation selections (0.69/0.05, selected with n=3); applying them unchanged
to all five test bundles gives a five-run sensitivity, not a new threshold
selection.

### Operational confidence-score instability

YOLO11s seed 271 is not a failed or non-converged training process. Its losses
decreased normally and validation mAP@0.5:0.95 peaked at `0.08958`, yet its
maximum held-out prediction score was only `0.0412735`. It produced zero
detections at score 0.25, yielding valid precision/recall/F1 values of
`0/0/0`, while AP@0.5 was `0.1587217` and AP@0.5:0.95 was `0.0555799` because
COCO AP retains ranked predictions down to 0.001. Low-score predictions also
localize targets, so this was not an unlucky zero-IoU-match event. It is a
seed-specific **operational confidence-score degeneracy despite normal
convergence**. The run remains in every unconditional all-attempt endpoint;
only its mathematically undefined matched-only IoU/Dice are omitted.

The defensible trade-off is detection quality versus computational cost.
Under the documented detector-specific profiling procedure on the measured
laptop, YOLO11s has about 3.0 times the measured throughput, 78% fewer
parameters, and about 21 times fewer estimated registered operations. Faster
R-CNN has the stronger precision-recall result over the retained predictions
and, within the evaluated 0.01--0.99 score sweep, higher observed sensitivity
at every reported FROC budget. The accuracy-efficiency Pareto analysis
therefore finds no strict cross-objective dominance. Phase 8's five-attempt
primary training-procedure intervals were wholly positive for recall, F1, and
both AP differences but crossed zero for precision. Separately, the
checkpoint-conditional Holm-adjusted permutation result favored YOLO11s for
precision at 0.25; conditional IoU/Dice were inconclusive and used four
complete seed pairs. The fixed-threshold precision result does not establish a
frontier advantage. See `THRESHOLD_ANALYSIS.md`, `PARETO_ANALYSIS.md`,
`FROC_ANALYSIS.md`, and `STATISTICAL_ANALYSIS.md`; the first three preserve the
pre-specified n=3 artifacts and separately label their n=5 sensitivities.

## Reproduction

The accepted seed-17 timing measurements are reused for seeds 42, 137, 271, and
314 because the model, data, optimizer, AMP, batch, resolution, software, and
GPU contracts are unchanged. `seed-gates` writes a new, non-fabricated
approval artifact that retains the original timing values and hashes while
explicitly recording the source run, source hash, target seed, current
implementation identity, and the reason for reuse. Actual training time and
memory are measured independently in every full run.

```powershell
# Activate the locked CUDA-capable environment documented in README.md first.
$benchmarkPython = (Get-Command python).Source

& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode seed-gates

& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed42.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed42_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed137.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed137_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed271.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed271_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed314.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed314_benchmark/benchmark_estimate.json

& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed42.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed137.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed271.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed314.yaml --mode train

& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode preflight
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode evaluate
```

The final artifacts are:

- `results/tables/detector_comparison_per_seed.csv` — the ten run-level rows,
  including seed 271's explicit null conditional-localization fields;
- `results/tables/detector_comparison_mean_std.csv` — detector/metric long form;
- `results/tables/detector_comparison.csv` — side-by-side mean ± standard
  deviation with detector-specific valid and attempted `n`; and
- `results/logs/phase5_evaluation/summary.json` plus `predictions/*.json.gz` —
  full provenance and prediction evidence.

The former n=3 comparison, long-form, and per-seed tables remain frozen under
their explicit `*_n3_archive.csv` names; they are audit evidence, not the
current headline result.
