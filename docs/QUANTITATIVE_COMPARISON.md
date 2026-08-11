# Unified quantitative comparison

## Scope and freeze point

Phase 5 compares the selected validation-best checkpoint from each training seed
on the untouched 750-image RSNA test split. The seed grid was fixed before the
additional runs as 17, 42, and 137. Seeds 42 and 137 change only RNG state and
artifact identity; detector-specific hyperparameters remain equal to the
accepted Batch 2 and Batch 3 configs. Training and early stopping continue to
use only train and validation data. `src/evaluate.py` is the first model entry
point authorized to open `instances_test.json`.

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
- the same score threshold (0.25), matching IoU (0.50), COCO minimum score
  (0.001), and maximum detections per image (100).

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
operations differ by architecture, so latency/FPS are the deployment-relevant
efficiency measures. Faster R-CNN preprocessing/resizing remains inside its
model forward, while the YOLO profile performs resize/tensor conversion before
the timed forward-plus-NMS interval; this inherited framework asymmetry is
reported rather than treated as an architecture-only difference.

## Across-seed summary

All requested metrics are retained per seed. The publication table reports the
arithmetic mean and sample standard deviation (`ddof=1`, `n=3`) across seeds.
This describes training-seed variation; it is not a confidence interval and is
not a substitute for Phase 8's paired image-level resampling and testing.

## Held-out results

The common evaluator processed all 750 test images and 268 reference boxes for
each of the six frozen checkpoints. Values below are mean ± sample standard
deviation over seeds 17, 42, and 137.

| Predictive metric | Faster R-CNN | YOLO11s |
|---|---:|---:|
| Precision | 0.1626 ± 0.0439 | **0.3730 ± 0.0395** |
| Recall | **0.6381 ± 0.0526** | 0.1356 ± 0.0094 |
| F1 | **0.2558 ± 0.0493** | 0.1981 ± 0.0048 |
| Matched-box IoU | 0.6732 ± 0.0084 | **0.6971 ± 0.0189** |
| Matched-box Dice | 0.7997 ± 0.0065 | **0.8172 ± 0.0134** |
| mAP@0.5 | **0.3084 ± 0.0123** | 0.1643 ± 0.0226 |
| mAP@0.5:0.95 | **0.1023 ± 0.0036** | 0.0549 ± 0.0080 |

| Compute metric | Faster R-CNN | YOLO11s |
|---|---:|---:|
| FPS, batch 1 | 17.42 ± 5.69 | **52.94 ± 10.65** |
| Mean inference time (ms/image) | 62.72 ± 24.58 | **19.36 ± 3.49** |
| Total parameters | 43,256,153 ± 0 | **9,428,179 ± 0** |
| Trainable parameters | 43,030,809 ± 0 | **9,428,163 ± 0** |
| Estimated GFLOPs/image | 450.764 ± 0 | **21.420 ± 0** |
| Peak training GPU memory (MiB) | 1,556.92 ± 0.27 | **1,148.16 ± 0** |
| Training time (seconds) | 6,211.41 ± 2,566.64 | **1,833.21 ± 214.60** |

Faster R-CNN provides the stronger detection result under the frozen protocol:
about 1.86 times YOLO11s' mAP@0.5:0.95, substantially higher recall, and higher
F1. YOLO11s is more selective and much cheaper: it has higher precision, about
3.0 times the measured throughput, 78% fewer parameters, and about 21 times
fewer estimated FLOPs. Its slightly higher IoU and Dice apply only to matched
true positives and do not offset its low recall. Phase 8's predeclared paired
tests retain Holm-corrected evidence for the recall and both AP differences,
as well as YOLO's precision advantage; see `docs/STATISTICAL_ANALYSIS.md`.

## Reproduction

The accepted seed-17 timing measurements are reused for seeds 42 and 137 because
the model, data, optimizer, AMP, batch, resolution, software, and GPU contracts
are unchanged. `seed-gates` writes a new, non-fabricated approval artifact that
retains the original timing values and hashes while explicitly recording the
source run, source hash, target seed, current implementation identity, and the
reason for reuse. Actual training time and memory are measured independently in
every full run.

```powershell
$benchmarkPython = 'C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe'

& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode seed-gates

& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed42.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed42_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed137.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed137_benchmark/benchmark_estimate.json

& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed42.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed137.yaml --mode train

& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode preflight
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode evaluate
```

The final artifacts are:

- `results/tables/detector_comparison_per_seed.csv` — the six run-level rows;
- `results/tables/detector_comparison_mean_std.csv` — detector/metric long form;
- `results/tables/detector_comparison.csv` — side-by-side mean ± standard
  deviation; and
- `results/logs/phase5_evaluation/summary.json` plus `predictions/*.json.gz` —
  full provenance and prediction evidence.
