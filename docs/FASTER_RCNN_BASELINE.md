# Faster R-CNN Baseline Protocol

## Scope and review gate

Batch 2 trains one seed of the Faster R-CNN baseline. It does not evaluate the
test split and does not spend the two additional seeds reserved for Batch 4.
Before full training, the exact chosen configuration must complete three real
train-plus-validation epochs. The resulting upper-bound time estimate requires
the user's approval; the full-run command validates that the approved estimate
was produced by the byte-identical YAML configuration on the same exact
train/validation pixels and annotations, implementation sources, dependency
versions, GPU, AMP/batch, and resolution settings.

The adopted local runtime is Python 3.11.15 with Torch 2.6.0+cu124 and
Torchvision 0.21.0+cu124 in the existing `torch-gpu` Anaconda environment.
CUDA availability, Torchvision CUDA NMS, float16 AMP with `GradScaler`, an
empty-negative Faster R-CNN training step, and FLOP profiling were verified
before the timed smoke/benchmark run. This runtime replaces the untrained,
unavailable Python 3.13/Torch 2.13/CUDA 13.0 setup; it is not a mid-experiment
change.

The data gate was satisfied on 2026-08-04 from the manually downloaded official
Kaggle aggregate archive. It passed a full ZIP CRC test; the 26,684 training
DICOMs include all 5,000 manifest-selected studies. Fresh conversion produced
exactly 5,000 PNGs with zero missing sources, conversion errors, or manifest
name differences. Benchmark and full modes still enforce the same completeness
check before CUDA/model initialization.

## Chosen configuration

The source of truth is `configs/faster_rcnn.yaml`.

| Setting | Value | Rationale |
|---|---:|---|
| Architecture | `fasterrcnn_resnet50_fpn_v2` | Required two-stage baseline at the scoped model size |
| Initialization | Torchvision default COCO weights | Transfer learning required by Phase 3 |
| Input transform | 640 short edge, 640 long edge | The source images are square; this fixes the model input at 640×640 |
| Physical batch | 2 | Within the 8 GB VRAM scope; batch 1 is prohibited |
| Accumulation | 2 | Effective optimizer batch 4 without claiming larger BatchNorm samples |
| Precision | float16 AMP + `GradScaler` | Mandatory on the RTX 4060 Laptop GPU |
| Data loading | 6 workers; non-persistent train/validation pools | Keeps the scoped worker count while avoiding simultaneous PyTorch worker pools that exceeded the 16 GB Windows commit limit |
| BatchNorm | freeze running statistics; affine parameters remain trainable | Torchvision's v2 builder uses ordinary BatchNorm; microbatch 2 would otherwise update noisy running statistics |
| Train augmentation | none | Explicit controlled baseline; YOLO augmentation parity remains a Batch 3 decision |
| Optimizer | SGD, LR 0.005, momentum 0.9, weight decay 0.0005 | Configured transfer-learning baseline |
| Scheduler | reduce LR on validation AP50:95 plateau | Couples learning-rate reduction to the same validation objective |
| Early stopping | validation AP50:95, patience 5, minimum 8 epochs, maximum 30 | Patience-based stopping required by the hardware scope |
| Seed | 17 | One primary-pipeline seed; extra seeds are reserved for Batch 4 |

Gradient accumulation averages gradients over two forward/backward
microbatches. It does not aggregate BatchNorm moments across those forwards.
The explicit running-statistics freeze, rather than accumulation, addresses the
BatchNorm concern while retaining a physical batch of two.

The first official-data timing attempt completed all epoch-1 training batches
but Windows failed while spawning the separate validation pool with
`WinError 1455` (paging file too small). Training workers were persistent, so
both six-worker PyTorch pools briefly coexisted. Setting both pools to
non-persistent keeps six workers active at a time and makes their startup cost
part of every measured epoch. The failed attempt produced no epoch metric or
checkpoint and is excluded from the timing estimate.

## Metrics

Every epoch records image-weighted mean training loss and its four torchvision
components. Validation is always inference-mode evaluation:

- AP50:95 is official COCO bounding-box AP averaged over IoU thresholds 0.50 to
  0.95, with at most 100 detections per image.
- AP50 is official COCO AP at IoU 0.50.
- Precision, recall, and F1 use global micro counts at score at least 0.25 and
  same-class greedy matching at IoU at least 0.50.
- Negative radiographs remain in validation; detections on them count as false
  positives.
- COCO AP receives model outputs down to the configured internal score
  threshold of zero. The 0.25 threshold applies only to the reported operating
  point.

The validation table names the two thresholds and prediction populations
separately: operating-point prediction counts correspond to precision/recall/F1,
while COCO prediction counts correspond to AP50 and AP50:95.

The detector uses contiguous model labels with zero reserved for background.
The dataset adapter derives the mapping from COCO categories and maps predictions
back to canonical category IDs before evaluation. No class count is hardcoded.

## Timing estimate

Let the three measured complete epoch durations be `d1`, `d2`, and `d3`, each
including training, validation, and equivalent best/last checkpoint I/O. The
steady duration is
`median(d2, d3)`, and the estimate for `N` epochs is:

```text
T(N) = d1 + (N - 1) × median(d2, d3)
```

The benchmark artifact reports the configured minimum-epoch estimate, the
30-epoch sign-off upper bound, and a conservative upper-bound range using the
minimum and maximum of `d2` and `d3`. Dependency and pretrained-weight downloads
are outside epoch timing. The approved full run restarts from COCO weights.

### Completed timing gate (2026-08-09)

The clean official-data benchmark completed exactly three epochs in 770.3,
551.5, and 473.0 seconds (29.91 minutes total). The configured steady-state
estimate is 512.2 seconds per epoch. This projects 1.21 hours through the
minimum eight epochs and 4.34 hours through the 30-epoch upper bound, with a
conservative upper-bound range of 4.02--4.66 hours. Early stopping may end the
full run earlier; its stopping epoch cannot be known from this timing gate.

The approval-bound artifact is
`results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json`, with
SHA-256
`232460ae09827dfb780b0f5c6506bf9f545bbdc0e1483082c2c440035e8e8e8b`.
The best three-epoch validation AP50:95 was 0.10993 at epoch 3; this is a timing
diagnostic, not the final baseline result.

## Completed one-seed baseline (2026-08-09)

After explicit timing approval, the full seed-17 run restarted from the same
COCO weights and stopped after epoch 11 by validation-AP50:95 early stopping.
Epoch 6 was retained as best with AP50:95 0.12764, AP50 0.33144, precision
0.14138, recall 0.68953, and F1 0.23464 on all 750 validation images. These
operating-point metrics use the configured score threshold 0.25 and matching
IoU 0.50. The held-out test split was not accessed.

The epoch loop took 7,017.8 seconds (1.95 hours) and peak allocated training
memory was 1,556.6 MiB. Final best-checkpoint profiling measured 11.00 FPS,
90.92 ms mean latency, 90.78 ms p50 latency, and 92.18 ms p95 latency using
batch-1 float16 AMP over 100 synchronized validation batches. The model has
43,256,153 total parameters (43,030,809 trainable), a 165.38 MiB checkpoint,
and 450.76 estimated GFLOPs under the registered-operation convention described
below. The best checkpoint SHA-256 is
`9ec35c5d761f8e4bf7a43f7999f388ac1ffc0d533f62746409db280706dffab4`.

The final metric table, compute table, and four-panel curve are respectively
`results/tables/faster_rcnn_baseline_validation.csv`,
`results/tables/faster_rcnn_compute.csv`, and
`results/figures/faster_rcnn_training_curves.png`. The curve shows the epoch-6
best marker, the validation fluctuation that triggered patience, and the
learning-rate reduction from 0.005 to 0.0005 at epoch 10.

## Final profiling contract

After early stopping, the best validation checkpoint is reloaded. Batch-1 AMP
inference uses 10 synchronized warmups and 100 synchronized validation batches;
PNG decoding and host-to-device transfer occur outside each timed interval.
The report includes throughput, mean/p50/p95 latency, exact total/trainable
parameter counts, checkpoint bytes/MiB and SHA-256, peak allocated training GPU
memory, and actual epoch-loop time.

GFLOPs use `torch.utils.flop_counter` on a configured 640-pixel validation
image. The installed counter's registered convolution and matrix-multiplication
formulas already use the conventional two operations per multiply-add, so its
total is reported directly rather than doubled again. The method is
proposal-dependent and excludes unsupported operations such as ROIAlign/NMS;
the limitation is stored beside the number and must use the same convention for
YOLO later. Finalization fails rather than publishing a blank GFLOP value.

## Artifacts

- Per-epoch logs and run environment: `results/logs/faster_rcnn_rsna_seed17_*`
- Best/last full checkpoints: `results/checkpoints/faster_rcnn_rsna_seed17_full/`
- Validation baseline table: `results/tables/faster_rcnn_baseline_validation.csv`
- Compute table: `results/tables/faster_rcnn_compute.csv`
- Training curves: `results/figures/faster_rcnn_training_curves.png`

The final tables and figure are intentionally not produced until an approved
full run completes. If training completes but profiling or figure generation
fails, `--mode finalize` regenerates final artifacts from the saved best
checkpoint without repeating optimization.
