# Accuracy-Efficiency Pareto Analysis

## Protocol and dominance rule

The figure joins the six frozen seed-level accuracy rows to their corresponding compute
CSVs and cross-checks mAP@0.5:0.95 means against `detector_comparison.csv`. Because mAP is
threshold-independent, panels (a) and (c) use the Phase 5 seed values directly; panels (b)
and (d) use recall at each detector's best observed mean-F1 threshold in
`threshold_sweep.csv`, with the per-seed companion table exposing the spread. Faster R-CNN
selects threshold 0.63 and YOLO11s selects 0.01, the lower sweep boundary, so the YOLO11s
point is an observed operating point rather than a claimed global optimum. A detector is
marked strictly Pareto-dominant only when every one of its seed points is better than every
seed of the other detector on both directed axes; mean-only ordering is not sufficient.

## (a) mAP@0.5:0.95 versus FPS

All Faster R-CNN seeds have higher mAP@0.5:0.95 (0.0984-0.1054) than all YOLO11s seeds
(0.0458-0.0607), while every YOLO11s seed has higher throughput (46.74-65.24 FPS) than every
Faster R-CNN seed (11.00-21.82 FPS). The corresponding means are 0.1023 versus 0.0549 mAP
and 17.42 versus 52.94 FPS. Neither detector strictly dominates because the detector with
greater detection accuracy is consistently slower. Faster R-CNN is therefore preferable
when AP is the binding requirement, whereas YOLO11s is preferable when batch-1 throughput
is the tighter constraint and the measured AP reduction is acceptable.

## (b) Recall versus latency

At their peak mean-F1 sweep points, Faster R-CNN recall spans 0.3955-0.4478 at threshold
0.63, while YOLO11s spans 0.2724-0.3731 at threshold 0.01. YOLO11s simultaneously has lower
mean latency for every seed (15.33-21.40 ms/image) than Faster R-CNN (45.82-90.92 ms/image).
Neither detector strictly dominates because the recall and latency objectives remain
opposed even after replacing the shared 0.25 threshold with threshold-aware operating
points. A recall-sensitive workflow can justify Faster R-CNN's additional delay, while a
latency-constrained human-reviewed workflow may favor YOLO11s only if its remaining
coverage limitation is acceptable.

## (c) mAP@0.5:0.95 versus parameters

Faster R-CNN's 43.26 million parameters accompany the higher seed-level mAP range, whereas
YOLO11s uses 9.43 million parameters, about 78.2% fewer, with the lower mAP range.
Parameter count is architecture-fixed here, so the vertical spread at each x-position is
training-seed variation in mAP rather than model-size variation. Neither detector strictly
dominates because higher AP and a smaller parameter footprint occur in different detector
arms. Faster R-CNN is preferable when model capacity is affordable and AP is primary;
YOLO11s is preferable when memory, storage, or model-transfer limits bind more strongly.

## (d) Recall versus estimated GFLOPs

Faster R-CNN combines mean recall 0.4266 at its selected threshold with 450.76 estimated
GFLOPs/image, while YOLO11s combines mean recall 0.3321 with 21.42 GFLOPs/image. The
registered-operation gap is about 21.04-fold, but the estimate excludes
unsupported operations and should be interpreted alongside measured latency rather than as
a complete runtime model. Neither detector strictly dominates because the higher-recall
seed cloud is also the more compute-intensive one. Faster R-CNN is preferable where missed
findings are the tighter constraint and GPU compute is available, whereas YOLO11s is the
more plausible option under a strict operation budget with mandatory human review.

## Reproduction

The analysis is entirely offline: it loads no checkpoint and performs no training or model
inference.

```powershell
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode preflight
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode run
```
