# Accuracy-Efficiency Pareto Analysis

## Historical n=3 scope and n=5 sensitivity

This figure and every value below are frozen **n=3** results for seeds 17, 42,
and 137; they were not regenerated for the later five-seed clean comparison.
`configs/pareto.yaml` reads
`detector_comparison_n3_archive.csv` and
`detector_comparison_per_seed_n3_archive.csv`. Its selected-threshold inputs are
also n=3: `configs/threshold_selection.yaml` is bound to
`configs/evaluation_n3_archive.yaml` and
`results/logs/phase5_evaluation/summary_n3_archive.json`. Historical output
filenames without an `n3` suffix do not change that scope.

Batch 35 preserves that figure unchanged and adds a separately labeled n=5
all-attempt sensitivity. The new path includes seeds 17, 42, 137, 271, and 314
for both detectors, including every available same-run compute row. It neither
reselects the historical validation thresholds nor mixes n=3 and n=5 points in
one frontier.

## Protocol and dominance rule

The figure joins the six frozen seed-level accuracy rows to their corresponding compute
CSVs and cross-checks mAP@0.5:0.95 means against `detector_comparison.csv`. Because mAP does
not depend on the selected single operating threshold, panels (a) and (c) use the Phase 5
seed values directly; AP nevertheless remains conditional on the retained predictions and
common evaluation/post-processing protocol. Panels (b) and (d) use
`selected_operating_points_per_seed.csv`. The selection rule maximizes mean F1 over the
three validation seeds, selecting 0.69 for Faster R-CNN and 0.05 for YOLO11s; each frozen
threshold is then evaluated once on the test bundles. Test results do not influence the
threshold choice. A detector is marked strictly Pareto-dominant only when every one of its
seed points is better than every seed of the other detector on both directed axes;
mean-only ordering is not sufficient. This correction changes only the recall panels and
their annotations: the AP data and compute axes in panels (a) and (c) are unchanged.

FPS and latency are implementation-specific measurements on the stated laptop/software
stack. Faster R-CNN resizing occurs inside its timed model forward, whereas YOLO tensor
resizing occurs before the timed forward-plus-NMS region; the compute axes therefore do not
define an architecture-family throughput law.

YOLO11s seed 271 is intentionally absent from the historical figure. Although its training
curves converged normally, its maximum test confidence is 0.0412735, so it emits zero
detections at the n=3-selected YOLO threshold 0.05. The current Pareto figure therefore must
not be described as an n=5 frontier or used to hide that later confidence-score degeneracy.

The n=5 sensitivity includes seed 271's observed AP, throughput, latency,
parameters, and GFLOPs. Its recall is the defined value zero because it emits
no detection at the unchanged 0.05 threshold. This point is shown rather than
filtered.

## (a) mAP@0.5:0.95 versus FPS

All Faster R-CNN seeds have higher mAP@0.5:0.95 (0.0984-0.1054) than all YOLO11s seeds
(0.0458-0.0607), while every YOLO11s seed has higher throughput (46.74-65.24 FPS) than every
Faster R-CNN seed (11.00-21.82 FPS). The corresponding means are 0.1023 versus 0.0549 mAP
and 17.42 versus 52.94 FPS. Neither detector strictly dominates because the detector with
greater detection accuracy is consistently slower. Faster R-CNN is therefore preferable
when AP is the binding requirement, whereas YOLO11s is preferable when batch-1 throughput
is the tighter constraint and the measured AP reduction is acceptable.

## (b) Recall versus latency

At the validation-selected thresholds, Faster R-CNN test recall spans 0.2910-0.4030 at
threshold 0.69, while YOLO11s spans 0.2090-0.2612 at threshold 0.05. YOLO11s
simultaneously has lower mean latency for every seed (15.33-21.40 ms/image) than Faster R-CNN
(45.82-90.92 ms/image).
Neither detector strictly dominates because the recall and latency objectives remain
opposed at thresholds chosen without test feedback. A recall-sensitive workflow can justify
Faster R-CNN's additional delay, while a latency-constrained human-reviewed workflow may
favor YOLO11s only if its remaining coverage limitation is acceptable.

## (c) mAP@0.5:0.95 versus parameters

Faster R-CNN's 43.26 million parameters accompany the higher seed-level mAP range, whereas
YOLO11s uses 9.43 million parameters, about 78.2% fewer, with the lower mAP range.
Parameter count is architecture-fixed here, so the vertical spread at each x-position is
training-seed variation in mAP rather than model-size variation. Neither detector strictly
dominates because higher AP and a smaller parameter footprint occur in different detector
arms. Faster R-CNN is preferable when model capacity is affordable and AP is primary;
YOLO11s is preferable when memory, storage, or model-transfer limits bind more strongly.

## (d) Recall versus estimated GFLOPs

Faster R-CNN combines mean test recall 0.3607 at its validation-selected threshold with
450.76 estimated GFLOPs/image, while YOLO11s combines mean recall 0.2438 with 21.42
GFLOPs/image. The registered-operation gap is about 21.04-fold, but the estimate excludes
unsupported operations and should be interpreted alongside measured latency rather than as
a complete runtime model. Neither detector strictly dominates because the higher-recall
seed cloud is also the more compute-intensive one. Faster R-CNN is preferable where missed
opacity annotations are the tighter constraint and GPU compute is available, whereas
YOLO11s is the more plausible option under a strict operation budget with mandatory human
review.

## Five-run sensitivity and aggregation definition

The n=5 point table has exactly one row per detector/run. Each row joins:

- that run's COCO AP@0.5:0.95 from the retained prediction population and
  common evaluation protocol;
- that run's test recall at the detector threshold selected from the original
  three validation runs (0.69 or 0.05); and
- that same run's measured FPS/latency and recorded parameter/GFLOP values.

No metric is pooled across scopes inside a point. The companion summary is an
equal-run arithmetic mean and sample SD across five rows per detector; it is
descriptive and is not used for the strict dominance label. Dominance retains
the conservative cloud rule: every candidate run must be strictly better than
every competing run on both axes.

Across n=5, Faster R-CNN AP ranges 0.0885--0.1054 versus 0.0458--0.0607 for
YOLO11s, while YOLO11s throughput ranges 46.74--73.03 FPS versus 11.00--25.19.
Fixed-threshold recall ranges 0.2910--0.4030 and 0--0.2612, respectively;
YOLO11s latency remains lower for every run. Mean AP/FPS is 0.0995/20.28 versus
0.0542/60.29, and mean fixed-threshold recall/latency is 0.3507/53.93 ms versus
0.1948/17.23 ms. Parameter and GFLOP counts remain 43.26 M/450.76 versus
9.43 M/21.42.

Neither detector strictly dominates in any of the four n=5 panels. All four
Pareto conclusions are therefore **unchanged** from n=3. Hardware sample sizes
are explicit: the sensitivity has five compute rows per detector; the
historical figure has three. No incomplete hardware metric is silently carried
into a mean.

## Reproduction

The Pareto rendering itself is entirely offline: it loads no checkpoint and performs no
training or model inference. Its selected operating-point inputs are generated by the
frozen n=3 validation-selection protocol documented in `THRESHOLD_ANALYSIS.md`.

```powershell
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode preflight
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto.yaml --mode run
```

These commands reproduce only the archived three-seed Pareto analysis because the config
is routed to the n=3 CSV inputs; they do not regenerate a five-seed figure.

The separate n=5 sensitivity is reproduced with:

```powershell
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto_n5_sensitivity.yaml --mode preflight
& $benchmarkPython -m src.plot_pareto_frontier --config configs/pareto_n5_sensitivity.yaml --mode run
```

It writes `pareto_points_n5_sensitivity.csv`,
`pareto_summary_n5_sensitivity.csv`,
`pareto_frontier_n5_sensitivity.png`, and the hash-bound
`phase35_operating_regime_n5/pareto_summary.json`. The figure and provenance
state both the five-run test scope and the three-run validation-selection scope.
