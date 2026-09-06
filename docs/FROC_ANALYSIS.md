# Free-Response ROC Analysis

## Current interpretation

The current result is the **observed exact-score frontier** computed from ten
versioned prediction bundles collected from the frozen Phase 5 checkpoints at
the user-approved candidate-score floor of 0.00001. It remains bounded by the
candidates emitted at that configured floor. One YOLO11s run ends just below 2 FP/image, so only
that run's 2-FP/image contribution and the corresponding aggregate remain
lower-bound observations.

No retraining, weight update, threshold selection, interpolation, or
extrapolation was performed. The approved collection changed only the
candidate floor from 0.0001 to 0.00001. The images, annotations, class,
preprocessing, checkpoints, matching, native NMS, maximum detections, and seed
scope are unchanged. The historical three-run and five-run grid artifacts and
the 0.001 and 0.0001 exact-score analyses remain intact as provenance.

## Historical-grid reproduction and fixed protocol

The historical grid was `{0.01, 0.02, ..., 0.99}`: 99 thresholds with lower
and upper limits 0.01 and 0.99. Before exact-score evaluation, the archive-safe
n=3 command reproduced the historical operating-point CSV byte for byte
(SHA-256
`c5b1fad22afbaedac90060056066beca62be2464f79cfb716a0db998836527c8`).
The Batch 35 n=5 table and summary hashes were also verified before use and are
bound in each exact-score summary.

The shared protocol is:

- historical-grid lower/upper limits: 0.01 and 0.99;
- original Phase 5 candidate floor: 0.001;
- first approved sensitivity candidate floor: 0.0001;
- current approved sensitivity candidate floor: 0.00001;
- matching IoU: 0.50, using the canonical stable descending-score greedy
  matcher with no target reuse;
- native, class-aware NMS IoU: 0.50;
- maximum detections: 100 per image;
- class: category 1, `Lung Opacity`;
- images and annotations: the unchanged 750-image internal-testing split with
  268 target boxes;
- seed scope: 17, 42, 137, 271, and 314 for each detector; and
- prespecified FP/image operating budgets: 0.125, 0.25, 0.5, 1, and 2.

The historical n=3 scope is seeds 17, 42, and 137. Batch 35 added the separate
five-run grid sensitivity without replacing the archive. Seed 271 remains in
the five-run scope exactly as observed.

## Exact-score method and endpoint behavior

For each detector/run bundle, the analysis evaluates every unique emitted
confidence score in deterministic descending order with `score >= threshold`.
An upper empty sentinel is the next representable float above that run's
maximum score, giving zero predictions, zero FP/image, and zero sensitivity.
A lower sentinel at exactly 0.00001 includes every candidate retained in the
approved bundle even when no emitted score equals 0.00001.

For each operating budget, each run contributes the maximum observed
sensitivity at or below the budget. There is no interpolation. Ties prefer
fewer FP/image and then the higher threshold. Aggregate sensitivity is the
arithmetic mean across all five runs; dispersions are sample standard
deviations. Sensitivity and FP/image are monotone nondecreasing as thresholds
relax in all ten runs. All 50 selected run-budget points were independently
re-evaluated through the canonical matcher with no disagreement.

## Historical grid versus approved lower-floor exact-score frontier

| FP/image budget | Historical grid Faster R-CNN | Historical grid YOLO11s | Exact Faster R-CNN | Exact YOLO11s | Exact gap status |
|---:|---:|---:|---:|---:|---|
| 0.125 | 0.2672 | 0.1761 | 0.2776 +/- 0.0101 | 0.1799 +/- 0.0182 | Faster higher; gap strengthened |
| 0.25 | 0.3642 | 0.2455 | 0.3664 +/- 0.0272 | 0.2664 +/- 0.0306 | Faster higher; gap weakened |
| 0.5 | 0.4836 | 0.2925 | 0.4858 +/- 0.0309 | 0.3821 +/- 0.0097 | Faster higher; gap weakened |
| 1 | 0.5970 | 0.2925 | 0.6000 +/- 0.0208 | 0.5075 +/- 0.0201 | Faster higher; gap weakened |
| 2 | 0.6873 | 0.2925 | 0.6978 +/- 0.0194 | 0.6090 +/- 0.0307 | Faster higher; gap weakened; YOLO lower bound |

The v4 inference leaves both detectors unchanged through 0.5 FP/image and
leaves Faster R-CNN unchanged at all five budgets. Versus v3, YOLO11s
sensitivity rises by 0.0090 at 1 and 0.0209 at 2 FP/image. The detector
ordering does not reverse:
Faster R-CNN remains higher at every prespecified budget. Relative to the
historical grid, the Faster-R-CNN-minus-YOLO11s gap weakens at 0.25, 0.5, 1,
and 2 FP/image and strengthens only at 0.125.

## Residual candidate-floor boundary and conservative bound

The 0.00001 candidate floor removes the former YOLO11s seed-137 limit at
1 FP/image. Faster R-CNN is not floor-limited at any prespecified budget.
YOLO11s seed 137 reaches the bundle endpoint at sensitivity 0.5634 and
1.9907 FP/image, just below the 2-FP/image budget. Therefore:

- no run is floor-limited at 0.125, 0.25, 0.5, or 1 FP/image;
- YOLO11s seed 137 alone is floor-limited at 2 FP/image; and
- the five-run YOLO11s aggregate at 2 FP/image remains a lower-bound
  observation, not evidence of a terminal plateau.

For the unobserved portion, the deliberately conservative bound assigns the
mathematical maximum sensitivity of 1.0 to the limited seed. This raises the
largest possible YOLO11s five-run aggregate from 0.6090 to 0.6963. Faster
R-CNN is observed at 0.6978, leaving a minimum gap of 0.0015. Thus the
frontier remains technically incomplete at 2 FP/image, but the missing portion
cannot reverse the Faster-R-CNN-versus-YOLO11s ordering at either 1 or
2 FP/image.

## Scientific value of another lower-floor run

Another detector-neutral lower-floor inference pass could replace the lone
2-FP/image lower bound with a directly observed value and better estimate the
YOLO11s curve shape near that budget. Its scientific value is therefore
limited to quantitative completeness: even the mathematical maximum missing
sensitivity cannot change the qualitative detector ordering. No 0.000001 run
was authorized or performed, and none should be started without a separate
review.

## Artifacts and reproduction

The current v4 outputs are:

- `results/logs/phase42_froc_lower_floor_v4/` (ten prediction bundles,
  environment capture, progress record, and completion summary);
- `results/tables/froc_lower_floor_prediction_inventory_v4.csv`;
- `results/tables/froc_lower_floor_score_boundaries_v4.csv`;
- `results/tables/froc_lower_floor_inference_contract_v4.csv`;
- `results/tables/froc_exact_score_per_seed_v4.csv`;
- `results/tables/froc_exact_score_aggregate_v4.csv`;
- `results/tables/froc_operating_points_per_seed_exact_score_v4.csv`;
- `results/tables/froc_operating_points_exact_score_v4.csv`;
- `results/tables/froc_grid_vs_exact_score_comparison_v4.csv`;
- `results/tables/froc_exact_score_v3_vs_v4_comparison_v4.csv`;
- `results/tables/froc_exact_score_v3_vs_v4_per_run_v4.csv`;
- `results/tables/froc_incomplete_frontier_bounds_v4.csv`;
- `results/figures/froc_exact_score_v4.png`; and
- `results/logs/phase42_froc_exact_score_v4/summary.json`.

From the repository root in the pinned CUDA environment, the exact full
reproduction is:

```powershell
& $benchmarkPython -m src.collect_froc_lower_floor_predictions --config configs/evaluation_froc_lower_floor_v4.yaml --source-config configs/evaluation.yaml --prior-config configs/evaluation_froc_lower_floor_v3.yaml --checkpoint-manifest results/checkpoint_release_manifest.json --mode preflight
& $benchmarkPython -m src.collect_froc_lower_floor_predictions --config configs/evaluation_froc_lower_floor_v4.yaml --source-config configs/evaluation.yaml --prior-config configs/evaluation_froc_lower_floor_v3.yaml --checkpoint-manifest results/checkpoint_release_manifest.json --mode run
& $benchmarkPython -m src.analyze_exact_score_froc --config configs/froc_exact_score_v4.yaml --mode preflight
& $benchmarkPython -m src.analyze_exact_score_froc --config configs/froc_exact_score_v4.yaml --mode run
```

The collector is restart-safe: it reuses an existing bundle only after its
checkpoint, annotation, evaluator, identity, image count, and bundle content
pass validation. It never trains or updates weights. The exact-score analysis
after collection is offline and does not load checkpoints.

The 0.001 and 0.0001 exact-score outputs remain reproducible with
`configs/froc_exact_score_v2.yaml` and `configs/froc_exact_score_v3.yaml`.
The archive-safe n=3 historical
reproduction uses `configs/froc_n3_archive.yaml`, and the five-run historical
grid remains verifiable with `configs/froc_n5_sensitivity.yaml`.
