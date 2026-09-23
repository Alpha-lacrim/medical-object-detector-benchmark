# Frozen VinDr-CXR inference results — Batch 49

Generated from [results/vindr_external_v1/inference_summary.json](../results/vindr_external_v1/inference_summary.json).
Source SHA-256: `d15070ebcafdfb2711b07db5090e67ef556b60d0042c87ac047b243bdf8fc656`.

All ten frozen RSNA checkpoints are retained, including both seed-271 runs.
The official VinDr-CXR v1.0.0 test set contains 3,000 released study/images,
84 strict `Lung Opacity` positive images, 95 boxes and 2,916 strict negatives.
Image percentages do not imply patient-level percentages. Other findings
are not merged into the target. No training, adaptation or external threshold selection.

These are per-run and equal-run descriptive results. Batch 50 bootstrap
uncertainty and internal/external synthesis have not been performed.
AP uses score support >=0.001; exact-score FROC uses the retained 0.00001
candidate support. Native candidate filters are strict >; common evaluation
uses >=. The NMS/matching IoU is 0.50 and the cap is 100 detections/image.
FROC budget sensitivities use observed points without interpolation.
A floor-limited value is an observed lower bound. No FROC score coordinate
is exported as a deployment threshold.

## AP and prespecified FROC sensitivities

| Detector | Seed | AP@0.50 | AP@0.50:0.95 | Sens. at 0.125 FP/image | Sens. at 0.25 FP/image | Sens. at 0.5 FP/image | Sens. at 1 FP/image | Sens. at 2 FP/image | Images at cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster_rcnn | 17 | 0.001172 | 0.000418 | 0.042105 | 0.052632 | 0.052632 | 0.052632 | 0.084211 | 3000 |
| faster_rcnn | 42 | 0.003999 | 0.000721 | 0.063158 | 0.136842 | 0.147368 | 0.178947 | 0.231579 | 2635 |
| faster_rcnn | 137 | 0.001216 | 0.000457 | 0.010526 | 0.042105 | 0.063158 | 0.084211 | 0.136842 | 3000 |
| faster_rcnn | 271 | 0.003880 | 0.000802 | 0.105263 | 0.115789 | 0.115789 | 0.157895 | 0.189474 | 3000 |
| faster_rcnn | 314 | 0.002258 | 0.000692 | 0.063158 | 0.084211 | 0.094737 | 0.115789 | 0.178947 | 209 |
| yolo11s | 17 | 0.000758 | 0.000129 | 0.021053 | 0.052632 | 0.084211 | 0.105263 | 0.147368 | 0 |
| yolo11s | 42 | 0.000582 | 0.000233 | 0.021053 | 0.021053 | 0.084211 | 0.094737 | 0.105263 | 0 |
| yolo11s | 137 | 0.000082 | 0.000049 | 0.010526 | 0.042105 | 0.063158 | 0.094737 | 0.094737 | 0 |
| yolo11s | 271 | 0.000392 | 0.000046 | 0.021053 | 0.021053 | 0.031579 | 0.031579 | 0.052632 | 0 |
| yolo11s | 314 | 0.000689 | 0.000140 | 0.021053 | 0.031579 | 0.042105 | 0.063158 | 0.094737 | 0 |

### Equal-run AP summaries

Arithmetic mean and sample SD; five runs per detector.

| Detector | Endpoint | Mean | Sample SD | Defined runs |
| --- | --- | --- | --- | --- |
| faster_rcnn | ap.ap50 | 0.002505 | 0.001380 | 5 |
| faster_rcnn | ap.ap50_95 | 0.000618 | 0.000170 | 5 |
| yolo11s | ap.ap50 | 0.000501 | 0.000272 | 5 |
| yolo11s | ap.ap50_95 | 0.000119 | 0.000077 | 5 |

### Achieved FP/image and candidate-floor limits

| Detector | Seed | Achieved at 0.125 | Achieved at 0.25 | Achieved at 0.5 | Achieved at 1 | Achieved at 2 | Floor FP/image | Floor sensitivity | Floor-limited budgets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster_rcnn | 17 | 0.119667 | 0.209667 | 0.209667 | 0.209667 | 1.711333 | 99.982667 | 0.547368 | none |
| faster_rcnn | 42 | 0.085667 | 0.208667 | 0.345000 | 0.789333 | 1.910667 | 99.169333 | 0.610526 | none |
| faster_rcnn | 137 | 0.020667 | 0.218333 | 0.439333 | 0.993667 | 1.680000 | 99.981000 | 0.600000 | none |
| faster_rcnn | 271 | 0.117000 | 0.212667 | 0.212667 | 0.928333 | 1.934667 | 99.982333 | 0.557895 | none |
| faster_rcnn | 314 | 0.112333 | 0.142333 | 0.252667 | 0.872000 | 1.422000 | 82.785333 | 0.526316 | none |
| yolo11s | 17 | 0.044667 | 0.219333 | 0.412667 | 0.655333 | 1.981000 | 5.349333 | 0.210526 | none |
| yolo11s | 42 | 0.034000 | 0.034000 | 0.419333 | 0.896667 | 1.619667 | 4.291333 | 0.157895 | none |
| yolo11s | 137 | 0.081000 | 0.208000 | 0.309333 | 0.793333 | 0.793333 | 0.793333 | 0.094737 | 1, 2 |
| yolo11s | 271 | 0.102667 | 0.102667 | 0.281000 | 0.281000 | 1.379667 | 15.160000 | 0.252632 | none |
| yolo11s | 314 | 0.071000 | 0.212000 | 0.423333 | 0.835000 | 1.834333 | 7.738333 | 0.189474 | none |

## Historical n=3 RSNA threshold transport — PRIMARY

Selection provenance and thresholds remain unchanged; external evaluation uses five runs per detector.

| Detector | Seed | Threshold | TP | FP | FN | Precision | Recall | F1 | FP/image | Detections/image | % images emitting | Emitting images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster_rcnn | 17 | 0.690000 | 3 | 241 | 92 | 0.012295 | 0.031579 | 0.017699 | 0.080333 | 0.081333 | 7.200000 | 216 |
| faster_rcnn | 42 | 0.690000 | 2 | 167 | 93 | 0.011834 | 0.021053 | 0.015152 | 0.055667 | 0.056333 | 4.800000 | 144 |
| faster_rcnn | 137 | 0.690000 | 0 | 46 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.015333 | 0.015333 | 1.500000 | 45 |
| faster_rcnn | 271 | 0.690000 | 3 | 131 | 92 | 0.022388 | 0.031579 | 0.026201 | 0.043667 | 0.044667 | 3.866667 | 116 |
| faster_rcnn | 314 | 0.690000 | 1 | 130 | 94 | 0.007634 | 0.010526 | 0.008850 | 0.043333 | 0.043667 | 3.933333 | 118 |
| yolo11s | 17 | 0.050000 | 1 | 110 | 94 | 0.009009 | 0.010526 | 0.009709 | 0.036667 | 0.037000 | 2.900000 | 87 |
| yolo11s | 42 | 0.050000 | 0 | 84 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.028000 | 0.028000 | 1.933333 | 58 |
| yolo11s | 137 | 0.050000 | 0 | 132 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.044000 | 0.044000 | 3.533333 | 106 |
| yolo11s | 271 | 0.050000 | 0 | 0 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| yolo11s | 314 | 0.050000 | 1 | 80 | 94 | 0.012346 | 0.010526 | 0.011364 | 0.026667 | 0.027000 | 2.333333 | 70 |

## Post-hoc n=5 RSNA threshold transport — SECONDARY sensitivity

Selection provenance and thresholds remain unchanged; external evaluation uses five runs per detector.

| Detector | Seed | Threshold | TP | FP | FN | Precision | Recall | F1 | FP/image | Detections/image | % images emitting | Emitting images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster_rcnn | 17 | 0.700000 | 3 | 217 | 92 | 0.013636 | 0.031579 | 0.019048 | 0.072333 | 0.073333 | 6.566667 | 197 |
| faster_rcnn | 42 | 0.700000 | 2 | 159 | 93 | 0.012422 | 0.021053 | 0.015625 | 0.053000 | 0.053667 | 4.666667 | 140 |
| faster_rcnn | 137 | 0.700000 | 0 | 33 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.011000 | 0.011000 | 1.066667 | 32 |
| faster_rcnn | 271 | 0.700000 | 3 | 128 | 92 | 0.022901 | 0.031579 | 0.026549 | 0.042667 | 0.043667 | 3.766667 | 113 |
| faster_rcnn | 314 | 0.700000 | 1 | 126 | 94 | 0.007874 | 0.010526 | 0.009009 | 0.042000 | 0.042333 | 3.866667 | 116 |
| yolo11s | 17 | 0.010000 | 2 | 266 | 93 | 0.007463 | 0.021053 | 0.011019 | 0.088667 | 0.089333 | 6.200000 | 186 |
| yolo11s | 42 | 0.010000 | 2 | 195 | 93 | 0.010152 | 0.021053 | 0.013699 | 0.065000 | 0.065667 | 4.600000 | 138 |
| yolo11s | 137 | 0.010000 | 0 | 182 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.060667 | 0.060667 | 4.633333 | 139 |
| yolo11s | 271 | 0.010000 | 0 | 54 | 95 | 0.000000 | 0.000000 | 0.000000 | 0.018000 | 0.018000 | 1.500000 | 45 |
| yolo11s | 314 | 0.010000 | 1 | 204 | 94 | 0.004878 | 0.010526 | 0.006667 | 0.068000 | 0.068333 | 5.066667 | 152 |

## Score distributions at the four prespecified supports

Emitted-detection descriptions, not calibrated patient probabilities.
Quantiles use the linear method; empty populations remain NA.

| Detector | Seed | Support | Count | Min | Q01 | Q05 | Q25 | Median | Q75 | Q95 | Q99 | Mean | Max | Zero-output images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster_rcnn | 17 | all_retained_candidates | 300000 | 0.000014 | 0.000026 | 0.000036 | 0.000074 | 0.000180 | 0.000696 | 0.017178 | 0.244995 | 0.008224 | 0.917377 | 0 |
| faster_rcnn | 17 | ap_candidates | 62377 | 0.001001 | 0.001020 | 0.001105 | 0.001734 | 0.003773 | 0.015785 | 0.236164 | 0.581271 | 0.038760 | 0.917377 | 0 |
| faster_rcnn | 17 | historical_threshold | 244 | 0.690350 | 0.691118 | 0.694634 | 0.721964 | 0.754280 | 0.792845 | 0.864594 | 0.895598 | 0.762641 | 0.917377 | 2784 |
| faster_rcnn | 17 | posthoc_n5_threshold | 220 | 0.700485 | 0.701628 | 0.707579 | 0.729484 | 0.764364 | 0.799297 | 0.865551 | 0.896718 | 0.770093 | 0.917377 | 2803 |
| faster_rcnn | 42 | all_retained_candidates | 297566 | 0.000010 | 0.000011 | 0.000014 | 0.000024 | 0.000045 | 0.000113 | 0.001156 | 0.018749 | 0.001936 | 0.964489 | 0 |
| faster_rcnn | 42 | ap_candidates | 16213 | 0.001001 | 0.001014 | 0.001088 | 0.001607 | 0.003148 | 0.010735 | 0.173414 | 0.698099 | 0.033908 | 0.964489 | 191 |
| faster_rcnn | 42 | historical_threshold | 169 | 0.691393 | 0.692939 | 0.701730 | 0.745954 | 0.804866 | 0.868102 | 0.937133 | 0.953504 | 0.809234 | 0.964489 | 2856 |
| faster_rcnn | 42 | posthoc_n5_threshold | 161 | 0.700178 | 0.704150 | 0.711581 | 0.754282 | 0.811825 | 0.869770 | 0.938011 | 0.954004 | 0.814892 | 0.964489 | 2860 |
| faster_rcnn | 137 | all_retained_candidates | 300000 | 0.000046 | 0.000102 | 0.000124 | 0.000190 | 0.000347 | 0.000993 | 0.015953 | 0.178598 | 0.006697 | 0.802865 | 0 |
| faster_rcnn | 137 | ap_candidates | 74713 | 0.001001 | 0.001016 | 0.001084 | 0.001585 | 0.003069 | 0.010428 | 0.146075 | 0.411808 | 0.025908 | 0.802865 | 0 |
| faster_rcnn | 137 | historical_threshold | 46 | 0.690768 | 0.691026 | 0.691458 | 0.698329 | 0.710452 | 0.737736 | 0.781559 | 0.798293 | 0.722354 | 0.802865 | 2955 |
| faster_rcnn | 137 | posthoc_n5_threshold | 33 | 0.702479 | 0.703131 | 0.704851 | 0.709372 | 0.722723 | 0.754644 | 0.786280 | 0.799614 | 0.733595 | 0.802865 | 2968 |
| faster_rcnn | 271 | all_retained_candidates | 300000 | 0.000010 | 0.000017 | 0.000024 | 0.000042 | 0.000066 | 0.000127 | 0.000807 | 0.012797 | 0.001568 | 0.968856 | 0 |
| faster_rcnn | 271 | ap_candidates | 13011 | 0.001001 | 0.001016 | 0.001084 | 0.001570 | 0.003142 | 0.011083 | 0.170405 | 0.703651 | 0.033793 | 0.968856 | 229 |
| faster_rcnn | 271 | historical_threshold | 134 | 0.692538 | 0.693125 | 0.711844 | 0.741686 | 0.797594 | 0.873751 | 0.925889 | 0.953130 | 0.810027 | 0.968856 | 2884 |
| faster_rcnn | 271 | posthoc_n5_threshold | 131 | 0.704770 | 0.707590 | 0.714279 | 0.742912 | 0.803174 | 0.875278 | 0.925969 | 0.953175 | 0.812707 | 0.968856 | 2887 |
| faster_rcnn | 314 | all_retained_candidates | 248406 | 0.000010 | 0.000010 | 0.000011 | 0.000014 | 0.000022 | 0.000044 | 0.000327 | 0.005946 | 0.001331 | 0.977414 | 0 |
| faster_rcnn | 314 | ap_candidates | 6228 | 0.001001 | 0.001014 | 0.001101 | 0.001681 | 0.003686 | 0.015514 | 0.350474 | 0.836296 | 0.051110 | 0.977414 | 830 |
| faster_rcnn | 314 | historical_threshold | 131 | 0.690089 | 0.693883 | 0.716391 | 0.774156 | 0.830732 | 0.890957 | 0.946977 | 0.962939 | 0.833257 | 0.977414 | 2882 |
| faster_rcnn | 314 | posthoc_n5_threshold | 127 | 0.707607 | 0.713234 | 0.736199 | 0.780815 | 0.835416 | 0.892840 | 0.947467 | 0.963019 | 0.837653 | 0.977414 | 2884 |
| yolo11s | 17 | all_retained_candidates | 16068 | 0.000010 | 0.000010 | 0.000011 | 0.000017 | 0.000031 | 0.000085 | 0.001099 | 0.024901 | 0.001718 | 0.613281 | 47 |
| yolo11s | 17 | ap_candidates | 846 | 0.001030 | 0.001030 | 0.001099 | 0.001808 | 0.004211 | 0.015869 | 0.149902 | 0.500781 | 0.031227 | 0.613281 | 2513 |
| yolo11s | 17 | historical_threshold | 111 | 0.050293 | 0.052075 | 0.056641 | 0.079834 | 0.115723 | 0.277344 | 0.523438 | 0.609375 | 0.189233 | 0.613281 | 2913 |
| yolo11s | 17 | posthoc_n5_threshold | 268 | 0.010010 | 0.010010 | 0.010986 | 0.017944 | 0.035156 | 0.090576 | 0.381152 | 0.598906 | 0.091489 | 0.613281 | 2814 |
| yolo11s | 42 | all_retained_candidates | 12889 | 0.000010 | 0.000010 | 0.000010 | 0.000015 | 0.000026 | 0.000066 | 0.000778 | 0.023726 | 0.001637 | 0.523438 | 126 |
| yolo11s | 42 | ap_candidates | 557 | 0.001030 | 0.001030 | 0.001099 | 0.001595 | 0.004333 | 0.020386 | 0.237500 | 0.445312 | 0.036458 | 0.523438 | 2674 |
| yolo11s | 42 | historical_threshold | 84 | 0.050293 | 0.050901 | 0.058350 | 0.087891 | 0.149414 | 0.285645 | 0.451953 | 0.491016 | 0.197722 | 0.523438 | 2942 |
| yolo11s | 42 | posthoc_n5_threshold | 197 | 0.010010 | 0.010010 | 0.011353 | 0.018311 | 0.037354 | 0.115723 | 0.410937 | 0.476875 | 0.097597 | 0.523438 | 2862 |
| yolo11s | 137 | all_retained_candidates | 2389 | 0.000010 | 0.000010 | 0.000010 | 0.000016 | 0.000033 | 0.000149 | 0.070020 | 0.594687 | 0.020517 | 0.871094 | 1880 |
| yolo11s | 137 | ap_candidates | 333 | 0.001030 | 0.001042 | 0.001366 | 0.002975 | 0.015442 | 0.225586 | 0.647656 | 0.808594 | 0.146702 | 0.871094 | 2770 |
| yolo11s | 137 | historical_threshold | 132 | 0.051025 | 0.054224 | 0.061035 | 0.152832 | 0.294922 | 0.566406 | 0.743555 | 0.828125 | 0.357039 | 0.871094 | 2894 |
| yolo11s | 137 | posthoc_n5_threshold | 182 | 0.010010 | 0.010859 | 0.012839 | 0.044128 | 0.174316 | 0.476562 | 0.678711 | 0.828125 | 0.265522 | 0.871094 | 2861 |
| yolo11s | 271 | all_retained_candidates | 45504 | 0.000010 | 0.000010 | 0.000010 | 0.000016 | 0.000031 | 0.000090 | 0.000404 | 0.001701 | 0.000142 | 0.038574 | 0 |
| yolo11s | 271 | ap_candidates | 784 | 0.001030 | 0.001030 | 0.001030 | 0.001366 | 0.001984 | 0.003601 | 0.012817 | 0.029076 | 0.003686 | 0.038574 | 2544 |
| yolo11s | 271 | historical_threshold | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 3000 |
| yolo11s | 271 | posthoc_n5_threshold | 54 | 0.010010 | 0.010172 | 0.010681 | 0.012817 | 0.015656 | 0.022949 | 0.033716 | 0.036763 | 0.018908 | 0.038574 | 2955 |
| yolo11s | 314 | all_retained_candidates | 23233 | 0.000010 | 0.000010 | 0.000010 | 0.000016 | 0.000029 | 0.000075 | 0.000519 | 0.008057 | 0.000797 | 0.421875 | 2 |
| yolo11s | 314 | ap_candidates | 738 | 0.001030 | 0.001030 | 0.001099 | 0.001595 | 0.003372 | 0.012817 | 0.130615 | 0.317715 | 0.022974 | 0.421875 | 2560 |
| yolo11s | 314 | historical_threshold | 81 | 0.050293 | 0.051465 | 0.056641 | 0.077148 | 0.117676 | 0.197266 | 0.351562 | 0.395313 | 0.153664 | 0.421875 | 2930 |
| yolo11s | 314 | posthoc_n5_threshold | 205 | 0.010010 | 0.010022 | 0.011060 | 0.018555 | 0.033691 | 0.087402 | 0.293359 | 0.376719 | 0.074796 | 0.421875 | 2848 |

## Execution and numerical handoff audit

The intended RTX 4060 Laptop GPU executed all runs sequentially under explicit AMP.
Historical internal YOLO accuracy used FP32, a disclosed comparison limitation.
One-ULP Torchvision inverse-resize upper-bound overshoots are numerically
canonicalized; raw coordinates are retained privately and replay-verified.
The initial failed attempt and implementation remain hash-preserved privately.

| Detector | Seed | Observed convolution dtype | Elapsed seconds | Repaired images | Repaired coordinates | Max correction (pixels) |
| --- | --- | --- | --- | --- | --- | --- |
| faster_rcnn | 17 | torch.float16 | 636.649808 | 17 | 36 | 0.000244 |
| faster_rcnn | 42 | torch.float16 | 616.853230 | 17 | 46 | 0.000244 |
| faster_rcnn | 137 | torch.float16 | 648.786194 | 13 | 25 | 0.000244 |
| faster_rcnn | 271 | torch.float16 | 1124.893487 | 15 | 68 | 0.000244 |
| faster_rcnn | 314 | torch.float16 | 611.615767 | 25 | 45 | 0.000244 |
| yolo11s | 17 | torch.bfloat16 | 269.557286 | 0 | 0 | 0.000000 |
| yolo11s | 42 | torch.bfloat16 | 265.578068 | 0 | 0 | 0.000000 |
| yolo11s | 137 | torch.bfloat16 | 260.145406 | 0 | 0 | 0.000000 |
| yolo11s | 271 | torch.bfloat16 | 266.600250 | 0 | 0 | 0.000000 |
| yolo11s | 314 | torch.bfloat16 | 267.532618 | 0 | 0 | 0.000000 |

Every checkpoint/config/protocol/dataset/environment and bundle hash is in the
source summary. Per-image prediction/count/score evidence remains private.
Equal-run mean/sample SD and defined-run counts for all scalar summaries
are also retained in that JSON; detections are never pooled across runs.
