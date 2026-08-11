# Medical Object Detector Benchmark

This repository is a controlled comparison of Faster R-CNN and YOLO11s on the
RSNA Pneumonia Detection Challenge. It measures clean detection performance,
compute, common-corruption robustness, Grad-CAM localization, and paired
statistical uncertainty under one patient-safe data and evaluation protocol.

The project is complete. The 12-section study is in
[`report/report.md`](report/report.md), the consolidated scope statement is in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), and the exact generated evidence
is under [`results/`](results/). This is a retrospective benchmark, not a
clinical device or diagnostic system.

## Headline result

Across seeds 17, 42, and 137, Faster R-CNN achieves mAP@0.5:0.95 of
0.1023 ± 0.0036 and recall of 0.6381 ± 0.0526, compared with 0.0549 ± 0.0080
and 0.1356 ± 0.0094 for YOLO11s. YOLO11s is more selective and much cheaper:
precision is 0.3730 ± 0.0395 versus 0.1626 ± 0.0439, throughput is
52.94 ± 10.65 FPS versus 17.42 ± 5.69, and it has 9.43 M parameters versus
43.26 M. The paired clean analysis retains Holm-corrected evidence for Faster
R-CNN's recall and AP advantages and YOLO11s' precision advantage.

On the seed-17 300-image common-corruption sample, mean mAP@0.5:0.95 retention
is 0.7638 for Faster R-CNN and 0.7091 for YOLO11s. Both detectors have weak
Grad-CAM localization: mean energy-in-box is 0.0869 and 0.0975, with pointing
accuracy 0.1091 and 0.1261. These values do not establish clinical validity.

## Reproduction assumptions

Run every command below from the repository root in Windows PowerShell. A clean
reproduction requires:

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/);
- an NVIDIA GPU/driver compatible with the pinned CUDA 12.4 Torch wheels;
- sufficient disk space for the RSNA archive, 5,000 processed images,
  checkpoints, prediction bundles, and logs;
- a Kaggle account that has joined the RSNA competition and accepted its rules;
  and
- either `KAGGLE_USERNAME`/`KAGGLE_KEY` or a valid
  `$HOME\.kaggle\kaggle.json`. Credentials must never be committed.

Raw images, processed images, downloaded pretrained weights, and trained
checkpoints are intentionally Git-ignored. The commands below regenerate them.
The measured workstation used an RTX 4060 Laptop GPU with 8 GB VRAM, 16 GB RAM,
and an i7-13650HX; timing will vary on other machines.

## 1. Create the pinned environment

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv --default-index https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps --editable .

$benchmarkPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $benchmarkPython -m pip check
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests
```

`requirements.txt` is the authoritative exact dependency list. Each experiment
also writes `pip_freeze.txt` and `run_environment.json` before CUDA
initialization. Deterministic algorithms use warning mode because the pinned
Torchvision CUDA ROI Align backward used by Grad-CAM is not bitwise
deterministic.

## 2. Acquire and prepare the dataset

The downloader requests the three official Stage 2 files declared in
`configs/dataset.yaml`. The official mapping is fetched directly from RSNA.

```powershell
& $benchmarkPython -m src.data.download --check-credentials
& $benchmarkPython -m src.data.download --config configs/dataset.yaml
Expand-Archive -LiteralPath data/raw/rsna-pneumonia/stage_2_train_images.zip -DestinationPath data/raw/rsna-pneumonia -Force
Invoke-WebRequest -Uri "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json" -OutFile data/raw/rsna-pneumonia/mappings.json
& $benchmarkPython -m src.data.prepare --config configs/dataset.yaml --convert-images
& $benchmarkPython -m src.data.visualize --config configs/dataset.yaml
```

Preparation verifies the pinned mapping digest, audits all 26,684 labeled
studies, reconstructs true NIH patient groups, writes the exact 5,000-study
70/15/15 split, converts the selected DICOMs, and creates canonical COCO JSON.
Visualization regenerates:

- `results/figures/rsna_class_distribution.png` (report Figure 1);
- `results/figures/rsna_annotation_samples.png` (report Figure 2); and
- `results/figures/rsna_eda_summary.json`.

The dataset table in report Section 3 is a rounded rendering of
`data/manifests/rsna-pneumonia-5000-audit.json` and the generated/committed split
manifests from the preparation command. To regenerate metadata, splits, and
COCO annotations without decoding pixels, use:

```powershell
& $benchmarkPython -m src.data.prepare --config configs/dataset.yaml --metadata-only
```

If Kaggle supplies the aggregate competition archive instead of the individual
image ZIP, extract only its training-image member before preparation:

```powershell
tar -xf data/raw/rsna-pneumonia/rsna-pneumonia-detection-challenge.zip -C data/raw/rsna-pneumonia stage_2_train_images
```

## 3. Train and finalize the primary Faster R-CNN run

`configs/faster_rcnn.yaml` fixes the seed-17 ResNet-50/FPN model, data,
optimizer, float16 AMP, 640-pixel transform, physical batch 2, accumulation 2,
and output identity. Run the required readiness, smoke, and three-epoch timing
gate before the full run:

```powershell
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode preflight
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode smoke
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode benchmark
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed17_benchmark/benchmark_estimate.json
```

The train command regenerates the report Section 5 evidence:

- `results/tables/faster_rcnn_baseline_validation.csv` (report Table 2);
- `results/tables/faster_rcnn_compute.csv` (report Table 2); and
- `results/figures/faster_rcnn_training_curves.png` (report Figure 3).

If optimization completed but final profiling or plotting was interrupted,
regenerate the derived artifacts from the saved best checkpoint without
retraining:

```powershell
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn.yaml --mode finalize
```

## 4. Train and finalize the primary YOLO11s run

The prepare mode makes an audited YOLO view of the canonical data. The remaining
commands verify the pinned pretrained weight, run the GPU smoke/timing gate, and
train seed 17:

```powershell
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode prepare
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode preflight
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode smoke
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode benchmark
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode train
```

The train command regenerates the report Section 6 evidence:

- `results/tables/yolo_baseline_validation.csv` (report Table 3);
- `results/tables/yolo_compute.csv` (report Table 3); and
- `results/figures/yolo_training_curves.png` (report Figure 4).

If reporting was interrupted after training, use:

```powershell
& $benchmarkPython -m src.models.train_yolo --config configs/yolo.yaml --mode finalize
```

## 5. Train the additional seeds and run the unified test evaluator

The additional configs change only seed and artifact identity. First derive
seed-specific timing approvals from the accepted seed-17 Faster R-CNN gate,
then train seeds 42 and 137 for both detectors:

```powershell
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode seed-gates

& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed42.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed42_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_faster_rcnn --config configs/faster_rcnn_seed137.yaml --mode train --approved-benchmark results/logs/faster_rcnn_rsna_seed137_benchmark/benchmark_estimate.json
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed42.yaml --mode train
& $benchmarkPython -m src.models.train_yolo --config configs/yolo_seed137.yaml --mode train

& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode preflight
& $benchmarkPython -m src.evaluate --config configs/evaluation.yaml --mode evaluate
```

Only the final two commands open the held-out test annotation. Both detector
adapters feed one pycocotools/operating-point evaluator; framework-native mAP is
not used in the comparison. The evaluate command regenerates:

- `results/tables/detector_comparison.csv` (report Tables 4a and 4b);
- `results/tables/detector_comparison_mean_std.csv`;
- `results/tables/detector_comparison_per_seed.csv`; and
- `results/logs/phase5_evaluation/summary.json` plus the six frozen prediction
  bundles used by statistics.

## 6. Run the common-corruption benchmark

The run command deterministically draws or verifies the 300-image sample,
filters the clean seed-17 prediction evidence, and infers every one of the 70
corrupted detector conditions. Completed condition bundles are resumable.

```powershell
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode preflight
& $benchmarkPython -m src.robustness.run_robustness --config configs/corruptions.yaml --mode run
```

This regenerates the report Section 8 evidence:

- `results/tables/robustness_results.csv` (report Table 5 and all raw/relative
  metrics);
- `results/tables/robustness_curves.csv`;
- `results/tables/robustness_family_mean_curves.csv`;
- `results/figures/robustness_map_50_95_raw.png` (report Figure 5);
- `results/figures/robustness_map_50_95_relative.png` (report Figure 6); and
- `results/logs/phase6_robustness/summary.json` plus 72 prediction bundles.

## 7. Run the Grad-CAM analysis

Preflight binds the analysis to the completed robustness sample and primary
checkpoint hashes. The smoke pass is bounded to two positive images per model;
the full run covers all 111 boxes and the predeclared qualitative cases.

```powershell
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode preflight
& $benchmarkPython -m pytest tests/test_gradcam.py tests/test_pointing_game.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode smoke
& $benchmarkPython -m src.explainability.run_explainability --config configs/explainability.yaml --mode run
```

This regenerates the report Section 9 evidence:

- `results/tables/gradcam_localization_summary.csv` (report Table 6);
- `results/tables/gradcam_localization_per_target.csv`;
- `results/tables/gradcam_qualitative_cases.csv`;
- `results/figures/gradcam_good_predictions.png` (report Figure 7);
- `results/figures/gradcam_bad_predictions.png` (report Figure 8);
- `results/figures/gradcam_failure_cases.png` (report Figure 9); and
- `results/logs/phase7_explainability/summary.json`.

## 8. Run the paired statistical analysis

This CPU phase reads the six frozen clean bundles and all 72 robustness
bundles. It does not rerun model inference.

```powershell
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode preflight
& $benchmarkPython -m src.stats.run_statistics --config configs/statistics.yaml --mode run
```

This regenerates:

- `results/tables/statistical_clean_comparison.csv` (report Table 7);
- `results/tables/statistical_robustness_comparison.csv` (the report Section 10
  corruption results); and
- `results/logs/phase8_statistics/summary.json`.

The configuration fixes 2,000 paired bootstrap draws, 5,000 paired
permutations, 95% pointwise percentile intervals, paired jackknife Cohen's d,
and Holm correction. Aggregate AP is reconstructed at every draw rather than
approximated as a mean of per-image AP.

## Report artifact-to-command index

Report tables are rounded Markdown views of generated machine-readable
artifacts; the report assembly step does not recompute values.

| Report item | Generated source | Regenerating command |
|---|---|---|
| Table 1; Figures 1–2 | audit/split manifests; `rsna_*.png` | `src.data.prepare`, then `src.data.visualize` in §2 |
| Table 2; Figure 3 | `faster_rcnn_*.csv`; Faster curve | seed-17 Faster R-CNN train/finalize in §3 |
| Table 3; Figure 4 | `yolo_*.csv`; YOLO curve | seed-17 YOLO train/finalize in §4 |
| Tables 4a–4b | `detector_comparison*.csv` | unified `src.evaluate --mode evaluate` in §5 |
| Table 5; Figures 5–6 | `robustness*.csv`; robustness plots | robustness `--mode run` in §6 |
| Table 6; Figures 7–9 | `gradcam*.csv`; Grad-CAM plots | explainability `--mode run` in §7 |
| Table 7 and corruption inference | `statistical_*.csv` | statistics `--mode run` in §8 |

## Definition of Done audit

Every item in `PROJECT_SPEC.md` §9 is satisfied:

- [x] **Two detectors under matched conditions.** Faster R-CNN and YOLO11s are
  trained on identical patient-safe splits, resolution, seed grid, evaluation
  protocol, and no-augmentation policy. Necessary precision, normalization,
  learning-rate, batch, and scheduler asymmetries are documented in the model
  reports and `docs/LIMITATIONS.md`.
- [x] **Standardized predictive and compute benchmark.** `src/evaluate.py`
  routes both adapters through the same operating-point matcher and
  pycocotools evaluator; comparison artifacts include accuracy, latency/FPS,
  parameters, GFLOPs, memory, and training time.
- [x] **Multi-family, multi-severity robustness.** Seven corruption types in
  four families are evaluated at five severities, with raw and clean-relative
  curves and complete result tables.
- [x] **Qualitative and quantitative Grad-CAM.** Three paired figure categories
  cover good predictions, false positives, and false-negative proxies; all 111
  boxes receive energy-in-box and pointing-game analysis, with one explicit
  zero-energy map.
- [x] **Statistical inference.** The frozen clean and corruption predictions
  have paired bootstrap CIs, paired permutation p-values, paired jackknife
  effect sizes, and Holm correction. Non-estimable rows and the reason McNemar
  is inapplicable are explicit.
- [x] **Scenario-grounded discussion.** Report Section 11 weighs measured
  accuracy, robustness, interpretability, and compute for high-sensitivity
  retrospective screening, constrained point-of-care assistance, and
  autonomous use.
- [x] **Honest consolidated limitations.** `docs/LIMITATIONS.md` covers the
  single dataset, three-seed headline scope, primary-seed 300-image/111-box
  robustness and explainability scope, augmentation choice, detector
  asymmetries, and RTX 4060 8 GB / 16 GB RAM constraints.
- [x] **Clean-checkout reproduction commands.** Sections 1–8 above provide the
  exact ordered commands and the artifact index maps every report table and
  figure to its generating command.

## Repository verification

After a reproduction or code change, run:

```powershell
& $benchmarkPython -m pytest -q
& $benchmarkPython -m ruff check src tests
git diff --check
```

See `docs/REPRODUCIBILITY.md` for the seed/environment contract and the
phase-specific documents under `docs/` for complete metric, corruption,
Grad-CAM, and inference definitions.
