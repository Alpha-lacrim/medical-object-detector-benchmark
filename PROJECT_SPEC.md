# Codex Task Prompt — Comparative Analysis of Object Detectors for Medical Imaging

> **This is the static requirements spec — it doesn't change.** For the session protocol (what to read/update each session), see `AGENTS.md`. For current project state (decisions made, files created), see `CODEX.md`. For the session-by-session log, see `HANDOFF.md`. For the sequence of batch instructions to work through this spec, see `BATCHES.md`.

## 0. Role and Framing

You are acting as a research engineer building a **controlled comparative study**, not a course-assignment script. The end product must satisfy two audiences at once: a grader checking off deliverables, and a reader who could reproduce your numbers from your repo alone. Treat every design choice (splits, seeds, hyperparameters, augmentation) as something that must be **documented and justified**, not just implemented. Where the assignment brief is ambiguous or internally inconsistent (noted below), state your interpretation explicitly in the repo rather than silently picking one.

**Known inconsistencies in the source assignment brief — resolve them as follows:**
- The brief's "Expected Deliverables" says *"three two detectors"* — this is a leftover edit artifact. Build **exactly two** detectors: Faster R-CNN (baseline) and one YOLO version.
- The brief's phase numbering skips from "Phase 8" to "Phase 10" with no Phase 9. Ignore the phase numbers; follow the **12-section Final Report Structure** in §7 below as the authoritative outline.
- Phase 2 says "verify five classes." This is almost certainly a template placeholder, not a property of any of the three linked datasets (see §1 — none is a clean 5-class detection set). **Do not hardcode 5 classes anywhere.** Determine the true class count from whichever dataset you select, document it, and derive all downstream code (label maps, model heads, confusion matrices) from that count programmatically.

---

## 1. Dataset — Decision Required First

The brief links three candidate datasets (verbatim URLs, extracted from the source PDF):

1. **RSNA Pneumonia Detection Challenge 2018** — https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
   Chest X-ray, single foreground class ("opacity"/pneumonia), bounding boxes, large (~26k images), hosted via Kaggle competition (requires Kaggle account + competition acceptance to download).
2. **Kaggle — Medical Image Dataset: Brain Tumor Detection** — https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection
   Brain MRI with bounding-box annotations, multi-class (tumor subtypes), smaller scale.
3. **Kaggle — MRI for Brain Tumor with Bounding Boxes** — https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes
   Similar to (2), different source/annotation set.

**Before writing any training code:**
- Inspect all three (image count, class list, annotation format, license/usage terms, class balance).
- Pick **one**. Write `docs/DATASET_CHOICE.md` justifying the choice on: annotation quality, class balance, appropriateness for a two-detector accuracy/robustness/explainability comparison, and licensing suitability for a course project.
- **Medical-imaging-specific rigor point (not in the original brief, but required here):** check whether the dataset has multiple images per patient/study. If so, split train/val/test **by patient ID**, not by image, to prevent patient-level leakage across splits. Document this explicitly even if the chosen dataset turns out to be single-image-per-patient.
- Downloading may require a Kaggle API token — write the download script to read credentials from environment variables / `~/.kaggle/kaggle.json`, never hardcode credentials, and fail with a clear message if absent.

---

## 2. Research Framing (what makes this a study, not a homework script)

State this explicitly in the report's Introduction and carry it through the code:

> **Research question:** Under identical data, identical evaluation protocol, and matched training budget, how do a two-stage anchor-based detector (Faster R-CNN) and a one-stage modern detector (YOLO) trade off detection accuracy, robustness to common image corruptions, and interpretability, on medical images — and which trade-off is preferable for a plausible deployment scenario (e.g., point-of-care triage vs. retrospective batch screening)?

Every phase should trace back to this question. Where compute or time forces a shortcut (e.g., single seed instead of multiple), state it as an explicit **limitation**, not a silent omission.

---

## 3. Compute Budget & Hardware-Specific Scoping

Target hardware: **ASUS ROG Strix G16, Intel i7-13650HX, RTX 4060 Laptop GPU (8GB VRAM), 16GB DDR5-4800 system RAM.** This is a capable single-GPU setup, but 8GB VRAM and 16GB system RAM are real ceilings — scope the experiment design to them explicitly rather than discovering OOM errors mid-run. Build all of the following as config options (`configs/*.yaml`), not hardcoded, but use these as the defaults:

**Model scale**
- Faster R-CNN: `fasterrcnn_resnet50_fpn_v2`, ResNet-50 backbone, fits in 8GB at batch size 2–4 with mixed precision. Do not go to a heavier backbone (e.g., ResNeXt-101) — it's not needed for this comparison and risks OOM.
- YOLO: use the **small or medium** scale (e.g. `yolo11s`/`yolo11m` or `yolo26s`/`yolo26m`), not `l`/`x`. Nano (`n`) is an option if VRAM or time gets tight but note it trades some accuracy that could confound the comparison — prefer `s`/`m` and drop to `n` only if you hit hard constraints, and say so if you do.

**Precision and memory**
- Mixed precision (AMP) is not optional here — enable it for both frameworks (`torch.cuda.amp.autocast` + `GradScaler` for Faster R-CNN; Ultralytics defaults to AMP, verify `amp=True`). Roughly halves memory pressure and speeds up training on the 4060's tensor cores.
- Image size: 640px (long side) is a reasonable default for both detectors on this GPU. If Faster R-CNN OOMs at batch size 2 even with AMP, drop resolution before dropping to batch size 1 — batch size 1 makes BatchNorm statistics unstable during fine-tuning.
- Gradient accumulation if you need an effectively larger batch than VRAM allows for stable training — don't just train at batch size 1 and call it done.
- DataLoader: `num_workers` around 6–8 (the 13650HX has 14 cores, leave headroom for the OS/GPU driver), `pin_memory=True`, `persistent_workers=True`.
- **RAM caution:** Ultralytics supports `cache="ram"` for faster training — with only 16GB system RAM this can start competing with the OS and your own preprocessing scripts. Use `cache="disk"` or no caching instead, and don't load a full dataset into memory as decoded arrays anywhere in `src/data/`.

**Dataset scale**
- RSNA Pneumonia (~26k images) is large for iterating on a single laptop GPU within a course-project timeframe. If you select it, **subsample to a fixed, stratified subset** (e.g. 3,000–5,000 images preserving positive/negative and class balance) and document the subsampling procedure and its effect on how far the conclusions generalize. The two brain-tumor datasets are smaller and likely more tractable end-to-end without subsampling — factor this into the §1 dataset decision, not just annotation quality.

**Seeds and the robustness/explainability grid — a pragmatic, honest scoping**
Running 3 full seeds through *every* phase (training × robustness grid × Grad-CAM × stats) multiplies compute 3× across the whole pipeline, which is unrealistic on this hardware within a course timeline. Scope it instead:
- Run the **primary pipeline (Phases 3–4, training) at 1 seed**, fully, for both detectors.
- If time allows, re-run **just the final headline metrics (Phase 5 quantitative comparison)** at 2 additional seeds (3 total) — this is what feeds the Phase 8 statistical tests, so it's the highest-value place to spend extra seed budget, not the robustness/explainability phases.
- For **Phase 6 (robustness)**, run the full corruption × severity grid on a **stratified subsample of the test set** (e.g. 200–400 images) rather than the entire test set — document the subsample size and how it was drawn. Running every corruption × severity combination over a full multi-thousand-image test set for both detectors is the single most expensive part of this project; don't let it silently balloon runtime.
- For **Phase 7 (Grad-CAM)**, qualitative heatmaps only need a handful of representative good/bad/failure cases (single digits to low tens); the quantitative pointing-game metric can run on the same robustness subsample from Phase 6 rather than a separate full pass.
- State all of this explicitly in `docs/LIMITATIONS.md` — reduced seed count outside the headline comparison and a subsampled robustness/explainability set are legitimate, common scoping decisions in resource-constrained research, but only if disclosed.

**Time-boxing**
- Before committing to a full training run, benchmark 2–3 epochs on the real data at the chosen resolution/batch size and extrapolate epoch time — don't discover a 6-hour-per-epoch run after starting it.
- Use early stopping on validation mAP (patience-based) rather than a large fixed epoch count, for both detectors, so training time is comparable and not arbitrarily different between them.
- If a particular run (e.g. the full robustness grid, or the 3-seed re-runs) turns out to be impractical locally within your time budget, a free-tier Kaggle Notebook or Colab GPU (T4-class, comparable or better VRAM) is a reasonable fallback for that specific piece — keep primary development and debugging local, but don't burn days of laptop time on something a free cloud GPU session would clear in an hour.

---

## 4. Repository Structure

```
project/
├── README.md                     # exact commands to reproduce every table/figure
├── environment.yml / requirements.txt
├── configs/
│   ├── dataset.yaml
│   ├── faster_rcnn.yaml
│   └── yolo.yaml
├── data/
│   ├── raw/            (gitignored)
│   ├── processed/       (gitignored)
│   └── splits/          (patient-level split manifests, committed)
├── src/
│   ├── data/            (download.py, prepare.py, visualize.py, datasheet.py)
│   ├── models/          (faster_rcnn.py, yolo.py)
│   ├── train.py
│   ├── evaluate.py       # unified evaluator — see §4, Phase 5
│   ├── robustness/       (corruptions.py, run_robustness.py)
│   ├── explainability/   (gradcam.py, pointing_game.py)
│   ├── stats/            (significance_tests.py)
│   └── utils/            (seed.py, logging_utils.py, flops.py)
├── notebooks/            (EDA only — no training logic lives here)
├── results/
│   ├── tables/  figures/  logs/  checkpoints/(gitignored)
├── report/
│   └── report.md          (or .tex — matches §7 structure)
├── tests/                  # pytest: data pipeline, metric functions, corruption functions
└── docs/
    ├── DATASET_CHOICE.md
    ├── LITERATURE_REVIEW.md
    └── LIMITATIONS.md
```

Config-driven, no hardcoded paths or magic numbers in training scripts. Every experiment run must be reproducible from a config file plus a seed.

---

## 5. Phase-by-Phase Implementation Instructions

### Phase 0 — Reproducibility scaffolding (not in brief, do first)
- Seed utility fixing Python/NumPy/PyTorch RNGs; log seed, package versions (`pip freeze`), and GPU/driver info to every run's output directory.
- Enable deterministic algorithms where feasible; document any operation that can't be made deterministic (e.g., certain CUDA kernels).
- Git-init the repo; commit after each phase with a message describing what changed and why.

### Phase 1 — Literature Review
- `docs/LITERATURE_REVIEW.md`: summarize, **in your own words** (no reproduced text), the architecture you're using for the YOLO side (YOLOv8, or a newer Ultralytics release — see note below), plus Grad-CAM and explainable AI for object detection.
- **YOLO version note:** the brief says "Yolo.v8 or a newer version." As of the current Ultralytics release lineup, YOLO11 and YOLO26 (Jan 2026, NMS-free head, is the current SOTA release) postdate v8. Pick one, state the version pinned (`ultralytics==<version>`) in `requirements.txt`, and briefly justify the choice (e.g., YOLO11 for a more literature-comparable anchor-free baseline vs. YOLO26 for current SOTA at the cost of a newer/less-studied architecture). Do not silently default to whatever `pip install ultralytics` resolves to — pin it.
- Include a short related-work table: prior detector benchmarks on medical imaging, with proper citations (BibTeX in `report/references.bib`).

### Phase 2 — Dataset Preparation
- Download, verify image count, verify annotation integrity, determine and record the **actual** class list (see §1), verify/construct patient-safe train/val/test splits, check annotation quality (malformed boxes, off-image boxes, duplicate labels), convert annotations to a single canonical intermediate format (e.g., COCO JSON) that both detectors' data loaders read from — this is what makes the later "unified evaluator" possible.
- Produce a **datasheet** (`docs/DATASET_CHOICE.md` or a separate `docs/DATASHEET.md`): collection method, known biases, class imbalance, licensing terms, and any exclusions you made.
- Visualize random samples with bounding boxes and class labels; save to `results/figures/`.
- Deliverables: dataset report, class distribution plot, sample images.

### Phase 3 — Baseline: Faster R-CNN
- `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` (or equivalent), pretrained on COCO, fine-tuned via transfer learning on the chosen dataset.
- Log training curves (loss, precision, recall, mAP) every epoch to `results/logs/`.
- Record FPS, parameter count, GFLOPs (e.g., `fvcore` or `thop`), and model size.
- Per §3, run 1 seed here fully; the additional 2 seeds (3 total, for mean ± std) get spent on the Phase 5 headline metrics specifically, not repeated through this whole phase. Note this scoping decision in `docs/LIMITATIONS.md` rather than presenting the single-seed training curves as if variance were characterized.
- Deliverables: baseline performance tables and figures.

### Phase 4 — YOLO Implementation
- Replace only the detector. Keep dataset, augmentation policy, and optimizer as close to identical as the two frameworks allow.
- **Important:** Ultralytics' training loop applies its own built-in augmentations (mosaic, etc.) that torchvision's Faster R-CNN pipeline won't replicate by default. This is a genuine confound — don't paper over it. Either (a) disable YOLO's extra augmentations to match Faster R-CNN's pipeline as closely as possible, or (b) leave them on and **explicitly document this as a threat to the validity of a "fair" comparison** in the Discussion section. Pick one and justify it; don't leave it unaddressed.
- Deliverables: performance report, training curves, inference speed.

### Phase 5 — Quantitative Performance Comparison
- **Build one unified evaluation harness** (`src/evaluate.py`) that both models' predictions are fed into, rather than trusting each library's own internal mAP calculator (they use different matching/NMS conventions and will not be numerically comparable otherwise). Use `pycocotools` or an equivalent COCO-style evaluator applied identically to both.
- Metrics — detection: Precision, Recall, F1, IoU, Dice, mAP@0.5, mAP@0.5:0.95.
- Metrics — computational: FPS, parameter count, GFLOPs, peak GPU memory, training time, inference time.
- Produce comparison tables (`results/tables/`).

### Phase 6 — Robustness Evaluation
- Corruption pipeline (`albumentations` or `imgaug`) covering: lighting (darker/brighter), noise (Gaussian, salt & pepper), blur (Gaussian, motion), compression (JPEG quality 20%, 50%).
- **Beyond the brief's single-severity ask:** parametrize each corruption by severity level (e.g., 3–5 levels each, ImageNet-C style — cite Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," in the literature review) so you can report a mean performance degradation curve per corruption type per model, not just one before/after number. This is what turns "robustness eyeballing" into a robustness benchmark. Per §3, run this full grid on the stratified test subsample, not the entire test set.
- Report both raw degraded performance and relative degradation (e.g., mean Performance under Corruption / clean performance) per detector, per corruption type and severity.

### Phase 7 — Explainability Analysis
- Grad-CAM (or Grad-CAM++/Eigen-CAM if Grad-CAM proves noisy for the one-stage detector's architecture) applied to **comparable backbone layers** in both models so the visual comparison is apples-to-apples.
- Qualitative: heatmaps for good predictions, bad predictions, and failure cases, for both models side by side.
- **Beyond the brief's qualitative-only ask:** add a quantitative explainability metric — e.g., a pointing-game accuracy or energy-based pointing score measuring the fraction of Grad-CAM mass that falls inside the ground-truth bounding box. Purely qualitative "where is it looking" claims are weak evidence on their own; back them with a number.
- Answer explicitly: where is the model looking, is it focusing on the actual lesion/finding vs. background/artifacts, and does that differ systematically between the two detectors.

### Phase 8 — Statistical Analysis
- Do not just compare point estimates. Use, as appropriate to what's being compared:
  - Bootstrap confidence intervals (per-image resampling) for each metric.
  - Paired permutation test / Wilcoxon signed-rank test for repeated per-image measurements between the two detectors.
  - McNemar's test only if you're comparing paired binary correct/incorrect classification decisions on matched detections — don't force it onto continuous metrics like mAP.
  - **If running multiple corruption/severity comparisons in Phase 6 with the same test repeatedly, correct for multiple comparisons** (Holm-Bonferroni or similar) rather than reporting a wall of uncorrected p-values.
- Report p-values, confidence intervals, and effect sizes (e.g., Cohen's d or rank-biserial correlation) — not p-values alone.

### Phases 9–12 — Discussion, Limitations, Conclusion
- Scientific Discussion: which detector is more suitable for which deployment scenario and **why** — tie back explicitly to the accuracy/robustness/interpretability/compute trade-offs measured above, not just "which had the higher mAP."
- Deployment considerations relevant to medical imaging specifically: FPS/model-size trade-off for point-of-care vs. batch/retrospective use, and a brief note that clinical deployment would require regulatory validation (FDA/CE) and prospective clinical validation beyond this benchmark — this is a scope-of-claims point, not something to implement.
- `docs/LIMITATIONS.md`: single-dataset scope, seed count, any augmentation asymmetry from Phase 4, dataset size/class imbalance effects on statistical power.
- Conclusions and Future Work.

---

## 6. Coding Standards

- Type hints and docstrings on all functions in `src/`.
- `pytest` unit tests for: annotation-format conversion, the unified evaluator's metric functions (test against known small hand-computed cases), and each corruption function (verify it actually degrades the image as intended, e.g., blur reduces high-frequency content).
- No hardcoded file paths, class counts, or hyperparameters inside `.py` files — everything through `configs/*.yaml`.
- Every number that ends up in `report/report.md` must be traceable to a script/command in `README.md` that regenerates it.

## 7. Execution Mode

Work through §4 phase by phase. After each phase:
1. Run the phase's tests (if any) and a quick smoke test on a small data subset before committing to a full run.
2. Write a short status note (what was built, what was verified, what's deferred) before moving to the next phase.
3. If a phase requires a decision that materially changes downstream work (dataset choice, YOLO version, whether to disable YOLO's built-in augmentations), stop and ask rather than assuming — everything else, including compute scoping (fixed in §3 for an RTX 4060 Laptop 8GB / 16GB RAM setup), make a documented default assumption and proceed.

If a benchmarked epoch time (per §3, "Time-boxing") makes the planned run impractical within a reasonable timeframe on this hardware, flag it and propose a concrete adjustment (smaller model scale, smaller resolution, smaller dataset subsample, or offloading that specific run to a free-tier cloud GPU) rather than silently shrinking scope or running it anyway.

## 8. Final Report Structure (authoritative — 12 sections)

1. Introduction
2. Literature Review
3. Materials and Dataset
4. Experimental Methodology
5. Baseline Faster R-CNN Implementation
6. YOLO Implementation
7. Quantitative Performance Comparison
8. Robustness Evaluation
9. Explainability Analysis
10. Statistical Analysis
11. Discussion
12. Conclusions and Future Work

## 9. Definition of Done

- [ ] Two detectors (Faster R-CNN + one YOLO version), trained under matched conditions, with any unavoidable asymmetries documented.
- [ ] Standardized benchmark: accuracy metrics + computational metrics, both computed through one unified evaluator.
- [ ] Robustness benchmark across ≥4 corruption families, multi-severity, with degradation curves.
- [ ] Grad-CAM explainability: qualitative heatmaps (good/bad/failure cases) + a quantitative localization metric.
- [ ] Statistical tests with p-values, CIs, effect sizes, multiple-comparison correction where relevant.
- [ ] Discussion that argues which paradigm suits which deployment scenario and why, grounded in the measured trade-offs.
- [ ] `docs/LIMITATIONS.md` stating scope honestly (single dataset, seed count, robustness/explainability subsample size, and the RTX 4060 8GB / 16GB RAM compute constraints that drove those choices per §3).
- [ ] `README.md` sufficient for a third party to reproduce every table and figure from a clean checkout.
