# BATCHES.md — Session-by-Session Prompts

Paste one batch at a time as your message to Codex CLI, in order. Each batch ends at a natural checkpoint — don't skip the "stop for review" instruction inside each one, even if Codex offers to keep going. Before pasting Batch N, make sure Batch N-1's review items actually got resolved (dataset looks right, training curves look sane, etc.) — these are the points where a bad call is cheap to catch and expensive to discover three phases later.

All four project files (`PROJECT_SPEC.md`, `AGENTS.md`, `CODEX.md`, `HANDOFF.md`) must exist at the repo root before running Batch 0.

---

## BATCH 0 — Repository Bootstrap

```
Read AGENTS.md, CODEX.md, and HANDOFF.md now — all three already exist in this repo root.

Set up the project skeleton per PROJECT_SPEC.md §4 (Repository Structure): full directory
tree, requirements.txt pinning exact versions of torch, torchvision, ultralytics, pycocotools,
albumentations, scipy, pytest (per PROJECT_SPEC.md §5 Phase 1's note on pinning the YOLO
version — leave the exact YOLO version as a placeholder for now, that gets decided in Batch 1's
literature review).

Implement src/utils/seed.py per PROJECT_SPEC.md §5 Phase 0: fixes Python/NumPy/PyTorch RNGs,
logs seed + package versions + GPU/driver info to each run's output directory. Write a unit
test for it under tests/.

Git init if not already done, first commit.

Do not touch any dataset or model code yet — this batch is infrastructure only.

When done: update CODEX.md's File Map with what you created, and append a HANDOFF.md entry
for this session. Then stop and summarize what you built so I can review before Batch 1.
```

---

## BATCH 1 — Dataset Investigation, Choice, and Preparation

```
Read HANDOFF.md and CODEX.md first for current state.

Follow PROJECT_SPEC.md §1 (Dataset) and §5 Phases 1–2.

1. Inspect all three linked datasets (image counts, actual class list, annotation format,
   license terms, class balance, whether there are multiple images per patient/study).
2. Pick one. Write docs/DATASET_CHOICE.md justifying the choice against the criteria in
   §1 — including whether it needs subsampling per §3's hardware scoping if it's RSNA-scale.
3. Build src/data/download.py reading Kaggle credentials from environment variables or
   ~/.kaggle/kaggle.json — never hardcoded, fail with a clear message if absent.
4. Convert annotations to one canonical intermediate format (COCO JSON) that both detectors'
   data loaders will read from later.
5. Determine and record the actual class list/count in CODEX.md — do not assume 5 classes.
6. Construct the train/val/test split. If the dataset has multiple images per patient/study,
   split by patient ID, not by image, to prevent leakage — document this either way in
   docs/DATASHEET.md even if it turns out not to apply.
7. Run EDA: class distribution plot, sample images with bounding boxes and labels, save to
   results/figures/. Check annotation quality (malformed/off-image/duplicate boxes).
8. Write docs/DATASHEET.md (collection method, biases, licensing, any exclusions made).

Do NOT start any model training in this batch.

When done: update CODEX.md's Decisions Log (dataset choice, class list, split strategy) and
File Map, append a HANDOFF.md entry. Stop for my review of the dataset choice and EDA before
Batch 2 — this decision affects everything downstream.
```

---

## BATCH 2 — Faster R-CNN Baseline

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §5 Phase 3 and the hardware scoping in §3: fasterrcnn_resnet50_fpn_v2,
AMP mandatory, batch size 2–4, drop resolution before dropping batch size below 2, gradient
accumulation if needed for stable BatchNorm statistics, early stopping on validation mAP.

Before committing to a full run: benchmark 2–3 epochs at the chosen config and report
estimated total training time. Stop and get my sign-off on that estimate before running the
full training.

Once approved: run the full training (1 seed per §3's seed-scoping — do not do 3 seeds here,
that budget is reserved for Batch 4). Log training curves (loss, precision, recall, mAP) to
results/logs/. Record FPS, parameter count, GFLOPs, model size. Produce baseline performance
tables/figures in results/tables/ and results/figures/.

When done: update CODEX.md (config used, final metrics, checkpoint path, actual training time)
and append a HANDOFF.md entry. Stop for my review of the training curves and benchmark numbers
before Batch 3.
```

---

## BATCH 3 — YOLO Implementation

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §5 Phase 4, using the YOLO version decided in Batch 1's literature
review (if not yet pinned in CODEX.md, do that first and log it there — small/medium scale
per §3, not large/x).

Explicitly decide and document, before training, how you're handling the augmentation
asymmetry called out in §5 Phase 4: either disable Ultralytics' built-in extra augmentations
to match the Faster R-CNN pipeline, or leave them on and document it as a threat to the
fairness of the comparison in docs/LIMITATIONS.md. Pick one now — don't defer this.

Train 1 seed, matching the hardware scoping in §3 (AMP, batch size, early stopping on
validation mAP so training time is comparable to the Faster R-CNN run, not arbitrarily
different). Log training curves, FPS, params, GFLOPs, model size, inference speed.

When done: update CODEX.md and append a HANDOFF.md entry. Stop for my review before Batch 4.
```

---

## BATCH 4 — Unified Evaluator, Quantitative Comparison, Extra Seeds

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §5 Phase 5. Build ONE evaluation harness (src/evaluate.py) that both
models' raw predictions are fed into — do not use torchvision's or Ultralytics' own internal
mAP calculators independently, they use different matching conventions and won't be
comparable. Use pycocotools or an equivalent applied identically to both.

Compute: Precision, Recall, F1, IoU, Dice, mAP@0.5, mAP@0.5:0.95, plus FPS, parameters,
GFLOPs, peak GPU memory, training time, inference time for both detectors.

Per §3's seed scoping: this is where the extra 2 seeds (3 total) get spent. Retrain both
detectors 2 more times each with the same configs from Batches 2–3, and report mean ± std
for the headline metrics. Produce comparison tables in results/tables/.

When done: update CODEX.md and append a HANDOFF.md entry. Stop for my review of the
comparison tables before Batch 5 — these numbers anchor the statistical tests in Batch 7.
```

---

## BATCH 5 — Robustness Evaluation

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §5 Phase 6 and the subsampling scope in §3: draw a stratified
subsample of the test set (200–400 images), document the sampling procedure in
docs/LIMITATIONS.md.

Implement the corruption pipeline (albumentations/imgaug): lighting (darker/brighter), noise
(Gaussian, salt & pepper), blur (Gaussian, motion), compression (JPEG 20%, 50%) — each at
3–5 severity levels, ImageNet-C style. Run the full grid for both detectors on the
subsample. Report mean performance degradation curves per corruption type per model, plus
relative degradation (corrupted / clean performance).

When done: update CODEX.md and append a HANDOFF.md entry. Stop for my review before Batch 6.
```

---

## BATCH 6 — Explainability Analysis

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §5 Phase 7. Implement Grad-CAM applied to comparable backbone layers
in both models. Produce qualitative heatmaps for a handful of good predictions, bad
predictions, and failure cases, for both models side by side.

Implement a quantitative pointing-game (or energy-based pointing) metric measuring how much
Grad-CAM mass falls inside the ground-truth box — run it on the same robustness subsample
from Batch 5 rather than a separate full pass.

Answer explicitly in a short write-up: where is each model looking, is it focused on the
actual finding vs. background/artifacts, and does that differ systematically between the two.

When done: update CODEX.md and append a HANDOFF.md entry. Stop for my review before Batch 7.
```

---

## BATCH 7 — Statistical Analysis

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §5 Phase 8, using the 3-seed results from Batch 4 and the corruption
grid from Batch 5. Implement bootstrap confidence intervals, a paired permutation test and/or
Wilcoxon signed-rank test for repeated per-image comparisons between the two detectors,
McNemar's test only if applicable (paired binary correct/incorrect on matched detections).

If running the same test repeatedly across the Batch 5 corruption/severity grid, apply a
multiple-comparison correction (Holm-Bonferroni or similar) rather than reporting uncorrected
p-values across the board.

Report p-values, confidence intervals, and effect sizes (Cohen's d or rank-biserial) for
every comparison — not point estimates alone.

When done: update CODEX.md and append a HANDOFF.md entry. Stop for my review before Batch 8.
```

---

## BATCH 8 — Report Assembly and Wrap-Up

```
Read HANDOFF.md and CODEX.md first.

Follow PROJECT_SPEC.md §8 (12-section report structure) and §5 Phases 9–12. Write
report/report.md, pulling from the tables/figures already produced in results/ — don't
recompute anything here, cite what already exists.

The Discussion section must argue which detection paradigm suits which deployment scenario
and why, grounded in the accuracy/robustness/interpretability/compute trade-offs actually
measured — not just which had the higher mAP. Include the deployment/regulatory scope note
from §5 Phases 9-12 (clinical deployment would need regulatory validation beyond this
benchmark — a scope-of-claims statement, not something to implement).

Consolidate docs/LIMITATIONS.md — it should already have entries from Batches 1, 3, and 5;
merge and review them into one coherent section rather than overwriting.

Write README.md with the exact commands needed to reproduce every table and figure in the
report from a clean checkout.

Go through PROJECT_SPEC.md §9 (Definition of Done) explicitly, item by item, and report
status against each one.

When done: update CODEX.md to reflect project completion, and write a final HANDOFF.md entry.
```
