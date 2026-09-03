# YOLO11s Baseline Protocol

## Scope

Batch 3 replaces only the detector with Ultralytics YOLO11s. It uses the same
official patient-safe 3,500/750 train/validation studies, one foreground class,
seed 17, 640-pixel input, RTX 4060 Laptop GPU, and Python 3.11/Torch 2.6/CUDA
12.4 environment as Faster R-CNN. The held-out test annotation is not opened in
this batch. The implementation is pinned to `ultralytics==8.4.110`, and the
downloaded `yolo11s.pt` hash is recorded before training.

The official Ultralytics v8.4.0 release asset was resolved by the pinned
Ultralytics downloader to `results/checkpoints/pretrained/yolo11s.pt` before
any Batch 3 training. Its SHA-256 is
`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`;
the preflight and every run summary bind to the same file identity.

## Augmentation decision made before training

Ultralytics' extra stochastic augmentations are disabled. Mosaic, mixup,
cutmix, copy-paste, HSV jitter, horizontal/vertical flips, affine and
perspective transforms, erasing, auto-augmentation, and multi-scale training
are all explicitly set to identity/off in `configs/yolo.yaml`. Faster R-CNN
used deterministic resize/normalization only, so leaving the YOLO defaults on
would change both the detector and data distribution. The controlled setting
is preferred even though it may reduce YOLO performance relative to its normal
augmentation-rich recipe.

## Matched optimization and hardware policy

YOLO11s uses physical/effective batch 4 (`batch=4`, `nbs=4`), bfloat16 AMP, SGD
with target LR 0.001, momentum 0.9, weight decay 0.0005, and no Nesterov.
The valid run uses a one-epoch linear learning-rate warmup from zero while
keeping momentum fixed at 0.9. A pre-training full-data diagnostic with no
warmup repeatedly produced non-finite classification loss at epoch 1, batch
29 as AMP's scale collapsed; the warmup is therefore a minimal numerical-
stability exception to the otherwise matched optimizer policy. A subsequent
0.005-target diagnostic remained finite but collapsed to all-zero losses and
near-zero scores after epoch 1, so the valid run lowers YOLO's target LR to
0.001. A 0.001 diagnostic still collapsed because Ultralytics performed task-
aligned target assignment on float16 scores: once the one-class logits became
negative enough, head scores underflowed before the float32 loss received
them. The valid mixed-precision policy therefore uses bfloat16 autocast for the
model forward/backward (supported natively by the RTX 4060) and computes target
assignment and detector losses in float32. Ordinary seeded shuffle is retained.
These failed attempts are not used for timing or performance results.
Their defining signature was optimizer/head failure--non-finite or all-zero losses with
near-zero output scores--and should not be conflated with the later seed-271 outcome
described below.
Ultralytics' bundled AMP equivalence probe is specific to its default float16
autocast and rejects bfloat16 on its output-tolerance comparison. The custom
trainer replaces that probe with an explicit CUDA bfloat16 capability/dtype
gate; the smoke test and permanent per-batch non-finite/zero-loss guards supply
the end-to-end numerical check, and the run aborts if Ultralytics disables AMP.
YOLO BatchNorm statistics remain trainable. Forced freezing produced stable
bfloat16 arithmetic but still drove the one-class head into a zero-score state
by epoch 3; unlike Faster R-CNN's frozen-normalization backbone, YOLO relies on
its native BatchNorm adaptation when transferring from COCO to radiographs.
This architecture-specific normalization difference is documented rather than
retaining a matched setting that prevents learning. Validation mAP50:95
drives early stopping with minimum 8 epochs, improvement delta 0.001, patience
5, and maximum 30. Ultralytics does not expose the Faster R-CNN
ReduceLROnPlateau scheduler through its standard trainer contract, so YOLO uses
a constant target LR after warmup (`lrf=1.0`); this unavoidable optimizer-
schedule difference is recorded rather than silently approximated with a
different decay schedule.

Two DataLoader workers are used because the earlier Windows run demonstrated a
16 GB commit-limit failure when multiple six-worker PyTorch pools coexisted.
Decoded-image RAM caching is disabled. A complete three-epoch real-data timing
benchmark is run before the full seed.

## Evaluation and artifacts

Ultralytics' per-epoch metrics drive training and checkpoint selection. Final
validation predictions are also converted to canonical records and passed to
the repository's shared COCO/operating-point functions; Batch 4 will build the
formal two-model `src/evaluate.py` comparison harness. Profiling records
synchronized batch-1 model-plus-NMS FPS/latency, parameters, mandatory finite
GFLOPs, peak allocated training memory, checkpoint size/hash, and training
time. Tables and curves are written under `results/tables/` and
`results/figures/` from documented commands in `README.md`.

## Measured seed-17 result

The accepted three-epoch timing benchmark completed in 141.79, 134.99, and
136.20 seconds. Its 135.59-second steady-state estimate projected 18.18 minutes
for the eight-epoch floor and 67.90 minutes for the 30-epoch ceiling. The full
run restarted from the pinned pretrained weights and stopped at epoch 15 after
five validation mAP50:95 non-improvements following the epoch-10 best. The
logged epoch loop took 1,975.64 seconds (32.93 minutes), with 1,148.16 MiB peak
allocated GPU memory. Ultralytics' native epoch-10 mAP50:95, used only for
checkpointing and early stopping, was 0.07335.

The final best checkpoint was evaluated on all 750 validation images and 277
boxes through the shared evaluator. It produced AP50 0.26464 and AP50:95
0.08692. At the fixed score threshold 0.25 and match IoU 0.50, precision was
0.57143, recall 0.20217, and F1 0.29867 (56 TP, 42 FP, 221 FN). The held-out
test split remained untouched.

Synchronized batch-1 bfloat16 profiling over 100 timed images measured 65.24
FPS with mean/p50/p95 latency 15.33/14.49/19.82 ms. YOLO11s has 9,428,179
parameters (9,428,163 trainable during training), 21.42 estimated GFLOPs under
the shared registered-operation convention, and a 19,172,819-byte (18.28 MiB)
best checkpoint. Its SHA-256 is
`65909164e82c1ef53c0d38e0d898d37bbbec5f46cb9f5cd029e76ba486c0371c`.

Post-training finalization initially exposed two reporting defects without
changing the trained weights: a list-source inference call materialized the
entire validation set and exceeded VRAM, and stripped Ultralytics checkpoints
store epoch `-1`. Finalization now streams the audited validation directory in
bounded batches, verifies all 750 filenames, and derives the best epoch from
the immutable `results.csv`. The summary records the benchmark-approved
training source identity separately from the corrected reporting-source
identity.

## Research-track seed expansion and seed-271 score-scale analysis

Batch 16 later repeated the frozen, augmentation-disabled training recipe for seeds 271 and
314, in addition to the accepted seeds 17, 42, and 137. This produced five completed YOLO11s
checkpoints for the clean all-attempt comparison; it did not change the Batch 3 seed-17
protocol or retroactively alter the historical diagnostics above.

Seed 271 did **not** reproduce the historical all-zero-loss/head collapse. Its train
box/classification/DFL losses decreased from 1.5827/4.8243/1.6195 to
0.9683/1.0934/0.9996, validation mAP@0.5:0.95 was nonzero and peaked at 0.08958 at epoch 7,
and the run completed 12 epochs through normal validation-map early stopping. Its low-score
test predictions retain plausible ranking and localization (AP@0.5 0.1587 and
AP@0.5:0.95 0.0556), but their maximum confidence is only 0.0412735. Consequently seed 271
emits zero test detections at both the fixed score threshold 0.25 and the frozen n=3-selected
YOLO threshold 0.05.

The retained failure classification is therefore a seed-specific **confidence-score/output-scale
degeneracy after otherwise normal convergence**, not the earlier numerical loss/head
collapse. The clean all-attempt analysis retains seed 271's valid AP and zero
fixed-threshold precision/recall/F1 outcomes. Matched-only IoU and Dice are undefined for
that seed because it has no fixed-threshold true positive; their descriptive YOLO summary
uses the four defined seeds and their paired inference uses the four complete detector seed
pairs. The validation threshold remains the frozen n=3 selection. Batch 35 retains the
original n=3 threshold, FROC, precision-recall, and Pareto artifacts as provenance and adds
separately labeled n=5 frozen-bundle sensitivities in which seed 271 is included exactly as
observed rather than filtered for its zero detections at 0.25 or 0.05.
