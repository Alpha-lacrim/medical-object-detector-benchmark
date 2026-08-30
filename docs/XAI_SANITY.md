# Grad-CAM Sensitivity Controls (v2)

## Scope and correction

Batch 31 corrects the terminology and extends the method used by the historical
Batch 21 Grad-CAM audit. Both analyses use the trained seed-17 Faster R-CNN and
YOLO11s checkpoints and the same nested 50-image/41-patient subset of the frozen
300-image robustness pool. The v2 subset manifest is byte-identical to the
historical manifest (SHA-256 `55c76d9f80bfbc450df755cd617145559cba37bb85e484cdc4044f2cedff6e03`).
No training or checkpoint update occurs.

Two historical labels require explicit correction:

- Batch 21's `data_randomization` condition permuted RGB pixel vectors within
  each test image at inference. It is an **input-pixel randomization control**,
  not the Adebayo et al. training-label data-randomization test
  [@adebayo2018sanity].
- Batch 21's one-shot reinitialization of every weight is a **full
  model-parameter randomization control**. It is not a reproduction of the
  paper's cascading analysis.

The canonical Adebayo data-randomization test was **not performed**. For this
detector study it would require training identical architectures on randomized
training annotations or data-label relationships and verifying that the
randomized tasks were actually fit. Batch 31 was not authorized to add that
retraining, so neither the historical nor v2 result establishes sensitivity to
the learned training data-label relationship.

The historical table, detail data, panel, and Phase 21 summary remain unchanged
for audit continuity:

| Historical artifact | SHA-256 |
|---|---|
| `results/tables/gradcam_sanity_summary.csv` | `d753cd479b5585749eccce26debc3b46e4c014eec3d30f7b5f5ef5818bbe5f06` |
| `results/tables/gradcam_sanity_per_image.csv` | `3c2ffef0997041f5d5100f5694389ff319d8ed246726bb3947ded652dd552da5` |
| `results/figures/gradcam_sanity_panel.png` | `ce32473824f59863deb4fe52febcd1053dfbfcc4bfbb1eb850efb6bce98e6256` |
| `results/logs/phase21_xai_sanity/summary.json` | `70d99e923ece3b338003e1b947abd976fe544ba3bcd9d3272b034d76dca86198` |

The `data_randomization` strings inside those frozen files are legacy labels
for the input-pixel control and must not be interpreted as randomized-label
training.

## Fixed target and input control

For every study, the trained detector's highest-score retained candidate on the
unmodified radiograph defines one reference region without using a ground-truth
box. Every condition differentiates the pre-activation foreground score at that
fixed region:

- Faster R-CNN rescores the fixed ROI through its ROI classifier.
- YOLO11s uses the raw anchor whose decoded center is closest to the fixed
  region center.

Ordinary ReLU Grad-CAM is extracted from the same stride-16, 40 by 40 layers as
Phase 7: Faster R-CNN `backbone.body.layer3` and YOLO11s `model.6`. This
fixed-region target avoids requiring randomized detector heads to emit valid
post-NMS boxes, but differs from Phase 7's ground-truth-associated,
post-activation localization target.

The input-pixel randomization control deterministically permutes whole RGB
pixel vectors without replacement over spatial positions. The pixel-vector
multiset is preserved, spatial anatomy is destroyed, both trained detectors
receive identical perturbed images, and model weights remain unchanged. This
is a severe input-perturbation stress control only.

## Cascading model-parameter randomization

Each cumulative stage starts from a fresh CPU deep copy of the trained in-memory
model. A fixed CPU `torch.Generator` is reset to seed 17 at every stage;
selected weights receive Xavier-normal values, selected biases are zeroed, and
non-weight/non-bias buffers are preserved. Stages proceed from detector outputs
toward the input. Runtime audits require every weight/bias-bearing module to
belong to exactly one group.

| Stage | Faster R-CNN group | Module prefixes | YOLO11s group | Module prefixes |
|---:|---|---|---|---|
| 1 | Detector output heads | `rpn.head`, `roi_heads.box_predictor` | Detect output head | `model.23` |
| 2 | ROI and pyramid heads | `roi_heads.box_head`, `backbone.fpn` | Deep PAN neck | `model.17`, `19`, `20`, `22` |
| 3 | Backbone C5 | `backbone.body.layer4` | Shallow PAN neck | `model.13`, `16` |
| 4 | Backbone C4/CAM stage | `backbone.body.layer3` | Deep backbone | `model.7`--`10` |
| 5 | Backbone C3 | `backbone.body.layer2` | Backbone/CAM stage | `model.5`, `6` |
| 6 | Backbone C2 and stem | `backbone.body.layer1`, `conv1`, `bn1` | Early backbone and stem | `model.0`--`4` |

Stage $j$ randomizes all groups from 1 through $j$. Stage 6 covers all
declared weights and biases and is therefore also the v2 full
model-parameter randomization control. Partition audits cover 137 Faster R-CNN
and 169 YOLO11s weight/bias-bearing modules.

The on-disk checkpoints are never written. Their hashes were identical before
and after the complete 700-map run:

- Faster R-CNN:
  `9ec35c5d761f8e4bf7a43f7999f388ac1ffc0d533f62746409db280706dffab4`;
- YOLO11s:
  `65909164e82c1ef53c0d38e0d898d37bbbec5f46cb9f5cd029e76ba486c0371c`.

## Similarity metrics and invalid maps

Each full-resolution map is bilinearly reduced to a 40 by 40 evaluation grid,
matching the source CAM resolution, and independently min-max normalized to
$[0,1]$. The same normalized pair supplies:

- Pearson correlation for continuity with the historical analysis;
- tie-aware Spearman rank correlation using average ranks; and
- Gaussian-window SSIM with data range 1, sigma 1.5, $K_1=0.01$, and
  $K_2=0.03$.

If either map is non-finite or has range at most $10^{-12}$ after
preprocessing, the pair is excluded from all three metrics and counted once.
Metrics are never imputed. Pearson and Spearman measure value/rank association;
SSIM retains local spatial structure and can therefore disagree with the two
correlations, especially for sparse nonnegative maps.

## Results

The table reports descriptive means over the valid map pairs. There are no
inferential intervals or hypothesis tests.

| Detector | Control / newly included group | Valid K / 50 | Pearson | Spearman | SSIM |
|---|---|---:|---:|---:|---:|
| Faster R-CNN | Input-pixel control | 43 | -0.0206 | -0.0188 | 0.2468 |
| Faster R-CNN | 1: detector output heads | 50 | 0.0644 | 0.0512 | 0.1222 |
| Faster R-CNN | 2: ROI and pyramid heads | 50 | 0.0419 | 0.0703 | 0.0676 |
| Faster R-CNN | 3: backbone layer 4 | 50 | 0.0432 | 0.0794 | 0.0835 |
| Faster R-CNN | 4: backbone layer 3 | 50 | -0.0385 | -0.0189 | 0.0756 |
| Faster R-CNN | 5: backbone layer 2 | 50 | 0.0282 | 0.0452 | 0.0930 |
| Faster R-CNN | 6: early backbone; full model | 50 | 0.0025 | 0.0287 | 0.0381 |
| YOLO11s | Input-pixel control | 50 | -0.0021 | 0.0133 | 0.0067 |
| YOLO11s | 1: detect output head | 50 | 0.0893 | 0.0886 | 0.0935 |
| YOLO11s | 2: deep PAN neck | 50 | 0.1810 | 0.1858 | 0.1643 |
| YOLO11s | 3: shallow PAN neck | 50 | 0.1725 | 0.2020 | 0.1581 |
| YOLO11s | 4: deep backbone | 50 | -0.0309 | -0.0397 | 0.0216 |
| YOLO11s | 5: backbone/CAM stage | 50 | 0.0750 | 0.1122 | 0.0537 |
| YOLO11s | 6: early backbone; full model | 46 | 0.0242 | 0.0795 | 0.0476 |

The head-to-backbone curves are not monotonic, which is unsurprising for one
random draw per cumulative stage and a fixed nonlinear target. The final
full-model stages nevertheless have low mean similarity on all three metrics.
This supports the narrow conclusion that the reported Grad-CAM construction is
sensitive to model parameters under this audit. It does not prove anatomical
correctness, causal faithfulness, clinical reasoning, or medical validity.

The input-pixel control produces near-zero mean Pearson and Spearman values for
both detectors. Faster R-CNN retains some local structural similarity
(mean SSIM 0.2468), illustrating why one correlation alone was insufficient.
These results show response to a severe spatial input perturbation; they do not
test or establish dependence on the learned data-label relationship.

## Localization audit and claim boundary

The original localization values remain descriptive. Faster R-CNN has mean
energy-in-box 0.0869 versus a 0.0713 box-area reference (absolute lift 0.0156)
and pointing accuracy 0.1091. YOLO11s has mean energy 0.0975 versus area 0.0718
(lift 0.0257) and pointing accuracy 0.1261. No inferential analysis tests these
small absolute differences. They are weak spatial overlaps, not strong
localization, even though their means slightly exceed the box-area references.

Accordingly, the v2 controls do not upgrade the Phase 7 interpretation.
Grad-CAM remains a coarse, target- and layer-dependent failure-analysis
association map. It does not explain proposal generation, NMS, box regression,
or the complete detector decision.

## Versioned artifacts and reproduction

```powershell
$benchmarkPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $benchmarkPython -m pytest tests/test_xai_sanity.py tests/test_gradcam.py tests/test_explainability.py -q
& $benchmarkPython -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode preflight
& 'C:\Users\Pouyan\.conda\envs\torch-gpu\python.exe' -m src.explainability.sanity_checks --config configs/xai_sanity.yaml --mode run
```

The v2 outputs are:

- `results/tables/gradcam_sanity_v2_summary.csv`
  (`23a8b47c65ba2d4d076262f71f3ec1f5c1aec9a6715ee1a67ee0ceaed3dc6f51`);
- `results/tables/gradcam_sanity_v2_per_image.csv`
  (`4e765237372318118cbd66041afb24413e0fe99cd859e58b3697b7ad2fb8074c`);
- `results/figures/gradcam_sanity_v2_panel.png`
  (`7eac38050cdb8f27932c24f150f5f0d8141e17105b81d25dd5b7968ff63e2204`); and
- `results/logs/phase31_xai_sanity_v2/summary.json`
  (`88d386c215bf483cea66ef02c3b2b4e90ad548f54a1a8f6fe4511da153463e176`).

The Phase 31 summary records the config/source identities, complete layer-group
partition, per-stage initialization audit, checkpoint before/after hashes,
historical artifact hashes, map-level records, and v2 artifact hashes.
