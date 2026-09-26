# YOLO numerical inference-path audit (Batch 53, v1)

2026-09-26. Secondary sensitivity only; historical internal FP32 and frozen
external evidence retain their status. Canonical manuscript:
`report/paper_draft.md` (D-010/D-022). Starting branch `main`, HEAD
`ac6975df799a20c55d97204ddcdcbd0dab6675f9`, empty index. Existing Batch 52 and
author-declaration edits are preserved; full starting status is retained locally
in `tmp/batch53/starting_state.json`. No Git publication action is authorized.

## Prerequisite evidence

- All five YOLO checkpoints (17, 42, 137, **271**, 314) exist and match
  `results/checkpoint_release_manifest.json` SHA-256 values.
- Both sets of five historical bundles match their Phase 5 / Phase 42 v4
  summary hashes and checkpoint identities. Each contains 750 images. Filtering
  each v4 bundle to the historical AP floor gives **exactly identical** boxes,
  scores, labels and counts to its Phase 5 bundle; support need not be invented.
- Canonical test annotations match the original SHA-256
  `86bbe6c238d651bfc0b017a447a67548aa631f95855d66b9b8b71a13807502cc`.
  The test manifest identity is bound by `configs/cohort_characteristics.yaml`;
  the new protocol additionally verifies its rows against the canonical loader.
- All 83 publication artifacts, 580 present inputs and 201 result references
  pass the scientific verifier; all 38 original external freeze bindings pass
  the archive-aware verifier. All five external YOLO prediction, FROC and metric
  files match their existing summary hashes. Each records an actual
  `torch.bfloat16` convolution output.
- Exact-score FROC is `src/analyze_exact_score_froc.py` with
  `configs/froc_exact_score_v4.yaml`. Historical n=3 and post-hoc n=5 YOLO
  cutoffs are 0.05 and 0.01, respectively; their existing selection summaries
  and tables are bound. No selection is performed here.
- Searches of code/configuration/documentation found no completed five-run
  internal precision harmonisation. Batch 45 checks timing-path parity under
  the same AMP; it does not answer FP32-versus-bfloat16 accuracy sensitivity.
- The local RTX 4060 Laptop supports bfloat16. Torch 2.6.0+cu124,
  Torchvision 0.21.0+cu124, Ultralytics 8.4.110, CUDA 12.4 and cuDNN 90100
  match the recorded scientific environment. No GPU prediction was needed for
  this forensic comparison; checkpoint/default inspection used CPU loading.

## Path comparison

| Component | Historical internal | Frozen external | Sensitivity design |
|---|---|---|---|
| Entry | `src.evaluate._collect_yolo_predictions`; Phase 42 calls the same helper | `src.evaluate_vindr_external` uses `src.benchmark_inference.Predictor` | Reuse that existing Predictor without editing it |
| Population/loader | `CocoDetectionDataset`, canonical RSNA PNG paths, one image per native call | Same metadata loader, prepared VinDr PNG paths | Immutable RSNA records, order and PNG paths |
| Decode/preprocess | Native `LoadImagesAndVideos`: OpenCV BGR; BGR-to-RGB, CHW, contiguous tensor, FP32 /255 | Same native file-source route (`rgb=None`) | Same internal files and native route; no decoded-array adapter |
| Resize | Native `LetterBox`, `YOLO.predict` rect=True, imgsz=640, stride-aware aspect ratio | Same; native VinDr dimensions differ | Same 1024-square RSNA source geometry and 640 input |
| Native precision | `amp=True` argument alone does not activate prediction autocast; quantize defaults to None/FP32 | Explicit CUDA bfloat16 autocast around native prediction; quantize=None | Copy explicit external numerical path and verify active convolution dtype |
| Backend precision | Historical seed helper does not set TF32 flags; fresh pinned runtime has matmul=False, cuDNN=True; historical effective flags were not logged | Explicit matmul=False, cuDNN=False | Match external flags; this is numerical-path sensitivity, not an isolated activation-dtype causal effect |
| Fusion/model state | Native Conv/BatchNorm fusion, eval, FP32 weights | Same fusion/API, eval, FP32 weights; initial native setup occurs inside outer CUDA autocast | Preserve initialization order; CPU-loaded model is fused on CPU before device transfer in this pinned backend |
| Forward/gradients | Native inference-mode prediction, no augmentation | Outer and native inference mode, no augmentation | Inference only, explicit training-call guard and module-state checks |
| NMS | Native class-aware NMS, IoU=0.5, max_det=100 | Identical | Identical; no second NMS |
| Candidate support | Phase 5 native strict >0.001; Phase 42 v4 native strict >0.00001 | Native strict >0.00001; common AP filter >=0.001 | Collect >0.00001, use unchanged AP filter; compare historical AP and FROC at their respective supports |
| Score comparisons | Common evaluator >= threshold, same-class score-ordered matching | Same | Preserve; never reselect 0.05/0.01 or the shared 0.25 cutoff |
| Coordinates | Native `scale_boxes` back to source, source-bound clipping, float64 serialization | Same YOLO path; documented one-ULP repair permits Faster R-CNN only | No YOLO repair or alternative restoration |
| Determinism | Deterministic algorithms, warning-only unsupported operations; backend defaults partly implicit | Deterministic algorithms, fail on unsupported operations; one Torch thread | External numerical controls; fail if unsupported; record all effective flags |
| Evaluation | Common COCO AP and exact-score FROC implementation | Same implementations and frozen definitions | Reuse evaluators; equal-run summaries and same-checkpoint differences |

The audit identifies no material non-numerical adapter change requiring a new
design. TF32 policy is part of the disclosed numerical-path intervention and
prevents describing the experiment as a pure bfloat16 activation-only ablation.
Historical FP32 denotes tensor/activation precision, not proof that every CUDA
kernel used full-mantissa arithmetic. Datasets, sites, populations, annotation
standards and preprocessing effects remain confounded even if this sensitivity
is small. No external results, primary estimates or checkpoint bytes change.

Installed source evidence: `ultralytics/engine/model.py` (`predict`),
`engine/predictor.py` (`preprocess`, `pre_transform`, `setup_model`),
`nn/backends/pytorch.py` (`load_model`), `data/loaders.py`,
`models/yolo/detect/predict.py`, and `utils/nms.py`. Exact installed source hashes
are captured with the new execution environment rather than inferred from the
version string alone.
