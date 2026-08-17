"""Run paired stride-16 Grad-CAM and localization analysis on the Phase 6 sample."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.evaluate import load_and_validate_training_configs, load_phase5_config, sha256_file
from src.explainability.config import (
    DetectorName,
    ExplainabilityConfig,
    load_explainability_config,
)
from src.explainability.gradcam import (
    ActivationCapture,
    gradcam_for_score,
    resolve_module,
    restore_letterboxed_heatmap,
)
from src.explainability.pointing_game import evaluate_box_attention
from src.explainability.selection import operating_evidence, select_qualitative_cases
from src.meddet_benchmark.evaluation import ImagePrediction, ImageTarget, box_iou
from src.robustness.run_robustness import CorruptedSubsetDataset, load_robustness_config

DETECTORS: tuple[DetectorName, DetectorName] = ("faster_rcnn", "yolo11s")
TARGET_FIELDS = (
    "detector",
    "image_id",
    "target_index",
    "category_id",
    "image_height",
    "image_width",
    "ground_truth_x1",
    "ground_truth_y1",
    "ground_truth_x2",
    "ground_truth_y2",
    "operating_status",
    "operating_prediction_score",
    "operating_prediction_iou",
    "cam_target_role",
    "cam_candidate_score",
    "cam_candidate_iou",
    "cam_candidate_above_operating_threshold",
    "candidate_x1",
    "candidate_y1",
    "candidate_x2",
    "candidate_y2",
    "target_layer",
    "target_stride",
    "feature_height",
    "feature_width",
    "valid_cam",
    "total_energy",
    "energy_in_box",
    "pointing_hit",
    "peak_x",
    "peak_y",
    "box_pixel_fraction",
    "energy_lift_over_area",
)


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_bytes(path, raw)


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"gzip JSON artifact must contain an object: {path}")
    return payload


def _deserialize_prediction(payload: Mapping[str, Any]) -> ImagePrediction:
    return ImagePrediction(
        image_id=str(payload["image_id"]),
        image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        boxes_xyxy=payload["boxes_xyxy"],
        labels=payload["labels"],
        scores=payload["scores"],
    )


def _targets_from_dataset(dataset: CorruptedSubsetDataset) -> dict[str, ImageTarget]:
    return {
        record.file_name: ImageTarget(
            image_id=record.file_name,
            image_size=(record.height, record.width),
            boxes_xyxy=[annotation.bbox_xyxy for annotation in record.annotations],
            labels=[annotation.category_id for annotation in record.annotations],
        )
        for record in dataset.records
    }


def _load_sample_names(
    config: ExplainabilityConfig,
    phase6: Any,
    phase6_summary: Mapping[str, Any],
) -> set[str]:
    manifest = phase6.resolve(phase6.sampling.output_manifest)
    sampling_summary = phase6_summary.get("sampling", {})
    if sampling_summary.get("output_manifest_sha256") != sha256_file(manifest):
        raise ValueError("Phase 6 sample manifest hash disagrees with its completed summary")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {phase6.sampling.id_column, phase6.sampling.split_column}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Phase 6 sample manifest lacks configured identity columns")
    if len(rows) != phase6.sampling.size or any(
        row[phase6.sampling.split_column] != phase6.sampling.split_value for row in rows
    ):
        raise ValueError("Phase 6 sample manifest size or split identity changed")
    names = {row[phase6.sampling.id_column] for row in rows}
    if len(names) != len(rows):
        raise ValueError("Phase 6 sample manifest image identities are not unique")
    if config.seed != int(sampling_summary.get("seed", -1)):
        raise ValueError("Phase 7 seed differs from the frozen Phase 6 sample seed")
    return names


def _load_clean_predictions(
    config: ExplainabilityConfig,
    phase6: Any,
    detector: DetectorName,
    selected_names: set[str],
    sample_sha256: str,
) -> dict[str, ImagePrediction]:
    path = phase6.resolve(phase6.outputs.prediction_bundles_dir) / f"{detector}__clean.json.gz"
    payload = _read_gzip_json(path)
    identity = payload.get("identity", {})
    detector_config = next(item for item in phase6.detectors if item.detector == detector)
    if (
        payload.get("status") != "complete"
        or identity.get("detector") != detector
        or identity.get("seed") != config.seed
        or identity.get("sample_manifest_sha256") != sample_sha256
        or identity.get("checkpoint_sha256")
        != sha256_file(phase6.resolve(detector_config.checkpoint))
        or identity.get("condition", {}).get("condition_id") != "clean"
    ):
        raise ValueError(f"incompatible frozen clean prediction bundle: {path}")
    predictions = {
        item.image_id: item
        for item in (_deserialize_prediction(raw) for raw in payload.get("predictions", []))
    }
    if set(predictions) != selected_names:
        raise ValueError(f"clean prediction bundle does not match Phase 6 sample: {path}")
    return predictions


def _prepare(config: ExplainabilityConfig) -> dict[str, Any]:
    phase6 = load_robustness_config(config.resolve(config.phase6_config))
    summary_path = config.resolve(config.phase6_summary)
    phase6_summary = _read_json(summary_path)
    if (
        phase6_summary.get("status") != "complete"
        or phase6_summary.get("config_sha256") != sha256_file(phase6.source_path)
        or phase6_summary.get("seed_scope", {}).get("training_seed") != config.seed
    ):
        raise ValueError("Phase 6 summary is incomplete or incompatible with Phase 7")
    if phase6.seed != config.seed:
        raise ValueError("Phase 7 must use the Phase 6 primary-seed detector pair")

    phase5 = load_phase5_config(phase6.resolve(phase6.phase5_evaluation_config))
    training_configs = load_and_validate_training_configs(phase5)
    selected_names = _load_sample_names(config, phase6, phase6_summary)
    sample_path = phase6.resolve(phase6.sampling.output_manifest)
    sample_sha256 = sha256_file(sample_path)

    from src.models.faster_rcnn_data import CocoDetectionDataset

    faster_config = training_configs[("faster_rcnn", config.seed)]
    base_dataset = CocoDetectionDataset(
        faster_config.resolve(faster_config.data.dataset_config),
        "test",
        mode="full",
        project_root=config.project_root,
    )
    subset = CorruptedSubsetDataset(
        base_dataset,
        selected_names,
        condition=None,
        seed=config.seed,
    )
    targets = _targets_from_dataset(subset)
    positive_images = sum(bool(len(item.boxes_xyxy)) for item in targets.values())
    target_count = sum(len(item.boxes_xyxy) for item in targets.values())
    sampling_summary = phase6_summary["sampling"]
    if (
        len(targets) != sampling_summary["sample_image_count"]
        or positive_images != sampling_summary["sample_positive_images"]
        or target_count != sampling_summary["sample_box_count"]
    ):
        raise ValueError("canonical annotations no longer match the Phase 6 sample audit")
    predictions = {
        detector: _load_clean_predictions(config, phase6, detector, selected_names, sample_sha256)
        for detector in DETECTORS
    }
    cases = select_qualitative_cases(
        predictions,
        targets,
        score_threshold=phase6.evaluation.score_threshold,
        iou_threshold=phase6.evaluation.match_iou_threshold,
        max_detections=phase6.evaluation.max_detections,
        cases_per_category=config.qualitative.cases_per_category,
        failure_quantiles=config.qualitative.failure_quantiles,
    )
    evidence = {
        detector: {
            image_id: operating_evidence(
                predictions[detector][image_id],
                target,
                score_threshold=phase6.evaluation.score_threshold,
                iou_threshold=phase6.evaluation.match_iou_threshold,
                max_detections=phase6.evaluation.max_detections,
            )
            for image_id, target in targets.items()
        }
        for detector in DETECTORS
    }
    return {
        "phase6": phase6,
        "phase6_summary": phase6_summary,
        "phase6_summary_path": summary_path,
        "training_configs": training_configs,
        "subset": subset,
        "targets": targets,
        "predictions": predictions,
        "evidence": evidence,
        "cases": cases,
        "sample_path": sample_path,
        "sample_sha256": sample_sha256,
    }


def preflight(config: ExplainabilityConfig) -> dict[str, Any]:
    """Validate frozen provenance, paired cases, and quantitative population."""

    prepared = _prepare(config)
    targets = prepared["targets"]
    return {
        "status": "ready",
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "sample_manifest": prepared["sample_path"].as_posix(),
        "sample_manifest_sha256": prepared["sample_sha256"],
        "sample_image_count": len(targets),
        "positive_image_count": sum(bool(len(item.boxes_xyxy)) for item in targets.values()),
        "ground_truth_box_count": sum(len(item.boxes_xyxy) for item in targets.values()),
        "target_layers": [item.model_dump(mode="json") for item in config.detectors],
        "qualitative_cases": [
            {
                "category": item["category"],
                "rank": item["rank"],
                "image_id": item["image_id"],
                "target_index": item["target_index"],
                "selection_score": item["selection_score"],
            }
            for item in prepared["cases"]
        ],
    }


def _best_candidate_index(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    reference_box: np.ndarray,
) -> tuple[int, float]:
    if len(boxes_xyxy) == 0:
        raise ValueError("detector emitted no positive-score candidate for Grad-CAM")
    overlaps = box_iou(boxes_xyxy, reference_box.reshape(1, 4))[:, 0]
    best_iou = float(overlaps.max())
    tied = np.flatnonzero(np.isclose(overlaps, best_iou, rtol=0, atol=1e-12))
    best = int(tied[np.argmax(scores[tied])])
    return best, best_iou


def _validate_feature(
    activation: Any, config: ExplainabilityConfig, detector: DetectorName
) -> None:
    layer = config.layer(detector)
    if activation is None or tuple(activation.shape[-2:]) != (
        layer.expected_spatial_size,
        layer.expected_spatial_size,
    ):
        shape = None if activation is None else tuple(activation.shape)
        raise ValueError(f"{detector} target layer emitted unexpected shape: {shape}")


def _load_faster_model(
    config: ExplainabilityConfig,
    prepared: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    import torch

    from src.models.faster_rcnn_config import config_fingerprint
    from src.models.faster_rcnn_model import build_faster_rcnn

    phase6 = prepared["phase6"]
    detector_config = next(item for item in phase6.detectors if item.detector == "faster_rcnn")
    model_config = prepared["training_configs"][("faster_rcnn", config.seed)]
    checkpoint = torch.load(
        phase6.resolve(detector_config.checkpoint), map_location="cpu", weights_only=False
    )
    if checkpoint.get("config_sha256") != config_fingerprint(model_config):
        raise ValueError("Faster R-CNN checkpoint/config mismatch")
    model, _metadata = build_faster_rcnn(
        prepared["subset"].num_foreground_classes,
        model_config.model,
        use_pretrained_weights=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(config.runtime.device)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del checkpoint
    return model, device, model_config


def _faster_forward(
    model: Any,
    device: Any,
    model_config: Any,
    capture: ActivationCapture,
    image: Any,
    config: ExplainabilityConfig,
) -> tuple[Any, np.ndarray, Any, tuple[int, int]]:
    import torch

    device_image = image.to(device).requires_grad_(True)
    capture.clear()
    with torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=model_config.runtime.amp
    ):
        output = model([device_image])[0]
    activation = capture.activation
    _validate_feature(activation, config, "faster_rcnn")
    scores = output["scores"][: config.gradcam.max_candidates]
    boxes = output["boxes"][: config.gradcam.max_candidates]
    keep = scores > config.gradcam.candidate_score_floor
    scores = scores[keep]
    boxes_numpy = boxes[keep].detach().float().cpu().numpy().astype(np.float64)
    if len(scores) == 0:
        raise ValueError("Faster R-CNN has no positive-score Grad-CAM candidate")
    return activation, boxes_numpy, scores, tuple(int(value) for value in image.shape[-2:])


def _load_yolo_model(
    config: ExplainabilityConfig,
    prepared: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    import torch
    from ultralytics import YOLO

    phase6 = prepared["phase6"]
    detector_config = next(item for item in phase6.detectors if item.detector == "yolo11s")
    model_config = prepared["training_configs"][("yolo11s", config.seed)]
    wrapper = YOLO(phase6.resolve(detector_config.checkpoint).as_posix())
    model = wrapper.model
    device = torch.device(f"cuda:{model_config.runtime.device}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device, model_config


def _yolo_forward(
    model: Any,
    device: Any,
    model_config: Any,
    capture: ActivationCapture,
    image: Image.Image,
    config: ExplainabilityConfig,
    class_count: int,
) -> tuple[Any, np.ndarray, Any, tuple[int, int]]:
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import nms, ops

    original = np.asarray(image.convert("RGB"))
    input_size = int(model_config.model.input_size)
    letterboxed = LetterBox(
        new_shape=(input_size, input_size),
        auto=False,
        scale_fill=False,
        scaleup=True,
        stride=config.layer("yolo11s").expected_stride * 2,
    )(image=original)
    tensor = torch.from_numpy(np.ascontiguousarray(letterboxed.transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(device).float().div_(255).requires_grad_(True)
    capture.clear()
    with torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=model_config.runtime.amp
    ):
        output = model(tensor)
    prediction = output[0] if isinstance(output, tuple) else output
    activation = capture.activation
    _validate_feature(activation, config, "yolo11s")
    detections = nms.non_max_suppression(
        prediction,
        conf_thres=config.gradcam.candidate_score_floor,
        iou_thres=config.gradcam.nms_iou_threshold,
        max_det=config.gradcam.max_candidates,
        nc=class_count,
        max_time_img=config.gradcam.yolo_nms_time_limit_seconds,
        max_nms=config.gradcam.yolo_max_nms_candidates,
    )[0]
    if len(detections) == 0:
        raise ValueError("YOLO11s has no positive-score Grad-CAM candidate")
    scaled = ops.scale_boxes(
        tuple(tensor.shape[-2:]),
        detections[:, :4].detach().float().clone(),
        original.shape[:2],
    )
    boxes_numpy = scaled.cpu().numpy().astype(np.float64)
    return activation, boxes_numpy, detections[:, 4], tuple(tensor.shape[-2:])


def _cam_numpy(
    score: Any,
    activation: Any,
    *,
    feature_output_size: tuple[int, int],
    original_size: tuple[int, int],
    detector: DetectorName,
    config: ExplainabilityConfig,
    retain_graph: bool,
) -> np.ndarray:
    heatmap = gradcam_for_score(
        score,
        activation,
        output_size=feature_output_size,
        interpolation_mode=config.gradcam.interpolation_mode,
        align_corners=config.gradcam.align_corners,
        epsilon=config.gradcam.epsilon,
        retain_graph=retain_graph,
    )
    if detector == "yolo11s":
        heatmap = restore_letterboxed_heatmap(
            heatmap,
            original_size=original_size,
            interpolation_mode=config.gradcam.interpolation_mode,
            align_corners=config.gradcam.align_corners,
            epsilon=config.gradcam.epsilon,
        )
    return heatmap.detach().float().cpu().numpy().astype(np.float32)


def _target_row(
    *,
    detector: DetectorName,
    image_id: str,
    target_index: int,
    target: ImageTarget,
    operating: Mapping[str, Any],
    candidate_index: int,
    candidate_iou: float,
    candidate_boxes: np.ndarray,
    candidate_scores: Any,
    activation: Any,
    heatmap: np.ndarray,
    config: ExplainabilityConfig,
    score_threshold: float,
) -> dict[str, Any]:
    box = target.boxes_xyxy[target_index]
    candidate_box = candidate_boxes[candidate_index]
    candidate_score = float(candidate_scores[candidate_index].detach().float().cpu())
    attention = evaluate_box_attention(
        heatmap,
        box,
        zero_energy_epsilon=config.gradcam.zero_energy_epsilon,
    )
    layer = config.layer(detector)
    values = attention.to_dict()
    return {
        "detector": detector,
        "image_id": image_id,
        "target_index": target_index,
        "category_id": int(target.labels[target_index]),
        "image_height": target.image_size[0],
        "image_width": target.image_size[1],
        "ground_truth_x1": float(box[0]),
        "ground_truth_y1": float(box[1]),
        "ground_truth_x2": float(box[2]),
        "ground_truth_y2": float(box[3]),
        "operating_status": operating["status"],
        "operating_prediction_score": operating["score"],
        "operating_prediction_iou": operating["iou"],
        "cam_target_role": (
            "emitted_true_positive"
            if operating["status"] == "true_positive"
            else "miss_proxy_candidate"
        ),
        "cam_candidate_score": candidate_score,
        "cam_candidate_iou": candidate_iou,
        "cam_candidate_above_operating_threshold": candidate_score >= score_threshold,
        "candidate_x1": float(candidate_box[0]),
        "candidate_y1": float(candidate_box[1]),
        "candidate_x2": float(candidate_box[2]),
        "candidate_y2": float(candidate_box[3]),
        "target_layer": layer.module_path,
        "target_stride": layer.expected_stride,
        "feature_height": int(activation.shape[-2]),
        "feature_width": int(activation.shape[-1]),
        "valid_cam": values["valid"],
        "total_energy": values["total_energy"],
        "energy_in_box": values["energy_in_box"],
        "pointing_hit": values["pointing_hit"],
        "peak_x": values["peak_x"],
        "peak_y": values["peak_y"],
        "box_pixel_fraction": values["box_pixel_fraction"],
        "energy_lift_over_area": values["energy_lift_over_area"],
    }


def _case_key(case: Mapping[str, Any]) -> tuple[str, int | None]:
    return str(case["image_id"]), case["target_index"]


def _run_detector(
    config: ExplainabilityConfig,
    prepared: Mapping[str, Any],
    detector: DetectorName,
    *,
    smoke: bool,
    case_maps: dict[tuple[str, int, DetectorName], dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    subset = prepared["subset"]
    targets: Mapping[str, ImageTarget] = prepared["targets"]
    phase6 = prepared["phase6"]
    indices = {record.file_name: index for index, record in enumerate(subset.records)}
    positive_names = sorted(
        image_id for image_id, target in targets.items() if len(target.boxes_xyxy)
    )
    if smoke:
        positive_names = positive_names[: config.runtime.smoke_positive_images]
    selected_cases = {
        _case_key(case): case for case in prepared["cases"] if case["category"] != "bad_prediction"
    }

    if detector == "faster_rcnn":
        model, device, model_config = _load_faster_model(config, prepared)
    else:
        model, device, model_config = _load_yolo_model(config, prepared)
    module = resolve_module(model, config.layer(detector).module_path)
    records: list[dict[str, Any]] = []
    with ActivationCapture(module) as capture:
        for image_number, image_id in enumerate(positive_names, start=1):
            index = indices[image_id]
            tensor_image, _unused_target = subset[index]
            pil_image = subset.load_pil(index)
            target = targets[image_id]
            if detector == "faster_rcnn":
                activation, candidate_boxes, candidate_scores, feature_output_size = (
                    _faster_forward(
                        model,
                        device,
                        model_config,
                        capture,
                        tensor_image,
                        config,
                    )
                )
            else:
                activation, candidate_boxes, candidate_scores, feature_output_size = _yolo_forward(
                    model,
                    device,
                    model_config,
                    capture,
                    pil_image,
                    config,
                    subset.num_foreground_classes,
                )
            score_values = candidate_scores.detach().float().cpu().numpy().astype(np.float64)
            selections = [
                _best_candidate_index(candidate_boxes, score_values, box)
                for box in target.boxes_xyxy
            ]
            for target_index, (candidate_index, candidate_iou) in enumerate(selections):
                heatmap = _cam_numpy(
                    candidate_scores[candidate_index],
                    activation,
                    feature_output_size=feature_output_size,
                    original_size=target.image_size,
                    detector=detector,
                    config=config,
                    retain_graph=target_index < len(selections) - 1,
                )
                operating = prepared["evidence"][detector][image_id]["targets"][target_index]
                row = _target_row(
                    detector=detector,
                    image_id=image_id,
                    target_index=target_index,
                    target=target,
                    operating=operating,
                    candidate_index=candidate_index,
                    candidate_iou=candidate_iou,
                    candidate_boxes=candidate_boxes,
                    candidate_scores=candidate_scores,
                    activation=activation,
                    heatmap=heatmap,
                    config=config,
                    score_threshold=phase6.evaluation.score_threshold,
                )
                records.append(row)
                case = selected_cases.get((image_id, target_index))
                if case is not None and not smoke:
                    case_maps[(case["category"], case["rank"], detector)] = {
                        "heatmap": heatmap,
                        "candidate_box": candidate_boxes[candidate_index].copy(),
                        "candidate_score": row["cam_candidate_score"],
                        "candidate_iou": candidate_iou,
                        "energy_in_box": row["energy_in_box"],
                        "pointing_hit": row["pointing_hit"],
                    }
            capture.clear()
            del activation, candidate_scores, tensor_image
            if image_number % config.runtime.progress_every_images == 0:
                print(
                    f"[{detector}] explained {image_number}/{len(positive_names)} positive images",
                    flush=True,
                )

        if not smoke:
            bad_cases = [case for case in prepared["cases"] if case["category"] == "bad_prediction"]
            for case in bad_cases:
                image_id = str(case["image_id"])
                index = indices[image_id]
                tensor_image, _unused_target = subset[index]
                pil_image = subset.load_pil(index)
                if detector == "faster_rcnn":
                    activation, candidate_boxes, candidate_scores, feature_output_size = (
                        _faster_forward(
                            model,
                            device,
                            model_config,
                            capture,
                            tensor_image,
                            config,
                        )
                    )
                else:
                    activation, candidate_boxes, candidate_scores, feature_output_size = (
                        _yolo_forward(
                            model,
                            device,
                            model_config,
                            capture,
                            pil_image,
                            config,
                            subset.num_foreground_classes,
                        )
                    )
                reference = np.asarray(
                    case["frozen_evidence"][detector]["box_xyxy"], dtype=np.float64
                )
                score_values = candidate_scores.detach().float().cpu().numpy().astype(np.float64)
                candidate_index, reference_iou = _best_candidate_index(
                    candidate_boxes, score_values, reference
                )
                heatmap = _cam_numpy(
                    candidate_scores[candidate_index],
                    activation,
                    feature_output_size=feature_output_size,
                    original_size=targets[image_id].image_size,
                    detector=detector,
                    config=config,
                    retain_graph=False,
                )
                case_maps[(case["category"], case["rank"], detector)] = {
                    "heatmap": heatmap,
                    "candidate_box": candidate_boxes[candidate_index].copy(),
                    "candidate_score": float(score_values[candidate_index]),
                    "candidate_iou_to_frozen_box": reference_iou,
                    "energy_in_box": None,
                    "pointing_hit": None,
                }
                capture.clear()
                del activation, candidate_scores, tensor_image

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def aggregate_localization(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate valid CAMs by detector and frozen operating-point status."""

    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        detector_records = [item for item in records if item["detector"] == detector]
        for status in ("all", "true_positive", "false_negative"):
            selected = (
                detector_records
                if status == "all"
                else [item for item in detector_records if item["operating_status"] == status]
            )
            valid = [item for item in selected if item["valid_cam"]]
            energies = np.asarray([item["energy_in_box"] for item in valid], dtype=np.float64)
            hits = np.asarray([item["pointing_hit"] for item in valid], dtype=np.float64)
            areas = np.asarray([item["box_pixel_fraction"] for item in valid], dtype=np.float64)
            lifts = np.asarray([item["energy_lift_over_area"] for item in valid], dtype=np.float64)
            by_image: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for item in valid:
                by_image[str(item["image_id"])].append(item)
            image_energies = [
                float(np.mean([item["energy_in_box"] for item in items]))
                for items in by_image.values()
            ]
            image_hits = [
                float(np.mean([item["pointing_hit"] for item in items]))
                for items in by_image.values()
            ]
            rows.append(
                {
                    "detector": detector,
                    "operating_status": status,
                    "target_count": len(selected),
                    "valid_cam_count": len(valid),
                    "zero_energy_cam_count": len(selected) - len(valid),
                    "image_count": len({str(item["image_id"]) for item in selected}),
                    "mean_energy_in_box": float(energies.mean()) if len(valid) else None,
                    "median_energy_in_box": float(np.median(energies)) if len(valid) else None,
                    "pointing_game_accuracy": float(hits.mean()) if len(valid) else None,
                    "mean_box_area_fraction": float(areas.mean()) if len(valid) else None,
                    "mean_energy_lift_over_area": float(lifts.mean()) if len(valid) else None,
                    "image_macro_mean_energy_in_box": (
                        float(np.mean(image_energies)) if image_energies else None
                    ),
                    "image_macro_pointing_game_accuracy": (
                        float(np.mean(image_hits)) if image_hits else None
                    ),
                }
            )
    return rows


def _paired_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (str(item["image_id"]), int(item["target_index"]), str(item["detector"])): item
        for item in records
    }
    keys = sorted({(key[0], key[1]) for key in by_key})
    paired = [
        (by_key[(*key, "faster_rcnn")], by_key[(*key, "yolo11s")])
        for key in keys
        if by_key[(*key, "faster_rcnn")]["valid_cam"] and by_key[(*key, "yolo11s")]["valid_cam"]
    ]
    differences = np.asarray(
        [float(faster["energy_in_box"]) - float(yolo["energy_in_box"]) for faster, yolo in paired],
        dtype=np.float64,
    )
    return {
        "paired_valid_target_count": len(paired),
        "mean_energy_difference_faster_minus_yolo": (
            float(differences.mean()) if len(differences) else None
        ),
        "median_energy_difference_faster_minus_yolo": (
            float(np.median(differences)) if len(differences) else None
        ),
        "faster_higher_energy_count": int(np.sum(differences > 0)),
        "yolo_higher_energy_count": int(np.sum(differences < 0)),
        "equal_energy_count": int(np.sum(differences == 0)),
        "both_pointing_hit_count": sum(
            bool(faster["pointing_hit"]) and bool(yolo["pointing_hit"]) for faster, yolo in paired
        ),
        "faster_only_pointing_hit_count": sum(
            bool(faster["pointing_hit"]) and not bool(yolo["pointing_hit"])
            for faster, yolo in paired
        ),
        "yolo_only_pointing_hit_count": sum(
            not bool(faster["pointing_hit"]) and bool(yolo["pointing_hit"])
            for faster, yolo in paired
        ),
        "neither_pointing_hit_count": sum(
            not bool(faster["pointing_hit"]) and not bool(yolo["pointing_hit"])
            for faster, yolo in paired
        ),
    }


def _qualitative_rows(
    cases: Sequence[Mapping[str, Any]],
    case_maps: Mapping[tuple[str, int, DetectorName], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for detector in DETECTORS:
            evidence = case["frozen_evidence"][detector]
            rendered = case_maps[(case["category"], case["rank"], detector)]
            rows.append(
                {
                    "category": case["category"],
                    "rank": case["rank"],
                    "image_id": case["image_id"],
                    "target_index": case["target_index"],
                    "selection_score": case["selection_score"],
                    "selection_quantile": case.get("selection_quantile"),
                    "detector": detector,
                    "frozen_score": evidence.get("score"),
                    "frozen_iou": evidence.get("iou"),
                    "cam_candidate_score": rendered["candidate_score"],
                    "cam_candidate_iou": rendered.get("candidate_iou"),
                    "energy_in_box": rendered["energy_in_box"],
                    "pointing_hit": rendered["pointing_hit"],
                }
            )
    return rows


def _draw_box(axis: Any, box: Sequence[float], *, color: str, line_width: float) -> None:
    from matplotlib.patches import Rectangle

    x1, y1, x2, y2 = (float(value) for value in box)
    axis.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=line_width,
        )
    )


def _write_case_figure(
    path: Path,
    *,
    category: str,
    cases: Sequence[Mapping[str, Any]],
    case_maps: Mapping[tuple[str, int, DetectorName], Mapping[str, Any]],
    prepared: Mapping[str, Any],
    config: ExplainabilityConfig,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = prepared["subset"]
    targets = prepared["targets"]
    indices = {record.file_name: index for index, record in enumerate(subset.records)}
    selected = sorted(
        (item for item in cases if item["category"] == category),
        key=lambda item: int(item["rank"]),
    )
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(
            config.qualitative.panel_width_inches * 3,
            config.qualitative.row_height_inches * len(selected),
        ),
        squeeze=False,
        constrained_layout=True,
    )
    titles = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
    for row_index, case in enumerate(selected):
        image_id = str(case["image_id"])
        image = np.asarray(subset.load_pil(indices[image_id]))
        target_index = case["target_index"]
        axes[row_index, 0].imshow(image, cmap="gray")
        axes[row_index, 0].set_title(f"Original | {image_id[:8]}…")
        if target_index is not None:
            _draw_box(
                axes[row_index, 0],
                targets[image_id].boxes_xyxy[int(target_index)],
                color=config.qualitative.ground_truth_color,
                line_width=config.qualitative.line_width,
            )
        for column_index, detector in enumerate(DETECTORS, start=1):
            rendered = case_maps[(category, case["rank"], detector)]
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap="gray")
            axis.imshow(
                rendered["heatmap"],
                cmap=config.qualitative.colormap,
                alpha=config.qualitative.overlay_alpha,
                vmin=0,
                vmax=1,
            )
            if target_index is not None:
                _draw_box(
                    axis,
                    targets[image_id].boxes_xyxy[int(target_index)],
                    color=config.qualitative.ground_truth_color,
                    line_width=config.qualitative.line_width,
                )
            _draw_box(
                axis,
                rendered["candidate_box"],
                color=config.qualitative.candidate_color,
                line_width=config.qualitative.line_width,
            )
            score = float(rendered["candidate_score"])
            energy = rendered["energy_in_box"]
            detail = f"score={score:.3f}"
            if energy is not None:
                hit = "hit" if rendered["pointing_hit"] else "miss"
                detail += f" | energy={float(energy):.3f} | {hit}"
            axis.set_title(f"{titles[detector]}\n{detail}")
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(category.replace("_", " ").title())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=config.qualitative.dpi)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return path


def _source_identity(config: ExplainabilityConfig) -> dict[str, str]:
    paths = (
        config.source_path,
        config.project_root / "src/explainability/config.py",
        config.project_root / "src/explainability/gradcam.py",
        config.project_root / "src/explainability/pointing_game.py",
        config.project_root / "src/explainability/selection.py",
        config.project_root / "src/explainability/run_explainability.py",
    )
    return {
        path.resolve().relative_to(config.project_root).as_posix(): sha256_file(path)
        for path in paths
    }


def run_explainability(config: ExplainabilityConfig, *, smoke: bool = False) -> dict[str, Any]:
    """Run the bounded smoke or complete paired Phase 7 experiment."""

    prepared = _prepare(config)
    from src.utils.seed import initialize_reproducibility, seed_everything

    initialize_reproducibility(config.seed, config.resolve(config.outputs.log_dir))
    case_maps: dict[tuple[str, int, DetectorName], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for detector in DETECTORS:
        seed_everything(config.seed)
        records.extend(
            _run_detector(
                config,
                prepared,
                detector,
                smoke=smoke,
                case_maps=case_maps,
            )
        )
    records.sort(
        key=lambda item: (
            str(item["detector"]),
            str(item["image_id"]),
            int(item["target_index"]),
        )
    )
    aggregate = aggregate_localization(records)
    paired = _paired_summary(records)
    if smoke:
        payload = {
            "schema_version": 1,
            "status": "smoke_complete",
            "experiment_id": config.experiment_id,
            "seed": config.seed,
            "positive_images_per_detector": config.runtime.smoke_positive_images,
            "target_record_count": len(records),
            "aggregate": aggregate,
            "paired": paired,
        }
        path = _atomic_json(config.resolve(config.outputs.smoke_summary), payload)
        return {"summary": payload, "artifacts": {"smoke_summary": path.as_posix()}}

    expected_targets = sum(len(item.boxes_xyxy) for item in prepared["targets"].values())
    if len(records) != expected_targets * len(DETECTORS):
        raise AssertionError("quantitative Grad-CAM target grid is incomplete")
    expected_case_maps = len(prepared["cases"]) * len(DETECTORS)
    if len(case_maps) != expected_case_maps:
        raise AssertionError("qualitative Grad-CAM case grid is incomplete")

    target_path = _atomic_csv(config.resolve(config.outputs.target_table), TARGET_FIELDS, records)
    aggregate_fields = tuple(aggregate[0])
    aggregate_path = _atomic_csv(
        config.resolve(config.outputs.aggregate_table), aggregate_fields, aggregate
    )
    qualitative_rows = _qualitative_rows(prepared["cases"], case_maps)
    qualitative_fields = tuple(qualitative_rows[0])
    qualitative_path = _atomic_csv(
        config.resolve(config.outputs.qualitative_manifest),
        qualitative_fields,
        qualitative_rows,
    )
    figure_paths = {
        "good_figure": _write_case_figure(
            config.resolve(config.outputs.good_figure),
            category="good_prediction",
            cases=prepared["cases"],
            case_maps=case_maps,
            prepared=prepared,
            config=config,
        ),
        "bad_figure": _write_case_figure(
            config.resolve(config.outputs.bad_figure),
            category="bad_prediction",
            cases=prepared["cases"],
            case_maps=case_maps,
            prepared=prepared,
            config=config,
        ),
        "failure_figure": _write_case_figure(
            config.resolve(config.outputs.failure_figure),
            category="failure_case",
            cases=prepared["cases"],
            case_maps=case_maps,
            prepared=prepared,
            config=config,
        ),
    }
    phase6 = prepared["phase6"]
    checkpoint_hashes = {
        item.detector: sha256_file(phase6.resolve(item.checkpoint)) for item in phase6.detectors
    }
    artifacts = {
        "target_table": target_path.as_posix(),
        "target_table_sha256": sha256_file(target_path),
        "aggregate_table": aggregate_path.as_posix(),
        "aggregate_table_sha256": sha256_file(aggregate_path),
        "qualitative_manifest": qualitative_path.as_posix(),
        "qualitative_manifest_sha256": sha256_file(qualitative_path),
    }
    for name, path in figure_paths.items():
        artifacts[name] = path.as_posix()
        artifacts[f"{name}_sha256"] = sha256_file(path)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": config.experiment_id,
        "seed_scope": {
            "training_seed": config.seed,
            "scope": "primary checkpoint only for both detectors",
        },
        "config_path": config.source_path.as_posix(),
        "config_sha256": sha256_file(config.source_path),
        "source_identity": _source_identity(config),
        "phase6_provenance": {
            "config_path": phase6.source_path.as_posix(),
            "config_sha256": sha256_file(phase6.source_path),
            "summary_path": prepared["phase6_summary_path"].as_posix(),
            "summary_sha256": sha256_file(prepared["phase6_summary_path"]),
            "sample_manifest": prepared["sample_path"].as_posix(),
            "sample_manifest_sha256": prepared["sample_sha256"],
        },
        "checkpoints": checkpoint_hashes,
        "method": config.gradcam.model_dump(mode="json"),
        "metric": config.metric.model_dump(mode="json"),
        "target_layers": [item.model_dump(mode="json") for item in config.detectors],
        "population": {
            "sample_image_count": len(prepared["targets"]),
            "positive_image_count": sum(
                bool(len(item.boxes_xyxy)) for item in prepared["targets"].values()
            ),
            "negative_image_count": sum(
                not len(item.boxes_xyxy) for item in prepared["targets"].values()
            ),
            "ground_truth_box_count": expected_targets,
            "quantitative_record_count": len(records),
            "negative_images_excluded_reason": (
                "Energy-in-box and pointing-game metrics require a ground-truth box."
            ),
        },
        "aggregate": aggregate,
        "paired_descriptive_comparison": paired,
        "qualitative_selection": {
            "settings": config.qualitative.model_dump(mode="json"),
            "case_count": len(prepared["cases"]),
            "cases": [
                {
                    "category": item["category"],
                    "rank": item["rank"],
                    "image_id": item["image_id"],
                    "target_index": item["target_index"],
                    "selection_score": item["selection_score"],
                    "selection_quantile": item.get("selection_quantile"),
                }
                for item in prepared["cases"]
            ],
        },
        "artifacts": artifacts,
    }
    summary_path = _atomic_json(config.resolve(config.outputs.summary_json), summary)
    artifacts["summary_json"] = summary_path.as_posix()
    artifacts["summary_json_sha256"] = sha256_file(summary_path)
    return {"summary": summary, "artifacts": artifacts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/explainability.yaml"))
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_explainability_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True), flush=True)
        return 0
    result = run_explainability(config, smoke=args.mode == "smoke")
    print(
        json.dumps(
            {
                "status": result["summary"]["status"],
                "artifacts": result["artifacts"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
