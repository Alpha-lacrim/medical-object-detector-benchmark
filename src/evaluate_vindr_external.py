"""Frozen ten-checkpoint VinDr inference and independently replayable metrics.

No training, threshold selection, bootstrap or internal-result modification.
Image-linked outputs remain in the protocol's private tree. Existing runs are
never overwritten or silently resumed; verification is read-only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import inspect
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.analyze_exact_score_froc import _selection_key, exact_score_froc_rows
from src.benchmark_inference import Predictor
from src.data.prepare_vindr import (
    build_targets,
    ensure_private,
    read_csv,
    resolve_root,
    sha256,
    targets_to_coco,
    verify_freeze,
    verify_inventory,
    write_json,
)
from src.evaluate import (
    _serialize_prediction,
    _targets_from_dataset,
    load_and_validate_training_configs,
    load_phase5_config,
)
from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import ImagePrediction, evaluate_operating_point
from src.meddet_benchmark.reproducibility import configure_reproducibility
from src.models.faster_rcnn_data import CocoDetectionDataset
from src.utils.seed import log_run_environment


def read_json(path: Path) -> Any:
    """Read a provenance/metric document without changing it."""
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    """Fail closed at a scientific or provenance gate."""
    if not condition:
        raise ValueError(message)


def artifact(path: Path) -> dict[str, Any]:
    """Bind exact bytes using a repository-relative path."""
    return {"path": path.as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def dump_gzip(path: Path, payload: Any) -> None:
    """Create a private compressed bundle without overwriting existing evidence."""
    raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0))


def read_gzip(path: Path) -> Any:
    """Read a local restricted prediction or frontier bundle."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def preflight(config_path: Path, operation_path: Path) -> tuple[Any, ...]:
    """Read-only full freeze/checkpoint/source/prepared-data prerequisite gate."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    operation = yaml.safe_load(operation_path.read_text(encoding="utf-8"))
    frozen = verify_freeze(config_path, config)
    require(bool(operation["authorization"]), "Explicit Batch 49 authorization is required")
    adapter_path = Path(operation["adapter_config"])
    adapter = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    private = Path(config["outputs"]["private_root"])
    receipt_path = (
        Path(config["outputs"]["aggregate_results_root"]) / adapter["aggregate_summary_name"]
    )
    receipt = read_json(receipt_path)
    require(receipt["status"] == "passed" and receipt["mode"] == "prepare", "Batch 48 incomplete")
    bindings = {
        "config_sha256": config_path,
        "protocol_sha256": Path(config["protocol_document"]),
        "adapter_config_sha256": adapter_path,
        "adapter_sha256": Path(inspect.getfile(build_targets)),
        "manifest_sha256": Path(config["outputs"]["local_manifest"]),
        "annotations_sha256": Path(config["outputs"]["local_annotations"]),
    }
    for key, path in bindings.items():
        require(receipt[key] == sha256(path), f"Batch 48 binding mismatch: {key}")
    audit = read_json(private / adapter["private_audit_name"])
    require(audit["summary"] == receipt, "Private/public preparation receipts disagree")
    require(receipt["dataset_version"] == config["dataset"]["version"], "Release version mismatch")
    for key in ("ontology", "preprocessing", "evaluation"):
        # JSON object keys are strings, unlike YAML category IDs.
        require(receipt[key] == json.loads(json.dumps(config[key])), f"Prepared {key} mismatch")
    inventory = verify_inventory(resolve_root(config), config)
    with Path(config["outputs"]["local_manifest"]).open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    require(len(manifest) == len(inventory) == receipt["test_count"], "Test population mismatch")
    sizes = {}
    for index, (entry, (image_id, source, digest)) in enumerate(
        zip(manifest, inventory, strict=True)
    ):
        require(entry["image_id"] == image_id, "Manifest order differs from official inventory")
        require(entry["source_sha256"] == digest == sha256(source), "DICOM hash mismatch")
        png = private / adapter["images_directory_name"] / entry["file_name"]
        ensure_private(png, private)
        require(entry["file_name"] == image_id + ".png", "Prepared filename mismatch")
        require(sha256(png) == entry["png_sha256"], "Prepared pixel file hash mismatch")
        sizes[image_id] = (int(entry["height"]), int(entry["width"]))
        if (index + 1) % operation["progress_every"] == 0:
            print(f"Integrity {index + 1}/{len(inventory)}", flush=True)
    _, source_rows = read_csv(resolve_root(config) / config["dataset"]["paths"]["boxes"])
    source_targets = build_targets(source_rows, sizes, config)
    require(
        targets_to_coco(source_targets, config) == read_json(bindings["annotations_sha256"]),
        "Canonical annotations differ from source",
    )
    dataset = CocoDetectionDataset(private / adapter["loader_config_name"], "test", mode="full")
    require(
        len(dataset) == config["dataset"]["expected_images"], "Common loader population mismatch"
    )
    for record, entry in zip(dataset.records, manifest, strict=True):
        require(
            record.file_name == entry["file_name"]
            and (record.height, record.width) == sizes[entry["image_id"]],
            "Loader order/shape mismatch",
        )
    phase = load_phase5_config(config["models"]["run_inventory_config"])
    source_phase = load_phase5_config(operation["source_evaluation_config"])
    models = load_and_validate_training_configs(source_phase)
    require(
        [r.model_dump(mode="json") for r in phase.runs]
        == [r.model_dump(mode="json") for r in source_phase.runs],
        "Run inventory mismatch",
    )
    require(
        {(r.detector, r.seed) for r in phase.runs}
        == {(d, s) for d in config["models"]["detectors"] for s in config["models"]["seeds"]},
        "All ten runs including seed 271 are required",
    )
    expected = {
        r["source_path"]: r
        for r in read_json(Path(config["models"]["checkpoint_manifest"]))["checkpoints"]
    }
    checkpoints = []
    for run in phase.runs:
        path = Path(run.checkpoint)
        identity = expected[path.as_posix()]
        require(
            sha256(path) == identity["sha256"] and path.stat().st_size == identity["size_bytes"],
            "Checkpoint identity mismatch",
        )
        require(
            sha256(Path(run.training_config)) == identity["config"]["sha256"],
            "Training config mismatch",
        )
        model = models[(run.detector, run.seed)]
        require(
            model.runtime.amp
            and model.runtime.amp_dtype == config["runtime"]["amp_dtype"][run.detector],
            "AMP config mismatch",
        )
        size = model.model.min_size if run.detector == "faster_rcnn" else model.model.input_size
        require(size == config["preprocessing"]["input_size"], "Native input size mismatch")
        checkpoints.append(
            {
                "detector": run.detector,
                "seed": run.seed,
                **artifact(path),
                "training_config_sha256": sha256(Path(run.training_config)),
            }
        )
    evaluation = config["evaluation"]
    require(
        phase.evaluation.coco_minimum_score == evaluation["candidate_score_floor"]
        and phase.evaluation.max_detections == evaluation["max_detections_per_image"]
        and phase.evaluation.nms_iou_threshold == evaluation["nms_iou_threshold"],
        "Collection contract mismatch",
    )
    inputs = dict(frozen)
    for path in [
        config_path,
        operation_path,
        receipt_path,
        *bindings.values(),
        private / adapter["loader_config_name"],
        private / adapter["private_audit_name"],
        *map(Path, operation["source_files"]),
        *[Path(r.training_config) for r in phase.runs],
    ]:
        inputs[path.as_posix()] = sha256(path)
    gate = {
        "status": "passed",
        "protocol_id": config["protocol_id"],
        "dataset_version": config["dataset"]["version"],
        "images": len(dataset),
        "target_boxes": receipt["strict_box_count"],
        "strict_positive_images": receipt["strict_positive_count"],
        "authorization": operation["authorization"],
        "input_sha256": inputs,
        "checkpoints": checkpoints,
        "freeze_files_verified": len(frozen),
    }
    return config, operation, phase, models, dataset, gate


def score_summary(scores: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Detection-level linear quantiles, with explicit empty populations."""
    values = np.asarray(scores, dtype=np.float64)
    quantiles = config["score_summaries"]["quantiles"]
    return {
        "detection_count": len(values),
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "quantiles": {
            str(q): float(
                np.quantile(values, q, method=config["score_summaries"]["quantile_method"])
            )
            if len(values)
            else None
            for q in quantiles
        },
    }


def select_budgets(curve: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the exact v4 per-run rule without its multi-run SD aggregation."""
    settings = config["evaluation"]["froc"]
    floor = curve[-1]
    points = []
    for budget in settings["fp_per_image_budgets"]:
        eligible = [r for r in curve if r["fp_per_image"] <= budget + settings["numeric_tolerance"]]
        selected = max(eligible, key=_selection_key)
        points.append(
            {
                "fp_per_image_budget": budget,
                "sensitivity": selected["sensitivity"],
                "achieved_fp_per_image": selected["fp_per_image"],
                "candidate_floor_endpoint_fp_per_image": floor["fp_per_image"],
                "candidate_floor_endpoint_sensitivity": floor["sensitivity"],
                "candidate_floor_limited": floor["fp_per_image"]
                < budget - settings["numeric_tolerance"],
                "selected_at_candidate_floor_boundary": selected["is_candidate_floor_boundary"],
            }
        )
    return points


def aggregate_metrics(runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Equal-run means/sample SDs with explicit counts; never pool detections."""

    def flatten(value: Any, prefix: str = "") -> dict[str, float | None]:
        if isinstance(value, dict):
            output = {}
            for key, child in value.items():
                if key not in {"detection_count_histogram", "seed", "selection_seeds"}:
                    output.update(flatten(child, f"{prefix}.{key}" if prefix else key))
            return output
        if isinstance(value, list):
            output = {}
            for index, child in enumerate(value):
                output.update(flatten(child, f"{prefix}.{index}"))
            return output
        if value is None or isinstance(value, (int, float)):
            return {prefix: value}
        return {}

    result = {}
    ddof = config["statistics"]["standard_deviation_ddof"]
    for detector in config["models"]["detectors"]:
        group = [r["metrics"] for r in runs if r["metrics"]["detector"] == detector]
        require(
            sorted(r["seed"] for r in group) == config["models"]["seeds"], "Aggregation lost a run"
        )
        flat = [flatten(r) for r in group]
        summary = {}
        for key in sorted(set().union(*(r.keys() for r in flat))):
            values = [r[key] for r in flat if r.get(key) is not None]
            summary[key] = {
                "defined_run_count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "sample_sd": float(np.std(values, ddof=ddof)) if len(values) > ddof else None,
            }
        result[detector] = summary
    return result


def compute_metrics(
    predictions: list[ImagePrediction],
    targets: list[Any],
    config: dict[str, Any],
    detector: str,
    seed: int,
) -> tuple[dict[str, Any], list[Any]]:
    """Replay frozen AP, exact FROC, transported cutoffs, and emitted scores."""
    evaluation = config["evaluation"]
    classes = tuple(config["ontology"]["canonical_classes"])
    cap = evaluation["max_detections_per_image"]
    require(
        [p.image_id for p in predictions] == [t.image_id for t in targets],
        "Bundle image ordering mismatch",
    )
    for prediction, target in zip(predictions, targets, strict=True):
        require(prediction.image_size == target.image_size, "Prediction native dimensions mismatch")
        require(
            len(prediction.scores) <= cap
            and all(prediction.scores >= evaluation["candidate_score_floor"]),
            "Candidate support mismatch",
        )
    ap = evaluate_coco(
        predictions,
        targets,
        class_ids=classes,
        class_names=config["ontology"]["canonical_classes"],
        minimum_score=evaluation["ap_minimum_score"],
        max_detections=cap,
    )
    curve, frontier = exact_score_froc_rows(
        predictions,
        targets,
        detector=detector,
        seed=seed,
        class_ids=classes,
        candidate_score_floor=evaluation["candidate_score_floor"],
        iou_threshold=evaluation["matching_iou_threshold"],
        max_detections=cap,
    )
    budgets = select_budgets(curve, config)
    policies, scores = {}, {}
    supports = {
        "all_retained_candidates": evaluation["candidate_score_floor"],
        "ap_candidates": evaluation["ap_minimum_score"],
    }
    for role, policy in config["threshold_transport"].items():
        if role not in ("primary", "secondary"):
            continue
        threshold = policy["thresholds"][detector]
        result = evaluate_operating_point(
            predictions,
            targets,
            class_ids=classes,
            score_threshold=threshold,
            iou_threshold=evaluation["matching_iou_threshold"],
            max_detections=cap,
        )["overall"]
        count = sum(int(np.count_nonzero(p.scores >= threshold)) for p in predictions)
        emitting = sum(bool(np.any(p.scores >= threshold)) for p in predictions)
        policies[role] = {
            "label": policy["label"],
            "threshold": threshold,
            "selection_seeds": policy["selection_seeds"],
            **{k: result[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "images": len(targets),
            "target_boxes": sum(len(t.labels) for t in targets),
            "detections": count,
            "images_with_detections": emitting,
            "zero_detection_images": len(targets) - emitting,
            "fp_per_image": result["fp"] / len(targets),
            "detections_per_image": count / len(targets),
            "percent_images_with_detections": 100 * emitting / len(targets),
        }
        supports["historical_threshold" if role == "primary" else "posthoc_n5_threshold"] = (
            threshold
        )
    for name, floor in supports.items():
        populations = [p.scores[p.scores >= floor] for p in predictions]
        counts = np.asarray([len(p) for p in populations])
        scores[name] = {
            "score_floor": floor,
            **score_summary(np.concatenate(populations), config),
            "zero_detection_images": int(np.count_nonzero(counts == 0)),
            "detection_count_histogram": {
                str(n): int(np.count_nonzero(counts == n)) for n in np.unique(counts)
            },
        }
    return {
        "detector": detector,
        "seed": seed,
        "ap": ap,
        "froc": frontier,
        "froc_budgets": budgets,
        "threshold_transport": policies,
        "score_summaries": scores,
        "images_at_detection_cap": sum(len(p.scores) == cap for p in predictions),
    }, curve


def native_floor_audit(floor: float) -> dict[str, Any]:
    """Test native strict-> equality behavior on synthetic boxes, without models."""
    import torch
    from torchvision.models.detection.roi_heads import RoIHeads
    from ultralytics.utils.nms import non_max_suppression

    roi = RoIHeads(None, None, None, 0.5, 0.5, 1, 0.5, None, floor, 0.5, 100)
    # Obtain the exact softmax-emitted value before setting the equality cutoff.
    logits = torch.tensor([[0.0, np.log(floor / (1 - floor))]], dtype=torch.float32)
    emitted = float(logits.softmax(-1)[0, 1])
    roi.score_thresh = emitted
    boxes = [torch.tensor([[0.0, 0.0, 10.0, 10.0]])]
    _, scores, _ = roi.postprocess_detections(logits, torch.zeros((1, 8)), boxes, [(20, 20)])
    require(len(scores[0]) == 0, "Torchvision native equality behavior changed")
    roi.score_thresh = float(np.nextafter(np.float32(emitted), np.float32(-np.inf)))
    _, scores, _ = roi.postprocess_detections(logits, torch.zeros((1, 8)), boxes, [(20, 20)])
    require(len(scores[0]) == 1, "Torchvision native above-floor behavior changed")
    candidates = torch.zeros((1, 5, 2), dtype=torch.float32)
    candidates[0, :4, :] = torch.tensor([[5, 15], [5, 15], [4, 4], [4, 4]])
    candidates[0, 4, :] = torch.tensor([floor, np.nextafter(np.float32(floor), np.float32(np.inf))])
    outputs = non_max_suppression(candidates, conf_thres=floor, nc=1)
    require(len(outputs[0]) == 1, "Ultralytics native strict-floor behavior changed")
    return {
        "torchvision_native_score_comparison": "strict_greater_than",
        "ultralytics_native_score_comparison": "strict_greater_than",
        "common_evaluator_score_comparison": "greater_than_or_equal",
        "equality_candidate_not_emitted_by_native_filters": True,
        "native_floor_unchanged": floor,
        "torchvision_roi_heads_sha256": sha256(Path(inspect.getfile(RoIHeads))),
        "ultralytics_nms_sha256": sha256(Path(inspect.getfile(non_max_suppression))),
    }


def deserialize(payload: list[dict[str, Any]]) -> list[ImagePrediction]:
    """Validate numeric prediction values through the shared record type."""
    return [
        ImagePrediction(
            image_id=p["image_id"],
            image_size=tuple(p["image_size"]),
            boxes_xyxy=np.asarray(p["boxes_xyxy"]).reshape(-1, 4),
            labels=np.asarray(p["labels"], dtype=np.int64),
            scores=np.asarray(p["scores"]),
        )
        for p in payload
    ]


def restore_native_bounds(
    boxes: np.ndarray, image_size: tuple[int, int], detector: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Repair only one-ULP float32 source-boundary restoration overshoots.

    Torchvision clips in resized coordinates before float32 inverse scaling.
    That multiplication can restore a clipped edge to nextafter(native_bound,
    +inf). This is numerical canonicalization, not general box clipping. Scores,
    labels, candidate counts, in-bound coordinates and larger errors are intact.
    """
    raw = np.asarray(boxes)
    height, width = image_size
    limits = np.asarray([width, height, width, height], dtype=np.float64)
    require(
        raw.ndim == 2 and raw.shape[1] == 4 and np.isfinite(raw).all(),
        "Invalid native box shape/value",
    )
    require(not np.any(raw < 0), "Negative native coordinates are not a rounding repair")
    outside = raw > limits
    if not np.any(outside):
        return raw, []
    require(
        detector == "faster_rcnn" and raw.dtype == np.float32,
        "Only documented Torchvision float32 restoration is repairable",
    )
    upper = np.nextafter(limits.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    require(not np.any(raw > upper), "Native box exceeds the one-ULP restoration bound")
    corrected = raw.copy()
    changes = []
    for row, column in np.argwhere(outside):
        changes.append(
            {
                "box_index": int(row),
                "coordinate_index": int(column),
                "raw_value": float(raw[row, column]),
                "canonical_value": float(limits[column]),
            }
        )
        corrected[row, column] = limits[column]
    return corrected, changes


def verify_boundary_repairs(bundle: dict[str, Any]) -> None:
    """Replay every recorded numerical correction from preserved raw coordinates."""
    predictions = {p["image_id"]: p for p in bundle["predictions"]}
    repairs = bundle["native_boundary_repairs"]
    require(
        len({r["image_id"] for r in repairs}) == len(repairs), "Duplicate boundary repair image"
    )
    for record in repairs:
        prediction = predictions[record["image_id"]]
        raw = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
        for change in record["changes"]:
            row, col = change["box_index"], change["coordinate_index"]
            require(float(raw[row, col]) == change["canonical_value"], "Repair coordinate mismatch")
            raw[row, col] = change["raw_value"]
        restored, changes = restore_native_bounds(
            raw, tuple(prediction["image_size"]), bundle["metadata"]["detector"]
        )
        require(
            changes == record["changes"] and np.array_equal(restored, prediction["boxes_xyxy"]),
            "Native boundary repair replay mismatch",
        )
    summary = bundle["metadata"]["native_boundary_repair"]
    require(
        summary["images"] == len(repairs)
        and summary["coordinates"] == sum(len(r["changes"]) for r in repairs)
        and summary["maximum_excess_pixels"]
        == max(
            (c["raw_value"] - c["canonical_value"] for r in repairs for c in r["changes"]),
            default=0.0,
        ),
        "Native boundary repair summary mismatch",
    )


def run(config_path: Path, operation_path: Path) -> dict[str, Any]:
    """Sequential CUDA inference; create immutable private and aggregate evidence."""
    config, operation, phase, models, dataset, gate = preflight(config_path, operation_path)
    import torch

    require(torch.cuda.is_available(), "Pinned local CUDA runtime unavailable; run locally")
    for package, expected in config["runtime"]["versions"].items():
        actual = torch.version.cuda if package == "cuda" else version(package)
        require(actual == expected, f"Pinned {package} mismatch: {actual}")
    require(
        torch.cuda.get_device_name(0) == operation["required_gpu_name"],
        "Intended local GPU unavailable",
    )
    private_root = Path(config["outputs"]["local_predictions"])
    ensure_private(private_root, Path(config["outputs"]["private_root"]))
    result_root = Path(config["outputs"]["aggregate_results_root"])
    require(
        not private_root.exists() and not (result_root / operation["summary_name"]).exists(),
        "External inference artifacts exist; verify/review, never overwrite",
    )
    native = native_floor_audit(config["evaluation"]["candidate_score_floor"])
    private_root.mkdir(parents=True)
    write_json(private_root / operation["gate_name"], gate)
    report = configure_reproducibility(
        phase.seeds[0],
        deterministic=config["runtime"]["deterministic"],
        allow_tf32=config["runtime"]["allow_tf32"],
    )
    torch.set_num_threads(config["runtime"]["torch_threads"])
    torch.cuda.set_device(torch.device(config["runtime"]["device"]))
    environment_dir = private_root / operation["environment_directory_name"]
    log_run_environment(environment_dir, report)
    environment = {
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "pycocotools")
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": subprocess.list2cmdline(
            [sys.executable, "-m", "src.evaluate_vindr_external", *sys.argv[1:]]
        ),
        "files": [artifact(p) for p in sorted(environment_dir.iterdir()) if p.is_file()],
    }
    execution = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "gate": gate,
        "native_candidate_floor_audit": native,
        "runs": [],
    }
    execution_path = private_root / operation["run_receipt_name"]
    write_json(execution_path, execution)
    targets = _targets_from_dataset(dataset)
    try:
        for run_spec in phase.runs:
            started, start_time = datetime.now(UTC).isoformat(), time.perf_counter()
            run_dir = private_root / f"{run_spec.detector}_seed{run_spec.seed}"
            run_dir.mkdir()
            configure_reproducibility(
                run_spec.seed,
                deterministic=config["runtime"]["deterministic"],
                allow_tf32=config["runtime"]["allow_tf32"],
            )
            predictor = Predictor(
                run_spec, phase, models[(run_spec.detector, run_spec.seed)], dataset
            )
            if run_spec.detector == "faster_rcnn":
                predictor.model.roi_heads.score_thresh = config["evaluation"][
                    "candidate_score_floor"
                ]
                require(
                    predictor.model.roi_heads.nms_thresh
                    == config["evaluation"]["nms_iou_threshold"]
                    and predictor.model.roi_heads.detections_per_img
                    == config["evaluation"]["max_detections_per_image"],
                    "Native Faster R-CNN contract mismatch",
                )
            # First-call YOLO fusion may replace modules; initialize before the hook.
            predictor.predict(0, None)
            observed_amp = predictor.verify_amp(0)
            require(
                torch.are_deterministic_algorithms_enabled()
                and not torch.backends.cudnn.benchmark
                and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32,
                "Native predictor changed deterministic runtime",
            )
            predictions = []
            boundary_repairs = []
            torch.cuda.reset_peak_memory_stats()
            for index, target in enumerate(targets):
                values = predictor.predict(index, None)
                boxes, changes = restore_native_bounds(
                    values["boxes"], target.image_size, run_spec.detector
                )
                if changes:
                    boundary_repairs.append({"image_id": target.image_id, "changes": changes})
                predictions.append(
                    ImagePrediction(
                        image_id=target.image_id,
                        image_size=target.image_size,
                        boxes_xyxy=boxes,
                        scores=values["scores"],
                        labels=values["labels"],
                    )
                )
                if (index + 1) % operation["progress_every"] == 0:
                    print(
                        f"{run_spec.detector} seed {run_spec.seed}: {index + 1}/{len(targets)}",
                        flush=True,
                    )
            checkpoint = next(
                c
                for c in gate["checkpoints"]
                if (c["detector"], c["seed"]) == (run_spec.detector, run_spec.seed)
            )
            require(
                sha256(Path(run_spec.checkpoint)) == checkpoint["sha256"],
                "Checkpoint changed during inference",
            )
            metadata = {
                "detector": run_spec.detector,
                "seed": run_spec.seed,
                "checkpoint": checkpoint,
                "input_sha256": gate["input_sha256"],
                "protocol_id": config["protocol_id"],
                "dataset_version": config["dataset"]["version"],
                "environment": environment,
                "started_utc": started,
                "completed_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - start_time,
                "amp_convolution_dtype": observed_amp,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "native_candidate_floor_audit": native,
                "native_boundary_repair": {
                    "rule": "Torchvision float32 upper-bound overshoot of at most one ULP only",
                    "images": len(boundary_repairs),
                    "coordinates": sum(len(r["changes"]) for r in boundary_repairs),
                    "maximum_excess_pixels": max(
                        (
                            c["raw_value"] - c["canonical_value"]
                            for r in boundary_repairs
                            for c in r["changes"]
                        ),
                        default=0.0,
                    ),
                },
            }
            bundle_path = run_dir / operation["prediction_name"]
            dump_gzip(
                bundle_path,
                {
                    "metadata": metadata,
                    "predictions": [_serialize_prediction(p) for p in predictions],
                    "native_boundary_repairs": boundary_repairs,
                },
            )
            metrics, curve = compute_metrics(
                predictions, targets, config, run_spec.detector, run_spec.seed
            )
            curve_path = run_dir / operation["private_froc_name"]
            dump_gzip(curve_path, curve)
            metrics_path = run_dir / operation["metrics_name"]
            write_json(metrics_path, metrics)
            execution["runs"].append(
                {
                    "metadata": metadata,
                    "metrics": metrics,
                    "artifacts": [artifact(p) for p in (bundle_path, curve_path, metrics_path)],
                }
            )
            write_json(execution_path, execution)
            del predictor, predictions, curve, metrics
            gc.collect()
            torch.cuda.empty_cache()
        execution["status"] = "complete"
        execution["completed_utc"] = datetime.now(UTC).isoformat()
        write_json(execution_path, execution)
        # Public summary contains no image identifiers, row-level scores or coordinates.
        public = {
            "status": "complete",
            "protocol_id": config["protocol_id"],
            "dataset_version": config["dataset"]["version"],
            "gate": gate,
            "environment": environment,
            "native_candidate_floor_audit": native,
            "runs": execution["runs"],
            "equal_run_descriptive_summaries": aggregate_metrics(execution["runs"], config),
            "execution_receipt": artifact(execution_path),
            "statistics_status": "Batch 50 not run; per-run descriptive results only",
        }
        write_json(result_root / operation["summary_name"], public)
        return {"status": "complete", "runs": len(execution["runs"])}
    except BaseException as error:
        execution["status"] = "stopped_on_error"
        execution["error_type"] = type(error).__name__
        execution["error_message"] = str(error)
        write_json(execution_path, execution)
        raise


def verify(config_path: Path, operation_path: Path) -> dict[str, Any]:
    """Recompute every per-run result from hash-bound private bundles, no inference."""
    config, operation, phase, _, dataset, gate = preflight(config_path, operation_path)
    root = Path(config["outputs"]["local_predictions"])
    execution = read_json(root / operation["run_receipt_name"])
    public = read_json(
        Path(config["outputs"]["aggregate_results_root"]) / operation["summary_name"]
    )
    require(execution["status"] == public["status"] == "complete", "Incomplete inference")
    require(execution["gate"] == public["gate"] == gate, "Inference input provenance changed")
    require(
        public["execution_receipt"] == artifact(root / operation["run_receipt_name"]),
        "Execution receipt changed",
    )
    require(execution["runs"] == public["runs"], "Public/private results disagree")
    require(
        public["equal_run_descriptive_summaries"] == aggregate_metrics(execution["runs"], config),
        "Equal-run aggregation replay mismatch",
    )
    expected = [(r.detector, r.seed) for r in phase.runs]
    require(
        [(r["metadata"]["detector"], r["metadata"]["seed"]) for r in execution["runs"]] == expected,
        "Incomplete/duplicate run inventory",
    )
    for item in execution["environment"]["files"]:
        require(item == artifact(Path(item["path"])), "Software environment artifact changed")
    targets = _targets_from_dataset(dataset)
    for result in execution["runs"]:
        for item in result["artifacts"]:
            require(item == artifact(Path(item["path"])), "Run artifact hash mismatch")
        bundle = read_gzip(Path(result["artifacts"][0]["path"]))
        verify_boundary_repairs(bundle)
        require(bundle["metadata"] == result["metadata"], "Bundle/receipt provenance mismatch")
        meta = bundle["metadata"]
        require(
            meta["input_sha256"] == gate["input_sha256"]
            and meta["checkpoint"] in gate["checkpoints"],
            "Run input identity mismatch",
        )
        require(
            meta["amp_convolution_dtype"]
            == "torch." + config["runtime"]["amp_dtype"][meta["detector"]],
            "Unverified AMP",
        )
        metrics, curve = compute_metrics(
            deserialize(bundle["predictions"]), targets, config, meta["detector"], meta["seed"]
        )
        require(
            metrics == result["metrics"] == read_json(Path(result["artifacts"][2]["path"])),
            "Metric replay mismatch",
        )
        require(
            curve == read_gzip(Path(result["artifacts"][1]["path"])),
            "Exact-score frontier replay mismatch",
        )
        print(f"Verified {meta['detector']} seed {meta['seed']}", flush=True)
    return {
        "status": "verified",
        "runs": len(expected),
        "images_per_run": len(targets),
        "summary_sha256": sha256(
            Path(config["outputs"]["aggregate_results_root"]) / operation["summary_name"]
        ),
    }


def main() -> None:
    """Expose explicit preflight, run and read-only verification modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operation-config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run", "verify"), required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = preflight(args.config, args.operation_config)[-1]
        result = {
            "status": result["status"],
            "images": result["images"],
            "checkpoints": len(result["checkpoints"]),
            "freeze_files_verified": result["freeze_files_verified"],
        }
    else:
        result = (run if args.mode == "run" else verify)(args.config, args.operation_config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
