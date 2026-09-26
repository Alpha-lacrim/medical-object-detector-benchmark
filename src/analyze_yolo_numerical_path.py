"""Versioned, inference-only RSNA sensitivity to the frozen external YOLO path."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from scipy.stats import spearmanr

from src.benchmark_inference import Predictor
from src.evaluate import _serialize_prediction, _targets_from_dataset, load_phase5_config
from src.evaluate_vindr_external import compute_metrics, deserialize, read_gzip, require
from src.meddet_benchmark.evaluation import ImagePrediction, evaluate_operating_point
from src.meddet_benchmark.reproducibility import configure_reproducibility
from src.models.faster_rcnn_data import CocoDetectionDataset
from src.models.yolo_config import load_yolo_config
from src.utils.seed import log_run_environment


def digest(path: str | Path) -> str:
    """Hash exact bytes without reading large checkpoints into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: str | Path) -> Any:
    """Read an existing UTF-8 JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    """Stable strict JSON, also used for deterministic gzip payloads."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: Any, *, compressed: bool = False) -> None:
    """Exclusive creation: partial/completed prior evidence is never replaced."""
    raw = canonical_bytes(payload)
    if compressed:
        raw = gzip.compress(raw, compresslevel=9, mtime=0)
    with path.open("xb") as stream:
        stream.write(raw)


def validate_contract(config: dict[str, Any], external: dict[str, Any]) -> None:
    """Require all original runs and unchanged scientific evaluator definitions."""
    require(config["schema_version"] == 1, "Unsupported sensitivity schema")
    require(config["status"].startswith("secondary_sensitivity"), "Secondary status required")
    expected = external["models"]["seeds"]
    require(config["seeds"] == expected, "All five seeds including 271 are required")
    require([r["seed"] for r in config["runs"]] == expected, "Run scope lost or reordered a seed")
    for key in ("training", "threshold_selection", "external_inference"):
        require(config["scope"][key] is False, f"Forbidden operation: {key}")
    require(config["scope"]["split"] == "test", "Only immutable internal test is permitted")
    for key in ("evaluation", "threshold_transport", "score_summaries"):
        require(config[key] == external[key], f"Frozen {key} differs")
    for key, value in external["runtime"].items():
        require(config["runtime"][key] == value, f"Frozen runtime differs: {key}")
    require(config["runtime"]["activation_dtype"] == "torch.bfloat16", "Explicit bfloat16 required")
    require(config["runtime"]["amp"] is True, "Explicit autocast required")
    require(
        config["preprocessing"]["input_size"] == external["preprocessing"]["input_size"],
        "Resolution changed",
    )
    require(config["preprocessing"]["source"] == "native_file_path", "Native file source required")
    require(config["preprocessing"]["rgb_argument"] is None, "No alternative array adapter")
    require(config["preprocessing"]["quantize"] is None, "Native FP32 weights required")
    root = Path(config["outputs"]["root"])
    require(
        root.parent.as_posix() == "results/logs"
        and root.name.startswith("phase53_")
        and root.name.endswith("_v1")
        and ".." not in root.parts,
        "Versioned Batch 53 output root required; historical paths forbidden",
    )
    names = [v for k, v in config["outputs"].items() if k != "root"]
    require(len(names) == len(set(names)), "Output names overlap")
    require(all(Path(n).name == n and n not in (".", "..") for n in names), "Output escapes root")
    for source in config["input_sha256"]:
        require(not Path(source).resolve().is_relative_to(root.resolve()), "Output overlaps input")


def load_config(path: Path) -> dict[str, Any]:
    """Load explicit versioned settings and check their frozen source contract."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    external = yaml.safe_load(Path(config["inputs"]["external_config"]).read_text(encoding="utf-8"))
    validate_contract(config, external)
    config["ontology"] = external["ontology"]
    return config


def check_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Rehash every binding and reconcile annotations, split and both FP32 supports."""
    for path, expected in config["input_sha256"].items():
        require(digest(path) == expected, f"Input hash mismatch: {path}")
    release = read_json(config["inputs"]["checkpoint_manifest"])
    expected_runs = [r for r in release["checkpoints"] if r["detector"] == "yolo11s"]
    dataset = CocoDetectionDataset(config["inputs"]["dataset_config"], "test", mode="full")
    require(len(dataset) == config["scope"]["expected_images"], "Image count changed")
    targets = _targets_from_dataset(dataset)
    require(
        sum(len(t.labels) for t in targets) == config["scope"]["expected_target_boxes"],
        "Annotations changed",
    )
    with Path(config["inputs"]["test_manifest"]).open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    names = [r.file_name for r in dataset.records]
    require(len(manifest) == len(names), "Split length changed")
    require(
        {r["processed_file"] for r in manifest} == set(names), "Split/annotation membership differs"
    )
    require(all(r["split"] == "test" for r in manifest), "Non-test image in manifest")
    cohort = yaml.safe_load(Path(config["inputs"]["cohort_config"]).read_text(encoding="utf-8"))
    require(
        digest(config["inputs"]["test_manifest"])
        == cohort["inputs"]["expected_split_sha256"]["test"],
        "Original split hash differs",
    )
    annotation_hash = digest(config["inputs"]["annotations"])
    require(
        dataset.annotation_file.resolve() == Path(config["inputs"]["annotations"]).resolve(),
        "Loader annotation path differs",
    )
    phase = load_phase5_config(config["inputs"]["froc_collection_config"])
    phase_runs = [r for r in phase.runs if r.detector == "yolo11s"]
    require([r.seed for r in phase_runs] == config["seeds"], "Original run grid differs")
    for run, original, spec in zip(config["runs"], expected_runs, phase_runs, strict=True):
        require(run["seed"] == original["seed"] == spec.seed, "Checkpoint seed differs")
        require(
            run["checkpoint"] == original["source_path"] == spec.checkpoint.as_posix(),
            "Checkpoint selection changed",
        )
        require(
            run["sha256"] == original["sha256"] == digest(run["checkpoint"]),
            "Checkpoint hash invalid",
        )
        require(
            Path(run["checkpoint"]).stat().st_size == run["size_bytes"] == original["size_bytes"],
            "Checkpoint size invalid",
        )
        require(
            run["training_config"] == original["config"]["path"],
            "Training config selection changed",
        )
        require(
            digest(run["training_config"]) == original["config"]["sha256"],
            "Training config hash invalid",
        )
        bundles = [read_gzip(Path(run[key])) for key in ("fp32_ap", "fp32_froc")]
        for bundle in bundles:
            require(
                bundle["seed"] == run["seed"] and bundle["detector"] == "yolo11s",
                "Wrong historical run",
            )
            require(bundle["checkpoint_sha256"] == run["sha256"], "Historical checkpoint differs")
            require(bundle["annotation_sha256"] == annotation_hash, "Historical annotations differ")
            require(
                [p["image_id"] for p in bundle["predictions"]] == names,
                "Historical image order differs",
            )
        ap, low = [deserialize(b["predictions"]) for b in bundles]
        for a, b in zip(ap, low, strict=True):
            keep = b.scores >= config["evaluation"]["ap_minimum_score"]
            require(
                np.array_equal(a.scores, b.scores[keep])
                and np.array_equal(a.boxes_xyxy, b.boxes_xyxy[keep])
                and np.array_equal(a.labels, b.labels[keep]),
                "Historical AP support differs from FROC collection",
            )
    image_hashes = [
        {"file_name": name, "sha256": digest(dataset.image_path(i))} for i, name in enumerate(names)
    ]
    return {
        "dataset": dataset,
        "targets": targets,
        "phase": phase,
        "runs": phase_runs,
        "image_hashes": image_hashes,
    }


def runtime_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """Gate inference to the pinned, bfloat16-capable NVIDIA environment."""
    import torch
    import ultralytics

    settings = config["runtime"]
    require(torch.cuda.is_available(), "Compatible NVIDIA environment unavailable")
    require(torch.cuda.is_bf16_supported(), "CUDA bfloat16 unsupported")
    require(
        torch.cuda.get_device_name(0) == settings["required_gpu_name"], "Unexpected reporting GPU"
    )
    for package, expected in settings["versions"].items():
        actual = torch.version.cuda if package == "cuda" else version(package)
        require(actual == expected, f"Pinned {package} mismatch: {actual}")
    require(torch.backends.cudnn.version() == settings["cudnn_version"], "cuDNN version differs")
    installed = Path(ultralytics.__file__).parent
    sources = (
        "engine/model.py",
        "engine/predictor.py",
        "nn/backends/pytorch.py",
        "data/loaders.py",
        "models/yolo/detect/predict.py",
        "utils/nms.py",
        "utils/ops.py",
        "utils/torch_utils.py",
        "cfg/default.yaml",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: version(p)
            for p in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "pycocotools")
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "bfloat16_supported": True,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "installed_ultralytics_sha256": {p: digest(installed / p) for p in sources},
        "source_sha256": digest(Path(__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": subprocess.list2cmdline(
            [sys.executable, "-m", "src.analyze_yolo_numerical_path", *sys.argv[1:]]
        ),
    }


def forbid_training(*_args: Any, **_kwargs: Any) -> None:
    """Fail before any Ultralytics training API can execute."""
    raise RuntimeError("Training is forbidden in numerical-path sensitivity")


@contextmanager
def inference_only() -> Iterator[None]:
    """Guard the public training entry even if a future adapter calls it."""
    from ultralytics.engine.model import Model

    with patch.object(Model, "train", forbid_training):
        yield


def setup_runtime(config: dict[str, Any], seed: int) -> Any:
    """Apply the exact external deterministic numerical controls."""
    import torch

    settings = config["runtime"]
    report = configure_reproducibility(
        seed, deterministic=settings["deterministic"], allow_tf32=settings["allow_tf32"]
    )
    torch.set_num_threads(settings["torch_threads"])
    torch.cuda.set_device(torch.device(settings["device"]))
    return report


def prepare_predictor(
    config: dict[str, Any], checked: dict[str, Any], spec: Any
) -> tuple[Any, dict[str, Any]]:
    """Initialize exactly as external inference, then check effective state."""
    import torch

    model_config = load_yolo_config(spec.training_config)
    require(
        model_config.model.input_size == config["preprocessing"]["input_size"],
        "Native size mismatch",
    )
    predictor = Predictor(spec, checked["phase"], model_config, checked["dataset"])
    predictor.predict(0, None)  # native CPU fusion before installing the activation hook
    observed = predictor.verify_amp(0)
    require(observed == config["runtime"]["activation_dtype"], "Convolution dtype differs")
    model = predictor.model.model
    require(not any(m.training for m in model.modules()), "A module entered training mode")
    require(
        not any(p.requires_grad for p in model.parameters()),
        "Inference parameters require gradients",
    )
    require(
        torch.are_deterministic_algorithms_enabled() and not torch.backends.cudnn.benchmark,
        "Determinism changed",
    )
    require(
        not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
        "TF32 policy changed",
    )
    require(predictor.model.predictor.args.quantize is None, "Native quantization changed")
    require(predictor.model.predictor.args.rect is True, "Native letterbox policy changed")
    return predictor, {
        "convolution_dtype": observed,
        "weight_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "training": False,
        "parameters_require_grad": False,
        "autocast_device": "cuda",
        "autocast_dtype": model_config.runtime.amp_dtype,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "native_quantize": None,
        "native_rect": True,
    }


def preflight(config: dict[str, Any], *, probe: bool = True) -> dict[str, Any]:
    """Read-only gate plus a two-call activation probe; never create output paths."""
    require(
        not Path(config["outputs"]["root"]).exists(),
        "Versioned output already exists; never overwrite",
    )
    checked = check_inputs(config)
    environment = runtime_evidence(config)
    evidence = None
    if probe:
        setup_runtime(config, checked["runs"][0].seed)
        with inference_only():
            predictor, evidence = prepare_predictor(config, checked, checked["runs"][0])
        del predictor
        gc.collect()
    return {
        "status": "passed",
        "runs": len(checked["runs"]),
        "images_per_run": len(checked["dataset"]),
        "model_image_evaluations": len(checked["runs"]) * len(checked["dataset"]),
        "expected_prediction_bundles": len(checked["runs"]),
        "input_bindings_verified": len(config["input_sha256"]),
        "environment": environment,
        "activation_probe": evidence,
        "training": False,
        "threshold_selection": False,
        "external_inference": False,
    }


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float | None]:
    """Flatten a deterministic numeric schema without pooling detections."""
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            if key not in ("seed", "selection_seeds", "detection_count_histogram"):
                result.update(numeric_leaves(value[key], f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(numeric_leaves(child, f"{prefix}.{index}"))
        return result
    if value is None or isinstance(value, (int, float)):
        return {prefix: value}
    return {}


def distribution(values: list[float], quantiles: list[float]) -> dict[str, Any]:
    """Describe finite paired differences; undefined empty support stays null."""
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": len(a),
        "mean": float(a.mean()) if len(a) else None,
        "min": float(a.min()) if len(a) else None,
        "max": float(a.max()) if len(a) else None,
        "quantiles": {str(q): float(np.quantile(a, q)) if len(a) else None for q in quantiles},
    }


def candidate_agreement(
    before: list[ImagePrediction], after: list[ImagePrediction], settings: dict[str, Any]
) -> dict[str, Any]:
    """Greedy IoU candidate correspondence, never forcing unmatched boxes to pair."""
    require(
        [p.image_id for p in before] == [p.image_id for p in after], "Agreement image order differs"
    )
    matched, count_equal = 0, 0
    old_scores, new_scores, coordinates, overlaps = [], [], [], []
    crossing = {
        str(t): {
            "matched_up": 0,
            "matched_down": 0,
            "unmatched_old_above": 0,
            "unmatched_new_above": 0,
        }
        for t in settings["key_thresholds"]
    }
    for a, b in zip(before, after, strict=True):
        count_equal += len(a.scores) == len(b.scores)
        x, y = a.boxes_xyxy, b.boxes_xyxy
        area_x = np.prod(x[:, 2:] - x[:, :2], axis=1)
        area_y = np.prod(y[:, 2:] - y[:, :2], axis=1)
        intersection = np.prod(
            np.maximum(
                0,
                np.minimum(x[:, None, 2:], y[None, :, 2:])
                - np.maximum(x[:, None, :2], y[None, :, :2]),
            ),
            axis=2,
        )
        iou = intersection / (area_x[:, None] + area_y[None, :] - intersection)
        indices = np.argwhere(
            (iou >= settings["minimum_iou"]) & (a.labels[:, None] == b.labels[None, :])
        )
        pairs = sorted(
            ((float(iou[i, j]), int(i), int(j)) for i, j in indices),
            key=lambda p: (-p[0], p[1], p[2]),
        )
        used_a, used_b = set(), set()
        for overlap, i, j in pairs:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += 1
            old_scores.append(float(a.scores[i]))
            new_scores.append(float(b.scores[j]))
            coordinates.extend(np.abs(x[i] - y[j]).tolist())
            overlaps.append(overlap)
            for t in settings["key_thresholds"]:
                crossing[str(t)]["matched_up"] += int(a.scores[i] < t <= b.scores[j])
                crossing[str(t)]["matched_down"] += int(b.scores[j] < t <= a.scores[i])
        for t in settings["key_thresholds"]:
            crossing[str(t)]["unmatched_old_above"] += sum(
                s >= t for i, s in enumerate(a.scores) if i not in used_a
            )
            crossing[str(t)]["unmatched_new_above"] += sum(
                s >= t for j, s in enumerate(b.scores) if j not in used_b
            )
    n_old, n_new = sum(len(p.scores) for p in before), sum(len(p.scores) for p in after)
    rho = (
        float(spearmanr(old_scores, new_scores).statistic)
        if len(set(old_scores)) > 1 and len(set(new_scores)) > 1
        else None
    )
    for value in crossing.values():
        for key in value:
            value[key] = int(value[key])
        value["matched_up_fraction"] = value["matched_up"] / matched if matched else None
        value["matched_down_fraction"] = value["matched_down"] / matched if matched else None
    delta = np.asarray(new_scores) - np.asarray(old_scores)
    return {
        "matching_rule": settings,
        "before_candidates": n_old,
        "after_candidates": n_new,
        "matched_candidates": matched,
        "disappeared_candidates": n_old - matched,
        "appeared_candidates": n_new - matched,
        "disappeared_fraction": (n_old - matched) / n_old if n_old else None,
        "appeared_fraction": (n_new - matched) / n_new if n_new else None,
        "images_equal_count": int(count_equal),
        "fraction_images_equal_count": count_equal / len(before),
        "box_absolute_coordinate_difference_pixels": distribution(
            coordinates, settings["quantiles"]
        ),
        "matched_iou": distribution(overlaps, settings["quantiles"]),
        "score_difference": distribution(delta.tolist(), settings["quantiles"]),
        "absolute_score_difference": distribution(np.abs(delta).tolist(), settings["quantiles"]),
        "spearman_matched_scores": rho,
        "threshold_crossings": crossing,
    }


def evaluate_pair(
    config: dict[str, Any], run: dict[str, Any], after: list[ImagePrediction], targets: list[Any]
) -> dict[str, Any]:
    """Replay existing evaluators, applying frozen cutoffs without selection."""
    before = deserialize(read_gzip(Path(run["fp32_froc"]))["predictions"])
    metrics = {}
    for name, predictions in (("fp32", before), ("bfloat16", after)):
        result, _curve = compute_metrics(predictions, targets, config, "yolo11s", run["seed"])
        shared = load_phase5_config(config["inputs"]["phase5_config"]).evaluation
        result["shared_operating_point"] = evaluate_operating_point(
            predictions,
            targets,
            class_ids=tuple(config["ontology"]["canonical_classes"]),
            score_threshold=shared.score_threshold,
            iou_threshold=shared.match_iou_threshold,
            max_detections=shared.max_detections,
        )["overall"]
        for support in result["score_summaries"].values():
            support["detections_per_image"] = support["detection_count"] / len(targets)
            support["zero_detection_fraction"] = support["zero_detection_images"] / len(targets)
        total = result["score_summaries"]["all_retained_candidates"]["detection_count"]
        result["threshold_emissions"] = {}
        for threshold in config["agreement"]["key_thresholds"]:
            count = sum(int(np.count_nonzero(p.scores >= threshold)) for p in predictions)
            result["threshold_emissions"][str(threshold)] = {
                "detections": count,
                "proportion_of_retained_candidates": count / total if total else None,
            }
        metrics[name] = result
    historical_ap = read_gzip(Path(run["fp32_ap"]))["coco"]
    for metric in ("ap50", "ap50_95"):
        require(
            abs(metrics["fp32"]["ap"][metric] - historical_ap[metric])
            <= config["evaluation"]["froc"]["numeric_tolerance"],
            "Historical AP replay differs",
        )
    return {
        "seed": run["seed"],
        **metrics,
        "agreement": candidate_agreement(before, after, config["agreement"]),
    }


def make_tables(pairs: list[dict[str, Any]], ddof: int) -> tuple[list[dict[str, Any]], ...]:
    """Per-run deltas and equal-run aggregates with deterministic row order."""
    per_run, agreement = [], []
    for pair in pairs:
        old, new = numeric_leaves(pair["fp32"]), numeric_leaves(pair["bfloat16"])
        require(old.keys() == new.keys(), "Metric schemas differ")
        for key in old:
            delta = new[key] - old[key] if old[key] is not None and new[key] is not None else None
            per_run.append(
                {
                    "seed": pair["seed"],
                    "metric": key,
                    "fp32": old[key],
                    "bfloat16": new[key],
                    "delta": delta,
                }
            )
        agreement.extend(
            {"seed": pair["seed"], "metric": k, "value": v}
            for k, v in numeric_leaves(pair["agreement"]).items()
        )
    aggregate = []
    for key in sorted({r["metric"] for r in per_run}):
        rows = [r for r in per_run if r["metric"] == key]
        row = {"metric": key, "runs": len(rows)}
        for role in ("fp32", "bfloat16", "delta"):
            values = [r[role] for r in rows if r[role] is not None]
            row[f"{role}_defined_runs"] = len(values)
            row[f"{role}_mean"] = float(np.mean(values)) if values else None
            row[f"{role}_sample_sd"] = (
                float(np.std(values, ddof=ddof)) if len(values) > ddof else None
            )
        aggregate.append(row)
    return per_run, aggregate, agreement


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    """Exclusive UTF-8/LF CSV with a deterministic schema."""
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Collect exactly five internal bundles; evaluate without tuning or training."""
    import torch

    gate = preflight(config, probe=False)
    checked = check_inputs(config)
    root = Path(config["outputs"]["root"])
    root.mkdir()  # exclusive root ownership; failure artifacts require human review
    prediction_dir = root / config["outputs"]["prediction_directory"]
    prediction_dir.mkdir()
    report = setup_runtime(config, config["seeds"][0])
    environment_dir = root / config["outputs"]["environment_directory"]
    log_run_environment(environment_dir, report)
    protocol = {
        "config_sha256": digest(config_path),
        "config": config,
        "gate": gate,
        "started_utc": datetime.now(UTC).isoformat(),
        "image_sha256": checked["image_hashes"],
    }
    write_new(root / config["outputs"]["protocol"], protocol)
    pairs, artifacts = [], []
    with inference_only():
        for entry, spec in zip(config["runs"], checked["runs"], strict=True):
            setup_runtime(config, spec.seed)
            started, clock = datetime.now(UTC).isoformat(), time.perf_counter()
            predictor, evidence = prepare_predictor(config, checked, spec)
            predictions = []
            for index, target in enumerate(checked["targets"]):
                values = predictor.predict(index, None)
                predictions.append(
                    ImagePrediction(
                        image_id=target.image_id,
                        image_size=target.image_size,
                        boxes_xyxy=values["boxes"],
                        scores=values["scores"],
                        labels=values["labels"],
                    )
                )
                if (index + 1) % config["progress_every"] == 0:
                    print(
                        f"YOLO seed {spec.seed}: {index + 1}/{len(checked['targets'])}", flush=True
                    )
            require(
                digest(entry["checkpoint"]) == entry["sha256"],
                "Checkpoint changed during inference",
            )
            bundle = {
                "schema_version": config["schema_version"],
                "analysis_id": config["analysis_id"],
                "detector": "yolo11s",
                "seed": spec.seed,
                "split": "test",
                "checkpoint_sha256": entry["sha256"],
                "config_sha256": digest(config_path),
                "annotation_sha256": config["input_sha256"][config["inputs"]["annotations"]],
                "manifest_sha256": config["input_sha256"][config["inputs"]["test_manifest"]],
                "environment": gate["environment"],
                "numerical_path": evidence,
                "evaluation": config["evaluation"],
                "started_utc": started,
                "completed_utc": datetime.now(UTC).isoformat(),
                "inference_seconds": time.perf_counter() - clock,
                "predictions": [_serialize_prediction(p) for p in predictions],
            }
            path = prediction_dir / f"yolo11s_seed{spec.seed}_test_predictions.json.gz"
            write_new(path, bundle, compressed=True)
            artifacts.append({"path": path.as_posix(), "sha256": digest(path)})
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
            pairs.append(evaluate_pair(config, entry, predictions, checked["targets"]))
    tables = make_tables(pairs, config["statistics"]["standard_deviation_ddof"])
    for name, rows in zip(
        ("per_run_table", "aggregate_table", "agreement_table"), tables, strict=True
    ):
        path = root / config["outputs"][name]
        write_csv_new(path, rows)
        artifacts.append({"path": path.as_posix(), "sha256": digest(path)})
    # Recheck frozen inputs and source image bytes after all work, before acceptance.
    after = check_inputs(config)
    require(after["image_hashes"] == checked["image_hashes"], "Source images changed")
    for path in [root / config["outputs"]["protocol"], *sorted(environment_dir.iterdir())]:
        artifacts.append({"path": path.as_posix(), "sha256": digest(path)})
    summary = {
        "schema_version": config["schema_version"],
        "analysis_id": config["analysis_id"],
        "status": "complete_secondary_sensitivity",
        "config_sha256": digest(config_path),
        "source_sha256": digest(Path(__file__)),
        "completed_utc": datetime.now(UTC).isoformat(),
        "gate": gate,
        "paired_runs": pairs,
        "aggregate": tables[1],
        "artifacts": artifacts,
        "training_performed": False,
        "threshold_selection_performed": False,
        "external_inference_performed": False,
    }
    write_new(root / config["outputs"]["summary"], summary)
    return {
        "status": summary["status"],
        "runs": len(pairs),
        "summary": str(root / config["outputs"]["summary"]),
    }


def verify(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """CPU-only full metric/agreement replay, without inference or writes."""
    checked = check_inputs(config)
    root = Path(config["outputs"]["root"])
    summary = read_json(root / config["outputs"]["summary"])
    require(summary["config_sha256"] == digest(config_path), "Sensitivity config changed")
    require(
        summary["source_sha256"] == digest(Path(__file__)), "Sensitivity implementation changed"
    )
    for artifact in summary["artifacts"]:
        require(
            digest(artifact["path"]) == artifact["sha256"],
            f"Output hash differs: {artifact['path']}",
        )
    protocol = read_json(root / config["outputs"]["protocol"])
    require(protocol["image_sha256"] == checked["image_hashes"], "Source image bytes differ")
    pairs = []
    for entry in config["runs"]:
        path = (
            root
            / config["outputs"]["prediction_directory"]
            / f"yolo11s_seed{entry['seed']}_test_predictions.json.gz"
        )
        bundle = read_gzip(path)
        require(
            bundle["seed"] == entry["seed"] and bundle["checkpoint_sha256"] == entry["sha256"],
            "Bundle run identity differs",
        )
        require(
            bundle["config_sha256"] == digest(config_path)
            and bundle["evaluation"] == json.loads(json.dumps(config["evaluation"])),
            "Bundle protocol differs",
        )
        require(
            bundle["numerical_path"]["convolution_dtype"] == config["runtime"]["activation_dtype"],
            "Bundle activation dtype differs",
        )
        pairs.append(
            evaluate_pair(config, entry, deserialize(bundle["predictions"]), checked["targets"])
        )
    require(pairs == summary["paired_runs"], "Metric or agreement replay differs")
    require(
        make_tables(pairs, config["statistics"]["standard_deviation_ddof"])[1]
        == summary["aggregate"],
        "Aggregate replay differs",
    )
    return {
        "status": "verified",
        "runs": len(pairs),
        "artifacts": len(summary["artifacts"]),
        "gpu_inference": False,
        "training": False,
    }


def main() -> None:
    """Expose only preflight, inference collection and read-only replay modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("preflight", "run", "verify"))
    args = parser.parse_args()
    config = load_config(args.config)
    result = (
        preflight(config)
        if args.mode == "preflight"
        else run(config, args.config)
        if args.mode == "run"
        else verify(config, args.config)
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
