"""Collect approved, lower-floor FROC prediction bundles without retraining."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluate import (
    Phase5Config,
    RunSpec,
    _atomic_csv,
    _atomic_json,
    _collect_faster_rcnn_predictions,
    _collect_yolo_predictions,
    _load_test_dataset,
    _recorded_training_config_sha256,
    _targets_from_dataset,
    _write_prediction_bundle,
    evaluate_prediction_records,
    load_and_validate_training_configs,
    load_phase5_config,
    sha256_file,
)

INVENTORY_FIELDS = (
    "detector",
    "seed",
    "candidate_score_floor",
    "image_count",
    "prediction_count",
    "images_with_predictions",
    "unique_score_count",
    "minimum_emitted_score",
    "maximum_emitted_score",
    "checkpoint",
    "checkpoint_sha256",
    "prediction_bundle",
    "prediction_bundle_sha256",
)

BOUNDARY_FIELDS = (
    "detector",
    "seed_count",
    "candidate_score_floor",
    "prediction_count_mean",
    "prediction_count_std",
    "minimum_emitted_score_min",
    "maximum_emitted_score_max",
)

CONTRACT_FIELDS = (
    "experiment_id",
    "source_experiment_id",
    "approved_candidate_score_floor",
    "source_candidate_score_floor",
    "score_threshold",
    "match_iou_threshold",
    "nms_iou_threshold",
    "max_detections",
    "seed_count",
    "model_count",
    "images_per_model",
    "model_image_evaluations",
    "training_performed",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected gzip JSON object: {path}")
    return payload


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _run_grid(config: Phase5Config) -> dict[tuple[str, int], RunSpec]:
    return {(run.detector, run.seed): run for run in config.runs}


def validate_protocol_contract(lower: Phase5Config, source: Phase5Config) -> None:
    """Require an inference-only floor change and byte-identical run scope."""

    if lower.source_path.resolve() == source.source_path.resolve():
        raise ValueError("lower-floor and source configs must be separate versioned files")
    if lower.seeds != source.seeds or lower.split != source.split:
        raise ValueError("lower-floor seed/split scope differs from the source evaluator")
    if lower.runtime != source.runtime:
        raise ValueError("lower-floor runtime contract differs from the source evaluator")
    if lower.evaluation.coco_minimum_score >= source.evaluation.coco_minimum_score:
        raise ValueError("approved candidate floor must be lower than the source floor")
    for field in (
        "score_threshold",
        "match_iou_threshold",
        "nms_iou_threshold",
        "max_detections",
    ):
        if getattr(lower.evaluation, field) != getattr(source.evaluation, field):
            raise ValueError(f"lower-floor inference changed forbidden setting: {field}")
    source_runs = _run_grid(source)
    lower_runs = _run_grid(lower)
    if set(lower_runs) != set(source_runs):
        raise ValueError("lower-floor run grid differs from the source evaluator")
    for key, lower_run in lower_runs.items():
        source_run = source_runs[key]
        if lower_run.model_dump(mode="json") != source_run.model_dump(mode="json"):
            raise ValueError(f"lower-floor run provenance differs from source: {key}")
    lower_outputs = {lower.resolve(path).resolve() for path in lower.outputs.model_dump().values()}
    source_outputs = {
        source.resolve(path).resolve() for path in source.outputs.model_dump().values()
    }
    if lower_outputs & source_outputs:
        raise ValueError("lower-floor outputs overlap frozen source outputs")


def _validate_source_summary(source: Phase5Config) -> tuple[Path, dict[str, Any]]:
    summary_path = source.resolve(source.outputs.summary_json)
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("source Phase 5 summary is not complete")
    if summary.get("config_sha256") != sha256_file(source.source_path):
        raise ValueError("source Phase 5 config hash mismatch")
    if summary.get("experiment_id") != source.experiment_id:
        raise ValueError("source Phase 5 experiment identity mismatch")
    if summary.get("evaluation") != source.evaluation.model_dump(mode="json"):
        raise ValueError("source Phase 5 evaluation contract mismatch")
    return summary_path, summary


def _validate_prior_summary(prior: Phase5Config) -> tuple[Path, dict[str, Any]]:
    """Verify the immediately preceding candidate-collection evidence."""

    summary_path, summary = _validate_source_summary(prior)
    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != len(prior.runs):
        raise ValueError("prior candidate-collection summary lacks the complete run grid")
    expected = set(_run_grid(prior))
    observed: set[tuple[str, int]] = set()
    for record in runs:
        key = str(record.get("detector")), int(record.get("seed"))
        comparison = record.get("comparison_row")
        if key in observed or key not in expected or not isinstance(comparison, dict):
            raise ValueError("prior candidate-collection run provenance is invalid")
        bundle_path = prior.resolve(Path(str(comparison.get("prediction_bundle"))))
        if record.get("prediction_bundle_sha256") != sha256_file(bundle_path):
            raise ValueError(f"prior prediction-bundle hash mismatch: {key}")
        observed.add(key)
    if observed != expected:
        raise ValueError("prior candidate-collection summary lacks the exact run grid")
    return summary_path, summary


def _validate_release_manifest(
    path: Path,
    source: Phase5Config,
    source_summary_path: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("checkpoint release manifest schema mismatch")
    summary_record = manifest.get("phase5_summary")
    if not isinstance(summary_record, dict):
        raise ValueError("checkpoint release manifest lacks Phase 5 provenance")
    if source.resolve(Path(str(summary_record.get("path")))).resolve() != source_summary_path:
        raise ValueError("checkpoint release manifest points to another Phase 5 summary")
    if summary_record.get("sha256") != sha256_file(source_summary_path):
        raise ValueError("checkpoint release manifest Phase 5 hash mismatch")
    records = manifest.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("checkpoint release manifest lacks checkpoint records")
    by_run: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = str(record.get("detector")), int(record.get("seed"))
        if key in by_run:
            raise ValueError(f"duplicate checkpoint release record: {key}")
        by_run[key] = record
    expected = set(_run_grid(source))
    if set(by_run) != expected:
        raise ValueError("checkpoint release manifest does not contain the exact ten-run grid")
    for key, run in _run_grid(source).items():
        record = by_run[key]
        checkpoint_path = source.resolve(run.checkpoint)
        if source.resolve(Path(str(record.get("source_path")))).resolve() != checkpoint_path:
            raise ValueError(f"checkpoint release path mismatch: {key}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        if int(record.get("size_bytes", -1)) != checkpoint_path.stat().st_size:
            raise ValueError(f"checkpoint release size mismatch: {key}")
        if record.get("sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"checkpoint release hash mismatch: {key}")
        config_record = record.get("config")
        if not isinstance(config_record, dict):
            raise ValueError(f"checkpoint release config record missing: {key}")
        training_path = source.resolve(run.training_config)
        if source.resolve(Path(str(config_record.get("path")))).resolve() != training_path:
            raise ValueError(f"checkpoint release training-config path mismatch: {key}")
        if config_record.get("sha256") != sha256_file(training_path):
            raise ValueError(f"checkpoint release training-config hash mismatch: {key}")
    return by_run


def _score_inventory(
    payload: Mapping[str, Any],
    *,
    path: Path,
    root: Path,
) -> dict[str, Any]:
    records = payload.get("predictions")
    if not isinstance(records, list):
        raise ValueError(f"prediction bundle lacks predictions: {path}")
    all_scores: list[float] = []
    images_with_predictions = 0
    for item in records:
        scores = [float(value) for value in item.get("scores", [])]
        if scores:
            images_with_predictions += 1
            all_scores.extend(scores)
    evaluation = payload["evaluation"]
    floor = float(evaluation["coco_minimum_score"])
    if any(not math.isfinite(score) or score < floor for score in all_scores):
        raise ValueError(f"prediction bundle contains invalid scores: {path}")
    return {
        "detector": str(payload["detector"]),
        "seed": int(payload["seed"]),
        "candidate_score_floor": floor,
        "image_count": len(records),
        "prediction_count": len(all_scores),
        "images_with_predictions": images_with_predictions,
        "unique_score_count": len(set(all_scores)),
        "minimum_emitted_score": min(all_scores) if all_scores else "",
        "maximum_emitted_score": max(all_scores) if all_scores else "",
        "prediction_bundle": _relative(path, root),
        "prediction_bundle_sha256": sha256_file(path),
    }


def _validate_bundle(
    path: Path,
    *,
    run: RunSpec,
    config: Phase5Config,
    annotation_sha256: str,
    checkpoint_sha256: str,
    image_count: int,
) -> dict[str, Any]:
    payload = _read_bundle(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("detector") != run.detector
        or int(payload.get("seed")) != run.seed
        or payload.get("split") != config.split
    ):
        raise ValueError(f"lower-floor bundle identity mismatch: {path}")
    if payload.get("evaluation") != config.evaluation.model_dump(mode="json"):
        raise ValueError(f"lower-floor bundle evaluator mismatch: {path}")
    if payload.get("annotation_sha256") != annotation_sha256:
        raise ValueError(f"lower-floor bundle annotation mismatch: {path}")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"lower-floor bundle checkpoint mismatch: {path}")
    inventory = _score_inventory(payload, path=path, root=config.project_root)
    if inventory["image_count"] != image_count:
        raise ValueError(f"lower-floor bundle image count mismatch: {path}")
    inventory["checkpoint"] = _relative(config.resolve(run.checkpoint), config.project_root)
    inventory["checkpoint_sha256"] = checkpoint_sha256
    return inventory


def preflight(
    lower: Phase5Config,
    source: Phase5Config,
    checkpoint_manifest_path: Path,
    prior: Phase5Config | None = None,
) -> dict[str, Any]:
    """Hash-check the approved ten-checkpoint inference scope before GPU work."""

    import torch

    validate_protocol_contract(lower, source)
    prior_summary_path: Path | None = None
    if prior is not None:
        validate_protocol_contract(lower, prior)
        prior_summary_path, _prior_summary = _validate_prior_summary(prior)
    source_summary_path, source_summary = _validate_source_summary(source)
    release = _validate_release_manifest(checkpoint_manifest_path, source, source_summary_path)
    training_configs = load_and_validate_training_configs(source)
    faster_reference = training_configs[("faster_rcnn", source.seeds[0])]
    dataset = _load_test_dataset(source, faster_reference)
    annotation_path = dataset.annotation_file.resolve()
    if source_summary.get("test_annotation_sha256") != sha256_file(annotation_path):
        raise ValueError("source Phase 5 annotation hash mismatch")
    if len(dataset) != int(source_summary.get("image_count", -1)):
        raise ValueError("source Phase 5 image count mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("approved lower-floor inference requires CUDA")
    free_bytes = shutil.disk_usage(lower.project_root).free
    return {
        "status": "ready",
        "experiment_id": lower.experiment_id,
        "source_experiment_id": source.experiment_id,
        "source_config_sha256": sha256_file(source.source_path),
        "source_summary_sha256": sha256_file(source_summary_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "checkpoints_verified": len(release),
        "checkpoint_bytes_verified": sum(
            source.resolve(run.checkpoint).stat().st_size for run in source.runs
        ),
        "candidate_score_floor": lower.evaluation.coco_minimum_score,
        "source_candidate_score_floor": (
            prior.evaluation.coco_minimum_score
            if prior is not None
            else source.evaluation.coco_minimum_score
        ),
        "prior_config_sha256": sha256_file(prior.source_path) if prior is not None else None,
        "prior_summary_sha256": (
            sha256_file(prior_summary_path) if prior_summary_path is not None else None
        ),
        "seeds": list(lower.seeds),
        "models": len(lower.runs),
        "images_per_model": len(dataset),
        "model_image_evaluations": len(lower.runs) * len(dataset),
        "annotations": sum(len(record.annotations) for record in dataset.records),
        "annotation_sha256": sha256_file(annotation_path),
        "cuda_device": torch.cuda.get_device_name(0),
        "free_disk_bytes": free_bytes,
        "performs_training": False,
        "performs_checkpoint_loading": False,
        "performs_model_inference": False,
    }


def _aggregate_boundaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["detector"])].append(row)
    output: list[dict[str, Any]] = []
    for detector in sorted(grouped):
        selected = grouped[detector]
        counts = np.asarray([float(row["prediction_count"]) for row in selected])
        minima = [float(row["minimum_emitted_score"]) for row in selected]
        maxima = [float(row["maximum_emitted_score"]) for row in selected]
        output.append(
            {
                "detector": detector,
                "seed_count": len(selected),
                "candidate_score_floor": selected[0]["candidate_score_floor"],
                "prediction_count_mean": float(np.mean(counts)),
                "prediction_count_std": float(np.std(counts, ddof=1)),
                "minimum_emitted_score_min": min(minima),
                "maximum_emitted_score_max": max(maxima),
            }
        )
    return output


def run_collection(
    lower: Phase5Config,
    source: Phase5Config,
    checkpoint_manifest_path: Path,
    prior: Phase5Config | None = None,
) -> dict[str, Any]:
    """Run approved inference and atomically materialize restart-safe bundles."""

    from src.utils.seed import initialize_reproducibility, seed_everything

    readiness = preflight(lower, source, checkpoint_manifest_path, prior)
    training_configs = load_and_validate_training_configs(source)
    faster_reference = training_configs[("faster_rcnn", source.seeds[0])]
    dataset = _load_test_dataset(source, faster_reference)
    targets = _targets_from_dataset(dataset)
    annotation_path = dataset.annotation_file.resolve()
    annotation_sha256 = sha256_file(annotation_path)
    log_dir = lower.resolve(lower.outputs.log_dir)
    initialize_reproducibility(lower.seeds[0], log_dir)
    progress_path = log_dir / "progress.json"
    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run in sorted(lower.runs, key=lambda item: (item.detector, item.seed)):
        checkpoint_path = lower.resolve(run.checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        bundle_path = lower.resolve(lower.outputs.prediction_bundles_dir) / (
            f"{run.detector}_seed{run.seed}_test_predictions.json.gz"
        )
        if bundle_path.exists():
            print(f"reusing verified {run.detector} seed {run.seed} bundle", flush=True)
            inventory = _validate_bundle(
                bundle_path,
                run=run,
                config=lower,
                annotation_sha256=annotation_sha256,
                checkpoint_sha256=checkpoint_sha256,
                image_count=len(dataset),
            )
        else:
            print(f"inferring {run.detector} seed {run.seed} at approved floor", flush=True)
            seed_everything(run.seed)
            model_config = training_configs[(run.detector, run.seed)]
            if run.detector == "faster_rcnn":
                predictions, _inference_seconds = _collect_faster_rcnn_predictions(
                    run,
                    lower,
                    model_config,
                    dataset,
                    expected_config_sha256=_recorded_training_config_sha256(run, source),
                )
            else:
                predictions, _inference_seconds = _collect_yolo_predictions(
                    run, lower, model_config, dataset
                )
            metrics = evaluate_prediction_records(
                predictions,
                targets,
                category_names=dataset.category_names,
                settings=lower.evaluation,
            )
            _write_prediction_bundle(
                bundle_path,
                run=run,
                checkpoint_sha256=checkpoint_sha256,
                annotation_path=annotation_path,
                predictions=predictions,
                metrics=metrics,
                settings=lower.evaluation,
            )
            inventory = _validate_bundle(
                bundle_path,
                run=run,
                config=lower,
                annotation_sha256=annotation_sha256,
                checkpoint_sha256=checkpoint_sha256,
                image_count=len(dataset),
            )
            del predictions, metrics
        rows.append(inventory)
        comparison_row = {
            "detector": run.detector,
            "seed": run.seed,
            "prediction_bundle": inventory["prediction_bundle"],
            "checkpoint": inventory["checkpoint"],
            "checkpoint_sha256": checkpoint_sha256,
            "candidate_score_floor": lower.evaluation.coco_minimum_score,
            "image_count": len(dataset),
            "prediction_count": inventory["prediction_count"],
        }
        run_summaries.append(
            {
                "detector": run.detector,
                "seed": run.seed,
                "comparison_row": comparison_row,
                "prediction_bundle_sha256": inventory["prediction_bundle_sha256"],
            }
        )
        _atomic_json(
            progress_path,
            {
                "status": "in_progress",
                "experiment_id": lower.experiment_id,
                "config_sha256": sha256_file(lower.source_path),
                "completed_runs": run_summaries,
            },
        )

    rows.sort(key=lambda row: (str(row["detector"]), int(row["seed"])))
    run_summaries.sort(key=lambda row: (str(row["detector"]), int(row["seed"])))
    boundaries = _aggregate_boundaries(rows)
    inventory_path = _atomic_csv(
        lower.resolve(lower.outputs.per_seed_table), INVENTORY_FIELDS, rows
    )
    boundaries_path = _atomic_csv(
        lower.resolve(lower.outputs.mean_std_table), BOUNDARY_FIELDS, boundaries
    )
    protocol_source = prior if prior is not None else source
    contract_rows = [
        {
            "experiment_id": lower.experiment_id,
            "source_experiment_id": protocol_source.experiment_id,
            "approved_candidate_score_floor": lower.evaluation.coco_minimum_score,
            "source_candidate_score_floor": protocol_source.evaluation.coco_minimum_score,
            "score_threshold": lower.evaluation.score_threshold,
            "match_iou_threshold": lower.evaluation.match_iou_threshold,
            "nms_iou_threshold": lower.evaluation.nms_iou_threshold,
            "max_detections": lower.evaluation.max_detections,
            "seed_count": len(lower.seeds),
            "model_count": len(lower.runs),
            "images_per_model": len(dataset),
            "model_image_evaluations": len(lower.runs) * len(dataset),
            "training_performed": False,
        }
    ]
    contract_path = _atomic_csv(
        lower.resolve(lower.outputs.publication_table), CONTRACT_FIELDS, contract_rows
    )
    source_summary_path, source_summary = _validate_source_summary(source)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": lower.experiment_id,
        "config_path": _relative(lower.source_path, lower.project_root),
        "config_sha256": sha256_file(lower.source_path),
        "source_identity": {
            _relative(Path(__file__), lower.project_root): sha256_file(Path(__file__)),
            "src/evaluate.py": sha256_file(lower.project_root / "src/evaluate.py"),
        },
        "approval": {
            "status": "user_approved",
            "scope": "Batch 42 inference-only sensitivity protocol",
            "approved_candidate_score_floor": lower.evaluation.coco_minimum_score,
        },
        "source_phase5": {
            "config": _artifact(source.source_path, lower.project_root),
            "summary": _artifact(source_summary_path, lower.project_root),
            "experiment_id": source.experiment_id,
            "candidate_score_floor": source.evaluation.coco_minimum_score,
        },
        "prior_candidate_collection": (
            {
                "config": _artifact(prior.source_path, lower.project_root),
                "summary": _artifact(prior.resolve(prior.outputs.summary_json), lower.project_root),
                "experiment_id": prior.experiment_id,
                "candidate_score_floor": prior.evaluation.coco_minimum_score,
            }
            if prior is not None
            else None
        ),
        "checkpoint_release_manifest": _artifact(checkpoint_manifest_path, lower.project_root),
        "split": lower.split,
        "test_annotation_path": annotation_path.as_posix(),
        "test_annotation_sha256": annotation_sha256,
        "image_count": len(dataset),
        "annotation_count": sum(len(record.annotations) for record in dataset.records),
        "category_names": {str(key): value for key, value in dataset.category_names.items()},
        "evaluation": lower.evaluation.model_dump(mode="json"),
        "protocol": {
            "changed_parameter": "candidate score floor only",
            "source_candidate_score_floor": protocol_source.evaluation.coco_minimum_score,
            "approved_candidate_score_floor": lower.evaluation.coco_minimum_score,
            "unchanged": [
                "test images and annotations",
                "Lung Opacity class",
                "trained checkpoint bytes",
                "image preprocessing and detector input resolution",
                "IoU 0.50 matching",
                "class-aware native NMS at IoU 0.50",
                "maximum 100 detections per image",
                "five predeclared seeds including 271",
            ],
            "faster_rcnn_adapter": (
                "model emits its configured top 100 post-NMS detections; the shared bundle "
                "filter changed from the immediately prior floor to the approved floor"
            ),
            "yolo11s_adapter": (
                "Ultralytics predict conf changed from the immediately prior floor to the "
                "approved floor"
            ),
            "performs_training": False,
            "performs_checkpoint_loading": True,
            "performs_model_inference": True,
        },
        "statistics": {
            "seeds": list(lower.seeds),
            "n": len(lower.seeds),
            "n_definition": "same predeclared attempted training seeds",
            "standard_deviation": "sample",
            "ddof": lower.runtime.statistics_ddof,
        },
        "runs": run_summaries,
        "score_boundaries": boundaries,
        "artifacts": {
            "prediction_inventory": _artifact(inventory_path, lower.project_root),
            "score_boundaries": _artifact(boundaries_path, lower.project_root),
            "inference_contract": _artifact(contract_path, lower.project_root),
            "prediction_bundles": [
                {
                    "detector": row["detector"],
                    "seed": row["seed"],
                    "path": row["prediction_bundle"],
                    "sha256": row["prediction_bundle_sha256"],
                }
                for row in rows
            ],
        },
        "preflight": {key: value for key, value in readiness.items() if key != "free_disk_bytes"},
        "source_test_split_accessed": bool(source_summary.get("test_split_accessed")),
        "test_split_accessed": True,
    }
    summary_path = _atomic_json(lower.resolve(lower.outputs.summary_json), summary)
    _atomic_json(
        progress_path,
        {
            "status": "complete",
            "experiment_id": lower.experiment_id,
            "config_sha256": sha256_file(lower.source_path),
            "completed_runs": run_summaries,
            "summary": _relative(summary_path, lower.project_root),
            "summary_sha256": sha256_file(summary_path),
        },
    )
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix()}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lower = load_phase5_config(args.config)
    source = load_phase5_config(args.source_config)
    prior = load_phase5_config(args.prior_config) if args.prior_config is not None else None
    checkpoint_manifest = args.checkpoint_manifest.resolve()
    if args.mode == "run":
        run_collection(lower, source, checkpoint_manifest, prior)
    else:
        print(
            json.dumps(
                preflight(lower, source, checkpoint_manifest, prior), indent=2, sort_keys=True
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
