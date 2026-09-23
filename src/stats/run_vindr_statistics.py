"""Batch 50: offline, hash-gated cross-dataset transportability analysis.

No model loading, checkpoint selection, threshold fitting or ontology changes.
Only aggregate tables/figures are public; observation-linked evidence stays private.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.verify_scientific_artifacts import verify_manifest
from src.data.prepare_vindr import ensure_private, verify_freeze
from src.evaluate_vindr_external import deserialize, read_gzip, score_summary
from src.stats.paired import build_patient_clusters, json_number, stable_rng_seed
from src.stats.run_statistics import load_coco_targets, load_patient_group_map
from src.stats.transportability import (
    RunEvidence,
    bootstrap,
    build_ap_cache,
    build_froc_cache,
    classify_change,
    interval_rows,
    operating_counts,
)


def read_json(path: str | Path) -> Any:
    """Read an input or existing result without changing it."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read the requested config; scientific settings remain inherited."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def artifact(path: str | Path) -> dict[str, Any]:
    """Record portable repository-relative paths and exact byte identities."""
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": path.resolve().relative_to(Path.cwd()).as_posix(),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def require(condition: bool, reason: str) -> None:
    """Fail closed if a prerequisite or a metric replay disagrees."""
    if not condition:
        raise ValueError(reason)


def verify_artifact(entry: dict[str, Any]) -> None:
    """Reject missing or changed evidence before analysis or rendering."""
    current = artifact(entry["path"])
    require(current["sha256"] == entry["sha256"], f"Hash mismatch: {entry['path']}")
    if "size_bytes" in entry:
        require(current["size_bytes"] == entry["size_bytes"], f"Size mismatch: {entry['path']}")


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with explicit LF bytes, avoiding Windows/Git hash drift."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_number(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregate rows with explicit missing values and stable column ordering."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(json_number(rows))


def metric_names(protocol: dict[str, Any]) -> list[str]:
    """Name the two AP, five budget and four historical-threshold endpoints."""
    return [
        "ap50",
        "ap50_95",
        *[f"froc_{b:g}" for b in protocol["evaluation"]["froc"]["fp_per_image_budgets"]],
        "primary_precision",
        "primary_recall",
        "primary_f1",
        "primary_fp_per_image",
    ]


def preflight(config_path: Path) -> dict[str, Any]:
    """Verify complete external receipts, all frozen hashes and internal comparison sources."""
    config = read_yaml(config_path)
    protocol_path = Path(config["protocol_config"])
    protocol = read_yaml(protocol_path)
    frozen = verify_freeze(protocol_path, protocol)
    stats = protocol["statistics"]
    require(
        stats["run_resampling"] == "independent_within_detector"
        and stats["observation_resampling"] == "paired_across_detectors_and_runs"
        and stats["hypothesis_tests"] == "none_planned"
        and stats["defensible_patient_group_id"] is None,
        "Unsupported departure from frozen estimands",
    )
    require(config["conditional_seed"] in protocol["models"]["seeds"], "Missing primary seed")
    require(
        f"checkpoint_conditional_seed{config['conditional_seed']}" == stats["secondary_estimand"],
        "Conditional checkpoint differs from the frozen protocol",
    )
    private = Path(config["outputs"]["private_root"])
    ensure_private(
        private / config["outputs"]["bootstrap_draws"], Path(protocol["outputs"]["private_root"])
    )
    external = read_json(config["external_summary"])
    operation = read_yaml(config["inference_config"])
    require(external["status"] == "complete", "Batch 49 is incomplete")
    require(external["gate"]["freeze_files_verified"] == len(frozen), "Freeze inventory mismatch")
    expected = {
        (d, s) for d in protocol["models"]["detectors"] for s in protocol["models"]["seeds"]
    }
    keys = [(r["metadata"]["detector"], r["metadata"]["seed"]) for r in external["runs"]]
    require(
        len(keys) == len(expected) and set(keys) == expected, "Incomplete/duplicate external runs"
    )
    paths = {
        str(config_path),
        config["protocol_config"],
        protocol["freeze_manifest"],
        config["inference_config"],
        config["external_summary"],
        *frozen,
        *config["source_files"],
        *operation["source_files"],
    }
    execution = external["execution_receipt"]
    verify_artifact(execution)
    receipt = read_json(execution["path"])
    require(
        receipt["runs"] == external["runs"]
        and receipt["status"] == "complete"
        and receipt["gate"] == external["gate"],
        "Public/private execution disagreement",
    )
    paths.add(execution["path"])
    for p, h in external["gate"]["input_sha256"].items():
        verify_artifact({"path": p, "sha256": h})
        paths.add(p)
    for item in external["environment"]["files"]:
        verify_artifact(item)
        paths.add(item["path"])
    for r in external["runs"]:
        meta = r["metadata"]
        require(meta["input_sha256"] == external["gate"]["input_sha256"], "Run input mismatch")
        verify_artifact(meta["checkpoint"])
        paths.add(meta["checkpoint"]["path"])
        for item in r["artifacts"]:
            verify_artifact(item)
            paths.add(item["path"])
    internal = config["internal"]
    verify_manifest(Path(internal["scientific_manifest"]))
    ic = read_yaml(internal["statistics_config"])
    fc = read_yaml(internal["froc_config"])
    for field in ("bootstrap_resamples", "confidence_level", "run_resampling"):
        require(ic["analysis"][field] == stats[field], f"Internal inferential mismatch: {field}")
    require(ic["seed"] == stats["bootstrap_seed"], "Bootstrap seed principle changed")
    require(
        fc["analysis"]["fp_per_image_budgets"]
        == protocol["evaluation"]["froc"]["fp_per_image_budgets"]
        and fc["analysis"]["candidate_score_floor"]
        == protocol["evaluation"]["candidate_score_floor"],
        "Internal/external FROC support mismatch",
    )
    paths.update(
        internal[k]
        for k in (
            "statistics_config",
            "scientific_manifest",
            "evaluation_summary",
            "froc_config",
            "threshold_config",
            "threshold_per_run",
        )
    )
    paths.update(
        [
            fc["inputs"]["phase5_summary"],
            fc["outputs"]["summary_json"],
            ic["inputs"]["test_annotations"],
            ic["inputs"]["test_split_manifest"],
        ]
    )
    for summary_path in [internal["evaluation_summary"], fc["inputs"]["phase5_summary"]]:
        summary = read_json(summary_path)
        require(summary["status"] == "complete", "Internal inference incomplete")
        keys = [(r["detector"], r["seed"]) for r in summary["runs"]]
        require(len(keys) == len(expected) and set(keys) == expected, "Incomplete internal runs")
        require(
            artifact(ic["inputs"]["test_annotations"])["sha256"]
            == summary["test_annotation_sha256"],
            "Internal annotation hash mismatch",
        )
        for r in summary["runs"]:
            p = r["comparison_row"]["prediction_bundle"]
            verify_artifact({"path": p, "sha256": r["prediction_bundle_sha256"]})
            paths.add(p)
            external_run = next(
                e
                for e in external["runs"]
                if e["metadata"]["detector"] == r["detector"] and e["metadata"]["seed"] == r["seed"]
            )
            require(
                r["comparison_row"]["checkpoint_sha256"]
                == external_run["metadata"]["checkpoint"]["sha256"],
                "Checkpoint identity differs",
            )
    for policy in ("primary", "secondary"):
        paths.update(
            protocol["threshold_transport"][policy][k]
            for k in ("source_config", "source_summary", "source_table")
        )
    return {
        "config": config,
        "protocol": protocol,
        "external": external,
        "internal_statistics": ic,
        "internal_froc": fc,
        "inputs": [artifact(p) for p in sorted(paths)],
        "freeze_files_verified": len(frozen),
    }


def close(actual: Any, expected: Any, reason: str, tolerance: float) -> None:
    """Require canonical metric agreement before any bootstrap calculation."""
    require(bool(np.allclose(actual, expected, rtol=0, atol=tolerance, equal_nan=True)), reason)


def prepare_dataset(
    gate: dict[str, Any],
    dataset: str,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Any]:
    """Load one run at a time, verify point estimates and retain only numerical caches."""
    config, protocol = gate["config"], gate["protocol"]
    evaluation = protocol["evaluation"]
    budgets = evaluation["froc"]["fp_per_image_budgets"]
    tolerance = evaluation["froc"]["numeric_tolerance"]
    classes = tuple(protocol["ontology"]["canonical_classes"])
    if dataset == "external":
        annotation_path = protocol["outputs"]["local_annotations"]
        ap_runs = {
            (r["metadata"]["detector"], r["metadata"]["seed"]): r for r in gate["external"]["runs"]
        }
        froc_runs = ap_runs
        threshold_rows = []
    else:
        annotation_path = gate["internal_statistics"]["inputs"]["test_annotations"]
        ap_runs = {
            (r["detector"], r["seed"]): r
            for r in read_json(config["internal"]["evaluation_summary"])["runs"]
        }
        froc_runs = {
            (r["detector"], r["seed"]): r
            for r in read_json(gate["internal_froc"]["inputs"]["phase5_summary"])["runs"]
        }
        with Path(config["internal"]["threshold_per_run"]).open(encoding="utf-8", newline="") as s:
            threshold_rows = list(csv.DictReader(s))
    targets, categories = load_coco_targets(Path(annotation_path))
    require(categories == protocol["ontology"]["canonical_classes"], "Category contract mismatch")
    ids = tuple(t.image_id for t in targets)
    clusters = None
    if dataset == "internal":
        internal = gate["internal_statistics"]
        groups = load_patient_group_map(
            Path(internal["inputs"]["test_split_manifest"]),
            image_column=internal["analysis"]["manifest_image_column"],
            patient_group_column=internal["analysis"]["patient_group_column"],
        )
        clusters = build_patient_clusters(ids, groups)
    ones = np.ones(len(targets), dtype=np.int64)
    rows, scores, plots, detector_evidence = [], [], [], []
    for detector in protocol["models"]["detectors"]:
        runs = []
        for seed in protocol["models"]["seeds"]:
            key = (detector, seed)
            source = ap_runs[key]
            if dataset == "external":
                predictions = deserialize(
                    read_gzip(Path(source["artifacts"][0]["path"]))["predictions"]
                )
                lower_predictions = predictions
                expected_ap = source["metrics"]["ap"]
            else:
                predictions = deserialize(
                    read_gzip(Path(source["comparison_row"]["prediction_bundle"]))["predictions"]
                )
                lower_predictions = deserialize(
                    read_gzip(Path(froc_runs[key]["comparison_row"]["prediction_bundle"]))[
                        "predictions"
                    ]
                )
                expected_ap = source["metrics"]["coco"]
            ap = build_ap_cache(
                predictions,
                targets,
                class_ids=classes,
                minimum_score=evaluation["ap_minimum_score"],
                max_dets=evaluation["ap"]["max_dets"],
                iou_thresholds=evaluation["ap"]["iou_thresholds"],
                recall_points=evaluation["ap"]["recall_grid_points"],
            )
            kwargs = {
                "class_ids": classes,
                "iou": evaluation["matching_iou_threshold"],
                "cap": evaluation["max_detections_per_image"],
            }
            froc = build_froc_cache(
                lower_predictions, targets, floor=evaluation["candidate_score_floor"], **kwargs
            )
            op_cache = (
                froc
                if dataset == "external"
                else build_froc_cache(
                    predictions, targets, floor=evaluation["ap_minimum_score"], **kwargs
                )
            )
            primary = operating_counts(
                op_cache, protocol["threshold_transport"]["primary"]["thresholds"][detector]
            )
            evidence = RunEvidence(seed, ap, froc, primary)
            runs.append(evidence)
            point = evidence.evaluate(ones, budgets, tolerance)
            close(
                point[:2],
                [expected_ap["ap50"], expected_ap["ap50_95"]],
                f"AP replay mismatch: {dataset} {detector} {seed}",
                tolerance,
            )
            _, floor_limited = froc.budgets(ones, budgets, tolerance)
            if dataset == "external":
                expected_froc = source["metrics"]["froc_budgets"]
            else:
                expected_froc = [
                    r
                    for r in read_json(gate["internal_froc"]["outputs"]["summary_json"])[
                        "operating_points_per_seed"
                    ]
                    if r["detector"] == detector and r["seed"] == seed
                ]
            expected_froc.sort(key=lambda r: r["fp_per_image_budget"])
            close(
                point[2 : 2 + len(budgets)],
                [r["sensitivity"] for r in expected_froc],
                "FROC point replay mismatch",
                tolerance,
            )
            require(
                floor_limited.tolist() == [r["candidate_floor_limited"] for r in expected_froc],
                "FROC support mismatch",
            )
            fp_curve, sensitivity_curve = froc.curve(ones)
            row = {
                "dataset": dataset,
                "detector": detector,
                "seed": seed,
                "images": len(ids),
                "target_boxes": int(ap.target_counts.sum()),
                "positive_images": int(np.count_nonzero(ap.target_counts)),
                "patient_groups": clusters.patient_group_count if clusters else None,
                **dict(zip(metric_names(protocol), map(float, point), strict=True)),
                "images_at_collection_cap": sum(
                    len(p.scores) == evaluation["max_detections_per_image"]
                    for p in lower_predictions
                ),
                "floor_fp_per_image": float(fp_curve[-1]),
                "floor_sensitivity": float(sensitivity_curve[-1]),
            }
            for b, flag in zip(budgets, floor_limited, strict=True):
                row[f"floor_limited_{b:g}"] = bool(flag)
            for policy in ("primary", "secondary"):
                settings = protocol["threshold_transport"][policy]
                threshold = settings["thresholds"][detector]
                counts = operating_counts(op_cache, threshold)
                tp, fp = counts.sum(axis=0)
                fn = int(ap.target_counts.sum() - tp)
                predicted = int(tp + fp)
                precision = float(tp / predicted) if predicted else 0.0
                recall = float(tp / ap.target_counts.sum())
                f1 = 2 * tp / (2 * tp + fp + fn)
                actual = [tp, fp, fn, precision, recall, f1]
                if dataset == "external":
                    expected = source["metrics"]["threshold_transport"][policy]
                    reference = [
                        expected[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")
                    ]
                else:
                    selected = [
                        r
                        for r in threshold_rows
                        if r["selection_scope"] == config["internal"][policy + "_scope"]
                        and r["detector"] == detector
                        and int(r["seed"]) == seed
                    ]
                    require(len(selected) == 1, "Threshold reference run missing or duplicated")
                    expected = selected[0]
                    close(
                        threshold,
                        float(expected["selected_threshold"]),
                        "Threshold changed",
                        tolerance,
                    )
                    reference = [
                        float(expected["test_" + k])
                        for k in (
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                            "precision",
                            "recall",
                            "f1",
                        )
                    ]
                close(actual, reference, "Frozen operating-point replay mismatch", tolerance)
                emitting = int(np.count_nonzero(counts.sum(axis=1)))
                metrics = {
                    "threshold": threshold,
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": float(f1),
                    "fp_per_image": float(fp / len(ids)),
                    "detections": predicted,
                    "detections_per_image": predicted / len(ids),
                    "percent_images_with_detections": 100 * emitting / len(ids),
                }
                row.update({f"{policy}_{k}": v for k, v in metrics.items()})
            populations = {
                "all_retained_candidates": (lower_predictions, evaluation["candidate_score_floor"]),
                "ap_candidates": (predictions, evaluation["ap_minimum_score"]),
                "historical_threshold": (predictions, row["primary_threshold"]),
                "posthoc_n5_threshold": (predictions, row["secondary_threshold"]),
            }
            ap_scores = None
            for support, (population, floor) in populations.items():
                counts = np.asarray([np.count_nonzero(p.scores >= floor) for p in population])
                values = np.concatenate([p.scores[p.scores >= floor] for p in population])
                stats = score_summary(values, protocol)
                score_row = {
                    "dataset": dataset,
                    "detector": detector,
                    "seed": seed,
                    "support": support,
                    "score_floor": floor,
                    "images": len(ids),
                    "detections_per_image": len(values) / len(ids),
                    "zero_detection_images": int(np.count_nonzero(counts == 0)),
                    "percent_zero_detection_images": 100 * np.mean(counts == 0),
                    **{k: v for k, v in stats.items() if k != "quantiles"},
                    **{f"q_{q}": v for q, v in stats["quantiles"].items()},
                }
                scores.append(score_row)
                if dataset == "external":
                    expected = source["metrics"]["score_summaries"][support]
                    require(
                        stats == {k: expected[k] for k in stats}, "Score-summary replay mismatch"
                    )
                if support == "ap_candidates":
                    ap_scores = np.sort(values).tolist()
            plots.append(
                {
                    "dataset": dataset,
                    "detector": detector,
                    "seed": seed,
                    "froc_fp": fp_curve.tolist(),
                    "froc_sensitivity": sensitivity_curve.tolist(),
                    "ap_scores": ap_scores,
                }
            )
            rows.append(row)
            print(f"Prepared and replayed {dataset} {detector} seed {seed}", flush=True)
        detector_evidence.append(tuple(runs))
    return tuple(detector_evidence), rows, scores, plots, clusters


def summarize_values(values: list[float | None]) -> dict[str, Any]:
    """Equal-run arithmetic mean and sample SD, with defined run counts."""
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return {
        "mean": float(np.mean(valid)) if valid else None,
        "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
        "defined_runs": len(valid),
        "attempted_runs": len(values),
    }


def comparisons(
    per_run: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate cohort estimates, signed gap changes and historical-versus-post-hoc policies."""
    detectors = protocol["models"]["detectors"]
    tol = protocol["evaluation"]["froc"]["numeric_tolerance"]
    endpoints = metric_names(protocol) + [
        f"secondary_{m}" for m in ("precision", "recall", "f1", "fp_per_image")
    ]
    endpoints += [
        f"{p}_{m}"
        for p in ("primary", "secondary")
        for m in ("detections_per_image", "percent_images_with_detections")
    ]
    comparison = []
    for metric in endpoints:
        row = {"metric": metric}
        for dataset in ("internal", "external"):
            for letter, detector in zip(("a", "b"), detectors, strict=True):
                selected = [
                    r for r in per_run if r["dataset"] == dataset and r["detector"] == detector
                ]
                row.update(
                    {
                        f"{dataset}_{letter}_{k}": v
                        for k, v in summarize_values([r[metric] for r in selected]).items()
                    }
                )
                if metric.startswith("froc_"):
                    budget = metric.removeprefix("froc_")
                    row[f"{dataset}_{letter}_floor_limited_runs"] = sum(
                        r["floor_limited_" + budget] for r in selected
                    )
                    row[f"{dataset}_{letter}_conservative_upper"] = float(
                        np.mean(
                            [1.0 if r["floor_limited_" + budget] else r[metric] for r in selected]
                        )
                    )
            row[f"{dataset}_a_minus_b"] = row[f"{dataset}_a_mean"] - row[f"{dataset}_b_mean"]
        row.update(classify_change(row["internal_a_minus_b"], row["external_a_minus_b"], tol))
        comparison.append(row)
    aggregate_scores = []
    for support in protocol["score_summaries"]["populations"]:
        for dataset in ("internal", "external"):
            for detector in detectors:
                selected = [
                    r
                    for r in score_rows
                    if r["dataset"] == dataset
                    and r["detector"] == detector
                    and r["support"] == support
                ]
                for metric in (
                    "detection_count",
                    "detections_per_image",
                    "percent_zero_detection_images",
                    "min",
                    "max",
                    "mean",
                    "median",
                    *[f"q_{q}" for q in protocol["score_summaries"]["quantiles"]],
                ):
                    aggregate_scores.append(
                        {
                            "dataset": dataset,
                            "detector": detector,
                            "support": support,
                            "metric": metric,
                            **summarize_values([r[metric] for r in selected]),
                        }
                    )
    for metric in ("mean", "median", "detections_per_image", "percent_zero_detection_images"):
        row = {"metric": "ap_scores_" + metric}
        for dataset in ("internal", "external"):
            for letter, detector in zip(("a", "b"), detectors, strict=True):
                agg = next(
                    r
                    for r in aggregate_scores
                    if r["dataset"] == dataset
                    and r["detector"] == detector
                    and r["support"] == "ap_candidates"
                    and r["metric"] == metric
                )
                row.update(
                    {
                        f"{dataset}_{letter}_{k}": agg[k]
                        for k in ("mean", "sd", "defined_runs", "attempted_runs")
                    }
                )
            row[f"{dataset}_a_minus_b"] = row[f"{dataset}_a_mean"] - row[f"{dataset}_b_mean"]
        row.update(classify_change(row["internal_a_minus_b"], row["external_a_minus_b"], tol))
        comparison.append(row)
    policy_comparison = []
    for dataset in ("internal", "external"):
        for metric in ("precision", "recall", "f1", "fp_per_image", "detections_per_image"):
            primary = next(r for r in comparison if r["metric"] == "primary_" + metric)
            secondary = next(r for r in comparison if r["metric"] == "secondary_" + metric)
            before, after = primary[dataset + "_a_minus_b"], secondary[dataset + "_a_minus_b"]
            policy_comparison.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "historical_n3_gap": before,
                    "posthoc_n5_gap": after,
                    **classify_change(before, after, tol),
                }
            )
    return comparison, policy_comparison, aggregate_scores


def analyze(gate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Recompute each cohort separately and retain all five runs and every attempted draw."""
    config, protocol = gate["config"], gate["protocol"]
    names, settings = metric_names(protocol), protocol["statistics"]
    budgets = protocol["evaluation"]["froc"]["fp_per_image_budgets"]
    tolerance = protocol["evaluation"]["froc"]["numeric_tolerance"]
    per_run, scores, plots, intervals, draws, populations = [], [], [], [], {}, {}
    for dataset in ("internal", "external"):
        evidence, rows, score_rows, plot_rows, clusters = prepare_dataset(gate, dataset)
        per_run.extend(rows)
        scores.extend(score_rows)
        plots.extend(plot_rows)
        label = f"{config['analysis_id']}:{dataset}"
        samples = bootstrap(
            evidence,
            clusters=clusters,
            settings=settings,
            label=label,
            budgets=budgets,
            tolerance=tolerance,
            conditional_seed=config["conditional_seed"],
            progress_every=config["progress_every"],
        )
        ones = np.ones(rows[0]["images"], dtype=np.int64)
        for estimand, values in samples.items():
            draws[f"{dataset}_{estimand}"] = values
            point = np.asarray(
                [
                    np.mean([r.evaluate(ones, budgets, tolerance) for r in runs], axis=0)
                    if estimand == "training_procedure"
                    else next(r for r in runs if r.seed == config["conditional_seed"]).evaluate(
                        ones, budgets, tolerance
                    )
                    for runs in evidence
                ]
            )
            intervals.extend(
                {"dataset": dataset, "estimand": estimand, **r}
                for r in interval_rows(point, values, names, settings["confidence_level"])
            )
        populations[dataset] = {
            "images": rows[0]["images"],
            "target_boxes": rows[0]["target_boxes"],
            "positive_images": rows[0]["positive_images"],
            "patient_groups": rows[0]["patient_groups"],
            "resampling_unit": "NIH_patient" if clusters else "released_study_image",
            "bootstrap_label": label,
            "derived_rng_seed": stable_rng_seed(settings["bootstrap_seed"], label),
        }
        del evidence
    comparison, policies, aggregate_scores = comparisons(per_run, scores, protocol)
    payload = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "freeze_files_verified": gate["freeze_files_verified"],
        "statistics": settings,
        "metrics": names,
        "populations": populations,
        "detectors": protocol["models"]["detectors"],
        "seeds": protocol["models"]["seeds"],
        "per_run": per_run,
        "scores": scores,
        "aggregate_scores": aggregate_scores,
        "intervals": intervals,
        "comparison": comparison,
        "policy_comparison": policies,
        "inputs": gate["inputs"],
        "environment": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("numpy", "pycocotools", "matplotlib", "scipy")},
        },
        "scope": {
            "new_internal_intervals": "Batch 50 aligned endpoint reconstruction; historical "
            "outputs preserved",
            "secondary_threshold_policy": "descriptive_only_posthoc_n5_validation_sensitivity",
            "cross_dataset_gap_changes": "descriptive_not_a_test_of_dataset_interaction",
            "threshold_selection_uncertainty": (
                "not_resampled_thresholds_conditioned_on_frozen_values"
            ),
            "multiplicity": "marginal_percentile_intervals_no_new_hypothesis_tests",
            "pooling": "no_pooling_of_datasets_or_run_predictions",
            "patient_limitation": "Unknown repeated-patient dependence in VinDr may narrow "
            "intervals",
            "precision_path": "Historical YOLO internal FP32 versus mandatory external bfloat16 "
            "AMP",
            "causality": "Cross-dataset transportability; dataset factors and implementation "
            "are confounded",
        },
    }
    return payload, draws, plots


def run(config_path: Path) -> dict[str, Any]:
    """Generate a new versioned analysis without overwriting existing evidence."""
    gate = preflight(config_path)
    config = gate["config"]
    out = config["outputs"]
    public, private = Path(out["public_root"]), Path(out["private_root"])
    require(
        not (public / out["summary"]).exists(), "Statistics already exist; use verify or replay"
    )
    require(
        not (private / out["bootstrap_draws"]).exists(),
        "Private draws already exist; preserve them",
    )
    payload, draws, plots = analyze(gate)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private / out["bootstrap_draws"], **draws)
    (private / out["plot_data"]).write_bytes(
        gzip.compress(json.dumps(plots, allow_nan=False).encode(), mtime=0)
    )
    for key, rows in (
        ("intervals", payload["intervals"]),
        ("per_run", payload["per_run"]),
        ("scores", payload["scores"]),
        ("comparison", payload["comparison"]),
        ("policy_comparison", payload["policy_comparison"]),
    ):
        write_csv(public / out[key], rows)
    from src.stats.report_transportability import render

    render(payload, plots, config, gate["protocol"])
    output_paths = [
        public / out[k]
        for k in ("intervals", "per_run", "scores", "comparison", "policy_comparison")
    ]
    output_paths += [public / name for name in out["figures"].values()]
    output_paths += [
        Path(out["report"]),
        private / out["bootstrap_draws"],
        private / out["plot_data"],
    ]
    payload["artifacts"] = [artifact(p) for p in output_paths]
    # Inputs must also remain unchanged while a long bootstrap is running.
    for entry in payload["inputs"]:
        verify_artifact(entry)
    write_json(public / out["summary"], payload)
    return {"status": "complete", "summary": artifact(public / out["summary"])}


def verify(config_path: Path, *, replay: bool = False) -> dict[str, Any]:
    """Read-only hash, aggregation and interval replay; optionally rerun every bootstrap draw."""
    gate = preflight(config_path)
    config, protocol = gate["config"], gate["protocol"]
    out = config["outputs"]
    summary_path = Path(out["public_root"]) / out["summary"]
    saved = read_json(summary_path)
    require(
        saved["status"] == "complete" and saved["inputs"] == gate["inputs"],
        "Statistical input provenance changed",
    )
    for item in saved["artifacts"]:
        verify_artifact(item)
    comparison, policies, aggregate_scores = comparisons(
        saved["per_run"], saved["scores"], protocol
    )
    require(
        json_number(comparison) == saved["comparison"]
        and json_number(policies) == saved["policy_comparison"]
        and json_number(aggregate_scores) == saved["aggregate_scores"],
        "Aggregate replay mismatch",
    )
    with np.load(Path(out["private_root"]) / out["bootstrap_draws"], allow_pickle=False) as arrays:
        for dataset in ("internal", "external"):
            for estimand in ("training_procedure", "checkpoint_conditional_seed17"):
                selected = [
                    r
                    for r in saved["intervals"]
                    if r["dataset"] == dataset and r["estimand"] == estimand
                ]
                point = np.asarray([[r[f"{d}_estimate"] for r in selected] for d in ("a", "b")])
                fresh = interval_rows(
                    point,
                    arrays[f"{dataset}_{estimand}"],
                    saved["metrics"],
                    protocol["statistics"]["confidence_level"],
                )
                require(
                    json_number([{"dataset": dataset, "estimand": estimand, **r} for r in fresh])
                    == selected,
                    "Percentile interval replay mismatch",
                )
        if replay:
            fresh, draws, _ = analyze(gate)
            for key in (
                "per_run",
                "scores",
                "aggregate_scores",
                "intervals",
                "comparison",
                "policy_comparison",
                "populations",
            ):
                require(json_number(fresh[key]) == saved[key], f"Full replay mismatch: {key}")
            for key, values in draws.items():
                require(
                    np.array_equal(values, arrays[key], equal_nan=True), "Bootstrap draws changed"
                )
    return {
        "status": "verified",
        "full_bootstrap_replay": replay,
        "summary": artifact(summary_path),
        "interval_rows": len(saved["intervals"]),
        "input_artifacts": len(saved["inputs"]),
    }


def main() -> None:
    """Expose explicit offline preflight, run, verify, full replay and rendering commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "run", "verify", "replay", "render"), required=True
    )
    args = parser.parse_args()
    if args.mode == "preflight":
        gate = preflight(args.config)
        result = {
            "status": "passed",
            "frozen_files": gate["freeze_files_verified"],
            "inputs": len(gate["inputs"]),
        }
    elif args.mode == "run":
        result = run(args.config)
    elif args.mode == "render":
        verify(args.config)
        config = read_yaml(args.config)
        out = config["outputs"]
        from src.stats.report_transportability import render

        render(
            read_json(Path(out["public_root"]) / out["summary"]),
            read_gzip(Path(out["private_root"]) / out["plot_data"]),
            config,
            read_yaml(config["protocol_config"]),
        )
        result = verify(args.config)
    else:
        result = verify(args.config, replay=args.mode == "replay")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
