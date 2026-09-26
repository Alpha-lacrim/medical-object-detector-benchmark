"""Maintenance-only addition of reviewed secondary evidence to the publication inventory."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any

from src import analyze_yolo_numerical_path as analysis


def binding(path: str, expected: str | None = None) -> dict[str, str]:
    """Bind exact bytes; local checkpoint/source-data availability remains explicit."""
    actual = analysis.digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Evidence changed: {path}")
    return {
        "path": path,
        "sha256": actual,
        "availability": (
            "external_or_ignored"
            if path.startswith(("data/processed/", "results/checkpoints/"))
            else "committed"
        ),
    }


def main() -> None:
    """Keep every prior entry intact and refuse a repeated or overlapping extension."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    config = analysis.load_config(args.config)
    root = Path(config["outputs"]["root"])
    summary_path = root / config["outputs"]["summary"]
    summary = analysis.read_json(summary_path)
    if summary["config_sha256"] != analysis.digest(args.config):
        raise ValueError("Sensitivity config changed")
    manifest = analysis.read_json(args.manifest)
    paths = [summary_path] + [
        root / config["outputs"][k]
        for k in ("protocol", "per_run_table", "aggregate_table", "agreement_table")
    ]
    previous = {entry["path"] for entry in manifest["artifacts"]}
    if any(path.as_posix() in previous for path in paths):
        raise ValueError("Sensitivity entries already exist; verify, never refresh hashes")
    source = Path(inspect.getfile(analysis)).resolve().relative_to(Path.cwd()).as_posix()
    if analysis.digest(source) != summary["source_sha256"]:
        raise ValueError("Inference implementation changed")
    inputs = [binding(p, h) for p, h in config["input_sha256"].items()]
    inputs += [binding(a["path"], a["sha256"]) for a in summary["artifacts"]]
    for path in paths:
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                columns = reader.fieldnames
            schema: dict[str, Any] = {
                "type": "csv",
                "required_columns": columns,
                "exact_columns": columns,
                "row_count": len(rows),
            }
        else:
            schema = {
                "type": "json",
                "top_level_type": "object",
                "required_keys": list(analysis.read_json(path)),
            }
        manifest["artifacts"].append(
            {
                "path": path.as_posix(),
                "sha256": analysis.digest(path),
                "generating_script": source,
                "generating_script_sha256": analysis.digest(source),
                "config": {"path": args.config.as_posix(), "sha256": analysis.digest(args.config)},
                "input_artifacts": inputs
                if path == summary_path
                else [binding(summary_path.as_posix())],
                "expected_schema": schema,
                "study_phase": "Batch 53 secondary internal YOLO numerical-path sensitivity",
                "reproduction_tier": "exact_inference"
                if path.name == config["outputs"]["protocol"]
                else "authorized_data_replay",
                "gpu_required": path.name == config["outputs"]["protocol"],
                "training_required": False,
                "referenced_results": [a["path"] for a in summary["artifacts"]]
                if path == summary_path
                else [],
            }
        )
    manifest["manuscript_critical_paths"] = [e["path"] for e in manifest["artifacts"]]
    manifest["publication_extension"]["secondary_numerical_path"] = {
        "analysis_id": config["analysis_id"],
        "maintenance_script": Path(__file__).resolve().relative_to(Path.cwd()).as_posix(),
        "note": "All pre-existing entries preserved; no primary or external estimate replaced.",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Added {len(paths)} secondary entries; preserved {len(previous)} previous entries")


if __name__ == "__main__":
    main()
