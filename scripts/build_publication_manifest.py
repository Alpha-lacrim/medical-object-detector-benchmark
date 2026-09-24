"""Build the reviewed publication inventory without rewriting the frozen internal one.

Maintenance command only: CI verifies the committed inventory and never rebuilds it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


def digest(path: str) -> str:
    """Return an exact byte hash."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Extend unchanged internal entries with frozen external aggregate evidence."""
    root = Path.cwd()
    internal_path = "results/scientific_artifact_manifest.json"
    manifest = json.loads(Path(internal_path).read_text(encoding="utf-8"))
    manifest["generated_by"] = "scripts/build_publication_manifest.py"
    manifest["verification_command"] = (
        "python scripts/verify_scientific_artifacts.py "
        "--manifest results/publication_artifact_manifest.json"
    )
    baseline = json.loads(
        Path("report/provenance/batch46/baseline_bindings.json").read_text(encoding="utf-8")
    )
    historical = {e["path"]: e["archived_path"] for e in baseline["artifacts"]}

    def binding(path: str, sha256: str | None = None) -> dict:
        if Path(path).is_absolute():
            path = Path(path).relative_to(root).as_posix()
        path = historical.get(path, path)
        return {
            "path": path,
            "sha256": sha256 or digest(path),
            "availability": (
                "external_or_ignored"
                if path.startswith(("data/processed/", "data/raw/", "results/checkpoints/"))
                else "committed"
            ),
        }

    def add(path: str, script: str, config: str, inputs: list[dict]) -> None:
        source = Path(path)
        if source.suffix == ".csv":
            with source.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                columns = reader.fieldnames
            schema = {
                "type": "csv",
                "required_columns": columns,
                "exact_columns": columns,
                "row_count": len(rows),
            }
        elif source.suffix == ".png":
            with Image.open(source) as image:
                schema = {"type": "png", "width": image.width, "height": image.height}
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            schema = {"type": "json", "top_level_type": "object", "required_keys": list(payload)}
        manifest["artifacts"].append(
            {
                "path": path,
                "sha256": digest(path),
                "generating_script": script,
                "generating_script_sha256": digest(script),
                "config": {"path": config, "sha256": digest(config)},
                "input_artifacts": inputs,
                "expected_schema": schema,
                "study_phase": "Frozen VinDr external evidence, publication integration",
                "reproduction_tier": "authorized_data_replay",
                "gpu_required": False,
                "training_required": False,
                "referenced_results": [],
            }
        )

    prefix = "results/vindr_external_v1/"
    inference_path = prefix + "inference_summary.json"
    inference = json.loads(Path(inference_path).read_text(encoding="utf-8"))
    protocol = "configs/vindr_external_v1.yaml"
    adapter = prefix + "adapter_preflight.json"
    add(
        adapter,
        "src/data/prepare_vindr.py",
        "configs/vindr_adapter_v1.yaml",
        [
            binding(protocol),
            binding("docs/VINDR_EXTERNAL_PROTOCOL_v1.sha256.json"),
        ],
    )
    inputs = [binding(p, h) for p, h in inference["gate"]["input_sha256"].items()]
    for run in inference["runs"]:
        inputs += [binding(a["path"], a["sha256"]) for a in run["artifacts"]]
        item = run["metadata"]["checkpoint"]
        inputs.append(binding(item["path"], item["sha256"]))
    inputs.append(
        binding(inference["execution_receipt"]["path"], inference["execution_receipt"]["sha256"])
    )
    add(inference_path, "src/evaluate_vindr_external.py", "configs/vindr_inference_v1.yaml", inputs)
    summary_path = prefix + "statistics/summary.json"
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    config = "configs/vindr_statistics_v1.yaml"
    add(
        summary_path,
        "src/stats/run_vindr_statistics.py",
        config,
        [binding(e["path"], e["sha256"]) for e in summary["inputs"]],
    )
    for item in summary["artifacts"]:
        if item["path"].startswith(prefix):
            add(item["path"], "src/stats/run_vindr_statistics.py", config, [binding(summary_path)])
    manifest["manuscript_critical_paths"] = [e["path"] for e in manifest["artifacts"]]
    manifest["publication_extension"] = {
        "internal_manifest": binding(internal_path),
        "note": (
            "Internal entries copied unchanged; original v1 freeze and scientific "
            "definitions retained."
        ),
        "historical_editorial_bindings": "report/provenance/batch46/baseline_bindings.json",
    }
    Path("results/publication_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
