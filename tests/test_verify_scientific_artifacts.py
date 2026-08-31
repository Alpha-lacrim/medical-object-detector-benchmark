"""Regression tests for the frozen scientific-artifact verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_scientific_artifacts import (
    ArtifactVerificationError,
    sha256_file,
    verify_manifest,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _fixture_manifest(root: Path) -> tuple[Path, dict[str, object]]:
    artifact = root / "results/tables/critical.csv"
    generator = root / "src/generate.py"
    config = root / "configs/study.yaml"
    source = root / "results/source.json"
    reference = root / "results/figures/critical.png"
    _write(artifact, "detector,score\nfaster_rcnn,0.5\n")
    _write(generator, "# generator\n")
    _write(config, "seed: 17\n")
    _write(source, '{"status": "complete"}\n')
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"referenced result")
    entry = {
        "path": "results/tables/critical.csv",
        "sha256": sha256_file(artifact),
        "generating_script": "src/generate.py",
        "generating_script_sha256": sha256_file(generator),
        "config": {
            "path": "configs/study.yaml",
            "sha256": sha256_file(config),
        },
        "input_artifacts": [
            {
                "path": "results/source.json",
                "sha256": sha256_file(source),
                "availability": "committed",
            },
            {
                "path": "data/raw/not-committed.csv",
                "sha256": "0" * 64,
                "availability": "external_or_ignored",
            },
        ],
        "expected_schema": {
            "type": "csv",
            "required_columns": ["detector", "score"],
            "exact_columns": ["detector", "score"],
            "row_count": 1,
        },
        "study_phase": "test phase",
        "reproduction_tier": "committed_analysis",
        "gpu_required": False,
        "training_required": False,
        "referenced_results": ["results/figures/critical.png"],
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "manuscript_critical_paths": ["results/tables/critical.csv"],
        "artifacts": [entry],
    }
    manifest_path = root / "results/scientific_artifact_manifest.json"
    _write(manifest_path, json.dumps(payload))
    return manifest_path, payload


def _rewrite_manifest(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload))


def test_verify_manifest_accepts_bound_artifact_and_missing_external_input(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _fixture_manifest(tmp_path)

    report = verify_manifest(manifest_path, project_root=tmp_path)

    assert report.artifact_count == 1
    assert report.checked_input_count == 1
    assert report.skipped_external_input_count == 1
    assert report.referenced_result_count == 1


def test_verify_manifest_rejects_stale_artifact_hash(tmp_path: Path) -> None:
    manifest_path, _ = _fixture_manifest(tmp_path)
    _write(tmp_path / "results/tables/critical.csv", "detector,score\nyolo11s,0.5\n")

    with pytest.raises(ArtifactVerificationError, match="is stale"):
        verify_manifest(manifest_path, project_root=tmp_path)


def test_verify_manifest_rejects_schema_drift_even_with_current_hash(tmp_path: Path) -> None:
    manifest_path, payload = _fixture_manifest(tmp_path)
    artifact = tmp_path / "results/tables/critical.csv"
    _write(artifact, "detector\nfaster_rcnn\n")
    entry = payload["artifacts"][0]
    assert isinstance(entry, dict)
    entry["sha256"] = sha256_file(artifact)
    _rewrite_manifest(manifest_path, payload)

    with pytest.raises(ArtifactVerificationError, match="missing required columns"):
        verify_manifest(manifest_path, project_root=tmp_path)


def test_verify_manifest_rejects_missing_referenced_result(tmp_path: Path) -> None:
    manifest_path, _ = _fixture_manifest(tmp_path)
    (tmp_path / "results/figures/critical.png").unlink()

    with pytest.raises(ArtifactVerificationError, match=r"referenced_results.*is missing"):
        verify_manifest(manifest_path, project_root=tmp_path)
