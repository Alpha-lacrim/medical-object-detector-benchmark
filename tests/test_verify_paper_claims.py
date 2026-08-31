"""Regression tests for machine-checkable manuscript claim bindings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.verify_paper_claims import PaperClaimVerificationError, verify_claims


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _claim_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    _write(
        root / "results/tables/metrics.csv",
        "detector,kind,value\nmodel_a,clean,10\nmodel_a,corrupted,2\nmodel_a,corrupted,4\n",
    )
    _write(root / "results/summary.json", '{"counts": {"images": 750}}\n')
    _write(
        root / "results/scientific_artifact_manifest.json",
        json.dumps(
            {
                "artifacts": [
                    {"path": "results/tables/metrics.csv"},
                    {"path": "results/summary.json"},
                ]
            }
        ),
    )
    _write(
        root / "report/paper.md",
        "Images: 750. Clean value: 10. Mean corrupted value: 3. Ratio: 2.5.\n",
    )
    claims: dict[str, object] = {
        "schema_version": 1,
        "manuscript": "report/paper.md",
        "scientific_artifact_manifest": "results/scientific_artifact_manifest.json",
        "claims": [
            {
                "id": "image_count",
                "manuscript": {"regex": r"Images: (?P<value>\d+)\."},
                "source": {
                    "kind": "json_value",
                    "path": "results/summary.json",
                    "pointer": ["counts", "images"],
                },
                "absolute_tolerance": 0,
            },
            {
                "id": "corruption_mean",
                "manuscript": {"regex": r"Mean corrupted value: (?P<value>\d+(?:\.\d+)?)\."},
                "source": {
                    "kind": "csv_mean",
                    "path": "results/tables/metrics.csv",
                    "filters": {"detector": "model_a"},
                    "exclude_filters": {"kind": "clean"},
                    "column": "value",
                },
                "absolute_tolerance": 0,
            },
            {
                "id": "ratio",
                "manuscript": {"regex": r"Ratio: (?P<value>\d+\.\d+)\."},
                "source": {
                    "kind": "calculation",
                    "operation": "divide",
                    "operands": [
                        {
                            "kind": "csv_cell",
                            "path": "results/tables/metrics.csv",
                            "filters": {"detector": "model_a", "kind": "clean"},
                            "column": "value",
                        },
                        {
                            "kind": "csv_cell",
                            "path": "results/tables/metrics.csv",
                            "filters": {"detector": "model_a", "kind": "corrupted"},
                            "column": "value",
                        },
                    ],
                },
                "absolute_tolerance": 0,
            },
        ],
    }
    # Make the ratio operand select one row while retaining two rows for csv_mean.
    ratio = claims["claims"][2]
    assert isinstance(ratio, dict)
    operands = ratio["source"]["operands"]
    operands[1]["filters"]["value"] = 4
    manifest_path = root / "report/paper_claim_sources.yaml"
    _write(manifest_path, yaml.safe_dump(claims, sort_keys=False))
    return manifest_path, claims


def test_verify_claims_accepts_cells_aggregations_and_calculations(tmp_path: Path) -> None:
    manifest_path, _ = _claim_fixture(tmp_path)

    assert verify_claims(manifest_path, project_root=tmp_path) == 3


def test_verify_claims_rejects_manuscript_value_outside_tolerance(tmp_path: Path) -> None:
    manifest_path, _ = _claim_fixture(tmp_path)
    paper = tmp_path / "report/paper.md"
    _write(paper, paper.read_text(encoding="utf-8").replace("Images: 750", "Images: 749"))

    with pytest.raises(PaperClaimVerificationError, match="image_count mismatch"):
        verify_claims(manifest_path, project_root=tmp_path)


def test_verify_claims_rejects_source_outside_artifact_manifest(tmp_path: Path) -> None:
    manifest_path, _ = _claim_fixture(tmp_path)
    scientific_manifest = tmp_path / "results/scientific_artifact_manifest.json"
    _write(scientific_manifest, json.dumps({"artifacts": []}))

    with pytest.raises(PaperClaimVerificationError, match="absent from"):
        verify_claims(manifest_path, project_root=tmp_path)


def test_verify_claims_requires_one_unique_manuscript_match(tmp_path: Path) -> None:
    manifest_path, _ = _claim_fixture(tmp_path)
    paper = tmp_path / "report/paper.md"
    text = paper.read_text(encoding="utf-8")
    _write(paper, text + text)

    with pytest.raises(PaperClaimVerificationError, match="expected one match"):
        verify_claims(manifest_path, project_root=tmp_path)
