"""An editorial rewrite must not relax frozen scientific provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_frozen_external import BASELINE, baseline_paths, historical_editorial_view


def fixture(root: Path) -> None:
    """Create old manuscript bytes, new live bytes and an immutable scientific input."""
    entries = []
    for name in ("paper_draft.md", "paper_claim_sources.yaml"):
        live = root / "report" / name
        archived = root / BASELINE.parent / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(b"historical")
        live.write_bytes(b"current editorial content")
        entries.append(
            {
                "path": "report/" + name,
                "archived_path": archived.relative_to(root).as_posix(),
                "role": "batch46_baseline",
                "sha256": hashlib.sha256(b"historical").hexdigest(),
            }
        )
    (root / "freeze.json").write_text(json.dumps({"artifacts": entries}))
    (root / BASELINE).write_text(
        json.dumps({"freeze_manifest": "freeze.json", "artifacts": entries})
    )


def test_archived_reads_and_sizes_preserve_live_bytes_even_on_exception(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    live = Path("report/paper_draft.md")
    with pytest.raises(RuntimeError), historical_editorial_view(tmp_path):
        assert live.read_bytes() == b"historical"
        assert live.stat().st_size == len(b"historical")
        with pytest.raises(ValueError, match="cannot write"):
            live.write_text("forbidden")
        raise RuntimeError("test cleanup")
    assert live.read_bytes() == b"current editorial content"


def test_archive_tampering_fails(tmp_path):
    fixture(tmp_path)
    (tmp_path / BASELINE.parent / "paper_draft.md").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="original freeze"):
        baseline_paths(tmp_path)


def test_cannot_redirect_a_scientific_input(tmp_path):
    fixture(tmp_path)
    path = tmp_path / BASELINE
    manifest = json.loads(path.read_text())
    manifest["artifacts"][0]["path"] = "configs/vindr_external_v1.yaml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Only the two editorial"):
        baseline_paths(tmp_path)


def test_publication_extension_preserves_all_internal_entries():
    internal = json.loads(Path("results/scientific_artifact_manifest.json").read_text())
    publication = json.loads(Path("results/publication_artifact_manifest.json").read_text())
    assert publication["artifacts"][: len(internal["artifacts"])] == internal["artifacts"]
