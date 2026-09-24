"""Replay frozen evidence using its archived editorial baseline, without disk swaps.

The v1 freeze bound two then-current manuscript files as historical context.
Only reads/stat calls for those explicitly allow-listed records resolve to their
hash-identical archive. All scientific paths, code, hashes and checks stay live.
Current publication claims are checked separately by verify_paper_claims.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

BASELINE = Path("report/provenance/batch46/baseline_bindings.json")
EDITORIAL_PATHS = {"report/paper_draft.md", "report/paper_claim_sources.yaml"}


def baseline_paths(root: Path) -> dict[Path, Path]:
    """Validate the exact two archived editorial records against the original freeze."""
    bindings = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    freeze = json.loads((root / bindings["freeze_manifest"]).read_text(encoding="utf-8"))
    entries = bindings["artifacts"]
    if len(entries) != len(EDITORIAL_PATHS) or {e["path"] for e in entries} != EDITORIAL_PATHS:
        raise ValueError("Only the two editorial baseline paths may be redirected")
    frozen = {e["path"]: e for e in freeze["artifacts"]}
    result = {}
    for entry in entries:
        source = frozen[entry["path"]]
        archived = (root / entry["archived_path"]).resolve()
        if not archived.is_relative_to((root / BASELINE.parent).resolve()):
            raise ValueError("Editorial archive escapes its provenance directory")
        digest = hashlib.sha256(archived.read_bytes()).hexdigest()
        if source["role"] != "batch46_baseline" or digest != source["sha256"]:
            raise ValueError("Archived editorial baseline differs from the original freeze")
        if digest != entry["sha256"]:
            raise ValueError("Editorial archive binding differs from frozen bytes")
        result[(root / entry["path"]).resolve()] = archived
    return result


@contextmanager
def historical_editorial_view(root: Path) -> Iterator[None]:
    """Expose archived baseline reads only; never mutate current or archived files."""
    mapping = baseline_paths(root)
    original_open, original_stat = Path.open, Path.stat

    def target(path: Path) -> Path:
        # absolute(), unlike resolve(), does not itself call the patched stat().
        return mapping.get(path.absolute(), path)

    def read_baseline(path: Path, mode: str = "r", *args: object, **kwargs: object):
        resolved = target(path)
        if resolved != path and any(flag in mode for flag in "wax+"):
            raise ValueError("Historical verification cannot write an editorial baseline")
        return original_open(resolved, mode, *args, **kwargs)

    def stat_baseline(path: Path, *args: object, **kwargs: object):
        return original_stat(target(path), *args, **kwargs)

    with patch.object(Path, "open", read_baseline), patch.object(Path, "stat", stat_baseline):
        yield


def main() -> None:
    """Run only existing read-only verifiers; inference and regeneration are unavailable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("freeze", "inference", "statistics", "replay"), required=True
    )
    args = parser.parse_args()
    root = Path.cwd()
    for live, archive in baseline_paths(root).items():
        print(
            f"Historical editorial input: {live.relative_to(root)} <- {archive.relative_to(root)}"
        )
    with historical_editorial_view(root):
        if args.mode == "freeze":
            import yaml

            from src.data.prepare_vindr import verify_freeze

            path = Path("configs/vindr_external_v1.yaml")
            result = {
                "frozen_files_verified": len(verify_freeze(path, yaml.safe_load(path.read_text())))
            }
        elif args.mode == "inference":
            from src.evaluate_vindr_external import verify

            result = verify(
                Path("configs/vindr_external_v1.yaml"), Path("configs/vindr_inference_v1.yaml")
            )
        else:
            from src.stats.run_vindr_statistics import verify

            result = verify(Path("configs/vindr_statistics_v1.yaml"), replay=args.mode == "replay")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
