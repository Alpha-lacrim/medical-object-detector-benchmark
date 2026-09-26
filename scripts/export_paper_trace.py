"""Export the verified manuscript bindings as a reviewable numerical trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import yaml

from scripts.verify_paper_claims import evaluate_source, verify_claims


def main() -> None:
    """Verify first, then export exact locators and values without changing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parents[1]
    count = verify_claims(manifest_path, project_root=root)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manuscript = (root / manifest["manuscript"]).read_text(encoding="utf-8")
    rows = []
    for claim in manifest["claims"]:
        match = re.search(claim["manuscript"]["regex"], manuscript, re.MULTILINE | re.DOTALL)
        assert match is not None  # The verifier above requires exactly one match.
        rows.append(
            {
                "claim_id": claim["id"],
                "manuscript": manifest["manuscript"],
                "line": manuscript.count("\n", 0, match.start("value")) + 1,
                "description": claim.get("description", claim["manuscript"].get("section", "")),
                "manuscript_value": match.group("value"),
                "source_locator_or_derivation": json.dumps(claim["source"], sort_keys=True),
                "source_value": repr(
                    evaluate_source(claim["source"], root=root, label=claim["id"])
                ),
                "rounding": claim.get("rounding", "see absolute tolerance"),
                "absolute_tolerance": claim["absolute_tolerance"],
                "status": "PASS",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {count} verified numerical bindings to {args.output}")


if __name__ == "__main__":
    main()
