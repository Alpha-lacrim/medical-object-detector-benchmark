"""Render aggregate and per-run numerical-path evidence from verified outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.analyze_yolo_numerical_path import digest, numeric_leaves, read_json


def number(value: Any) -> str:
    """Render enough digits for sensitivity comparisons, preserving undefined."""
    return "undefined" if value is None else f"{value:.9g}"


def render(config: dict[str, Any]) -> str:
    """Render the bound evidence without model access or scientific choices."""
    for path, expected in config["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Report input hash mismatch: {path}")
    summary = read_json(config["summary"])
    aggregate = {r["metric"]: r for r in summary["aggregate"]}
    lines = [
        "# YOLO numerical-path sensitivity: results v1",
        "",
        "Secondary sensitivity; five frozen YOLO checkpoints, paired within checkpoint",
        "on the same 750 internal images. Historical FP32 and external results remain",
        "unchanged. Deltas are bfloat16 minus FP32. Means and sample SDs weight runs",
        "equally. No new uncertainty test or threshold selection is performed.",
        "",
        "See [protocol and interpretation](YOLO_NUMERICAL_PATH_SENSITIVITY_V1.md)",
        "for precision scope, candidate matching and residual confounding.",
        "",
        f"Source summary SHA-256: `{digest(config['summary'])}`.",
    ]
    for group in config["aggregate_groups"]:
        lines.extend(
            [
                "",
                "## " + group["title"],
                "",
                "| Metric | FP32 mean +/- SD | bfloat16 mean +/- SD | Mean delta +/- SD |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, label in group["metrics"].items():
            row = aggregate[key]
            cells = [
                f"{number(row[role + '_mean'])} +/- {number(row[role + '_sample_sd'])}"
                for role in ("fp32", "bfloat16", "delta")
            ]
            lines.append("| " + label + " | " + " | ".join(cells) + " |")
    for group in config["per_run_groups"]:
        lines.extend(
            [
                "",
                "## " + group["title"],
                "",
                "| Seed | Metric | FP32 | bfloat16 | Delta |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for pair in summary["paired_runs"]:
            old, new = numeric_leaves(pair["fp32"]), numeric_leaves(pair["bfloat16"])
            for key, label in group["metrics"].items():
                a, b = old[key], new[key]
                delta = b - a if a is not None and b is not None else None
                lines.append(
                    f"| {pair['seed']} | {label} | {number(a)} | {number(b)} | {number(delta)} |"
                )
    lines.extend(
        [
            "",
            "## Candidate agreement at retained support",
            "",
            "Unmatched boxes are unmatched under the prespecified IoU rule, not proven",
            "identities of distinct pre-NMS proposals. Score rank correlation is conditional",
            "on successfully matched candidates within each run.",
            "",
            "| Seed | Metric | Value |",
            "|---:|---|---:|",
        ]
    )
    for pair in summary["paired_runs"]:
        leaves = numeric_leaves(pair["agreement"])
        for key, label in config["agreement_metrics"].items():
            lines.append(f"| {pair['seed']} | {label} | {number(leaves[key])} |")
    lines.extend(
        [
            "",
            "## Threshold-crossing decomposition",
            "",
            "Matched fractions use all matched candidates as denominator. Unmatched counts",
            "above each cutoff are shown separately; net emission changes do not require",
            "one-to-one candidate correspondence.",
            "",
            "| Seed | Cutoff | Matched up (fraction) | Matched down (fraction) | "
            "Unmatched FP32 above | Unmatched bfloat16 above |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in summary["paired_runs"]:
        for cutoff, row in pair["agreement"]["threshold_crossings"].items():
            lines.append(
                f"| {pair['seed']} | {cutoff} | {row['matched_up']} "
                f"({number(row['matched_up_fraction'])}) | {row['matched_down']} "
                f"({number(row['matched_down_fraction'])}) | {row['unmatched_old_above']} | "
                f"{row['unmatched_new_above']} |"
            )
    lines.extend(
        [
            "",
            "Complete min/quantiles/median/mean/max at every defined score support, all",
            "emission proportions, per-run deltas and floor/cap diagnostics remain in the",
            "bound summary and CSVs. The retained-candidate floor/cap limits and dataset/",
            "site/population/annotation/preprocessing confounding are not removed by this",
            "sensitivity. Candidate agreement is post-NMS, not a pre-NMS tensor comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Create an exclusive report, or verify its deterministic text read-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    text = render(config)
    output = Path(config["output"])
    if args.verify:
        if output.read_text(encoding="utf-8") != text:
            raise ValueError("Sensitivity report differs from verified evidence")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(f"Sensitivity report {'verified' if args.verify else 'created'}: {output}")


if __name__ == "__main__":
    main()
