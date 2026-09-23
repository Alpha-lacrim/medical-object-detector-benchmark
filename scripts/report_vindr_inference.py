"""Render Batch 49 descriptive tables from generated aggregate evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def render(config_path: Path) -> Path:
    """Format saved values; do not infer, recompute, tune or select metrics."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source, destination = Path(config["summary"]), Path(config["output"])
    summary = json.loads(source.read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Cannot report incomplete inference")
    runs = summary["runs"]
    if [(r["metadata"]["detector"], r["metadata"]["seed"]) for r in runs] != [
        (r["detector"], r["seed"]) for r in summary["gate"]["checkpoints"]
    ]:
        raise ValueError("Report must retain every checkpoint")
    digits = config["decimal_places"]
    relative_source = Path(os.path.relpath(source, destination.parent)).as_posix()
    gate = summary["gate"]

    def number(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    lines = [
        "# Frozen VinDr-CXR inference results — Batch 49",
        "",
        f"Generated from [{source.as_posix()}]({relative_source}).",
        f"Source SHA-256: `{hashlib.sha256(source.read_bytes()).hexdigest()}`.",
        "",
        "All ten frozen RSNA checkpoints are retained, including both seed-271 runs.",
        f"The official VinDr-CXR v{summary['dataset_version']} test set contains "
        f"{gate['images']:,} released study/images,",
        f"{gate['strict_positive_images']} strict `Lung Opacity` positive images, "
        f"{gate['target_boxes']} boxes and "
        f"{gate['images'] - gate['strict_positive_images']:,} strict negatives.",
        "Image percentages do not imply patient-level percentages. Other findings",
        "are not merged into the target. No training, adaptation or external threshold selection.",
        "",
        "These are per-run and equal-run descriptive results. Batch 50 bootstrap",
        "uncertainty and internal/external synthesis have not been performed.",
        "AP uses score support >=0.001; exact-score FROC uses the retained 0.00001",
        "candidate support. Native candidate filters are strict >; common evaluation",
        "uses >=. The NMS/matching IoU is 0.50 and the cap is 100 detections/image.",
        "FROC budget sensitivities use observed points without interpolation.",
        "A floor-limited value is an observed lower bound. No FROC score coordinate",
        "is exported as a deployment threshold.",
        "",
    ]

    def table(headers: list[str], rows: list[list[Any]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        lines.extend("| " + " | ".join(number(v) for v in row) + " |" for row in rows)
        lines.append("")

    lines.extend(["## AP and prespecified FROC sensitivities", ""])
    budgets = [p["fp_per_image_budget"] for p in runs[0]["metrics"]["froc_budgets"]]
    table(
        [
            "Detector",
            "Seed",
            "AP@0.50",
            "AP@0.50:0.95",
            *[f"Sens. at {b:g} FP/image" for b in budgets],
            "Images at cap",
        ],
        [
            [
                m["detector"],
                m["seed"],
                m["ap"]["ap50"],
                m["ap"]["ap50_95"],
                *[p["sensitivity"] for p in m["froc_budgets"]],
                m["images_at_detection_cap"],
            ]
            for m in (r["metrics"] for r in runs)
        ],
    )
    lines.extend(
        [
            "### Equal-run AP summaries",
            "",
            "Arithmetic mean and sample SD; five runs per detector.",
            "",
        ]
    )
    table(
        ["Detector", "Endpoint", "Mean", "Sample SD", "Defined runs"],
        [
            [
                detector,
                metric,
                values[metric]["mean"],
                values[metric]["sample_sd"],
                values[metric]["defined_run_count"],
            ]
            for detector, values in summary["equal_run_descriptive_summaries"].items()
            for metric in ("ap.ap50", "ap.ap50_95")
        ],
    )
    lines.extend(["### Achieved FP/image and candidate-floor limits", ""])
    table(
        [
            "Detector",
            "Seed",
            *[f"Achieved at {b:g}" for b in budgets],
            "Floor FP/image",
            "Floor sensitivity",
            "Floor-limited budgets",
        ],
        [
            [
                m["detector"],
                m["seed"],
                *[p["achieved_fp_per_image"] for p in m["froc_budgets"]],
                m["froc"]["candidate_floor_endpoint_fp_per_image"],
                m["froc"]["candidate_floor_endpoint_sensitivity"],
                ", ".join(
                    f"{p['fp_per_image_budget']:g}"
                    for p in m["froc_budgets"]
                    if p["candidate_floor_limited"]
                )
                or "none",
            ]
            for m in (r["metrics"] for r in runs)
        ],
    )
    for role, heading in (
        ("primary", "Historical n=3 RSNA threshold transport — PRIMARY"),
        ("secondary", "Post-hoc n=5 RSNA threshold transport — SECONDARY sensitivity"),
    ):
        lines.extend(
            [
                f"## {heading}",
                "",
                "Selection provenance and thresholds remain unchanged; "
                "external evaluation uses five runs per detector.",
                "",
            ]
        )
        table(
            [
                "Detector",
                "Seed",
                "Threshold",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "FP/image",
                "Detections/image",
                "% images emitting",
                "Emitting images",
            ],
            [
                [
                    m["detector"],
                    m["seed"],
                    *[
                        p[k]
                        for k in (
                            "threshold",
                            "tp",
                            "fp",
                            "fn",
                            "precision",
                            "recall",
                            "f1",
                            "fp_per_image",
                            "detections_per_image",
                            "percent_images_with_detections",
                            "images_with_detections",
                        )
                    ],
                ]
                for m in (r["metrics"] for r in runs)
                for p in [m["threshold_transport"][role]]
            ],
        )
    lines.extend(
        [
            "## Score distributions at the four prespecified supports",
            "",
            "Emitted-detection descriptions, not calibrated patient probabilities.",
            "Quantiles use the linear method; empty populations remain NA.",
            "",
        ]
    )
    table(
        [
            "Detector",
            "Seed",
            "Support",
            "Count",
            "Min",
            "Q01",
            "Q05",
            "Q25",
            "Median",
            "Q75",
            "Q95",
            "Q99",
            "Mean",
            "Max",
            "Zero-output images",
        ],
        [
            [
                m["detector"],
                m["seed"],
                name,
                p["detection_count"],
                p["min"],
                *[p["quantiles"][q] for q in ("0.01", "0.05", "0.25")],
                p["median"],
                *[p["quantiles"][q] for q in ("0.75", "0.95", "0.99")],
                p["mean"],
                p["max"],
                p["zero_detection_images"],
            ]
            for m in (r["metrics"] for r in runs)
            for name, p in m["score_summaries"].items()
        ],
    )
    lines.extend(
        [
            "## Execution and numerical handoff audit",
            "",
            "The intended RTX 4060 Laptop GPU executed all runs sequentially under explicit AMP.",
            "Historical internal YOLO accuracy used FP32, a disclosed comparison limitation.",
            "One-ULP Torchvision inverse-resize upper-bound overshoots are numerically",
            "canonicalized; raw coordinates are retained privately and replay-verified.",
            "The initial failed attempt and implementation remain hash-preserved privately.",
            "",
        ]
    )
    table(
        [
            "Detector",
            "Seed",
            "Observed convolution dtype",
            "Elapsed seconds",
            "Repaired images",
            "Repaired coordinates",
            "Max correction (pixels)",
        ],
        [
            [
                m["detector"],
                m["seed"],
                m["amp_convolution_dtype"],
                m["elapsed_seconds"],
                m["native_boundary_repair"]["images"],
                m["native_boundary_repair"]["coordinates"],
                m["native_boundary_repair"]["maximum_excess_pixels"],
            ]
            for m in (r["metadata"] for r in runs)
        ],
    )
    lines.extend(
        [
            "Every checkpoint/config/protocol/dataset/environment and bundle hash is in the",
            "source summary. Per-image prediction/count/score evidence remains private.",
            "Equal-run mean/sample SD and defined-run counts for all scalar summaries",
            "are also retained in that JSON; detections are never pooled across runs.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    print(render(parser.parse_args().config))
