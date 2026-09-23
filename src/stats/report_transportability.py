"""Three aggregate figures and a source-bound Batch 50 review document."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _style(value: Any) -> Any:
    """Convert YAML's dash sequence to Matplotlib's tuple convention."""
    return (value[0], tuple(value[1])) if isinstance(value, list) else value


def _save(figure: Any, path: Path, dpi: int) -> None:
    """Render a static publication artifact with deterministic metadata."""
    figure.savefig(
        path, dpi=dpi, facecolor="white", metadata={"Software": "Batch 50 transportability"}
    )
    plt.close(figure)


def plot_froc(
    summary: dict[str, Any],
    plots: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Separate cohort panels: every observed curve, fixed budgets and floor endpoints."""
    settings = config["plot"]
    budgets = protocol["evaluation"]["froc"]["fp_per_image_budgets"]
    figure, axes = plt.subplots(1, 2, figsize=settings["froc_size"], sharey=True)
    for ax, dataset in zip(axes, ("internal", "external"), strict=True):
        for run in plots:
            if run["dataset"] != dataset:
                continue
            color = settings["colors"][run["detector"]]
            index = summary["seeds"].index(run["seed"])
            x, y = np.asarray(run["froc_fp"]), np.asarray(run["froc_sensitivity"])
            # All visible exact events are retained; step rendering is not numerical interpolation.
            keep = x <= settings["froc_maximum_x"]
            if np.any(~keep):
                keep[np.flatnonzero(~keep)[0]] = True
            ax.step(
                x[keep],
                y[keep],
                where="post",
                color=color,
                alpha=0.68,
                linewidth=1.05,
                linestyle=_style(settings["seed_styles"][index]),
            )
            if x[-1] <= settings["froc_maximum_x"]:
                ax.scatter(x[-1], y[-1], color=color, marker="v", s=55, zorder=5)
        for detector in summary["detectors"]:
            rows = [
                r
                for r in summary["per_run"]
                if r["dataset"] == dataset and r["detector"] == detector
            ]
            means = [np.mean([r[f"froc_{b:g}"] for r in rows]) for b in budgets]
            ax.scatter(
                budgets,
                means,
                marker="D",
                color=settings["colors"][detector],
                s=27,
                edgecolors="white",
                linewidths=0.5,
                zorder=6,
            )
        for budget in budgets:
            ax.axvline(budget, color="0.86", linewidth=0.65, zorder=0)
        population = summary["populations"][dataset]
        ax.set_title(
            f"{settings['dataset_labels'][dataset]}\n"
            f"{population['images']:,} images; {population['target_boxes']} target boxes"
        )
        ax.set(
            xlim=(0, settings["froc_maximum_x"]),
            ylim=(settings["minimum_y"], settings["maximum_y"]),
            xlabel="False positives / image",
        )
        ax.grid(axis="y", alpha=0.17)
    axes[0].set_ylabel("Matched target-box sensitivity")
    handles = [
        Line2D([0], [0], color=settings["colors"][d], label=settings["detector_labels"][d])
        for d in summary["detectors"]
    ]
    handles += [
        Line2D([0], [0], color="0.35", linestyle=_style(style), label=f"Seed {seed}")
        for seed, style in zip(summary["seeds"], settings["seed_styles"], strict=True)
    ]
    handles += [
        Line2D(
            [0], [0], color="0.35", marker="D", linestyle="none", label="Equal-run mean at budget"
        ),
        Line2D(
            [0], [0], color="0.35", marker="v", linestyle="none", label="Candidate-floor endpoint"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle(
        "Ranking at fixed false-positive budgets: separate observed frontiers", fontsize=12
    )
    figure.text(
        0.5,
        0.005,
        "Floor 0.00001; cap 100/image. Diamonds may include floor-limited lower bounds; no "
        "extrapolation.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.17, 1, 0.95))
    _save(
        figure,
        Path(config["outputs"]["public_root"]) / config["outputs"]["figures"]["froc"],
        settings["dpi"],
    )


def plot_scores(
    summary: dict[str, Any],
    plots: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Per-run ECDFs at common AP support; count changes accompany conditional distributions."""
    settings = config["plot"]
    figure, axes = plt.subplots(2, 2, figsize=settings["score_size"], sharex=True, sharey=True)
    for i, detector in enumerate(summary["detectors"]):
        for j, dataset in enumerate(("internal", "external")):
            ax = axes[i, j]
            for run in plots:
                if (run["dataset"], run["detector"]) != (dataset, detector):
                    continue
                values = np.asarray(run["ap_scores"])
                if len(values):
                    index = summary["seeds"].index(run["seed"])
                    ax.step(
                        values,
                        np.arange(1, len(values) + 1) / len(values),
                        where="post",
                        color=settings["colors"][detector],
                        linewidth=1.2,
                        alpha=0.8,
                        linestyle=_style(settings["seed_styles"][index]),
                    )
            rows = [
                r
                for r in summary["scores"]
                if r["dataset"] == dataset
                and r["detector"] == detector
                and r["support"] == "ap_candidates"
            ]
            counts = [r["detections_per_image"] for r in rows]
            zeros = [r["percent_zero_detection_images"] for r in rows]
            ax.text(
                0.97,
                0.07,
                f"Detections/image: {min(counts):.2f}-{max(counts):.2f}\n"
                f"Zero-output images: {min(zeros):.1f}-{max(zeros):.1f}%",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            ax.set_title(
                f"{settings['detector_labels'][detector]} | {settings['dataset_labels'][dataset]}"
            )
            ax.set_xscale("log")
            ax.set(xlim=(protocol["evaluation"]["ap_minimum_score"], 1), ylim=(0, 1.01))
            ax.grid(alpha=0.16)
            if j == 0:
                ax.set_ylabel("Fraction of emitted detections")
            if i == 1:
                ax.set_xlabel("Raw confidence (log scale)")
    handles = [
        Line2D([0], [0], color="0.3", linestyle=_style(style), label=f"Seed {seed}")
        for seed, style in zip(summary["seeds"], settings["seed_styles"], strict=True)
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.035)
    )
    figure.suptitle("Score scale and emitted counts at the common 0.001 AP floor", fontsize=12)
    figure.text(
        0.5,
        0.013,
        "Separate per-run ECDFs; no detection pooling. Scores are not calibrated patient "
        "probabilities.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    _save(
        figure,
        Path(config["outputs"]["public_root"]) / config["outputs"]["figures"]["scores"],
        settings["dpi"],
    )


def plot_thresholds(summary: dict[str, Any], config: dict[str, Any]) -> None:
    """Frozen policies: all run points and primary marginal intervals, without n=5 inference."""
    settings = config["plot"]
    figure, axes = plt.subplots(2, 4, figsize=settings["threshold_size"], sharex=True, sharey="col")
    metric_titles = {
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1",
        "fp_per_image": "FP / image",
    }
    for row_index, policy in enumerate(("primary", "secondary")):
        for col, (metric, title) in enumerate(metric_titles.items()):
            ax = axes[row_index, col]
            for dataset_index, dataset in enumerate(("internal", "external")):
                for d, detector in enumerate(summary["detectors"]):
                    x = dataset_index + (d - 0.5) * 0.22
                    color = settings["colors"][detector]
                    rows = [
                        r
                        for r in summary["per_run"]
                        if r["dataset"] == dataset and r["detector"] == detector
                    ]
                    values = [r[f"{policy}_{metric}"] for r in rows]
                    ax.scatter(
                        x + np.linspace(-0.06, 0.06, len(values)),
                        values,
                        color=color,
                        alpha=0.43,
                        s=16,
                        zorder=3,
                    )
                    mean = float(np.mean(values))
                    ax.scatter(
                        x,
                        mean,
                        marker="D" if policy == "primary" else "s",
                        color=color,
                        s=31,
                        zorder=5,
                    )
                    if policy == "primary":
                        interval = next(
                            r
                            for r in summary["intervals"]
                            if r["dataset"] == dataset
                            and r["estimand"] == "training_procedure"
                            and r["metric"] == f"primary_{metric}"
                        )
                        prefix = "a" if d == 0 else "b"
                        lo, hi = interval[f"{prefix}_ci_low"], interval[f"{prefix}_ci_high"]
                        # Percentile intervals need not contain the point estimate.
                        ax.plot([x, x], [lo, hi], color=color, linewidth=1.4, zorder=4)
                        ax.plot([x - 0.025, x + 0.025], [lo, lo], color=color, linewidth=1)
                        ax.plot([x - 0.025, x + 0.025], [hi, hi], color=color, linewidth=1)
            ax.set_title(title)
            ax.set(
                xticks=[0, 1],
                xticklabels=["RSNA", "VinDr"],
                xlim=(-0.35, 1.35),
                ylim=(settings["minimum_y"], settings["maximum_y"]),
            )
            ax.grid(axis="y", alpha=0.17)
        axes[row_index, 0].set_ylabel(
            "Historical n=3 selection\n95% marginal intervals"
            if policy == "primary"
            else "Post-hoc n=5 selection\nDescriptive sensitivity"
        )
    handles = [
        Line2D(
            [0], [0], color=settings["colors"][d], marker="D", label=settings["detector_labels"][d]
        )
        for d in summary["detectors"]
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035)
    )
    figure.suptitle(
        "RSNA-frozen thresholds applied unchanged to all five runs on each dataset", fontsize=12
    )
    figure.text(
        0.5,
        0.012,
        "Dots: retained runs. Larger symbols: equal-run means. Primary intervals: shared "
        "observations, independent detector runs.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    _save(
        figure,
        Path(config["outputs"]["public_root"]) / config["outputs"]["figures"]["thresholds"],
        settings["dpi"],
    )


def number(value: Any) -> str:
    """Preserve small adverse results and render missing quantities explicitly."""
    return "NA" if value is None or not np.isfinite(value) else f"{value:.6f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a compact deterministic Markdown table."""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows],
        ]
    )


def report(summary: dict[str, Any], config: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Generate every displayed number from the frozen aggregate analysis."""
    out = config["outputs"]
    report_path = Path(out["report"])
    root = Path(out["public_root"])

    def link(path: Path) -> str:
        return Path(os.path.relpath(path, report_path.parent)).as_posix()

    def mean_sd(row: dict[str, Any], prefix: str) -> str:
        return f"{number(row[prefix + '_mean'])} +/- {number(row[prefix + '_sd'])}"

    def interval(row: dict[str, Any], prefix: str) -> str:
        return (
            f"{number(row[prefix + '_estimate'])} "
            f"[{number(row[prefix + '_ci_low'])}, {number(row[prefix + '_ci_high'])}]"
        )

    c = {r["metric"]: r for r in summary["comparison"]}
    lines = [
        "# VinDr statistics and cross-dataset transportability — Batch 50",
        "",
        f"Generated from [{out['summary']}]({link(root / out['summary'])}). "
        "Methods and estimands: [VINDR_STATISTICS.md](VINDR_STATISTICS.md).",
        "",
        "## Central question",
        "",
        "**Detector ordering and RSNA-selected operating points do not transport in the same "
        "sense.** "
        "The observed equal-run AP and fixed-budget FROC ordering remains Faster R-CNN above "
        "YOLO11s, "
        "while both pipelines lose substantial absolute performance and the frozen thresholds "
        "yield "
        "very low external recall. Preserved ordering is not preserved performance. "
        "The detector gaps and the evidence about them are reported separately below.",
        "",
        "No ontology, threshold, checkpoint, score floor or detection cap was tuned. "
        "All five runs per detector, including seed 271, are retained. "
        "RSNA and VinDr observations and predictions are never pooled. "
        "Changes describe **cross-dataset transportability**; acquisition, institution, country, "
        "patient population, annotation ontology and numerical implementation are confounded. "
        "No specific factor is identified as the cause.",
        "",
        "## Population and estimands",
        "",
    ]
    lines += [
        table(
            [
                "Dataset",
                "Images",
                "Positive images",
                "Target boxes",
                "Patient groups",
                "Resampling unit",
            ],
            [
                [
                    d,
                    p["images"],
                    p["positive_images"],
                    p["target_boxes"],
                    p["patient_groups"] if p["patient_groups"] is not None else "Unavailable",
                    p["resampling_unit"],
                ]
                for d in ("internal", "external")
                for p in (summary["populations"][d],)
            ],
        ),
        "",
        f"Primary intervals use {protocol['statistics']['bootstrap_resamples']:,} draws, "
        "paired observation sampling across both detectors and all runs, and independent "
        "within-detector "
        "trained-run resampling. RSNA moves all studies from a known NIH patient together. "
        "VinDr uses released images because no defensible patient linkage exists; unobserved "
        "repeat patients "
        "may make those intervals too narrow. Seed-17 checkpoint-conditional intervals resample "
        "only observations. "
        "All intervals are marginal 95% percentile intervals, conditional on the frozen "
        "thresholds and retained "
        "candidate support. There are no new hypothesis tests or simultaneous guarantees. "
        "New internal intervals here are aligned Batch 50 reconstructions, not replacements for "
        "historical outputs.",
        "",
        "## A. Ranking and B. false-positive-regime transport",
        "",
        "Equal-run means +/- sample SD (n=5 each); A = Faster R-CNN, B = YOLO11s. "
        "AP uses the common 0.001 support; FROC uses exact scores at the frozen 0.00001 "
        "collection floor. "
        "Budget sensitivities are observed maxima at or below the same FP/image budget, without "
        "interpolation.",
        "",
    ]
    metrics = [
        "ap50",
        "ap50_95",
        *[f"froc_{b:g}" for b in protocol["evaluation"]["froc"]["fp_per_image_budgets"]],
    ]
    lines += [
        table(
            ["Endpoint", "RSNA A", "RSNA B", "VinDr A", "VinDr B", "Ordering", "Gap"],
            [
                [
                    m,
                    *[
                        mean_sd(c[m], p)
                        for p in ("internal_a", "internal_b", "external_a", "external_b")
                    ],
                    c[m]["ordering_change"],
                    c[m]["gap_change"],
                ]
                for m in metrics
            ],
        ),
        "",
        "The change labels compare the signed A-minus-B equal-run gaps: **unchanged** within the "
        "configured numerical tolerance; **strengthened** if the magnitude increases without a "
        "sign reversal; "
        "**weakened** if it decreases; **reversed** if its sign flips. These are descriptive "
        "labels, "
        "not significance decisions or tests of a dataset interaction. A smaller raw AP gap "
        "near zero "
        "does not mean that the pipelines have become clinically equivalent.",
        "",
        "### Candidate support remains incomplete",
        "",
    ]
    lines += [
        table(
            [
                "Budget",
                "RSNA YOLO limited runs",
                "VinDr YOLO limited runs",
                "VinDr observed A",
                "VinDr observed B",
                "VinDr B conservative upper",
            ],
            [
                [
                    m.removeprefix("froc_"),
                    c[m]["internal_b_floor_limited_runs"],
                    c[m]["external_b_floor_limited_runs"],
                    number(c[m]["external_a_mean"]),
                    number(c[m]["external_b_mean"]),
                    number(c[m]["external_b_conservative_upper"]),
                ]
                for m in metrics
                if m.startswith("froc_")
            ],
        ),
        "",
        "The conservative upper replaces each floor-limited run's observed sensitivity with 1.0; "
        "it is a missing-support bound, not a confidence interval. Unlike the preserved internal "
        "no-reversal bound at 2 FP/image, the external upper bounds at 1 and 2 FP/image exceed the "
        "observed Faster R-CNN mean. Thus the external ordering is established only on the "
        "retained "
        "candidate support; reversal beyond that support cannot be ruled out. No further floor "
        "change is made. "
        "Substantial Faster R-CNN detection-cap saturation is also retained and reported per run.",
        "",
        "![Separate exact-score FROC panels, all five runs and floor endpoints.]"
        f"({link(root / out['figures']['froc'])})",
        "",
        "## Primary uncertainty on VinDr",
        "",
        "Estimate [marginal 95% CI]; the contrast is A minus B. FROC intervals describe the "
        "observed, "
        "support-limited estimand, not an untruncated frontier.",
        "",
    ]
    rows = [
        r
        for r in summary["intervals"]
        if r["dataset"] == "external" and r["estimand"] == "training_procedure"
    ]
    lines += [
        table(
            ["Endpoint", "A", "B", "A minus B", "Valid draws A/B/difference"],
            [
                [
                    r["metric"],
                    interval(r, "a"),
                    interval(r, "b"),
                    interval(r, "difference"),
                    f"{r['a_valid']}/{r['b_valid']}/{r['difference_valid']}",
                ]
                for r in rows
            ],
        ),
        "",
        "Intervals that include zero do not resolve a detector difference under this "
        "training-procedure "
        "estimand. A positive mean alone does not establish a robust detector advantage. "
        "Undefined draws are not retried or imputed; all counts are in the interval table.",
        "",
        "## C. Score scale and detection counts",
        "",
        "Each row first summarizes each run's emitted detections at >=0.001, then weights the "
        "five runs "
        "equally. Counts are also normalized by cohort size. Confidence is not a calibrated "
        "patient probability.",
        "",
    ]
    score_metrics = [
        "ap_scores_mean",
        "ap_scores_median",
        "ap_scores_detections_per_image",
        "ap_scores_percent_zero_detection_images",
    ]
    lines += [
        table(
            ["Endpoint", "RSNA A", "RSNA B", "VinDr A", "VinDr B", "Gap change"],
            [
                [
                    m,
                    *[
                        mean_sd(c[m], p)
                        for p in ("internal_a", "internal_b", "external_a", "external_b")
                    ],
                    c[m]["gap_change"],
                ]
                for m in score_metrics
            ],
        ),
        "",
        "All four frozen score supports, full quantiles, per-run detection counts and defined-run "
        "counts are preserved in the score CSV and summary. Empty emitted populations have "
        "count zero "
        "and undefined score summaries; they are not assigned zero confidence or omitted from "
        "run inventories. "
        "The all-retained support uses Batch 42 v4 internally; AP and threshold supports use "
        "the original "
        "internal bundles. Historical YOLO internal inference was FP32, whereas external "
        "inference used "
        "verified bfloat16 AMP, limiting attribution of score shifts.",
        "",
        "![Per-run score distributions at the common AP support, with detection-count ranges.]"
        f"({link(root / out['figures']['scores'])})",
        "",
        "## D. Frozen operating points and n=3 versus n=5 sensitivity",
        "",
        "The primary historical policy selected thresholds from three RSNA validation runs; the "
        "secondary "
        "policy is Batch 43's post-hoc five-run validation sensitivity. Each is evaluated on "
        "five runs "
        "per detector in both datasets, with unchanged numerical thresholds. The secondary "
        "policy remains descriptive.",
        "",
    ]
    lines += [
        table(
            ["Policy", "Faster R-CNN threshold", "YOLO11s threshold", "Validation runs"],
            [
                [
                    p,
                    protocol["threshold_transport"][p]["thresholds"]["faster_rcnn"],
                    protocol["threshold_transport"][p]["thresholds"]["yolo11s"],
                    len(protocol["threshold_transport"][p]["selection_seeds"]),
                ]
                for p in ("primary", "secondary")
            ],
        ),
        "",
        table(
            ["Endpoint", "RSNA A", "RSNA B", "VinDr A", "VinDr B", "Cross-dataset gap"],
            [
                [
                    m,
                    *[
                        mean_sd(c[m], p)
                        for p in ("internal_a", "internal_b", "external_a", "external_b")
                    ],
                    c[m]["gap_change"],
                ]
                for m in c
                if m.startswith(("primary_", "secondary_"))
            ],
        ),
        "",
        table(
            ["Dataset", "Metric", "Historical n=3 gap", "Post-hoc n=5 gap", "Policy change"],
            [
                [
                    r["dataset"],
                    r["metric"],
                    number(r["historical_n3_gap"]),
                    number(r["posthoc_n5_gap"]),
                    r["gap_change"],
                ]
                for r in summary["policy_comparison"]
            ],
        ),
        "",
        "Lower FP/image alone does not establish improvement when recall and emission rates "
        "collapse. "
        "The historical-versus-secondary FP/image ordering reverses within both cohorts. "
        "Neither policy restores external recall. YOLO seed 271 emits nothing at its historical "
        "threshold "
        "in either dataset; its zero-valued defined metrics remain in all five-run summaries. "
        "No threshold selection uncertainty or clinical utility is inferred.",
        "",
        "![All-run frozen-threshold transport with primary marginal intervals "
        "and separate descriptive sensitivity.]"
        f"({link(root / out['figures']['thresholds'])})",
        "",
        "## Checkpoint-conditional sensitivity and complete provenance",
        "",
        "The following intervals hold the seed-17 checkpoints fixed and resample only common "
        "external images. "
        "They are secondary; their narrower or differently centered contrasts cannot replace "
        "the primary "
        "training-procedure estimand.",
        "",
    ]
    rows = [
        r
        for r in summary["intervals"]
        if r["dataset"] == "external" and r["estimand"] == "checkpoint_conditional_seed17"
    ]
    lines += [
        table(
            ["Endpoint", "A", "B", "A minus B"],
            [
                [r["metric"], interval(r, "a"), interval(r, "b"), interval(r, "difference")]
                for r in rows
            ],
        ),
        "",
        f"- [All 20 run rows, support flags, and both policies]({link(root / out['per_run'])})",
        "- [All marginal intervals for both separate datasets and estimands]"
        f"({link(root / out['intervals'])})",
        f"- [Per-run score/count summaries]({link(root / out['scores'])})",
        "- [Internal/external contrasts and change classifications]"
        f"({link(root / out['comparison'])})",
        f"- [Historical/post-hoc policy contrasts]({link(root / out['policy_comparison'])})",
        "",
        "The summary binds every input, code/config dependency, bootstrap-array artifact, "
        "aggregate table "
        "and figure by SHA-256. Exact commands, including full deterministic bootstrap replay, "
        "are in "
        "[README](../README.md#vindr-statistics-and-cross-dataset-transportability-batch-50). "
        "Private image-linked inputs and plot/bootstrap working files remain in ignored local "
        "storage. "
        "No manuscript rewrite, new inference, training, commit or publication is part of Batch "
        "50.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def render(
    summary: dict[str, Any],
    plots: list[dict[str, Any]],
    config: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Render only the three prespecified, nonidentifying information figures."""
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        plot_froc(summary, plots, config, protocol)
        plot_scores(summary, plots, config, protocol)
        plot_thresholds(summary, config)
    report(summary, config, protocol)
