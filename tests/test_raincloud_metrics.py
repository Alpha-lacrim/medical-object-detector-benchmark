import csv
from pathlib import Path

import pytest

from src.plot_raincloud_metrics import (
    RaincloudConfig,
    load_and_audit_observations,
    write_summary,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _config(summary: Path, per_seed: Path, figure: Path, provenance: Path) -> RaincloudConfig:
    return RaincloudConfig.model_validate(
        {
            "schema_version": 1,
            "analysis_id": "test-raincloud",
            "seed": 17,
            "inputs": {"summary_table": str(summary), "per_seed_table": str(per_seed)},
            "detectors": [
                {
                    "key": "a",
                    "summary_prefix": "a",
                    "label": "A",
                    "color": "#123456",
                },
                {
                    "key": "b",
                    "summary_prefix": "b",
                    "label": "B",
                    "color": "#654321",
                },
            ],
            "metrics": [
                {
                    "key": "score",
                    "label": "Score",
                    "group": "predictive",
                    "panel_index": 0,
                    "display_unit": "ratio",
                    "scale": 1.0,
                }
            ],
            "plot": {
                "rows": 1,
                "columns": 1,
                "width_inches": 4.0,
                "height_inches": 3.0,
                "dpi": 80,
                "violin_bandwidth_adjust": 0.7,
                "violin_width": 0.8,
                "box_width": 0.2,
                "point_size": 4.0,
                "point_jitter": 0.05,
                "panel_title_size": 10.0,
                "axis_label_size": 9.0,
                "tick_label_size": 8.0,
                "sample_label_size": 7.0,
            },
            "outputs": {"figure": str(figure), "summary_json": str(provenance)},
        }
    )


def _tables(tmp_path: Path) -> tuple[Path, Path]:
    summary = tmp_path / "summary.csv"
    per_seed = tmp_path / "per_seed.csv"
    _write_csv(
        per_seed,
        ["detector", "seed", "score"],
        [
            {"detector": "a", "seed": 1, "score": 1.0},
            {"detector": "a", "seed": 2, "score": 3.0},
            {"detector": "b", "seed": 1, "score": 2.0},
            {"detector": "b", "seed": 2, "score": ""},
        ],
    )
    _write_csv(
        summary,
        [
            "metric",
            "a_mean",
            "a_std",
            "a_n",
            "a_attempted_n",
            "a_undefined_seeds",
            "a_undefined_reason",
            "b_mean",
            "b_std",
            "b_n",
            "b_attempted_n",
            "b_undefined_seeds",
            "b_undefined_reason",
            "sample_size_note",
        ],
        [
            {
                "metric": "score",
                "a_mean": 2.0,
                "a_std": 2**0.5,
                "a_n": 2,
                "a_attempted_n": 2,
                "a_undefined_seeds": "",
                "a_undefined_reason": "",
                "b_mean": 2.0,
                "b_std": 0.0,
                "b_n": 1,
                "b_attempted_n": 2,
                "b_undefined_seeds": "2",
                "b_undefined_reason": "no_output",
                "sample_size_note": "conditional endpoint",
            }
        ],
    )
    return summary, per_seed


def test_audit_preserves_actual_metric_specific_n(tmp_path: Path) -> None:
    summary, per_seed = _tables(tmp_path)
    config = _config(summary, per_seed, tmp_path / "figure.png", tmp_path / "summary.json")

    observations, audits = load_and_audit_observations(config)

    assert len(observations) == 3
    assert audits[0].counts == {"a": 2, "b": 1}
    assert audits[0].attempted_counts == {"a": 2, "b": 2}
    assert audits[0].undefined_seeds["b"] == "2"


def test_audit_rejects_summary_drift(tmp_path: Path) -> None:
    summary, per_seed = _tables(tmp_path)
    rows = list(csv.DictReader(summary.open("r", encoding="utf-8", newline="")))
    rows[0]["a_mean"] = "9.0"
    _write_csv(summary, list(rows[0]), rows)
    config = _config(summary, per_seed, tmp_path / "figure.png", tmp_path / "summary.json")

    with pytest.raises(ValueError, match="aggregate mismatch"):
        load_and_audit_observations(config)


def test_summary_uses_clone_stable_lf_newlines(tmp_path: Path) -> None:
    summary, per_seed = _tables(tmp_path)
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"deterministic-test-figure")
    provenance = tmp_path / "summary.json"
    config = _config(summary, per_seed, figure, provenance)
    _, audits = load_and_audit_observations(config)

    write_summary(config, audits, figure)

    assert b"\r\n" not in provenance.read_bytes()
