"""Offline command-line checks for the benchmark foundation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from meddet_benchmark.config import assert_run_allowed, config_fingerprint, load_experiment
from meddet_benchmark.data_audit import audit_yolo_dataset
from meddet_benchmark.reproducibility import configure_reproducibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m meddet_benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a configuration and run gate")
    validate.add_argument("config", type=Path)
    validate.add_argument("--operation", choices=("smoke", "train", "test"), default="smoke")

    smoke = subparsers.add_parser("smoke", help="run the deterministic offline smoke check")
    smoke.add_argument("config", type=Path)

    audit = subparsers.add_parser("audit-data", help="audit a local YOLO dataset")
    audit.add_argument("root", type=Path)
    audit.add_argument("--class-name", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-data":
        report = audit_yolo_dataset(args.root, class_names=tuple(args.class_name))
        print(json.dumps(report, allow_nan=False, sort_keys=True))
        return 0 if report["passed"] else 2

    config = load_experiment(args.config)
    operation = args.operation if args.command == "validate" else "smoke"
    assert_run_allowed(config, operation)

    output: dict = {
        "experiment_id": config.experiment_id,
        "status": config.status,
        "operation": operation,
        "config_sha256": config_fingerprint(config),
    }
    if args.command == "smoke":
        output["reproducibility"] = asdict(
            configure_reproducibility(
                config.seeds[0],
                deterministic=config.runtime.deterministic,
                allow_tf32=config.runtime.allow_tf32,
            )
        )
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
