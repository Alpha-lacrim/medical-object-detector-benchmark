"""Verify manuscript numerical claims against frozen machine-readable sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class PaperClaimVerificationError(RuntimeError):
    """Raised after collecting every manuscript-claim verification failure."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot parse YAML {path}: {error}") from error


def _relative_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative string")
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{label} escapes or is not normalized: {value!r}")
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes project root: {value!r}") from error
    return candidate


def _number(value: Any, *, label: str) -> float:
    try:
        result = float(str(value).replace(",", ""))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _scalar_filters(filters: Any, *, label: str) -> dict[str, str]:
    if not isinstance(filters, dict) or not all(
        isinstance(key, str) and isinstance(value, (str, int, float, bool))
        for key, value in filters.items()
    ):
        raise ValueError(f"{label} must be a scalar mapping")
    return {key: str(value) for key, value in filters.items()}


def _filtered_csv_rows(
    path: Path,
    filters: Any,
    exclude_filters: Any,
    *,
    label: str,
) -> list[dict[str, str]]:
    expected = _scalar_filters(filters, label=f"{label}.filters")
    excluded = _scalar_filters(exclude_filters, label=f"{label}.exclude_filters")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot parse CSV {path}: {error}") from error
    return [
        row
        for row in rows
        if all(row.get(key) == value for key, value in expected.items())
        and not (excluded and all(row.get(key) == value for key, value in excluded.items()))
    ]


def _json_pointer(payload: Any, pointer: Any, *, label: str) -> Any:
    if not isinstance(pointer, list) or not pointer:
        raise ValueError(f"{label}.pointer must be a non-empty key/index list")
    value = payload
    for part in pointer:
        mapping_match = isinstance(value, dict) and isinstance(part, str) and part in value
        sequence_match = (
            isinstance(value, list) and isinstance(part, int) and 0 <= part < len(value)
        )
        if mapping_match or sequence_match:
            value = value[part]
        else:
            raise ValueError(f"{label}.pointer cannot resolve component {part!r}")
    return value


def _source_paths(source: Any) -> list[str]:
    if not isinstance(source, dict):
        return []
    paths: list[str] = []
    path = source.get("path")
    if isinstance(path, str):
        paths.append(path)
    operands = source.get("operands", [])
    if isinstance(operands, list):
        for operand in operands:
            paths.extend(_source_paths(operand))
    return paths


def evaluate_source(source: Any, *, root: Path, label: str) -> float:
    """Evaluate one allow-listed deterministic source specification."""

    if not isinstance(source, dict):
        raise ValueError(f"{label} must be an object")
    kind = source.get("kind")
    if kind in {"csv_cell", "csv_mean", "csv_row_count"}:
        path = _relative_path(root, source.get("path"), label=f"{label}.path")
        if not path.is_file():
            raise ValueError(f"{label}.path is missing: {source.get('path')}")
        rows = _filtered_csv_rows(
            path,
            source.get("filters", {}),
            source.get("exclude_filters", {}),
            label=label,
        )
        if kind == "csv_row_count":
            return float(len(rows))
        if kind == "csv_mean":
            if not rows:
                raise ValueError(f"{label} expected one or more CSV rows, found none")
            column = source.get("column")
            if not isinstance(column, str) or column not in rows[0]:
                raise ValueError(f"{label}.column is absent: {column!r}")
            values = [
                _number(row[column], label=f"{label}.{column}[{index}]")
                for index, row in enumerate(rows)
            ]
            return sum(values) / len(values)
        if len(rows) != 1:
            raise ValueError(f"{label} expected exactly one CSV row, found {len(rows)}")
        column = source.get("column")
        if not isinstance(column, str) or column not in rows[0]:
            raise ValueError(f"{label}.column is absent: {column!r}")
        return _number(rows[0][column], label=f"{label}.{column}")
    if kind == "json_value":
        path = _relative_path(root, source.get("path"), label=f"{label}.path")
        if not path.is_file():
            raise ValueError(f"{label}.path is missing: {source.get('path')}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot parse JSON {path}: {error}") from error
        value = _json_pointer(payload, source.get("pointer"), label=label)
        return _number(value, label=label)
    if kind == "calculation":
        operation = source.get("operation")
        operands = source.get("operands")
        if not isinstance(operands, list) or len(operands) != 2:
            raise ValueError(f"{label}.operands must contain exactly two sources")
        left = evaluate_source(operands[0], root=root, label=f"{label}.operands[0]")
        right = evaluate_source(operands[1], root=root, label=f"{label}.operands[1]")
        if operation == "divide":
            if right == 0:
                raise ValueError(f"{label} division by zero")
            result = left / right
        elif operation == "subtract":
            result = left - right
        elif operation == "percent_reduction":
            if left == 0:
                raise ValueError(f"{label} percent reduction has zero baseline")
            result = 100.0 * (left - right) / left
        elif operation == "add":
            result = left + right
        elif operation == "multiply":
            result = left * right
        else:
            raise ValueError(f"{label}.operation is unsupported: {operation!r}")
        scale = source.get("scale", 1.0)
        return result * _number(scale, label=f"{label}.scale")
    raise ValueError(f"{label}.kind is unsupported: {kind!r}")


def _artifact_paths(manifest_path: Path) -> set[str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot parse scientific artifact manifest {manifest_path}: {error}"
        ) from error
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(artifacts, list):
        raise ValueError("scientific artifact manifest has no artifacts list")
    return {
        artifact["path"]
        for artifact in artifacts
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }


def _verify_manuscript_semantic_guards(
    guards: Any, manuscript_text: str, *, errors: list[str]
) -> None:
    """Apply declarative required/forbidden regex guards to manuscript prose."""

    if guards is None:
        return
    if not isinstance(guards, list):
        errors.append("manuscript_semantic_guards must be a list")
        return

    guard_ids: list[str] = []
    for index, guard in enumerate(guards):
        label = f"manuscript_semantic_guards[{index}]"
        if not isinstance(guard, dict):
            errors.append(f"{label} must be an object")
            continue
        guard_id = guard.get("id")
        if not isinstance(guard_id, str) or not guard_id:
            errors.append(f"{label}.id must be a non-empty string")
            guard_id = label
        guard_ids.append(guard_id)

        required = guard.get("required_regex")
        forbidden = guard.get("forbidden_regex")
        if (required is None) == (forbidden is None):
            errors.append(
                f"manuscript guard {guard_id} must define exactly one of "
                "required_regex or forbidden_regex"
            )
            continue
        pattern = required if required is not None else forbidden
        if not isinstance(pattern, str) or not pattern:
            errors.append(f"manuscript guard {guard_id} regex must be a non-empty string")
            continue
        try:
            matched = re.search(pattern, manuscript_text, flags=re.MULTILINE | re.DOTALL)
        except re.error as error:
            errors.append(f"manuscript guard {guard_id} has invalid regex: {error}")
            continue
        if required is not None and matched is None:
            errors.append(f"manuscript guard {guard_id} required pattern was not found")
        if forbidden is not None and matched is not None:
            errors.append(f"manuscript guard {guard_id} forbidden pattern was found")

    duplicates = sorted({guard_id for guard_id in guard_ids if guard_ids.count(guard_id) > 1})
    if duplicates:
        errors.append(f"duplicate manuscript semantic guard ids: {', '.join(duplicates)}")


def verify_claims(manifest_path: Path, *, project_root: Path | None = None) -> int:
    """Verify one claim manifest and return its checked claim count."""

    manifest_path = manifest_path.resolve()
    root = (project_root or manifest_path.parents[1]).resolve()
    errors: list[str] = []
    try:
        manifest = _load_yaml(manifest_path)
    except ValueError as error:
        raise PaperClaimVerificationError([str(error)]) from error
    if not isinstance(manifest, dict):
        raise PaperClaimVerificationError(["paper claim manifest top level must be an object"])
    if manifest.get("schema_version") != 1:
        errors.append("paper claim manifest schema_version must equal 1")
    try:
        manuscript_path = _relative_path(root, manifest.get("manuscript"), label="manuscript")
        manuscript_text = manuscript_path.read_text(encoding="utf-8")
    except (ValueError, OSError, UnicodeError) as error:
        errors.append(f"cannot load manuscript: {error}")
        manuscript_text = ""
    _verify_manuscript_semantic_guards(
        manifest.get("manuscript_semantic_guards"), manuscript_text, errors=errors
    )
    scientific_manifest_value = manifest.get("scientific_artifact_manifest")
    try:
        scientific_manifest_path = _relative_path(
            root,
            scientific_manifest_value,
            label="scientific_artifact_manifest",
        )
        allowed_source_paths = _artifact_paths(scientific_manifest_path)
    except ValueError as error:
        errors.append(str(error))
        allowed_source_paths = set()

    claims = manifest.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("paper claim manifest claims must be a non-empty list")
        raise PaperClaimVerificationError(errors)
    claim_ids: list[str] = []
    for index, claim in enumerate(claims):
        label = f"claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{label} must be an object")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id:
            errors.append(f"{label}.id must be a non-empty string")
            claim_id = label
        claim_ids.append(claim_id)
        claim_label = f"claim {claim_id}"
        source = claim.get("source")
        unbound_paths = sorted(set(_source_paths(source)) - allowed_source_paths)
        if unbound_paths:
            errors.append(
                f"{claim_label} uses sources absent from the scientific artifact manifest: "
                + ", ".join(unbound_paths)
            )
        try:
            source_value = evaluate_source(source, root=root, label=f"{claim_label}.source")
        except ValueError as error:
            errors.append(str(error))
            continue
        manuscript = claim.get("manuscript")
        if not isinstance(manuscript, dict):
            errors.append(f"{claim_label}.manuscript must be an object")
            continue
        pattern = manuscript.get("regex")
        if not isinstance(pattern, str) or "(?P<value>" not in pattern:
            errors.append(f"{claim_label}.manuscript.regex must contain named group 'value'")
            continue
        try:
            matches = list(re.finditer(pattern, manuscript_text, flags=re.MULTILINE | re.DOTALL))
        except re.error as error:
            errors.append(f"{claim_label} has invalid manuscript regex: {error}")
            continue
        if len(matches) != 1:
            errors.append(
                f"{claim_label} manuscript regex expected one match, found {len(matches)}"
            )
            continue
        try:
            manuscript_value = _number(
                matches[0].group("value"), label=f"{claim_label}.manuscript value"
            )
            tolerance = _number(claim.get("absolute_tolerance"), label=f"{claim_label}.tolerance")
        except (IndexError, ValueError) as error:
            errors.append(str(error))
            continue
        if tolerance < 0:
            errors.append(f"{claim_label}.absolute_tolerance must be non-negative")
            continue
        difference = abs(manuscript_value - source_value)
        if difference > tolerance + 1e-15:
            errors.append(
                f"{claim_label} mismatch: manuscript {manuscript_value}, source {source_value}, "
                f"absolute difference {difference} exceeds tolerance {tolerance}"
            )
    duplicates = sorted({claim_id for claim_id in claim_ids if claim_ids.count(claim_id) > 1})
    if duplicates:
        errors.append(f"duplicate claim ids: {', '.join(duplicates)}")
    if errors:
        raise PaperClaimVerificationError(errors)
    return len(claims)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("report/paper_claim_sources.yaml"),
        help="paper claim-to-source manifest path",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repository root (defaults to the manifest's grandparent)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the paper-claim verifier CLI."""

    args = _parser().parse_args(argv)
    try:
        count = verify_claims(args.manifest, project_root=args.project_root)
    except PaperClaimVerificationError as error:
        print("Paper claim verification failed:", file=sys.stderr)
        for item in error.errors:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(
        f"Paper claim verification passed: {count} numerical claims and all configured "
        "semantic guards."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
