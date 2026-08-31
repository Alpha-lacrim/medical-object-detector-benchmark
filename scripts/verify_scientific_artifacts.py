"""Verify the frozen, manuscript-critical scientific artifact inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ArtifactVerificationError(RuntimeError):
    """Raised after collecting every scientific artifact verification failure."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class VerificationReport:
    """Counts from a successful artifact-manifest verification."""

    artifact_count: int
    checked_input_count: int
    skipped_external_input_count: int
    referenced_result_count: int


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse JSON {path}: {error}") from error


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


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))


def _verify_hash_binding(
    root: Path,
    path_value: Any,
    expected_hash: Any,
    *,
    label: str,
    required: bool,
    errors: list[str],
) -> tuple[bool, bool]:
    """Verify one path/hash pair and return ``(checked, skipped)``."""

    try:
        path = _relative_path(root, path_value, label=f"{label}.path")
    except ValueError as error:
        errors.append(str(error))
        return False, False
    if not _valid_sha256(expected_hash):
        errors.append(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
        return False, False
    if not path.is_file():
        if required:
            errors.append(f"{label} is missing: {path_value}")
            return False, False
        return False, True
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        errors.append(f"{label} is stale: {path_value} expected {expected_hash}, got {actual_hash}")
    return True, False


def _verify_csv_schema(path: Path, schema: dict[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        return [f"{label} cannot be parsed as CSV: {error}"]
    if not rows:
        return [f"{label} is an empty CSV"]
    header = rows[0]
    if len(header) != len(set(header)):
        errors.append(f"{label} has duplicate CSV column names")
    required_columns = schema.get("required_columns")
    if not isinstance(required_columns, list) or not all(
        isinstance(column, str) and column for column in required_columns
    ):
        errors.append(f"{label} expected_schema.required_columns must be a string list")
    else:
        missing = [column for column in required_columns if column not in header]
        if missing:
            errors.append(f"{label} is missing required columns: {', '.join(missing)}")
    exact_columns = schema.get("exact_columns")
    if exact_columns is not None and header != exact_columns:
        errors.append(f"{label} CSV header/order differs from the frozen schema")
    expected_rows = schema.get("row_count")
    if not isinstance(expected_rows, int) or expected_rows < 0:
        errors.append(f"{label} expected_schema.row_count must be a non-negative integer")
    elif len(rows) - 1 != expected_rows:
        errors.append(f"{label} row count expected {expected_rows}, got {len(rows) - 1}")
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            errors.append(f"{label} row {row_number} has {len(row)} fields; expected {len(header)}")
            break
    return errors


def _verify_json_schema(path: Path, schema: dict[str, Any], *, label: str) -> list[str]:
    try:
        payload = _load_json(path)
    except ValueError as error:
        return [f"{label}: {error}"]
    expected_type = schema.get("top_level_type")
    if expected_type == "object" and not isinstance(payload, dict):
        return [f"{label} JSON top level must be an object"]
    if expected_type == "array" and not isinstance(payload, list):
        return [f"{label} JSON top level must be an array"]
    if expected_type not in {"object", "array"}:
        return [f"{label} expected_schema.top_level_type must be object or array"]
    required_keys = schema.get("required_top_level_keys", [])
    if not isinstance(required_keys, list) or not all(
        isinstance(key, str) for key in required_keys
    ):
        return [f"{label} expected_schema.required_top_level_keys must be a string list"]
    if isinstance(payload, dict):
        missing = [key for key in required_keys if key not in payload]
        if missing:
            return [f"{label} JSON is missing top-level keys: {', '.join(missing)}"]
    return []


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        prefix = handle.read(24)
    if len(prefix) != 24 or prefix[:8] != _PNG_SIGNATURE or prefix[12:16] != b"IHDR":
        raise ValueError("invalid PNG signature or IHDR")
    return struct.unpack(">II", prefix[16:24])


def _verify_png_schema(path: Path, schema: dict[str, Any], *, label: str) -> list[str]:
    try:
        width, height = _png_dimensions(path)
    except (OSError, ValueError, struct.error) as error:
        return [f"{label} cannot be parsed as PNG: {error}"]
    errors: list[str] = []
    if width != schema.get("width"):
        errors.append(f"{label} PNG width expected {schema.get('width')}, got {width}")
    if height != schema.get("height"):
        errors.append(f"{label} PNG height expected {schema.get('height')}, got {height}")
    return errors


def _verify_schema(path: Path, schema: Any, *, label: str) -> list[str]:
    if not isinstance(schema, dict):
        return [f"{label}.expected_schema must be an object"]
    schema_type = schema.get("type")
    if schema_type == "csv":
        return _verify_csv_schema(path, schema, label=label)
    if schema_type == "json":
        return _verify_json_schema(path, schema, label=label)
    if schema_type == "png":
        return _verify_png_schema(path, schema, label=label)
    return [f"{label}.expected_schema.type is unsupported: {schema_type!r}"]


def verify_manifest(manifest_path: Path, *, project_root: Path | None = None) -> VerificationReport:
    """Verify one scientific artifact manifest or raise with all detected failures."""

    manifest_path = manifest_path.resolve()
    root = (project_root or manifest_path.parents[1]).resolve()
    errors: list[str] = []
    try:
        manifest = _load_json(manifest_path)
    except ValueError as error:
        raise ArtifactVerificationError([str(error)]) from error
    if not isinstance(manifest, dict):
        raise ArtifactVerificationError(
            ["scientific artifact manifest top level must be an object"]
        )
    if manifest.get("schema_version") != 1:
        errors.append("scientific artifact manifest schema_version must equal 1")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("scientific artifact manifest artifacts must be a non-empty list")
        raise ArtifactVerificationError(errors)

    artifact_paths: list[str] = []
    checked_inputs = 0
    skipped_inputs = 0
    referenced_results = 0
    for index, artifact in enumerate(artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{label} must be an object")
            continue
        path_value = artifact.get("path")
        if isinstance(path_value, str):
            artifact_paths.append(path_value)
        checked, _ = _verify_hash_binding(
            root,
            path_value,
            artifact.get("sha256"),
            label=label,
            required=True,
            errors=errors,
        )
        if checked:
            try:
                artifact_path = _relative_path(root, path_value, label=f"{label}.path")
            except ValueError:
                artifact_path = None
            if artifact_path is not None:
                errors.extend(
                    _verify_schema(
                        artifact_path,
                        artifact.get("expected_schema"),
                        label=path_value,
                    )
                )

        _verify_hash_binding(
            root,
            artifact.get("generating_script"),
            artifact.get("generating_script_sha256"),
            label=f"{label}.generating_script",
            required=True,
            errors=errors,
        )
        config = artifact.get("config")
        if not isinstance(config, dict):
            errors.append(f"{label}.config must be an object with path and sha256")
        else:
            _verify_hash_binding(
                root,
                config.get("path"),
                config.get("sha256"),
                label=f"{label}.config",
                required=True,
                errors=errors,
            )
        if not isinstance(artifact.get("study_phase"), str) or not artifact["study_phase"]:
            errors.append(f"{label}.study_phase must be a non-empty string")
        for flag in ("gpu_required", "training_required"):
            if not isinstance(artifact.get(flag), bool):
                errors.append(f"{label}.{flag} must be boolean")

        inputs = artifact.get("input_artifacts")
        if not isinstance(inputs, list) or not inputs:
            errors.append(f"{label}.input_artifacts must be a non-empty list")
        else:
            for input_index, input_artifact in enumerate(inputs):
                input_label = f"{label}.input_artifacts[{input_index}]"
                if not isinstance(input_artifact, dict):
                    errors.append(f"{input_label} must be an object")
                    continue
                availability = input_artifact.get("availability", "committed")
                if availability not in {"committed", "external_or_ignored"}:
                    errors.append(f"{input_label}.availability is invalid: {availability!r}")
                    continue
                checked, skipped = _verify_hash_binding(
                    root,
                    input_artifact.get("path"),
                    input_artifact.get("sha256"),
                    label=input_label,
                    required=availability == "committed",
                    errors=errors,
                )
                checked_inputs += int(checked)
                skipped_inputs += int(skipped)

        references = artifact.get("referenced_results", [])
        if not isinstance(references, list) or not all(
            isinstance(item, str) for item in references
        ):
            errors.append(f"{label}.referenced_results must be a string list")
        else:
            for reference_index, reference in enumerate(references):
                reference_label = f"{label}.referenced_results[{reference_index}]"
                try:
                    reference_path = _relative_path(root, reference, label=reference_label)
                except ValueError as error:
                    errors.append(str(error))
                    continue
                if not reference.startswith("results/"):
                    errors.append(f"{reference_label} must point below results/: {reference}")
                elif not reference_path.is_file():
                    errors.append(f"{reference_label} is missing: {reference}")
                else:
                    referenced_results += 1

    duplicates = sorted({path for path in artifact_paths if artifact_paths.count(path) > 1})
    if duplicates:
        errors.append(f"duplicate artifact paths: {', '.join(duplicates)}")
    critical_paths = manifest.get("manuscript_critical_paths")
    if not isinstance(critical_paths, list) or not all(
        isinstance(path, str) for path in critical_paths
    ):
        errors.append("manuscript_critical_paths must be a string list")
    elif critical_paths != artifact_paths:
        errors.append("manuscript_critical_paths must exactly match artifacts[].path in order")

    if errors:
        raise ArtifactVerificationError(errors)
    return VerificationReport(
        artifact_count=len(artifacts),
        checked_input_count=checked_inputs,
        skipped_external_input_count=skipped_inputs,
        referenced_result_count=referenced_results,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/scientific_artifact_manifest.json"),
        help="scientific artifact manifest path",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repository root (defaults to the manifest's grandparent)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the artifact verifier CLI."""

    args = _parser().parse_args(argv)
    try:
        report = verify_manifest(args.manifest, project_root=args.project_root)
    except ArtifactVerificationError as error:
        print("Scientific artifact verification failed:", file=sys.stderr)
        for item in error.errors:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(
        "Scientific artifact verification passed: "
        f"{report.artifact_count} artifacts, "
        f"{report.checked_input_count} present inputs, "
        f"{report.skipped_external_input_count} unavailable external/ignored inputs, "
        f"{report.referenced_result_count} referenced result files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
