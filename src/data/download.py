"""Download configured Kaggle competition files without embedding credentials.

The expected dataset configuration shape is::

    dataset:
      kaggle:
        competition: competition-slug
        files: [first-file.zip, labels.csv]
      paths:
        raw_dir: data/raw/example

Credentials are resolved from ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` together,
or from ``kaggle.json`` beneath ``KAGGLE_CONFIG_DIR`` (defaulting to
``~/.kaggle``). Credential values are never included in object representations,
commands, status messages, or raised diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/dataset.yaml")
_ACCEPTANCE_MARKERS = (
    "accept the rules",
    "accept this competition's rules",
    "accept this competition\u2019s rules",
    "competition rules before",
    "join this competition",
    "403 - forbidden",
    "403 forbidden",
)
_GEOGRAPHIC_MARKERS = (
    "not available in your location",
    "geographic restriction",
    "geographically restricted",
    "geo-restricted",
    "country is not supported",
    "region is not supported",
)


class DownloadError(RuntimeError):
    """Base class for expected downloader failures."""


class ConfigurationError(DownloadError):
    """Raised when the dataset YAML is absent or invalid."""


class CredentialError(DownloadError):
    """Raised when Kaggle credentials are absent, partial, or malformed."""


class KaggleCliError(DownloadError):
    """Raised when the Kaggle CLI cannot be found or executed."""


class KaggleDownloadError(DownloadError):
    """Raised when Kaggle rejects or cannot complete a download."""


class CredentialSource(StrEnum):
    """Supported sources of Kaggle credentials."""

    ENVIRONMENT = "environment"
    FILE = "file"


@dataclass(frozen=True)
class KaggleCredentials:
    """Validated Kaggle credential context with secret-safe representation."""

    source: CredentialSource
    username: str = field(repr=False)
    key: str = field(repr=False)
    config_dir: Path | None = None
    config_file: Path | None = None

    @property
    def redaction_values(self) -> tuple[str, ...]:
        """Return credential values that must be removed from diagnostics."""

        return tuple(value for value in (self.username, self.key) if value)


@dataclass(frozen=True)
class DownloadSettings:
    """Kaggle competition files and local destination loaded from YAML."""

    competition: str
    files: tuple[str, ...]
    raw_dir: Path


def _require_mapping(value: Any, *, key_path: str) -> Mapping[str, Any]:
    """Return *value* as a mapping or raise a path-specific config error."""

    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key_path} must be a mapping")
    return value


def _require_nonempty_string(value: Any, *, key_path: str) -> str:
    """Validate a non-blank YAML string without coercing other scalar types."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key_path} must be a non-empty string")
    return value.strip()


def load_download_settings(config_path: str | Path) -> DownloadSettings:
    """Load the Kaggle competition download settings from ``config_path``.

    The strict keys are ``dataset.kaggle.competition``,
    ``dataset.kaggle.files``, and ``dataset.paths.raw_dir``. Relative paths are
    intentionally retained as project-working-directory-relative paths.
    """

    path = Path(config_path).expanduser()
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise ConfigurationError(f"dataset config not found: {path}") from error
    except OSError as error:
        raise ConfigurationError(f"could not read dataset config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"dataset config is not valid YAML: {path}") from error

    root = _require_mapping(payload, key_path="config root")
    dataset = _require_mapping(root.get("dataset"), key_path="dataset")
    kaggle = _require_mapping(dataset.get("kaggle"), key_path="dataset.kaggle")
    paths = _require_mapping(dataset.get("paths"), key_path="dataset.paths")

    competition = _require_nonempty_string(
        kaggle.get("competition"), key_path="dataset.kaggle.competition"
    )
    raw_files = kaggle.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ConfigurationError("dataset.kaggle.files must be a non-empty list")
    files = tuple(
        _require_nonempty_string(value, key_path=f"dataset.kaggle.files[{index}]")
        for index, value in enumerate(raw_files)
    )
    if len(set(files)) != len(files):
        raise ConfigurationError("dataset.kaggle.files must not contain duplicates")

    raw_dir_value = _require_nonempty_string(
        paths.get("raw_dir"), key_path="dataset.paths.raw_dir"
    )
    return DownloadSettings(competition=competition, files=files, raw_dir=Path(raw_dir_value))


def _credential_file_path(
    environ: Mapping[str, str], *, home: str | Path | None = None
) -> Path:
    """Resolve ``kaggle.json`` while honoring Kaggle's config-dir convention."""

    configured_dir = environ.get("KAGGLE_CONFIG_DIR")
    if configured_dir is not None:
        if not configured_dir.strip():
            raise CredentialError("KAGGLE_CONFIG_DIR is set but blank")
        config_dir = Path(configured_dir).expanduser()
    else:
        home_dir = Path(home).expanduser() if home is not None else Path.home()
        config_dir = home_dir / ".kaggle"
    return config_dir / "kaggle.json"


def _load_credential_file(path: Path) -> KaggleCredentials:
    """Validate a Kaggle JSON credential file without exposing its contents."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as error:
        raise CredentialError(
            "Kaggle credentials were not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            f"together, or create {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise CredentialError(f"Kaggle credential file is malformed JSON: {path}") from error
    except OSError as error:
        raise CredentialError(f"Kaggle credential file could not be read: {path}") from error

    if not isinstance(payload, Mapping):
        raise CredentialError(f"Kaggle credential file must contain a JSON object: {path}")
    username = payload.get("username")
    key = payload.get("key")
    if not isinstance(username, str) or not username.strip():
        raise CredentialError(
            f"Kaggle credential file has a missing or invalid 'username' field: {path}"
        )
    if not isinstance(key, str) or not key.strip():
        raise CredentialError(
            f"Kaggle credential file has a missing or invalid 'key' field: {path}"
        )
    return KaggleCredentials(
        source=CredentialSource.FILE,
        username=username.strip(),
        key=key.strip(),
        config_dir=path.parent,
        config_file=path,
    )


def resolve_kaggle_credentials(
    environ: Mapping[str, str] | None = None,
    *,
    home: str | Path | None = None,
) -> KaggleCredentials:
    """Resolve and validate Kaggle credentials without making a network call.

    Environment credentials take precedence only when both variables are
    present and non-blank. A partial pair is an error rather than silently
    falling back to a file, because Kaggle itself would otherwise emit a much
    less actionable authentication failure.
    """

    effective_env = os.environ if environ is None else environ
    username = effective_env.get("KAGGLE_USERNAME")
    key = effective_env.get("KAGGLE_KEY")
    username_present = isinstance(username, str) and bool(username.strip())
    key_present = isinstance(key, str) and bool(key.strip())

    if username_present != key_present:
        missing = "KAGGLE_KEY" if username_present else "KAGGLE_USERNAME"
        raise CredentialError(
            "partial Kaggle environment credentials: "
            f"{missing} is missing or blank; set both variables or unset both"
        )
    if username_present and key_present:
        return KaggleCredentials(
            source=CredentialSource.ENVIRONMENT,
            username=username.strip(),
            key=key.strip(),
        )

    credential_path = _credential_file_path(effective_env, home=home)
    return _load_credential_file(credential_path)


def _redact_diagnostic(text: str, credentials: KaggleCredentials) -> str:
    """Remove known credential values and common assignment forms from text."""

    redacted = text
    for secret in sorted(credentials.redaction_values, key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    redacted = re.sub(
        r"(?i)(KAGGLE_(?:USERNAME|KEY)\s*[:=]\s*)\S+",
        r"\1<redacted>",
        redacted,
    )
    redacted = re.sub(
        r'(?i)(["\']?(?:username|key)["\']?\s*:\s*["\'])[^"\']*',
        r"\1<redacted>",
        redacted,
    )
    return redacted.strip()


def _child_environment(
    environ: Mapping[str, str], credentials: KaggleCredentials
) -> dict[str, str]:
    """Build the CLI environment for the selected credential source."""

    child_env = dict(environ)
    if credentials.source is CredentialSource.ENVIRONMENT:
        child_env["KAGGLE_USERNAME"] = credentials.username
        child_env["KAGGLE_KEY"] = credentials.key
    else:
        if credentials.config_dir is None:
            raise CredentialError("file-based credentials have no configuration directory")
        child_env.pop("KAGGLE_USERNAME", None)
        child_env.pop("KAGGLE_KEY", None)
        child_env["KAGGLE_CONFIG_DIR"] = str(credentials.config_dir)
    return child_env


def _download_failure_message(
    *,
    competition: str,
    file_name: str,
    output: str,
    returncode: int,
) -> str:
    """Translate common Kaggle/storage failures into actionable diagnostics."""

    normalized = output.casefold()
    if any(marker in normalized for marker in _GEOGRAPHIC_MARKERS):
        return (
            "Kaggle or its storage provider reports that the competition download is "
            "unavailable from this geographic location. Use an authorized available "
            "network/location or download through the Kaggle website, then retry."
        )
    if any(marker in normalized for marker in _ACCEPTANCE_MARKERS):
        return (
            "Kaggle denied access because the competition rules have not been accepted. "
            f"Join the competition and accept its rules at "
            f"https://www.kaggle.com/c/{competition}/rules, then retry."
        )

    detail = f" Kaggle diagnostic: {output}" if output else ""
    return (
        f"Kaggle failed to download configured file {file_name!r} from competition "
        f"{competition!r} (exit code {returncode}).{detail}"
    )


def download_competition_files(
    settings: DownloadSettings,
    *,
    credentials: KaggleCredentials | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Download every configured competition file with the Kaggle CLI.

    Commands are always passed to :func:`subprocess.run` as argument lists with
    ``shell=False``. The function returns the expected destination paths after
    all commands succeed; Kaggle remains responsible for the actual file names.
    """

    effective_env = dict(os.environ if environ is None else environ)
    resolved_credentials = credentials or resolve_kaggle_credentials(effective_env)
    executable = shutil.which("kaggle", path=effective_env.get("PATH"))
    if executable is None:
        raise KaggleCliError(
            "Kaggle CLI was not found on PATH. Install the 'kaggle' package and ensure "
            "its executable is available before retrying."
        )

    try:
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise KaggleDownloadError(
            f"could not create configured raw-data directory {settings.raw_dir}: {error}"
        ) from error

    child_env = _child_environment(effective_env, resolved_credentials)
    destinations: list[Path] = []
    for file_name in settings.files:
        command = [
            executable,
            "competitions",
            "download",
            "-c",
            settings.competition,
            "-f",
            file_name,
            "-p",
            str(settings.raw_dir),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env=child_env,
            )
        except FileNotFoundError as error:
            raise KaggleCliError(
                "Kaggle CLI disappeared from PATH before it could be executed."
            ) from error
        except OSError as error:
            raise KaggleCliError(f"Kaggle CLI could not be executed: {error}") from error

        if completed.returncode != 0:
            combined_output = "\n".join(
                value for value in (completed.stderr, completed.stdout) if value
            )
            safe_output = _redact_diagnostic(combined_output, resolved_credentials)
            message = _download_failure_message(
                competition=settings.competition,
                file_name=file_name,
                output=safe_output,
                returncode=completed.returncode,
            )
            raise KaggleDownloadError(message)
        destinations.append(settings.raw_dir / file_name)
    return tuple(destinations)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the dataset downloader."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"dataset YAML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--check-credentials",
        action="store_true",
        help="validate credential discovery without invoking Kaggle or reading the dataset config",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader CLI and return a process-style exit code."""

    args = build_parser().parse_args(argv)
    try:
        credentials = resolve_kaggle_credentials()
        if args.check_credentials:
            if credentials.source is CredentialSource.ENVIRONMENT:
                print("Kaggle credentials are validly configured in environment variables.")
            else:
                print(f"Kaggle credential file is structurally valid: {credentials.config_file}")
            return 0

        settings = load_download_settings(args.config)
        downloaded = download_competition_files(settings, credentials=credentials)
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Downloaded {len(downloaded)} configured file(s) to {settings.raw_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
