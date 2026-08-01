from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.data import download


def _write_config(
    path: Path,
    *,
    competition: str = "example-competition",
    files: list[str] | None = None,
    raw_dir: str | None = None,
) -> Path:
    configured_files = files or ["images.zip", "labels.csv"]
    configured_raw_dir = raw_dir or str(path.parent / "raw")
    path.write_text(
        "\n".join(
            [
                "dataset:",
                "  kaggle:",
                f"    competition: {competition}",
                "    files:",
                *(f"      - {file_name}" for file_name in configured_files),
                "  paths:",
                f"    raw_dir: {configured_raw_dir}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _environment_credentials() -> download.KaggleCredentials:
    return download.KaggleCredentials(
        source=download.CredentialSource.ENVIRONMENT,
        username="safe-user",
        key="super-secret-key",
    )


def test_load_download_settings_uses_repository_schema(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "dataset.yaml",
        competition="medical-challenge",
        files=["train.zip", "labels.csv", "classes.csv"],
        raw_dir="data/raw/medical",
    )

    settings = download.load_download_settings(config_path)

    assert settings == download.DownloadSettings(
        competition="medical-challenge",
        files=("train.zip", "labels.csv", "classes.csv"),
        raw_dir=Path("data/raw/medical"),
    )


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        ("dataset: null\n", "dataset must be a mapping"),
        (
            "dataset:\n  kaggle:\n    competition: c\n    files: []\n  paths:\n"
            "    raw_dir: raw\n",
            "dataset.kaggle.files must be a non-empty list",
        ),
        (
            "dataset:\n  kaggle:\n    competition: c\n    files: [a.zip, a.zip]\n"
            "  paths:\n    raw_dir: raw\n",
            "must not contain duplicates",
        ),
        ("dataset: [unterminated\n", "not valid YAML"),
    ],
)
def test_load_download_settings_reports_invalid_config(
    tmp_path: Path, yaml_text: str, expected: str
) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(download.ConfigurationError, match=expected):
        download.load_download_settings(config_path)


def test_load_download_settings_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(download.ConfigurationError, match="dataset config not found"):
        download.load_download_settings(tmp_path / "missing.yaml")


def test_environment_credentials_require_and_prefer_complete_pair(tmp_path: Path) -> None:
    config_dir = tmp_path / "credentials"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text("not valid json", encoding="utf-8")
    environ = {
        "KAGGLE_USERNAME": "env-user",
        "KAGGLE_KEY": "env-secret",
        "KAGGLE_CONFIG_DIR": str(config_dir),
    }

    credentials = download.resolve_kaggle_credentials(environ)

    assert credentials.source is download.CredentialSource.ENVIRONMENT
    assert credentials.username == "env-user"
    assert credentials.key == "env-secret"
    assert "env-user" not in repr(credentials)
    assert "env-secret" not in repr(credentials)


@pytest.mark.parametrize(
    "environ",
    [
        {"KAGGLE_USERNAME": "user"},
        {"KAGGLE_KEY": "key"},
        {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "   "},
    ],
)
def test_partial_environment_credentials_fail_clearly(environ: dict[str, str]) -> None:
    with pytest.raises(download.CredentialError, match="partial Kaggle environment credentials"):
        download.resolve_kaggle_credentials(environ)


def test_credential_file_respects_kaggle_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "custom-kaggle"
    config_dir.mkdir()
    credential_path = config_dir / "kaggle.json"
    credential_path.write_text(
        json.dumps({"username": "file-user", "key": "file-secret"}), encoding="utf-8"
    )

    credentials = download.resolve_kaggle_credentials({"KAGGLE_CONFIG_DIR": str(config_dir)})

    assert credentials.source is download.CredentialSource.FILE
    assert credentials.config_dir == config_dir
    assert credentials.config_file == credential_path
    assert credentials.username == "file-user"
    assert credentials.key == "file-secret"
    assert "file-secret" not in repr(credentials)


def test_credential_file_defaults_to_home_dot_kaggle(tmp_path: Path) -> None:
    config_dir = tmp_path / ".kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(
        json.dumps({"username": "user", "key": "key"}), encoding="utf-8"
    )

    credentials = download.resolve_kaggle_credentials({}, home=tmp_path)

    assert credentials.config_file == config_dir / "kaggle.json"


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("this is not json secret-value", "malformed JSON"),
        (json.dumps({"username": "user"}), "invalid 'key' field"),
        (json.dumps(["user", "secret-value"]), "must contain a JSON object"),
    ],
)
def test_malformed_credential_file_is_clear_and_does_not_echo_contents(
    tmp_path: Path, contents: str, expected: str
) -> None:
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(contents, encoding="utf-8")

    with pytest.raises(download.CredentialError, match=expected) as captured:
        download.resolve_kaggle_credentials({"KAGGLE_CONFIG_DIR": str(config_dir)})

    assert "secret-value" not in str(captured.value)


def test_missing_credentials_report_both_supported_sources(tmp_path: Path) -> None:
    with pytest.raises(download.CredentialError) as captured:
        download.resolve_kaggle_credentials({}, home=tmp_path)

    message = str(captured.value)
    assert "KAGGLE_USERNAME and KAGGLE_KEY together" in message
    assert str(tmp_path / ".kaggle" / "kaggle.json") in message


def test_blank_kaggle_config_dir_is_rejected() -> None:
    with pytest.raises(download.CredentialError, match="set but blank"):
        download.resolve_kaggle_credentials({"KAGGLE_CONFIG_DIR": "  "})


def test_download_uses_argument_lists_and_one_command_per_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = download.DownloadSettings(
        competition="safe-competition",
        files=("images.zip", "labels.csv"),
        raw_dir=tmp_path / "raw data",
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []

    monkeypatch.setattr(download.shutil, "which", lambda _name, path=None: "kaggle-exe")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    paths = download.download_competition_files(
        settings,
        credentials=_environment_credentials(),
        environ={"PATH": "test-path"},
    )

    assert paths == (settings.raw_dir / "images.zip", settings.raw_dir / "labels.csv")
    assert settings.raw_dir.is_dir()
    assert [call[0] for call in calls] == [
        [
            "kaggle-exe",
            "competitions",
            "download",
            "-c",
            "safe-competition",
            "-f",
            file_name,
            "-p",
            str(settings.raw_dir),
        ]
        for file_name in settings.files
    ]
    for _command, kwargs in calls:
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["env"]["KAGGLE_USERNAME"] == "safe-user"
        assert kwargs["env"]["KAGGLE_KEY"] == "super-secret-key"


def test_file_credentials_are_passed_as_config_dir_not_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = download.DownloadSettings("competition", ("data.zip",), tmp_path / "raw")
    config_dir = tmp_path / "kaggle"
    credentials = download.KaggleCredentials(
        source=download.CredentialSource.FILE,
        username="file-user",
        key="file-key",
        config_dir=config_dir,
        config_file=config_dir / "kaggle.json",
    )
    captured_env: dict[str, str] = {}
    monkeypatch.setattr(download.shutil, "which", lambda _name, path=None: "kaggle")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    download.download_competition_files(
        settings,
        credentials=credentials,
        environ={"PATH": "test", "KAGGLE_USERNAME": "stale", "KAGGLE_KEY": "stale"},
    )

    assert captured_env["KAGGLE_CONFIG_DIR"] == str(config_dir)
    assert "KAGGLE_USERNAME" not in captured_env
    assert "KAGGLE_KEY" not in captured_env


def test_missing_kaggle_cli_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = download.DownloadSettings("competition", ("data.zip",), tmp_path / "raw")
    monkeypatch.setattr(download.shutil, "which", lambda _name, path=None: None)

    with pytest.raises(download.KaggleCliError, match="not found on PATH"):
        download.download_competition_files(
            settings, credentials=_environment_credentials(), environ={}
        )


@pytest.mark.parametrize(
    ("diagnostic", "expected"),
    [
        (
            "403 Forbidden: You must accept the rules before downloading",
            "competition rules have not been accepted",
        ),
        (
            "AccessDenied: this service is not available in your location",
            "geographic location",
        ),
    ],
)
def test_known_download_failures_are_translated_and_credentials_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic: str,
    expected: str,
) -> None:
    settings = download.DownloadSettings("medical-challenge", ("data.zip",), tmp_path / "raw")
    credentials = _environment_credentials()
    monkeypatch.setattr(download.shutil, "which", lambda _name, path=None: "kaggle")

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        stderr = f"{diagnostic}; KAGGLE_KEY={credentials.key}"
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    with pytest.raises(download.KaggleDownloadError, match=expected) as captured:
        download.download_competition_files(
            settings, credentials=credentials, environ={"PATH": "test"}
        )

    assert credentials.key not in str(captured.value)
    assert credentials.username not in str(captured.value)


def test_generic_cli_failure_preserves_safe_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = download.DownloadSettings("competition", ("data.zip",), tmp_path / "raw")
    credentials = _environment_credentials()
    monkeypatch.setattr(download.shutil, "which", lambda _name, path=None: "kaggle")

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            returncode=7,
            stdout="",
            stderr=f"request failed with token {credentials.key}",
        )

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    with pytest.raises(download.KaggleDownloadError) as captured:
        download.download_competition_files(
            settings, credentials=credentials, environ={"PATH": "test"}
        )

    message = str(captured.value)
    assert "request failed with token <redacted>" in message
    assert credentials.key not in message


def test_check_credentials_cli_does_not_read_config_or_invoke_kaggle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(download, "resolve_kaggle_credentials", _environment_credentials)

    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("download configuration or Kaggle should not be used for credential checks")

    monkeypatch.setattr(download, "load_download_settings", fail_if_called)
    monkeypatch.setattr(download, "download_competition_files", fail_if_called)

    result = download.main(["--config", "missing.yaml", "--check-credentials"])

    captured = capsys.readouterr()
    assert result == 0
    assert "validly configured in environment variables" in captured.out
    assert "super-secret-key" not in captured.out
    assert captured.err == ""


def test_cli_returns_clear_error_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def raise_missing() -> download.KaggleCredentials:
        raise download.CredentialError("credentials are absent")

    monkeypatch.setattr(download, "resolve_kaggle_credentials", raise_missing)

    result = download.main(["--check-credentials"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "error: credentials are absent\n"
