"""Build the review article from its sole canonical Markdown source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run Pandoc/citeproc and the configured TeX engine; record artifact identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pandoc", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pandoc = str(args.pandoc) if args.pandoc else shutil.which("pandoc")
    if not pandoc:
        import pypandoc

        pandoc = pypandoc.get_pandoc_path()
    manuscript = root / config["manuscript"]
    work = root / config["work_directory"]
    output = root / config["output"]
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tex = work / "article.tex"
    env = os.environ.copy()
    env["PAPER_SUPPORT_URL"] = config["repository_url"] + "/blob/" + config["support_revision"]
    command = [
        pandoc,
        str(manuscript),
        "--standalone",
        "--citeproc",
        "--from=markdown",
        "--to=latex",
        "--top-level-division=section",
        "--bibliography=" + str(root / config["bibliography"]),
        "--include-in-header=" + str(root / config["header"]),
        "--lua-filter=" + str(root / config["filter"]),
        "--output=" + str(tex),
    ]
    for key, value in config["variables"].items():
        command.extend(
            ["--variable", f"{key}={str(value).lower() if isinstance(value, bool) else value}"]
        )
    subprocess.run(command, cwd=manuscript.parent, env=env, check=True)
    latex_command = [
        config["pdf_engine"],
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-output-directory=" + str(work),
        str(tex),
    ]
    for run in range(2):
        with (work / f"tex-{run + 1}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                latex_command,
                cwd=manuscript.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    shutil.copyfile(work / "article.pdf", output)
    inputs = [
        config_path,
        manuscript,
        root / config["bibliography"],
        root / config["header"],
        root / config["filter"],
        Path(__file__).resolve(),
    ]
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", manuscript.read_text(encoding="utf-8")):
        inputs.append((manuscript.parent / target).resolve())
    manifest = {
        "status": "author_review_derivative_not_submission_clearance",
        "canonical_manuscript": config["manuscript"],
        "support_revision": config["support_revision"],
        "pandoc_version": subprocess.check_output([pandoc, "--version"], text=True).splitlines()[0],
        "tex_version": subprocess.check_output(
            [config["pdf_engine"], "--version"], text=True
        ).splitlines()[0],
        "inputs": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)} for p in inputs],
        "output": {
            "path": config["output"],
            "sha256": sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "reproduction": "python -m scripts.build_paper_pdf --config configs/paper_build.yaml",
        "note": (
            "PDF timestamps/tool versions can change byte hashes; scientific content "
            "comes only from the canonical source."
        ),
    }
    (root / config["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["output"], indent=2))


if __name__ == "__main__":
    main()
