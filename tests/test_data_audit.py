import json
from pathlib import Path

from PIL import Image

from meddet_benchmark.__main__ import main
from meddet_benchmark.data_audit import audit_yolo_dataset


def add_case(
    root: Path,
    split: str,
    name: str,
    *,
    color: int,
    label: str | None,
) -> None:
    image_path = root / split / "images" / f"{name}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (12, 10), color=color).save(image_path)
    if label is not None:
        label_path = root / split / "labels" / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label, encoding="utf-8")
    else:
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def complete_layout(root: Path) -> None:
    add_case(root, "train", "positive", color=10, label="0 0.5 0.5 0.4 0.4\n")
    add_case(root, "valid", "negative", color=20, label="")
    add_case(root, "test", "negative", color=30, label="")


def test_clean_dataset_produces_deterministic_manifest(tmp_path: Path) -> None:
    complete_layout(tmp_path)

    first = audit_yolo_dataset(tmp_path, class_names=("tumor",))
    second = audit_yolo_dataset(tmp_path, class_names=("tumor",))

    assert first["passed"]
    assert first["image_count"] == 3
    assert first["box_count"] == 1
    assert first["class_box_counts"] == {"0": 1}
    assert first["split_counts"]["validation"]["empty_labels"] == 1
    assert first["manifest_sha256"] == second["manifest_sha256"]
    json.dumps(first, allow_nan=False)


def test_invalid_annotation_and_cross_split_duplicate_fail_audit(tmp_path: Path) -> None:
    add_case(tmp_path, "train", "a", color=10, label="2 0.5 0.5 0.4 0.4\n")
    add_case(tmp_path, "valid", "b", color=20, label="")
    add_case(tmp_path, "test", "c", color=10, label="0 0.9 0.5 0.4 0.4\n")

    report = audit_yolo_dataset(tmp_path, class_names=("tumor",))
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["passed"]
    assert {"unknown_class", "invalid_box", "cross_split_duplicate"} <= codes


def test_missing_label_is_visible_and_orphan_label_is_an_error(tmp_path: Path) -> None:
    complete_layout(tmp_path)
    (tmp_path / "test" / "labels" / "negative.txt").unlink()
    orphan = tmp_path / "train" / "labels" / "orphan.txt"
    orphan.write_text("", encoding="utf-8")

    report = audit_yolo_dataset(tmp_path, class_names=("tumor",))
    severities = {issue["code"]: issue["severity"] for issue in report["issues"]}

    assert severities["missing_label"] == "warning"
    assert severities["orphan_label"] == "error"
    assert not report["passed"]


def test_audit_cli_returns_machine_readable_report(tmp_path: Path, capsys) -> None:
    complete_layout(tmp_path)

    assert main(["audit-data", str(tmp_path), "--class-name", "tumor"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["passed"]
    assert report["manifest_sha256"]
