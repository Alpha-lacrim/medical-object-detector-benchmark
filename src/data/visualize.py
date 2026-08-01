"""Create deterministic EDA figures from split manifests and canonical COCO files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the dataset section of a YAML config."""

    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("dataset"), dict):
        raise ValueError("dataset config must contain a 'dataset' mapping")
    return payload["dataset"]


def read_split_rows(splits_dir: Path) -> list[dict[str, str]]:
    """Read all train/validation/test manifest rows from a split directory."""

    rows: list[dict[str, str]] = []
    paths = sorted(splits_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no split CSV manifests found in {splits_dir}")
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                record = dict(row)
                record.setdefault("split", path.stem)
                rows.append(record)
    return rows


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    *,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=_font())
    width = box[2] - box[0]
    draw.text((position[0] - width / 2, position[1]), text, fill=fill, font=_font())


def plot_study_distribution(
    rows: Iterable[dict[str, str]],
    *,
    strata: list[str],
    split_names: list[str],
    output_path: Path,
    width: int,
    height: int,
) -> dict[str, dict[str, int]]:
    """Plot study-level strata by split and return the plotted counts."""

    counts: dict[str, Counter[str]] = {name: Counter() for name in split_names}
    for row in rows:
        split = row.get("split", "")
        stratum = row.get("study_stratum", "")
        if split in counts and stratum in strata:
            counts[split][stratum] += 1

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    left, right, top, bottom = 95, 35, 65, 105
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = max((value for counter in counts.values() for value in counter.values()), default=1)
    maximum = max(maximum, 1)
    colors = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"]

    draw.text(
        (left, 22),
        "RSNA study-level distribution after patient-grouped split",
        fill="#111827",
    )
    draw.line((left, top, left, top + chart_height), fill="#374151", width=2)
    draw.line(
        (left, top + chart_height, width - right, top + chart_height),
        fill="#374151",
        width=2,
    )

    for tick in range(0, 6):
        value = round(maximum * tick / 5)
        y = top + chart_height - chart_height * tick / 5
        draw.line((left, y, width - right, y), fill="#e5e7eb", width=1)
        draw.text((10, y - 6), str(value), fill="#4b5563", font=_font())

    group_width = chart_width / max(len(strata), 1)
    bar_width = group_width * 0.72 / max(len(split_names), 1)
    for stratum_index, stratum in enumerate(strata):
        center = left + group_width * (stratum_index + 0.5)
        for split_index, split in enumerate(split_names):
            value = counts[split][stratum]
            bar_height = chart_height * value / maximum
            x0 = center - group_width * 0.36 + split_index * bar_width
            y0 = top + chart_height - bar_height
            x1 = x0 + bar_width * 0.86
            draw.rectangle((x0, y0, x1, top + chart_height), fill=colors[split_index])
            _draw_centered_text(
                draw,
                ((x0 + x1) / 2, max(top, y0 - 16)),
                str(value),
                fill="#111827",
            )
        wrapped = stratum.replace(" / ", " /\n")
        for line_index, line in enumerate(wrapped.splitlines()):
            _draw_centered_text(
                draw,
                (center, top + chart_height + 18 + line_index * 16),
                line,
                fill="#111827",
            )

    legend_x = left
    legend_y = height - 28
    for index, split in enumerate(split_names):
        draw.rectangle((legend_x, legend_y, legend_x + 13, legend_y + 13), fill=colors[index])
        draw.text((legend_x + 19, legend_y), split, fill="#111827", font=_font())
        legend_x += 100

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {split: {stratum: counts[split][stratum] for stratum in strata} for split in split_names}


def _stable_key(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _resolve_image_path(dataset: dict[str, Any], file_name: str) -> Path:
    paths = dataset["paths"]
    processed_dir = Path(paths["processed_dir"])
    first = processed_dir / file_name
    if first.is_file():
        return first
    return Path(paths["processed_images_dir"]) / Path(file_name).name


def _load_coco_records(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    annotations_dir = Path(dataset["paths"]["annotations_dir"])
    records: list[dict[str, Any]] = []
    for path in sorted(annotations_dir.glob("*.json")):
        split_name = path.stem.removeprefix("instances_")
        payload = json.loads(path.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in payload.get("images", [])}
        annotations: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations[int(annotation["image_id"])].append(annotation)
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        for image_id, image_record in images.items():
            file_name = str(image_record["file_name"])
            image_path = _resolve_image_path(dataset, file_name)
            if not image_path.is_file():
                continue
            records.append(
                {
                    "split": split_name,
                    "image": image_record,
                    "annotations": annotations[image_id],
                    "categories": categories,
                    "path": image_path,
                }
            )
    return records


def _balanced_sample(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stratum = str(record["image"].get("study_stratum", "unknown"))
        buckets[stratum].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: _stable_key(seed, str(item["image"]["file_name"])))
    selected: list[dict[str, Any]] = []
    names = sorted(buckets)
    while len(selected) < count and names:
        remaining: list[str] = []
        for name in names:
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop(0))
            if buckets[name]:
                remaining.append(name)
        names = remaining
    return selected


def _thumbnail_with_boxes(record: dict[str, Any], size: int) -> Image.Image:
    with Image.open(record["path"]) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size))
    tile = Image.new("RGB", (size, size + 42), "#111827")
    offset_x = (size - image.width) // 2
    offset_y = 42 + (size - image.height) // 2
    tile.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    image_record = record["image"]
    source_width = float(image_record["width"])
    source_height = float(image_record["height"])
    scale_x = image.width / source_width
    scale_y = image.height / source_height
    for annotation in record["annotations"]:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        box = (
            offset_x + x * scale_x,
            offset_y + y * scale_y,
            offset_x + (x + width) * scale_x,
            offset_y + (y + height) * scale_y,
        )
        draw.rectangle(box, outline="#ef4444", width=3)
        label = record["categories"].get(int(annotation["category_id"]), "unknown")
        draw.text((box[0] + 3, max(offset_y, box[1] - 13)), label, fill="#ef4444")
    stratum = str(image_record.get("study_stratum", "unknown"))
    title = f"{record['split']} | {stratum}"
    draw.text((8, 8), title[:58], fill="white", font=_font())
    return tile


def plot_annotation_samples(dataset: dict[str, Any], output_path: Path) -> int:
    """Create a deterministic grid of locally available images with COCO boxes."""

    eda = dataset["eda"]
    records = _load_coco_records(dataset)
    if not records:
        raise FileNotFoundError(
            "no processed PNG images referenced by the COCO files are available; "
            "download/convert images before running sample visualization"
        )
    requested = int(eda["sample_images"])
    selected = _balanced_sample(records, count=requested, seed=int(eda["sample_seed"]))
    columns = int(eda["grid_columns"])
    size = int(eda["thumbnail_size"])
    rows = math.ceil(len(selected) / columns)
    grid = Image.new("RGB", (columns * size, rows * (size + 42)), "#030712")
    for index, record in enumerate(selected):
        tile = _thumbnail_with_boxes(record, size)
        grid.paste(tile, ((index % columns) * size, (index // columns) * (size + 42)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return len(selected)


def run_eda(config_path: str | Path) -> dict[str, Any]:
    """Generate all Batch 1 EDA artifacts and a machine-readable summary."""

    dataset = load_dataset_config(config_path)
    paths = dataset["paths"]
    eda = dataset["eda"]
    split_names = list(dataset["split"]["ratios"])
    rows = read_split_rows(Path(paths["splits_dir"]))
    figures_dir = Path(paths["figures_dir"])
    distribution_path = figures_dir / "rsna_class_distribution.png"
    sample_path = figures_dir / "rsna_annotation_samples.png"
    counts = plot_study_distribution(
        rows,
        strata=list(dataset["classes"]["study_strata"]),
        split_names=split_names,
        output_path=distribution_path,
        width=int(eda["distribution_width"]),
        height=int(eda["distribution_height"]),
    )
    sample_count = plot_annotation_samples(dataset, sample_path)
    summary = {
        "distribution_figure": distribution_path.as_posix(),
        "sample_figure": sample_path.as_posix(),
        "sample_count": sample_count,
        "study_counts": counts,
    }
    summary_path = figures_dir / "rsna_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml", help="dataset YAML path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the EDA CLI."""

    args = build_parser().parse_args(argv)
    print(json.dumps(run_eda(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
