"""Reproduce the first failed native-coordinate handoff without evaluating metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from src.benchmark_inference import Predictor
from src.data.prepare_vindr import sha256, verify_freeze, write_json
from src.evaluate import load_and_validate_training_configs, load_phase5_config
from src.meddet_benchmark.reproducibility import configure_reproducibility


def main() -> None:
    """Diagnose only the failed first run; keep raw coordinates in private storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operation-config", type=Path, required=True)
    parser.add_argument("--maximum-images", type=int, required=True)
    args = parser.parse_args()
    import torch

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    operation = yaml.safe_load(args.operation_config.read_text(encoding="utf-8"))
    verify_freeze(args.config, config)
    root = Path(config["outputs"]["private_root"])
    failed = json.loads(
        (Path(config["outputs"]["local_predictions"]) / operation["run_receipt_name"]).read_text()
    )
    if failed["status"] != "stopped_on_error" or failed["runs"]:
        raise ValueError("This diagnostic requires the first-run failure without completed results")
    output = Path(config["outputs"]["local_predictions"]) / "bounds_diagnostic.json"
    if output.exists():
        raise FileExistsError(output)
    phase = load_phase5_config(config["models"]["run_inventory_config"])
    models = load_and_validate_training_configs(
        load_phase5_config(operation["source_evaluation_config"])
    )
    run = phase.runs[0]
    identity = failed["gate"]["checkpoints"][0]
    if sha256(Path(run.checkpoint)) != identity["sha256"]:
        raise ValueError("Checkpoint identity mismatch")
    configure_reproducibility(run.seed, deterministic=True, allow_tf32=False)
    torch.set_num_threads(config["runtime"]["torch_threads"])
    classes = config["ontology"]["canonical_classes"]
    metadata = SimpleNamespace(
        num_foreground_classes=len(classes),
        label_to_category_id={i + 1: c for i, c in enumerate(classes)},
    )
    predictor = Predictor(run, phase, models[(run.detector, run.seed)], metadata)
    predictor.model.roi_heads.score_thresh = config["evaluation"]["candidate_score_floor"]
    adapter = yaml.safe_load(Path(operation["adapter_config"]).read_text())
    images = json.loads(Path(config["outputs"]["local_annotations"]).read_text())["images"]
    for index, record in enumerate(images[: args.maximum_images]):
        path = root / adapter["images_directory_name"] / record["file_name"]
        with Image.open(path) as source:
            rgb = np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
        values = predictor.predict(index, rgb)
        boxes = values["boxes"]
        limits = np.array([record["width"], record["height"]] * 2)
        invalid = (boxes < 0) | (boxes > limits)
        if np.any(invalid):
            excess = np.maximum(boxes - limits, -boxes)
            payload = {
                "status": "native_boundary_overshoot_reproduced",
                "index": index,
                "image_id": record["file_name"],
                "image_size": [record["height"], record["width"]],
                "checkpoint_sha256": identity["sha256"],
                "boxes_dtype": str(boxes.dtype),
                "maximum_excess_pixels": float(excess[invalid].max()),
                "raw_prediction": {k: v.tolist() for k, v in values.items()},
                "invalid_coordinates": np.argwhere(invalid).tolist(),
                "diagnostic_source_sha256": sha256(Path(__file__)),
            }
            write_json(output, payload)
            print(
                json.dumps(
                    {
                        k: payload[k]
                        for k in ("status", "index", "boxes_dtype", "maximum_excess_pixels")
                    }
                )
            )
            return
    raise ValueError(
        "The bounded diagnostic did not reproduce the failure; do not assume its cause"
    )


if __name__ == "__main__":
    main()
