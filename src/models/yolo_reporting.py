"""Shared validation, profiling, tables, and curves for the YOLO11s arm."""

from __future__ import annotations

import csv
import gc
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

from src.meddet_benchmark.coco_evaluation import evaluate_coco
from src.meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    evaluate_operating_point,
)
from src.models.faster_rcnn_reporting import (
    summarize_inference_timings,
    summarize_model_artifact,
    summarize_parameter_counts,
    write_compute_metrics_csv,
    write_validation_metrics_csv,
)
from src.models.yolo_config import YoloConfig
from src.models.yolo_data import CocoImageRecord, load_coco_records


def _target(record: CocoImageRecord) -> ImageTarget:
    """Convert one canonical COCO record to the shared target type."""

    boxes = np.asarray(
        [
            (x, y, x + width, y + height)
            for x, y, width, height in record.boxes_xywh
        ],
        dtype=np.float64,
    ).reshape(-1, 4)
    return ImageTarget(
        image_id=record.file_name,
        image_size=(record.height, record.width),
        boxes_xyxy=boxes,
        labels=np.asarray(record.category_ids, dtype=np.int64),
    )


def evaluate_yolo_checkpoint(config: YoloConfig, checkpoint: Path) -> dict[str, Any]:
    """Evaluate a YOLO checkpoint through the repository's shared metric functions."""

    records, category_names = load_coco_records(config, "validation")
    category_ids = tuple(sorted(category_names))
    yolo_to_category = {index: category_id for index, category_id in enumerate(category_ids)}
    by_name = {record.file_name: record for record in records}
    model = YOLO(checkpoint.as_posix())
    predictions: list[ImagePrediction] = []
    targets: list[ImageTarget] = []
    seen: set[str] = set()
    validation_directory = config.resolve(config.data.yolo_root) / "images" / "val"
    streamed_names = {path.name for path in validation_directory.iterdir() if path.is_file()}
    if streamed_names != set(by_name):
        raise ValueError("YOLO validation view does not match the canonical validation records")
    # Directory streaming preserves source filenames and honors ``batch``.
    # A Python list source is unsuitable here: Ultralytics materializes the
    # entire list as one batch and assigns synthetic image0.jpg-style names.
    results = model.predict(
        source=validation_directory.as_posix(),
        stream=True,
        batch=config.runtime.batch_size,
        imgsz=config.model.input_size,
        device=config.runtime.device,
        amp=config.runtime.amp,
        conf=config.evaluation.inference_minimum_score,
        iou=config.evaluation.nms_iou_threshold,
        max_det=config.evaluation.max_detections,
        agnostic_nms=False,
        augment=False,
        verbose=False,
        save=False,
    )
    for result in results:
        file_name = Path(result.path).name
        if file_name in seen or file_name not in by_name:
            raise ValueError(f"unexpected or duplicate YOLO validation result: {file_name}")
        record = by_name[file_name]
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64, copy=False)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float64, copy=False)
        yolo_labels = boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
        try:
            labels = np.asarray(
                [yolo_to_category[int(label)] for label in yolo_labels], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(
                f"YOLO predicted an unknown class label: {error.args[0]}"
            ) from error
        predictions.append(
            ImagePrediction(
                image_id=file_name,
                image_size=(record.height, record.width),
                boxes_xyxy=xyxy,
                labels=labels,
                scores=scores,
            )
        )
        targets.append(_target(record))
        seen.add(file_name)
    del model, results, result, boxes
    gc.collect()
    torch.cuda.empty_cache()
    if seen != set(by_name):
        missing = sorted(set(by_name) - seen)
        raise RuntimeError(f"YOLO validation omitted {len(missing)} configured images")

    operating = evaluate_operating_point(
        predictions,
        targets,
        class_ids=category_ids,
        score_threshold=config.evaluation.score_threshold,
        iou_threshold=config.evaluation.match_iou_threshold,
        max_detections=config.evaluation.max_detections,
    )
    coco = evaluate_coco(
        predictions,
        targets,
        class_ids=category_ids,
        class_names=category_names,
        minimum_score=config.evaluation.inference_minimum_score,
        max_detections=config.evaluation.max_detections,
    )
    return {
        "operating_point": operating,
        "coco": coco,
        "image_count": len(records),
        "annotation_count": sum(len(record.boxes_xywh) for record in records),
        "test_split_accessed": False,
    }


def _profile_tensor(
    record: CocoImageRecord, config: YoloConfig, device: torch.device
) -> torch.Tensor:
    """Decode, resize, normalize, and transfer one image outside the timed interval."""

    with Image.open(record.path) as image:
        rgb = image.convert("RGB").resize(
            (config.model.input_size, config.model.input_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(rgb, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)


def profile_yolo_checkpoint(config: YoloConfig, checkpoint: Path) -> dict[str, Any]:
    """Profile batch-1 AMP forward plus native NMS and registered-operation FLOPs."""

    records, category_names = load_coco_records(config, "validation")
    yolo = YOLO(checkpoint.as_posix())
    module = yolo.model.float()
    parameters = summarize_parameter_counts(module.parameters())
    # Ultralytics strips optimizer state by marking every saved parameter as
    # non-trainable. Reconstruct the actual training count: the trainer updates
    # all parameters except its explicitly frozen DFL convolution.
    training_frozen = {
        name: parameter.numel()
        for name, parameter in module.named_parameters()
        if name.endswith(".dfl.conv.weight")
    }
    if not training_frozen:
        raise RuntimeError("could not identify Ultralytics' frozen DFL parameter")
    frozen_count = sum(training_frozen.values())
    parameters.update(
        {
            "trainable_parameters": parameters["total_parameters"] - frozen_count,
            "frozen_parameters": frozen_count,
            "training_frozen_parameter_names": sorted(training_frozen),
        }
    )
    device = torch.device(f"cuda:{config.runtime.device}")
    module.to(device).eval()
    module.fuse(verbose=False, imgsz=config.model.input_size)
    total_iterations = config.profiling.warmup_batches + config.profiling.timed_batches
    timings: list[float] = []
    estimated_gflops: float | None = None

    with torch.inference_mode():
        for index in range(total_iterations):
            record = records[index % len(records)]
            tensor = _profile_tensor(record, config, device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            amp_dtype = getattr(torch, config.runtime.amp_dtype)
            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=config.runtime.amp
            ):
                raw = module(tensor)
                prediction = raw[0] if isinstance(raw, tuple) else raw
                prediction = prediction.float()
            non_max_suppression(
                prediction,
                conf_thres=config.evaluation.inference_minimum_score,
                iou_thres=config.evaluation.nms_iou_threshold,
                max_det=config.evaluation.max_detections,
                nc=len(category_names),
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if index >= config.profiling.warmup_batches:
                timings.append(elapsed)

            if index == 0 and config.profiling.profile_flops:
                try:
                    from torch.utils.flop_counter import FlopCounterMode

                    with (
                        FlopCounterMode(display=False) as counter,
                        torch.amp.autocast(
                            device_type="cuda", dtype=amp_dtype, enabled=config.runtime.amp
                        ),
                    ):
                        module(tensor)
                    estimated_gflops = float(counter.get_total_flops()) / 1e9
                except Exception as error:
                    raise RuntimeError(
                        "mandatory YOLO GFLOP profiling failed: "
                        f"{type(error).__name__}: {error}"
                    ) from error

    if estimated_gflops is None or not math.isfinite(estimated_gflops) or estimated_gflops <= 0:
        raise RuntimeError("mandatory YOLO GFLOP profiling did not produce a finite value")
    return {
        "inference": summarize_inference_timings(timings, images_per_batch=1),
        "parameters": parameters,
        "estimated_gflops": estimated_gflops,
        "flop_count_method": (
            "torch.utils.flop_counter registered-operation FLOPs; multiply-add formulas "
            "count 2 FLOPs; fused YOLO forward on one 640-pixel validation image; "
            "native NMS excluded from FLOPs but included in latency"
        ),
    }


def write_yolo_tables(
    config: YoloConfig,
    *,
    run_id: str,
    best_epoch: int,
    validation: dict[str, Any],
    profile: dict[str, Any],
    model_artifact: dict[str, Any],
    training_seconds: float,
    peak_gpu_memory_mib: float,
) -> dict[str, Path]:
    """Write validation and computational tables with the shared schemas."""

    overall = validation["operating_point"]["overall"]
    coco = validation["coco"]
    validation_path = config.resolve(config.outputs.validation_table)
    compute_path = config.resolve(config.outputs.compute_table)
    write_validation_metrics_csv(
        validation_path,
        [
            {
                "run_id": run_id,
                "seed": config.seed,
                "best_epoch": best_epoch,
                "split": "validation",
                "operating_point_score_threshold": config.evaluation.score_threshold,
                "operating_point_match_iou_threshold": config.evaluation.match_iou_threshold,
                "coco_minimum_score": config.evaluation.inference_minimum_score,
                "max_detections": config.evaluation.max_detections,
                "image_count": validation["image_count"],
                "target_count": validation["annotation_count"],
                "operating_point_prediction_count": overall["prediction_count"],
                "coco_prediction_count": coco["prediction_count"],
                "true_positives": overall["tp"],
                "false_positives": overall["fp"],
                "false_negatives": overall["fn"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "map_50": coco["ap50"],
                "map_50_95": coco["ap50_95"],
            }
        ],
    )
    inference = profile["inference"]
    parameters = profile["parameters"]
    write_compute_metrics_csv(
        compute_path,
        [
            {
                "run_id": run_id,
                "seed": config.seed,
                "input_min_size": config.model.input_size,
                "input_max_size": config.model.input_size,
                "amp_dtype": config.runtime.amp_dtype,
                "inference_batch_size": config.profiling.batch_size,
                "timed_batches": inference["timed_batches"],
                "timed_images": inference["timed_images"],
                "throughput_fps": inference["throughput_fps"],
                "mean_latency_ms": inference["mean_latency_ms"],
                "p50_latency_ms": inference["p50_latency_ms"],
                "p95_latency_ms": inference["p95_latency_ms"],
                "total_parameters": parameters["total_parameters"],
                "trainable_parameters": parameters["trainable_parameters"],
                "estimated_gflops": profile["estimated_gflops"],
                "flop_count_method": profile["flop_count_method"],
                "model_size_bytes": model_artifact["size_bytes"],
                "model_size_mib": model_artifact["size_mib"],
                "model_sha256": model_artifact["sha256"],
                "training_seconds": training_seconds,
                "peak_train_gpu_memory_mib": peak_gpu_memory_mib,
            }
        ],
    )
    return {"validation_table": validation_path, "compute_table": compute_path}


def plot_yolo_training_curves(results_csv: Path, destination: Path, *, best_epoch: int) -> Path:
    """Create the Batch 3 four-panel curve figure from Ultralytics' CSV log."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with results_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("Ultralytics results.csv contains no epochs")
    rows = [{key.strip(): float(value) for key, value in row.items()} for row in raw_rows]
    epochs = [int(row["epoch"]) for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            "Training loss",
            "Loss",
            (("train/box_loss", "box"), ("train/cls_loss", "class"), ("train/dfl_loss", "DFL")),
        ),
        (
            axes[0, 1],
            "Validation operating point",
            "Metric",
            (("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")),
        ),
        (
            axes[1, 0],
            "Validation average precision",
            "Average precision",
            (("metrics/mAP50(B)", "mAP@0.5"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95")),
        ),
        (
            axes[1, 1],
            "Learning rate",
            "Learning rate",
            (("lr/pg0", "parameter group 0"),),
        ),
    )
    for axis, title, ylabel, series in panels:
        for key, label in series:
            axis.plot(epochs, [row[key] for row in rows], marker="o", label=label)
        axis.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("YOLO11s training curves — yolo11s_rsna_seed17_full")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def summarize_yolo_artifact(path: Path) -> dict[str, Any]:
    """Expose the shared checkpoint size/hash summary under a detector-neutral name."""

    return summarize_model_artifact(path)
