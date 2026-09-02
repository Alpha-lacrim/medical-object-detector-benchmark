"""Shared infrastructure for the medical object-detector benchmark."""

from .coco_evaluation import evaluate_coco
from .config import ExperimentConfig, config_fingerprint, load_experiment
from .corruptions import apply_corruption, load_corruptions
from .data_audit import audit_yolo_dataset
from .evaluation import (
    ImagePrediction,
    ImageTarget,
    evaluate_operating_point,
)
from .reproducibility import configure_reproducibility

__all__ = [
    "ExperimentConfig",
    "ImagePrediction",
    "ImageTarget",
    "apply_corruption",
    "audit_yolo_dataset",
    "config_fingerprint",
    "configure_reproducibility",
    "evaluate_coco",
    "evaluate_operating_point",
    "load_corruptions",
    "load_experiment",
]
__version__ = "2.0.0"
