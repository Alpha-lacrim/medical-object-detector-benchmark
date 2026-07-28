"""Shared infrastructure for the medical object-detector benchmark."""

from meddet_benchmark.config import ExperimentConfig, config_fingerprint, load_experiment
from meddet_benchmark.evaluation import (
    ImagePrediction,
    ImageTarget,
    evaluate_operating_point,
)
from meddet_benchmark.reproducibility import configure_reproducibility

__all__ = [
    "ExperimentConfig",
    "ImagePrediction",
    "ImageTarget",
    "config_fingerprint",
    "configure_reproducibility",
    "evaluate_operating_point",
    "load_experiment",
]
__version__ = "0.1.0"
