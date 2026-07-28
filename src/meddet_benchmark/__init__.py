"""Shared infrastructure for the medical object-detector benchmark."""

from meddet_benchmark.config import ExperimentConfig, config_fingerprint, load_experiment

__all__ = ["ExperimentConfig", "config_fingerprint", "load_experiment"]
__version__ = "0.1.0"
