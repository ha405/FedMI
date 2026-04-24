"""
FedMI — Unified Package
========================
One-stop import for the entire FedMI framework.

Quick Start:
    >>> from fedmi.config import ExperimentConfig
    >>> from fedmi.runner import ExperimentRunner
    >>> from fedmi.models import get_model
    >>> from fedmi.dataset import get_dataset
    >>> from fedmi.circuits import discover_client_circuit
    >>> from fedmi.playground import load_config, eval_full, REGISTRY

Sub-modules:
    fedmi.config      — ExperimentConfig dataclass
    fedmi.dataset     — Dataset loading, partitioning, transforms
    fedmi.models      — SimpleCNN, get_model
    fedmi.circuits    — Circuit discovery, evaluation, hooks, pruning
    fedmi.runner      — ExperimentRunner, StandaloneRunner
    fedmi.playground  — Playground experiments (apply, stitch, ensemble_distill, lth_prune)
"""

import os, sys

# Ensure the repo root is always importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fedmi import config
from fedmi import dataset
from fedmi import models
from fedmi import circuits
from fedmi import runner
from fedmi import playground

__all__ = [
    "config",
    "dataset",
    "models",
    "circuits",
    "runner",
    "playground",
]
