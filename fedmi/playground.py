"""Re-export: Playground experiments and utilities."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Playground core utilities ---
from playground.core.loader import load_config, load_model, load_dataset, load_circuits
from playground.core.evaluator import eval_full, eval_sufficiency, eval_necessity, apply_circuit

# --- Playground experiment classes ---
from playground.experiments import (
    REGISTRY,
    ApplyCircuitExperiment,
    CKACompareExperiment,
)

__all__ = [
    # Loader
    "load_config",
    "load_model",
    "load_dataset",
    "load_circuits",
    # Evaluator
    "eval_full",
    "eval_sufficiency",
    "eval_necessity",
    "apply_circuit",
    # Experiment classes
    "REGISTRY",
    "ApplyCircuitExperiment",
    "CKACompareExperiment",
]
