"""Re-export: Playground experiments and utilities."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Playground core utilities ---
from playground.core.loader import load_config, load_model, load_dataset, load_circuits
from playground.core.evaluator import eval_full, eval_sufficiency, eval_necessity, apply_circuit
from playground.core.stitcher import stitch_circuits, available_strategies

# --- Playground experiment classes ---
from playground.experiments import (
    REGISTRY,
    ApplyCircuitExperiment,
    StitchCircuitExperiment,
    EnsembleDistillExperiment,
    LTHPruneExperiment,
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
    # Stitcher
    "stitch_circuits",
    "available_strategies",
    # Experiment classes
    "REGISTRY",
    "ApplyCircuitExperiment",
    "StitchCircuitExperiment",
    "EnsembleDistillExperiment",
    "LTHPruneExperiment",
    "CKACompareExperiment",
]
