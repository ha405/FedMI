"""Re-export: Circuits — discovery, evaluation, hooks, pruning."""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from circuits.discovery import discover_client_circuit, compute_layer_means
from circuits.evaluation import evaluate_circuit, evaluate_circuit_necessity, evaluate_detailed
from circuits.hooks import (
    binary_gate,
    get_gate_hook,
    get_gate_mean_hook,
    get_hard_mask_hook,
    get_inverse_mask_hook,
    get_mean_ablation_hook,
)
from circuits.pruning import get_current_sparsity, apply_weight_sparsity

__all__ = [
    "discover_client_circuit",
    "compute_layer_means",
    "evaluate_circuit",
    "evaluate_circuit_necessity",
    "evaluate_detailed",
    "binary_gate",
    "get_gate_hook",
    "get_gate_mean_hook",
    "get_hard_mask_hook",
    "get_inverse_mask_hook",
    "get_mean_ablation_hook",
    "get_current_sparsity",
    "apply_weight_sparsity",
]
