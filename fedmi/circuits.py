"""Re-export: Circuits — discovery, evaluation, hooks, CKA."""

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
from circuits.cka import linear_cka, extract_circuit_activations, extract_prehead_latents, cka_matrix

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
    "linear_cka",
    "extract_circuit_activations",
    "extract_prehead_latents",
    "cka_matrix",
    "get_current_sparsity",
    "apply_weight_sparsity",
]
