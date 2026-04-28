"""
Circuit Overlap Visualizer
==========================
Three visualization types for circuit mechanistic analysis:

1. Inter-Client IoU  — pairwise circuit overlap across clients for 5 sampled classes
2. Intra-Client Stability — how stable each client's circuit is across rounds (IoU t vs t-1)
3. Local vs Global Divergence — IoU between local and global circuit per class per round
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_circuits(circuits_dir: str) -> dict:
    """Load all_circuits.json from the circuits directory."""
    path = os.path.join(circuits_dir, "all_circuits.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"all_circuits.json not found in {circuits_dir}")
    with open(path) as f:
        return json.load(f)


def _circuit_iou(circ_a: dict, circ_b: dict) -> float:
    """Compute mean Jaccard similarity across all layers between two circuits."""
    layers = set(circ_a) & set(circ_b)
    if not layers:
        return 0.0
    ious = []
    for layer in layers:
        s1, s2 = set(circ_a[layer]), set(circ_b[layer])
        union = s1 | s2
        ious.append(len(s1 & s2) / len(union) if union else 1.0)
    return float(np.mean(ious))


def _pick_classes(all_circuits: dict, n: int = 5) -> list:
    """Sample up to n class names that appear consistently across rounds."""
    first_round = list(all_circuits.values())[0]
    all_class_names = set()
    for client_data in first_round.get("clients_local_model", {}).values():
        all_class_names.update(client_data.keys())
    sampled = sorted(all_class_names)
    random.shuffle(sampled)
    return sampled[:n]


def _style():
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "ggplot")


# ── Plot 1: Inter-Client IoU ────────────────────────────────────────────────

def plot_inter_client_iou(all_circuits: dict, figures_dir: str, n_classes: int = 5):
    """
    For 5 sampled classes, compute pairwise IoU between all client pairs
    at the final round. Saves one heatmap per class.
    """
    _style()
    sampled_classes = _pick_classes(all_circuits, n_classes)
    rounds = sorted(all_circuits.keys())
    last_round = all_circuits[rounds[-1]]["clients_local_model"]
    client_ids = sorted(last_round.keys())

    if len(client_ids) < 2:
        return

    for cls_name in sampled_classes:
        n = len(client_ids)
        matrix = np.zeros((n, n))
        for i, ci in enumerate(client_ids):
            for j, cj in enumerate(client_ids):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                circ_i = last_round.get(ci, {}).get(cls_name, {}).get("active_nodes", {})
                circ_j = last_round.get(cj, {}).get(cls_name, {}).get("active_nodes", {})
                matrix[i][j] = _circuit_iou(circ_i, circ_j)

        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(client_ids, fontsize=9)
        ax.set_yticklabels(client_ids, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Circuit IoU")
        ax.set_title(f"Inter-Client Circuit IoU — Class: {cls_name} (Round {len(rounds)})", fontsize=12)
        ax.set_xlabel("Client")
        ax.set_ylabel("Client")
        safe_name = cls_name.replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(figures_dir, f"iou_inter_client_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Plot 2: Intra-Client Circuit Stability ──────────────────────────────────

def plot_intra_client_stability(all_circuits: dict, figures_dir: str, n_classes: int = 5):
    """
    For 5 sampled classes, track circuit stability (IoU between consecutive rounds)
    for each client. Plots IoU(round_t, round_{t-1}) over time.
    """
    _style()
    sampled_classes = _pick_classes(all_circuits, n_classes)
    rounds = sorted(all_circuits.keys())
    if len(rounds) < 2:
        return

    first_round_data = all_circuits[rounds[0]]["clients_local_model"]
    client_ids = sorted(first_round_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(client_ids), 1)))

    for cls_name in sampled_classes:
        fig, ax = plt.subplots(figsize=(12, 5))
        for ci, client_id in enumerate(client_ids):
            stability = []
            round_nums = []
            for r in range(1, len(rounds)):
                prev = all_circuits[rounds[r - 1]]["clients_local_model"].get(client_id, {}).get(cls_name, {}).get("active_nodes", {})
                curr = all_circuits[rounds[r]]["clients_local_model"].get(client_id, {}).get(cls_name, {}).get("active_nodes", {})
                if prev and curr:
                    stability.append(_circuit_iou(prev, curr))
                    round_nums.append(r + 1)
            if stability:
                ax.plot(round_nums, stability, "-o", markersize=3, linewidth=1.3,
                        color=colors[ci], label=client_id, alpha=0.85)

        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Circuit Stability (IoU vs prev round)", fontsize=11)
        ax.set_title(f"Intra-Client Circuit Stability — Class: {cls_name}", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, ncol=2)
        safe_name = cls_name.replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(figures_dir, f"iou_intra_stability_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Plot 3: Local vs Global Circuit Divergence ──────────────────────────────

def plot_local_vs_global(all_circuits: dict, figures_dir: str, n_classes: int = 5):
    """
    For 5 sampled classes, compare local circuit vs global circuit IoU across rounds
    for each client. Shows whether local specialization diverges from or aligns with global.
    """
    _style()
    sampled_classes = _pick_classes(all_circuits, n_classes)
    rounds = sorted(all_circuits.keys())
    first_local = all_circuits[rounds[0]].get("clients_local_model", {})
    client_ids = sorted(first_local.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(client_ids), 1)))

    for cls_name in sampled_classes:
        fig, ax = plt.subplots(figsize=(12, 5))
        for ci, client_id in enumerate(client_ids):
            divergence = []
            round_nums = []
            for r_idx, r_key in enumerate(rounds):
                local_circ = all_circuits[r_key].get("clients_local_model", {}).get(client_id, {}).get(cls_name, {}).get("active_nodes", {})
                global_circ = all_circuits[r_key].get("clients_global_model", {}).get(client_id, {}).get(cls_name, {}).get("active_nodes", {})
                if local_circ and global_circ:
                    divergence.append(_circuit_iou(local_circ, global_circ))
                    round_nums.append(r_idx + 1)
            if divergence:
                ax.plot(round_nums, divergence, "-o", markersize=3, linewidth=1.3,
                        color=colors[ci], label=client_id, alpha=0.85)

        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Local ↔ Global Circuit IoU", fontsize=11)
        ax.set_title(f"Local vs Global Circuit Alignment — Class: {cls_name}", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, ncol=2)
        safe_name = cls_name.replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(figures_dir, f"iou_local_vs_global_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Entry Point ────────────────────────────────────────────────────────────

class CircuitOverlapVisualizer:
    def __init__(self, output_dir: str, n_classes: int = 5):
        self.output_dir = output_dir
        self.circuits_dir = os.path.join(output_dir, "circuits")
        self.figures_dir = os.path.join(output_dir, "figures")
        self.n_classes = n_classes
        os.makedirs(self.figures_dir, exist_ok=True)

    def run(self):
        try:
            all_circuits = _load_circuits(self.circuits_dir)
        except FileNotFoundError as e:
            print(f"[CircuitOverlapVisualizer] Skipping: {e}")
            return

        print("[CircuitOverlapVisualizer] Generating circuit overlap plots...")
        plot_inter_client_iou(all_circuits, self.figures_dir, self.n_classes)
        plot_intra_client_stability(all_circuits, self.figures_dir, self.n_classes)
        plot_local_vs_global(all_circuits, self.figures_dir, self.n_classes)
        print(f"[CircuitOverlapVisualizer] Saved to {self.figures_dir}")
