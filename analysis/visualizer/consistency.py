import os
import json
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from analysis.visualizer.base import BaseVisualizer

def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
    set1 = set(circuit1)
    set2 = set(circuit2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0
    return intersection / union

def get_active_nodes(circuit_data: Dict, layer_name: str) -> List[int]:
    try:
        if "active_nodes" in circuit_data and layer_name in circuit_data["active_nodes"]:
            nodes = circuit_data["active_nodes"][layer_name]
        elif layer_name in circuit_data:
            nodes = circuit_data[layer_name]
        else:
            return []
        return [int(idx) for idx in nodes]
    except (TypeError, ValueError):
        return []

def _style_ax(ax):
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')

class ConsistencyVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.layer_names = None

    def _extract_layer_names(self, circuit_data: Dict) -> List[str]:
        if self.layer_names is not None:
            return self.layer_names
        layer_names = set()
        for client_data in circuit_data.values():
            for class_data in client_data.values():
                active_nodes = class_data.get("active_nodes", {})
                layer_names.update(active_nodes.keys())
        self.layer_names = sorted(list(layer_names))
        return self.layer_names

    def analyze_inter_client_consistency(self):
        print("\n--- Inter-Client Consistency (Per Round) ---")
        if not self.data: return

        round_keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
        first_round_data = self.data[round_keys[0]]["clients_global_model"]
        client_keys = sorted(first_round_data.keys())
        all_classes = sorted(first_round_data[client_keys[0]].keys())
        class_names = random.sample(all_classes, min(5, len(all_classes)))

        layer_names = self._extract_layer_names(first_round_data)
        plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

        for round_key in round_keys:
            round_data = self.data[round_key]["clients_global_model"]
            for class_name in class_names:
                for layer_name in layer_names:
                    consistency_ious = []
                    for c1, c2 in itertools.combinations(client_keys, 2):
                        iou = calculate_iou(
                            get_active_nodes(round_data[c1][class_name], layer_name),
                            get_active_nodes(round_data[c2][class_name], layer_name),
                        )
                        consistency_ious.append(iou)
                    plot_data[class_name][layer_name].append(
                        np.mean(consistency_ious) if consistency_ious else 0
                    )

        with plt.style.context('default'):
            fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 5 * len(class_names)), sharex=True)
            if len(class_names) == 1: axes = [axes]
            fig.patch.set_facecolor('white')
            fig.suptitle('Inter-Client Circuit Consistency Across Rounds', fontsize=16)

            for i, class_name in enumerate(class_names):
                ax = axes[i]
                for layer_name in layer_names:
                    ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
                ax.set_title(f'Class: "{class_name}"')
                ax.set_ylabel('Avg IoU')
                ax.set_ylim(-0.05, 1.05)
                _style_ax(ax)
                ax.legend()

            axes[-1].set_xlabel('Federated Round')
            plt.xticks(range(1, len(round_keys) + 1))
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self.save_plot(fig, "analysis_Inter_Client_Consistency.png")

    def analyze_intra_client_stability(self, target_class=None):
        print("\n--- Intra-Client Stability (Per Client) ---")
        if not self.data: return

        round_keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
        if len(round_keys) < 2:
            print("Not enough rounds to analyze stability.")
            return

        first_round_data = self.data[round_keys[0]]["clients_global_model"]
        client_keys = sorted(first_round_data.keys())
        available_classes = sorted(first_round_data[client_keys[0]].keys())

        if target_class is None or target_class not in available_classes:
            target_class = available_classes[0]
        print(f"Analyzing stability for Class: '{target_class}'")

        layer_names = self._extract_layer_names(self.data[round_keys[0]]["clients_global_model"])
        round_transitions = [f"R{i}->R{i+1}" for i in range(1, len(round_keys))]

        with plt.style.context('default'):
            fig, axes = plt.subplots(len(client_keys), 1, figsize=(10, 4 * len(client_keys)), sharex=True, sharey=True)
            if len(client_keys) == 1: axes = [axes]
            fig.patch.set_facecolor('white')
            fig.suptitle(f'Circuit Stability Per Client (Class: {target_class})', fontsize=16)

            for i, client_key in enumerate(client_keys):
                ax = axes[i]
                plot_data = {layer: [] for layer in layer_names}

                for r_idx in range(len(round_keys) - 1):
                    r1, r2 = round_keys[r_idx], round_keys[r_idx + 1]
                    try:
                        d1 = self.data[r1]["clients_global_model"][client_key][target_class]
                        d2 = self.data[r2]["clients_global_model"][client_key][target_class]
                        for layer in layer_names:
                            plot_data[layer].append(calculate_iou(
                                get_active_nodes(d1, layer), get_active_nodes(d2, layer)
                            ))
                    except KeyError:
                        for layer in layer_names: plot_data[layer].append(np.nan)

                for layer in layer_names:
                    ax.plot(round_transitions, plot_data[layer], marker='o', label=layer)

                ax.set_title(f'Client: {client_key}')
                ax.set_ylabel('Stability IoU')
                _style_ax(ax)
                if i == 0:
                    ax.legend(loc='lower right')

            axes[-1].set_xlabel('Round Transition')
            plt.ylim(-0.05, 1.05)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            self.save_plot(fig, f"analysis_Intra_Client_Stability_{target_class.replace(' ', '_')}.png")

    def analyze_local_vs_global(self):
        print("\n--- Local vs Global Circuit Similarity (Per Client) ---")
        if not self.data: return

        round_keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
        first_local = self.data[round_keys[0]].get("clients_local_model", {})
        client_keys = sorted(first_local.keys())
        if not client_keys: return

        all_classes = sorted(first_local[client_keys[0]].keys())
        class_names = random.sample(all_classes, min(5, len(all_classes)))
        layer_names = self._extract_layer_names(first_local)

        for client_key in client_keys:
            plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

            for round_key in round_keys:
                local_data = self.data[round_key].get("clients_local_model", {}).get(client_key)
                global_data = self.data[round_key].get("clients_global_model", {}).get(client_key)
                if not local_data or not global_data:
                    for cls in class_names:
                        for layer in layer_names:
                            plot_data[cls][layer].append(np.nan)
                    continue
                for class_name in class_names:
                    for layer in layer_names:
                        plot_data[class_name][layer].append(calculate_iou(
                            get_active_nodes(local_data.get(class_name, {}), layer),
                            get_active_nodes(global_data.get(class_name, {}), layer),
                        ))

            with plt.style.context('default'):
                fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 5 * len(class_names)), sharex=True)
                if len(class_names) == 1: axes = [axes]
                fig.patch.set_facecolor('white')
                fig.suptitle(f'Local vs Global Similarity: {client_key}', fontsize=16)

                for i, class_name in enumerate(class_names):
                    ax = axes[i]
                    for layer in layer_names:
                        ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer], marker='o', label=layer)
                    ax.set_title(f'Class: "{class_name}"')
                    ax.set_ylabel('IoU')
                    ax.set_ylim(-0.05, 1.05)
                    _style_ax(ax)
                    ax.legend()

                axes[-1].set_xlabel('Federated Round')
                plt.xticks(range(1, len(round_keys) + 1))
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                self.save_plot(fig, f"Local_vs_Global_{client_key}.png")

    def run(self):
        if not self.load_data(): return
        self.analyze_inter_client_consistency()
        self.analyze_intra_client_stability()
        self.analyze_local_vs_global()
