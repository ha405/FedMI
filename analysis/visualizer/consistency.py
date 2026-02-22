import os
import json
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
        # Ensure all indices are integers
        return [int(idx) for idx in nodes]
    except (TypeError, ValueError):
        return []

class ConsistencyVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.layer_names = None
    
    def _extract_layer_names(self, circuit_data: Dict) -> List[str]:
        """Extract layer names dynamically from circuit data."""
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
        class_names = sorted(first_round_data[client_keys[0]].keys())
        
        # Extract layer names dynamically
        layer_names = self._extract_layer_names(first_round_data)
        
        plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

        for round_key in round_keys:
            round_data = self.data[round_key]["clients_global_model"]
            for class_name in class_names:
                for layer_name in layer_names:
                    consistency_ious = []
                    for client1_key, client2_key in itertools.combinations(client_keys, 2):
                        circuit1 = get_active_nodes(round_data[client1_key][class_name], layer_name)
                        circuit2 = get_active_nodes(round_data[client2_key][class_name], layer_name)
                        iou = calculate_iou(circuit1, circuit2)
                        consistency_ious.append(iou)
                    
                    avg_iou = np.mean(consistency_ious) if consistency_ious else 0
                    plot_data[class_name][layer_name].append(avg_iou)

        num_classes = len(class_names)
        fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
        if num_classes == 1: axes = [axes]
        fig.suptitle('Inter-Client Circuit Consistency Across Rounds', fontsize=16)

        for i, class_name in enumerate(class_names):
            ax = axes[i]
            for layer_name in layer_names:
                ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
            ax.set_title(f'Class: "{class_name}"')
            ax.set_ylabel('Avg IoU')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.6)
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
        
        # Extract layer names dynamically
        layer_names = self._extract_layer_names(self.data[round_keys[0]]["clients_global_model"])
        
        round_transitions = [f"R{i}->R{i+1}" for i in range(1, len(round_keys))]
        num_clients = len(client_keys)
        fig, axes = plt.subplots(num_clients, 1, figsize=(10, 4 * num_clients), sharex=True, sharey=True)
        if num_clients == 1: axes = [axes]

        fig.suptitle(f'Circuit Stability Per Client (Class: {target_class})', fontsize=16)

        for i, client_key in enumerate(client_keys):
            ax = axes[i]
            plot_data = {layer: [] for layer in layer_names}
            
            for r_idx in range(len(round_keys) - 1):
                round1_key, round2_key = round_keys[r_idx], round_keys[r_idx+1]
                try:
                    c1_data_full = self.data[round1_key]["clients_global_model"][client_key][target_class]
                    c2_data_full = self.data[round2_key]["clients_global_model"][client_key][target_class]
                    
                    for layer in layer_names:
                        c1_nodes = get_active_nodes(c1_data_full, layer)
                        c2_nodes = get_active_nodes(c2_data_full, layer)
                        iou = calculate_iou(c1_nodes, c2_nodes)
                        plot_data[layer].append(iou)
                except KeyError:
                    for layer in layer_names: plot_data[layer].append(np.nan)

            for layer in layer_names:
                ax.plot(round_transitions, plot_data[layer], marker='o', label=layer)
            
            ax.set_title(f'Client: {client_key}')
            ax.set_ylabel('Stability IoU')
            ax.grid(True, linestyle='--', alpha=0.6)
            if i == 0:
                ax.legend(loc='lower right')

        axes[-1].set_xlabel('Round Transition')
        plt.ylim(-0.05, 1.05)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        self.save_plot(fig, f"analysis_Intra_Client_Stability_{target_class.replace(' ', '_')}.png")

    def run(self):
        if not self.load_data(): return
        self.analyze_inter_client_consistency()
        self.analyze_intra_client_stability()
