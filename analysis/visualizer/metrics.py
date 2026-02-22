import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from analysis.visualizer.base import BaseVisualizer

def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
    set1, set2 = set(circuit1), set(circuit2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 1.0

def get_active_indices(node_data, layer_name):
    try:
        if "active_nodes" in node_data:
            nodes = node_data["active_nodes"].get(layer_name, [])
        else:
            nodes = node_data.get(layer_name, [])
        # Ensure all indices are integers
        return [int(idx) for idx in nodes]
    except (TypeError, ValueError):
        return []

class MetricsVisualizer(BaseVisualizer):
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

    def analyze_specialist_distinctness(self, final_round_data: Dict):
        print("\n--- Analysis: Specialist Distinctness (Final Round) ---")
        
        client_keys = sorted(final_round_data.keys(), key=lambda c: int(c.split('_')[1]))
        
        # Extract layer names dynamically
        layer_names = self._extract_layer_names(final_round_data)
        
        client_aggregated_circuits = {c: {l: set() for l in layer_names} for c in client_keys}
        
        for client in client_keys:
            classes = final_round_data[client].keys()
            for cls in classes:
                for layer in layer_names:
                    indices = get_active_indices(final_round_data[client][cls], layer)
                    client_aggregated_circuits[client][layer].update(indices)

        avg_iou_by_layer = {layer: [] for layer in layer_names}
        
        for layer_name in layer_names:
            layer_ious = []
            for i, j in itertools.combinations(range(len(client_keys)), 2):
                client1, client2 = client_keys[i], client_keys[j]
                
                circuit1 = list(client_aggregated_circuits[client1][layer_name])
                circuit2 = list(client_aggregated_circuits[client2][layer_name])
                
                iou = calculate_iou(circuit1, circuit2)
                layer_ious.append(iou)
            
            avg = np.mean(layer_ious) if layer_ious else 0
            avg_iou_by_layer[layer_name] = avg

        fig = plt.figure(figsize=(10, 6))
        plt.bar(avg_iou_by_layer.keys(), avg_iou_by_layer.values(), color='skyblue')
        plt.title('Inter-Client Circuit Overlap (Should be Low for Disjoint Tasks)')
        plt.ylabel('Average IoU')
        plt.xlabel('Layer')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        self.save_plot(fig, "Specialist_Distinctness.png")

    def analyze_local_vs_global_shift_controlled(self):
        print("\n--- Analysis: Local vs Global Shift ---")
        
        round_keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
        last_round = self.data[round_keys[-1]]["clients_local_model"]
        client_keys = sorted(last_round.keys(), key=lambda c: int(c.split('_')[1]))
        
        # Extract layer names dynamically
        layer_names = self._extract_layer_names(last_round)
        
        for client_key in client_keys:
            classes = sorted(last_round[client_key].keys())
            num_classes = len(classes)
            if num_classes == 0: continue

            fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
            if num_classes == 1: axes = [axes]
            
            fig.suptitle(f'Local vs Global Similarity: {client_key}', fontsize=16)

            for i, class_name in enumerate(classes):
                ax = axes[i]
                plot_data = {layer: [] for layer in layer_names}

                for round_key in round_keys:
                    try:
                        local_node = self.data[round_key]["clients_local_model"][client_key][class_name]
                        global_node = self.data[round_key]["clients_global_model"][client_key][class_name]
                        
                        for layer in layer_names:
                            l_circ = get_active_indices(local_node, layer)
                            g_circ = get_active_indices(global_node, layer)
                            plot_data[layer].append(calculate_iou(l_circ, g_circ))
                    except KeyError:
                        for layer in layer_names: plot_data[layer].append(np.nan)

                for layer in layer_names:
                    valid_indices = [j for j, val in enumerate(plot_data[layer]) if not np.isnan(val)]
                    valid_rounds = [j+1 for j in valid_indices]
                    valid_ious = [plot_data[layer][j] for j in valid_indices]
                    
                    ax.plot(valid_rounds, valid_ious, marker='o', label=layer)
                
                ax.set_title(f"Class: {class_name}")
                ax.set_ylabel('IoU')
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                
            axes[-1].set_xlabel('Federated Round')
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            self.save_plot(fig, f"Local_vs_Global_{client_key}.png")

    def run(self):
        if not self.load_data(): return
        
        sorted_keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
        if not sorted_keys: return
        final_round_key = sorted_keys[-1]
        
        self.analyze_specialist_distinctness(self.data[final_round_key]["clients_global_model"])
        self.analyze_local_vs_global_shift_controlled()
