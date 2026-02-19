import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from analysis.visualizer.base import BaseVisualizer

LAYER_SIZES = [64, 128, 256]
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def visualize_circuit(client_class_data, title, ax):
    layer_spacing = 3.0
    neuron_radius = 0.15
    active_color = '#2ecc71'
    inactive_color = '#ecf0f1'
    connection_color = '#3498db'
    connection_alpha = 0.4
    
    max_neurons = max(LAYER_SIZES)
    vertical_span = max_neurons * 0.15 

    layer_positions = {}
    
    active_nodes_dict = client_class_data.get("active_nodes", {})
    connectivity_dict = client_class_data.get("connectivity", {})

    for i, (layer_name, layer_size) in enumerate(zip(LAYER_NAMES, LAYER_SIZES)):
        active_indices = set(active_nodes_dict.get(layer_name, []))
        x = i * layer_spacing
        
        active_list = sorted(list(active_indices))
        inactive_sample = [j for j in range(layer_size) if j not in active_indices]
        
        num_context = min(15, len(inactive_sample))
        context_indices = [inactive_sample[int(j * len(inactive_sample) / num_context)] for j in range(num_context)] if num_context > 0 else []
        
        neurons_to_draw = sorted(list(set(active_list + context_indices)))
        
        if not neurons_to_draw: continue

        vertical_spacing = vertical_span / (len(neurons_to_draw) + 1)
        y_positions = [vertical_span - (j + 1) * vertical_spacing for j in range(len(neurons_to_draw))]
        
        layer_positions[layer_name] = {}
        
        for idx, neuron_idx in enumerate(neurons_to_draw):
            y = y_positions[idx]
            layer_positions[layer_name][neuron_idx] = (x, y)
            is_active = neuron_idx in active_indices
            
            circle = plt.Circle((x, y), neuron_radius, color=(active_color if is_active else inactive_color),
                              alpha=(1.0 if is_active else 0.3), ec='black',
                              linewidth=(0.5 if is_active else 0.2), zorder=3)
            ax.add_patch(circle)
            
            if is_active:
                ax.text(x, y, str(neuron_idx), ha='center', va='center', fontsize=5, fontweight='bold', zorder=4)
        
        ax.text(x, vertical_span + 0.5, layer_name, ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i in range(len(LAYER_NAMES) - 1):
        current_layer = LAYER_NAMES[i]
        next_layer = LAYER_NAMES[i + 1]
        
        layer_edges = connectivity_dict.get(next_layer, {})
        
        for dest_idx, sources in layer_edges.items():
            dest_idx = int(dest_idx)
            if dest_idx in layer_positions.get(next_layer, {}):
                for src_idx in sources:
                    if src_idx in layer_positions.get(current_layer, {}):
                        x1, y1 = layer_positions[current_layer][src_idx]
                        x2, y2 = layer_positions[next_layer][dest_idx]
                        
                        arrow = FancyArrowPatch((x1 + neuron_radius, y1), (x2 - neuron_radius, y2),
                                              arrowstyle='-', color=connection_color, alpha=connection_alpha,
                                              linewidth=0.5, zorder=1)
                        ax.add_patch(arrow)
    
    ax.set_xlim(-1, len(LAYER_NAMES) * layer_spacing - 1)
    ax.set_ylim(-1, vertical_span + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

class GraphVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, round_key=None):
        if not self.load_data(): return

        if round_key is None:
            keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
            if keys:
                round_key = keys[-1]
            else:
                return

        if round_key not in self.data: return

        for model_type in ["clients_local_model", "clients_global_model"]:
            if model_type not in self.data[round_key]: continue
            
            print(f"[{self.__class__.__name__}] Generating graph visualization for {model_type}...")
            round_data = self.data[round_key][model_type]
            
            client_keys = sorted(round_data.keys(), key=lambda c: int(c.split('_')[1]))
            num_clients = len(client_keys)
            if num_clients == 0: continue

            cols = min(num_clients, 4)
            rows = (num_clients + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
            axes = axes.flatten()

            model_type_str = "Local" if "local" in model_type else "Global"
            fig.suptitle(f"Circuit Graphs ({model_type_str} Models, {round_key})",
                        fontsize=16, fontweight='bold', y=1.0)
            
            for i, client_key in enumerate(client_keys):
                ax = axes[i]
                
                client_data = round_data[client_key]
                classes = sorted(client_data.keys())
                
                if not classes:
                    ax.axis('off')
                    continue
                    
                class_name = classes[0]
                circuit_data = client_data[class_name]
                
                title = f"{client_key}\nClass: '{class_name}'"
                visualize_circuit(circuit_data, title, ax)
            
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self.save_plot(fig, f"graph_analysis_{model_type_str.lower()}_{round_key}.png")
