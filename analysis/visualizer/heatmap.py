import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from analysis.visualizer.base import BaseVisualizer

LAYER_DIMS = {'conv1': 32, 'conv2': 64, 'conv3': 128}
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

class HeatmapVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def create_overlap_heatmap(self, round_key):
        local_data = self.data[round_key]["clients_local_model"]
        global_data = self.data[round_key]["clients_global_model"]
        
        client_keys = sorted(local_data.keys(), key=lambda c: int(c.split('_')[1]))

        total_rows = 0
        for client in client_keys:
            total_rows += len(local_data[client].keys())

        fig, axes = plt.subplots(len(LAYER_NAMES), 1, figsize=(20, 0.5 * total_rows + 4), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f"Local vs Global Circuit Overlap (Round {round_key.split('_')[1]})", fontsize=16)

        for i, layer in enumerate(LAYER_NAMES):
            ax = axes[i]
            num_channels = LAYER_DIMS[layer]
            
            heatmap_data = np.zeros((total_rows, num_channels))
            y_labels = []
            
            current_row_idx = 0

            for client_key in client_keys:
                classes = sorted(local_data[client_key].keys())
                
                for class_name in classes:
                    y_labels.append(f"{client_key} | {class_name}")

                    try:
                        local_indices = set(local_data[client_key][class_name]["active_nodes"].get(layer, []))
                    except (KeyError, TypeError):
                        local_indices = set()

                    try:
                        global_indices = set(global_data[client_key][class_name]["active_nodes"].get(layer, []))
                    except (KeyError, TypeError):
                        global_indices = set()

                    union_indices = local_indices.union(global_indices)
                    
                    for idx in union_indices:
                        if idx >= num_channels: continue
                        
                        is_local = idx in local_indices
                        is_global = idx in global_indices
                        
                        if is_local and is_global:
                            heatmap_data[current_row_idx, idx] = 3
                        elif is_global:
                            heatmap_data[current_row_idx, idx] = 2
                        elif is_local:
                            heatmap_data[current_row_idx, idx] = 1
                    
                    current_row_idx += 1

            cmap = ListedColormap(['#ffffff', '#e74c3c', '#3498db', '#2ecc71'])
            
            sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, 
                        linewidths=0.5, linecolor='lightgray',
                        yticklabels=y_labels, vmin=0, vmax=3)
            
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0.37, 1.1, 1.85, 2.6])
            cbar.set_ticklabels(['Inactive', 'Local Only (Lost)', 'Global Only (New)', 'Intersection (Preserved)'])
            
            ax.set_title(f"Layer: {layer}", fontsize=14, pad=10)
            ax.set_xlabel("Channel Index")
            ax.set_ylabel("")
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig

    def run(self, round_key=None):
        if not self.load_data(): return

        if round_key is None:
            keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
            if keys:
                round_key = keys[-1]
            else:
                return

        if round_key not in self.data:
            print(f"[{self.__class__.__name__}] Round '{round_key}' not found in data.")
            return

        print(f"[{self.__class__.__name__}] Generating heatmap for {round_key}...")
        fig = self.create_overlap_heatmap(round_key)
        self.save_plot(fig, f"heatmap_overlap_{round_key}.png")
