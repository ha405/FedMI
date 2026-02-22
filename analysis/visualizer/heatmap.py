import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from analysis.visualizer.base import BaseVisualizer

class HeatmapVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.config = self._load_config()
        self.layer_names = None
        self.layer_dims = None
    
    def _load_config(self):
        """Load config.json to get model parameters."""
        config_path = os.path.join(self.output_dir, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error loading config: {e}")
        return {}
    
    def _infer_layer_info(self, round_data):
        """Infer layer names and dimensions from circuit data and config."""
        if self.layer_names is not None:
            return self.layer_names, self.layer_dims
        
        # Extract layer names from circuit data
        layer_names = set()
        if "clients_local_model" in round_data:
            for client_data in round_data["clients_local_model"].values():
                for class_data in client_data.values():
                    active_nodes = class_data.get("active_nodes", {})
                    layer_names.update(active_nodes.keys())
        
        layer_names = sorted(list(layer_names))
        
        # Get layer dimensions from config
        conv_channels = self.config.get('conv_channels', [32, 64, 128])
        layer_dims = {}
        
        for ln in layer_names:
            if 'conv' in ln.lower():
                # Extract conv layer number
                conv_idx = int(''.join(filter(str.isdigit, ln))) - 1
                if conv_idx < len(conv_channels):
                    layer_dims[ln] = conv_channels[conv_idx]
                else:
                    layer_dims[ln] = conv_channels[-1]
            else:
                layer_dims[ln] = 128  # Default for non-conv layers
        
        self.layer_names = layer_names
        self.layer_dims = layer_dims
        
        return self.layer_names, self.layer_dims

    def create_overlap_heatmap(self, round_key):
        local_data = self.data[round_key]["clients_local_model"]
        global_data = self.data[round_key]["clients_global_model"]
        
        # Infer layer info from data
        self._infer_layer_info(self.data[round_key])
        
        client_keys = sorted(local_data.keys(), key=lambda c: int(c.split('_')[1]))

        total_rows = 0
        for client in client_keys:
            total_rows += len(local_data[client].keys())

        fig, axes = plt.subplots(len(self.layer_names), 1, figsize=(20, 0.5 * total_rows + 4), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f"Local vs Global Circuit Overlap (Round {round_key.split('_')[1]})", fontsize=16)

        for i, layer in enumerate(self.layer_names):
            ax = axes[i]
            num_channels = self.layer_dims[layer]
            
            heatmap_data = np.zeros((total_rows, num_channels))
            y_labels = []
            
            current_row_idx = 0

            for client_key in client_keys:
                classes = sorted(local_data[client_key].keys())
                
                for class_name in classes:
                    y_labels.append(f"{client_key} | {class_name}")

                    try:
                        local_node_list = local_data[client_key][class_name]["active_nodes"].get(layer, [])
                        local_indices = set(int(idx) for idx in local_node_list)
                    except (KeyError, TypeError, ValueError):
                        local_indices = set()

                    try:
                        global_node_list = global_data[client_key][class_name]["active_nodes"].get(layer, [])
                        global_indices = set(int(idx) for idx in global_node_list)
                    except (KeyError, TypeError, ValueError):
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
