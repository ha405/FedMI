import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from analysis.visualizer.base import BaseVisualizer

def visualize_circuit(client_class_data, title, ax, layer_names=None, layer_sizes=None):
    layer_spacing = 3.0
    neuron_radius = 0.15
    active_color = '#2ecc71'
    inactive_color = '#ecf0f1'
    connection_color = '#3498db'
    connection_alpha = 0.4
    
    active_nodes_dict = client_class_data.get("active_nodes", {})
    connectivity_dict = client_class_data.get("connectivity", {})
    
    # Use provided layer info or extract from data
    if layer_names is None:
        layer_names = sorted(active_nodes_dict.keys())
    if layer_sizes is None:
        layer_sizes = [max(active_nodes_dict.get(ln, [0])) + 1 for ln in layer_names]
    
    max_neurons = max(layer_sizes) if layer_sizes else 64
    vertical_span = max_neurons * 0.15 

    layer_positions = {}
    
    for i, (layer_name, layer_size) in enumerate(zip(layer_names, layer_sizes)):
        try:
            active_indices = set(int(idx) for idx in active_nodes_dict.get(layer_name, []))
        except (TypeError, ValueError):
            active_indices = set()
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

    for i in range(len(layer_names) - 1):
        current_layer = layer_names[i]
        next_layer = layer_names[i + 1]
        
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
    
    ax.set_xlim(-1, len(layer_names) * layer_spacing - 1)
    ax.set_ylim(-1, vertical_span + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

class GraphVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.config = self._load_config()
        self.layer_names = None
        self.layer_sizes = None
        self.class_names = None
    
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
    
    def _load_class_names(self):
        """
        Load class names based on config.num_classes.
        Use numeric labels 0..N-1, derived from config.
        """
        if self.class_names is not None:
            return self.class_names
        
        num_classes = self.config.get('num_classes', 10)
        # Always use numeric labels based on config.num_classes
        self.class_names = [str(i) for i in range(num_classes)]
        
        return self.class_names
    
    def _infer_layer_info(self, circuit_data):
        """Infer layer names and sizes from circuit data and config."""
        if self.layer_names is not None:
            return self.layer_names, self.layer_sizes
        
        # Extract layer names from circuit data
        layer_names = set()
        for client_data in circuit_data.values():
            for class_data in client_data.values():
                active_nodes = class_data.get("active_nodes", {})
                layer_names.update(active_nodes.keys())
        
        layer_names = sorted(list(layer_names))
        
        # Get layer sizes from config
        conv_channels = self.config.get('conv_channels', [32, 64, 128])
        layer_sizes = {}
        
        for i, ln in enumerate(layer_names):
            if 'conv' in ln.lower():
                # Extract conv layer number
                conv_idx = int(''.join(filter(str.isdigit, ln))) - 1
                if conv_idx < len(conv_channels):
                    layer_sizes[ln] = conv_channels[conv_idx]
                else:
                    layer_sizes[ln] = conv_channels[-1]
            else:
                layer_sizes[ln] = 128  # Default for non-conv layers
        
        self.layer_names = layer_names
        self.layer_sizes = [layer_sizes.get(ln, 64) for ln in layer_names]
        
        return self.layer_names, self.layer_sizes

    def run(self, round_key=None):
        if not self.load_data(): return

        if round_key is None:
            keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
            if keys:
                round_key = keys[-1]
            else:
                return

        if round_key not in self.data: return
        
        # Infer layer info from data and load class names
        if self.data[round_key].get("clients_local_model"):
            self._infer_layer_info(self.data[round_key].get("clients_local_model", {}))
        self._load_class_names()

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
                    
                # Choose the class most representative for this client (most active nodes)
                best_class = None
                best_count = -1
                for cls in classes:
                    data = client_data[cls]
                    # Count active nodes across layers if available
                    active = data.get("active_nodes", {})
                    count = 0
                    for v in active.values():
                        try:
                            count += len(v)
                        except Exception:
                            pass
                    if count > best_count:
                        best_count = count
                        best_class = cls

                if best_class is None:
                    ax.axis('off')
                    continue

                circuit_data = client_data[best_class]
                # Map class index to display name using config-driven numeric labels
                try:
                    class_idx = int(best_class)
                    if class_idx < len(self.class_names):
                        display_class_name = self.class_names[class_idx]
                    else:
                        display_class_name = str(class_idx)
                except (ValueError, TypeError):
                    display_class_name = str(best_class)

                title = f"{client_key}\nClass: '{display_class_name}'"
                visualize_circuit(circuit_data, title, ax, self.layer_names, self.layer_sizes)
            
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self.save_plot(fig, f"graph_analysis_{model_type_str.lower()}_{round_key}.png")
