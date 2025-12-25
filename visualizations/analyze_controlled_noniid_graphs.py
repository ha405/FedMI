import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import argparse
import os

LAYER_SIZES = [64, 128, 256]
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

CLASSES_TO_DISCOVER = {
    0: (0, "0 - zero"),
    1: (3, "3 - three"),
    2: (6, "6 - six")
}

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def visualize_circuit(client_class_data, title, ax):
    """
    client_class_data: The dictionary for a specific class containing "active_nodes" and "connectivity"
    """
    layer_spacing = 3.0
    neuron_radius = 0.15
    active_color = '#2ecc71'
    inactive_color = '#ecf0f1'
    connection_color = '#3498db'
    connection_alpha = 0.4
    
    max_neurons = max(LAYER_SIZES)
    vertical_span = max_neurons * 0.15 

    layer_positions = {}
    
    # --- KEY CHANGE: Access "active_nodes" ---
    active_nodes_dict = client_class_data.get("active_nodes", {})
    connectivity_dict = client_class_data.get("connectivity", {})
    # -----------------------------------------

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

    # Draw Connections
    for i in range(len(LAYER_NAMES) - 1):
        current_layer = LAYER_NAMES[i]
        next_layer = LAYER_NAMES[i + 1]
        
        # --- KEY CHANGE: Use the new "connectivity" dict structure ---
        # The structure is layer_name -> {dest_idx: [src_idx, ...]}
        # We need edges for the NEXT layer (where connections are defined)
        layer_edges = connectivity_dict.get(next_layer, {})
        
        for dest_idx, sources in layer_edges.items():
            dest_idx = int(dest_idx) # JSON keys are strings
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

def main():
    parser = argparse.ArgumentParser(description="Visualize graphs for the Controlled Non-IID experiment.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the experiment directory (e.g., ./checkpoints/controlled_noniid).")
    parser.add_argument("--round", type=str, default="round_10", help="Which round to visualize.")
    args = parser.parse_args()

    json_file = os.path.join(args.dir, "circuits_per_round_controlled_noniid.json")
    data = load_data(json_file)
    if not data: return

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    if args.round not in data:
        print(f"Error: Round '{args.round}' not found. Available: {list(data.keys())}")
        return

    num_clients = len(CLASSES_TO_DISCOVER)

    for model_type in ["clients_local_model", "clients_global_model"]:
        print(f"\nGenerating graph visualization for {model_type}...")
        
        fig, axes = plt.subplots(1, num_clients, figsize=(7 * num_clients, 7), squeeze=False)
        axes = axes.flatten()

        model_type_str = "Local" if "local" in model_type else "Global"
        fig.suptitle(f"Controlled Non-IID Circuit Graphs ({model_type_str} Models, Round {args.round.split('_')[1]})",
                     fontsize=16, fontweight='bold', y=1.0)
        
        round_data = data[args.round][model_type]
        
        for client_id in range(num_clients):
            client_key = f"client_{client_id}"
            _, class_name = CLASSES_TO_DISCOVER[client_id]
            ax = axes[client_id]
            
            if client_key in round_data and class_name in round_data[client_key]:
                # Pass the WHOLE class object (containing active_nodes AND connectivity)
                circuit_data = round_data[client_key][class_name]
                title = f"{client_key}\n(Specialized on Class '{class_name}')"
                visualize_circuit(circuit_data, title, ax)
            else:
                ax.text(0.5, 0.5, f"No circuit data for\n{client_key} - {class_name}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"controlled_noniid_graphs_{model_type_str.lower()}_{args.round}.png"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()