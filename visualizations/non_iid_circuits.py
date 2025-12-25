import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

# Updated to match your Config (64, 128, 256)
LAYER_DIMS = {'conv1': 32, 'conv2': 64, 'conv3': 128}
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def create_overlap_heatmap(data, round_key):
    """
    Creates a heatmap showing:
    0: Inactive
    1: Local Only (Red)
    2: Global Only (Blue)
    3: Intersection (Purple/Green)
    """
    # 1. Setup Data & Dimensions
    local_data = data[round_key]["clients_local_model"]
    global_data = data[round_key]["clients_global_model"]
    
    # Sort clients numerically
    client_keys = sorted(local_data.keys(), key=lambda c: int(c.split('_')[1]))

    # Calculate total rows required (Sum of classes across all clients)
    total_rows = 0
    for client in client_keys:
        total_rows += len(local_data[client].keys())

    # Dynamic figure height based on rows
    fig, axes = plt.subplots(len(LAYER_NAMES), 1, figsize=(20, 0.5 * total_rows + 4), squeeze=False)
    axes = axes.flatten()

    fig.suptitle(f"Local vs Global Circuit Overlap (Round {round_key.split('_')[1]})", fontsize=16)

    # 2. Iterate Layers
    for i, layer in enumerate(LAYER_NAMES):
        ax = axes[i]
        num_channels = LAYER_DIMS[layer]
        
        heatmap_data = np.zeros((total_rows, num_channels))
        y_labels = []
        
        current_row_idx = 0

        # 3. Iterate Clients & Classes
        for client_key in client_keys:
            # Sort classes to keep order consistent (e.g., "0 - zero", "1 - one")
            classes = sorted(local_data[client_key].keys())
            
            for class_name in classes:
                y_labels.append(f"{client_key} | {class_name}")

                # --- Get Local Indices ---
                try:
                    local_indices = set(local_data[client_key][class_name]["active_nodes"].get(layer, []))
                except (KeyError, TypeError):
                    local_indices = set()

                # --- Get Global Indices ---
                try:
                    global_indices = set(global_data[client_key][class_name]["active_nodes"].get(layer, []))
                except (KeyError, TypeError):
                    global_indices = set()

                # --- Populate Matrix ---
                union_indices = local_indices.union(global_indices)
                
                for idx in union_indices:
                    if idx >= num_channels: continue
                    
                    is_local = idx in local_indices
                    is_global = idx in global_indices
                    
                    if is_local and is_global:
                        heatmap_data[current_row_idx, idx] = 3  # Intersection
                    elif is_global:
                        heatmap_data[current_row_idx, idx] = 2  # Global Only
                    elif is_local:
                        heatmap_data[current_row_idx, idx] = 1  # Local Only
                
                current_row_idx += 1

        # 4. Plotting
        # 0: White, 1: Red (Lost), 2: Blue (Added), 3: Green (Kept)
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
        
        # Rotate y-labels for readability
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize Circuits for the Controlled Non-IID experiment.")
    parser.add_argument("--dir", type=str, required=True, help="Path to experiment dir")
    parser.add_argument("--round", type=str, default="round_10", help="Round to visualize")
    args = parser.parse_args()

    json_file = os.path.join(args.dir, "circuits_per_round_controlled_noniid.json")
    data = load_data(json_file)
    if not data: return

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    if args.round not in data:
        print(f"Round '{args.round}' not found.")
        return

    print(f"\nGenerating Overlap Heatmap for all classes...")
    fig = create_overlap_heatmap(data, args.round)
    if fig:
        filename = f"controlled_noniid_OVERLAP_{args.round}.png"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()