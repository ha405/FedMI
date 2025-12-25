import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
from matplotlib.colors import ListedColormap

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def create_circuit_heatmap(data, round_key, class_name, layer_names, model_type="clients_global_model"):
    """
    Creates a heatmap where:
    - X-axis: Neurons/Channels
    - Y-axis: Clients
    - Color: Active (1) or Inactive (0)
    """
    round_data = data[round_key][model_type]
    client_keys = sorted(round_data.keys())
    
    # We need to determine the maximum number of channels to set the X-axis limit
    # For SimpleCNN: conv1=32, conv2=64, conv3=128 (based on your config)
    # But we can just find the max index found in the data to be safe, or hardcode based on architecture
    layer_dims = {'conv1': 32, 'conv2': 64, 'conv3': 128} 

    fig, axes = plt.subplots(len(layer_names), 1, figsize=(12, 4 * len(layer_names)))
    if len(layer_names) == 1: axes = [axes]

    fig.suptitle(f"Circuit Visualization - Round: {round_key} | Class: {class_name}", fontsize=16)

    for i, layer in enumerate(layer_names):
        ax = axes[i]
        num_channels = layer_dims.get(layer, 128) # Default to 128 if unknown
        
        # Build the matrix: [num_clients x num_channels]
        heatmap_data = np.zeros((len(client_keys), num_channels))
        
        for client_idx, client_key in enumerate(client_keys):
            try:
                active_indices = round_data[client_key][class_name][layer]
                for channel_idx in active_indices:
                    if channel_idx < num_channels:
                        heatmap_data[client_idx, channel_idx] = 1
            except KeyError:
                pass # Circuit might be missing/empty

        # Create the Heatmap
        # Using a custom binary colormap: White for 0, Blue for 1
        cmap = ListedColormap(['#f7f7f7', '#2b8cbe']) 
        
        sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=False, 
                    linewidths=0.5, linecolor='gray',
                    yticklabels=client_keys)
        
        ax.set_title(f"Layer: {layer}")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Client")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize Circuits as Heatmaps")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON file (e.g., mnist_sparse_iid.json)")
    parser.add_argument("--round", type=str, default="round_10", help="Which round to visualize (e.g., round_10)")
    parser.add_argument("--cls", type=str, default="1 - one", help="Target class to visualize")
    parser.add_argument("--prefix", type=str, default="viz", help="Prefix for saved image")
    
    args = parser.parse_args()

    data = load_data(args.file)
    if data is None: return

    # Check if round exists
    if args.round not in data:
        print(f"Error: {args.round} not found in data. Available rounds: {list(data.keys())}")
        # Fallback to last round
        args.round = sorted(data.keys())[-1]
        print(f"Falling back to last round: {args.round}")

    # Check if class exists
    sample_client = list(data[args.round]["clients_global_model"].keys())[0]
    available_classes = list(data[args.round]["clients_global_model"][sample_client].keys())
    
    if args.cls not in available_classes:
        print(f"Error: Class '{args.cls}' not found. Available: {available_classes}")
        return

    layers = ["conv1", "conv2", "conv3"]
    
    print(f"Generating visualization for {args.round}, Class: {args.cls}...")
    fig = create_circuit_heatmap(data, args.round, args.cls, layers)
    
    filename = f"{args.prefix}_heatmap_{args.round}_{args.cls.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved visualization to {filename}")
    # plt.show() # Uncomment if you want to see it pop up

if __name__ == "__main__":
    main()