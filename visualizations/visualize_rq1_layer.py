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

def create_combined_heatmap(data, round_key, layer_name, class_names):
    """
    Creates a master heatmap showing all clients for all classes in one layer.
    """
    round_data = data[round_key]["clients_global_model"]
    client_keys = sorted(round_data.keys())
    
    # 1. Determine Dimensions
    # SimpleCNN conv3 is usually 128, but let's be dynamic
    max_idx = 0
    for client in client_keys:
        for cls in class_names:
            if round_data[client][cls][layer_name]:
                max_idx = max(max_idx, max(round_data[client][cls][layer_name]))
    
    num_channels = max(max_idx + 1, 64) # Ensure decent width
    
    # 2. Build the Matrix
    # Total Rows = Num_Classes * Num_Clients
    # We add a small "gap" row of -1s between classes for visual separation
    rows_per_class = len(client_keys)
    gap_size = 1 
    total_rows = (rows_per_class * len(class_names)) + (gap_size * (len(class_names) - 1))
    
    heatmap_data = np.zeros((total_rows, num_channels))
    
    # We create a mask for the gap rows to color them grey later
    mask = np.zeros_like(heatmap_data, dtype=bool)

    y_labels = []
    
    current_row = 0
    for i, cls in enumerate(class_names):
        for client in client_keys:
            indices = round_data[client][cls][layer_name]
            for idx in indices:
                if idx < num_channels:
                    heatmap_data[current_row, idx] = 1
            
            y_labels.append(f"{cls} ({client})")
            current_row += 1
        
        # Add gap row (if not last class)
        if i < len(class_names) - 1:
            # Mark this row as masked (for grey color)
            mask[current_row, :] = True
            y_labels.append("") 
            current_row += 1

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(16, 0.4 * total_rows))
    
    # Custom Colormap: White=0, Blue=1
    cmap = ListedColormap(['#f7f7f7', '#2b8cbe'])
    
    # We plot the data using seaborn
    # We use 'mask' to hide the gap rows, then overlay a grey background for them
    ax.set_facecolor('#505050') # Dark grey background for gaps
    
    sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=False, mask=mask,
                linewidths=0.5, linecolor='gray',
                yticklabels=y_labels)
    
    ax.set_title(f"RQ1 & RQ3 Combined: Circuit Distinctness & Consistency\nLayer: {layer_name} | Round: {round_key}", fontsize=14)
    ax.set_xlabel(f"Channel Index ({layer_name})")
    
    # Adjust tick labels for readability
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to JSON")
    parser.add_argument("--round", type=str, default="round_10")
    parser.add_argument("--layer", type=str, default="conv3")
    parser.add_argument("--prefix", type=str, default="combined_viz")
    args = parser.parse_args()

    data = load_data(args.file)
    if not data: return

    # Validate Round
    if args.round not in data:
        args.round = sorted(data.keys())[-1]
        print(f"Round not found. Defaulting to {args.round}")

    # Auto-detect classes
    sample_client = list(data[args.round]["clients_global_model"].keys())[0]
    classes = sorted(data[args.round]["clients_global_model"][sample_client].keys())

    print(f"Generating Combined Visualization for {args.layer}...")
    fig = create_combined_heatmap(data, args.round, args.layer, classes)
    
    # --- FIX: Define directory and create it if missing ---
    import os
    output_dir = "RQs" # Saves to FedMI/RQs/
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"{args.prefix}_{args.layer}_all_classes.png")
    # ------------------------------------------------------

    plt.savefig(filename, dpi=150)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    main()