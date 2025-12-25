import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
from matplotlib.colors import ListedColormap
import os
import glob
import re

LAYER_DIMS = {'conv1': 64, 'conv2': 128, 'conv3': 256}
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def create_combined_heatmap(data, round_key, layer_name, class_names, model_type, alpha):
    fig, ax = plt.subplots(figsize=(20, 0.4 * len(class_names) * 5 + 2)) # Dynamic height
    
    try:
        round_data = data[round_key][model_type]
        client_keys = sorted(round_data.keys())
    except KeyError:
        plt.close(fig)
        return None

    num_channels = LAYER_DIMS[layer_name]
    
    rows_per_class = len(client_keys)
    total_rows = (rows_per_class * len(class_names)) + (len(class_names) - 1)
    
    heatmap_data = np.zeros((total_rows, num_channels))
    mask = np.zeros_like(heatmap_data, dtype=bool)
    y_labels = []
    
    current_row = 0
    for i, cls in enumerate(class_names):
        for client in client_keys:
            if cls in round_data.get(client, {}):
                indices = round_data[client][cls].get(layer_name, [])
                heatmap_data[current_row, indices] = 1
            y_labels.append(f"{cls} ({client})")
            current_row += 1
        
        if i < len(class_names) - 1:
            mask[current_row, :] = True
            y_labels.append("") 
            current_row += 1

    cmap = ListedColormap(['#f0f0f0', '#1f77b4'])
    ax.set_facecolor('#606060')
    
    sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=False, mask=mask,
                linewidths=0.5, linecolor='white', yticklabels=y_labels)
    
    model_type_str = "Local" if "local" in model_type else "Global"
    ax.set_title(f"Circuit Map ({model_type_str} Models) | Layer: {layer_name} | Alpha: {alpha}", fontsize=16, pad=20)
    ax.set_xlabel(f"Channel Index ({layer_name})")
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate combined circuit heatmaps for all classes and clients.")
    parser.add_argument("--dir", type=str, required=True, help="Directory with experiment results (e.g., ./checkpoints).")
    parser.add_argument("--round", type=str, default="round_10", help="Which round to visualize.")
    parser.add_argument("--layer", type=str, default="conv3", choices=LAYER_NAMES, help="Which layer to visualize.")
    args = parser.parse_args()

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    search_pattern = os.path.join(args.dir, "**/circuits_per_round_*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"FATAL: No 'circuits_per_round_*.json' files found in '{args.dir}'")
        return

    for file_path in json_files:
        print(f"\nProcessing: {file_path}")
        data = load_data(file_path)
        if not data: continue

        match = re.search(r'alpha_([\d.]+)|(controlled_noniid)|(iid)', os.path.basename(file_path))
        alpha = next((g for g in match.groups() if g is not None), "unknown") if match else "unknown"

        if args.round not in data:
            print(f"  Round '{args.round}' not found. Skipping.")
            continue
        
        sample_client = list(data[args.round]["clients_global_model"].keys())[0]
        classes = sorted(data[args.round]["clients_global_model"][sample_client].keys())

        for model_type in ["clients_local_model", "clients_global_model"]:
            fig = create_combined_heatmap(data, args.round, args.layer, classes, model_type, alpha)
            if fig:
                model_type_suffix = "local" if "local" in model_type else "global"
                filename = f"combined_heatmap_{model_type_suffix}_layer_{args.layer}_alpha_{alpha}.png"
                save_path = os.path.join(output_dir, filename)
                fig.savefig(save_path, dpi=200, bbox_inches='tight')
                print(f"  Saved: {save_path}")
                plt.close(fig)

if __name__ == "__main__":
    main()