import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import itertools

LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def calculate_iou(circuit1, circuit2):
    """Calculates Intersection over Union for two lists of indices."""
    set1 = set(circuit1)
    set2 = set(circuit2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        # If both are empty, they are identical (1.0). If one is empty, 0.0.
        return 1.0 if len(set1) == 0 and len(set2) == 0 else 0.0
    return intersection / union

def get_active_indices(node_data, layer_name):
    """Robustly retrieves indices whether using old or new JSON format."""
    if "active_nodes" in node_data:
        return node_data["active_nodes"].get(layer_name, [])
    else:
        return node_data.get(layer_name, [])

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def analyze_pairwise_iou(data, round_key, output_dir):
    print(f"\n--- Analyzing Pairwise IoU for {round_key} ---")
    
    # 1. Gather all circuits by class
    # Structure: class_circuits = { "0 - zero": { "conv1": [...], ... }, ... }
    class_circuits = {}
    
    # Using Local Models to see distinctness
    model_type = "clients_local_model" 
    
    if model_type not in data[round_key]:
        print(f"Key {model_type} not found in round data.")
        return

    round_data = data[round_key][model_type]
    
    for client_key, client_data in round_data.items():
        for class_name, class_node_data in client_data.items():
            # Collect unique specialist circuit for each class
            class_circuits[class_name] = class_node_data

    # Sort classes for consistent plotting
    class_names = sorted(list(class_circuits.keys()))
    num_classes = len(class_names)
    
    if num_classes < 2:
        print("Not enough classes found to compare.")
        return

    # 2. Generate Pairs
    pairs = list(itertools.combinations(class_names, 2))
    num_pairs = len(pairs)
    print(f"Found {num_classes} classes. Generating {num_pairs} pairwise comparisons.")

    # 3. Plotting Setup
    # Calculate grid size (e.g., 3 columns)
    cols = 3
    rows = (num_pairs + cols - 1) // cols
    
    # Increase height slightly to accommodate titles
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), constrained_layout=True)
    axes = axes.flatten() # Flatten 2D array to 1D for easy iteration

    # 4. Calculate and Plot
    for i, (class_a, class_b) in enumerate(pairs):
        ax = axes[i]
        
        ious = []
        for layer in LAYER_NAMES:
            # Extract indices
            indices_a = get_active_indices(class_circuits[class_a], layer)
            indices_b = get_active_indices(class_circuits[class_b], layer)
            
            # Calculate IoU
            iou = calculate_iou(indices_a, indices_b)
            ious.append(iou)
        
        # Calculate Average IoU across layers
        avg_iou = np.mean(ious)

        # Bar Plot
        ax.bar(LAYER_NAMES, ious, color=['#3498db', '#e74c3c', '#2ecc71'])
        
        # Formatting
        short_name_a = class_a.split(' - ')[0] # "0" instead of "0 - zero"
        short_name_b = class_b.split(' - ')[0]
        
        # Title with Average
        ax.set_title(f"Class {short_name_a} vs {short_name_b}\nAvg IoU: {avg_iou:.2f}", fontsize=11, fontweight='bold')
        
        ax.set_ylim(0, 1.05) 
        ax.set_ylabel("IoU")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add value labels on top of bars
        for j, v in enumerate(ious):
            ax.text(j, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Pairwise Circuit Overlap (IoU) - {round_key}", fontsize=16)
    
    filename = f"pairwise_iou_{round_key}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise IoU between all classes in a specific round.")
    parser.add_argument("--dir", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument("--round", type=str, default="round_10", help="Round to analyze (e.g., round_10).")
    args = parser.parse_args()

    # Find the JSON file
    json_path = os.path.join(args.dir, "circuits_per_round_controlled_noniid.json")
    if not os.path.exists(json_path):
        import glob
        files = glob.glob(os.path.join(args.dir, "circuits_per_round_*.json"))
        if files:
            json_path = files[0]
        else:
            print(f"No circuit JSON found in {args.dir}")
            return

    print(f"Loading: {json_path}")
    data = load_data(json_path)
    if not data: return

    if args.round not in data:
        print(f"Round '{args.round}' not found in data.")
        return

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)

    analyze_pairwise_iou(data, args.round, output_dir)

if __name__ == "__main__":
    main()