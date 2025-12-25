import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# Define the specific targets based on your description
TARGET_A = {"client": "client_1", "class": "5 - five"}
TARGET_B = {"client": "client_2", "class": "7 - seven"}
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def calculate_iou(circuit1, circuit2):
    """Calculates Intersection over Union for two lists of indices."""
    set1 = set(circuit1)
    set2 = set(circuit2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0 if len(set1) == 0 and len(set2) == 0 else 0.0
    return intersection / union

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def analyze_cross_client_class_iou(data, output_dir):
    # Sort rounds numerically
    round_keys = sorted(data.keys(), key=lambda r: int(r.split('_')[1]))
    rounds = [int(r.split('_')[1]) for r in round_keys]
    
    iou_history = {layer: [] for layer in LAYER_NAMES}

    print(f"\n--- Comparing {TARGET_A['client']}/{TARGET_A['class']} vs {TARGET_B['client']}/{TARGET_B['class']} ---")

    for r in round_keys:
        # Access Local Models
        local_data = data[r]["clients_local_model"]
        
        # Extract Active Nodes
        # Handle potential missing keys gracefully
        try:
            # Note: Using .get() chains to avoid crashes if a client/class is missing in a specific round
            nodes_a = local_data.get(TARGET_A["client"], {}).get(TARGET_A["class"], {}).get("active_nodes", {})
            nodes_b = local_data.get(TARGET_B["client"], {}).get(TARGET_B["class"], {}).get("active_nodes", {})
        except AttributeError:
            # Fallback for old JSON structure if "active_nodes" key doesn't exist
            nodes_a = local_data.get(TARGET_A["client"], {}).get(TARGET_A["class"], {})
            nodes_b = local_data.get(TARGET_B["client"], {}).get(TARGET_B["class"], {})

        for layer in LAYER_NAMES:
            circ_a = nodes_a.get(layer, [])
            circ_b = nodes_b.get(layer, [])
            
            iou = calculate_iou(circ_a, circ_b)
            iou_history[layer].append(iou)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    for layer in LAYER_NAMES:
        plt.plot(rounds, iou_history[layer], marker='o', linestyle='-', label=layer)

    plt.title(f"Circuit Overlap (IoU): Class 5 (Client 1) vs Class 6 (Client 2)\n(Higher = More Shared Neurons)", fontsize=14)
    plt.xlabel("Federated Round")
    plt.ylabel("Intersection over Union (IoU)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xticks(rounds)
    
    filename = "iou_class5_vs_class6.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot to: {save_path}")
    plt.close()

    # Print final round stats
    print(f"Final Round ({rounds[-1]}) IoUs:")
    for layer in LAYER_NAMES:
        print(f"  {layer}: {iou_history[layer][-1]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Compare specific local circuits across clients.")
    parser.add_argument("--dir", type=str, required=True, help="Path to experiment directory.")
    args = parser.parse_args()

    # Locate the JSON file
    json_path = os.path.join(args.dir, "circuits_per_round_controlled_noniid.json")
    
    data = load_data(json_path)
    if not data: return

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)

    analyze_cross_client_class_iou(data, output_dir)

if __name__ == "__main__":
    main()