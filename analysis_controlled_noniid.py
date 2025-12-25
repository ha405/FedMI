import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict
import os

LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
    set1, set2 = set(circuit1), set(circuit2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 1.0

def get_active_indices(node_data, layer_name):
    """Robustly retrieves indices whether using old or new JSON format."""
    if "active_nodes" in node_data:
        return node_data["active_nodes"].get(layer_name, [])
    else:
        return node_data.get(layer_name, [])

def analyze_specialist_distinctness(final_round_data: Dict, output_prefix: str):
    """
    Checks if Client 0's neurons are different from Client 1's neurons.
    Since they train on disjoint classes, the IoU should be LOW.
    """
    print("\n--- Analysis: Specialist Distinctness (Final Round) ---")
    
    client_keys = sorted(final_round_data.keys(), key=lambda c: int(c.split('_')[1]))
    
    # 1. Aggregate all active neurons for each client across ALL their classes
    client_aggregated_circuits = {c: {l: set() for l in LAYER_NAMES} for c in client_keys}
    
    for client in client_keys:
        classes = final_round_data[client].keys()
        for cls in classes:
            for layer in LAYER_NAMES:
                indices = get_active_indices(final_round_data[client][cls], layer)
                client_aggregated_circuits[client][layer].update(indices)

    # 2. Compare Clients
    avg_iou_by_layer = {layer: [] for layer in LAYER_NAMES}
    
    for layer_name in LAYER_NAMES:
        layer_ious = []
        for i, j in itertools.combinations(range(len(client_keys)), 2):
            client1, client2 = client_keys[i], client_keys[j]
            
            circuit1 = list(client_aggregated_circuits[client1][layer_name])
            circuit2 = list(client_aggregated_circuits[client2][layer_name])
            
            iou = calculate_iou(circuit1, circuit2)
            layer_ious.append(iou)
        
        avg = np.mean(layer_ious) if layer_ious else 0
        avg_iou_by_layer[layer_name] = avg
        print(f"  {layer_name:<8} | Avg. Inter-Client Overlap: {avg:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(avg_iou_by_layer.keys(), avg_iou_by_layer.values(), color='skyblue')
    plt.title('Inter-Client Circuit Overlap (Should be Low for Disjoint Tasks)')
    plt.ylabel('Average IoU')
    plt.xlabel('Layer')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig_path = os.path.join(output_prefix, "Specialist_Distinctness.png")
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()

def analyze_local_vs_global_shift_controlled(all_data: Dict, output_prefix: str):
    """
    Plots IoU between Local Model and Global Model per Client, per Class.
    """
    print("\n--- Analysis: Local vs Global Shift ---")
    
    round_keys = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))
    
    # Detect clients and classes from the last round (most complete)
    last_round = all_data[round_keys[-1]]["clients_local_model"]
    client_keys = sorted(last_round.keys(), key=lambda c: int(c.split('_')[1]))
    
    for client_key in client_keys:
        classes = sorted(last_round[client_key].keys())
        
        num_classes = len(classes)
        if num_classes == 0: continue

        fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
        if num_classes == 1: axes = [axes]
        
        fig.suptitle(f'Local vs Global Similarity: {client_key}', fontsize=16)

        for i, class_name in enumerate(classes):
            ax = axes[i]
            plot_data = {layer: [] for layer in LAYER_NAMES}

            for round_key in round_keys:
                try:
                    local_node = all_data[round_key]["clients_local_model"][client_key][class_name]
                    global_node = all_data[round_key]["clients_global_model"][client_key][class_name]
                    
                    for layer in LAYER_NAMES:
                        l_circ = get_active_indices(local_node, layer)
                        g_circ = get_active_indices(global_node, layer)
                        plot_data[layer].append(calculate_iou(l_circ, g_circ))
                except KeyError:
                    for layer in LAYER_NAMES: plot_data[layer].append(np.nan)

            for layer in LAYER_NAMES:
                # Filter out NaNs for plotting
                valid_indices = [j for j, val in enumerate(plot_data[layer]) if not np.isnan(val)]
                valid_rounds = [j+1 for j in valid_indices]
                valid_ious = [plot_data[layer][j] for j in valid_indices]
                
                ax.plot(valid_rounds, valid_ious, marker='o', label=layer)
            
            ax.set_title(f"Class: {class_name}")
            ax.set_ylabel('IoU')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
        axes[-1].set_xlabel('Federated Round')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = os.path.join(output_prefix, f"Local_vs_Global_{client_key}.png")
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze results from the Controlled Non-IID experiment.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the controlled_noniid experiment directory.")
    args = parser.parse_args()

    json_file = os.path.join(args.dir, "circuits_per_round_controlled_noniid.json")
    
    print(f"Loading results from: {json_file}")
    try:
        with open(json_file, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: The file '{json_file}' was not found.")
        return

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    final_round_key = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))[-1]
    
    analyze_specialist_distinctness(all_data[final_round_key]["clients_global_model"], output_dir)
    analyze_local_vs_global_shift_controlled(all_data, output_dir)

if __name__ == "__main__":
    main()