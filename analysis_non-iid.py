import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re
from typing import List, Dict

def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
    """Calculates the Intersection over Union (IoU) for two circuits."""
    set1 = set(circuit1)
    set2 = set(circuit2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0
    return intersection / union

def analyze_rq1_inter_class_distinction(final_round_data: Dict, output_prefix: str, alpha: str):
    print(f"\n--- RQ1: Inter-Class Distinction (Alpha: {alpha}) ---")
    
    client_keys = sorted(final_round_data.keys())
    if not client_keys:
        print("  No client data found for RQ1.")
        return

    first_client_data = final_round_data[client_keys[0]]
    class_names = sorted(first_client_data.keys())
    layer_names = sorted(first_client_data[class_names[0]].keys())
    
    avg_iou_by_layer: Dict[str, float] = {layer: [] for layer in layer_names}

    for layer_name in layer_names:
        layer_ious = []
        for client_key in client_keys:
            client_circuits = final_round_data[client_key]
            for class1, class2 in itertools.combinations(class_names, 2):
                circuit1 = client_circuits[class1][layer_name]
                circuit2 = client_circuits[class2][layer_name]
                iou = calculate_iou(circuit1, circuit2)
                layer_ious.append(iou)
        
        avg_iou_by_layer[layer_name] = np.mean(layer_ious) if layer_ious else 0
        print(f"  {layer_name:<8} | Avg. IoU between classes: {avg_iou_by_layer[layer_name]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.bar(avg_iou_by_layer.keys(), avg_iou_by_layer.values(), color='coral')
    plt.title(f'RQ1: Inter-Class Circuit Overlap (Final Round) - Alpha: {alpha}')
    plt.ylabel('Average IoU (Lower is Better)')
    plt.xlabel('Layer')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig_path = f"{output_prefix}_RQ1_alpha_{alpha}.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()

def analyze_rq2_intra_client_stability(all_data: Dict, output_prefix: str, alpha: str, target_class: str = None):
    print(f"\n--- RQ2: Intra-Client Stability (Alpha: {alpha}) ---")
    
    round_keys = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))
    if len(round_keys) < 2:
        print("  Not enough rounds to analyze stability.")
        return

    first_round_data = all_data[round_keys[0]]["clients_global_model"]
    client_keys = sorted(first_round_data.keys())
    available_classes = sorted(first_round_data[client_keys[0]].keys())
    
    if target_class is None or target_class not in available_classes:
        target_class = available_classes[0]
    
    print(f"  Analyzing stability for Class: '{target_class}'")
    
    layer_names = sorted(first_round_data[client_keys[0]][target_class].keys())
    round_transitions = [f"R{i+1}->R{i+2}" for i in range(len(round_keys) - 1)]

    num_clients = len(client_keys)
    fig, axes = plt.subplots(num_clients, 1, figsize=(10, 4 * num_clients), sharex=True, sharey=True)
    if num_clients == 1: axes = [axes]

    fig.suptitle(f'RQ2: Circuit Stability Per Client (Class: {target_class}) - Alpha: {alpha}', fontsize=16)

    for i, client_key in enumerate(client_keys):
        ax = axes[i]
        plot_data = {layer: [] for layer in layer_names}
        
        for r_idx in range(len(round_keys) - 1):
            round1_key, round2_key = round_keys[r_idx], round_keys[r_idx+1]
            try:
                c1_data = all_data[round1_key]["clients_global_model"][client_key][target_class]
                c2_data = all_data[round2_key]["clients_global_model"][client_key][target_class]
                for layer in layer_names:
                    iou = calculate_iou(c1_data[layer], c2_data[layer])
                    plot_data[layer].append(iou)
            except KeyError:
                for layer in layer_names: plot_data[layer].append(0)

        for layer in layer_names:
            ax.plot(round_transitions, plot_data[layer], marker='o', label=layer)
        
        ax.set_title(f'Client: {client_key}')
        ax.set_ylabel('Stability IoU')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0: ax.legend(loc='lower right')

    axes[-1].set_xlabel('Round Transition')
    plt.ylim(-0.05, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = f"{output_prefix}_RQ2_alpha_{alpha}_{target_class.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()

def analyze_rq3_inter_client_consistency(all_data: Dict, output_prefix: str, alpha: str):
    print(f"\n--- RQ3: Inter-Client Consistency (Alpha: {alpha}) ---")

    round_keys = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))
    first_round_data = all_data[round_keys[0]]["clients_global_model"]
    client_keys = sorted(first_round_data.keys())
    class_names = sorted(first_round_data[client_keys[0]].keys())
    layer_names = sorted(first_round_data[client_keys[0]][class_names[0]].keys())
    
    plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

    for round_key in round_keys:
        round_data = all_data[round_key]["clients_global_model"]
        for class_name in class_names:
            for layer_name in layer_names:
                consistency_ious = []
                for client1_key, client2_key in itertools.combinations(client_keys, 2):
                    circuit1 = round_data[client1_key][class_name][layer_name]
                    circuit2 = round_data[client2_key][class_name][layer_name]
                    iou = calculate_iou(circuit1, circuit2)
                    consistency_ious.append(iou)
                
                avg_iou = np.mean(consistency_ious) if consistency_ious else 0
                plot_data[class_name][layer_name].append(avg_iou)

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
    if num_classes == 1: axes = [axes]
    fig.suptitle(f'RQ3: Inter-Client Circuit Consistency - Alpha: {alpha}', fontsize=16)

    for i, class_name in enumerate(class_names):
        ax = axes[i]
        for layer_name in layer_names:
            ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
        ax.set_title(f'Class: "{class_name}"')
        ax.set_ylabel('Avg IoU')
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
    axes[-1].set_xlabel('Federated Round')
    plt.xticks(range(1, len(round_keys) + 1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = f"{output_prefix}_RQ3_alpha_{alpha}.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()

def analyze_rq4_local_vs_global_shift(all_data: Dict, output_prefix: str, alpha: str, client_key: str = "client_0"):
    print(f"\n--- RQ4: Local vs Global Shift (Alpha: {alpha}, Client: {client_key}) ---")
    
    round_keys = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))
    
    try:
        first_round_local = all_data[round_keys[0]]["clients_local_model"]
        if client_key not in first_round_local:
            client_key = sorted(first_round_local.keys())[0]
            
        first_round_data = first_round_local[client_key]
        class_names = sorted(first_round_data.keys())
        layer_names = sorted(first_round_data[class_names[0]].keys())
    except (KeyError, IndexError):
        return

    plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

    for round_key in round_keys:
        local_data = all_data[round_key]["clients_local_model"].get(client_key)
        global_data = all_data[round_key]["clients_global_model"].get(client_key)
        
        if not local_data or not global_data: continue

        for class_name in class_names:
            for layer_name in layer_names:
                local_circuit = local_data[class_name][layer_name]
                global_circuit = global_data[class_name][layer_name]
                iou = calculate_iou(local_circuit, global_circuit)
                plot_data[class_name][layer_name].append(iou)

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
    if num_classes == 1: axes = [axes]
    
    fig.suptitle(f'RQ4: Local vs Global Circuit Similarity - Alpha: {alpha}, {client_key}', fontsize=16)

    for i, class_name in enumerate(class_names):
        ax = axes[i]
        for layer_name in layer_names:
            ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
        ax.set_title(f'Class: "{class_name}"')
        ax.set_ylabel('IoU')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
    axes[-1].set_xlabel('Federated Round')
    plt.xticks(range(1, len(round_keys) + 1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = f"{output_prefix}_RQ4_alpha_{alpha}_{client_key}.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze circuit evolution from FedMI results.")
    # --- UPDATED: Accept a directory ---
    parser.add_argument("--dir", type=str, required=False, help="Directory containing JSON files.")
    parser.add_argument("--prefix", type=str, default="analysis_results", help="Prefix for output image files.")
    parser.add_argument("--target_class", type=str, default="1 - one", help="Class for RQ2 per-client analysis.")
    args = parser.parse_args()

    # --- UPDATED: Find all circuit JSON files ---
    search_path = os.path.join(args.dir, "json_files_non-iid/circuits_per_round_*.json")
    json_files = glob.glob(search_path)
    
    if not json_files:
        print(f"FATAL: No 'circuits_per_round_*.json' files found in '{args.dir}'")
        return
        
    # Create output directory for plots
    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    # --- UPDATED: Loop through each file and run analysis ---
    for file_path in json_files:
        print(f"\n=======================================================")
        print(f"Analyzing file: {os.path.basename(file_path)}")
        
        # Extract alpha from filename using regex
        match = re.search(r'alpha_([\d.]+)\.json', file_path)
        if match:
            alpha = match.group(1)
        elif 'iid' in file_path:
            alpha = "iid"
        else:
            alpha = "unknown"
        
        print(f"Detected Alpha: {alpha}")
        print(f"=======================================================")

        try:
            with open(file_path, 'r') as f:
                all_data = json.load(f)
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue

        # Get final round data for RQ1
        final_round_key = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))[-1]
        final_round_global_data = all_data[final_round_key]["clients_global_model"]
        
        # Create a unique prefix for each alpha's plots
        file_prefix = os.path.join(output_dir, f"alpha_{alpha}")

        # Run all analyses
        analyze_rq1_inter_class_distinction(final_round_global_data, file_prefix, alpha)
        analyze_rq2_intra_client_stability(all_data, file_prefix, alpha, target_class=args.target_class)
        analyze_rq3_inter_client_consistency(all_data, file_prefix, alpha)
        analyze_rq4_local_vs_global_shift(all_data, file_prefix, alpha, client_key="client_0")

if __name__ == "__main__":
    main()