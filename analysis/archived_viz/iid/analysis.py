# import json
# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# from typing import List, Dict

# def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
#     """Calculates the Intersection over Union (IoU) for two circuits."""
#     set1 = set(circuit1)
#     set2 = set(circuit2)
    
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
    
#     if union == 0:
#         return 1.0  # If both circuits are empty, they are identical.
#     return intersection / union

# def analyze_rq1_inter_class_distinction(final_round_data: Dict, output_prefix: str):
#     print("\n--- Inter-Class Distinction (Final Round) ---")
    
#     client_keys = sorted(final_round_data.keys())
#     if not client_keys:
#         print("No client data found for RQ1.")
#         return

#     first_client_data = final_round_data[client_keys[0]]
#     class_names = sorted(first_client_data.keys())
#     layer_names = sorted(first_client_data[class_names[0]].keys())
    
#     avg_iou_by_layer: Dict[str, float] = {layer: [] for layer in layer_names}

#     for layer_name in layer_names:
#         layer_ious = []
#         for client_key in client_keys:
#             client_circuits = final_round_data[client_key]
#             for class1, class2 in itertools.combinations(class_names, 2):
#                 circuit1 = client_circuits[class1][layer_name]
#                 circuit2 = client_circuits[class2][layer_name]
#                 iou = calculate_iou(circuit1, circuit2)
#                 layer_ious.append(iou)
        
#         avg_iou_by_layer[layer_name] = np.mean(layer_ious) if layer_ious else 0
#         print(f"  {layer_name:<8} | Avg. IoU between classes: {avg_iou_by_layer[layer_name]:.4f}")

#     plt.figure(figsize=(10, 6))
#     plt.bar(avg_iou_by_layer.keys(), avg_iou_by_layer.values(), color='coral')
#     plt.title('Average Inter-Class Circuit Overlap (Final Round)')
#     plt.ylabel('Average IoU (Lower is Better)')
#     plt.xlabel('Layer') 
#     plt.ylim(0, 1.05)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     fig_path = f"{output_prefix}_RQ1_Inter_Class_Distinction.png"
#     plt.savefig(fig_path)
#     print(f"Figure saved to {fig_path}")
#     plt.close()

# def analyze_rq2_intra_client_stability(all_data: Dict, output_prefix: str, target_class: str = None):
#     """
#     RQ2: Intra-Client Stability (Per Client).
#     Generates subplots (one per client) showing circuit stability over rounds for a SINGLE class.
#     """
#     print("\n--- Intra-Client Stability (Per Client) ---")
    
#     round_keys = sorted(all_data.keys())
#     if len(round_keys) < 2:
#         print("  Not enough rounds to analyze stability.")
#         return

#     first_round_data = all_data[round_keys[0]]["clients_global_model"]
#     client_keys = sorted(first_round_data.keys())
#     available_classes = sorted(first_round_data[client_keys[0]].keys())
    
#     # Select target class
#     if target_class is None or target_class not in available_classes:
#         target_class = available_classes[0] # Default to first class if not specified
    
#     print(f"  Analyzing stability for Class: '{target_class}'")
    
#     layer_names = sorted(first_round_data[client_keys[0]][target_class].keys())
#     round_transitions = [f"R{i+1}->R{i+2}" for i in range(len(round_keys) - 1)]

#     # Setup plot: One subplot per client
#     num_clients = len(client_keys)
#     fig, axes = plt.subplots(num_clients, 1, figsize=(10, 4 * num_clients), sharex=True, sharey=True)
#     if num_clients == 1: axes = [axes]

#     fig.suptitle(f'Circuit Stability Per Client (Class: {target_class})', fontsize=16)

#     for i, client_key in enumerate(client_keys):
#         ax = axes[i]
        
#         # Prepare data containers for this client
#         plot_data = {layer: [] for layer in layer_names}
        
#         for r_idx in range(len(round_keys) - 1):
#             round1_key = round_keys[r_idx]
#             round2_key = round_keys[r_idx+1]
            
#             # Get data for this client in consecutive rounds
#             try:
#                 c1_data = all_data[round1_key]["clients_global_model"][client_key][target_class]
#                 c2_data = all_data[round2_key]["clients_global_model"][client_key][target_class]
                
#                 for layer in layer_names:
#                     iou = calculate_iou(c1_data[layer], c2_data[layer])
#                     plot_data[layer].append(iou)
#             except KeyError:
#                 print(f"Data missing for {client_key} in {round1_key}/{round2_key}")
#                 for layer in layer_names: plot_data[layer].append(0)

#         # Plot lines for this client
#         for layer in layer_names:
#             ax.plot(round_transitions, plot_data[layer], marker='o', label=layer)
        
#         ax.set_title(f'Client: {client_key}')
#         ax.set_ylabel('Stability IoU')
#         ax.grid(True, linestyle='--', alpha=0.6)
#         if i == 0: # Only legend on top plot to save space
#             ax.legend(loc='lower right')

#     axes[-1].set_xlabel('Round Transition')
#     plt.ylim(-0.05, 1.05)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])

#     fig_path = f"{output_prefix}_RQ2_Intra_Client_Stability_{target_class}.png"
#     plt.savefig(fig_path)
#     print(f"Figure saved to {fig_path}")
#     plt.close()

# def analyze_rq3_inter_client_consistency(all_data: Dict, output_prefix: str):
#     print("\n--- Inter-Client Consistency (Per Round) ---")

#     round_keys = sorted(all_data.keys())
#     first_round_data = all_data[round_keys[0]]["clients_global_model"]
#     client_keys = sorted(first_round_data.keys())
#     class_names = sorted(first_round_data[client_keys[0]].keys())
#     layer_names = sorted(first_round_data[client_keys[0]][class_names[0]].keys())
    
#     plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

#     for round_key in round_keys:
#         round_data = all_data[round_key]["clients_global_model"]
#         for class_name in class_names:
#             for layer_name in layer_names:
#                 consistency_ious = []
#                 for client1_key, client2_key in itertools.combinations(client_keys, 2):
#                     circuit1 = round_data[client1_key][class_name][layer_name]
#                     circuit2 = round_data[client2_key][class_name][layer_name]
#                     iou = calculate_iou(circuit1, circuit2)
#                     consistency_ious.append(iou)
                
#                 avg_iou = np.mean(consistency_ious) if consistency_ious else 0
#                 plot_data[class_name][layer_name].append(avg_iou)

#     num_classes = len(class_names)
#     fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
#     if num_classes == 1: axes = [axes]
#     fig.suptitle('Inter-Client Circuit Consistency Across Rounds', fontsize=16)

#     for i, class_name in enumerate(class_names):
#         ax = axes[i]
#         for layer_name in layer_names:
#             ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
#         ax.set_title(f'Class: "{class_name}"')
#         ax.set_ylabel('Avg IoU')
#         ax.set_ylim(0, 1.05)
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend()
        
#     axes[-1].set_xlabel('Federated Round')
#     plt.xticks(range(1, len(round_keys) + 1))
#     plt.tight_layout(rect=[0, 0, 1, 0.96])

#     fig_path = f"{output_prefix}_RQ3_Inter_Client_Consistency.png"
#     plt.savefig(fig_path)
#     print(f"Figure saved to {fig_path}")
#     plt.close()

# def analyze_rq4_local_vs_global_shift(all_data: Dict, output_prefix: str, client_key: str = "client_0"):
#     print(f"\n--- RQ4: Local vs Global Circuit Shift (Client: {client_key}) ---")
    
#     round_keys = sorted(all_data.keys())
    
#     try:
#         first_round_local = all_data[round_keys[0]]["clients_local_model"]
#         if client_key not in first_round_local:
#             client_key = sorted(first_round_local.keys())[0]
#             print(f"  Note: {client_key} not found, switching to {client_key}")
            
#         first_round_data = first_round_local[client_key]
#         class_names = sorted(first_round_data.keys())
#         layer_names = sorted(first_round_data[class_names[0]].keys())
#     except (KeyError, IndexError):
#         print(f"  Error initializing RQ4. Skipping.")
#         return

#     plot_data = {cls: {layer: [] for layer in layer_names} for cls in class_names}

#     for round_key in round_keys:
#         local_data = all_data[round_key]["clients_local_model"].get(client_key)
#         global_data = all_data[round_key]["clients_global_model"].get(client_key)
        
#         if not local_data or not global_data:
#             continue

#         for class_name in class_names:
#             for layer_name in layer_names:
#                 local_circuit = local_data[class_name][layer_name]
#                 global_circuit = global_data[class_name][layer_name]
#                 iou = calculate_iou(local_circuit, global_circuit)
#                 plot_data[class_name][layer_name].append(iou)

#     num_classes = len(class_names)
#     fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
#     if num_classes == 1: axes = [axes]
    
#     fig.suptitle(f'Local vs Global Circuit Similarity - {client_key}', fontsize=16)

#     for i, class_name in enumerate(class_names):
#         ax = axes[i]
#         for layer_name in layer_names:
#             ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
#         ax.set_title(f'Class: "{class_name}"')
#         ax.set_ylabel('IoU')
#         ax.set_ylim(-0.05, 1.05)
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend()
        
#     axes[-1].set_xlabel('Federated Round')
#     plt.xticks(range(1, len(round_keys) + 1))
#     plt.tight_layout(rect=[0, 0, 1, 0.96])

#     fig_path = f"{output_prefix}_RQ4_Local_vs_Global.png"
#     plt.savefig(fig_path)
#     print(f"Figure saved to {fig_path}")
#     plt.close()

# def main():
#     parser = argparse.ArgumentParser(description="Analyze circuit evolution from FedMI results.")
#     parser.add_argument("--file", type=str, required=True, help="Path to JSON.")
#     parser.add_argument("--prefix", type=str, default="analysis", help="Prefix for output.")
#     parser.add_argument("--target_class", type=str, default="bird", help="Class for RQ2 per-client analysis.")
    
#     args = parser.parse_args()

#     print(f"Loading results from: {args.file}")
#     try:
#         with open(args.file, 'r') as f:
#             all_data = json.load(f)
#     except FileNotFoundError:
#         print(f"FATAL: The file '{args.file}' was not found.")
#         return

#     final_round_key = sorted(all_data.keys())[-1]
#     final_round_global_data = all_data[final_round_key]["clients_global_model"]

#     analyze_rq1_inter_class_distinction(final_round_global_data, args.prefix)
    
#     # Pass the target class to RQ2
#     analyze_rq2_intra_client_stability(all_data, args.prefix, target_class=args.target_class)
    
#     analyze_rq3_inter_client_consistency(all_data, args.prefix)
#     analyze_rq4_local_vs_global_shift(all_data, args.prefix, client_key="client_0")

# if __name__ == "__main__":
#     main()

import json, os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict

# <<< FIX 1: Define layer names statically, as they are no longer discoverable from the top-level keys.
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
    """Calculates the Intersection over Union (IoU) for two circuits."""
    set1 = set(circuit1)
    set2 = set(circuit2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0
    return intersection / union

def get_active_nodes(circuit_data: Dict, layer_name: str) -> List[int]:
    """
    <<< FIX 2: New helper function to safely get the list of active nodes
    from the new, nested JSON structure.
    """
    if "active_nodes" in circuit_data and layer_name in circuit_data["active_nodes"]:
        return circuit_data["active_nodes"][layer_name]
    # Fallback for old format, just in case
    elif layer_name in circuit_data:
        return circuit_data[layer_name]
    return []


def analyze_inter_client_consistency(all_data: Dict, output_prefix: str):
    print("\n--- Inter-Client Consistency (Per Round) ---")

    round_keys = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))
    first_round_data = all_data[round_keys[0]]["clients_global_model"]
    client_keys = sorted(first_round_data.keys())
    
    # Get class names from the first client in the first round
    class_names = sorted(first_round_data[client_keys[0]].keys())
    
    plot_data = {cls: {layer: [] for layer in LAYER_NAMES} for cls in class_names}

    for round_key in round_keys:
        round_data = all_data[round_key]["clients_global_model"]
        for class_name in class_names:
            for layer_name in LAYER_NAMES:
                consistency_ious = []
                for client1_key, client2_key in itertools.combinations(client_keys, 2):
                    # <<< FIX 3: Use the new helper function to get node lists
                    circuit1 = get_active_nodes(round_data[client1_key][class_name], layer_name)
                    circuit2 = get_active_nodes(round_data[client2_key][class_name], layer_name)
                    iou = calculate_iou(circuit1, circuit2)
                    consistency_ious.append(iou)
                
                avg_iou = np.mean(consistency_ious) if consistency_ious else 0
                plot_data[class_name][layer_name].append(avg_iou)

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
    if num_classes == 1: axes = [axes]
    fig.suptitle('Inter-Client Circuit Consistency Across Rounds', fontsize=16)

    for i, class_name in enumerate(class_names):
        ax = axes[i]
        for layer_name in LAYER_NAMES:
            ax.plot(range(1, len(round_keys) + 1), plot_data[class_name][layer_name], marker='o', label=layer_name)
        ax.set_title(f'Class: "{class_name}"')
        ax.set_ylabel('Avg IoU')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
    axes[-1].set_xlabel('Federated Round')
    plt.xticks(range(1, len(round_keys) + 1))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = f"{output_prefix}_Inter_Client_Consistency.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()

def analyze_intra_client_stability(all_data: Dict, output_prefix: str, target_class: str = None):
    print("\n--- Intra-Client Stability (Per Client) ---")
    
    round_keys = sorted(all_data.keys(), key=lambda r: int(r.split('_')[1]))
    if len(round_keys) < 2:
        print("Not enough rounds to analyze stability.")
        return

    first_round_data = all_data[round_keys[0]]["clients_global_model"]
    client_keys = sorted(first_round_data.keys())
    available_classes = sorted(first_round_data[client_keys[0]].keys())
    
    if target_class is None or target_class not in available_classes:
        target_class = available_classes[0]
    
    print(f"Analyzing stability for Class: '{target_class}'")
    
    round_transitions = [f"R{i}->R{i+1}" for i in range(1, len(round_keys))]

    num_clients = len(client_keys)
    fig, axes = plt.subplots(num_clients, 1, figsize=(10, 4 * num_clients), sharex=True, sharey=True)
    if num_clients == 1: axes = [axes]

    fig.suptitle(f'Circuit Stability Per Client (Class: {target_class})', fontsize=16)

    for i, client_key in enumerate(client_keys):
        ax = axes[i]
        plot_data = {layer: [] for layer in LAYER_NAMES}
        
        for r_idx in range(len(round_keys) - 1):
            round1_key, round2_key = round_keys[r_idx], round_keys[r_idx+1]
            try:
                c1_data_full = all_data[round1_key]["clients_global_model"][client_key][target_class]
                c2_data_full = all_data[round2_key]["clients_global_model"][client_key][target_class]
                
                for layer in LAYER_NAMES:
                    # <<< FIX 4: Use the new helper function here as well
                    c1_nodes = get_active_nodes(c1_data_full, layer)
                    c2_nodes = get_active_nodes(c2_data_full, layer)
                    iou = calculate_iou(c1_nodes, c2_nodes)
                    plot_data[layer].append(iou)
            except KeyError:
                for layer in LAYER_NAMES: plot_data[layer].append(np.nan)

        for layer in LAYER_NAMES:
            ax.plot(round_transitions, plot_data[layer], marker='o', label=layer)
        
        ax.set_title(f'Client: {client_key}')
        ax.set_ylabel('Stability IoU')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend(loc='lower right')

    axes[-1].set_xlabel('Round Transition')
    plt.ylim(-0.05, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = f"{output_prefix}_Intra_Client_Stability_{target_class.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze IID circuit evolution from FedMI results.")
    parser.add_argument("--file", type=str, required=True, help="Path to the IID JSON results file.")
    parser.add_argument("--prefix", type=str, default="figures/iid/analysis", help="Prefix for output image files.")
    parser.add_argument("--target_class", type=str, default="0 - zero", help="Class for stability analysis.")
    
    args = parser.parse_args()

    print(f"Loading results from: {args.file}")
    try:
        with open(args.file, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: The file '{args.file}' was not found.")
        return

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the analyses
    analyze_inter_client_consistency(all_data, args.prefix)
    analyze_intra_client_stability(all_data, args.prefix, target_class=args.target_class)

if __name__ == "__main__":
    main()