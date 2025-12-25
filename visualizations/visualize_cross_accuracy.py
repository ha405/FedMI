import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def extract_metrics(data):
    """
    Parses the JSON to extract:
    1. Cross Accuracy (Local Mask on Global Weights)
    2. Global Accuracy (Global Mask on Global Weights) - for baseline comparison
    """
    # Sort rounds numerically (round_1, round_2...)
    round_keys = sorted(data.keys(), key=lambda r: int(r.split('_')[1]))
    
    # Structure: {client: {class: {'rounds': [], 'cross_acc': [], 'global_acc': []}}}
    extracted = {}

    for r in round_keys:
        round_num = int(r.split('_')[1])
        
        # We look inside 'clients_global_model' because that is where we saved the cross-eval metric
        global_model_data = data[r].get("clients_global_model", {})
        
        for client, classes_dict in global_model_data.items():
            if client not in extracted:
                extracted[client] = {}
            
            for class_name, class_data in classes_dict.items():
                if class_name not in extracted[client]:
                    extracted[client][class_name] = {'rounds': [], 'cross_acc': [], 'global_acc': []}
                
                metrics = class_data.get("metrics", {})
                
                # Get the specific metric we care about
                cross_acc = metrics.get("local_mask_on_global_weights_acc", 0)
                global_acc = metrics.get("accuracy", 0) # The upper bound baseline
                
                extracted[client][class_name]['rounds'].append(round_num)
                extracted[client][class_name]['cross_acc'].append(cross_acc)
                extracted[client][class_name]['global_acc'].append(global_acc)
                
    return extracted

def plot_cross_accuracy(extracted_data, output_dir):
    """
    Plot 1: Absolute Accuracy of Local Masks on Global Weights.
    Shows if the Local Circuit shape remains valid in the Global Weights.
    """
    clients = sorted(extracted_data.keys())
    fig, axes = plt.subplots(len(clients), 1, figsize=(10, 5 * len(clients)), sharex=True)
    if len(clients) == 1: axes = [axes]
    
    fig.suptitle("Cross-Evaluation: Accuracy of LOCAL MASK on GLOBAL WEIGHTS\n(Higher is Better = Circuit Location Preserved)", fontsize=16)
    
    for i, client in enumerate(clients):
        ax = axes[i]
        client_data = extracted_data[client]
        
        # Sort classes to keep colors consistent
        classes = sorted(client_data.keys())
        
        for cls in classes:
            rounds = client_data[cls]['rounds']
            accs = client_data[cls]['cross_acc']
            
            if not rounds: continue
            
            ax.plot(rounds, accs, marker='o', linestyle='-', linewidth=2, label=f"{cls}")
        
        ax.set_title(f"{client}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
        
    axes[-1].set_xlabel("Federated Round")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, "cross_accuracy_absolute.png")
    plt.savefig(save_path, dpi=200)
    print(f"Saved Absolute Accuracy Plot: {save_path}")
    plt.close()

def plot_drift_gap(extracted_data, output_dir):
    """
    Plot 2: The 'Drift Gap'. 
    Gap = (Global Mask Acc) - (Local Mask Acc).
    High Gap = The Global model works, but it MOVED the circuit to different neurons.
    Low Gap = The Global model works AND kept the circuit in the same place.
    """
    clients = sorted(extracted_data.keys())
    fig, axes = plt.subplots(len(clients), 1, figsize=(10, 5 * len(clients)), sharex=True)
    if len(clients) == 1: axes = [axes]
    
    fig.suptitle("Circuit Drift Gap: (Global Mask Acc - Local Mask Acc)\n(Lower is Better = Less Physical Drift)", fontsize=16)
    
    for i, client in enumerate(clients):
        ax = axes[i]
        client_data = extracted_data[client]
        classes = sorted(client_data.keys())
        
        for cls in classes:
            rounds = client_data[cls]['rounds']
            cross = np.array(client_data[cls]['cross_acc'])
            glob = np.array(client_data[cls]['global_acc'])
            
            if not rounds: continue
            
            # Gap Calculation
            gap = glob - cross
            
            ax.plot(rounds, gap, marker='s', linestyle='--', linewidth=2, label=f"{cls}")
            
            # Add a zero line reference (Ideal state)
            ax.axhline(0, color='black', linewidth=1, alpha=0.3)

        ax.set_title(f"{client}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Accuracy Gap (%)")
        # Y-lim can be dynamic, but usually 0 to 100
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        
    axes[-1].set_xlabel("Federated Round")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, "cross_accuracy_drift_gap.png")
    plt.savefig(save_path, dpi=200)
    print(f"Saved Drift Gap Plot: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Experiment directory containing json")
    args = parser.parse_args()

    # Find the JSON file (assuming controlled non-iid name pattern)
    json_path = os.path.join(args.dir, "draft1_circuits_per_round_controlled_noniid.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    print(f"Loading: {json_path}")
    data = load_data(json_path)
    
    if not data: return

    output_dir = os.path.join(args.dir, "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse Data
    extracted = extract_metrics(data)
    
    # 2. Plot Absolute Accuracy (Does the local mask work on global weights?)
    plot_cross_accuracy(extracted, output_dir)
    
    # 3. Plot Gap (Did the circuit physically move?)
    plot_drift_gap(extracted, output_dir)

if __name__ == "__main__":
    main()