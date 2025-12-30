"""
Shared Neuron Replacement Study.

Objective: Investigate the impact of "Polysemantic" Shared Neurons.
Method:
1. Start with Global FedAvg.
2. Identify neurons shared by >1 experts.
3. For each Expert, overwrite ONLY the Shared Neurons in the Global Model 
   with that Expert's original local weights.
4. Measure how this "Bias Injection" affects all classes.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
import copy
from config import Config
from models import get_model
from dataset import get_dataset, get_test_dataloader

# --- Configuration ---
# Update based on your previous logs
EXPERT_MAP = {
    1: 0, # Class 1 -> Client 0
    5: 1, # Class 5 -> Client 1
    9: 2  # Class 9 -> Client 2
}

def load_circuits(checkpoint_dir: str) -> dict:
    json_path = os.path.join(checkpoint_dir, "circuits_per_round_controlled_noniid.json")
    with open(json_path, 'r') as f: return json.load(f)

def load_client_models(checkpoint_dir: str, round_num: int, num_clients: int, config) -> list:
    client_models = []
    for i in range(num_clients):
        model = get_model(config)
        path = os.path.join(checkpoint_dir, f"round_{round_num}", f"client_{i}_model.pt")
        model.load_state_dict(torch.load(path, map_location=config.device))
        model.eval()
        client_models.append(model)
    return client_models

def create_fedavg_model(client_models, config) -> nn.Module:
    global_model = get_model(config)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float)
    for model in client_models:
        local_dict = model.state_dict()
        for k in global_dict.keys():
            global_dict[k] += local_dict[k]
    num_clients = len(client_models)
    for k in global_dict.keys():
        if global_dict[k].is_floating_point():
            global_dict[k] /= num_clients
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate_per_class(model, testloader, config, target_classes):
    model.eval()
    class_correct = {c: 0 for c in target_classes}
    class_total = {c: 0 for c in target_classes}
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, predicted = outputs.max(dim=1)
            
            for c in target_classes:
                mask = (labels == c)
                if mask.sum() > 0:
                    class_correct[c] += (predicted[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()
                    
    results = {}
    for c in target_classes:
        acc = 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        results[c] = acc
    return results

def get_shared_indices(circuits, round_num, expert_map, class_names):
    """
    Identifies which channel indices are used by MORE THAN ONE expert.
    Returns: Dict {layer_name: set(shared_indices)}
    """
    round_key = f"round_{round_num}"
    conv_layers = ['conv1', 'conv2', 'conv3']
    
    # Count occurrences
    counts = {l: {} for l in conv_layers} # {layer: {idx: count}}
    
    for cls_idx, client_idx in expert_map.items():
        cls_name = class_names[cls_idx]
        try:
            active_nodes = circuits[round_key]["clients_local_model"][f"client_{client_idx}"][cls_name]["active_nodes"]
            for layer in conv_layers:
                indices = active_nodes.get(layer, [])
                for idx in indices:
                    counts[layer][idx] = counts[layer].get(idx, 0) + 1
        except KeyError: continue
        
    # Filter for Shared (>1)
    shared_indices = {l: set() for l in conv_layers}
    for layer in conv_layers:
        for idx, count in counts[layer].items():
            if count > 1:
                shared_indices[layer].add(idx)
                
    return shared_indices

def inject_shared_weights(base_model, source_client_model, shared_indices, device):
    """
    Overwrites ONLY the shared neurons in base_model with values from source_client_model.
    """
    modified_model = copy.deepcopy(base_model)
    conv_layers = ['conv1', 'conv2', 'conv3']
    
    total_injected = 0
    
    for layer_name in conv_layers:
        global_layer = modified_model.get_submodule(layer_name)
        source_layer = source_client_model.get_submodule(layer_name)
        
        target_indices = list(shared_indices[layer_name])
        if len(target_indices) == 0: continue
        
        # Convert to tensor index
        idx_tensor = torch.tensor(target_indices).to(device)
        
        with torch.no_grad():
            # Overwrite Weights: [Out, In, k, k]
            # We select the Out channels corresponding to shared indices
            global_layer.weight.data[idx_tensor] = source_layer.weight.data[idx_tensor].clone()
            
            # Overwrite Bias
            if global_layer.bias is not None:
                global_layer.bias.data[idx_tensor] = source_layer.bias.data[idx_tensor].clone()
                
        total_injected += len(target_indices)
        
    return modified_model, total_injected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/controlled_noniid")
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
    
    config = Config()
    config.checkpoint_dir = args.checkpoint_dir
    config.num_clients = args.num_clients
    
    print("="*60)
    print("SHARED NEURON REPLACEMENT STUDY")
    print("="*60)

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes

    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)

    # 1. Base FedAvg
    print("Initializing Dense FedAvg Model...")
    base_model = create_fedavg_model(client_models, config)
    
    # 2. Identify Shared Neurons
    print("Identifying Polysemantic (Shared) Neurons...")
    shared_indices = get_shared_indices(circuits, args.round, EXPERT_MAP, class_names)
    for l, inds in shared_indices.items():
        print(f"  {l}: {len(inds)} shared neurons")

    # 3. Baseline Evaluation
    print("\n--- Baseline FedAvg Accuracy ---")
    base_accs = evaluate_per_class(base_model, testloader, config, list(EXPERT_MAP.keys()))
    for c, acc in base_accs.items():
        print(f"  Class {c}: {acc:.2f}%")

    # 4. Replacement Loop
    print("\n--- Injection Results ---")
    print(f"{'Source':<20} | {'Class 1':<8} | {'Class 5':<8} | {'Class 9':<8}")
    print("-" * 60)
    
    # Print Baseline Row
    print(f"{'FedAvg (Baseline)':<20} | {base_accs[1]:<8.2f} | {base_accs[5]:<8.2f} | {base_accs[9]:<8.2f}")

    for cls_idx, client_idx in EXPERT_MAP.items():
        source_name = f"Client {client_idx} (Cls {cls_idx})"
        
        # Inject Shared Weights from this client
        test_model, count = inject_shared_weights(
            base_model, client_models[client_idx], shared_indices, config.device
        )
        
        # Evaluate
        accs = evaluate_per_class(test_model, testloader, config, list(EXPERT_MAP.keys()))
        
        print(f"{source_name:<20} | {accs[1]:<8.2f} | {accs[5]:<8.2f} | {accs[9]:<8.2f}")

if __name__ == "__main__":
    main()