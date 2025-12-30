"""
Single Circuit Ablation Study.

1. Evaluate FedAvg Baseline.
2. For each Expert Circuit:
   - Isolate the circuit (Zero out irrelevant neurons).
   - Evaluate accuracy.
   - Report overlap stats (How many neurons are shared with other experts?).
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
DEFAULT_EXPERT_MAP = {
    1: 0, 5: 1, 9: 2
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

def isolate_circuit(model, circuits, round_num, target_map, all_maps):
    round_key = f"round_{round_num}"
    target_client = target_map['client']
    target_class = target_map['class']
    
    try:
        target_nodes = circuits[round_key]["clients_local_model"][f"client_{target_client}"][target_class]["active_nodes"]
    except KeyError: return model, 0

    other_nodes_union = {'conv1': set(), 'conv2': set(), 'conv3': set()}
    for cls, client in all_maps.items():
        if cls == target_map['class_idx']: continue
        try:
            c_nodes = circuits[round_key]["clients_local_model"][f"client_{client}"][target_map['names'][cls]]["active_nodes"]
            for layer in ['conv1', 'conv2', 'conv3']:
                other_nodes_union[layer].update(c_nodes.get(layer, []))
        except KeyError: continue

    total_shared_neurons = 0
    conv_layers = ['conv1', 'conv2', 'conv3']
    
    for layer_name in conv_layers:
        layer = model.get_submodule(layer_name)
        keep_indices = set(target_nodes.get(layer_name, []))
        
        # Calculate Shared
        overlap = keep_indices.intersection(other_nodes_union[layer_name])
        total_shared_neurons += len(overlap)
        
        print(f"  [{layer_name}] Keeping {len(keep_indices)} channels (Shared: {len(overlap)})")
        
        total_channels = layer.out_channels
        mask = torch.zeros(total_channels, dtype=torch.bool)
        for idx in keep_indices:
            if idx < total_channels: mask[idx] = True
                
        with torch.no_grad():
            layer.weight.data[~mask, :, :, :] = 0.0
            if layer.bias is not None:
                layer.bias.data[~mask] = 0.0
                
    return model, total_shared_neurons

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
    print("SINGLE CIRCUIT ABLATION (With Overlap Stats)")
    print("="*60)

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes

    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)

    print("Initializing Dense FedAvg Model...")
    base_model = create_fedavg_model(client_models, config)
    
    print("\n--- Baseline FedAvg Accuracy ---")
    baseline_accs = evaluate_per_class(base_model, testloader, config, list(DEFAULT_EXPERT_MAP.keys()))
    for cls, acc in baseline_accs.items():
        print(f"  Class {cls}: {acc:.2f}%")
    
    ablation_results = {}
    shared_stats = {}
    
    for class_idx, client_idx in DEFAULT_EXPERT_MAP.items():
        class_name = class_names[class_idx]
        print(f"\n--- Testing Circuit: Class {class_idx} ({class_name}) ---")
        
        test_model = copy.deepcopy(base_model)
        target_info = {
            'client': client_idx, 'class': class_name, 'class_idx': class_idx, 'names': class_names
        }
        
        test_model, shared_count = isolate_circuit(test_model, circuits, args.round, target_info, DEFAULT_EXPERT_MAP)
        shared_stats[class_idx] = shared_count
        
        res = evaluate_per_class(test_model, testloader, config, [class_idx])
        acc = res[class_idx]
        print(f"  -> Circuit Integrity (Accuracy on Class {class_idx}): {acc:.2f}%")
        ablation_results[class_idx] = acc

    print("\n" + "="*75)
    print("FINAL SUMMARY (Circuit Validity)")
    print("="*75)
    print(f"{'Class':<8} | {'FedAvg Acc':<12} | {'Circuit Acc':<12} | {'Shared Neurons':<15} | {'Drop':<8}")
    print("-" * 75)
    for cls in DEFAULT_EXPERT_MAP.keys():
        base = baseline_accs[cls]
        abl = ablation_results[cls]
        shared = shared_stats[cls]
        drop = base - abl
        print(f"{cls:<8} | {base:<12.2f} | {abl:<12.2f} | {shared:<15} | {drop:<8.2f}")

if __name__ == "__main__":
    main()
