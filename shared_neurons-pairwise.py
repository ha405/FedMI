"""
Pairwise Compatibility Study (All Pairs).

Goal: Determine which pairs of clients have "Compatible" shared neurons vs "Conflicting" ones.
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
# Update this based on your dataset/setup
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

def create_pairwise_fedavg(model_a, model_b, config) -> nn.Module:
    global_model = get_model(config)
    dict_a = model_a.state_dict()
    dict_b = model_b.state_dict()
    global_dict = global_model.state_dict()
    
    for k in global_dict.keys():
        if global_dict[k].is_floating_point():
            global_dict[k] = (dict_a[k] + dict_b[k]) / 2.0
        else:
            global_dict[k] = dict_a[k]
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

def get_pairwise_shared_indices(circuits, round_num, client_a, class_a, client_b, class_b):
    round_key = f"round_{round_num}"
    conv_layers = ['conv1', 'conv2', 'conv3']
    shared_indices = {l: set() for l in conv_layers}
    
    try:
        nodes_a = circuits[round_key]["clients_local_model"][f"client_{client_a}"][class_a]["active_nodes"]
        nodes_b = circuits[round_key]["clients_local_model"][f"client_{client_b}"][class_b]["active_nodes"]
    except KeyError: return shared_indices
    
    for layer in conv_layers:
        set_a = set(nodes_a.get(layer, []))
        set_b = set(nodes_b.get(layer, []))
        shared_indices[layer] = set_a.intersection(set_b)
        
    return shared_indices

def inject_shared_weights(base_model, source_client_model, shared_indices, device):
    modified_model = copy.deepcopy(base_model)
    conv_layers = ['conv1', 'conv2', 'conv3']
    
    for layer_name in conv_layers:
        global_layer = modified_model.get_submodule(layer_name)
        source_layer = source_client_model.get_submodule(layer_name)
        
        target_indices = list(shared_indices[layer_name])
        if len(target_indices) == 0: continue
        
        idx_tensor = torch.tensor(target_indices).to(device)
        with torch.no_grad():
            global_layer.weight.data[idx_tensor] = source_layer.weight.data[idx_tensor].clone()
            if global_layer.bias is not None:
                global_layer.bias.data[idx_tensor] = source_layer.bias.data[idx_tensor].clone()
                
    return modified_model

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
    print("ALL-PAIRS SHARED NEURON STUDY")
    print("="*60)

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes

    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)

    # Define Pairs (Class A vs Class B)
    pairs = [
        (1, 5),
        (1, 9),
        (5, 9)
    ]

    for cls_a, cls_b in pairs:
        print(f"\n" + "="*40)
        print(f"ANALYZING PAIR: Class {cls_a} vs Class {cls_b}")
        print("="*40)
        
        client_a = EXPERT_MAP[cls_a]
        client_b = EXPERT_MAP[cls_b]
        
        # 1. FedAvg
        base_model = create_pairwise_fedavg(client_models[client_a], client_models[client_b], config)
        
        # 2. Shared Neurons
        shared_indices = get_pairwise_shared_indices(
            circuits, args.round, client_a, class_names[cls_a], client_b, class_names[cls_b]
        )
        total_shared = sum(len(v) for v in shared_indices.values())
        print(f"Total Shared Neurons: {total_shared}")
        
        # 3. Baseline Eval
        accs_base = evaluate_per_class(base_model, testloader, config, [cls_a, cls_b])
        print(f"Baseline FedAvg -> C{cls_a}: {accs_base[cls_a]:.2f}% | C{cls_b}: {accs_base[cls_b]:.2f}%")
        
        # 4. Inject A
        model_a = inject_shared_weights(base_model, client_models[client_a], shared_indices, config.device)
        accs_a = evaluate_per_class(model_a, testloader, config, [cls_a, cls_b])
        
        # 5. Inject B
        model_b = inject_shared_weights(base_model, client_models[client_b], shared_indices, config.device)
        accs_b = evaluate_per_class(model_b, testloader, config, [cls_a, cls_b])
        
        # Summary Table
        print("-" * 50)
        print(f"{'Injection Source':<20} | {'Class ' + str(cls_a):<10} | {'Class ' + str(cls_b):<10}")
        print("-" * 50)
        print(f"{'None (Baseline)':<20} | {accs_base[cls_a]:<10.2f} | {accs_base[cls_b]:<10.2f}")
        print(f"{'Client ' + str(client_a) + ' (' + str(cls_a) + ')':<20} | {accs_a[cls_a]:<10.2f} | {accs_a[cls_b]:<10.2f}")
        print(f"{'Client ' + str(client_b) + ' (' + str(cls_b) + ')':<20} | {accs_b[cls_a]:<10.2f} | {accs_b[cls_b]:<10.2f}")

if __name__ == "__main__":
    main()