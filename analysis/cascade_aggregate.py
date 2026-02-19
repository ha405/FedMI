"""
Circuit Cascade Aggregation for Non-IID Federated Learning.

Instead of FedAvg (which blurs all weights), this script assembles
a global model by:
1. Conv Layers: Union of active channels, averaging overlaps.
2. FC Layer: Row-by-row assembly from class experts.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
from config import Config
from models import get_model
from dataset import get_dataset, get_test_dataloader

# --- Configuration ---
DEFAULT_EXPERT_MAP = {
    0: 0, 1: 0, 2: 0,  # Client 0 is expert for classes 0, 1, 2
    3: 1, 4: 1, 5: 1,  # Client 1 is expert for classes 3, 4, 5
    6: 2, 7: 2, 8: 2, 9: 2  # Client 2 is expert for classes 6, 7, 8, 9
}

def load_circuits(checkpoint_dir: str) -> dict:
    json_path = os.path.join(checkpoint_dir, "circuits_per_round_controlled_noniid.json")
    with open(json_path, 'r') as f:
        return json.load(f)

def load_client_models(checkpoint_dir: str, round_num: int, num_clients: int, config) -> list:
    client_models = []
    for i in range(num_clients):
        model = get_model(config)
        path = os.path.join(checkpoint_dir, f"round_{round_num}", f"client_{i}_model.pt")
        model.load_state_dict(torch.load(path, map_location=config.device))
        model.eval()
        client_models.append(model)
    return client_models

def get_conv3_circuit_indices(circuits: dict, round_num: int, client_idx: int, class_name: str) -> list:
    round_key = f"round_{round_num}"
    client_key = f"client_{client_idx}"
    
    try:
        class_data = circuits[round_key]["clients_local_model"][client_key][class_name]
        return class_data["active_nodes"].get("conv3", [])
    except KeyError:
        return []

def cascade_aggregate(
    global_model: nn.Module,
    client_models: list,
    circuits: dict,
    round_num: int,
    expert_map: dict,
    class_names: list,
    config
) -> nn.Module:
    """    
    Args:
        global_model: The target global model to populate.
        client_models: List of trained client models.
        circuits: The circuits dictionary from JSON.
        round_num: Which round's circuits to use.
        expert_map: Dict mapping class_idx -> client_idx (the expert).
        class_names: List of class name strings.
        config: Config object.

    Returns:
        The assembled global model.
    """
    device = config.device
    spatial_dim = global_model.spatial_dim 
    block_size = spatial_dim * spatial_dim
    num_classes = config.num_classes
    num_conv3_channels = config.conv_channels[2]
    print("\n[Cascade] Aggregating Convolutional Layers...")
    
    conv_layers = ['conv1', 'conv2', 'conv3']
    
    for layer_name in conv_layers:
        print(f"  Processing {layer_name}...")
        channel_usage = {}  
        for class_idx, client_idx in expert_map.items():
            class_name = class_names[class_idx]
            round_key = f"round_{round_num}"
            client_key = f"client_{client_idx}"
            
            try:
                active_nodes = circuits[round_key]["clients_local_model"][client_key][class_name]["active_nodes"]
                active_channels = active_nodes.get(layer_name, [])
                
                for ch in active_channels:
                    if ch not in channel_usage:
                        channel_usage[ch] = []
                    if client_idx not in channel_usage[ch]:
                        channel_usage[ch].append(client_idx)
            except KeyError:
                continue

        global_layer = global_model.get_submodule(layer_name)
        global_layer.weight.data.zero_()
        if global_layer.bias is not None:
            global_layer.bias.data.zero_()

        for ch, contributing_clients in channel_usage.items():
            weight_sum = torch.zeros_like(global_layer.weight[ch])
            bias_sum = 0.0
            
            for client_idx in contributing_clients:
                client_layer = client_models[client_idx].get_submodule(layer_name)
                weight_sum += client_layer.weight[ch].to(device)
                if client_layer.bias is not None:
                    bias_sum += client_layer.bias[ch].item()

            num_contributors = len(contributing_clients)
            global_layer.weight.data[ch] = weight_sum / num_contributors
            if global_layer.bias is not None:
                global_layer.bias.data[ch] = bias_sum / num_contributors
        
        print(f"    Active channels: {len(channel_usage)} / {global_layer.weight.shape[0]}")

    print("\n[Cascade] Assembling FC Layer (Row-by-Row from Experts)...")
    
    global_fc = global_model.fc
    global_fc.weight.data.zero_()
    global_fc.bias.data.zero_()
    
    for class_idx, client_idx in expert_map.items():
        class_name = class_names[class_idx]

        active_conv3 = get_conv3_circuit_indices(circuits, round_num, client_idx, class_name)
        
        if not active_conv3:
            print(f"  Class {class_idx} ({class_name}): No circuit found, skipping.")
            continue

        expert_fc = client_models[client_idx].fc

        global_fc.bias.data[class_idx] = expert_fc.bias[class_idx].to(device)

        for ch in active_conv3:
            start_col = ch * block_size
            end_col = (ch + 1) * block_size
            
            global_fc.weight.data[class_idx, start_col:end_col] = \
                expert_fc.weight[class_idx, start_col:end_col].to(device)
        
        print(f"  Class {class_idx} ({class_name}): Copied {len(active_conv3)} channel blocks from Client {client_idx}")

    print("\n[Cascade] Aggregation Complete!")
    return global_model

def evaluate(model, testloader, config):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Circuit Cascade Aggregation")
    parser.add_argument("--round", type=int, default=5, help="Round number to use")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/controlled_noniid")
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
    
    config = Config()
    config.checkpoint_dir = args.checkpoint_dir
    config.num_clients = args.num_clients
    
    print("=" * 60)
    print("CIRCUIT CASCADE AGGREGATION")
    print("=" * 60)
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Using Round: {args.round}")
    print(f"Num Clients: {args.num_clients}")
    print(f"Expert Map: {DEFAULT_EXPERT_MAP}")

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes

    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)

    global_model = get_model(config)

    global_model = cascade_aggregate(
        global_model, client_models, circuits, args.round,
        DEFAULT_EXPERT_MAP, class_names, config
    )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    accuracy = evaluate(global_model, testloader, config)
    print(f"Cascade-Aggregated Model Accuracy: {accuracy:.2f}%")

    fedavg_path = os.path.join(args.checkpoint_dir, f"checkpoint_round_{args.round}.pt")
    if os.path.exists(fedavg_path):
        fedavg_model = get_model(config)
        checkpoint = torch.load(fedavg_path, map_location=config.device)
        fedavg_model.load_state_dict(checkpoint['model_state_dict'])
        fedavg_acc = evaluate(fedavg_model, testloader, config)
        print(f"FedAvg Model Accuracy (Round {args.round}): {fedavg_acc:.2f}%")
        print(f"Difference: {accuracy - fedavg_acc:+.2f}%")

    save_path = os.path.join(args.checkpoint_dir, f"cascaded_model_round_{args.round}.pt")
    torch.save(global_model.state_dict(), save_path)
    print(f"\nSaved cascaded model to: {save_path}")

if __name__ == "__main__":
    main()
