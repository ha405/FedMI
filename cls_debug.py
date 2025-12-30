"""
FC Weight Analysis on Shared Channels.

Goal: Identify if Class 9's FC weights are systematically larger 
      on the 'Shared' channels than Class 1's weights.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
import numpy as np
from config import Config
from models import get_model

# --- Configuration ---
CLASS_A = 1
CLASS_B = 9
CLIENT_A = 0
CLIENT_B = 2

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

def create_pairwise_fedavg(model_a, model_b, config):
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

def analyze_shared_weights(model, circuits, round_num, config):
    round_key = f"round_{round_num}"
    
    # 1. Identify Shared Conv3 Channels (The bottleneck features)
    try:
        nodes_a = circuits[round_key]["clients_local_model"][f"client_{CLIENT_A}"][f"{CLASS_A} - one"]["active_nodes"]["conv3"]
        nodes_b = circuits[round_key]["clients_local_model"][f"client_{CLIENT_B}"][f"{CLASS_B} - nine"]["active_nodes"]["conv3"]
    except KeyError:
        print("Error reading circuits.")
        return

    shared_channels = sorted(list(set(nodes_a).intersection(set(nodes_b))))
    unique_a = sorted(list(set(nodes_a) - set(nodes_b)))
    unique_b = sorted(list(set(nodes_b) - set(nodes_a)))
    
    print(f"\n" + "="*60)
    print(f"FC WEIGHT ANALYSIS (Shared Channels in Conv3)")
    print("="*60)
    print(f"Shared Channels: {shared_channels}")
    print(f"Unique to Class {CLASS_A}: {unique_a}")
    print(f"Unique to Class {CLASS_B}: {unique_b}")
    
    # 2. Get FC Weights
    # FC Layer shape: [10, 512]. 
    # Each Conv3 channel corresponds to 'block_size' columns in FC.
    spatial_dim = model.spatial_dim 
    block_size = spatial_dim * spatial_dim
    
    W_a = model.fc.weight.data[CLASS_A] # [512]
    W_b = model.fc.weight.data[CLASS_B] # [512]
    
    print(f"\n{'Channel Type':<15} | {'Channel Idx':<12} | {'Avg W_1':<10} | {'Avg W_9':<10} | {'Winner'}")
    print("-" * 65)
    
    def get_avg_weight(channel_idx, weight_row):
        start = channel_idx * block_size
        end = (channel_idx + 1) * block_size
        # Get mean absolute weight for this channel's block
        # Using ABS because sign might flip, but magnitude indicates importance
        # Actually, let's use signed mean to see directionality
        return weight_row[start:end].mean().item()

    # Check Shared
    for ch in shared_channels:
        wa = get_avg_weight(ch, W_a)
        wb = get_avg_weight(ch, W_b)
        win = f"C{CLASS_B}" if abs(wb) > abs(wa) else f"C{CLASS_A}"
        print(f"{'SHARED':<15} | {ch:<12} | {wa:<10.4f} | {wb:<10.4f} | {win}")
        
    print("-" * 65)
        
    # Check Unique A
    for ch in unique_a:
        wa = get_avg_weight(ch, W_a)
        wb = get_avg_weight(ch, W_b)
        win = f"C{CLASS_B}" if abs(wb) > abs(wa) else f"C{CLASS_A}"
        print(f"{'UNIQUE ' + str(CLASS_A):<15} | {ch:<12} | {wa:<10.4f} | {wb:<10.4f} | {win}")

    print("-" * 65)

    # Check Unique B
    for ch in unique_b:
        wa = get_avg_weight(ch, W_a)
        wb = get_avg_weight(ch, W_b)
        win = f"C{CLASS_B}" if abs(wb) > abs(wa) else f"C{CLASS_A}"
        print(f"{'UNIQUE ' + str(CLASS_B):<15} | {ch:<12} | {wa:<10.4f} | {wb:<10.4f} | {win}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/controlled_noniid")
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
    
    config = Config()
    config.checkpoint_dir = args.checkpoint_dir
    config.num_clients = args.num_clients
    
    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)
    
    print(f"Building Pairwise FedAvg (Client {CLIENT_A} + Client {CLIENT_B})...")
    model = create_pairwise_fedavg(client_models[CLIENT_A], client_models[CLIENT_B], config)
    
    analyze_shared_weights(model, circuits, args.round, config)

if __name__ == "__main__":
    main()