"""
Pairwise Deep Representation Analysis.

Goal: Determine if "Feature Indistinguishability" (Centroid Collapse) occurs 
      in specific pairs, or if it is a global phenomenon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import copy
from config import Config
from models import get_model
from dataset import get_dataset, get_test_dataloader
import os

# --- Configuration ---
# Class -> Client Mapping
EXPERT_MAP = {
    1: 0, 
    5: 1, 
    9: 2
}

def load_client_models(checkpoint_dir, round_num, num_clients, config):
    client_models = []
    for i in range(num_clients):
        model = get_model(config)
        path = os.path.join(checkpoint_dir, f"round_{round_num}", f"client_{i}_model.pt")
        model.load_state_dict(torch.load(path, map_location=config.device))
        model.eval()
        client_models.append(model)
    return client_models

def create_pairwise_fedavg(model_a, model_b, config):
    """Creates a FedAvg model from exactly two clients."""
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

def analyze_centroid_collapse(model, testloader, config, class_a, class_b):
    device = config.device
    model.eval()
    
    features = {class_a: [], class_b: []}
    
    # Collect latent vectors
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            
            # Forward to get features (before FC)
            # Assuming SimpleCNN structure
            x = model.pool(F.relu(model.conv1(inputs)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            feats = x.view(x.size(0), -1)
            
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl in [class_a, class_b]:
                    features[lbl].append(feats[i].cpu())
            
            if len(features[class_a]) > 50 and len(features[class_b]) > 50: break

    # Compute Centroids
    if len(features[class_a]) == 0 or len(features[class_b]) == 0:
        print(f"  [WARN] Not enough samples for comparison.")
        return

    cent_a = torch.stack(features[class_a]).mean(dim=0)
    cent_b = torch.stack(features[class_b]).mean(dim=0)
    
    # Normalize for cosine calc
    cent_a_norm = F.normalize(cent_a.unsqueeze(0), p=2)
    cent_b_norm = F.normalize(cent_b.unsqueeze(0), p=2)
    
    dist_cos = 1.0 - F.cosine_similarity(cent_a_norm, cent_b_norm).item()
    
    # Check Magnitudes
    mag_a = torch.norm(cent_a).item()
    mag_b = torch.norm(cent_b).item()
    
    print(f"  Cosine Distance: {dist_cos:.6f} (0=Synonyms, 1=Distinct)")
    print(f"  Magnitude Ratio: {mag_a:.2f} vs {mag_b:.2f}")
    
    if dist_cos < 0.01:
        print(f"  -> [DIAGNOSIS] INDISTINGUISHABLE FEATURES (Synonyms).")
        if abs(mag_a - mag_b) > 5.0:
            winner = class_a if mag_a > mag_b else class_b
            print(f"  -> [PREDICTION] Class {winner} will win due to Volume.")
    else:
        print(f"  -> [DIAGNOSIS] Features are distinct.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/controlled_noniid")
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()
    
    config = Config()
    config.checkpoint_dir = args.checkpoint_dir
    config.num_clients = args.num_clients
    
    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)

    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)
    
    # Pairs to test
    pairs = [
        (1, 5), # Client 0 vs 1
        (1, 9), # Client 0 vs 2
        (5, 9)  # Client 1 vs 2
    ]
    
    print("="*60)
    print("PAIRWISE FEATURE COLLAPSE ANALYSIS")
    print("="*60)

    for cls_a, cls_b in pairs:
        print(f"\nAnalyzing Pair: Class {cls_a} vs Class {cls_b}")
        print("-" * 40)
        
        client_a_idx = EXPERT_MAP[cls_a]
        client_b_idx = EXPERT_MAP[cls_b]
        
        # Create 2-Client Average
        pairwise_model = create_pairwise_fedavg(
            client_models[client_a_idx], 
            client_models[client_b_idx], 
            config
        )
        
        analyze_centroid_collapse(pairwise_model, testloader, config, cls_a, cls_b)

if __name__ == "__main__":
    main()