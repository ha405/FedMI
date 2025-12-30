import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
import copy
import collections
from config import Config
from models import get_model
from dataset import get_dataset, get_test_dataloader

EXPERT_MAP = { 1: 0, 5: 1, 9: 2 }

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

def analyze_predictions(model, testloader, config, target_classes):
    model.eval()
    
    correct_top1 = {c: 0 for c in target_classes}
    correct_top3 = {c: 0 for c in target_classes}
    confidence_sum = {c: 0.0 for c in target_classes}
    total_samples = {c: 0 for c in target_classes}
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1)
            
            _, top3_preds = outputs.topk(3, dim=1)
            top1_preds = top3_preds[:, 0]
            
            for c in target_classes:
                mask = (labels == c)
                if mask.sum() == 0: continue
                
                subset_labels = labels[mask]
                subset_top1 = top1_preds[mask]
                subset_top3 = top3_preds[mask]
                subset_probs = probs[mask, c] 
                
                total_samples[c] += mask.sum().item()
                
                correct_top1[c] += (subset_top1 == subset_labels).sum().item()
                
                hits_top3 = (subset_top3 == subset_labels.unsqueeze(1)).any(dim=1)
                correct_top3[c] += hits_top3.sum().item()
                
                confidence_sum[c] += subset_probs.sum().item()

    stats = {}
    for c in target_classes:
        n = total_samples[c]
        if n == 0: continue
        
        acc1 = 100 * correct_top1[c] / n
        acc3 = 100 * correct_top3[c] / n
        avg_conf = confidence_sum[c] / n
            
        stats[c] = {
            "top1": acc1,
            "top3": acc3,
            "avg_conf": avg_conf
        }
        
    return stats

def mask_dominator_unique_nodes(model, circuits, round_num, class_names, dominator_class):
    round_key = f"round_{round_num}"
    dom_client = EXPERT_MAP[dominator_class]
    dom_name = class_names[dominator_class]
    
    try:
        dom_nodes = circuits[round_key]["clients_local_model"][f"client_{dom_client}"][dom_name]["active_nodes"]
    except KeyError: return model

    victim_nodes_union = {'conv1': set(), 'conv2': set(), 'conv3': set()}
    for cls_idx, client_idx in EXPERT_MAP.items():
        if cls_idx == dominator_class: continue
        cls_name = class_names[cls_idx]
        try:
            c_nodes = circuits[round_key]["clients_local_model"][f"client_{client_idx}"][cls_name]["active_nodes"]
            for layer in ['conv1', 'conv2', 'conv3']:
                victim_nodes_union[layer].update(c_nodes.get(layer, []))
        except KeyError: continue

    print(f"\n--- Masking Unique Nodes of Dominator (Class {dominator_class}) ---")
    conv_layers = ['conv1', 'conv2', 'conv3']
    for layer_name in conv_layers:
        layer = model.get_submodule(layer_name)
        dom_indices = set(dom_nodes.get(layer_name, []))
        victim_indices = victim_nodes_union[layer_name]
        to_kill = dom_indices - victim_indices
        print(f"  [{layer_name}] Unique to Dominator (Killed): {len(to_kill)}")
        
        with torch.no_grad():
            for idx in to_kill:
                if idx < layer.out_channels:
                    layer.weight.data[idx, :, :, :] = 0.0
                    if layer.bias is not None:
                        layer.bias.data[idx] = 0.0
    return model

def print_robust_report(title, stats):
    print(f"\n{title}")
    print(f"{'Class':<6} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'Avg Softmax Prob (Confidence)'}")
    print("-" * 80)
    for cls, data in stats.items():
        print(f"{cls:<6} | {data['top1']:<10.2f} | {data['top3']:<10.2f} | {data['avg_conf']:.4f}")

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
    print("ROBUST DOMINATOR MASKING ANALYSIS")
    print("="*60)

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes

    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)

    print("Initializing Dense FedAvg Model...")
    base_model = create_fedavg_model(client_models, config)
    
    base_stats = analyze_predictions(base_model, testloader, config, list(EXPERT_MAP.keys()))
    print_robust_report("BASELINE (FEDAVG) PERFORMANCE", base_stats)
    
    best_acc = -1
    dominator_class = -1
    for cls, data in base_stats.items():
        if data['top1'] > best_acc:
            best_acc = data['top1']
            dominator_class = cls
    print(f"\n[Detected Dominator] Class {dominator_class}")
        
    masked_model = copy.deepcopy(base_model)
    masked_model = mask_dominator_unique_nodes(masked_model, circuits, args.round, class_names, dominator_class)
    
    new_stats = analyze_predictions(masked_model, testloader, config, list(EXPERT_MAP.keys()))
    print_robust_report("POST-MASKING PERFORMANCE", new_stats)
    
    print("\n" + "="*60)
    print("FINAL DELTA SUMMARY")
    print("="*60)
    print(f"{'Class':<6} | {'Top-1 Delta':<15} | {'Confidence Delta':<15}")
    print("-" * 60)
    for cls in EXPERT_MAP.keys():
        d1 = new_stats[cls]['top1'] - base_stats[cls]['top1']
        d_conf = new_stats[cls]['avg_conf'] - base_stats[cls]['avg_conf']
        print(f"{cls:<6} | {d1:<+15.2f} | {d_conf:<+15.4f}")

if __name__ == "__main__":
    main()