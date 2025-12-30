import torch
import torch.nn as nn
import json
import os
import argparse
import copy
from config import Config
from models import get_model
from dataset import get_dataset, get_test_dataloader

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

def get_circuit_weights(base_model, circuits, round_num, client_idx, class_name):
    round_key = f"round_{round_num}"
    try:
        active_nodes = circuits[round_key]["clients_local_model"][f"client_{client_idx}"][class_name]["active_nodes"]
    except KeyError:
        return base_model.state_dict()

    masked_weights = copy.deepcopy(base_model.state_dict())
    conv_layers = ['conv1', 'conv2', 'conv3']
    
    for layer_name in conv_layers:
        key_w = f"{layer_name}.weight"
        key_b = f"{layer_name}.bias"
        
        keep_indices = active_nodes.get(layer_name, [])
        out_channels = masked_weights[key_w].shape[0]
        mask = torch.zeros(out_channels, dtype=torch.bool)
        for idx in keep_indices:
            if idx < out_channels: mask[idx] = True
        
        masked_weights[key_w][~mask] = 0.0
        if key_b in masked_weights:
            masked_weights[key_b][~mask] = 0.0
            
    return masked_weights

def evaluate_fedavg(model, testloader, config, active_classes):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for c in active_classes: mask |= (labels == c)
            if mask.sum() == 0: continue
            inputs, labels = inputs[mask], labels[mask]
            
            outputs = model(inputs)
            
            logit_mask = torch.full_like(outputs, -1e9)
            logit_mask[:, active_classes] = 0.0
            outputs = outputs + logit_mask
            
            _, predicted = outputs.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total > 0 else 0.0

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
    print("CIRCUIT ENSEMBLE INFERENCE (MAX-LOGIT)")
    print("="*60)

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes

    EXPERT_MAP = {
        0: 0, 
        3: 1, 
        6: 2  
    }
    TARGET_CLASSES = [0, 3, 6]

    circuits = load_circuits(args.checkpoint_dir)
    client_models = load_client_models(args.checkpoint_dir, args.round, args.num_clients, config)
    
    print("Building Base FedAvg Model...")
    base_model = create_fedavg_model(client_models, config)
    base_model.eval()
    
    # 1. Evaluate FedAvg Baseline
    print("\nEvaluating Standard FedAvg...")
    fedavg_acc = evaluate_fedavg(base_model, testloader, config, TARGET_CLASSES)
    print(f"FedAvg Accuracy: {fedavg_acc:.2f}%")
    
    # 2. Ensemble Inference
    print("\nPre-calculating Circuit Masks...")
    circuit_weights = {}
    for cls_idx in TARGET_CLASSES:
        client_id = EXPERT_MAP[cls_idx]
        class_name = class_names[cls_idx]
        print(f"  Circuit {cls_idx} -> Client {client_id} ({class_name})")
        circuit_weights[cls_idx] = get_circuit_weights(base_model, circuits, args.round, client_id, class_name)

    print("Running Ensemble Inference...")
    correct = 0
    total = 0
    
    ensemble_models = {}
    for cls_idx in TARGET_CLASSES:
        m = get_model(config)
        m.load_state_dict(circuit_weights[cls_idx])
        m.eval()
        ensemble_models[cls_idx] = m

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for c in TARGET_CLASSES: mask |= (labels == c)
            if mask.sum() == 0: continue
            inputs, labels = inputs[mask], labels[mask]
            
            logits_list = []
            
            for cls_idx in TARGET_CLASSES:
                outputs = ensemble_models[cls_idx](inputs)
                logit_val = outputs[:, cls_idx]
                logits_list.append(logit_val.unsqueeze(1))
            
            ensemble_scores = torch.cat(logits_list, dim=1)
            pred_indices = ensemble_scores.argmax(dim=1)
            
            pred_classes = torch.tensor([TARGET_CLASSES[i] for i in pred_indices.cpu().numpy()]).to(config.device)
            
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)
            
    ensemble_acc = 100 * correct / total
    print(f"\nEnsemble Accuracy: {ensemble_acc:.2f}%")
    print(f"Improvement: {ensemble_acc - fedavg_acc:+.2f}%")

if __name__ == "__main__":
    main()