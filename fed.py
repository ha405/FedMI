import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import os
import numpy as np
from typing import List, Dict, Set
from torch.utils.data import DataLoader
from train import binary_gate, get_gate_hook


def apply_weight_sparsity(model, sparsity_level=0.90):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                flat_param = param.abs().flatten()
                k = int((1 - sparsity_level) * flat_param.numel())
                if k < 1:
                    continue
                threshold = torch.topk(flat_param, k).values[-1]
                mask = (param.abs() >= threshold).float()
                param.data.mul_(mask)


def local_train(model, dataloader, config, epochs=None):
    """Per-client training. Supports 'sparse' or 'dense' modes via config.train_mode."""
    if epochs is None:
        epochs = config.local_epochs

    # Default to sparse if not specified, or checks explicitly
    train_mode = getattr(config, 'train_mode', 'sparse') 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Apply sparsity only if mode is sparse
            if train_mode == 'sparse':
                apply_weight_sparsity(model, config.target_sparsity)

    return model


def fedavg(global_model, client_models, weights=None):
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)

    global_state = global_model.state_dict()
    target_device = next(global_model.parameters()).device

    for key in global_state.keys():
        # Ensure we accumulate on the correct device
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32).to(target_device)
        for i, client_model in enumerate(client_models):
            global_state[key] += weights[i] * client_model.state_dict()[key].to(target_device).float()

    global_model.load_state_dict(global_state)


def evaluate(model, testloader, config):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def discover_client_circuit(model, dataloader, target_class, config):
    """Discover circuit for a specific class on a client model with robust search and safety."""
    device = config.device
    criterion = nn.CrossEntropyLoss()

    original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}

    # Freeze model parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    gate_params = {}
    hooks = []
    layers_to_prune = []

    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layers_to_prune.append(name)
                # Initialize gates: 2.0 (sigmoid(2.0) ~= 0.88, start open)
                gate_params[name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
                hooks.append(module.register_forward_hook(get_gate_hook(gate_params[name])))

        gate_optimizer = optim.Adam(gate_params.values(), lr=config.gate_lr)
        # Scheduler for better convergence (Cosine Annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(gate_optimizer, T_max=config.discovery_steps)

        data_iter = iter(dataloader)

        for step in range(config.discovery_steps):
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, labels = next(data_iter)

            inputs, labels = inputs.to(device), labels.to(device)
            target_mask = (labels == target_class)
            
            # Skip batch if no target class samples presence
            if target_mask.sum() == 0:
                # Stepping scheduler anyway to keep sync? Probably fine to skip or not. 
                # Let's skip optimization step but not scheduler step to avoid error if it expects step?
                # Actually, standard PyTorch scheduler steps are decoupled. Let's just continue.
                continue

            inputs = inputs[target_mask]
            labels = labels[target_mask]

            gate_optimizer.zero_grad()
            outputs = model(inputs)
            task_loss = criterion(outputs, labels)

            l0_loss = 0
            for name in gate_params:
                l0_loss += torch.sigmoid(gate_params[name]).sum()

            total_loss = task_loss + (config.l0_lambda * l0_loss)
            total_loss.backward()
            gate_optimizer.step()
            scheduler.step()

        # Extract circuit
        circuit = {}
        for name in layers_to_prune:
            # Use threshold > 0 (sigmoid > 0.5)
            mask = (gate_params[name] > 0).float().detach().cpu().numpy().flatten()
            indices = np.where(mask == 1)[0]
            circuit[name] = indices.tolist()

    finally:
        # Security/Safety: Ensure hooks are ALWAYS removed
        for h in hooks:
            h.remove()
        
        # Restore gradients
        for name, param in model.named_parameters():
            param.requires_grad = original_grad_states.get(name, True)

    return circuit


def save_circuits_to_json(all_circuits, path):
    """Save all circuits to JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, 'w') as f:
        json.dump(all_circuits, f, indent=2)
    print(f"Circuits saved to {path}")


def run_federated_round(global_model, client_dataloaders, config, round_num, class_names, all_circuits):
    """Run one FL round with per-client circuit discovery."""
    client_models = []
    round_circuits = {"clients_local_model": {}, "clients_global_model": {}}

    print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")

    # 1. Local Training & Discovery on Local Models
    for client_id, dataloader in enumerate(client_dataloaders):
        print(f"  Client {client_id}: Local Training...")
        client_model = copy.deepcopy(global_model)
        client_model = local_train(client_model, dataloader, config)
        client_models.append(client_model)

        # Discover circuits on the TRAINED LOCAL MODEL
        client_circuits = {}
        for target_class in config.classes_to_analyze:
            class_name = class_names[target_class]
            circuit = discover_client_circuit(client_model, dataloader, target_class, config)
            client_circuits[class_name] = circuit
            # active_counts = {layer: len(indices) for layer, indices in circuit.items()}
            # print(f"    Local Model - Client {client_id} - {class_name}: {active_counts}")

        round_circuits["clients_local_model"][f"client_{client_id}"] = client_circuits

    # 2. Aggregation
    fedavg(global_model, client_models)

    # 3. Discovery on Global Model using Per-Client Data
    print(f"  Global Model: Running per-client circuit discovery...")
    for client_id, dataloader in enumerate(client_dataloaders):
        # Deepcopy to ensure safety during discovery (hooks overlap prevention)
        gm_copy = copy.deepcopy(global_model)
        
        client_global_circuits = {}
        for target_class in config.classes_to_analyze:
            class_name = class_names[target_class]
            circuit = discover_client_circuit(gm_copy, dataloader, target_class, config)
            client_global_circuits[class_name] = circuit
            active_counts = {layer: len(indices) for layer, indices in circuit.items()}
            print(f"    Global Model + Client {client_id} Data - {class_name}: {active_counts}")
            
        round_circuits["clients_global_model"][f"client_{client_id}"] = client_global_circuits
        del gm_copy

    all_circuits[f"round_{round_num + 1}"] = round_circuits

    return global_model


def run_federated_training(global_model, client_dataloaders, testloader, config, class_names):
    """Full FL training with per-round, per-client circuit discovery."""
    print("\n=== Federated Training (IID) with Circuit Discovery ===")
    
    # Check if we are doing sparse or dense training based on config
    mode = getattr(config, 'train_mode', 'sparse')
    print(f"Training Mode: {mode.upper()}")

    all_circuits = {}

    for round_num in range(config.num_rounds):
        global_model = run_federated_round(
            global_model, client_dataloaders, config, round_num, class_names, all_circuits
        )
        acc = evaluate(global_model, testloader, config)
        print(f"  Round {round_num + 1} Global Acc: {acc:.2f}%")

    # Save all circuits to JSON
    save_circuits_to_json(all_circuits, "circuits_per_round.json")

    # Freeze weights after FL training (optional, but good for safety)
    for param in global_model.parameters():
        param.requires_grad = False

    return global_model, all_circuits
