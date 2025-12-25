import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from tqdm import tqdm
import os
from train import (
    get_gate_hook,
    apply_weight_sparsity, 
    get_current_sparsity,
    compute_layer_means,
    get_gate_mean_hook,
    extract_sparse_connectivity,
    filter_connectivity_by_circuit
)
from utils import (
    evaluate_circuit,
    evaluate_circuit_necessity,
    save_local_model,
    save_checkpoint,
    load_latest_checkpoint,
    save_circuits_to_json,
    evaluate_detailed,
    get_client_class_counts
)

def local_train(model, dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    use_rs = getattr(config, 'use_fedrs', False)
    cdist = None
    
    if use_rs:
        cnts = get_client_class_counts(dataloader, config.num_classes)
        if cnts.sum() > 0:
            dist = cnts / cnts.sum()
        else:
            dist = torch.ones(config.num_classes) / config.num_classes
        if dist.max() > 0:
            cdist = dist / dist.max()
        else:
            cdist = torch.ones(config.num_classes)
        alpha = getattr(config, 'fedrs_alpha', 0.5)
        cdist = cdist * (1.0 - alpha) + alpha
        # Move to device and reshape for broadcasting [1, num_classes]
        cdist = cdist.to(config.device).view(1, -1)
    total_steps = len(dataloader) * config.local_epochs
    current_step = 0
    
    for epoch in range(config.local_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.local_epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if use_rs and cdist is not None:
                # Element-wise multiplication of logits by the Restricted Softmax weights
                outputs = outputs * cdist
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if getattr(config, 'train_mode', 'dense') == 'sparse':
                sparsity_to_apply = get_current_sparsity(
                    current_step=current_step, total_steps=total_steps,
                    final_sparsity=config.target_sparsity
                )
                apply_weight_sparsity(model, sparsity_to_apply)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", sp=f"{sparsity_to_apply:.2f}")
            else:
                 progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            current_step += 1
    if getattr(config, 'train_mode', 'dense') == 'sparse':
        apply_weight_sparsity(model, config.target_sparsity)      
    return model

def fedavg(global_model, client_models):
    weights = [1.0 / len(client_models)] * len(client_models)
    global_state = global_model.state_dict()
    target_device = next(global_model.parameters()).device
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32).to(target_device)
        for i, client_model in enumerate(client_models):
            global_state[key] += weights[i] * client_model.state_dict()[key].to(target_device).float()
    global_model.load_state_dict(global_state)

def discover_client_circuit(model, dataloader, target_class, config, layer_means=None):
    device = config.device
    
    has_target_class = any(target_class in labels for _, labels in dataloader)
    layers_to_discover = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]

    if not has_target_class:
        return {name: [] for name in layers_to_discover}

    criterion = nn.CrossEntropyLoss()
    original_grads = {name: param.requires_grad for name, param in model.named_parameters()}
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    
    gate_params, hooks, layers = {}, [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append(name)
            gate_params[name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
            
            if config.use_mean_ablation and layer_means and name in layer_means:
                hooks.append(module.register_forward_hook(get_gate_mean_hook(gate_params[name], layer_means[name])))
            else:
                hooks.append(module.register_forward_hook(get_gate_hook(gate_params[name])))

    optimizer = optim.Adam(gate_params.values(), lr=config.gate_lr)
    
    for _ in range(config.discovery_steps):
        data_iter = iter(dataloader)
        found_batch = False
        for inputs, labels in data_iter:
            if target_class in labels:
                inputs, labels = inputs.to(device), labels.to(device)
                found_batch = True
                break
        if not found_batch: continue 

        mask = (labels == target_class)
        if mask.sum() == 0: continue
        
        optimizer.zero_grad()
        l0_loss = sum(torch.sigmoid(p).sum() for p in gate_params.values())
        loss = criterion(model(inputs[mask]), labels[mask]) + (config.l0_lambda * l0_loss)
        loss.backward()
        optimizer.step()
        
    circuit = {name: np.where((gate_params[name] > 0).float().cpu().numpy().flatten() == 1)[0].tolist() for name in layers}
    for h in hooks: h.remove()
    for name, param in model.named_parameters(): param.requires_grad = original_grads.get(name, True)
    return circuit

def run_federated_round(global_model, client_dataloaders, testloader, config, round_num, class_names, all_circuits, classes_to_discover_per_client=None, log_file=None):
    client_models = []
    
    round_circuits = {
        "clients_local_model": {}, 
        "clients_global_model": {}
    }
    
    print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")
    
    # --- 1. LOCAL TRAINING & DISCOVERY ---
    for i, dl in enumerate(client_dataloaders):
        print(f"  Client {i}: Local Training...")
        c_model = local_train(copy.deepcopy(global_model), dl, config)
        save_local_model(c_model, round_num, i, config)
        
        physical_connectivity = extract_sparse_connectivity(c_model)
        client_models.append(c_model)
        client_means = compute_layer_means(c_model, dl, config) if config.use_mean_ablation else None

        c_circs = {}
        classes_for_this_client = classes_to_discover_per_client.get(i) if classes_to_discover_per_client else config.classes_to_analyze
        
        for tc in classes_for_this_client:
            name = class_names[tc]
            circ = discover_client_circuit(c_model, dl, tc, config, layer_means=client_means)
            func_conn = filter_connectivity_by_circuit(physical_connectivity, circ)
            
            c_circs[name] = {
                "active_nodes": circ,
                "connectivity": func_conn,
                "metrics": {}
            }
            
            # Local evaluation
            acc = evaluate_circuit(c_model, testloader, circ, tc, config, layer_means=client_means)
            inv_acc = evaluate_circuit_necessity(c_model, testloader, circ, tc, config)
            
            c_circs[name]["metrics"] = {"accuracy": acc, "necessity": inv_acc}
            
        round_circuits["clients_local_model"][f"client_{i}"] = c_circs
        
    # --- 2. AGGREGATION ---
    fedavg(global_model, client_models)
    
    # --- 3. GLOBAL DISCOVERY & CROSS-EVALUATION ---
    print(f"  Global Model: Running per-client circuit discovery & evaluation...")
    for i, dl in enumerate(client_dataloaders):
        gm_copy = copy.deepcopy(global_model)
        global_phys_conn = extract_sparse_connectivity(gm_copy)
        global_means = compute_layer_means(gm_copy, dl, config) if config.use_mean_ablation else None

        cg_circs = {}
        classes_for_this_client = classes_to_discover_per_client.get(i) if classes_to_discover_per_client else config.classes_to_analyze

        for tc in classes_for_this_client:
            name = class_names[tc]
            
            # A. Discovery on Global Model
            circ_global = discover_client_circuit(gm_copy, dl, tc, config, layer_means=global_means)
            func_conn = filter_connectivity_by_circuit(global_phys_conn, circ_global)
            
            cg_circs[name] = {
                "active_nodes": circ_global,
                "connectivity": func_conn,
                "metrics": {}
            }
            
            # B. Evaluate Global Circuit
            acc_global = evaluate_circuit(gm_copy, testloader, circ_global, tc, config, layer_means=global_means)
            inv_acc = evaluate_circuit_necessity(gm_copy, testloader, circ_global, tc, config)
            
            cg_circs[name]["metrics"]["accuracy"] = acc_global
            cg_circs[name]["metrics"]["necessity"] = inv_acc

            # --- C. Evaluate LOCAL Circuit on GLOBAL Weights (Cross-Evaluation) ---
            acc_cross = 0.0
            try:
                local_circ_nodes = round_circuits["clients_local_model"][f"client_{i}"][name]["active_nodes"]
                
                # Write header for this specific check if logging
                if log_file:
                    log_file.write(f"\n[Round {round_num + 1} | Client {i} | Class {name}] Cross-Eval:\n")
                
                acc_cross = evaluate_circuit(
                    gm_copy, testloader, local_circ_nodes, tc, config, 
                    layer_means=global_means, 
                    log_file=log_file, 
                    class_names=class_names
                )
            except KeyError:
                acc_cross = 0.0 
            
            cg_circs[name]["metrics"]["local_mask_on_global_weights_acc"] = acc_cross

            counts = {layer: len(idx) for layer, idx in circ_global.items()}
            print(f"    Global + Client {i} Data - {name}: {counts}")
            print(f"      > Global Mask Acc: {acc_global:.2f}% | Local Mask on Global Weights Acc: {acc_cross:.2f}%")
            
        round_circuits["clients_global_model"][f"client_{i}"] = cg_circs
        
    all_circuits[f"round_{round_num + 1}"] = round_circuits
    return global_model

def run_federated_training(global_model, client_dataloaders, testloader, config, class_names, classes_to_discover_per_client=None):
    print("\n=== Federated Training with Circuit Discovery ===")
    mode = getattr(config, 'train_mode', 'sparse')
    ablation_type = "MEAN" if config.use_mean_ablation else "ZERO"
    print(f"Training Mode: {mode.upper()} | Ablation: {ablation_type}")
    
    # --- SETUP LOGGING ---
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(config.checkpoint_dir, "training_log.txt")
    print(f"Logging detailed analysis to: {log_path}")
    with open(log_path, 'w') as f:
        f.write("=== Federated Training Log ===\n")

    all_circuits = {}
    start_round, loaded_circuits = load_latest_checkpoint(global_model, config) if config.resume else (0, {})
    
    if start_round >= config.num_rounds:
        print("Training already completed!")
        return global_model, loaded_circuits

    for round_num in range(start_round, config.num_rounds):
        with open(log_path, 'a') as log_f:
            global_model = run_federated_round(
                global_model, client_dataloaders, testloader, config, round_num, class_names, 
                all_circuits, classes_to_discover_per_client, 
                log_file=log_f
            )
            
            acc = evaluate_detailed(
                global_model, testloader, config, 
                log_file=log_f, 
                class_names=class_names, 
                title=f"Round {round_num + 1} Global Full Model Evaluation"
            )
            print(f"  Round {round_num + 1} Global Full Model Acc: {acc:.2f}%")
        
        save_checkpoint(global_model, round_num + 1, all_circuits, config)
        
        if config.partition_method == "dirichlet":
            filename_suffix = f"alpha_{config.dirichlet_alpha}"
        elif config.partition_method == "by_class":
            filename_suffix = "controlled_noniid"
        else:
            filename_suffix = "iid"
        json_path = os.path.join(config.checkpoint_dir, f"circuits_per_round_{filename_suffix}.json")
        save_circuits_to_json(all_circuits, json_path)
    
    for param in global_model.parameters():
        param.requires_grad = False
    return global_model, all_circuits