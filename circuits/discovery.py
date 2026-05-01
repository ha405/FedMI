import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from .hooks import get_gate_hook, get_multi_class_gate_hook

def is_valid_layer(name, module):
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return False
    # Exclude typical useless layers and classification heads
    exclude_terms = ['patch_embed', 'head', 'classifier', 'embed']
    if any(term in name for term in exclude_terms):
        return False
    if name == 'fc': # typical resnet head
        return False
    return True

def precollect_all_class_samples(dataloader, classes: List[int], max_per_class: int = 1024,
                                  device: str = 'cuda') -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Scan a dataloader ONCE and collect samples for ALL requested classes.
    """
    class_inputs = {c: [] for c in classes}
    class_labels = {c: [] for c in classes}
    counts = {c: 0 for c in classes}

    for b_inputs, b_labels in dataloader:
        for c in classes:
            if counts[c] >= max_per_class:
                continue
            mask = (b_labels == c)
            if mask.any():
                class_inputs[c].append(b_inputs[mask])
                class_labels[c].append(b_labels[mask])
                counts[c] += mask.sum().item()

        # Early exit if all classes are full
        if all(counts[c] >= max_per_class for c in classes):
            break

    result = {}
    for c in classes:
        if class_inputs[c]:
            result[c] = (
                torch.cat(class_inputs[c]).to(device),
                torch.cat(class_labels[c]).to(device),
            )
    return result


def discover_client_circuit_cached(model, class_inputs: torch.Tensor, class_labels: torch.Tensor,
                                    target_class: int, config):
    device = config.device

    if class_inputs is None or class_inputs.shape[0] == 0:
        return {name: [] for name, m in model.named_modules() if is_valid_layer(name, m)}

    all_inputs = class_inputs 
    all_labels = class_labels
    num_avail = all_inputs.shape[0]

    criterion = nn.CrossEntropyLoss()
    original_grads = {name: param.requires_grad for name, param in model.named_parameters()}
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    gate_params, hooks, layers = {}, [], []
    for name, module in model.named_modules():
        # Strip '_orig_mod.' prefix if model is compiled
        clean_name = name.replace("_orig_mod.", "")
        if is_valid_layer(clean_name, module):
            layers.append(clean_name)
            if isinstance(module, nn.Conv2d):
                gate_params[clean_name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
            else:
                gate_params[clean_name] = nn.Parameter(torch.ones(module.out_features).to(device) * 2.0)

            hooks.append(module.register_forward_hook(get_gate_hook(gate_params[clean_name])))

    optimizer = optim.Adam(gate_params.values(), lr=config.gate_lr)

    for _ in range(config.discovery_steps):
        if num_avail <= config.batch_size:
            idx = torch.arange(num_avail, device=device)
        else:
            idx = torch.randperm(num_avail, device=device)[:config.batch_size]

        inputs = all_inputs[idx]
        labels = all_labels[idx]

        optimizer.zero_grad()
        l0_loss = sum(torch.sigmoid(p).sum() for p in gate_params.values())
        logits = model(inputs)

        cls_loss = criterion(logits, labels)
        loss = cls_loss + (config.l0_lambda * l0_loss)

        loss.backward()
        optimizer.step()

    circuit = {name: np.where((gate_params[name] > 0).float().cpu().numpy().flatten() == 1)[0].tolist() for name in layers}

    for h in hooks:
        h.remove()
    for name, param in model.named_parameters():
        param.requires_grad = original_grads.get(name, True)

    return circuit


def discover_all_classes_cached(model, class_samples_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]], config):

    device = config.device
    num_classes = config.num_classes
    
    # 1. Pool all samples together
    all_inputs = []
    all_labels = []
    for c, (inputs, labels) in class_samples_dict.items():
        all_inputs.append(inputs)
        all_labels.append(labels)
        
    if not all_inputs:
        return {c: {n: [] for n, m in model.named_modules() if is_valid_layer(n, m)} for c in range(num_classes)}
        
    discovery_inputs = torch.cat(all_inputs)
    discovery_labels = torch.cat(all_labels)
    num_avail = discovery_inputs.shape[0]

    criterion = nn.CrossEntropyLoss()
    original_grads = {name: param.requires_grad for name, param in model.named_parameters()}
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    gate_params, hooks, layers = {}, [], []
    for name, module in model.named_modules():
        clean_name = name.replace("_orig_mod.", "")
        if is_valid_layer(clean_name, module):
            layers.append(clean_name)
            channels = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
            gate_params[clean_name] = nn.Parameter(torch.ones(num_classes, channels, device=device) * 2.0)
            
            hooks.append(module.register_forward_hook(get_multi_class_gate_hook(gate_params[clean_name])))

    optimizer = optim.Adam(gate_params.values(), lr=config.gate_lr)

    for _ in range(config.discovery_steps):
        if num_avail <= config.batch_size:
            idx = torch.arange(num_avail, device=device)
        else:
            idx = torch.randperm(num_avail, device=device)[:config.batch_size]

        inputs = discovery_inputs[idx]
        labels = discovery_labels[idx]

        for name, module in model.named_modules():
            clean_name = name.replace("_orig_mod.", "")
            if is_valid_layer(clean_name, module):
                module.current_labels = labels

        optimizer.zero_grad()
        
        unique_labels = labels.unique()
        l0_loss = sum(torch.sigmoid(p[unique_labels]).sum() for p in gate_params.values())
        
        logits = model(inputs)
        cls_loss = criterion(logits, labels)
        loss = cls_loss + (config.l0_lambda * l0_loss)

        loss.backward()
        optimizer.step()

    all_circuits = {}
    with torch.no_grad():
        for c in range(num_classes):
            circuit = {}
            for name in layers:
                class_gate_row = gate_params[name][c]
                indices = torch.where(class_gate_row > 0)[0].cpu().numpy().tolist()
                circuit[name] = indices
            all_circuits[c] = circuit

    # Cleanup
    for h in hooks:
        h.remove()
    for name, module in model.named_modules():
        if hasattr(module, 'current_labels'):
            del module.current_labels
    for name, param in model.named_parameters():
        param.requires_grad = original_grads.get(name, True)

    return all_circuits
