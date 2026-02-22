import copy
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from circuits.hooks import get_hard_mask_hook, get_inverse_mask_hook


def apply_circuit(model, circuit: dict, cfg, remove_after=False):
    device = cfg.device
    hooks = []
    for layer_name, indices in circuit.items():
        try:
            module = model.get_submodule(layer_name)
        except AttributeError:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        hooks.append(module.register_forward_hook(get_hard_mask_hook(idx_tensor, device)))
    return hooks


def eval_full(model, testloader, cfg) -> dict:
    device = cfg.device
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    correct = 0
    total = 0
    class_correct = [0] * cfg.num_classes
    class_total   = [0] * cfg.num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_copy(inputs)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                if 0 <= lbl < cfg.num_classes:
                    class_correct[lbl] += (predicted[i] == labels[i]).item()
                    class_total[lbl]   += 1

    per_class = {}
    for i in range(cfg.num_classes):
        per_class[str(i)] = round(100 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else None

    return {"overall": round(100 * correct / total, 2), "per_class": per_class}


def eval_sufficiency(model, testloader, circuit: dict, target_class: int, cfg) -> float:
    device = cfg.device
    model_copy = copy.deepcopy(model)
    hooks = apply_circuit(model_copy, circuit, cfg)
    model_copy.eval()

    correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mask = labels == target_class
            if mask.sum() == 0:
                continue
            outputs = model_copy(inputs[mask])
            _, predicted = torch.max(outputs, 1)
            total   += mask.sum().item()
            correct += (predicted == labels[mask]).sum().item()

    for h in hooks:
        h.remove()
    return round(100 * correct / total, 2) if total > 0 else 0.0


def eval_necessity(model, testloader, circuit: dict, target_class: int, cfg) -> float:
    device = cfg.device
    model_copy = copy.deepcopy(model)
    hooks = []
    for layer_name, indices in circuit.items():
        try:
            module = model_copy.get_submodule(layer_name)
        except AttributeError:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        hooks.append(module.register_forward_hook(get_inverse_mask_hook(idx_tensor, device)))

    model_copy.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mask = labels == target_class
            if mask.sum() == 0:
                continue
            outputs = model_copy(inputs[mask])
            _, predicted = torch.max(outputs, 1)
            total   += mask.sum().item()
            correct += (predicted == labels[mask]).sum().item()

    for h in hooks:
        h.remove()
    return round(100 * correct / total, 2) if total > 0 else 0.0
