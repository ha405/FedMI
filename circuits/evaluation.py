import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .hooks import get_hard_mask_hook, get_inverse_mask_hook, get_mean_ablation_hook

def evaluate_circuit(model, testloader, circuit, target_class, config, layer_means=None, log_file=None, class_names=None):
    """
    Sufficiency Test: Can the circuit perform the task alone?
    If log_file provided: Calculates the AVERAGE Top-5 Probability distribution across all failures.
    """
    device = config.device
    hooks = []
    
    # 1. Register Hooks
    for layer_name, indices in circuit.items():
        module = model.get_submodule(layer_name)
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        
        if config.use_mean_ablation and layer_means is not None and layer_name in layer_means:
            mean_tensor = layer_means[layer_name]
            hooks.append(module.register_forward_hook(
                get_mean_ablation_hook(idx_tensor, mean_tensor, device)
            ))
        else:
            hooks.append(module.register_forward_hook(
                get_hard_mask_hook(idx_tensor, device)
            ))
    
    model.eval()
    correct, total = 0, 0
    
    # Trackers for "Average Top 5" calculation
    # Sum of probabilities for ALL failed images
    failure_prob_sum = torch.zeros(config.num_classes, device=device)
    failure_count = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Filter for target class
            mask = (labels == target_class)
            if mask.sum() == 0: continue
            
            masked_inputs = inputs[mask]
            masked_labels = labels[mask]
            
            outputs = model(masked_inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += mask.sum().item()
            correct += (predicted == masked_labels).sum().item()
            
            # --- Logic for Average Failure Analysis ---
            if log_file:
                # Find indices where prediction was WRONG
                wrong_mask = (predicted != masked_labels)
                if wrong_mask.sum() > 0:
                    # Calculate Softmax probabilities for the whole batch
                    probs = F.softmax(outputs, dim=1)
                    
                    # Extract probabilities for only the WRONG images
                    wrong_probs = probs[wrong_mask]
                    
                    # Sum them up (dim=0 is the batch dimension)
                    failure_prob_sum += wrong_probs.sum(dim=0)
                    failure_count += wrong_mask.sum().item()
    
    for h in hooks: h.remove()
    
    # Write summary log
    if log_file and failure_count > 0:
        # Calculate Average Probability Distribution of Failures
        avg_failure_probs = failure_prob_sum / failure_count
        
        # Get Top 5 from the Average
        top5_vals, top5_inds = torch.topk(avg_failure_probs, k=min(5, config.num_classes))
        
        error_msg_parts = []
        for i in range(len(top5_inds)):
            idx = top5_inds[i].item()
            prob = top5_vals[i].item()
            if class_names and 0 <= idx < len(class_names):
                name = class_names[idx]
            else:
                name = str(idx)
            error_msg_parts.append(f"{name} ({prob*100:.2f}%)")
            
        error_msg = ", ".join(error_msg_parts)
        
        acc = 100 * correct / total
        if class_names and 0 <= target_class < len(class_names):
            t_name = class_names[target_class]
        else:
            t_name = str(target_class)
        
        log_file.write(f"  [Cross-Eval] Target: {t_name} | Acc: {acc:.2f}% | Failures: {failure_count}/{total}\n")
        log_file.write(f"  Avg Top-5 Confusion: {error_msg}\n")

    return (100 * correct / total) if total > 0 else 0.0

def evaluate_circuit_necessity(model, testloader, circuit, target_class, config):
    """Necessity Test (Inverse Pruning)"""
    device = config.device
    hooks = []
    
    # Need to be careful: If circuit indices are empty, inverse mask should be all 1s (No pruning).
    # The hook already handles empty indices correctly.
    
    for layer_name, indices in circuit.items():
        module = model.get_submodule(layer_name)
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        hooks.append(module.register_forward_hook(get_inverse_mask_hook(idx_tensor, device)))
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mask = (labels == target_class)
            if mask.sum() == 0: continue
            outputs = model(inputs[mask])
            _, predicted = torch.max(outputs.data, 1)
            total += mask.sum().item()
            correct += (predicted == labels[mask]).sum().item()
    
    for h in hooks: h.remove()
    return (100 * correct / total) if total > 0 else 0.0

def evaluate_detailed(model, testloader, config, log_file=None, class_names=None, title="Model Evaluation"):
    """
    Evaluates model on the full test set.
    Logs ONLY the per-class accuracy summary table to log_file.
    """
    model.eval()
    
    correct = 0
    total = 0
    
    num_classes = config.num_classes
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    if log_file:
        log_file.write(f"\n--- {title} ---\n")
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class stats
            c = (predicted == labels)
            for i in range(labels.size(0)):
                label = labels[i].item()
                if 0 <= label < num_classes:
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    # Log Per-Class Accuracy Table
    if log_file:
        log_file.write(f"Overall Accuracy: {100 * correct / total:.2f}%\n")
        log_file.write("Per-Class Accuracy:\n")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                if class_names and 0 <= i < len(class_names):
                    c_name = class_names[i]
                else:
                    c_name = str(i)
                log_file.write(f"  Class {c_name}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
            else:
                log_file.write(f"  Class {i}: N/A (No samples)\n")
        log_file.write("-" * 30 + "\n")
        log_file.flush()

    return 100 * correct / total

def extract_sparse_connectivity(model):
    connectivity = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu()
            # Sum over spatial dims (2,3) to get [Out, In] magnitude
            w_spatial_sum = w.abs().sum(dim=(2, 3))
            # Returns indices [[row, col], ...] -> [[dest, src], ...]
            connected_indices = torch.nonzero(w_spatial_sum > 0, as_tuple=False)
            
            connectivity[name] = connected_indices.tolist()
        elif isinstance(module, nn.Linear):
            w = module.weight.detach().cpu()
            connected_indices = torch.nonzero(w.abs() > 0, as_tuple=False)
            connectivity[name] = connected_indices.tolist()        
    return connectivity

def filter_connectivity_by_circuit(physical_connectivity, active_circuit):
    functional_connectivity = {}
    layer_order = list(physical_connectivity.keys())
    
    for i, layer_name in enumerate(layer_order):
        edges = physical_connectivity.get(layer_name, [])
        if not edges: continue
            
        active_dest_nodes = set(active_circuit.get(layer_name, []))
        
        if i == 0:
            valid_edges = [
                (dest, src) for dest, src in edges 
                if dest in active_dest_nodes
            ]
        else:
            prev_layer_name = layer_order[i-1]
            active_src_nodes = set(active_circuit.get(prev_layer_name, []))
            
            valid_edges = [
                (dest, src) for dest, src in edges 
                if dest in active_dest_nodes and src in active_src_nodes
            ]
            
        layer_dict = {}
        for dest, src in valid_edges:
            if dest not in layer_dict:
                layer_dict[dest] = []
            layer_dict[dest].append(src)
            
        functional_connectivity[layer_name] = layer_dict
        
    return functional_connectivity

def analyze_iou(circuit_storage):
    analyzed_names = list(circuit_storage.keys())
    if len(analyzed_names) < 2: return
    layers = list(circuit_storage[analyzed_names[0]].keys())
    for layer in layers:
        print(f"\nLayer: {layer}")
        for c1, c2 in itertools.combinations(analyzed_names, 2):
            set1, set2 = set(circuit_storage[c1][layer]), set(circuit_storage[c2][layer])
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            iou = intersection / union if union > 0 else 0
            print(f"  IoU ({c1} vs {c2}): {iou:.4f}")
