import torch
import torch.nn.functional as F
import os
import glob
import json
from train import get_hard_mask_hook, get_mean_ablation_hook, get_inverse_mask_hook

# --- Evaluation Functions ---

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
            name = class_names[idx] if class_names else str(idx)
            error_msg_parts.append(f"{name} ({prob*100:.2f}%)")
            
        error_msg = ", ".join(error_msg_parts)
        
        acc = 100 * correct / total
        t_name = class_names[target_class] if class_names else target_class
        
        log_file.write(f"  [Cross-Eval] Target: {t_name} | Acc: {acc:.2f}% | Failures: {failure_count}/{total}\n")
        log_file.write(f"  Avg Top-5 Confusion: {error_msg}\n")

    return (100 * correct / total) if total > 0 else 0.0

def evaluate_circuit_necessity(model, testloader, circuit, target_class, config):
    """Necessity Test (Inverse Pruning)"""
    device = config.device
    hooks = []
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
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Log Per-Class Accuracy Table
    if log_file:
        log_file.write(f"Overall Accuracy: {100 * correct / total:.2f}%\n")
        log_file.write("Per-Class Accuracy:\n")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                c_name = class_names[i] if class_names else str(i)
                log_file.write(f"  Class {c_name}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})\n")
            else:
                log_file.write(f"  Class {i}: N/A (No samples)\n")
        log_file.write("-" * 30 + "\n")
        log_file.flush()

    return 100 * correct / total

# --- I/O and Checkpointing Functions ---

def save_local_model(model, round_num, client_idx, config):
    round_dir = os.path.join(config.checkpoint_dir, f"round_{round_num + 1}")
    os.makedirs(round_dir, exist_ok=True)
    path = os.path.join(round_dir, f"client_{client_idx}_model.pt")
    torch.save(model.state_dict(), path)

def save_checkpoint(global_model, round_num, all_circuits, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_round_{round_num}.pt")
    state = {
        'round': round_num,
        'model_state_dict': global_model.state_dict(),
        'all_circuits': all_circuits,
        'config': config 
    }
    torch.save(state, path)
    print(f"  [Checkpoint] Saved global state to {path}")

def load_latest_checkpoint(global_model, config):
    if not os.path.exists(config.checkpoint_dir):
        return 0, {}
    files = glob.glob(os.path.join(config.checkpoint_dir, "checkpoint_round_*.pt"))
    if not files:
        return 0, {}
    latest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(f"  [Resume] Loading checkpoint: {latest_file}")
    checkpoint = torch.load(latest_file, map_location=config.device)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    start_round = checkpoint['round']
    all_circuits = checkpoint['all_circuits']
    print(f"  [Resume] Resuming from Round {start_round + 1}")
    return start_round, all_circuits

def save_circuits_to_json(all_circuits, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(all_circuits, f, indent=2)
    print(f"Circuits saved to {path}")

def get_client_class_counts(dataloader, num_classes):
    counts = torch.zeros(num_classes)
    for _, labels in dataloader:
        unique, c = torch.unique(labels, return_counts=True)
        for label, count in zip(unique, c):
            counts[label] += count
    return counts