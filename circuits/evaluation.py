import torch
import torch.nn as nn
import torch.nn.functional as F
from .hooks import get_hard_mask_hook, get_inverse_mask_hook, get_mean_ablation_hook
from core.data_cache import EvaluationCache


def evaluate_circuit_cached(model, cache: EvaluationCache, circuit, target_class, config,
                             layer_means=None, log_file=None, class_names=None) -> float:
    """Sufficiency test: can the circuit alone perform the task?"""
    device = config.device
    hooks = []
    for layer_name, indices in circuit.items():
        module = model.get_submodule(layer_name)
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        if config.use_mean_ablation and layer_means is not None and layer_name in layer_means:
            hooks.append(module.register_forward_hook(get_mean_ablation_hook(idx_t, layer_means[layer_name], device)))
        else:
            hooks.append(module.register_forward_hook(get_hard_mask_hook(idx_t, device)))

    model.eval()
    correct, total = 0, 0
    failure_prob_sum = torch.zeros(config.num_classes, device=device)
    failure_count = 0

    class_inputs, class_labels = cache.get_class_data(target_class)
    if class_inputs.shape[0] > 0:
        with torch.no_grad():
            for i in range(0, class_inputs.shape[0], config.batch_size):
                inp = class_inputs[i:i + config.batch_size]
                lbl = class_labels[i:i + config.batch_size]
                out = model(inp)
                _, pred = torch.max(out, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()
                if log_file:
                    wrong = pred != lbl
                    if wrong.any():
                        failure_prob_sum += F.softmax(out, dim=1)[wrong].sum(dim=0)
                        failure_count += wrong.sum().item()

    for h in hooks:
        h.remove()

    if log_file and failure_count > 0:
        avg_fp = failure_prob_sum / failure_count
        top5_v, top5_i = torch.topk(avg_fp, k=min(5, config.num_classes))
        parts = [f"{class_names[idx.item()] if class_names else idx.item()} ({v.item()*100:.1f}%)" for idx, v in zip(top5_i, top5_v)]
        t_name = class_names[target_class] if class_names else str(target_class)
        log_file.write(f"  [Cross-Eval] {t_name} | Acc: {100*correct/total if total else 0:.2f}% | Top-5 confusion: {', '.join(parts)}\n")

    return (100 * correct / total) if total > 0 else 0.0


def evaluate_circuit_necessity_cached(model, cache: EvaluationCache, circuit, target_class, config) -> float:
    """Necessity test: does performance collapse when the circuit is ablated?"""
    device = config.device
    hooks = []
    for layer_name, indices in circuit.items():
        module = model.get_submodule(layer_name)
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        hooks.append(module.register_forward_hook(get_inverse_mask_hook(idx_t, device)))

    model.eval()
    correct, total = 0, 0
    class_inputs, class_labels = cache.get_class_data(target_class)
    if class_inputs.shape[0] > 0:
        with torch.no_grad():
            for i in range(0, class_inputs.shape[0], config.batch_size):
                inp = class_inputs[i:i + config.batch_size]
                lbl = class_labels[i:i + config.batch_size]
                out = model(inp)
                _, pred = torch.max(out, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()

    for h in hooks:
        h.remove()
    return (100 * correct / total) if total > 0 else 0.0


def evaluate_detailed_with_loss(model, cache: EvaluationCache, config,
                                class_names=None, title="Global Eval",
                                active_classes=None, log_file=None):
    """
    Evaluates model on EvaluationCache. Returns (overall_acc, avg_loss, class_acc_dict).
    class_acc_dict: {class_id: accuracy_float | None}
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_classes = config.num_classes
    class_correct = [0.0] * num_classes
    class_total   = [0.0] * num_classes
    correct, total, running_loss, n_batches = 0, 0, 0.0, 0

    if log_file:
        log_file.write(f"\n--- {title} ---\n")

    with torch.no_grad():
        for inputs, labels in cache.iterate_batches(config.batch_size):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            valid_mask = labels < num_classes
            if active_classes is not None:
                ac_mask = torch.zeros_like(valid_mask)
                for c in active_classes:
                    ac_mask |= (labels == c)
                valid_mask &= ac_mask
            if not valid_mask.any():
                continue
            inputs, labels = inputs[valid_mask], labels[valid_mask]
            outputs = model(inputs)
            running_loss += criterion(outputs, labels).item()
            n_batches += 1
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_correct[lbl] += (predicted[i] == labels[i]).item()
                class_total[lbl] += 1

    overall_acc = 100 * correct / total if total > 0 else 0.0
    avg_loss    = running_loss / n_batches if n_batches > 0 else 0.0

    eval_classes = sorted(active_classes) if active_classes is not None else list(range(num_classes))
    class_acc = {c: (100 * class_correct[c] / class_total[c] if class_total[c] > 0 else None) for c in eval_classes}

    if log_file:
        log_file.write(f"Overall: {overall_acc:.2f}% | Loss: {avg_loss:.4f}\n")
        for c in eval_classes:
            name = class_names[c] if class_names else str(c)
            if class_acc[c] is not None:
                log_file.write(f"  {name}: {class_acc[c]:.2f}% ({int(class_correct[c])}/{int(class_total[c])})\n")
            else:
                log_file.write(f"  {name}: N/A\n")
        log_file.write("-" * 30 + "\n")
        log_file.flush()

    return overall_acc, avg_loss, class_acc


def extract_sparse_connectivity(model):
    from .discovery import is_valid_layer
    connectivity = {}
    for name, module in model.named_modules():
        if not is_valid_layer(name, module):
            continue
        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu()
            connectivity[name] = torch.nonzero(w.abs().sum(dim=(2, 3)) > 0, as_tuple=False).tolist()
        elif isinstance(module, nn.Linear):
            w = module.weight.detach().cpu()
            connectivity[name] = torch.nonzero(w.abs() > 0, as_tuple=False).tolist()
    return connectivity


def filter_connectivity_by_circuit(physical_connectivity, active_circuit):
    layer_order = list(physical_connectivity.keys())
    functional_connectivity = {}
    for i, layer_name in enumerate(layer_order):
        edges = physical_connectivity.get(layer_name, [])
        if not edges:
            continue
        active_dest = set(active_circuit.get(layer_name, []))
        if i == 0:
            valid = [(d, s) for d, s in edges if d in active_dest]
        else:
            active_src = set(active_circuit.get(layer_order[i - 1], []))
            valid = [(d, s) for d, s in edges if d in active_dest and s in active_src]
        layer_dict = {}
        for d, s in valid:
            layer_dict.setdefault(d, []).append(s)
        functional_connectivity[layer_name] = layer_dict
    return functional_connectivity
