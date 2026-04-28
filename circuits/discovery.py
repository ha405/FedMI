import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from .hooks import get_gate_hook, get_gate_mean_hook

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
    
    This replaces the pattern of calling discover_client_circuit 10 times,
    each scanning the entire dataloader to find one class. Instead we do
    1 pass and hand pre-collected tensors to the discovery function.
    
    Returns:
        {class_id: (inputs_tensor, labels_tensor)}  — already on device.
        Missing classes (no samples found) are omitted from the dict.
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
                                    target_class: int, config, layer_means=None):
    """
    Circuit discovery using PRE-COLLECTED samples (already on device).
    
    This is the optimized variant of discover_client_circuit that avoids
    scanning the dataloader.  Call precollect_all_class_samples() first,
    then pass the tensors here.
    """
    device = config.device

    if class_inputs is None or class_inputs.shape[0] == 0:
        return {name: [] for name, m in model.named_modules() if is_valid_layer(name, m)}

    all_inputs = class_inputs  # already on device
    all_labels = class_labels
    num_avail = all_inputs.shape[0]

    criterion = nn.CrossEntropyLoss()
    original_grads = {name: param.requires_grad for name, param in model.named_parameters()}
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    gate_params, hooks, layers = {}, [], []
    for name, module in model.named_modules():
        if is_valid_layer(name, module):
            layers.append(name)
            if isinstance(module, nn.Conv2d):
                gate_params[name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
            else:
                gate_params[name] = nn.Parameter(torch.ones(module.out_features).to(device) * 2.0)

            if config.use_mean_ablation and layer_means and name in layer_means:
                hooks.append(module.register_forward_hook(get_gate_mean_hook(gate_params[name], layer_means[name])))
            else:
                hooks.append(module.register_forward_hook(get_gate_hook(gate_params[name])))

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





def compute_layer_means(model, dataloader, config):
    """
    CALIBRATION STEP:
    Calculates the temporal mean activation of every channel across the dataset.
    """
    model.eval()
    device = config.device
    
    # Store sums and counts
    layer_sums = {} 
    layer_counts = {}
    
    def get_activation_hook(name, module):
        def hook(mod, input, output):
            if isinstance(module, nn.Conv2d):
                if name not in layer_sums:
                    layer_sums[name] = torch.zeros(output.shape[1], device=device)
                    layer_counts[name] = 0
                layer_sums[name] += output.sum(dim=(0, 2, 3))
                layer_counts[name] += output.shape[0] * output.shape[2] * output.shape[3]
            else: # Linear
                if name not in layer_sums:
                    layer_sums[name] = torch.zeros(output.shape[-1], device=device)
                    layer_counts[name] = 0
                if len(output.shape) == 3: # ViT [B, Seq, C]
                    layer_sums[name] += output.sum(dim=(0, 1))
                    layer_counts[name] += output.shape[0] * output.shape[1]
                else: # [B, C]
                    layer_sums[name] += output.sum(dim=0)
                    layer_counts[name] += output.shape[0]
        return hook

    hooks = []
    layer_types = {}
    for name, module in model.named_modules():
        if is_valid_layer(name, module):
            layer_types[name] = type(module)
            hooks.append(module.register_forward_hook(get_activation_hook(name, module)))
            
    # Run pass
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            
    for h in hooks: h.remove()
    
    # Compute Means
    layer_means = {}
    for name in layer_sums:
        if layer_counts[name] > 0:
            mean_val = layer_sums[name] / layer_counts[name]
            if issubclass(layer_types[name], nn.Conv2d):
                mean_val = mean_val.view(1, -1, 1, 1)
            layer_means[name] = mean_val
        
    return layer_means
