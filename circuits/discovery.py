import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
def discover_client_circuit(model, dataloader, target_class, config, layer_means=None):
    device = config.device
    
    # 1. Pre-collect target class samples (Optimization for non-IID/sparse data)
    # This prevents scanning the entire dataloader in every discovery step.
    # We collect up to a reasonable buffer (e.g., 1024 samples) which is plenty for discovery.
    target_samples = []
    target_labels = []
    max_buf = 1024
    
    for b_inputs, b_labels in dataloader:
        mask = (b_labels == target_class)
        if mask.any():
            target_samples.append(b_inputs[mask])
            target_labels.append(b_labels[mask])
            
            # Check buffer size
            if sum(x.shape[0] for x in target_samples) >= max_buf:
                break
                
    if not target_samples:
        return {name: [] for name in [n for n, m in model.named_modules() if is_valid_layer(n, m)]}

    # Combine into a single batch for easier sampling
    all_inputs = torch.cat(target_samples, dim=0).to(device)
    all_labels = torch.cat(target_labels, dim=0).to(device)
    num_avail = all_inputs.shape[0]

    criterion = nn.CrossEntropyLoss()
    original_grads = {name: param.requires_grad for name, param in model.named_parameters()}
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    
    gate_params, hooks, layers = {}, [], []
    for name, module in model.named_modules():
        if is_valid_layer(name, module):
            layers.append(name)
            if isinstance(module, nn.Conv2d):
                gate_params[name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
            else: # Linear
                gate_params[name] = nn.Parameter(torch.ones(module.out_features).to(device) * 2.0)
            
            if config.use_mean_ablation and layer_means and name in layer_means:
                hooks.append(module.register_forward_hook(get_gate_mean_hook(gate_params[name], layer_means[name])))
            else:
                hooks.append(module.register_forward_hook(get_gate_hook(gate_params[name])))

    optimizer = optim.Adam(gate_params.values(), lr=config.gate_lr)
    
    # Discovery Loop: Now significantly faster as we sample from pre-collected buffer
    for _ in range(config.discovery_steps):
        # Sample a batch from our collected samples
        # If we have fewer than batch_size, take all. Otherwise, take a random subset.
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
    
    for h in hooks: h.remove()
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
