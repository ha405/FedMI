import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .hooks import get_gate_hook, get_gate_mean_hook

def discover_client_circuit(model, dataloader, target_class, config, layer_means=None):
    device = config.device
    
    # Check if target class exists in this dataloader
    # This check is expensive if dataloader is large? 
    # Original code iterated it.
    # To be safe and efficient, we just try to find one batch with the target.
    found_batch = False
    for _, labels in dataloader:
        if target_class in labels:
            found_batch = True
            break
    
    layers_to_discover = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]

    if not found_batch:
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
        inputs, labels = None, None
        
        # Find a batch with the target class
        for b_inputs, b_labels in data_iter:
            if target_class in b_labels:
                inputs, labels = b_inputs.to(device), b_labels.to(device)
                found_batch = True
                break
                
        if not found_batch: continue 

        mask = (labels == target_class)
        if mask.sum() == 0: continue
        
        optimizer.zero_grad()
        l0_loss = sum(torch.sigmoid(p).sum() for p in gate_params.values())
        logits = model(inputs[mask])
        
        # Loss calculation
        cls_loss = criterion(logits, labels[mask])
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
    
    def get_activation_hook(name):
        def hook(model, input, output):
            # output shape: [Batch, Channel, Height, Width]
            if name not in layer_sums:
                layer_sums[name] = torch.zeros(output.shape[1], device=device)
                layer_counts[name] = 0
            
            # Sum over Batch(0), Height(2), Width(3) -> Keep Channel(1)
            batch_sum = output.sum(dim=(0, 2, 3)) 
            layer_sums[name] += batch_sum
            
            # Count total pixels seen
            layer_counts[name] += output.shape[0] * output.shape[2] * output.shape[3]
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
            
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
            # Reshape to [1, C, 1, 1] for broadcasting
            mean_val = (layer_sums[name] / layer_counts[name]).view(1, -1, 1, 1)
            layer_means[name] = mean_val
        
    return layer_means
