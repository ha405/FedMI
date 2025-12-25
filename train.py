import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import itertools

def get_current_sparsity(current_step, total_steps, final_sparsity, anneal_frac=0.5):
    anneal_steps = int(total_steps * anneal_frac)
    if current_step < anneal_steps:
        return final_sparsity * (current_step / anneal_steps)
    else:
        return final_sparsity

def binary_gate(x):
    return (x > 0).float() - torch.sigmoid(x).detach() + torch.sigmoid(x)

def get_gate_hook(gate_param):
    def hook(module, input, output):
        return output * binary_gate(gate_param)
    return hook

def get_gate_mean_hook(gate_param, mean_tensor):
    """
    Gating hook that uses STE but falls back to MEAN instead of ZERO.
    Used during Circuit Discovery training.
    """
    def hook(module, input, output):
        # 1. Get the binary mask (via STE)
        mask = binary_gate(gate_param)
        
        # 2. Apply Mean Ablation Logic during Discovery
        # Output = (Signal * Mask) + (Mean * (1 - Mask))
        return (output * mask) + (mean_tensor * (1.0 - mask))
    return hook

def get_hard_mask_hook(indices, device):
    """Zero Ablation Hook: Keeps ONLY the indices."""
    def hook(module, input, output):
        mask = torch.zeros(1, output.shape[1], 1, 1).to(device)
        if len(indices) > 0:
            mask[:, indices, :, :] = 1.0
        return output * mask
    return hook

def get_inverse_mask_hook(indices, device):
    """
    Inverse Mask Hook: ZEROs the circuit indices and keeps everything else ON.
    Used for Necessity testing.
    """
    def hook(module, input, output):
        # Start with everything ON (1s)
        mask = torch.ones(1, output.shape[1], 1, 1).to(device)
        if len(indices) > 0:
            # Turn OFF the specific circuit indices
            mask[:, indices, :, :] = 0.0
        return output * mask
    return hook

# --- Mean Ablation Logic ---

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

def get_mean_ablation_hook(indices, mean_tensor, device):
    """
    Mean Ablation Hook:
    If kept: Return Output.
    If pruned: Return Mean Value.
    """
    def hook(module, input, output):
        # 1. Create Mask (1 for Keep, 0 for Prune)
        mask = torch.zeros(1, output.shape[1], 1, 1).to(device)
        if len(indices) > 0:
            mask[:, indices, :, :] = 1.0
            
        # 2. Apply: (Signal * Mask) + (Mean * InverseMask)
        return (output * mask) + (mean_tensor * (1.0 - mask))
    return hook

# --------------------------------

def apply_weight_sparsity(model, sparsity_level=0.90, min_alive=4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                flat_param = param.abs().flatten()
                num_keep = int((1 - sparsity_level) * flat_param.numel())
                if num_keep < 1: num_keep = 1
                
                threshold = torch.topk(flat_param, num_keep).values[-1]
                mask = (param.abs() >= threshold).float()

                if 'conv' in name and param.dim() == 4:
                    for i in range(param.shape[0]):
                        filter_weights = param[i]
                        alive_count = (mask[i] > 0).sum().item()
                        
                        if alive_count < min_alive:
                            top_k_vals = torch.topk(filter_weights.abs().flatten(), min_alive).values
                            if top_k_vals.numel() > 0:
                                revival_threshold = top_k_vals[-1]
                                revival_mask = (filter_weights.abs() >= revival_threshold).float()
                                mask[i] = torch.max(mask[i], revival_mask)
                
                param.data.mul_(mask)

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

def extract_sparse_connectivity(model):
    """
    Returns a dictionary where key is layer name and value is a list of [dest, src] pairs
    representing non-zero weights.
    """
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
    """
    Filters physical connections to keep only those where both Source and Destination 
    nodes are active in the specific class circuit.
    
    Args:
        physical_connectivity: Dict {layer_name: [[dest, src], ...]}
        active_circuit: Dict {layer_name: [list_of_active_indices]}
        
    Returns:
        functional_connectivity: Dict {layer_name: {dest_idx: [src_idx_1, src_idx_2...]}}
    """
    functional_connectivity = {}
    
    # We need to know the order of layers to check Previous Layer (Source) activation
    # This assumes keys are inserted in order (Python 3.7+)
    layer_order = list(physical_connectivity.keys())
    
    for i, layer_name in enumerate(layer_order):
        # 1. Get physical edges [[dest, src], ...]
        edges = physical_connectivity.get(layer_name, [])
        if not edges: continue
            
        # 2. Get active nodes for CURRENT layer (Destination)
        # Handle cases where layer might not be in circuit dict (e.g., if FC layer wasn't pruned)
        active_dest_nodes = set(active_circuit.get(layer_name, []))
        
        # 3. Get active nodes for PREVIOUS layer (Source)
        if i == 0:
            # Special case for first layer (conv1): 
            # Input image channels (0, 1, 2) are assumed ALWAYS active.
            # So we only filter based on Destination.
            valid_edges = [
                (dest, src) for dest, src in edges 
                if dest in active_dest_nodes
            ]
        else:
            prev_layer_name = layer_order[i-1]
            active_src_nodes = set(active_circuit.get(prev_layer_name, []))
            
            # Check BOTH Dest and Source are active
            valid_edges = [
                (dest, src) for dest, src in edges 
                if dest in active_dest_nodes and src in active_src_nodes
            ]
            
        # 4. Format as Dictionary: {Dest: [Source1, Source2...]}
        layer_dict = {}
        for dest, src in valid_edges:
            if dest not in layer_dict:
                layer_dict[dest] = []
            layer_dict[dest].append(src)
            
        functional_connectivity[layer_name] = layer_dict
        
    return functional_connectivity