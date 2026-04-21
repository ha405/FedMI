import torch
import torch.nn as nn
from .hooks import get_hard_mask_hook

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA) between two matrices X and Y.
    X: [N, D1]
    Y: [N, D2]
    N is the number of samples, D1 and D2 are the feature dimensions.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples (rows).")
    
    X = X.view(X.size(0), -1)
    Y = Y.view(Y.size(0), -1)

    # Center the columns of X and Y
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Compute inner products
    # Instead of computing the full N x N Gram matrices K and L and doing dot(K, L),
    # we can use the identity trace(K L) = ||X^T Y||_F^2 for linear kernels
    # This is much more memory efficient for large N
    dot_xy = torch.norm(torch.matmul(X_centered.t(), Y_centered), p='fro') ** 2
    norm_xx = torch.norm(torch.matmul(X_centered.t(), X_centered), p='fro')
    norm_yy = torch.norm(torch.matmul(Y_centered.t(), Y_centered), p='fro')
    
    if norm_xx == 0 or norm_yy == 0:
        return 0.0
        
    cka_score = dot_xy / (norm_xx * norm_yy)
    return cka_score.item()

def _get_target_layer_name(model):
    """Finds a layer before the classifier for default extraction"""
    valid_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            valid_names.append(name)
    
    # Try to find the last conv or last feature block before the final head/fc
    candidates = [name for name in valid_names if 'fc' not in name.lower() and 'head' not in name.lower() and 'classifier' not in name.lower()]
    if candidates:
        return candidates[-1]
    return valid_names[-2] if len(valid_names) > 1 else valid_names[0]

def extract_circuit_activations(model, dataloader, circuit, config, layer_name=None, max_samples=1000):
    """
    Run inputs through model WITH circuit mask, and get activations at a target layer.
    Returns: torch.Tensor [N, D]
    """
    device = config.device
    hooks = []
    
    if layer_name is None:
        layer_name = _get_target_layer_name(model)
        
    # Apply circuit mask
    for circuit_layer_name, indices in circuit.items():
        try:
            module = model.get_submodule(circuit_layer_name)
        except AttributeError:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        hooks.append(module.register_forward_hook(get_hard_mask_hook(idx_tensor, device)))
        
    activations = []
    target_module = model.get_submodule(layer_name)
    
    def capture_hook(module, input, output):
        activations.append(output.detach().cpu())
        
    capture_h = target_module.register_forward_hook(capture_hook)
    
    model.eval()
    samples_collected = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            samples_collected += inputs.size(0)
            if samples_collected >= max_samples:
                break
                
    for h in hooks:
        h.remove()
    capture_h.remove()
    
    if not activations:
        return torch.empty(0)
    
    out = torch.cat(activations, dim=0)
    return out[:max_samples]

def extract_prehead_latents(model, dataloader, config, max_samples=1000):
    """
    Run inputs through FULL model (no masking), get activations just before the classification head.
    Returns: torch.Tensor [N, D]
    """
    device = config.device
    layer_name = _get_target_layer_name(model)
    
    activations = []
    target_module = model.get_submodule(layer_name)
    
    def capture_hook(module, input, output):
        activations.append(output.detach().cpu())
        
    capture_h = target_module.register_forward_hook(capture_hook)
    
    model.eval()
    samples_collected = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            samples_collected += inputs.size(0)
            if samples_collected >= max_samples:
                break
                
    capture_h.remove()
    
    if not activations:
        return torch.empty(0)
        
    out = torch.cat(activations, dim=0)
    return out[:max_samples]

def cka_matrix(activations_a, activations_b, class_labels):
    """
    Given two dicts of {class_label: activation_tensor}, compute CKA for each matching class.
    Returns: dict {class_label: cka_score}
    """
    scores = {}
    for label in class_labels:
        if str(label) in activations_a and str(label) in activations_b:
            X = activations_a[str(label)]
            Y = activations_b[str(label)]
            scores[str(label)] = linear_cka(X, Y)
        else:
            scores[str(label)] = None
    return scores
