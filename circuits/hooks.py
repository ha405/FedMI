import torch
import torch.nn as nn

def binary_gate(x):
    return (x > 0).float() - torch.sigmoid(x).detach() + torch.sigmoid(x)

def get_gate_hook(gate_param):
    def hook(module, input, output):
        return output * binary_gate(gate_param)
    return hook

def apply_mask(output, indices, device, mask_value=1.0, default_value=0.0):
    mask = torch.full_like(output, default_value, device=device)
    if len(indices) > 0:
        if len(output.shape) == 4: # Conv2d
            mask[:, indices, :, :] = mask_value
        elif len(output.shape) == 3: # Linear (Sequence e.g. ViT)
            mask[:, :, indices] = mask_value
        elif len(output.shape) == 2: # Linear (Standard)
            mask[:, indices] = mask_value
        else:
            # Fallback
            mask[..., indices] = mask_value
    return mask

def get_hard_mask_hook(indices, device):
    """Zero Ablation Hook: Keeps ONLY the indices."""
    def hook(module, input, output):
        mask = apply_mask(output, indices, device, mask_value=1.0, default_value=0.0)
        return output * mask
    return hook

def get_inverse_mask_hook(indices, device):
    """
    Inverse Mask Hook: ZEROs the circuit indices and keeps everything else ON.
    Used for Necessity testing.
    """
    def hook(module, input, output):
        mask = apply_mask(output, indices, device, mask_value=0.0, default_value=1.0)
        return output * mask
    return hook


