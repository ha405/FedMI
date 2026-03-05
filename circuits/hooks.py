import torch
import torch.nn as nn

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
        mask = binary_gate(gate_param)
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
        mask = torch.ones(1, output.shape[1], 1, 1).to(device)
        if len(indices) > 0:
            mask[:, indices, :, :] = 0.0
        return output * mask
    return hook

def get_mean_ablation_hook(indices, mean_tensor, device):
    """
    Mean Ablation Hook:
    If kept: Return Output.
    If pruned: Return Mean Value.
    """
    def hook(module, input, output):
        mask = torch.zeros(1, output.shape[1], 1, 1).to(device)
        if len(indices) > 0:
            mask[:, indices, :, :] = 1.0
        return (output * mask) + (mean_tensor * (1.0 - mask))
    return hook
