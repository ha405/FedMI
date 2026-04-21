import torch

def get_current_sparsity(current_step, total_steps, final_sparsity, anneal_frac=0.5):
    anneal_steps = int(total_steps * anneal_frac)
    if current_step < anneal_steps:
        return final_sparsity * (current_step / anneal_steps)
    else:
        return final_sparsity

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
