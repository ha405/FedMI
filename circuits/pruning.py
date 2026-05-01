import torch

def get_current_sparsity(current_step, total_steps, final_sparsity, anneal_frac=0.5):
    anneal_steps = int(total_steps * anneal_frac)
    if current_step < anneal_steps:
        return final_sparsity * (current_step / anneal_steps)
    else:
        return final_sparsity

def apply_weight_sparsity(model, sparsity_level=0.90, min_alive=4):
    exclude_terms = ['patch_embed', 'head', 'classifier', 'embed', 'fc']
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(term in name for term in exclude_terms):
                continue
            if 'weight' in name and param.dim() > 1:
                flat_param = param.abs().flatten()
                num_keep = int((1 - sparsity_level) * flat_param.numel())
                if num_keep < 1: num_keep = 1
                
                threshold = torch.topk(flat_param, num_keep).values[-1]
                mask = (param.abs() >= threshold).float()

                if 'conv' in name and param.dim() == 4:
                    # Vectorized revival of filters
                    alive_per_filter = mask.view(param.shape[0], -1).sum(dim=1)
                    dead_filters = (alive_per_filter < min_alive).nonzero(as_tuple=True)[0]
                    
                    if dead_filters.numel() > 0:
                        # Extract only weights for filters
                        flat_dead = param[dead_filters].view(dead_filters.numel(), -1).abs()
                        # Get the threshold for the top min_alive weights
                        revival_thresholds = torch.topk(flat_dead, min_alive, dim=1).values[:, -1]
                        revival_mask = (param[dead_filters].abs() >= revival_thresholds.view(-1, 1, 1, 1)).float()
                        mask[dead_filters] = torch.max(mask[dead_filters], revival_mask)
                
                param.data.mul_(mask)
