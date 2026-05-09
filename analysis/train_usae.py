import os
import sys
import json
import argparse
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader
from core.models import get_model

class UniversalSAE(nn.Module):
    def __init__(self, num_models, input_dim, num_concepts, top_k=32):
        super().__init__()
        self.num_models = num_models
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.top_k = top_k

        self.W_enc = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, num_concepts) / (input_dim**0.5))
            for _ in range(num_models)
        ])
        self.b_pre = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_models)
        ])
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(num_concepts)
            for _ in range(num_models)
        ])
        self.D = nn.ParameterList([
            nn.Parameter(torch.randn(num_concepts, input_dim) / (num_concepts**0.5))
            for _ in range(num_models)
        ])

    def encode(self, x, model_idx):
        h = (x - self.b_pre[model_idx]) @ self.W_enc[model_idx]
        h = F.relu(self.bn[model_idx](h))
        
        top_vals, top_idx = torch.topk(h, self.top_k, dim=-1)
        z = torch.zeros_like(h)
        z.scatter_(-1, top_idx, top_vals)
        return z

    def decode(self, z, model_idx):
        return z @ self.D[model_idx]

def _get_stages(model):
    if hasattr(model, 'block1'):
        stem = lambda x: model.maxpool(model.relu(model.bn1(model.conv1(x))))
        return [stem, model.block1, model.block2, model.block3, model.block4], model.avgpool, 1
    stem = lambda x: model.pool(torch.relu(model.conv1(x)))
    return [
        stem,
        lambda x: model.pool(torch.relu(model.conv2(x))),
        lambda x: model.pool(torch.relu(model.conv3(x))),
    ], nn.Identity(), 0

def _forward_prefix(model, x, block_idx):
    stages, _, offset = _get_stages(model)
    for stage in stages[:block_idx + offset]:
        x = stage(x)
    return x

def _forward_suffix(model, x, block_idx):
    stages, pool, offset = _get_stages(model)
    for stage in stages[block_idx + offset:]:
        x = stage(x)
    return pool(x)

def extract_activations(model, dataloader, device, block_idx=4):
    model.eval()
    activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            x = _forward_prefix(model, x, block_idx)
            b, c, h, w = x.shape
            activations.append(rearrange(x, 'b c h w -> (b h w) c').cpu())
            
    return torch.cat(activations, dim=0)

def calculate_r2(true_A, recon_A):
    ss_res = ((true_A - recon_A)**2).sum()
    ss_tot = ((true_A - true_A.mean(dim=0))**2).sum()
    return max(0.0, (1 - ss_res / ss_tot).item())

def evaluate_usae_performance(source_idx, target_idx, source_model, target_model, usae, dataloader, device, block_idx=4, mask=None):
    source_model.eval()
    target_model.eval()
    usae.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            feat = _forward_prefix(source_model, x, block_idx)
            b, c, h, w = feat.shape
            feat_flat = rearrange(feat, 'b c h w -> (b h w) c')

            Z = usae.encode(feat_flat, source_idx)
            if mask is not None:
                Z = Z * mask

            recon_flat = usae.decode(Z, target_idx)
            recon_feat = rearrange(recon_flat, '(b h w) c -> b c h w', b=b, h=h, w=w).contiguous()

            out = _forward_suffix(target_model, recon_feat, block_idx)
            out = out.view(out.size(0), -1)
            logits = target_model.fc(out)

            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return correct / total

def get_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            outputs = outputs.view(outputs.size(0), -1) if not hasattr(model, 'block1') else outputs
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def _find_latest_checkpoint(checkpoints_dir):
    import glob as _glob
    files = _glob.glob(os.path.join(checkpoints_dir, "checkpoint_round_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_round_*.pt files in {checkpoints_dir}")
    return max(files, key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))


def main():
    parser = argparse.ArgumentParser(description="Train Universal SAE (USAE)")
    parser.add_argument("--model1", type=str, default=None, help="Path to first model checkpoint")
    parser.add_argument("--model2", type=str, default=None, help="Path to second model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config.json (shared config)")
    parser.add_argument("--config_a", type=str, default=None, help="Config of first experiment (auto-finds checkpoint)")
    parser.add_argument("--config_b", type=str, default=None, help="Config of second experiment (auto-finds checkpoint)")
    parser.add_argument("--output", type=str, default=None, help="Optional: Override output directory")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block", type=int, choices=[1, 2, 3, 4], default=3, help="Block/Layer index to extract from")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--expansion_factor", type=int, default=8, help="SAE expansion factor")
    parser.add_argument("--top_k", type=int, default=48, help="Top-K sparsity")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--subset_size", type=int, default=50000, help="Max activations")

    args = parser.parse_args()

    # Resolve paths when using config_a / config_b mode (run_suite.py style)
    if args.config_a and args.config_b:
        results_dir_a = os.path.dirname(os.path.abspath(args.config_a))
        results_dir_b = os.path.dirname(os.path.abspath(args.config_b))
        args.model1 = _find_latest_checkpoint(os.path.join(results_dir_a, "checkpoints"))
        args.model2 = _find_latest_checkpoint(os.path.join(results_dir_b, "checkpoints"))
        args.config = args.config_a  # use first config for model arch / dataset
        name_a = os.path.basename(results_dir_a)
        name_b = os.path.basename(results_dir_b)
    else:
        if args.model1 is None or args.model2 is None or args.config is None:
            parser.error("Provide either (--config_a + --config_b) or (--model1 + --model2 + --config)")
        name_a = os.path.basename(os.path.dirname(os.path.dirname(args.model1)))
        name_b = os.path.basename(os.path.dirname(os.path.dirname(args.model2)))

    if args.output and ("/" in args.output or "\\" in args.output):
        output_dir = os.path.dirname(os.path.abspath(args.output))
    else:
        output_dir = os.path.abspath(f"./results/usae_{name_a}_vs_{name_b}_block{args.block}")

    os.makedirs(output_dir, exist_ok=True)

    model_output = os.path.join(output_dir, "usae_model.pt")
    log_file = os.path.join(output_dir, "training_log.txt")
    eval_plot = os.path.join(output_dir, "evaluation_results.png")
    results_json = os.path.join(output_dir, "results.json")

    config = ExperimentConfig.load(args.config)
    device = torch.device(args.device)
    config.device = str(device)
    print(f"Using device: {device}")
    print(f"Results will be saved to: {output_dir}\n")

    def load_model(path):
        m = get_model(config)
        sd = torch.load(path, map_location=device)
        if 'model_state_dict' in sd:
            m.load_state_dict(sd['model_state_dict'])
        else:
            m.load_state_dict(sd)
        m.eval()
        return m

    print(f"Loading model 1 from {args.model1}...")
    model1 = load_model(args.model1)

    print(f"Loading model 2 from {args.model2}...")
    model2 = load_model(args.model2)
    
    print("Loading dataset...")
    _, testset = get_dataset(config)
    test_loader = get_test_dataloader(testset, config)
    
    print(f"Extracting activations from model 1 (Block {args.block})...")
    acts1 = extract_activations(model1, test_loader, device, args.block)
    print(f"Model 1 activations: {acts1.shape}")
    
    print(f"Extracting activations from model 2 (Block {args.block})...")
    acts2 = extract_activations(model2, test_loader, device, args.block)
    print(f"Model 2 activations: {acts2.shape}")
    
    min_size = min(len(acts1), len(acts2))
    subset_size = min(args.subset_size, min_size)
    A1_t = acts1[:subset_size].to(device)
    A2_t = acts2[:subset_size].to(device)
    
    dataloader = DataLoader(TensorDataset(A1_t, A2_t), batch_size=args.batch_size, shuffle=True)
    
    input_dim = A1_t.shape[-1]
    num_concepts = input_dim * args.expansion_factor
    print(f"Initializing USAE with {num_concepts} concepts (input_dim={input_dim})...")
    
    usae = UniversalSAE(num_models=2, input_dim=input_dim, num_concepts=num_concepts, top_k=args.top_k).to(device)
    optimizer = torch.optim.Adam(usae.parameters(), lr=args.lr)
    
    print(f"Starting USAE training for {args.epochs} epochs...")
    usae.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            A_batch = batch 
            i = random.randint(0, 1)
            
            optimizer.zero_grad()
            Z = usae.encode(A_batch[i], i)
            
            loss = 0.0
            for j in range(2):
                loss += F.mse_loss(usae.decode(Z, j), A_batch[j])
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} - Avg Loss: {total_loss/len(dataloader):.6f}")
            
    print("Training completed.")
    
    usae.eval()
    with torch.no_grad():
        A_eval = [A1_t, A2_t]
        labels = ["Model 1", "Model 2"]
        print(f"\n{'Encoder':<10} | {'Decoder':<10} | R^2 Score")
        print("-" * 35)
        for i in range(2):
            Z = usae.encode(A_eval[i], i)
            for j in range(2):
                r2 = calculate_r2(A_eval[j], usae.decode(Z, j))
                print(f"{labels[i]:<10} | {labels[j]:<10} | {r2:.4f}")
                
    acc1_base = get_accuracy(model1, test_loader, device)
    acc2_base = get_accuracy(model2, test_loader, device)
    print(f"\nModel 1 Base Accuracy: {acc1_base:.4f}")
    print(f"Model 2 Base Accuracy: {acc2_base:.4f}")
    
    with torch.no_grad():
        Z1 = usae.encode(A1_t, 0)
        Z2 = usae.encode(A2_t, 1)
        active1 = (Z1 > 0).any(dim=0)
        active2 = (Z2 > 0).any(dim=0)
        
    families = [
        ("Shared Concepts", (active1 & active2).float()),
        ("Model 1 Unique", (active1 & ~active2).float()),
        ("Model 2 Unique", (~active1 & active2).float()),
        ("Full USAE", None)
    ]
    
    results_data = [] 
    print(f"\n{'Concept Family':<20} | {'Source':<8} -> {'Target':<8} | Accuracy | % of Base")
    print("-" * 75)
    
    models = [model1, model2]
    base_accs = [acc1_base, acc2_base]
    for name, mask in families:
        active_count = int(mask.sum().item()) if mask is not None else num_concepts
        print(f"{name:<20} ({active_count:4d} concepts)")
        for i in range(2):
            for j in range(2):
                stitch_acc = evaluate_usae_performance(i, j, models[i], models[j], usae, test_loader, device, args.block, mask)
                rel_perf = (stitch_acc / base_accs[j]) * 100
                print(f"{'':<20} | {labels[i]:<8} -> {labels[j]:<8} | {stitch_acc:.4f}   | {rel_perf:6.1f}%")
                results_data.append({
                    'Family': name, 
                    'Source': labels[i], 
                    'Target': labels[j], 
                    'Accuracy': stitch_acc, 
                    'Relative': rel_perf
                })
        print("-" * 75)
        
    try:
        import pandas as pd
        df = pd.DataFrame(results_data)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.barplot(data=df, x='Family', y='Accuracy', hue='Source', ax=axes[0])
        axes[0].axhline(y=acc1_base, color='blue', linestyle='--', label='M1 Base')
        axes[0].axhline(y=acc2_base, color='orange', linestyle='--', label='M2 Base')
        axes[0].set_title("Stitching Accuracy")
        axes[0].set_ylim(0, 1.1)
        
        shared_df = df[df['Family'] == 'Shared Concepts'].pivot(index='Source', columns='Target', values='Relative')
        sns.heatmap(shared_df, annot=True, fmt=".1f", ax=axes[1])
        axes[1].set_title("Shared Concepts Relative (%)")
        
        plt.tight_layout()
        plt.savefig(eval_plot)
        print(f"Evaluation plot saved to: {eval_plot}")
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Save results as JSON
    try:
        results_summary = {
            'model1': args.model1,
            'model2': args.model2,
            'config': args.config,
            'block': args.block,
            'epochs': args.epochs,
            'base_acc_model1': float(acc1_base),
            'base_acc_model2': float(acc2_base),
            'num_concepts': num_concepts,
            'top_k': args.top_k,
            'results': results_data
        }
        with open(results_json, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Results summary saved to: {results_json}")
    except Exception as e:
        print(f"JSON save failed: {e}")
        
    torch.save({
        'model_state_dict': usae.state_dict(), 
        'input_dim': input_dim, 
        'num_concepts': num_concepts, 
        'top_k': args.top_k
    }, model_output)
    print(f"USAE model saved to: {model_output}")
    print(f"\n[OK] All results saved to: {output_dir}")

if __name__ == "__main__":
    main()