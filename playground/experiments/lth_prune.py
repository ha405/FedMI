import os
import sys
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseExperiment
from .ensemble_distill import EnsembleModel, load_client_models
from playground.core import load_config, load_dataset
from playground.core.evaluator import eval_full

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.models import get_model
from core.dataset import get_dataset, split_public_data, get_test_dataloader
from circuits.discovery import discover_client_circuit, compute_layer_means
from circuits.evaluation import evaluate_circuit, evaluate_circuit_necessity


def count_nonzero_params(model):
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += (param.data != 0).sum().item()
    return nonzero, total


def global_magnitude_prune(model, prune_rate, target_nonzero=None):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            all_weights.append(param.data.abs().flatten())

    if not all_weights:
        return

    all_weights = torch.cat(all_weights)

    current_nonzero = (all_weights > 0).sum().item()
    num_to_keep = int(current_nonzero * (1.0 - prune_rate))
    
    if target_nonzero is not None and num_to_keep < target_nonzero:
        num_to_keep = target_nonzero
        
    if num_to_keep < 1:
        num_to_keep = 1

    nonzero_weights = all_weights[all_weights > 0]
    if len(nonzero_weights) <= num_to_keep:
        return

    threshold = torch.topk(nonzero_weights, num_to_keep).values[-1]

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                mask = (param.data.abs() >= threshold).float()
                param.data.mul_(mask)


def finetune(model, dataloader, cfg, epochs, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Zero out gradients for pruned weights (keep them dead)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and param.dim() > 1:
                        param.grad[param.data == 0] = 0.0

            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"    Finetune Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    return model


def discover_all_circuits(model, dataloader, testloader, cfg, class_names):
    circuits = {}
    layer_means = None
    if cfg.use_mean_ablation:
        layer_means = compute_layer_means(model, dataloader, cfg)

    for tc in range(cfg.num_classes):
        if class_names and 0 <= tc < len(class_names):
            name = class_names[tc]
        else:
            name = str(tc)

        circ = discover_client_circuit(model, dataloader, tc, cfg, layer_means=layer_means)
        acc = evaluate_circuit(model, testloader, circ, tc, cfg, layer_means=layer_means)
        nec = evaluate_circuit_necessity(model, testloader, circ, tc, cfg)

        circuits[name] = {
            "active_nodes": circ,
            "metrics": {"accuracy": acc, "necessity": nec}
        }

        counts = {layer: len(idx) for layer, idx in circ.items()}
        print(f"    Class {name}: {counts} | Suff: {acc:.2f}% | Nec: {nec:.2f}%")

    return circuits


class LTHPruneExperiment(BaseExperiment):
    def run(self):
        args = self.args
        cfg = load_config(args.exp_dir)
        device = cfg.device

        trainset, testset = get_dataset(cfg)
        testloader = get_test_dataloader(testset, cfg)
        class_names = list(trainset.classes) if hasattr(trainset, 'classes') else [str(i) for i in range(cfg.num_classes)]

        public_fraction = args.public_fraction
        _, public_loader = split_public_data(trainset, public_fraction, cfg.public_data_seed, cfg)

        if public_loader is None:
            sys.exit("[ERROR] public_fraction is 0. Set --public_fraction > 0.")

        round_num = args.round or cfg.num_rounds

        # --- Step 1: Build Ensemble (Logit Averaging) ---
        self.print_header("Building Ensemble (Logit Averaging)")
        client_models = load_client_models(args.exp_dir, round_num, cfg.num_clients, cfg)
        
        # Calculate the average non-zero elements across all client models to define target
        avg_single_nonzero = sum(count_nonzero_params(m)[0] for m in client_models) // len(client_models)
        
        model = EnsembleModel(client_models)
        model.to(device)
        model.eval()

        num_ensemble = len(client_models)
        nonzero_start, total_params = count_nonzero_params(model)
        
        print(f"  Ensemble size: {num_ensemble} models")
        print(f"  Total params (ensemble): {total_params:,}")
        print(f"  Average single model non-zero params: {avg_single_nonzero:,}")
        print(f"  Non-zero params (start): {nonzero_start:,}")
        print(f"  Initial sparsity: {1.0 - nonzero_start/total_params:.4f}")

        # Target non-zero elements
        target_nonzero = args.target_nonzero if args.target_nonzero is not None else avg_single_nonzero
        print(f"  Target non-zero params: {target_nonzero:,} (match average non-zero of a single model)")

        # Baseline evaluation
        self.print_header("Baseline Ensemble Evaluation")
        baseline_result = eval_full(model, testloader, cfg)
        self.print_result("Baseline Overall Acc:", f"{baseline_result['overall']:.2f}%")
        for cls, acc in baseline_result["per_class"].items():
            self.print_result(f"  Class {cls}:", f"{acc}%" if acc is not None else "N/A")

        # --- Step 2: Iterative Magnitude Pruning ---
        self.print_header(f"Iterative Magnitude Pruning (rate={args.prune_rate}, target_nonzero={target_nonzero:,})")

        pruning_log = []

        for iteration in range(1, args.prune_iters + 1):
            print(f"\n  --- Prune Iteration {iteration}/{args.prune_iters} ---")

            global_magnitude_prune(model, args.prune_rate, target_nonzero=target_nonzero)

            nonzero_now, _ = count_nonzero_params(model)
            current_sparsity = 1.0 - nonzero_now / total_params
            print(f"    Non-zero: {nonzero_now:,} | Sparsity: {current_sparsity:.4f}")

            # Evaluate after pruning (before finetune)
            pre_ft_result = eval_full(model, testloader, cfg)
            print(f"    Acc after pruning (Pre-Finetune): {pre_ft_result['overall']:.2f}%")

            # Finetune
            print(f"    Finetuning ({args.finetune_epochs} epochs)...")
            model = finetune(model, public_loader, cfg, args.finetune_epochs, args.finetune_lr)

            # Evaluate after finetune
            post_ft_result = eval_full(model, testloader, cfg)
            print(f"    Acc after finetuning (Post-Finetune): {post_ft_result['overall']:.2f}%")

            pruning_log.append({
                "iteration": iteration,
                "nonzero_params": nonzero_now,
                "sparsity": round(current_sparsity, 6),
                "pre_finetune_acc": pre_ft_result["overall"],
                "post_finetune_acc": post_ft_result["overall"]
            })

            if nonzero_now <= target_nonzero:
                print(f"\n    Target non-zero params ({target_nonzero}) reached. Stopping.")
                break

        # --- Step 3: Final Evaluation ---
        self.print_header("Final Pruned Model Evaluation")
        final_result = eval_full(model, testloader, cfg)
        self.print_result("Final Overall Acc:", f"{final_result['overall']:.2f}%")
        for cls, acc in final_result["per_class"].items():
            self.print_result(f"  Class {cls}:", f"{acc}%" if acc is not None else "N/A")

        nonzero_final, _ = count_nonzero_params(model)
        final_sparsity = 1.0 - nonzero_final / total_params
        self.print_result("Final Sparsity:", f"{final_sparsity:.4f}")
        self.print_result("Non-zero Params:", f"{nonzero_final:,} / {total_params:,}")

        # --- Step 4: Circuit Discovery on Pruned Model ---
        self.print_header("Circuit Discovery on Pruned Model")
        pruned_circuits = discover_all_circuits(model, public_loader, testloader, cfg, class_names)

        # --- Step 5: Save ---
        output_dir = os.path.join(args.exp_dir, "playground_results", "lth_prune")
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(output_dir, "pruned_model.pt"))

        results = {
            "baseline_accuracy": baseline_result,
            "final_accuracy": final_result,
            "final_sparsity": round(final_sparsity, 6),
            "final_nonzero_params": nonzero_final,
            "total_params": total_params,
            "pruning_log": pruning_log,
            "pruned_circuits": pruned_circuits,
            "config": {
                "prune_rate": args.prune_rate,
                "prune_iters": args.prune_iters,
                "finetune_epochs": args.finetune_epochs,
                "finetune_lr": args.finetune_lr,
                "target_nonzero": args.target_nonzero,
                "public_fraction": args.public_fraction,
                "round": round_num
            }
        }

        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to: {output_dir}")


def add_args(subparsers):
    p = subparsers.add_parser("lth_prune",
                              help="Apply LTH-style iterative magnitude pruning to an ensemble model.")
    p.add_argument("--exp_dir", required=True, help="Path to completed experiment directory")
    p.add_argument("--round", type=int, default=None, help="Round to load client models from (default: last)")
    p.add_argument("--prune_rate", type=float, default=0.2, help="Fraction of remaining weights to prune per iteration")
    p.add_argument("--prune_iters", type=int, default=10, help="Maximum number of pruning iterations")
    p.add_argument("--finetune_epochs", type=int, default=0, help="Epochs of finetuning after each prune")
    p.add_argument("--finetune_lr", type=float, default=0.0005, help="Finetuning learning rate")
    p.add_argument("--target_nonzero", type=int, default=None, help="Target non-zero elements (default: auto = average non-zero of a single model)")
    p.add_argument("--public_fraction", type=float, default=0.1, help="Fraction of train set to use as public data")
    return p