import os
import sys
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseExperiment
from playground.core import load_config, load_dataset, load_circuits
from playground.core.evaluator import eval_full

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.models import get_model
from core.dataset import get_dataset, split_public_data, get_test_dataloader
from circuits.discovery import discover_client_circuit, compute_layer_means
from circuits.evaluation import evaluate_circuit, evaluate_circuit_necessity


def load_client_models(exp_dir, round_num, num_clients, cfg):
    models = []
    for i in range(num_clients):
        path = os.path.join(exp_dir, "checkpoints", f"round_{round_num}", f"client_{i}_model.pt")
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Client model not found: {path}")
        model = get_model(cfg)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        model.eval()
        models.append(model)
    return models


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = [m(x) for m in self.models]
        return torch.stack(logits).mean(dim=0)


def distill(teacher, student, public_loader, cfg, temperature, epochs, lr):
    device = cfg.device
    teacher.eval()
    student.train()
    student.to(device)
    teacher.to(device)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    history = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for inputs, _ in public_loader:
            inputs = inputs.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

            student_logits = student(inputs)
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)

            loss = kl_loss_fn(student_log_soft, teacher_soft) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        history.append(avg_loss)
        print(f"  Distill Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    return student, history


def discover_all_circuits(model, public_loader, testloader, cfg, class_names):
    circuits = {}
    layer_means = None
    if cfg.use_mean_ablation:
        layer_means = compute_layer_means(model, public_loader, cfg)

    for tc in range(cfg.num_classes):
        if class_names and 0 <= tc < len(class_names):
            name = class_names[tc]
        else:
            name = str(tc)

        circ = discover_client_circuit(model, public_loader, tc, cfg, layer_means=layer_means)
        acc = evaluate_circuit(model, testloader, circ, tc, cfg, layer_means=layer_means)
        nec = evaluate_circuit_necessity(model, testloader, circ, tc, cfg)

        circuits[name] = {
            "active_nodes": circ,
            "metrics": {"accuracy": acc, "necessity": nec}
        }

        counts = {layer: len(idx) for layer, idx in circ.items()}
        print(f"    Class {name}: {counts} | Suff: {acc:.2f}% | Nec: {nec:.2f}%")

    return circuits


class EnsembleDistillExperiment(BaseExperiment):
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
            sys.exit("[ERROR] public_fraction is 0 — no public data to distill with. Set --public_fraction > 0.")

        round_num = args.round or cfg.num_rounds
        client_models = load_client_models(args.exp_dir, round_num, cfg.num_clients, cfg)

        # --- Step 1: Ensemble Evaluation ---
        self.print_header("Ensemble Evaluation (Logit Averaging)")
        ensemble = EnsembleModel(client_models)
        ensemble.eval()
        ensemble.to(device)
        ensemble_result = eval_full(ensemble, testloader, cfg)
        self.print_result("Ensemble Overall Acc:", f"{ensemble_result['overall']:.2f}%")
        for cls, acc in ensemble_result["per_class"].items():
            self.print_result(f"  Class {cls}:", f"{acc}%" if acc is not None else "N/A")

        # --- Step 2: Distillation ---
        self.print_header("Knowledge Distillation → Seed Model")
        student = get_model(cfg)

        init_path = os.path.join(args.exp_dir, "checkpoints", "initialization.pt")
        if os.path.exists(init_path):
            student.load_state_dict(torch.load(init_path, map_location=device))
            print("  Loaded original initialization weights as student seed.")
        else:
            print("  Using fresh random initialization for student.")

        student, loss_history = distill(
            teacher=ensemble,
            student=student,
            public_loader=public_loader,
            cfg=cfg,
            temperature=args.temperature,
            epochs=args.distill_epochs,
            lr=args.distill_lr
        )

        # --- Step 3: Distilled Model Evaluation ---
        self.print_header("Distilled Model Evaluation")
        student.eval()
        distilled_result = eval_full(student, testloader, cfg)
        self.print_result("Distilled Overall Acc:", f"{distilled_result['overall']:.2f}%")
        for cls, acc in distilled_result["per_class"].items():
            self.print_result(f"  Class {cls}:", f"{acc}%" if acc is not None else "N/A")

        # --- Step 4: Circuit Discovery on Distilled Model ---
        self.print_header("Circuit Discovery on Distilled Model")
        distilled_circuits = discover_all_circuits(student, public_loader, testloader, cfg, class_names)

        # --- Step 5: Save ---
        output_dir = os.path.join(args.exp_dir, "playground_results", "ensemble_distill")
        os.makedirs(output_dir, exist_ok=True)

        torch.save(student.state_dict(), os.path.join(output_dir, "distilled_model.pt"))

        results = {
            "ensemble_accuracy": ensemble_result,
            "distilled_accuracy": distilled_result,
            "distillation_loss_history": loss_history,
            "distilled_circuits": distilled_circuits,
            "config": {
                "temperature": args.temperature,
                "distill_epochs": args.distill_epochs,
                "distill_lr": args.distill_lr,
                "public_fraction": args.public_fraction,
                "round": round_num
            }
        }

        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to: {output_dir}")


def add_args(subparsers):
    p = subparsers.add_parser("ensemble_distill",
                              help="Ensemble client models and distill into a seed model via public data.")
    p.add_argument("--exp_dir", required=True, help="Path to completed experiment directory")
    p.add_argument("--round", type=int, default=None, help="Round to load client models from (default: last)")
    p.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    p.add_argument("--distill_epochs", type=int, default=50, help="Number of distillation epochs")
    p.add_argument("--distill_lr", type=float, default=0.0001, help="Distillation learning rate")
    p.add_argument("--public_fraction", type=float, default=0.2, help="Fraction of train set to use as public data")
    return p
