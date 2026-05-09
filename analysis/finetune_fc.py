"""
FC Finetune Script
==================
Loads the latest global checkpoint from a results directory, freezes the
backbone, and finetunes the fc layer only on a small balanced (IID) subset
of the training data. Results are written to <results_dir>/finetuning_results/.

Method (matches FedMI_Optimal_FC_Finetune notebook):
  - Freeze all params except those whose name contains "fc"
  - AdamW(lr=1e-3, weight_decay=5e-4)
  - CosineAnnealingLR(T_max=EPOCHS), step once per epoch
  - CrossEntropyLoss

Per-task overrides:
  - EPOCHS = 20
  - Subset size N = 256 (balanced across num_classes; bs=64 for training,
    so 4 gradient steps per epoch)
  - Eval batch size = 256
  - Log every epoch

Usage:
  python finetune_fc.py --results_dir results/mnist_cnn_100_5e_10c_dirichlet_alpha_0.5
"""

import os
import sys
import json
import copy
import glob
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Make repo root importable so `from core...` works regardless of CWD
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.config import ExperimentConfig
from core.models import get_model
from core.dataset import get_dataset, get_labels


# ── Hyperparameters ───────────────────────────────────────────────
EPOCHS         = 5
SUBSET_SIZE    = 200
TRAIN_BS       = 16
EVAL_BS        = 256
LR             = 1e-3
WD             = 5e-4
SEED           = 42

def set_seed(s: int = SEED):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


def find_latest_checkpoint(checkpoints_dir: str) -> str:
    files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_round_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_round_*.pt files in {checkpoints_dir}")
    return max(files, key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))


def load_global_model(config, checkpoint_path: str) -> nn.Module:
    model = get_model(config)
    raw = torch.load(checkpoint_path, map_location=config.device)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(sd)
    return model


def make_iid_subset(trainset, num_classes: int, size: int, seed: int) -> list:
    """Balanced subset of `size` indices across the first `num_classes` classes."""
    rng = np.random.RandomState(seed)
    labels = get_labels(trainset)
    per_class = size // num_classes
    remainder = size % num_classes
    indices = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = per_class + (1 if c < remainder else 0)
        indices.extend(idx[:n].tolist())
    rng.shuffle(indices)
    return indices


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    correct = total = 0
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            for i in range(labels.size(0)):
                l = labels[i].item()
                if l < num_classes:
                    class_correct[l] += int(preds[i].item() == l)
                    class_total[l]   += 1
    overall = 100.0 * correct / total if total > 0 else 0.0
    per_class = {
        c: (100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0)
        for c in range(num_classes)
    }
    return overall, per_class


def train_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def finetune_fc(base_model, trainset, train_indices, testloader, config, class_names):
    """Freeze backbone, finetune fc only. AdamW + cosine schedule."""
    device = config.device
    num_classes = config.num_classes
    model = copy.deepcopy(base_model).to(device)

    # Freeze everything except fc
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)

    train_subset = Subset(trainset, train_indices)
    bs = min(len(train_subset), TRAIN_BS)
    pin = "cuda" in str(device)
    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=pin)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WD,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_epoch = 0
    best_pc    = {}
    history    = []

    sep = "-" * 58
    print(f"\n{'='*58}")
    print(f"  Finetuning global model  |  N = {len(train_subset)}")
    print(f"  Trainable params: {trainable}")
    print(f"  Frozen params: {len(frozen)} tensors")
    print(f"{'='*58}")
    print(f"  {'Epoch':>6}  {'Train':>8}  {'Test':>8}  {'Best':>8}")
    print(sep)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        tr_acc        = train_accuracy(model, train_loader, device)
        te_acc, te_pc = evaluate(model, testloader, device, num_classes)
        history.append((epoch, tr_acc, te_acc))

        if te_acc > best_acc:
            best_acc   = te_acc
            best_epoch = epoch
            best_pc    = te_pc

        marker = " <- best" if te_acc == best_acc else ""
        print(f"  {epoch:>6}  {tr_acc:>7.2f}%  {te_acc:>7.2f}%  {best_acc:>7.2f}%{marker}")

    print(sep)
    print(f"  Best test acc: {best_acc:.2f}%  (epoch {best_epoch})")
    print(f"  Per-class breakdown:")
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"    {name:>14}: {best_pc[c]:.2f}%")

    return {
        "best_acc"   : best_acc,
        "best_epoch" : best_epoch,
        "best_pc"    : best_pc,
        "final_acc"  : history[-1][2] if history else 0.0,
        "history"    : history,
    }


def serialize(d):
    if isinstance(d, dict):  return {str(k): serialize(v) for k, v in d.items()}
    if isinstance(d, list):  return [serialize(x) for x in d]
    if isinstance(d, tuple): return list(d)
    if isinstance(d, (np.floating, np.integer)): return float(d)
    return d


def get_class_names(testset, num_classes: int):
    base = testset.dataset if isinstance(testset, Subset) else testset
    if hasattr(base, "classes"):
        return [str(c) for c in base.classes[:num_classes]]
    return [str(c) for c in range(num_classes)]


MODEL_FAMILIES = [
    ("CIFAR-10 · CNN",        ["CIFAR_CNN",    "CIFAR_CNN_05",    "CIFAR_CNN_02",    "CIFAR_CNN_005"]),
    ("CIFAR-10 · ResNet",     ["CIFAR_ResNet", "CIFAR_ResNet_05", "CIFAR_ResNet_02", "CIFAR_ResNet_005"]),
    ("Fashion-MNIST · CNN",   ["FMNIST_CNN",   "FMNIST_CNN_05",   "FMNIST_CNN_02",   "FMNIST_CNN_005"]),
    ("Fashion-MNIST · ResNet",["fmnist_resnet","fmnist_resnet_05","fmnist_resnet_02","fmnist_resnet_005"]),
]


def run_one(results_dir: str, device: str) -> None:
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        print(f"  WARNING: directory not found, skipping: {results_dir}")
        return

    config_path = os.path.join(results_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"  WARNING: config.json not found, skipping: {results_dir}")
        return

    config = ExperimentConfig.load(config_path)
    config.device = device
    if not os.path.isabs(config.data_root):
        config.data_root = os.path.join(REPO_ROOT, config.data_root)

    set_seed(SEED)
    print(f"  Dataset : {config.dataset_name}  model={config.model_name}")

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")

    global_model = load_global_model(config, ckpt_path)

    trainset, testset = get_dataset(config)
    pin = "cuda" in device
    testloader = DataLoader(testset, batch_size=EVAL_BS, shuffle=False,
                            num_workers=0, pin_memory=pin)
    class_names = get_class_names(testset, config.num_classes)

    pre_acc, pre_pc = evaluate(global_model, testloader, config.device, config.num_classes)
    print(f"  Pre-finetune accuracy: {pre_acc:.2f}%")

    train_indices = make_iid_subset(trainset, config.num_classes,
                                    SUBSET_SIZE, seed=SEED + SUBSET_SIZE)
    result = finetune_fc(global_model, trainset, train_indices,
                         testloader, config, class_names)

    out_dir = os.path.join(results_dir, "finetuning_results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "results_dir"           : results_dir,
        "checkpoint"            : os.path.basename(ckpt_path),
        "dataset_name"          : config.dataset_name,
        "model_name"            : config.model_name,
        "num_classes"           : config.num_classes,
        "class_names"           : class_names,
        "settings": {
            "epochs"      : EPOCHS,
            "subset_size" : SUBSET_SIZE,
            "train_bs"    : TRAIN_BS,
            "eval_bs"     : EVAL_BS,
            "lr"          : LR,
            "weight_decay": WD,
            "optimizer"   : "AdamW",
            "scheduler"   : "CosineAnnealingLR",
            "seed"        : SEED,
        },
        "pre_finetune_acc"      : pre_acc,
        "pre_finetune_per_class": pre_pc,
        "result"                : result,
        "completed_at"          : datetime.now().isoformat(timespec="seconds"),
    }
    out_path = os.path.join(out_dir, "finetuning_results.json")
    with open(out_path, "w") as f:
        json.dump(serialize(payload), f, indent=2)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="FC-only finetune on the global model.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Single results sub-directory (e.g. results/CIFAR_CNN).")
    parser.add_argument("--all", action="store_true",
                        help="Run finetune for all 4 model/dataset families.")
    parser.add_argument("--results_base", type=str, default="results",
                        help="Base results directory used with --all (default: results).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}\n")

    if args.all:
        for display, dirnames in MODEL_FAMILIES:
            print(f"\n[{display}]")
            for dirname in dirnames:
                print(f"\n  -- {dirname}")
                run_one(os.path.join(args.results_base, dirname), device)
    else:
        if args.results_dir is None:
            parser.error("Provide --results_dir or use --all")
        # single-run mode: verbose output matching original behaviour
        results_dir = os.path.abspath(args.results_dir)
        if not os.path.isdir(results_dir):
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        config_path = os.path.join(results_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {results_dir}")
        config = ExperimentConfig.load(config_path)
        config.device = device
        if not os.path.isabs(config.data_root):
            config.data_root = os.path.join(REPO_ROOT, config.data_root)
        set_seed(SEED)
        print(f"Dataset      : {config.dataset_name}  (num_classes={config.num_classes})")
        print(f"Model        : {config.model_name}")
        print(f"Results dir  : {results_dir}")
        ckpt_dir = os.path.join(results_dir, "checkpoints")
        ckpt_path = find_latest_checkpoint(ckpt_dir)
        print(f"Checkpoint   : {os.path.basename(ckpt_path)}")
        global_model = load_global_model(config, ckpt_path)
        n_params = sum(p.numel() for p in global_model.parameters())
        print(f"Loaded global model ({n_params:,} params)")
        trainset, testset = get_dataset(config)
        pin = "cuda" in device
        testloader = DataLoader(testset, batch_size=EVAL_BS, shuffle=False,
                                num_workers=0, pin_memory=pin)
        class_names = get_class_names(testset, config.num_classes)
        print(f"Train pool   : {len(trainset)} samples")
        print(f"Test set     : {len(testset)} samples")
        print(f"Classes      : {class_names}")
        pre_acc, pre_pc = evaluate(global_model, testloader, config.device, config.num_classes)
        print(f"\nPre-finetune test accuracy: {pre_acc:.2f}%")
        train_indices = make_iid_subset(trainset, config.num_classes,
                                        SUBSET_SIZE, seed=SEED + SUBSET_SIZE)
        result = finetune_fc(global_model, trainset, train_indices,
                             testloader, config, class_names)
        out_dir = os.path.join(results_dir, "finetuning_results")
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "results_dir"           : results_dir,
            "checkpoint"            : os.path.basename(ckpt_path),
            "dataset_name"          : config.dataset_name,
            "model_name"            : config.model_name,
            "num_classes"           : config.num_classes,
            "class_names"           : class_names,
            "settings": {
                "epochs"      : EPOCHS,
                "subset_size" : SUBSET_SIZE,
                "train_bs"    : TRAIN_BS,
                "eval_bs"     : EVAL_BS,
                "lr"          : LR,
                "weight_decay": WD,
                "optimizer"   : "AdamW",
                "scheduler"   : "CosineAnnealingLR",
                "seed"        : SEED,
            },
            "pre_finetune_acc"      : pre_acc,
            "pre_finetune_per_class": pre_pc,
            "result"                : result,
            "completed_at"          : datetime.now().isoformat(timespec="seconds"),
        }
        out_path = os.path.join(out_dir, "finetuning_results.json")
        with open(out_path, "w") as f:
            json.dump(serialize(payload), f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
