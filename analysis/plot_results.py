"""
Convergence Plot Generator
===========================
Reads metrics.json from an experiment directory and generates convergence curves.

Usage:
  python scripts/plot_convergence.py results/mnist_simplecnn_iid
  python scripts/plot_convergence.py results/mnist_simplecnn_iid results/mnist_simplecnn_niid  # comparison
"""

import os
import sys
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(exp_dir):
    """Load metrics.json from an experiment directory."""
    path = os.path.join(exp_dir, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metrics.json found in {exp_dir}")
    with open(path) as f:
        return json.load(f)


def plot_convergence(exp_dir, save=True):
    """Generate all convergence plots for a single experiment."""
    data = load_metrics(exp_dir)
    rounds_data = data["rounds"]
    
    if not rounds_data:
        print(f"No rounds data in {exp_dir}")
        return

    figures_dir = os.path.join(exp_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    rounds = [r["round"] for r in rounds_data]
    global_acc = [r["global_accuracy"] for r in rounds_data]
    global_loss = [r["global_loss"] for r in rounds_data]

    # Try to load experiment name from config
    config_path = os.path.join(exp_dir, "config.json")
    exp_name = os.path.basename(exp_dir)
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            exp_name = f"{cfg.get('dataset_name', '')} {cfg.get('model_name', '')} {cfg.get('partition_method', '')}"
        except Exception:
            pass

    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')

    # ── Plot 1: Global Accuracy ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, global_acc, 'b-o', markersize=2, linewidth=1.5, label='Global Accuracy')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Global Model Accuracy — {exp_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if save:
        fig.savefig(os.path.join(figures_dir, "convergence_global_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 2: Global Loss ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, global_loss, 'r-o', markersize=2, linewidth=1.5, label='Global Loss')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Global Model Loss — {exp_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if save:
        fig.savefig(os.path.join(figures_dir, "convergence_global_loss.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 3 & 4: Per-Client Training Curves ──
    # Collect client data
    client_ids = set()
    for r in rounds_data:
        if "clients" in r:
            client_ids.update(r["clients"].keys())
    client_ids = sorted(client_ids, key=lambda x: int(x))

    if client_ids:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(client_ids), 1)))

        # Per-client training accuracy
        fig, ax = plt.subplots(figsize=(10, 6))
        for ci, cid in enumerate(client_ids):
            c_acc = []
            c_rounds = []
            for r in rounds_data:
                if "clients" in r and cid in r["clients"]:
                    c_rounds.append(r["round"])
                    c_acc.append(r["clients"][cid].get("train_accuracy", 0))
            if c_acc:
                ax.plot(c_rounds, c_acc, '-', color=colors[ci], linewidth=1.2,
                        label=f'Client {cid}', alpha=0.8)
        ax.plot(rounds, global_acc, 'k--', linewidth=2, label='Global (test)', alpha=0.9)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Per-Client Training Accuracy — {exp_name}', fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(os.path.join(figures_dir, "convergence_client_accuracy.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Per-client training loss
        fig, ax = plt.subplots(figsize=(10, 6))
        for ci, cid in enumerate(client_ids):
            c_loss = []
            c_rounds = []
            for r in rounds_data:
                if "clients" in r and cid in r["clients"]:
                    c_rounds.append(r["round"])
                    c_loss.append(r["clients"][cid].get("train_loss", 0))
            if c_loss:
                ax.plot(c_rounds, c_loss, '-', color=colors[ci], linewidth=1.2,
                        label=f'Client {cid}', alpha=0.8)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'Per-Client Training Loss — {exp_name}', fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(os.path.join(figures_dir, "convergence_client_loss.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Plot 5: Client accuracy variance band ──
        fig, ax = plt.subplots(figsize=(10, 6))
        all_client_acc = []
        for r in rounds_data:
            vals = []
            if "clients" in r:
                for cid in client_ids:
                    if cid in r["clients"]:
                        vals.append(r["clients"][cid].get("train_accuracy", 0))
            all_client_acc.append(vals if vals else [0])

        means = [np.mean(v) for v in all_client_acc]
        stds = [np.std(v) for v in all_client_acc]
        means, stds = np.array(means), np.array(stds)

        ax.plot(rounds, means, 'b-', linewidth=1.5, label='Client Mean')
        ax.fill_between(rounds, means - stds, means + stds, alpha=0.2, color='blue', label='± 1 Std Dev')
        ax.plot(rounds, global_acc, 'k--', linewidth=2, label='Global (test)', alpha=0.9)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Client Accuracy Variance — {exp_name}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(os.path.join(figures_dir, "convergence_client_variance.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  [Convergence] Plots saved to {figures_dir}")


def plot_comparison(exp_dirs, save_dir="results"):
    """Generate comparison plots across multiple experiments."""
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')

    fig_acc, ax_acc = plt.subplots(figsize=(12, 7))
    fig_loss, ax_loss = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(exp_dirs), 1)))

    for i, exp_dir in enumerate(exp_dirs):
        try:
            data = load_metrics(exp_dir)
        except FileNotFoundError:
            print(f"  Skipping {exp_dir} — no metrics.json")
            continue

        rounds_data = data["rounds"]
        if not rounds_data:
            continue

        label = os.path.basename(exp_dir)
        rounds = [r["round"] for r in rounds_data]
        global_acc = [r["global_accuracy"] for r in rounds_data]
        global_loss = [r["global_loss"] for r in rounds_data]

        ax_acc.plot(rounds, global_acc, '-', color=colors[i], linewidth=1.5, label=label)
        ax_loss.plot(rounds, global_loss, '-', color=colors[i], linewidth=1.5, label=label)

    ax_acc.set_xlabel('Round', fontsize=12)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
    ax_acc.set_title('Global Accuracy Comparison', fontsize=14)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    fig_acc.savefig(os.path.join(save_dir, "figures", "comparison_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_acc)

    ax_loss.set_xlabel('Round', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title('Global Loss Comparison', fontsize=14)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    fig_loss.savefig(os.path.join(save_dir, "figures", "comparison_loss.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_loss)

    print(f"  [Comparison] Plots saved to {os.path.join(save_dir, 'figures')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate convergence plots from FedMI experiment results")
    parser.add_argument("exp_dirs", nargs="+", help="One or more experiment directories")
    args = parser.parse_args()

    if len(args.exp_dirs) == 1:
        plot_convergence(args.exp_dirs[0])
    else:
        # Generate individual plots for each
        for d in args.exp_dirs:
            try:
                plot_convergence(d)
            except Exception as e:
                print(f"  Error plotting {d}: {e}")
        # Generate comparison
        plot_comparison(args.exp_dirs)
