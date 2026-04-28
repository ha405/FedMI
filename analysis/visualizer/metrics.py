import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict
from analysis.visualizer.base import BaseVisualizer

class MetricsVisualizer(BaseVisualizer):
    """
    Visualizes performance metrics from metrics.json:
    1. Global vs Local model per-class accuracy
    2. Circuit metrics: Sufficiency (accuracy) and Necessity (inverse accuracy)
    """
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.metrics_data = None

    def load_metrics(self):
        path = os.path.join(self.output_dir, "metrics.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            self.metrics_data = json.load(f).get("rounds", [])
        return len(self.metrics_data) > 0

    def plot_circuit_metrics(self):
        """Plots circuit sufficiency (accuracy) and necessity across rounds for sampled classes."""
        if not self.metrics_data or "circuit_metrics" not in self.metrics_data[0]:
            return

        rounds = [r["round"] for r in self.metrics_data]
        
        # Pick 5 classes that have metrics
        sample_classes = set()
        for r in self.metrics_data:
            if "circuit_metrics" in r:
                sample_classes.update(r["circuit_metrics"]["global"].get("client_0", {}).keys())
        
        sample_classes = sorted(list(sample_classes))[:5]
        if not sample_classes:
            return

        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "ggplot")

        for cls_name in sample_classes:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # 1. Sufficiency (Accuracy)
            ax1.set_title(f"Circuit Metrics: {cls_name}")
            for target in ["local", "global"]:
                # Use client 0 as representative or mean across clients?
                # Let's plot mean across all clients that have this class
                vals = []
                for r in self.metrics_data:
                    round_vals = []
                    for cid, class_dict in r.get("circuit_metrics", {}).get(target, {}).items():
                        if cls_name in class_dict:
                            round_vals.append(class_dict[cls_name]["accuracy"])
                    vals.append(np.mean(round_vals) if round_vals else np.nan)
                
                ax1.plot(rounds, vals, "-o", markersize=3, label=f"{target.capitalize()} Sufficiency")
            
            ax1.set_ylabel("Accuracy (%)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Necessity
            for target in ["local", "global"]:
                vals = []
                for r in self.metrics_data:
                    round_vals = []
                    for cid, class_dict in r.get("circuit_metrics", {}).get(target, {}).items():
                        if cls_name in class_dict:
                            round_vals.append(class_dict[cls_name]["necessity"])
                    vals.append(np.mean(round_vals) if round_vals else np.nan)
                
                ax2.plot(rounds, vals, "-s", markersize=3, label=f"{target.capitalize()} Necessity")
            
            ax2.set_ylabel("Inverse Acc (%)")
            ax2.set_xlabel("Round")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            safe_name = cls_name.replace("/", "_").replace(" ", "_")
            self.save_plot(fig, f"circuit_metrics_{safe_name}.png")
            plt.close(fig)

    def plot_local_vs_global_accuracy(self):
        """Compares local model test accuracy vs global model accuracy on global test set."""
        if not self.metrics_data or "clients" not in self.metrics_data[0]:
            return

        rounds = [r["round"] for r in self.metrics_data]
        client_ids = sorted(self.metrics_data[0]["clients"].keys(), key=lambda x: int(x))
        
        # Plot mean client test accuracy vs global accuracy
        global_acc = [r["global_accuracy"] for r in self.metrics_data]
        
        mean_client_test_acc = []
        for r in self.metrics_data:
            c_accs = []
            for cid in client_ids:
                if "test_class_accuracy" in r["clients"][cid]:
                    # Mean accuracy across all classes seen by this client
                    vals = list(r["clients"][cid]["test_class_accuracy"].values())
                    if vals: c_accs.append(np.mean(vals))
            mean_client_test_acc.append(np.mean(c_accs) if c_accs else np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rounds, global_acc, "k--", linewidth=2, label="Global Model (Global Test)")
        ax.plot(rounds, mean_client_test_acc, "b-", linewidth=1.5, label="Local Models (Avg on Global Test)")
        
        ax.set_title("Generalization Gap: Local vs Global Models")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, "generalization_gap.png")
        plt.close(fig)

    def run(self):
        if not self.load_metrics():
            # Fallback to old behavior if needed, but we prefer metrics.json
            return
        
        print("[MetricsVisualizer] Generating performance and circuit plots...")
        self.plot_circuit_metrics()
        self.plot_local_vs_global_accuracy()
