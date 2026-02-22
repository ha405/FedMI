import os
import json
import matplotlib.pyplot as plt
import numpy as np
from analysis.visualizer.base import BaseVisualizer

class CrossEvalVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.config = self._load_config()
        self.class_names = None
    
    def _load_config(self):
        """Load config.json to get model parameters."""
        config_path = os.path.join(self.output_dir, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _load_class_names(self):
        """
        Load class names based on config.num_classes.
        Use numeric labels 0..N-1, derived from config.
        """
        if self.class_names is not None:
            return self.class_names
        
        num_classes = self.config.get('num_classes', 10)
        # Always use numeric labels based on config.num_classes
        self.class_names = [str(i) for i in range(num_classes)]
        
        return self.class_names
    
    def _get_class_display_name(self, class_idx):
        """Map class index to display name."""
        class_names = self._load_class_names()
        try:
            idx = int(class_idx)
            if 0 <= idx < len(class_names):
                return str(class_names[idx])
        except (ValueError, TypeError):
            pass
        return str(class_idx)

    def extract_metrics(self):
        round_keys = sorted(self.data.keys(), key=lambda r: int(r.split('_')[1]))
        extracted = {}

        for r in round_keys:
            round_num = int(r.split('_')[1])
            global_model_data = self.data[r].get("clients_global_model", {})
            
            for client, classes_dict in global_model_data.items():
                if client not in extracted:
                    extracted[client] = {}
                
                for class_name, class_data in classes_dict.items():
                    if class_name not in extracted[client]:
                        extracted[client][class_name] = {'rounds': [], 'cross_acc': [], 'global_acc': []}
                    
                    metrics = class_data.get("metrics", {})
                    cross_acc = metrics.get("local_mask_on_global_weights_acc", 0)
                    global_acc = metrics.get("accuracy", 0)
                    
                    extracted[client][class_name]['rounds'].append(round_num)
                    extracted[client][class_name]['cross_acc'].append(cross_acc)
                    extracted[client][class_name]['global_acc'].append(global_acc)
        return extracted

    def plot_cross_accuracy(self, extracted_data):
        clients = sorted(extracted_data.keys())
        if not clients: return

        fig, axes = plt.subplots(len(clients), 1, figsize=(10, 5 * len(clients)), sharex=True)
        if len(clients) == 1: axes = [axes]
        
        fig.suptitle("Cross-Evaluation: Accuracy of LOCAL MASK on GLOBAL WEIGHTS\n(Higher is Better = Circuit Location Preserved)", fontsize=16)
        
        for i, client in enumerate(clients):
            ax = axes[i]
            client_data = extracted_data[client]
            class_indices = sorted(client_data.keys())
            
            for class_idx in class_indices:
                class_display_name = self._get_class_display_name(class_idx)
                data = client_data[class_idx]
                rounds = client_data[class_idx]['rounds']
                accs = client_data[class_idx]['cross_acc']
                if not rounds: continue
                ax.plot(rounds, accs, marker='o', linestyle='-', linewidth=2, label=f"Class {class_display_name}")
            
            ax.set_title(f"{client}", fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(-5, 105)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right')
            
        axes[-1].set_xlabel("Federated Round")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        self.save_plot(fig, "cross_accuracy_absolute.png")

    def plot_drift_gap(self, extracted_data):
        clients = sorted(extracted_data.keys())
        if not clients: return

        fig, axes = plt.subplots(len(clients), 1, figsize=(10, 5 * len(clients)), sharex=True)
        if len(clients) == 1: axes = [axes]
        
        fig.suptitle("Circuit Drift Gap: (Global Mask Acc - Local Mask Acc)\n(Lower is Better = Less Physical Drift)", fontsize=16)
        
        for i, client in enumerate(clients):
            ax = axes[i]
            client_data = extracted_data[client]
            classes = sorted(client_data.keys())
            
            for cls in classes:
                rounds = client_data[cls]['rounds']
                cross = np.array(client_data[cls]['cross_acc'])
                glob = np.array(client_data[cls]['global_acc'])
                
                if not rounds: continue
                gap = glob - cross
                # map class index to display name
                disp = self._get_class_display_name(cls)
                ax.plot(rounds, gap, marker='s', linestyle='--', linewidth=2, label=f"Class {disp}")
                ax.axhline(0, color='black', linewidth=1, alpha=0.3)

            ax.set_title(f"{client}", fontsize=14, fontweight='bold')
            ax.set_ylabel("Accuracy Gap (%)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')
            
        axes[-1].set_xlabel("Federated Round")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        self.save_plot(fig, "cross_accuracy_drift_gap.png")

    def run(self):
        if not self.load_data(): return
        extracted = self.extract_metrics()
        self.plot_cross_accuracy(extracted)
        self.plot_drift_gap(extracted)
