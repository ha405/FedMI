import os
import json
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.visualizer.base import BaseVisualizer
from core.config import ExperimentConfig
from core.models import get_model
from core.dataset import get_dataset, get_test_dataloader

class SensitivityVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.config = None
        self.testloader = None
        self.device = "cpu"
        
    def setup(self):
        """Loads config, dataset, and prepares for inference."""
        config_path = os.path.join(self.output_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"[{self.__class__.__name__}] Config not found at {config_path}. Skipping functional analysis.")
            return False
            
        self.config = ExperimentConfig.load(config_path)
        self.device = self.config.device
        
        # Load Dataset
        try:
            _, testset = get_dataset(self.config)
            self.testloader = get_test_dataloader(testset, self.config)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error loading dataset: {e}")
            return False
            
        return True

    def load_model(self, client_id=None, round_num=None, global_model=False):
        """Helper to load a model (Global or Client)."""
        model = get_model(self.config)
        model.to(self.device)
        model.eval()
        
        if round_num is None:
            # Default to last round if not specified
            sorted_rounds = sorted([r for r in self.data.keys() if r.startswith("round_")], 
                                 key=lambda x: int(x.split('_')[1]))
            if not sorted_rounds: return None
            round_key = sorted_rounds[-1]
            round_num = int(round_key.split('_')[1])
        
        # Paths
        if global_model:
            # Checkpoints might be in checkpoints/ or checkpoints/round_X/
            # Standard path: output_dir/checkpoints/global_model_round_X.pt
             path = os.path.join(self.output_dir, "checkpoints", f"global_model_round_{round_num}.pt")
        else:
             path = os.path.join(self.output_dir, "checkpoints", f"client_{client_id}_model_round_{round_num}.pt")
             if not os.path.exists(path):
                 # Try old structure just in case
                 path = os.path.join(self.output_dir, "checkpoints", f"round_{round_num}", f"client_{client_id}_model.pt")

        if not os.path.exists(path):
            # print(f"[{self.__class__.__name__}] Model not found: {path}")
            return None
            
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error loading state dict: {e}")
            return None
            
        return model

    def evaluate(self, model, target_classes=None):
        model.eval()
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                if target_classes:
                    # Only count specific classes
                    mask = torch.tensor([l.item() in target_classes for l in labels], device=self.device)
                    if mask.sum() == 0: continue
                    correct += (predicted[mask] == labels[mask]).sum().item()
                    total += mask.sum().item()
                    
                    for c in target_classes:
                         c_mask = (labels == c) & mask
                         if c_mask.sum() > 0:
                             class_correct[c] = class_correct.get(c, 0) + (predicted[c_mask] == labels[c_mask]).sum().item()
                             class_total[c] = class_total.get(c, 0) + c_mask.sum().item()
                else:
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
        
        acc = 100 * correct / total if total > 0 else 0.0
        return acc, class_correct, class_total

    def analyze_similarity_impact(self):
        """
        Replicates similarity.py:
        Identifies shared neurons and measures impact of injecting them into global model.
        """
        print(f"[{self.__class__.__name__}] Running Similarity Impact Analysis...")
        
        # 1. Identify Last Round
        sorted_rounds = sorted([r for r in self.data.keys() if r.startswith("round_")], 
                                key=lambda x: int(x.split('_')[1]))
        if not sorted_rounds: return
        last_round_key = sorted_rounds[-1]
        round_num = int(last_round_key.split('_')[1])
        
        # 2. Identify Shared Neurons
        local_data = self.data[last_round_key]["clients_local_model"]
        
        # Map Client -> Classes
        client_classes = {}
        for c_key in local_data:
            c_id = int(c_key.split('_')[1])
            client_classes[c_id] = list(local_data[c_key].keys())
        
        # Count neuron usage
        conv_layers = ['conv1', 'conv2', 'conv3']
        neuron_counts = {l: {} for l in conv_layers}
        
        for c_key, c_data in local_data.items():
            for class_data in c_data.values():
                active = class_data.get("active_nodes", {})
                for l in conv_layers:
                    for idx in active.get(l, []):
                        neuron_counts[l][idx] = neuron_counts[l].get(idx, 0) + 1
                        
        shared_indices = {l: set() for l in conv_layers}
        for l in conv_layers:
            for idx, count in neuron_counts[l].items():
                if count > 1:
                    shared_indices[l].add(idx)
                    
        total_shared = sum(len(s) for s in shared_indices.values())
        if total_shared == 0:
            print("  No shared neurons found.")
            return
        
        print(f"  Found {total_shared} shared neurons across all layers.")

        # 3. Load Global Model
        global_model = self.load_model(round_num=round_num, global_model=True)
        if not global_model: return
        
        # Baseline Eval
        _, _, base_class_stats = self.evaluate(global_model, target_classes=list(range(self.config.num_classes)))
        base_total_correct = sum(base_class_stats.values())
        print(f"  Baseline Global Accuracy (Total Correct): {base_total_correct}")
        
        # 4. Injection Analysis
        results = []
        
        for client_id in sorted(client_classes.keys()):
            client_model = self.load_model(client_id=client_id, round_num=round_num)
            if not client_model: continue
            
            # Inject
            test_model = copy.deepcopy(global_model)
            for l in conv_layers:
                target_idxs = list(shared_indices[l])
                if not target_idxs: continue
                
                idx_tensor = torch.tensor(target_idxs).to(self.device)
                
                gl = test_model.get_submodule(l)
                cl = client_model.get_submodule(l)
                
                with torch.no_grad():
                    gl.weight.data[idx_tensor] = cl.weight.data[idx_tensor].clone()
                    if gl.bias is not None:
                        gl.bias.data[idx_tensor] = cl.bias.data[idx_tensor].clone()
            
            # Evaluate
            _, _, class_stats = self.evaluate(test_model, target_classes=list(range(self.config.num_classes)))
            
            results.append({
                "client": client_id,
                "total_correct": sum(class_stats.values())
            })
            print(f"  Injection from Client {client_id}: {sum(class_stats.values())} correct")

        # 5. Plotting
        labels = ["Baseline"] + [f"Inj: C{r['client']}" for r in results]
        values = [base_total_correct] + [r['total_correct'] for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=['gray'] + ['skyblue']*len(results))
        ax.set_title(f"Impact of Shared Neuron Injection (Round {round_num})")
        ax.set_ylabel("Total Correct Predictions")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        self.save_plot(fig, "sensitivity_shared_neuron_impact.png")


    def run(self):
        if not self.load_data(): return
        if not self.setup(): return
        
        self.analyze_similarity_impact()

