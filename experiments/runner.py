import torch
import numpy as np
import random
import os
import copy
import json
import shutil
import sys

from core.dataset import get_dataset, get_test_dataloader, get_dataloader
from core.dataset import partition_iid, partition_dirichlet, partition_by_class, partition_systematic_skew
from core.dataset import get_classes_for_client
from core.models import get_model
from core.utils import load_latest_checkpoint, save_checkpoint, save_circuits_to_json
from federated.client import FederatedClient
from federated.server import FederatedServer
from circuits.evaluation import evaluate_detailed
from analysis.visualizer.class_distribution import ClassDistributionVisualizer

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        
    def set_seed(self):
        seed = self.config.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def setup(self):
        self.set_seed()
        
        # 1. Output Dir
        if not self.config.resume:
            if os.path.exists(self.config.output_dir):
                print(f"Cleaning existing directory: {self.config.output_dir}")
                shutil.rmtree(self.config.output_dir, ignore_errors=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create Subdirectories
        self.dirs = {
            "checkpoints": os.path.join(self.config.output_dir, "checkpoints"),
            "logs": os.path.join(self.config.output_dir, "logs"),
            "circuits": os.path.join(self.config.output_dir, "circuits"),
            "figures": os.path.join(self.config.output_dir, "figures"),
            "partitions": os.path.join(self.config.output_dir, "partitions")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
        # Save Config
        with open(os.path.join(self.config.output_dir, "config.json"), 'w') as f:
             # Basic serialization of config dataclass
             json.dump(self.config.__dict__, f, indent=4, default=str)
        
        # 2. Data
        print(f"Loading dataset: {self.config.dataset_name}")
        trainset, testset = get_dataset(self.config)
        self.testloader = get_test_dataloader(testset, self.config)
        # Determine class names based on config.num_classes.
        # If dataset provides class names and matches configured size, use them; otherwise use numeric labels 0..N-1
        configured_nc = getattr(self.config, 'num_classes', None)
        if hasattr(trainset, 'classes') and configured_nc is not None and len(trainset.classes) == configured_nc:
            self.class_names = list(trainset.classes)
        else:
            nc = configured_nc if configured_nc is not None else getattr(self.config, 'num_classes', 10)
            self.class_names = [str(i) for i in range(nc)]
        
        # 3. Partitioning
        print(f"Partitioning data using method: {self.config.partition_method}")
        if self.config.partition_method == "iid":
            client_indices = partition_iid(trainset, self.config.num_clients)
        elif self.config.partition_method == "dirichlet":
            client_indices = partition_dirichlet(trainset, self.config.num_clients, self.config.dirichlet_alpha, self.config.num_classes)
        elif self.config.partition_method == "manual": # Old "by_class"
            if not self.config.manual_allocation:
                raise ValueError("Manual allocation map required for 'manual' partition method.")
            client_indices = partition_by_class(trainset, self.config.manual_allocation)
        elif self.config.partition_method == "systematic_skew":
            if not self.config.skew_profile:
                raise ValueError("Skew profile required for 'systematic_skew' partition method.")
            client_indices = partition_systematic_skew(trainset, self.config.skew_profile, self.config.num_clients, self.config.num_classes)
        elif self.config.partition_method == "exact_amounts":
            from core.dataset import partition_exact_amounts
            if not self.config.exact_allocation:
                raise ValueError("exact_allocation map required for 'exact_amounts' partition method.")
            client_indices = partition_exact_amounts(trainset, self.config.exact_allocation, self.config.num_clients, self.config.num_classes)
        else:
            raise ValueError(f"Unknown partition method: {self.config.partition_method}")
            
        # Save partitions
        partition_path = os.path.join(self.dirs["partitions"], "client_partitions.json")
        with open(partition_path, 'w') as f:
            json.dump(client_indices, f)
            
        # Generate class distribution visualizations (especially useful for non-IID partitions)
        if self.config.partition_method in ["dirichlet", "systematic_skew", "manual"]:
            try:
                from core.dataset import get_labels
                labels = get_labels(trainset)
                
                visualizer = ClassDistributionVisualizer(self.config.output_dir, 
                                                        partition_method=self.config.partition_method)
                visualizer.run(dataset_name=self.config.dataset_name, 
                              class_names=list(self.class_names),
                              labels=labels)
            except Exception as e:
                print(f"Warning: Failed to generate class distribution plots: {e}")
                import traceback
                traceback.print_exc()
            
        # Auto-populate classes_to_discover_per_client from partition if not already set.
        # This ensures circuit discovery always targets the classes that actually exist in
        # each client's data, rather than relying on any hardcoded fallback.
        if self.config.classes_to_discover_per_client is None:
            self.config.classes_to_discover_per_client = {
                i: get_classes_for_client(trainset, indices)
                for i, indices in enumerate(client_indices)
            }
            print(f"[Runner] Auto-derived classes_to_discover_per_client: {self.config.classes_to_discover_per_client}")

        # 4. Clients
        self.clients = []
        for i, indices in enumerate(client_indices):
            dl = get_dataloader(trainset, indices, self.config)
            self.clients.append(FederatedClient(i, dl, self.config, self.class_names))
            
        # 5. Global Model
        self.global_model = get_model(self.config)
        torch.save(self.global_model.state_dict(), os.path.join(self.dirs["checkpoints"], "initialization.pt"))
        
        self.server = FederatedServer(self.global_model, self.testloader, self.config, self.class_names)
        
    def run(self):
        print(f"\n==================================================")
        print(f"STARTING EXPERIMENT: {self.config.output_dir}")
        print(f"Partition: {self.config.partition_method}")
        print(f"==================================================")
        
        all_circuits = {}
        start_round = 0
        
        if self.config.resume:
            start_round, all_circuits = load_latest_checkpoint(self.global_model, self.config)
            
        if start_round >= self.config.num_rounds:
            print("Training already completed!")
            return
            
        log_path = os.path.join(self.dirs["logs"], "training_log.txt")
        mode = "a" if self.config.resume else "w"
        
        with open(log_path, mode) as log_f:
            if not self.config.resume:
                log_f.write("=== Federated Training Log ===\n")
                
            for round_num in range(start_round, self.config.num_rounds):
                # Run Round
                round_circuits = self.server.orchestrate_round(round_num, self.clients, log_file=log_f)
                all_circuits[f"round_{round_num + 1}"] = round_circuits
                
                # Full Eval
                acc = evaluate_detailed(
                    self.global_model, self.testloader, self.config, 
                    log_file=log_f, 
                    class_names=self.class_names, 
                    title=f"Round {round_num + 1} Global Full Model Evaluation"
                )
                print(f"  Round {round_num + 1} Global Full Model Acc: {acc:.2f}%")
                
                # Periodic Save
                save_checkpoint(self.global_model, round_num + 1, all_circuits, self.config, self.dirs["checkpoints"])
                
                # Save Circuits JSON (Per Round)
                json_path_round = os.path.join(self.dirs["circuits"], f"circuits_round_{round_num + 1}.json")
                save_circuits_to_json(round_circuits, json_path_round) # Save just this round's data if possible, or all?
                
                # Save Master Circuits JSON (Cumulative)
                json_path_master = os.path.join(self.dirs["circuits"], "all_circuits.json")
                save_circuits_to_json(all_circuits, json_path_master)

        print("\n--- Experiment Complete ---")
        
        # --- Integration: Automatic Visualization ---
        try:
            print("\n=== Generating Automatic Visualizations ===")
            from analysis.visualizer.cross_eval import CrossEvalVisualizer
            from analysis.visualizer.heatmap import HeatmapVisualizer
            from analysis.visualizer.metrics import MetricsVisualizer
            from analysis.visualizer.consistency import ConsistencyVisualizer
            from analysis.visualizer.graph import GraphVisualizer
            from analysis.visualizer.sensitivity import SensitivityVisualizer

            CrossEvalVisualizer(self.config.output_dir).run()
            HeatmapVisualizer(self.config.output_dir).run()
            MetricsVisualizer(self.config.output_dir).run()
            ConsistencyVisualizer(self.config.output_dir).run()
            GraphVisualizer(self.config.output_dir).run()
            
            # Sensitivity Analysis (Functional)
            # Note: This loads models and runs inference, so it may be slower.
            SensitivityVisualizer(self.config.output_dir).run()
            
            print("=== Visualization Complete ===")
            
            # Copy Static Visualizer
            viz_src = os.path.join("analysis", "visualizer", "fl_visualizer.html")
            if os.path.exists(viz_src):
                viz_dst = os.path.join(self.config.output_dir, "visualizer.html")
                shutil.copy(viz_src, viz_dst)
                print(f"  [Visualizer] Copied interactive visualizer to {viz_dst}")
            
        except Exception as e:
            print(f"Error during visualization generation: {e}")
