import torch
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ExperimentConfig:
    # --- System ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    output_dir: str = "./checkpoints/experiment_run"
    
    # --- Data ---
    dataset_name: str = "MNIST"
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 0
    
    # --- Model ---
    model_name: str = "SimpleCNN"
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    num_classes: int = 10
    
    # --- Federation ---
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5
    
    # --- Partitioning ---
    partition_method: str = "iid"  # "iid", "dirichlet", "systematic_skew", "manual"
    dirichlet_alpha: float = 0.5
    # For "systematic_skew": {client_id: {class_id: probability}}
    # e.g. {0: {0: 0.9, 1: 0.1}, 1: {0: 0.1, 1: 0.9}}
    skew_profile: Optional[Dict[int, Dict[int, float]]] = None
    # For "manual" (aka "by_class" in original code): {client_id: [class_list]}
    manual_allocation: Optional[Dict[int, List[int]]] = None

    # --- Training ---
    train_mode: str = "sparse"  # "sparse", "dense"
    learning_rate: float = 0.001
    
    # --- Sparsity / Circuits ---
    target_sparsity: float = 0.99
    gate_lr: float = 0.1
    l0_lambda: float = 0.01
    discovery_steps: int = 200
    classes_to_analyze: List[int] = field(default_factory=lambda: [0, 1, 2, 7])
    # Which classes to discover per client (if None, uses classes_to_analyze)
    # {client_id: [class_list]}
    classes_to_discover_per_client: Optional[Dict[int, List[int]]] = None
    
    use_mean_ablation: bool = False
    
    # --- FedRS (Restricted Softmax) ---
    use_fedrs: bool = False
    fedrs_alpha: float = 0.4
    
    # --- Resume ---
    resume: bool = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str):
        import json
        if not os.path.exists(path):
            return cls() # Default
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
