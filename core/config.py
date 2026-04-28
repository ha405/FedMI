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
    partition_method: str = "iid"  # "iid", "dirichlet"
    dirichlet_alpha: float = 0.5
    # For "exact_amounts": {client_id: {class_id: exact_sample_count}}
    exact_allocation: Optional[Dict[int, Dict[int, int]]] = None

    # --- Training ---
    train_mode: str = "sparse"  # "sparse", "dense"
    learning_rate: float = 0.001
    
    # --- Sparsity / Circuits ---
    target_sparsity: float = 0.99
    gate_lr: float = 0.1
    l0_lambda: float = 0.01
    discovery_steps: int = 200
    

    
    # --- FedRS (Restricted Softmax) ---
    use_fedrs: bool = False
    fedrs_alpha: float = 0.4
    
    # --- Public Data (Server-Side) ---
    public_data_fraction: float = 0.0
    public_data_seed: int = 99

    # --- Discovery Pool ---
    discovery_samples_per_class: Optional[int] = None # If set, overrides fraction
    discovery_pool_fraction: float = 0.20             # Default 20% of data for discovery pool

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
        import dataclasses
        if not os.path.exists(path):
            return cls() # Default
        with open(path, 'r') as f:
            data = json.load(f)
        # Strip documentation-only keys (prefixed with "_") and any unknown fields
        # so config JSONs can carry inline comments without breaking construction.
        known_fields = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**data)

