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
    # If None (default), classes are auto-derived from the client's actual partition at runtime.
    # Set explicitly (e.g. [0, 1, 3]) to override and analyze specific classes regardless of partition.
    classes_to_analyze: Optional[List[int]] = None
    # Per-client class override: {client_id: [class_list]}.
    # If None, populated automatically from the partition during runner.setup().
    classes_to_discover_per_client: Optional[Dict[int, List[int]]] = None
    
    use_mean_ablation: bool = False
    
    # --- FedRS (Restricted Softmax) ---
    use_fedrs: bool = False
    fedrs_alpha: float = 0.4
    
    # --- Public Data (Server-Side) ---
    public_data_fraction: float = 0.0
    public_data_seed: int = 99

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
        data = {k: v for k, v in data.items() if k in known_fields and v is not None}
        return cls(**data)

