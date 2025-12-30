import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    model_name: str = "SimpleCNN"
    conv_channels: List[int] = field(default_factory=lambda: [32, 64,128])
    num_classes: int = 10
    dataset_name: str = "MNIST" 
    
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 0

    learning_rate: float = 0.001
    target_sparsity: float = 0.997

    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5 

    gate_lr: float = 0.1
    l0_lambda: float = 0.01
    discovery_steps: int = 200
    classes_to_analyze: List[int] = field(default_factory=lambda: [0, 1, 2, 7])
    
    train_mode: str = "sparse" 
    use_mean_ablation: bool = False
    partition_method: str = "iid"
    checkpoint_dir: str = "./checkpoints/iid_run"
    resume: bool = False

    use_fedrs: bool = False
    fedrs_alpha: float = 0.4

@dataclass
class NonIIDConfig(Config):
    partition_method: str = "dirichlet"
    dirichlet_alpha: float = 0.5 
    checkpoint_dir: str = "./checkpoints/noniid_run"