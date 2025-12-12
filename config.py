import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    model_name: str = "SimpleCNN"
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_classes: int = 10

    dataset_name: str = "CIFAR10"
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2

    epochs: int = 25
    learning_rate: float = 0.001

    target_sparsity: float = 0.95

    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 1

    gate_lr: float = 0.1
    l0_lambda: float = 0.1
    discovery_steps: int = 200
    classes_to_analyze: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    train_mode: str = "sparse"  # 'sparse' or 'dense'

    checkpoint_dir: str = "./checkpoints"
