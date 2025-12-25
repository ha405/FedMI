import torch
import numpy as np
import random
import json
import os
import shutil

from config import Config
from models import get_model
from dataset import get_dataset, partition_iid, partition_dirichlet, get_dataloader, get_test_dataloader 
from train import analyze_iou
from fed import run_federated_training

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_experiment(alpha_value):
    config = Config()
    config.use_mean_ablation = False 
    config.partition_method = "dirichlet"
    config.dirichlet_alpha = alpha_value
    config.checkpoint_dir = f"./checkpoints/noniid_alpha_{alpha_value}"
    
    print(f"\n==================================================")
    print(f"STARTING EXPERIMENT: Alpha = {alpha_value}")
    print(f"Output Directory: {config.checkpoint_dir}")
    print(f"==================================================")

    if os.path.exists(config.checkpoint_dir):
        print(f"Cleaning existing directory: {config.checkpoint_dir}")
        try:
            shutil.rmtree(config.checkpoint_dir)
        except Exception as e:
            print(f"Warning: Could not delete dir {e}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    set_seed(config.seed)

    # 2. Prepare Data
    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes 

    # 3. Partition Data (The Non-IID Step)
    print(f"Partitioning data (Dirichlet Alpha={alpha_value})...")
    client_indices = partition_dirichlet(
        trainset, 
        config.num_clients, 
        alpha=config.dirichlet_alpha, 
        num_classes=config.num_classes
    )
    
    # Save partitions for analysis later
    partition_path = os.path.join(config.checkpoint_dir, "client_partitions.json")
    with open(partition_path, 'w') as f:
        json.dump(client_indices, f)
        
    client_dataloaders = [
        get_dataloader(trainset, indices, config) for indices in client_indices
    ]

    # 4. Initialize & Save Global Model
    global_model = get_model(config)
    torch.save(global_model.state_dict(), os.path.join(config.checkpoint_dir, "initialization.pt"))

    # 5. Run Training
    global_model, all_circuits = run_federated_training(
        global_model, client_dataloaders, testloader, config, class_names, alpha_value
    )

    # 6. Final Analysis
    print(f"\n--- Final Analysis for Alpha={alpha_value} ---")
    final_round_key = f"round_{config.num_rounds}"
    if final_round_key in all_circuits:
        final_global_circuits = all_circuits[final_round_key]["clients_global_model"]
        for client_id, class_circuits in final_global_circuits.items():
            formatted_circuits = {}
            for class_name, layers in class_circuits.items():
                formatted_circuits[class_name] = {
                    layer: set(indices) for layer, indices in layers.items()
                }
            analyze_iou(formatted_circuits)

if __name__ == "__main__":
    alphas_to_run = [10.0, 0.5, 0.2, 0.1, 0.05] 

    for alpha in alphas_to_run:
        run_experiment(alpha)