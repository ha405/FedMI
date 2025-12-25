import torch
import numpy as np
import random
import json
import os
import shutil
from typing import Dict, List

from config import Config
from models import get_model
from dataset import get_dataset, partition_by_class, get_dataloader, get_test_dataloader 
from fed import run_federated_training

CLIENT_CLASS_MAP = {
    0: [0, 1, 2],       
    1: [3, 4, 5],       
    2: [6, 7, 8, 9]     
}

# CLASSES_TO_DISCOVER = {
#     0: [2], 
#     1: [5],
#     2: [6]   
# }
CLASSES_TO_DISCOVER = {
    0: [0, 1, 2], 
    1: [3, 4, 5],
    2: [6, 7, 8, 9]   
}
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    config = Config()
    config.num_clients = 3
    config.use_mean_ablation = False 
    config.partition_method = "by_class" 
    config.checkpoint_dir = "./checkpoints/controlled_noniid"
    
    print(f"\n==================================================")
    print(f"STARTING CONTROLLED NON-IID EXPERIMENT")
    print(f"Client Data Map: {CLIENT_CLASS_MAP}")
    print(f"Circuits to Discover: {CLASSES_TO_DISCOVER}")
    print(f"Output Directory: {config.checkpoint_dir}")
    print(f"==================================================")

    if os.path.exists(config.checkpoint_dir):
        print(f"Cleaning existing directory: {config.checkpoint_dir}")
        shutil.rmtree(config.checkpoint_dir, ignore_errors=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    set_seed(config.seed)

    trainset, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    class_names = trainset.classes 

    print("Partitioning data by assigned classes...")
    client_indices = partition_by_class(trainset, CLIENT_CLASS_MAP)
    
    partition_path = os.path.join(config.checkpoint_dir, "client_partitions.json")
    with open(partition_path, 'w') as f:
        json.dump(client_indices, f)
        
    client_dataloaders = [
        get_dataloader(trainset, indices, config) for indices in client_indices
    ]

    global_model = get_model(config)
    torch.save(global_model.state_dict(), os.path.join(config.checkpoint_dir, "initialization.pt"))
    global_model, all_circuits = run_federated_training(
        global_model, client_dataloaders, testloader, config, class_names,
        classes_to_discover_per_client=CLASSES_TO_DISCOVER
    )

    print("\n--- Experiment Complete ---")

if __name__ == "__main__":
    main()