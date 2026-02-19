import argparse
import sys
sys.dont_write_bytecode = True
import os
import json
import ast

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from core.config import ExperimentConfig
from experiments.runner import ExperimentRunner

def parse_args():
    parser = argparse.ArgumentParser(description="FedMI Unified Runner")
    
    # Config setup
    parser.add_argument("--config_file", type=str, help="Path to JSON config file to override defaults")
    
    # Overrides
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--partition", type=str, help="Partition method (iid, dirichlet, systematic_skew, manual)")
    parser.add_argument("--alpha", type=float, help="Dirichlet alpha")
    
    # Interactive/Complex overrides passed as strings (e.g. "{0: {0: 0.9}}")
    parser.add_argument("--skew_profile", type=str, help="Skew profile dict as string")
    parser.add_argument("--manual_allocation", type=str, help="Manual allocation dict as string")
    
    # Training / Federated Overrides
    parser.add_argument("--num_rounds", type=int, default=None, help="Number of federated rounds")
    parser.add_argument("--num_clients", type=int, default=None, help="Number of clients")
    parser.add_argument("--local_epochs", type=int, default=None, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--train_mode", type=str, help="Training mode (sparse/dense)")
    
    parser.add_argument("--resume", action="store_true", help="Resume experiment")
    
    return parser.parse_args()

def update_config(config, args):
    if args.config_file:
        with open(args.config_file, 'r') as f:
            json_config = json.load(f)
            for k, v in json_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)
    
    import datetime
    
    if args.output_dir: 
        config.output_dir = args.output_dir
        
    if args.device: config.device = args.device
    if args.seed: config.seed = args.seed
    if args.dataset: config.dataset_name = args.dataset
    if args.partition: config.partition_method = args.partition
    if args.alpha: config.dirichlet_alpha = args.alpha
    if args.resume: config.resume = True
    
    if args.skew_profile:
        config.skew_profile = ast.literal_eval(args.skew_profile)
    if args.manual_allocation:
        config.manual_allocation = ast.literal_eval(args.manual_allocation)
        
    if args.num_rounds: config.num_rounds = args.num_rounds
    if args.num_clients: config.num_clients = args.num_clients
    if args.local_epochs: config.local_epochs = args.local_epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.lr: config.learning_rate = args.lr
    if args.train_mode: config.train_mode = args.train_mode
    
    return config

def main():
    args = parse_args()
    config = ExperimentConfig()
    config = update_config(config, args)
    
    runner = ExperimentRunner(config)
    runner.setup()
    runner.run()

if __name__ == "__main__":
    main()
