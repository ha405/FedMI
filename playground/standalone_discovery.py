import sys
import os
import argparse
import json
import torch
import numpy as np

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader
from core.models import get_model
from torch.utils.data import DataLoader
from circuits.discovery import discover_client_circuit, compute_layer_means
from circuits.evaluation import extract_sparse_connectivity, filter_connectivity_by_circuit, evaluate_circuit, evaluate_circuit_necessity
from core.utils import save_circuits_to_json

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Circuit Discovery")
    parser.add_argument("--model_name", type=str, default="SimpleCNN", help="Name of the model (e.g. SimpleCNN, vit_tiny_patch16_224)")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for discovery and evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights (optional)")
    parser.add_argument("--output_file", type=str, default="circuits_output/circuits.json", help="Relative path in playground for saving the output json")
    parser.add_argument("--l0_lambda", type=float, default=0.01, help="Regularization strength for discovery")
    parser.add_argument("--discovery_steps", type=int, default=200, help="Number of steps for circuit gating optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize Configuration natively
    config = ExperimentConfig()
    config.model_name = args.model_name
    config.dataset_name = args.dataset
    config.batch_size = args.batch_size
    config.device = args.device
    config.l0_lambda = args.l0_lambda
    config.discovery_steps = args.discovery_steps
    
    # Resolve output directory exactly in playground/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, args.output_file)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Dataset: {config.dataset_name}")
    # Use actual valid repo functions:
    trainset, testset = get_dataset(config)
    global_testloader = get_test_dataloader(testset, config)
    
    # Standalone mode: we use the entire dataset directly without federated partitioning
    dataloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    # Try getting class names via valid native dataset logic
    configured_nc = getattr(config, 'num_classes', 10)
    if hasattr(trainset, 'classes') and len(trainset.classes) == configured_nc:
        class_names = list(trainset.classes)
    else:
        class_names = [str(i) for i in range(configured_nc)]
        
    print(f"Initializing Model: {args.model_name}")
    model = None
    if "vit" in args.model_name.lower():
        try:
            import timm
            model = timm.create_model(args.model_name, pretrained=False, num_classes=configured_nc)
            model = model.to(config.device)
        except ImportError:
            print("timm not found. Please pip install timm to use ViT models.")
            sys.exit(1)
        except Exception as e:
            print(f"Failed to load timm model: {e}")
            sys.exit(1)
    else:
        model = get_model(config)
        
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
        
    print("Calculating Layer Means (for ablation tests)...")
    config.use_mean_ablation = True
    layer_means = compute_layer_means(model, dataloader, config)
    
    print("Extracting physical connectivity...")
    physical_conn = extract_sparse_connectivity(model)
    
    # Emulate the exact nested structure generated natively by the Runner ("clients_local_model", "client_0", etc)
    round_circuits = {
        "clients_local_model": {
            "client_0": {}
        },
        "clients_global_model": {}
    }
    
    cg_circs = {}
    
    print("-" * 50)
    for tc in range(configured_nc):
        c_name = class_names[tc]
        print(f"Discovering circuit for class: \033[96m{c_name} (ID: {tc})\033[0m")
        
        # 1. Discover Circuit Nodes natively
        circ = discover_client_circuit(model, dataloader, tc, config, layer_means=layer_means)
        
        # 2. Extract Functional Graph
        func_conn = filter_connectivity_by_circuit(physical_conn, circ)
        
        # 3. Evaluate Metrics natively
        acc = evaluate_circuit(model, global_testloader, circ, tc, config, layer_means=layer_means)
        necessity_acc = evaluate_circuit_necessity(model, global_testloader, circ, tc, config)
        
        print(f"  > Accuracy (Circuit Alone): {acc:.2f}%")
        print(f"  > Necessity (Model w/o Circuit): {necessity_acc:.2f}%")
        print("-" * 50)
        
        cg_circs[c_name] = {
            "active_nodes": circ,
            "connectivity": func_conn,
            "metrics": {
                "accuracy": acc,
                "necessity": necessity_acc
            }
        }
    
    # Align format exactly
    round_circuits["clients_local_model"]["client_0"] = cg_circs
    all_circuits = {
        "standalone_model_discovery": round_circuits
    }
    
    print(f"\nSaving circuit output natively to: {output_path}")
    save_circuits_to_json(all_circuits, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
