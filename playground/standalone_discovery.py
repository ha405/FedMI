import sys
import os
import argparse
import json
import torch

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ExperimentConfig
from core.dataset import get_dataloaders
from core.models import get_model
from circuits.discovery import discover_client_circuit, compute_layer_means
from circuits.evaluation import extract_sparse_connectivity, filter_connectivity_by_circuit, evaluate_circuit, evaluate_circuit_necessity

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Circuit Discovery")
    parser.add_argument("--model_name", type=str, default="SimpleCNN", help="Name of the model (e.g. SimpleCNN, vit_tiny_patch16_224)")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for discovery and evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights (optional)")
    parser.add_argument("--output_file", type=str, default="circuits_output/circuits.json", help="Relative path for saving the output json")
    parser.add_argument("--l0_lambda", type=float, default=0.01, help="Regularization strength for discovery")
    parser.add_argument("--discovery_steps", type=int, default=200, help="Number of steps for circuit gating optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize Configuration
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
    # Force single full-dataset dataloader for this standalone task
    config.partition_method = "iid"
    config.num_clients = 1
    
    client_loaders, global_testloader, class_names, num_classes = get_dataloaders(config)
    dataloader = client_loaders[0] # The entire dataset effectively
    config.num_classes = num_classes
    
    print(f"Initializing Model: {args.model_name}")
    model = None
    if "vit" in args.model_name.lower():
        try:
            import timm
            model = timm.create_model(args.model_name, pretrained=False, num_classes=num_classes)
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
    
    results = {}
    
    print("-" * 50)
    for tc in range(num_classes):
        c_name = class_names[tc] if (class_names and tc < len(class_names)) else str(tc)
        print(f"Discovering circuit for class: \033[96m{c_name} (ID: {tc})\033[0m")
        
        # 1. Discover Circuit Nodes
        circ = discover_client_circuit(model, dataloader, tc, config, layer_means=layer_means)
        
        # 2. Extract Functional Graph
        func_conn = filter_connectivity_by_circuit(physical_conn, circ)
        
        # 3. Evaluate Metrics
        acc = evaluate_circuit(model, global_testloader, circ, tc, config, layer_means=layer_means)
        necessity_acc = evaluate_circuit_necessity(model, global_testloader, circ, tc, config)
        
        print(f"  > Accuracy (Circuit Alone): {acc:.2f}%")
        print(f"  > Necessity (Model w/o Circuit): {necessity_acc:.2f}%")
        print("-" * 50)
        
        results[c_name] = {
            "active_nodes": circ,
            "connectivity": func_conn,
            "metrics": {
                "accuracy": acc,
                "necessity": necessity_acc
            }
        }
        
    print(f"\nSaving circuit output to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
