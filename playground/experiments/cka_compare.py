import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

from .base import BaseExperiment
from playground.core.loader import load_config, load_model, load_dataset, load_circuits
from circuits.cka import extract_circuit_activations, extract_prehead_latents, linear_cka

class CKACompareExperiment(BaseExperiment):
    def run(self):
        args = self.args
        from core.config import ExperimentConfig
        import json
        
        # --- Helper to load configurations ---
        if getattr(args, 'cfg_a', None):
            cfg_a = ExperimentConfig.load(args.cfg_a)
        elif args.exp_a:
            cfg_a = load_config(args.exp_a)
        else:
            sys.exit("[ERROR] Must provide either --exp_a or --cfg_a")

        if getattr(args, 'cfg_b', None):
            cfg_b = ExperimentConfig.load(args.cfg_b)
        elif args.exp_b:
            cfg_b = load_config(args.exp_b)
        elif getattr(args, 'cfg_a', None) or args.exp_a:
            cfg_b = cfg_a  # Fallback to A
        else:
            sys.exit("[ERROR] Must provide either --exp_b or --cfg_b, or rely on A fallback")
            
        # --- Helper to load models ---
        from core.models import get_model
        def _load_direct_model(ckpt_path, cfg):
            model = get_model(cfg)
            checkpoint = torch.load(ckpt_path, map_location=cfg.device)
            # Handle if the checkpoint is just state_dict or nested under 'model_state_dict'
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(cfg.device)
            return model

        if getattr(args, 'ckpt_a', None):
            model_a = _load_direct_model(args.ckpt_a, cfg_a)
        elif args.exp_a:
            model_a = load_model(args.exp_a, cfg_a, round_num=args.round)
        else:
            sys.exit("[ERROR] Secondary fallback failed for model A.")
            
        if getattr(args, 'ckpt_b', None):
            model_b = _load_direct_model(args.ckpt_b, cfg_b)
        elif args.exp_b:
            model_b = load_model(args.exp_b, cfg_b, round_num=args.round)
        elif getattr(args, 'ckpt_a', None) or args.exp_a:
            model_b = model_a # Fallback to A
        else:
            sys.exit("[ERROR] Secondary fallback failed for model B.")
            
        # --- Helper to load circuits ---
        def _load_direct_circuits(circ_path, source, round_key):
            with open(circ_path) as f:
                data = json.load(f)
            # If the user uploaded an isolated round JSON, wrap it or handle it
            if round_key and round_key in data:
                round_data = data[round_key]
            else:
                round_data = data # assume it's directly the round data or we use latest
                rounds = sorted([k for k in data.keys() if k.startswith("round_")], key=lambda r: int(r.split("_")[1])) if isinstance(data, dict) else []
                if round_key == "last" and rounds:
                    round_data = data[rounds[-1]]
            
            source_key = "clients_local_model" if source == "local" else "clients_global_model"
            if source_key in round_data:
                return round_data[source_key]
            return round_data # Assume they just uploaded the local models directly

        def get_circuits_a():
            if getattr(args, 'circ_a', None): return _load_direct_circuits(args.circ_a, args.source, args.round_key or "last")
            elif args.exp_a: return load_circuits(args.exp_a, source=args.source, round_key=args.round_key or "last")
            return None
            
        def get_circuits_b():
            if getattr(args, 'circ_b', None): return _load_direct_circuits(args.circ_b, args.source, args.round_key or "last")
            elif args.exp_b: return load_circuits(args.exp_b, source=args.source, round_key=args.round_key or "last")
            elif getattr(args, 'circ_a', None) or args.exp_a: return get_circuits_a()
            return None
        
        # Ensure data_root is valid (especially for Colab/Kaggle)
        if not cfg_a.data_root:
            cfg_a.data_root = "./data"
        if not cfg_b.data_root:
            cfg_b.data_root = "./data"

        # Test split used to generate matching samples for CKA
        # We can just use testloader from exp_a as long as datasets match
        testloader = load_dataset(cfg_a)
        
        if args.mode == "prehead":
            self.print_header(f"Pre-head Latent CKA")
            
            act_a = extract_prehead_latents(model_a, testloader, cfg_a, max_samples=args.max_samples)
            act_b = extract_prehead_latents(model_b, testloader, cfg_b, max_samples=args.max_samples)
            
            cka_val = linear_cka(act_a, act_b)
            self.print_result(f"Pre-head CKA", f"{cka_val:.4f}")
            
            if args.output:
                # Can't plot heatmap for a single value really, but we'll save the result
                print(f"Skipping heatmap for single value (Prehead CKA: {cka_val:.4f})")
                
        elif args.mode == "circuit":
            circuits_a = get_circuits_a()
            circuits_b = get_circuits_b()
            
            client_key_a = f"client_{args.client_a}"
            client_key_b = f"client_{args.client_b}"
            
            if client_key_a not in circuits_a:
                sys.exit(f"[ERROR] {client_key_a} not found in side A circuits.")
                
            if client_key_b not in circuits_b:
                sys.exit(f"[ERROR] {client_key_b} not found in side B circuits.")
                
            client_circ_a = circuits_a[client_key_a]
            client_circ_b = circuits_b[client_key_b]
            
            # Determine which classes to evaluate
            avail_classes_a = set(client_circ_a.keys())
            avail_classes_b = set(client_circ_b.keys())
            intersect_classes = list(avail_classes_a.intersection(avail_classes_b))
            
            if args.classes:
                classes_to_compare = [str(c) for c in args.classes]
                classes_to_compare = [c for c in classes_to_compare if c in intersect_classes]
            else:
                classes_to_compare = sorted(intersect_classes)
                
            if not classes_to_compare:
                sys.exit("[WARNING] No overlapping classes found between the two selected clients.")
                
            self.print_header(f"CKA Circuit Comparison: Side A ({client_key_a}) vs Side B ({client_key_b})")
            print(f"  Comparing classes: {', '.join(classes_to_compare)}")
            print("-" * 60)
            
            results = {}
            for class_name in classes_to_compare:
                circ_a = client_circ_a[class_name].get("active_nodes", {})
                circ_b = client_circ_b[class_name].get("active_nodes", {})
                
                if not circ_a or not circ_b:
                    self.print_result(f"Class {class_name}:", "Missing circuit nodes, skipped.")
                    continue
                
                act_a = extract_circuit_activations(model_a, testloader, circ_a, cfg_a, layer_name=args.layer, max_samples=args.max_samples)
                act_b = extract_circuit_activations(model_b, testloader, circ_b, cfg_b, layer_name=args.layer, max_samples=args.max_samples)
                
                cka_val = linear_cka(act_a, act_b)
                results[class_name] = cka_val
                
                self.print_result(f"Class {class_name}:", f"CKA = {cka_val:.4f}")
                
            if results:
                avg_cka = sum(results.values()) / len(results)
                print("-" * 60)
                self.print_result(f"Average:", f"CKA = {avg_cka:.4f}")
                
            if args.output and results:
                try:
                    import pandas as pd
                    # Creates a 1D heatmap
                    df = pd.DataFrame([list(results.values())], columns=list(results.keys()), index=["CKA Score"])
                    plt.figure(figsize=(10, 2))
                    sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1.0)
                    plt.title(f"Cross-client Circuit CKA\\n{client_key_a} vs {client_key_b}")
                    output_path = os.path.abspath(args.output)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    plt.savefig(output_path, bbox_inches="tight")
                    print(f"\nSaved CKA heatmap to: {args.output}")
                except Exception as e:
                    print(f"Failed to generate heatmap: {e}")

def add_args(subparsers):
    p = subparsers.add_parser("cka", help="Compare CKA of circuit activations or latents across clients or runs.")
    p.add_argument("--exp_a", default=None, help="First experiment directory")
    p.add_argument("--exp_b", default=None, help="Second experiment directory (optional)")
    p.add_argument("--ckpt_a", default=None, help="Direct path to checkpoint A")
    p.add_argument("--ckpt_b", default=None, help="Direct path to checkpoint B")
    p.add_argument("--cfg_a", default=None, help="Direct path to config.json A")
    p.add_argument("--cfg_b", default=None, help="Direct path to config.json B")
    p.add_argument("--circ_a", default=None, help="Direct path to circuits json A")
    p.add_argument("--circ_b", default=None, help="Direct path to circuits json B")
    p.add_argument("--client_a", type=int, default=0, help="Client ID for exp_a")
    p.add_argument("--client_b", type=int, default=0, help="Client ID for exp_b")
    p.add_argument("--mode", choices=["circuit", "prehead"], default="circuit", help="Comparison mode")
    p.add_argument("--classes", type=int, nargs="+", help="Class IDs to compare")
    p.add_argument("--source", choices=["local", "global"], default="local", help="Source for circuits (local/global)")
    p.add_argument("--round", type=int, default=None, help="Checkpoint round (default: last)")
    p.add_argument("--round_key", default=None, help="Circuit round key (default: last)")
    p.add_argument("--layer", type=str, default=None, help="Layer to extract from (for circuit mode). Default: last before head")
    p.add_argument("--max_samples", type=int, default=1000, help="Max test samples to forward pass")
    p.add_argument("--output", type=str, help="Path to save heatmap")
    return p
