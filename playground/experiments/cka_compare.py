import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import json

from .base import BaseExperiment
from core.config import ExperimentConfig
from core.models import get_model
from playground.core.loader import load_dataset
from circuits.cka import extract_circuit_activations, extract_prehead_latents, linear_cka

class CKACompareExperiment(BaseExperiment):
    def run(self):
        args = self.args
        
        # --- Side A Loading ---
        cfg_a = self._prepare_config(
            getattr(args, 'cfg_a', None), 
            getattr(args, 'model_a', None), 
            getattr(args, 'classes_a', None), 
            getattr(args, 'dataset_a', None)
        )
        model_a = self._load_model(cfg_a, getattr(args, 'ckpt_a', None))
        circuits_a = self._load_circuits(
            getattr(args, 'circ_a', None), 
            getattr(args, 'source', 'local'), 
            getattr(args, 'round_key', None)
        )
        
        # --- Side B Loading (Defaults to A if not provided) ---
        ckpt_b = getattr(args, 'ckpt_b', None)
        cfg_b_arg = getattr(args, 'cfg_b', None)
        circ_b = getattr(args, 'circ_b', None)
        
        if ckpt_b or cfg_b_arg or circ_b:
            cfg_b = self._prepare_config(
                cfg_b_arg or getattr(args, 'cfg_a', None), 
                getattr(args, 'model_b', None), 
                getattr(args, 'classes_b', None), 
                getattr(args, 'dataset_b', None)
            )
            model_b = self._load_model(cfg_b, ckpt_b)
            circuits_b = self._load_circuits(
                circ_b, 
                getattr(args, 'source', 'local'), 
                getattr(args, 'round_key', None)
            )
        else:
            cfg_b, model_b, circuits_b = cfg_a, model_a, circuits_a
            
        testloader = load_dataset(cfg_a)
        
        if getattr(args, 'mode', 'circuit') == "prehead":
            self.print_header("Pre-head Latent CKA")
            act_a = extract_prehead_latents(model_a, testloader, cfg_a, max_samples=getattr(args, 'max_samples', 1000))
            act_b = extract_prehead_latents(model_b, testloader, cfg_b, max_samples=getattr(args, 'max_samples', 1000))
            
            cka_val = linear_cka(act_a, act_b)
            self.print_result("Pre-head CKA", f"{cka_val:.4f}")
            
        elif getattr(args, 'mode', 'circuit') == "circuit":
            client_key_a = f"client_{getattr(args, 'client_a', 0)}"
            client_key_b = f"client_{getattr(args, 'client_b', 0)}"
            
            if client_key_a not in circuits_a:
                sys.exit(f"[ERROR] {client_key_a} not found in Side A circuits.")
            if client_key_b not in circuits_b:
                sys.exit(f"[ERROR] {client_key_b} not found in Side B circuits.")
                
            circ_data_a = circuits_a[client_key_a]
            circ_data_b = circuits_b[client_key_b]
            
            # Intersection of classes
            common_classes = sorted(list(set(circ_data_a.keys()).intersection(set(circ_data_b.keys()))))
            if args.classes:
                classes_to_compare = [str(c) for c in args.classes if str(c) in common_classes]
            else:
                classes_to_compare = common_classes
                
            if not classes_to_compare:
                sys.exit("[ERROR] No overlapping classes found between selected clients.")
                
            self.print_header(f"CKA Circuit Comparison: Side A ({client_key_a}) vs Side B ({client_key_b})")
            print(f"Comparing classes: {', '.join(classes_to_compare)}\n")
            
            results = {}
            for cls in classes_to_compare:
                nodes_a = circ_data_a[cls].get("active_nodes", {})
                nodes_b = circ_data_b[cls].get("active_nodes", {})
                
                act_a = extract_circuit_activations(model_a, testloader, nodes_a, cfg_a, layer_name=getattr(args, 'layer', None), max_samples=getattr(args, 'max_samples', 1000))
                act_b = extract_circuit_activations(model_b, testloader, nodes_b, cfg_b, layer_name=getattr(args, 'layer', None), max_samples=getattr(args, 'max_samples', 1000))
                
                score = linear_cka(act_a, act_b)
                results[cls] = score
                self.print_result(f"Class {cls}", f"CKA = {score:.4f}")
                
            if results:
                avg = sum(results.values()) / len(results)
                print("-" * 60)
                self.print_result("Average CKA", f"{avg:.4f}")
                
            output_path = getattr(args, 'output', None)
            if output_path and results:
                self._save_heatmap(results, output_path, client_key_a, client_key_b)

    def _prepare_config(self, cfg_path, model_name, num_classes, dataset):
        cfg = ExperimentConfig.load(cfg_path) if cfg_path else ExperimentConfig()
        if model_name: cfg.model_name = model_name
        if num_classes: cfg.num_classes = int(num_classes)
        if dataset: cfg.dataset_name = dataset
        # Ensure data_root defaults to local/colab expectations if not set
        if not cfg.data_root or cfg.data_root == "./data":
            # Check if we are in colab
            if os.path.exists("/content"):
                cfg.data_root = "/content/FedMI/data"
            else:
                cfg.data_root = "./data"
        return cfg

    def _load_model(self, cfg, ckpt_path):
        if not ckpt_path:
            sys.exit("[ERROR] No checkpoint path provided for model.")
        model = get_model(cfg)
        state = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(cfg.device)
        model.eval()
        return model

    def _load_circuits(self, circ_path, source, round_key):
        if not circ_path:
            sys.exit("[ERROR] No circuit JSON path provided.")
        with open(circ_path) as f:
            data = json.load(f)
        
        # Handle all_circuits.json nested structure
        if round_key and round_key in data:
            round_data = data[round_key]
        elif isinstance(data, dict) and any(k.startswith("round_") for k in data.keys()):
            # Detect latest round
            rounds = sorted([k for k in data.keys() if k.startswith("round_")], key=lambda r: int(r.split("_")[1]))
            round_data = data[rounds[-1]]
        else:
            round_data = data
            
        source_key = "clients_local_model" if source == "local" else "clients_global_model"
        return round_data.get(source_key, round_data)

    def _save_heatmap(self, results, path, label_a, label_b):
        try:
            import pandas as pd
            df = pd.DataFrame([list(results.values())], columns=list(results.keys()), index=["CKA"])
            plt.figure(figsize=(10, 2))
            sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1.0)
            plt.title(f"CKA: {label_a} vs {label_b}")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            plt.savefig(path, bbox_inches="tight")
            print(f"\n[info] Heatmap saved to {path}")
        except Exception as e:
            print(f"[warning] Heatmap failed: {e}")

def add_args(subparsers):
    p = subparsers.add_parser("cka", help="Standalone CKA Comparison (Manual Paths)")
    # Side A
    p.add_argument("--ckpt_a", required=True, help="Path to checkpoint A (.pt)")
    p.add_argument("--cfg_a", help="Path to config A (.json)")
    p.add_argument("--circ_a", help="Path to circuits A (.json)")
    p.add_argument("--model_a", help="Override: model name for Side A (e.g. ResNet)")
    p.add_argument("--classes_a", type=int, help="Override: num classes for Side A")
    p.add_argument("--dataset_a", help="Override: dataset name for Side A")
    
    # Side B
    p.add_argument("--ckpt_b", help="Path to checkpoint B (optional, defaults to side A)")
    p.add_argument("--cfg_b", help="Path to config B")
    p.add_argument("--circ_b", help="Path to circuits B")
    p.add_argument("--model_b", help="Override: model name for Side B")
    p.add_argument("--classes_b", type=int, help="Override: num classes for Side B")
    p.add_argument("--dataset_b", help="Override: dataset name for Side B")

    # Shared Context
    p.add_argument("--mode", choices=["circuit", "prehead"], default="circuit")
    p.add_argument("--client_a", type=int, default=0)
    p.add_argument("--client_b", type=int, default=0)
    p.add_argument("--source", choices=["local", "global"], default="local")
    p.add_argument("--round_key", help="Explicit round key in JSON (e.g. round_10)")
    p.add_argument("--classes", type=int, nargs="+", help="Specific class IDs to compare")
    p.add_argument("--layer", help="Target layer for activations")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--output", help="Path to save heatmap")
    return p
