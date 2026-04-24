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
        
        # --- Path Resolution for Folders ---
        self._resolve_folder_paths(args)
        
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
        max_s = getattr(args, 'max_samples', 2048)
        
        mode = getattr(args, 'mode', 'all')
        
        # --- 1. Pre-head Latent CKA ---
        if mode in ["all", "prehead", "latent"]:
            self.print_header("Step 1: Pre-head Latent CKA")
            print(f"Using {max_s} samples from {cfg_a.dataset_name} test set.\n")
            act_a = extract_prehead_latents(model_a, testloader, cfg_a, max_samples=max_s)
            act_b = extract_prehead_latents(model_b, testloader, cfg_b, max_samples=max_s)
            
            cka_val = linear_cka(act_a, act_b)
            self.print_result("Full Model Similarity (Latent)", f"{cka_val:.4f}")
            print("\n")

        # --- 2. Circuit CKA (Individual Class Comparison) ---
        if mode in ["all", "circuit"]:
            client_key_a = f"client_{getattr(args, 'client_a', 0)}"
            client_key_b = f"client_{getattr(args, 'client_b', 0)}"
            
            if client_key_a not in circuits_a:
                sys.exit(f"[ERROR] {client_key_a} not found in Side A circuits.")
            if client_key_b not in circuits_b:
                sys.exit(f"[ERROR] {client_key_b} not found in Side B circuits.")
                
            circ_data_a = circuits_a[client_key_a]
            circ_data_b = circuits_b[client_key_b]
            
            common_classes = sorted(list(set(circ_data_a.keys()).intersection(set(circ_data_b.keys()))))
            classes_arg = getattr(args, 'classes', None)
            if classes_arg:
                classes_to_compare = [str(c) for c in classes_arg if str(c) in common_classes]
            else:
                classes_to_compare = common_classes
                
            if not classes_to_compare:
                sys.exit("[ERROR] No overlapping classes found for circuit comparison.")
                
            self.print_header(f"Step 2: Circuit CKA Comparison ({client_key_a} vs {client_key_b})")
            print(f"Comparing classes: {', '.join(classes_to_compare)}\n")
            
            results = {}
            for cls in classes_to_compare:
                nodes_a = circ_data_a[cls].get("active_nodes", {})
                nodes_b = circ_data_b[cls].get("active_nodes", {})
                
                act_a = extract_circuit_activations(model_a, testloader, nodes_a, cfg_a, layer_name=getattr(args, 'layer', None), max_samples=max_s)
                act_b = extract_circuit_activations(model_b, testloader, nodes_b, cfg_b, layer_name=getattr(args, 'layer', None), max_samples=max_s)
                
                score = linear_cka(act_a, act_b)
                results[cls] = score
                self.print_result(f"Class {cls}", f"CKA = {score:.4f}")
                
            if results:
                avg = sum(results.values()) / len(results)
                print("-" * 60)
                self.print_result("Average Circuit CKA", f"{avg:.4f}")
                
            output_path = getattr(args, 'output', None)
            if output_path and results:
                self._save_heatmap(results, output_path, client_key_a, client_key_b)

        # --- 3. Per-Class Latent CKA (Unmasked) ---
        if mode in ["all", "per_class", "latent_class"]:
            self.print_header("Step 3: Per-Class Latent CKA (Unmasked)")
            print(f"Comparing raw model representations without masks.\n")
            
            # Determine classes to compare (reuse from Step 2 logic or default to Side A config)
            classes_arg = getattr(args, 'classes', None)
            if classes_arg:
                classes_to_compare = [str(c) for c in classes_arg]
            else:
                classes_to_compare = [str(c) for c in range(cfg_a.num_classes)]
            
            results_latent = {}
            for cls in classes_to_compare:
                act_a = extract_prehead_latents(model_a, testloader, cfg_a, target_label=cls, max_samples=max_s)
                act_b = extract_prehead_latents(model_b, testloader, cfg_b, target_label=cls, max_samples=max_s)
                
                if act_a.size(0) > 0 and act_b.size(0) > 0:
                    score = linear_cka(act_a, act_b)
                    results_latent[cls] = score
                    self.print_result(f"Class {cls} (Latent)", f"{score:.4f}")
                else:
                    print(f"  [skip] Class {cls}: Not enough samples.")
            
            if results_latent:
                avg_l = sum(results_latent.values()) / len(results_latent)
                print("-" * 60)
                self.print_result("Average Latent Per-Class CKA", f"{avg_l:.4f}")

    def _resolve_folder_paths(self, args):
        rnd_a = getattr(args, 'round_a', None) or getattr(args, 'round', None)
        rnd_b = getattr(args, 'round_b', None) or getattr(args, 'round', None)
        
        # Side A
        exp_a = getattr(args, 'exp_a', None)
        if exp_a:
            if not getattr(args, 'cfg_a', None):
                args.cfg_a = os.path.join(exp_a, "config.json")
            if rnd_a is not None:
                if not getattr(args, 'ckpt_a', None):
                    args.ckpt_a = os.path.join(exp_a, "checkpoints", f"checkpoint_round_{rnd_a}.pt")
                if not getattr(args, 'circ_a', None):
                    args.circ_a = os.path.join(exp_a, "circuits", f"circuits_round_{rnd_a}.json")
                    
        # Side B
        exp_b = getattr(args, 'exp_b', None)
        if exp_b:
            if not getattr(args, 'cfg_b', None):
                args.cfg_b = os.path.join(exp_b, "config.json")
            if rnd_b is not None:
                if not getattr(args, 'ckpt_b', None):
                    args.ckpt_b = os.path.join(exp_b, "checkpoints", f"checkpoint_round_{rnd_b}.pt")
                if not getattr(args, 'circ_b', None):
                    args.circ_b = os.path.join(exp_b, "circuits", f"circuits_round_{rnd_b}.json")

    def _prepare_config(self, cfg_path, model_name, num_classes, dataset):
        cfg = ExperimentConfig.load(cfg_path) if cfg_path else ExperimentConfig()
        if model_name: cfg.model_name = model_name
        if num_classes: cfg.num_classes = int(num_classes)
        if dataset: cfg.dataset_name = dataset
        return cfg

    def _load_model(self, cfg, ckpt_path):
        if not ckpt_path:
            sys.exit("[ERROR] No checkpoint path provided for model.")
        if not os.path.exists(ckpt_path):
            sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")
        model = get_model(cfg)
        state = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(cfg.device)
        model.eval()
        return model

    def _load_circuits(self, circ_path, source, round_key):
        if not circ_path:
            sys.exit("[ERROR] No circuit JSON path provided.")
        if not os.path.exists(circ_path):
            sys.exit(f"[ERROR] Circuit file not found: {circ_path}")
        with open(circ_path) as f:
            data = json.load(f)
        
        if round_key and str(round_key) in data:
            round_data = data[str(round_key)]
        elif isinstance(data, dict) and any(str(k).startswith("round_") for k in data.keys()):
            rounds = sorted([k for k in data.keys() if str(k).startswith("round_")], key=lambda r: int(r.split("_")[1]))
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
    p = subparsers.add_parser("cka", help="Local/Standalone CKA Comparison")
    # Folder-based
    p.add_argument("--exp_a", help="Path to experiment root directory A")
    p.add_argument("--exp_b", help="Path to experiment root directory B")
    p.add_argument("--round", type=int, help="Round number to automatically find weights/circuits")
    p.add_argument("--round_a", type=int, help="Round number for Side A")
    p.add_argument("--round_b", type=int, help="Round number for Side B")
    
    # Side A (Explicit Overrides)
    p.add_argument("--ckpt_a", help="Direct path to checkpoint A (.pt)")
    p.add_argument("--cfg_a", help="Direct path to config A (.json)")
    p.add_argument("--circ_a", help="Direct path to circuits A (.json)")
    p.add_argument("--model_a", help="Override: model name for Side A")
    p.add_argument("--classes_a", type=int, help="Override: num classes for Side A")
    p.add_argument("--dataset_a", help="Override: dataset name for Side A")
    
    # Side B (Explicit Overrides)
    p.add_argument("--ckpt_b", help="Direct path to checkpoint B")
    p.add_argument("--cfg_b", help="Direct path to config B")
    p.add_argument("--circ_b", help="Direct path to circuits B")
    p.add_argument("--model_b", help="Override: model name for Side B")
    p.add_argument("--classes_b", type=int, help="Override: num classes for Side B")
    p.add_argument("--dataset_b", help="Override: dataset name for Side B")

    # Shared Context
    p.add_argument("--mode", choices=["all", "circuit", "prehead", "latent"], default="all")
    p.add_argument("--client_a", type=int, default=0)
    p.add_argument("--client_b", type=int, default=0)
    p.add_argument("--source", choices=["local", "global"], default="local")
    p.add_argument("--round_key", help="Explicit round key in JSON (e.g. round_10)")
    p.add_argument("--classes", type=int, nargs="+", help="Specific class IDs to compare")
    p.add_argument("--layer", help="Target layer for activations")
    p.add_argument("--max_samples", type=int, default=2048)
    p.add_argument("--output", help="Path to save heatmap")
    return p
