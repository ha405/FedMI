import argparse
import sys
import os
import json
import time
import torch
import numpy as np
import random
import shutil

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())

from core.config import ExperimentConfig
from core.dataset import get_dataset, get_test_dataloader, get_dataloader, partition_iid, partition_dirichlet, get_classes_for_client, get_labels
from torch.utils.data import Subset
from core.models import get_model
from core.utils import load_latest_checkpoint, save_checkpoint, save_circuits_to_json
from core.data_cache import EvaluationCache
from core.metrics import MetricsTracker, RunInfo
from federated.client import FederatedClient
from federated.server import FederatedServer
from circuits.evaluation import evaluate_detailed_with_loss


class ExperimentRunner:
    def __init__(self, config):
        self.config = config

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True

    def setup(self):
        self._set_seed()

        if not self.config.resume and os.path.exists(self.config.output_dir):
            shutil.rmtree(self.config.output_dir, ignore_errors=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.dirs = {
            "checkpoints": os.path.join(self.config.output_dir, "checkpoints"),
            "logs":        os.path.join(self.config.output_dir, "logs"),
            "circuits":    os.path.join(self.config.output_dir, "circuits"),
            "figures":     os.path.join(self.config.output_dir, "figures"),
            "partitions":  os.path.join(self.config.output_dir, "partitions"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=4, default=str)

        print(f"Loading dataset: {self.config.dataset_name}")
        trainset, testset = get_dataset(self.config)
        self.testloader = get_test_dataloader(testset, self.config)

        if hasattr(trainset, "classes") and len(trainset.classes) == self.config.num_classes:
            self.class_names = list(trainset.classes)
        else:
            self.class_names = [str(i) for i in range(self.config.num_classes)]

        self.evaluation_cache = EvaluationCache(self.testloader, self.config.device, self.config.num_classes)

        if self.config.partition_method == "iid":
            client_indices = partition_iid(trainset, self.config.num_clients)
        elif self.config.partition_method == "dirichlet":
            client_indices = partition_dirichlet(trainset, self.config.num_clients, self.config.dirichlet_alpha, self.config.num_classes)
        else:
            raise ValueError(f"Unknown partition method: {self.config.partition_method}")

        with open(os.path.join(self.dirs["partitions"], "client_partitions.json"), "w") as f:
            json.dump(client_indices, f)

        # Build balanced global discovery pool: Fraction (20%) or fixed count per class
        all_labels = get_labels(trainset)
        class_to_discovery_idx = {}
        for k in range(self.config.num_classes):
            idx_k = np.where(all_labels == k)[0]
            if len(idx_k) > 0:
                if self.config.discovery_samples_per_class is not None:
                    n_samples = min(len(idx_k), self.config.discovery_samples_per_class)
                else:
                    n_samples = int(len(idx_k) * self.config.discovery_pool_fraction)
                
                picked = np.random.choice(idx_k, n_samples, replace=False)
                class_to_discovery_idx[k] = picked.tolist()

        global_discovery_idx = [idx for indices in class_to_discovery_idx.values() for idx in indices]

        self.clients = []
        for i, indices in enumerate(client_indices):
            if not indices:
                continue
            self.clients.append(FederatedClient(
                i,
                get_dataloader(trainset, indices, self.config),
                get_dataloader(trainset, global_discovery_idx, self.config) if global_discovery_idx else None,
                self.config,
                self.class_names,
            ))

        self.global_model = get_model(self.config)
        torch.save(self.global_model.state_dict(), os.path.join(self.dirs["checkpoints"], "initialization.pt"))

        self.server = FederatedServer(self.global_model, self.config, self.class_names, self.evaluation_cache)

        self.tracker = MetricsTracker(self.config.output_dir)
        if self.config.resume:
            self.tracker.load()
        self.run_info = RunInfo(self.config.output_dir, self.config)

    def run(self):
        print(f"\n{'='*50}\nEXPERIMENT: {self.config.output_dir}\n{'='*50}")

        all_circuits = {}
        start_round = 0
        if self.config.resume:
            start_round, all_circuits = load_latest_checkpoint(self.global_model, self.config)

        if start_round >= self.config.num_rounds:
            print("Training already complete.")
            return

        log_path = os.path.join(self.dirs["logs"], "training_log.txt")
        final_acc = 0.0

        with open(log_path, "a" if self.config.resume else "w") as log_f:
            for round_num in range(start_round, self.config.num_rounds):
                t0 = time.time()
                round_circuits, client_metrics, client_test_metrics = self.server.orchestrate_round(round_num, self.clients, log_file=log_f)
                all_circuits[f"round_{round_num + 1}"] = round_circuits

                # Global model evaluated on ALL classes for benchmark
                acc, loss, class_acc = evaluate_detailed_with_loss(
                    self.global_model, self.evaluation_cache, self.config,
                    log_file=log_f, class_names=self.class_names,
                    title=f"Round {round_num + 1} Global Eval",
                    active_classes=None, 
                )
                final_acc = acc
                print(f"  Round {round_num + 1} | Acc: {acc:.2f}% | Loss: {loss:.4f} | {time.time() - t0:.1f}s")

                self.tracker.log_round(
                    round_num, acc, loss,
                    global_class_acc=class_acc,
                    client_train_metrics=client_metrics,
                    client_test_metrics=client_test_metrics,
                    round_circuits=round_circuits,
                )
                self.tracker.save()
                save_checkpoint(self.global_model, round_num + 1, all_circuits, self.config, self.dirs["checkpoints"])
                save_circuits_to_json(round_circuits, os.path.join(self.dirs["circuits"], f"circuits_round_{round_num + 1}.json"))
                save_circuits_to_json(all_circuits, os.path.join(self.dirs["circuits"], "all_circuits.json"))

        self.tracker.save_csv()
        self.run_info.finish(final_accuracy=final_acc)
        self._run_analysis()

    def _run_analysis(self):
        try:
            from analysis.visualizer.consistency import ConsistencyVisualizer
            from analysis.plot_results import plot_convergence

            ConsistencyVisualizer(self.config.output_dir).run()
            plot_convergence(self.config.output_dir)

            viz_src = os.path.join("analysis", "visualizer", "fl_visualizer.html")
            if os.path.exists(viz_src):
                shutil.copy(viz_src, os.path.join(self.config.output_dir, "visualizer.html"))
        except Exception as e:
            import traceback
            print(f"Analysis error: {e}")
            traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="FedMI")
    parser.add_argument("--config_file",   type=str)
    parser.add_argument("--device",        type=str)
    parser.add_argument("--seed",          type=int)
    parser.add_argument("--output_dir",    type=str)
    parser.add_argument("--dataset",       type=str)
    parser.add_argument("--partition",     type=str)
    parser.add_argument("--alpha",         type=float)
    parser.add_argument("--model",         type=str)
    parser.add_argument("--num_rounds",    type=int)
    parser.add_argument("--num_clients",   type=int)
    parser.add_argument("--local_epochs",  type=int)
    parser.add_argument("--batch_size",    type=int)
    parser.add_argument("--lr",            type=float)
    parser.add_argument("--train_mode",    type=str)
    parser.add_argument("--resume",        action="store_true")
    return parser.parse_args()


def update_config(config, args):
    if args.config_file:
        with open(args.config_file) as f:
            for k, v in json.load(f).items():
                if hasattr(config, k):
                    setattr(config, k, v)

    mapping = {
        "output_dir": "output_dir", "device": "device", "seed": "seed",
        "dataset": "dataset_name", "partition": "partition_method", "alpha": "dirichlet_alpha",
        "model": "model_name", "num_rounds": "num_rounds", "num_clients": "num_clients",
        "local_epochs": "local_epochs", "batch_size": "batch_size",
        "lr": "learning_rate", "train_mode": "train_mode",
    }
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, cfg_name, val)

    if args.resume:
        config.resume = True
    return config


def main():
    args = parse_args()
    config = update_config(ExperimentConfig(), args)
    runner = ExperimentRunner(config)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
