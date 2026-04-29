import os
import json
import time
import csv
import platform
import subprocess


class MetricsTracker:
    """
    Collects per-round, per-client metrics for convergence analysis.
    
    Saves incrementally to metrics.json so partial results survive crashes.
    Can also export a flat CSV for LaTeX / plotting.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.rounds = []
        self._metrics_path = os.path.join(output_dir, "metrics.json")
        self._csv_path = os.path.join(output_dir, "metrics.csv")

    def load(self):
        """Reload previous metrics from disk (for resume after crash)."""
        if os.path.exists(self._metrics_path):
            try:
                with open(self._metrics_path) as f:
                    data = json.load(f)
                self.rounds = data.get("rounds", [])
                print(f"  [Metrics] Resumed {len(self.rounds)} rounds from {self._metrics_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [Metrics] Could not load existing metrics: {e}")
                self.rounds = []

    def log_round(self, round_num: int, global_acc: float, global_loss: float,
                  global_class_acc: dict = None,
                  client_train_metrics: dict = None,
                  client_test_metrics: dict = None,
                  round_circuits: dict = None,
                  round_time: float = None):
        entry = {
            "round": round_num + 1,
            "global_accuracy": round(global_acc, 4),
            "global_loss": round(global_loss, 6),
            "timestamp": time.time(),
        }

        if round_time is not None:
            entry["round_time_seconds"] = round(round_time, 1)

        if global_class_acc:
            entry["global_class_accuracy"] = {str(k): round(v, 4) for k, v in global_class_acc.items() if v is not None}

        if client_train_metrics:
            entry["clients"] = {}
            for cid, metrics in client_train_metrics.items():
                entry["clients"][str(cid)] = {
                    "train_loss":     round(metrics.get("loss", 0), 6),
                    "train_accuracy": round(metrics.get("accuracy", 0), 4),
                }

        if client_test_metrics:
            if "clients" not in entry:
                entry["clients"] = {}
            for cid, class_acc in client_test_metrics.items():
                cid_str = str(cid)
                if cid_str not in entry["clients"]:
                    entry["clients"][cid_str] = {}
                entry["clients"][cid_str]["test_class_accuracy"] = {str(k): v for k, v in class_acc.items()}

        if round_circuits:
            entry["circuit_metrics"] = {"local": {}, "global": {}}
            entry["circuit_sizes"]   = {"local": {}, "global": {}}
            for group, target in [("clients_local_model", "local"), ("clients_global_model", "global")]:
                for cid_str, class_dict in round_circuits.get(group, {}).items():
                    entry["circuit_metrics"][target][cid_str] = {}
                    entry["circuit_sizes"][target][cid_str]   = {}
                    for class_name, data in class_dict.items():
                        m = data.get("metrics", {})
                        entry["circuit_metrics"][target][cid_str][class_name] = {
                            "accuracy":  round(m.get("accuracy", 0), 4),
                            "necessity": round(m.get("necessity", 0), 4),
                        }
                        entry["circuit_sizes"][target][cid_str][class_name] = {
                            layer: len(nodes)
                            for layer, nodes in data.get("active_nodes", {}).items()
                        }

        self.rounds.append(entry)

    def save(self):
        """Save current state to metrics.json (overwrites — call after each round)."""
        data = {"rounds": self.rounds}
        with open(self._metrics_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_csv(self):
        """
        Export a flat CSV with columns:
        round, global_acc, global_loss, client_0_loss, client_0_acc, ...
        """
        if not self.rounds:
            return

        # Determine client IDs from first round that has them
        client_ids = []
        for r in self.rounds:
            if "clients" in r:
                client_ids = sorted(r["clients"].keys(), key=lambda x: int(x))
                break

        header = ["round", "global_accuracy", "global_loss"]
        for cid in client_ids:
            header.extend([f"client_{cid}_train_loss", f"client_{cid}_train_accuracy"])

        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for r in self.rounds:
                row = [r["round"], r["global_accuracy"], r["global_loss"]]
                for cid in client_ids:
                    client_data = r.get("clients", {}).get(cid, {})
                    row.append(client_data.get("train_loss", ""))
                    row.append(client_data.get("train_accuracy", ""))
                writer.writerow(row)


class RunInfo:
    """Captures system/environment metadata for reproducibility."""

    def __init__(self, output_dir: str, config):
        self.path = os.path.join(output_dir, "run_info.json")
        self.data = {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": None,
            "duration_seconds": None,
            "final_accuracy": None,
            "system": self._get_system_info(),
            "config": config.to_dict() if hasattr(config, 'to_dict') else str(config),
        }
        self._start = time.time()
        self.save()

    def _get_system_info(self):
        import torch
        info = {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["gpu_memory_gb"] = round(mem, 1)
        else:
            info["gpu"] = None

        # Git commit hash (best-effort)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            info["git_commit"] = commit
        except Exception:
            info["git_commit"] = None

        return info

    def finish(self, final_accuracy: float = None):
        self.data["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.data["duration_seconds"] = round(time.time() - self._start, 1)
        if final_accuracy is not None:
            self.data["final_accuracy"] = round(final_accuracy, 4)
        self.save()

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
