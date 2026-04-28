"""
FedMI Experiment Orchestrator
===========================
Runs the experiment matrix using main.py with CLI overrides.

Matrix: {MNIST, CIFAR10} x {SimpleCNN, ResNet} x {IID, Non-IID}

Usage:
  python scripts/reproduce.py                    # Run all baseline experiments
  python scripts/reproduce.py --filter mnist     # Run only MNIST experiments
  python scripts/reproduce.py --rounds 10        # Run with fewer rounds for testing
  python scripts/reproduce.py --alpha 0.3        # Set custom Dirichlet alpha
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime


# ── Experiment Matrix ──────────────────────────────────────────────

EXPERIMENTS = [
    {"name": "mnist_simplecnn_iid",    "dataset": "MNIST",   "model": "SimpleCNN", "partition": "iid"},
    {"name": "mnist_simplecnn_niid",   "dataset": "MNIST",   "model": "SimpleCNN", "partition": "dirichlet"},
    {"name": "mnist_resnet_iid",       "dataset": "MNIST",   "model": "ResNet",    "partition": "iid"},
    {"name": "mnist_resnet_niid",      "dataset": "MNIST",   "model": "ResNet",    "partition": "dirichlet"},
    {"name": "cifar10_simplecnn_iid",  "dataset": "CIFAR10", "model": "SimpleCNN", "partition": "iid"},
    {"name": "cifar10_simplecnn_niid", "dataset": "CIFAR10", "model": "SimpleCNN", "partition": "dirichlet"},
    {"name": "cifar10_resnet_iid",     "dataset": "CIFAR10", "model": "ResNet",    "partition": "iid"},
    {"name": "cifar10_resnet_niid",    "dataset": "CIFAR10", "model": "ResNet",    "partition": "dirichlet"},
]


def build_command(exp, args):
    """Construct CLI command for main.py at root."""
    output_dir = f"./results/{exp['name']}"
    cmd = [
        sys.executable, "main.py",
        "--dataset", exp["dataset"],
        "--model", exp["model"],
        "--partition", exp["partition"],
        "--output_dir", output_dir,
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.rounds:
        cmd += ["--num_rounds", str(args.rounds)]
    if exp["partition"] == "dirichlet":
        cmd += ["--alpha", str(args.alpha)]
    return cmd, output_dir


def run_experiment(exp, args, idx, total):
    """Execute a single experiment run."""
    cmd, output_dir = build_command(exp, args)
    cmd_str = " ".join(cmd)

    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}] RUNNING: {exp['name']}")
    print(f"  Dataset: {exp['dataset']} | Model: {exp['model']} | Partition: {exp['partition']}")
    print(f"  Command: {cmd_str}")
    print(f"{'='*60}")

    if args.dry_run:
        print("  [Dry Run] Skipping...")
        return {"name": exp["name"], "status": "dry_run", "duration": 0, "output_dir": output_dir}

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    start_time = time.time()

    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"Experiment: {exp['name']}\n")
            log_file.write(f"Command: {cmd_str}\n")
            log_file.write(f"Started: {datetime.now().isoformat()}\n\n")
            log_file.flush()
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(f"  {line}", end="")
                log_file.write(line)
            process.wait()
            
            duration = time.time() - start_time
            log_file.write(f"\nDuration: {duration:.1f}s | Exit Code: {process.returncode}\n")
        
        status = "success" if process.returncode == 0 else f"failed ({process.returncode})"
    except Exception as e:
        duration = time.time() - start_time
        status = f"error: {str(e)}"

    print(f"\n  [{exp['name']}] Status: {status} | Duration: {duration:.1f}s")
    return {"name": exp["name"], "status": status, "duration": round(duration, 1), "output_dir": output_dir}


def print_summary(results):
    """Display final experiment matrix results."""
    print(f"\n{'='*60}")
    print(f"  FINAL EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<30} {'Status':<15} {'Accuracy':>10} {'Time':>10}")
    print("-" * 65)
    
    for r in results:
        accuracy = "N/A"
        if r["status"] == "success":
            metrics_path = os.path.join(r["output_dir"], "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path) as f:
                        data = json.load(f)
                    if data.get("rounds"):
                        accuracy = f"{data['rounds'][-1]['global_accuracy']:.2f}%"
                except Exception:
                    pass
        print(f"  {r['name']:<30} {r['status']:<15} {accuracy:>10} {r['duration']:.0f}s")


def run_post_analysis(results):
    """Generate aggregate comparison plots."""
    successful_dirs = [r["output_dir"] for r in results if r["status"] == "success"]
    if len(successful_dirs) > 1:
        print("\n=== Generating Comparison Analysis ===")
        try:
            from analysis.plot_results import plot_comparison
            plot_comparison(successful_dirs)
            print("Comparison plots generated in ./results/figures/")
        except Exception as e:
            print(f"Post-analysis visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="FedMI Experiment Orchestrator")
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name substring")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID partitions")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of communication rounds")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.filter:
        experiments = [e for e in experiments if args.filter.lower() in e["name"].lower()]

    if not experiments:
        print(f"No experiments matched filter: '{args.filter}'")
        return

    print(f"\nFedMI Experiment Orchestrator | {len(experiments)} experiments queued")
    results = []
    for idx, exp in enumerate(experiments, 1):
        results.append(run_experiment(exp, args, idx, len(experiments)))

    print_summary(results)
    run_post_analysis(results)

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
