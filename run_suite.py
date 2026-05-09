"""
End-to-end experiment suite for FedMI.

Phases (all run by default):
  1. train     — federated learning for all selected families
  2. analysis  — circuit consistency figures (inter/intra/local-global)
  3. probes    — linear probe training
  4. finetune  — head finetuning
  5. usae      — universal sparse autoencoder

Usage:
  python run_suite.py                              # all families, all phases
  python run_suite.py --families cifar_cnn         # one family only
  python run_suite.py --skip train                 # skip federated training
  python run_suite.py --config configs/cifar_cnn/iid.json   # single config, train only
  python run_suite.py --test                       # 1-round smoke test
  python run_suite.py --probe-dir results/CIFAR_CNN          # probes on one dir
  python run_suite.py --finetune-dir results/CIFAR_CNN       # finetune on one dir
  python run_suite.py --usae-dirs results/CIFAR_CNN results/CIFAR_CNN_005  # USAE on two dirs
"""

import argparse
import json
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

FAMILIES = {
    "fmnist_cnn": [
        "configs/fmnist_cnn/iid.json",
        "configs/fmnist_cnn/alpha_05.json",
        "configs/fmnist_cnn/alpha_02.json",
        "configs/fmnist_cnn/alpha_005.json",
    ],
    "fmnist_resnet": [
        "configs/fmnist_resnet/iid.json",
        "configs/fmnist_resnet/alpha_05.json",
        "configs/fmnist_resnet/alpha_02.json",
        "configs/fmnist_resnet/alpha_005.json",
    ],
    "cifar_cnn": [
        "configs/cifar_cnn/iid.json",
        "configs/cifar_cnn/alpha_05.json",
        "configs/cifar_cnn/alpha_02.json",
        "configs/cifar_cnn/alpha_005.json",
    ],
    "cifar_resnet": [
        "configs/cifar_resnet/iid.json",
        "configs/cifar_resnet/alpha_05.json",
        "configs/cifar_resnet/alpha_02.json",
        "configs/cifar_resnet/alpha_005.json",
    ],
}

ALL_PHASES = ["train", "analysis", "probes", "finetune", "usae"]


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {desc} — {elapsed:.0f}s")
    return result.returncode == 0


def _test_output_dir(cfg_path):
    """Read output_dir from a config file and append _test suffix."""
    with open(os.path.join(ROOT, cfg_path)) as f:
        return json.load(f)["output_dir"].rstrip("/").rstrip("\\") + "_test"


def _run_probes_one(results_dir, device, desc=""):
    return run(
        [sys.executable, "analysis/train_probes.py", "--results_dir", results_dir, "--device", device],
        f"Probes: {desc or results_dir}",
    )


def _run_finetune_one(results_dir, device, desc=""):
    return run(
        [sys.executable, "analysis/finetune_fc.py", "--results_dir", results_dir, "--device", device],
        f"Finetune: {desc or results_dir}",
    )


def phase_train(families, device, test=False, run_probes=False, run_finetune=False):
    failures = []
    for slug in families:
        for cfg in FAMILIES[slug]:
            cmd = [sys.executable, "main.py", "--config_file", cfg, "--device", device]
            out_dir = None
            if test:
                out_dir = _test_output_dir(cfg)
                cmd += ["--num_rounds", "1", "--output_dir", out_dir]
            ok = run(cmd, f"Train: {cfg}")
            if not ok:
                failures.append(cfg)
                continue
            if test and out_dir:
                if run_probes:
                    if not _run_probes_one(out_dir, device, cfg):
                        failures.append(f"probes:{cfg}")
                if run_finetune:
                    if not _run_finetune_one(out_dir, device, cfg):
                        failures.append(f"finetune:{cfg}")
    return failures


def phase_analysis(results_dir):
    failures = []
    for mode in ["--compare", "--local-global", "--intra-client"]:
        ok = run(
            [sys.executable, "analysis/circuit_consistency.py", mode, results_dir],
            f"Analysis: {mode}",
        )
        if not ok:
            failures.append(mode)
    return failures


def phase_probes(device, results_base="results"):
    ok = run(
        [sys.executable, "analysis/train_probes.py", "--all",
         "--results_base", results_base, "--device", device],
        "Linear probes — all families",
    )
    return [] if ok else ["train_probes"]


def phase_finetune(device, results_base="results"):
    ok = run(
        [sys.executable, "analysis/finetune_fc.py", "--all",
         "--results_base", results_base, "--device", device],
        "Head finetuning — all families",
    )
    return [] if ok else ["finetune_fc"]


# IID vs alpha=0.05 USAE pairs (dirname in results/)
_USAE_PAIRS = [
    ("CIFAR_CNN",     "CIFAR_CNN_005"),
    ("CIFAR_ResNet",  "CIFAR_ResNet_005"),
    ("FMNIST_CNN",    "FMNIST_CNN_005"),
    ("fmnist_resnet", "fmnist_resnet_005"),
]


def phase_usae(device, results_base="results", test=False):
    failures = []
    suffix = "_test" if test else ""
    for name_a, name_b in _USAE_PAIRS:
        dir_a = os.path.join(results_base, name_a + suffix)
        dir_b = os.path.join(results_base, name_b + suffix)
        cfg_a = os.path.join(dir_a, "config.json")
        cfg_b = os.path.join(dir_b, "config.json")
        if not os.path.exists(cfg_a) or not os.path.exists(cfg_b):
            print(f"  WARNING: skipping USAE pair {name_a + suffix} / {name_b + suffix} — config not found")
            continue
        ok = run(
            [sys.executable, "analysis/train_usae.py",
             "--config_a", cfg_a, "--config_b", cfg_b, "--device", device],
            f"USAE: {name_a + suffix} vs {name_b + suffix}",
        )
        if not ok:
            failures.append(f"{name_a}+{name_b}")
    return failures


def main():
    parser = argparse.ArgumentParser(description="FedMI end-to-end experiment suite")
    parser.add_argument(
        "--families", nargs="+", choices=list(FAMILIES.keys()), default=list(FAMILIES.keys()),
        help="Which model/dataset families to run (default: all)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run a single config file only (train phase, then exit). "
             "E.g. --config configs/cifar_cnn/iid.json",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=ALL_PHASES, default=[],
        help="Phases to skip",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device")
    parser.add_argument("--results", default="results", help="Results base directory")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 1 round, _test-suffixed dirs; "
                             "probes/finetune run inline after each config")
    parser.add_argument("--probe-dir", type=str, default=None, metavar="DIR",
                        help="Run linear probes on one results directory and exit")
    parser.add_argument("--finetune-dir", type=str, default=None, metavar="DIR",
                        help="Run FC finetuning on one results directory and exit")
    parser.add_argument("--usae-dirs", nargs=2, default=None, metavar=("DIR_A", "DIR_B"),
                        help="Run USAE on two results directories and exit")
    args = parser.parse_args()

    if args.probe_dir:
        sys.exit(0 if _run_probes_one(args.probe_dir, args.device) else 1)

    if args.finetune_dir:
        sys.exit(0 if _run_finetune_one(args.finetune_dir, args.device) else 1)

    if args.usae_dirs:
        dir_a, dir_b = args.usae_dirs
        ok = run(
            [sys.executable, "analysis/train_usae.py",
             "--config_a", os.path.join(dir_a, "config.json"),
             "--config_b", os.path.join(dir_b, "config.json"),
             "--device", args.device],
            f"USAE: {os.path.basename(dir_a)} vs {os.path.basename(dir_b)}",
        )
        sys.exit(0 if ok else 1)

    if args.config:
        ok = run(
            [sys.executable, "main.py", "--config_file", args.config, "--device", args.device],
            f"Train: {args.config}",
        )
        sys.exit(0 if ok else 1)

    phases = [p for p in ALL_PHASES if p not in args.skip]
    print(f"\nFedMI Suite{'  [TEST MODE]' if args.test else ''}")
    print(f"  Families : {args.families}")
    print(f"  Phases   : {phases}")
    print(f"  Device   : {args.device}")

    all_failures = {}
    t_start = time.time()

    inline_probes  = args.test and "probes"  in phases
    inline_finetune = args.test and "finetune" in phases

    if "train" in phases:
        f = phase_train(args.families, args.device, test=args.test,
                        run_probes=inline_probes, run_finetune=inline_finetune)
        if f: all_failures["train"] = f

    if "analysis" in phases:
        f = phase_analysis(args.results)
        if f: all_failures["analysis"] = f

    if "probes" in phases and not inline_probes:
        f = phase_probes(args.device, results_base=args.results)
        if f: all_failures["probes"] = f

    if "finetune" in phases and not inline_finetune:
        f = phase_finetune(args.device, results_base=args.results)
        if f: all_failures["finetune"] = f

    if "usae" in phases:
        f = phase_usae(args.device, results_base=args.results, test=args.test)
        if f: all_failures["usae"] = f

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Suite complete — {elapsed/60:.1f} min")
    if all_failures:
        print("  FAILURES:")
        for phase, items in all_failures.items():
            for item in items:
                print(f"    [{phase}] {item}")
    else:
        print("  All phases completed successfully.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
