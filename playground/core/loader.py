import os
import sys
import contextlib

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.config import ExperimentConfig
from core.models import get_model
from core.dataset import get_dataset, get_test_dataloader


def load_config(exp_dir: str) -> ExperimentConfig:
    path = os.path.join(exp_dir, "config.json")
    return ExperimentConfig.load(path)


def load_model(exp_dir: str, cfg: ExperimentConfig, round_num=None):
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if round_num is not None:
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_round_{round_num}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[loader] Round {round_num} not found, falling back to last.")
            round_num = None

    if round_num is None:
        candidates = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_round_")],
            key=lambda f: int(f.split("_")[-1].replace(".pt", ""))
        )
        if not candidates:
            sys.exit(f"[ERROR] No checkpoints in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, candidates[-1])

    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    model.to(cfg.device)
    print(f"[loader] Model loaded: {ckpt_path}")
    return model


def load_dataset(cfg: ExperimentConfig):
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        _, testset = get_dataset(cfg)
    return get_test_dataloader(testset, cfg)


def load_circuits(exp_dir: str, source: str = "local", round_key: str = "last") -> dict:
    path = os.path.join(exp_dir, "circuits", "all_circuits.json")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] No circuits at {path}")
    with open(path) as f:
        data = json.load(f)

    rounds = sorted(data.keys(), key=lambda r: int(r.split("_")[1]))
    if round_key == "last":
        round_key = rounds[-1]
    elif round_key not in data:
        sys.exit(f"[ERROR] Round '{round_key}' not found. Available: {rounds}")

    round_data = data[round_key]
    source_key = "clients_local_model" if source == "local" else "clients_global_model"

    if source_key not in round_data:
        sys.exit(f"[ERROR] Source '{source}' not in {round_key}")

    print(f"[loader] Circuits loaded: {round_key} / {source}")
    return round_data[source_key]
