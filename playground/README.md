# Circuit Playground

Post-FedAvg circuit experiment toolkit. All commands run from the project root.

**Prerequisites:** a completed experiment directory with `config.json`, `circuits/all_circuits.json`, and `checkpoints/checkpoint_round_N.pt`.

---

## Structure

```
playground/
  core/
    loader.py       load config, model, dataset, circuits
    evaluator.py    eval_full, eval_sufficiency, eval_necessity
    stitcher.py     stitch_circuits (union / intersection / majority)
  experiments/
    base.py         BaseExperiment — inherit to add new experiments
    apply.py        ApplyCircuitExperiment
    stitch.py       StitchCircuitExperiment
    __init__.py     REGISTRY — maps command name → experiment class
  circuit_lab.py    unified CLI entry point
```

---

## Commands

### `apply` — Load circuits, apply to model, evaluate

Loads circuits for one client (local or global model weights) and evaluates
sufficiency (circuit alone) and necessity (circuit removed) on the test set.

```bash
python playground/circuit_lab.py apply \
    --exp_dir checkpoints/non_iid_pathological \
    --client_id 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--exp_dir` | required | Path to experiment output directory |
| `--client_id` | required | Which client's circuits to use |
| `--source` | `local` | `local` or `global` — which model's circuits |
| `--round` | last | Checkpoint round number |
| `--round_key` | last | Circuit round key, e.g. `round_5` |
| `--eval_modes` | `all` | `all`, `sufficient`, `necessary` (space-separated) |

```bash
# Use global model circuits
python playground/circuit_lab.py apply --exp_dir checkpoints/non_iid_pathological --client_id 0 --source global

# Only run sufficiency test
python playground/circuit_lab.py apply --exp_dir checkpoints/non_iid_pathological --client_id 0 --eval_modes sufficient

# Use circuits and checkpoint from round 5
python playground/circuit_lab.py apply --exp_dir checkpoints/non_iid_pathological --client_id 0 --round 5 --round_key round_5
```

---

### `stitch` — Merge client circuits, evaluate stitched result

Collects all client circuits for each class, merges them with the chosen strategy,
and evaluates the stitched circuit on the global model.

```bash
python playground/circuit_lab.py stitch \
    --exp_dir checkpoints/non_iid_pathological
```

| Flag | Default | Description |
|------|---------|-------------|
| `--exp_dir` | required | Path to experiment output directory |
| `--strategy` | `union` | `union`, `intersection`, or `majority` |
| `--classes` | all | Class indices to stitch, e.g. `--classes 0 1 2` |
| `--round` | last | Checkpoint round number |
| `--round_key` | last | Circuit round key |
| `--compare_clients` | off | Also print per-client individual sufficiency |

```bash
# Majority-vote stitch, compare against individual client circuits
python playground/circuit_lab.py stitch --exp_dir checkpoints/non_iid_pathological \
    --strategy majority --compare_clients

# Intersection stitch for classes 0 and 1 only
python playground/circuit_lab.py stitch --exp_dir checkpoints/non_iid_pathological \
    --strategy intersection --classes 0 1
```

**Stitch strategies:**

| Strategy | Keeps neuron if… |
|----------|-----------------|
| `union` | active in **any** client |
| `intersection` | active in **all** clients |
| `majority` | active in **> 50%** of clients |

---

## Adding a New Experiment

1. Create `playground/experiments/my_experiment.py`:

```python
from .base import BaseExperiment
from playground.core import load_config, load_model, load_dataset, load_circuits

class MyExperiment(BaseExperiment):
    def run(self):
        cfg = load_config(self.args.exp_dir)
        ...

def add_args(subparsers):
    p = subparsers.add_parser("myexp", help="...")
    p.add_argument("--exp_dir", required=True)
    return p
```

2. Register in `playground/experiments/__init__.py`:

```python
from .my_experiment import MyExperiment

REGISTRY = {
    "apply":  ApplyCircuitExperiment,
    "stitch": StitchCircuitExperiment,
    "myexp":  MyExperiment,         # ← add here
}
```

3. Import `add_args` in `circuit_lab.py`:

```python
from playground.experiments import my_experiment
...
my_experiment.add_args(sub)
```
