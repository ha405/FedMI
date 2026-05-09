# FedMI — Mechanistic Interpretability in Federated Learning

Mechanistic analysis of circuit preservation across federated learning clients under IID and non-IID data distributions.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Single config

```bash
python main.py --config_file configs/cifar_cnn/iid.json
```

### Full suite (all 4 families × 4 conditions)

```bash
python run_suite.py
```

### One family only

```bash
python run_suite.py --families cifar_cnn
```

### Skip phases

```bash
python run_suite.py --skip probes finetune usae
```

### Smoke test (1 round, _test-suffixed dirs)

```bash
python run_suite.py --test --families cifar_cnn
```

---

## Config Layout

```
configs/
  cifar_cnn/      iid.json  alpha_05.json  alpha_02.json  alpha_005.json
  cifar_resnet/   iid.json  alpha_05.json  alpha_02.json  alpha_005.json
  fmnist_cnn/     iid.json  alpha_05.json  alpha_02.json  alpha_005.json
  fmnist_resnet/  iid.json  alpha_05.json  alpha_02.json  alpha_005.json
config.json       # root template
```

---

## Analysis

Circuit consistency figures (inter-client IoU, local-vs-global IoU, intra-client IoU):

```bash
python analysis/circuit_consistency.py --compare      results/
python analysis/circuit_consistency.py --local-global results/
python analysis/circuit_consistency.py --intra-client results/
```

---

## Post-hoc Analysis (manual)

### Linear probes on a single experiment

```bash
python run_suite.py --probe-dir results/CIFAR_CNN
```

### FC finetuning on a single experiment

```bash
python run_suite.py --finetune-dir results/CIFAR_CNN
```

### Universal SAE on two experiments

```bash
python run_suite.py --usae-dirs results/CIFAR_CNN results/CIFAR_CNN_005
```

---

## Output Structure

```
results/<experiment>/
  config.json
  checkpoints/         # round-by-round global model checkpoints
  circuits/            # per-round circuit JSON + all_circuits.json
  figures/             # IoU plots (inter, intra, local-global)
  logs/                # training log
  partitions/          # client data partition indices
  probes/              # linear probe results (if run)
  finetuning_results/  # FC finetune results (if run)
```
