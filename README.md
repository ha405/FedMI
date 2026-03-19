# FedMI
Advanced ML Research Project - Mechanistic Analysis of Circuit Preservation in Federated Learning

ArXiv link: https://arxiv.org/abs/2512.23043

## Run on Colab / Kaggle 

FedMI ships with a **unified `fedmi/` package** that re-exports the entire codebase and auto-configures paths, device, and dependencies for cloud notebooks.

| Notebook | Description | Open in Colab |
| :--- | :--- | :--- |
| `notebooks/fedmi_train.ipynb` | Full federated training pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ha405/FedMI/blob/cvpr/notebooks/fedmi_train.ipynb) |
| `notebooks/fedmi_playground.ipynb` | Playground experiments (apply, stitch, ensemble distillation, LTH pruning) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ha405/FedMI/blob/cvpr/notebooks/fedmi_playground.ipynb) |
| `notebooks/fedmi_sae.ipynb` | SAE interpretability analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ha405/FedMI/blob/cvpr/notebooks/fedmi_sae.ipynb) |

> **Kaggle**: Upload any notebook and enable GPU in *Accelerator* settings. The repo will be auto-cloned .

## Quick Start

### 1. Run Experiments
Choose a configuration based on your scenario:

- **IID Baseline**:
  ```bash
  python main.py --config_file configs/iid.json
  ```
- **Non-IID (Label Skew)**:
  ```bash
  python main.py --config_file configs/non_iid_pathological.json
  ```
- **Full Reproducibility Suite (IID + Non-IID)**:
  ```bash
  bash experiments/run_all.sh
  ```

### 2. View Results
Outputs are saved in `checkpoints/<experiment_name>/`.

#### Automatic Visualizations (in `figures/`)
| Plot Name | What it Shows |
| :--- | :--- |
| `class_distribution_individual_*.png` | **Per-Client Class Distribution** - Histograms showing sample count per class for each client (Non-IID partitions only). |
| `class_distribution_stacked_*.png` | **Stacked Bar Chart** - All clients' class composition side-by-side for easy comparison. |
| `class_distribution_heatmap_*.png` | **Class Proportion Heatmap** - Color intensity shows what fraction of each client's data is from each class. |
| `heatmap_overlap_round_X.png` | **Red**=Local-only, **Blue**=Global-only, **Green**=Preserved/Shared neurons. |
| `cross_accuracy_drift_gap.png` | Performance gap between Client Model and Global Model on local data. |
| `Specialist_Distinctness.png` | Low IoU = Clients are specializing in disjoint tasks. |
| `sensitivity_shared_neuron_impact.png` | Validation accuracy after injecting shared neurons. |

**Note:** Class distribution visualizations are automatically generated for non-IID partitioning methods (Dirichlet, systematic skew, manual). See [Class Distribution README](analysis/visualizer/CLASS_DISTRIBUTION_README.md) for details.

#### Interactive Viewer
1. Open `checkpoints/<experiment_name>/visualizer.html` in your browser.
2. Load the `circuits/all_circuits.json` file from the same directory to explore network graphs.
