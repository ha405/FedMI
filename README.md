# FedMI
Advanced ML Research Project - Mechanistic Analysis of Circuit Preservation in Federated Learning

ArXiv link: https://arxiv.org/abs/2512.23043

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
| `heatmap_overlap_round_X.png` | **Red**=Local-only, **Blue**=Global-only, **Green**=Preserved/Shared neurons. |
| `cross_accuracy_drift_gap.png` | Performance gap between Client Model and Global Model on local data. |
| `Specialist_Distinctness.png` | Low IoU = Clients are specializing in disjoint tasks. |
| `sensitivity_shared_neuron_impact.png` | Validation accuracy after injecting shared neurons. |

#### Interactive Viewer
1. Open `checkpoints/<experiment_name>/visualizer.html` in your browser.
2. Load the `circuits/all_circuits.json` file from the same directory to explore network graphs.
