# Class Distribution Visualizer

## Overview

The `ClassDistributionVisualizer` automatically generates histograms and heatmaps showing the class distribution across clients for non-IID (non-Independent and Identically Distributed) data partitioning, particularly useful for **Dirichlet** and **systematic skew** distributions.

## Features

### Auto-Integration
The visualizer is **automatically integrated** into the federated learning pipeline. When you run an experiment with non-IID partitioning methods (dirichlet, systematic_skew, or manual), the visualizations are generated automatically after data partitioning.

### Generated Visualizations

1. **Individual Client Distributions** (`class_distribution_individual_*.png`)
   - Subplots showing the class composition for each client
   - Displays sample counts per class
   - Useful for understanding individual client data heterogeneity

2. **Stacked Bar Chart** (`class_distribution_stacked_*.png`)
   - Shows all classes across all clients
   - Each client is a bar, stacked by class
   - Useful for comparing total and composition across clients

3. **Proportion Heatmap** (`class_distribution_heatmap_*.png`)
   - Shows the proportion (0-1) of each class per client
   - Color intensity represents the proportion (darker = higher proportion)
   - Useful for identifying class imbalances and heterogeneity patterns

4. **Statistics JSON** (`class_distribution_stats_*.json`)
   - Detailed statistics for each client and overall
   - Includes sample counts and proportions per class
   - Useful for further analysis

## Usage

### Automatic Usage (Recommended)

Simply run your experiment as normal:

```bash
python main.py --config_file configs/non_iid_dirichlet.json
```

The visualizations will be automatically generated and saved to:
```
checkpoints/<experiment_name>/figures/
```

### Manual Usage

If you need to visualize existing partitions:

```python
from analysis.visualizer.class_distribution import ClassDistributionVisualizer

# Initialize visualizer with experiment output directory
visualizer = ClassDistributionVisualizer(
    output_dir="./checkpoints/non_iid_dirichlet",
    partition_method="dirichlet"
)

# Generate all plots
visualizer.run(
    dataset_name="MNIST",
    class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
)
```

## Output Structure

```
checkpoints/<experiment_dir>/
├── figures/
│   ├── class_distribution_individual_dirichlet.png
│   ├── class_distribution_stacked_dirichlet.png
│   ├── class_distribution_heatmap_dirichlet.png
│   └── class_distribution_stats_dirichlet.json
├── partitions/
│   └── client_partitions.json
└── config.json
```

## Interpreting the Plots

### Individual Client Distributions
- Each subplot represents one client
- Bar heights show the number of samples for each class
- Imbalanced bars indicate high class heterogeneity at that client

### Stacked Bar Chart
- Useful for comparing total data volume per client
- Color consistency shows how classes are distributed
- Wide color variations across clients indicate high non-IID-ness

### Heatmap
- Values close to 1 (darker red) mean that client has mostly that class
- Values close to 0 (lighter yellow) mean that client has very few of that class
- Diagonal-like patterns indicate biased distributions (more non-IID)
- Uniform gray pattern indicates more IID distribution

## Configuration

The visualizer works with these partition methods:
- `dirichlet`: Dirichlet distribution-based partitioning
- `systematic_skew`: Custom skew profile-based partitioning
- `manual`: Manual class allocation to clients
- `iid`: (visualization skipped - data is balanced anyway)

## Notes

- The visualizer automatically loads the dataset from the configured data directory
- Supports MNIST and CIFAR10 datasets
- Visualizations have high DPI (200) for publication-quality output
- Statistics are saved as JSON for programmatic access
- The visualizer gracefully handles missing data with informative warnings

## Troubleshooting

**"Partition file not found"**
- Ensure the experiment has been run and partitions have been created
- Check that the output directory path is correct

**"Error loading dataset labels"**
- Verify the dataset is configured correctly in your config
- Ensure the dataset download location is correct
- Check that the dataset name is supported (MNIST or CIFAR10)

**Missing visualizations**
- Check the figures directory in your experiment output
- Look for warning messages in the console output
- Ensure matplotlib is properly installed
