import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from .base import BaseVisualizer


class ClassDistributionVisualizer(BaseVisualizer):
    """
    Visualizes class distribution histograms across clients for non-IID data partitioning.
    """
    
    def __init__(self, output_dir, partition_method="dirichlet"):
        super().__init__(output_dir)
        self.partition_method = partition_method
        self.partition_file = os.path.join(output_dir, "partitions", "client_partitions.json")
        self.config_file = os.path.join(output_dir, "config.json")
        
    def load_partition_data(self):
        """Load partition data from JSON."""
        if not os.path.exists(self.partition_file):
            print(f"[ClassDistributionVisualizer] Partition file not found: {self.partition_file}")
            return None
        
        try:
            with open(self.partition_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ClassDistributionVisualizer] Error loading partition data: {e}")
            return None
    
    def load_config(self):
        """Load config to get dataset and class info."""
        if not os.path.exists(self.config_file):
            print(f"[ClassDistributionVisualizer] Config file not found: {self.config_file}")
            return None
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ClassDistributionVisualizer] Error loading config: {e}")
            return None
    
    def get_dataset_labels(self, dataset_name: str = "MNIST") -> np.ndarray:
        """
        Load dataset labels from the original dataset.
        Supports MNIST and CIFAR10.
        """
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.ToTensor()
        
        if dataset_name == "MNIST":
            trainset = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
        elif dataset_name == "CIFAR10":
            trainset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if hasattr(trainset, 'targets'):
            return np.array(trainset.targets)
        else:
            return np.array([y for _, y in trainset])
    
    def plot_individual_clients(self, class_distributions: Dict[int, np.ndarray], 
                               num_classes: int, class_names: List[str] = None):
        """
        Create subplots showing class distribution histogram for each client.
        """
        num_clients = len(class_distributions)
        cols = min(4, num_clients)
        rows = (num_clients + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()
        
        for client_id, dist in class_distributions.items():
            ax = axes[client_id]
            
            classes = [class_names[i] if class_names else str(i) for i in range(num_classes)]
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            
            ax.bar(classes, dist, color=colors, edgecolor='black', alpha=0.7)
            ax.set_title(f"Client {client_id} (n={sum(dist)})", fontsize=12, fontweight='bold')
            ax.set_ylabel("Number of Samples", fontsize=10)
            ax.set_xlabel("Class", fontsize=10)
            
            # Add count labels on bars
            for i, count in enumerate(dist):
                ax.text(i, count, str(int(count)), ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_clients, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def run(self, dataset_name: str = "MNIST", class_names: List[str] = None):
        """
        Generate class distribution histogram.
        
        Args:
            dataset_name: Name of the dataset (MNIST, CIFAR10)
            class_names: List of class names for labeling
        """
        print(f"[ClassDistributionVisualizer] Generating class distribution histogram...")
        
        # Load data
        partitions = self.load_partition_data()
        config = self.load_config()
        
        if partitions is None or config is None:
            print("[ClassDistributionVisualizer] Failed to load required data.")
            return False
        
        # Get dataset info
        if dataset_name is None:
            dataset_name = config.get("dataset_name", "MNIST")
        
        num_classes = config.get("num_classes", 10)
        
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        # Load labels
        try:
            labels = self.get_dataset_labels(dataset_name)
        except Exception as e:
            print(f"[ClassDistributionVisualizer] Error loading dataset labels: {e}")
            return False
        
        # Calculate class distributions per client
        class_distributions = {}
        for client_id, indices in enumerate(partitions):
            counts = np.zeros(num_classes, dtype=int)
            for idx in indices:
                counts[labels[idx]] += 1
            class_distributions[client_id] = counts
        
        # Generate histogram
        try:
            fig = self.plot_individual_clients(class_distributions, num_classes, class_names)
            self.save_plot(fig, f"class_distribution_{self.partition_method}.png")
            print(f"[ClassDistributionVisualizer] Successfully generated histogram!")
            return True
            
        except Exception as e:
            print(f"[ClassDistributionVisualizer] Error generating histogram: {e}")
            import traceback
            traceback.print_exc()
            return False
