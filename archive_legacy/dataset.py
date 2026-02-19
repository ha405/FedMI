import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple, List
from typing import Dict, List
def get_transforms(dataset_name="MNIST"):
    if dataset_name == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_dataset(config):
    transform = get_transforms(config.dataset_name)
    
    if config.dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root=config.data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=config.data_root, train=False, download=True, transform=transform
        )
    elif config.dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
        
    return trainset, testset

def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    return [idx.tolist() for idx in split_indices]


def partition_dirichlet(dataset, num_clients: int, alpha: float, num_classes: int) -> List[List[int]]:
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    min_size = 0
    client_indices = [[] for _ in range(num_clients)]
    while min_size < 10:
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_k) < num_clients / 10.0 and 1.0 / num_clients or 1) for p in proportions])
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_idx_k = np.split(idx_k, split_points)
            
            for i in range(num_clients):
                client_indices[i].extend(split_idx_k[i].tolist())
        
        min_size = min([len(idx) for idx in client_indices])
    
    return client_indices

def get_dataloader(dataset, indices: List[int], config, shuffle: bool = True) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers
    )

def get_test_dataloader(testset, config) -> DataLoader:
    return DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )


def partition_by_class(dataset, client_class_map: Dict[int, List[int]]) -> List[List[int]]:
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    num_clients = len(client_class_map)
    client_indices = [[] for _ in range(num_clients)]
    
    for client_id, classes in client_class_map.items():
        for class_label in classes:
            idx_k = np.where(labels == class_label)[0]
            client_indices[client_id].extend(idx_k.tolist())
            
    return client_indices