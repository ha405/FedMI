import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple, List, Dict

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
    
    train_labels = get_labels(trainset)
    test_labels = get_labels(testset)
    train_valid = np.where(train_labels < config.num_classes)[0]
    test_valid = np.where(test_labels < config.num_classes)[0]
    trainset = Subset(trainset, train_valid)
    testset = Subset(testset, test_valid)
        
    return trainset, testset

def get_labels(dataset):
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    else:
        return np.array([y for _, y in dataset])

# --- Partitioning Logic ---

def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    return [idx.tolist() for idx in split_indices]

def partition_dirichlet(dataset, num_clients: int, alpha: float, num_classes: int) -> List[List[int]]:
    labels = get_labels(dataset)
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

def partition_by_class(dataset, client_class_map: Dict[int, List[int]]) -> List[List[int]]:
    """Older manual partition method where clients get specific classes exclusively."""
    labels = get_labels(dataset)
    num_clients = len(client_class_map)
    client_indices = [[] for _ in range(num_clients)]
    
    for client_id, classes in client_class_map.items():
        for class_label in classes:
            idx_k = np.where(labels == class_label)[0]
            client_indices[client_id].extend(idx_k.tolist())
            
    return client_indices

def partition_systematic_skew(dataset, skew_profile: Dict[int, Dict[int, float]], num_clients: int, num_classes: int) -> List[List[int]]:
    """
    Partition data based on a systematic skew profile.
    
    Args:
        skew_profile: {client_id: {class_id: probability_mass, ...}}
        
    Example:
        If client 0 should have 90% class 0 and 10% class 1:
        {0: {0: 0.9, 1: 0.1}}
        
        Note: The actual implementation distributes available data. If a class is exhausted, 
        it might not perfectly match the requested proportion if data is limited.
        However, for typical MNIST/CIFAR, we distribute the *global* pool of data 
        according to the *normalized* demand of all clients.
    """
    labels = get_labels(dataset)
    client_indices = [[] for _ in range(num_clients)]
    
    # 1. Organize all indices by class
    class_indices = {k: np.where(labels == k)[0] for k in range(num_classes)}
    for k in class_indices:
        np.random.shuffle(class_indices[k])
    
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients # Approximation
    
    
    for k in range(num_classes):
        # Gather weights for class k from all clients
        client_weights = []
        for c_id in range(num_clients):
            # perform safe get, default to 0 if not specified
            w = 0.0
            if c_id in skew_profile and k in skew_profile[c_id]:
                w = skew_profile[c_id][k]
            client_weights.append(w)
            
        client_weights = np.array(client_weights)
        total_weight = client_weights.sum()
        
        if total_weight == 0:
            proportions = np.ones(num_clients) / num_clients
        else:
            proportions = client_weights / total_weight

        idx_k = class_indices[k]
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_idx_k = np.split(idx_k, split_points)
        
        for i in range(num_clients):
            client_indices[i].extend(split_idx_k[i].tolist())
            
    return client_indices


def split_public_data(dataset, fraction: float, seed: int, config) -> tuple:
    labels = get_labels(dataset)
    n = len(dataset)
    valid_indices = np.where(labels < config.num_classes)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(valid_indices) 
    num_valid = len(valid_indices)
    num_public = int(num_valid * fraction)
    if num_public == 0:
        return valid_indices.tolist(), None
    public_indices = valid_indices[:num_public].tolist()
    private_indices = valid_indices[num_public:].tolist()
    public_subset = Subset(dataset, public_indices)
    public_loader = DataLoader(
        public_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    return private_indices, public_loader


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

def get_client_class_counts(dataloader, num_classes):
    counts = torch.zeros(num_classes)
    for _, labels in dataloader:
        unique, c = torch.unique(labels, return_counts=True)
        for label, count in zip(unique, c):
            counts[label] += count
    return counts

def get_classes_for_client(dataset, indices: List[int]) -> List[int]:

    labels = get_labels(dataset)
    client_labels = labels[np.array(indices, dtype=int)]
    return sorted(int(c) for c in np.unique(client_labels))

