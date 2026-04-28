import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List


def get_transforms(dataset_name: str) -> transforms.Compose:
    if dataset_name == "MNIST":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_dataset(config):
    transform = get_transforms(config.dataset_name)

    if config.dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root=config.data_root, train=True,  download=True, transform=transform)
        testset  = torchvision.datasets.MNIST(root=config.data_root, train=False, download=True, transform=transform)
    elif config.dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root=config.data_root, train=True,  download=True, transform=transform)
        testset  = torchvision.datasets.CIFAR10(root=config.data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    train_labels = get_labels(trainset)
    test_labels  = get_labels(testset)
    trainset = Subset(trainset, np.where(train_labels < config.num_classes)[0])
    testset  = Subset(testset,  np.where(test_labels  < config.num_classes)[0])
    return trainset, testset


def get_labels(dataset) -> np.ndarray:
    # Fast path: raw dataset with .targets attribute
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    # Fast path: Subset wrapping a dataset with .targets
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "targets"):
        return np.array(dataset.dataset.targets)[np.array(dataset.indices)]
    # Fallback: iterate (slow, only for unknown wrappers)
    return np.array([y for _, y in dataset])


def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    indices = np.random.permutation(len(dataset))
    return [s.tolist() for s in np.array_split(indices, num_clients)]


def partition_dirichlet(dataset, num_clients: int, alpha: float, num_classes: int) -> List[List[int]]:
    labels = get_labels(dataset)
    client_indices = [[] for _ in range(num_clients)]
    while min(len(c) for c in client_indices) < 10:
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for i, split in enumerate(np.split(idx_k, split_points)):
                client_indices[i].extend(split.tolist())
    return client_indices


def get_classes_for_client(dataset, indices: List[int]) -> List[int]:
    labels = get_labels(dataset)
    return sorted(int(c) for c in np.unique(labels[np.array(indices, dtype=int)]))


def get_dataloader(dataset, indices: List[int], config, shuffle: bool = True) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)


def get_test_dataloader(testset, config) -> DataLoader:
    return DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
