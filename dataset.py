import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple, List


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_cifar10(config):
    transform = get_transforms()
    trainset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=False, download=True, transform=transform
    )
    return trainset, testset


def partition_iid(dataset, num_clients: int) -> List[List[int]]:
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    return [list(idx) for idx in split_indices]


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
