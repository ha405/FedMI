import torch
import torch.nn as nn
from typing import List

class SimpleCNN(nn.Module):
    def __init__(self, conv_channels: List[int] = None, num_classes: int = 10, input_channels: int = 1):
        super(SimpleCNN, self).__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        
        # Input channels (1 for MNIST, 3 for CIFAR)
        self.conv1 = nn.Conv2d(input_channels, conv_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], 3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dynamic calculation for Linear layer size
        # MNIST (28x28) -> 3 pools -> 3x3 spatial dimension
        # CIFAR (32x32) -> 3 pools -> 4x4 spatial dimension
        self.spatial_dim = 3 if input_channels == 1 else 4
        
        self.fc = nn.Linear(conv_channels[2] * self.spatial_dim * self.spatial_dim, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(config) -> nn.Module:
    # Determine input channels based on dataset
    if config.dataset_name == "MNIST":
        in_channels = 1
    else:
        in_channels = 3

    return SimpleCNN(
        conv_channels=config.conv_channels,
        num_classes=config.num_classes,
        input_channels=in_channels
    ).to(config.device)