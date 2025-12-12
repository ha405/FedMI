import torch
import torch.nn as nn
from typing import List


class SimpleCNN(nn.Module):
    def __init__(self, conv_channels: List[int] = None, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        
        self.conv1 = nn.Conv2d(3, conv_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], 3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(conv_channels[2] * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(config) -> nn.Module:
    return SimpleCNN(
        conv_channels=config.conv_channels,
        num_classes=config.num_classes
    ).to(config.device)
