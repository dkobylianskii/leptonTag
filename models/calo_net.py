import numpy as np
import torch
import torch.nn as nn


class CaloNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.fc1 = nn.Linear(10 * 3 * 3, 50)
        self.fc2 = nn.Linear(50, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x