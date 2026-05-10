import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Architecture must match the notebook's DigitCNN exactly so safetensors load cleanly."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.20)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.drop2 = nn.Dropout(0.30)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)
