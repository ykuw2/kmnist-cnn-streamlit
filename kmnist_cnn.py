import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN (Convolutional Neural Network) for KMNIST Classification. This uses 3 convolution layers and 2 fully connected layers.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 1 input channel (grayscale), 32 channels, 3x3 kernel size (feature), moving filter by 1 pixel at a time, then adding the padding for 28 x 28, 32 x 28 x 28 -> 32 x 14 x 14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32 -> 64 channels, extracting more features, 14 x 14, 32 x 14 x 14 -> 64 x 7 x 7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 64 -> 128 channels, extracting even more features, 7 x 7 -> 128 x 7 x 7
        self.bn1 = nn.BatchNorm2d(32) # Normalizes the inputs of each layer so it makes faster computations, 32 channels here
        self.bn2 = nn.BatchNorm2d(64) # Normalization, 64 channels here
        self.bn3 = nn.BatchNorm2d(128) # Normaliztion, 128 channels here
        self.pool = nn.MaxPool2d(2) # 2x2 pooling to extract the essential features
        self.dropout = nn.Dropout2d(0.3) # This drops 30% of the features so basically that the model can "learn better"

        self.fc1 = nn.Linear(128 * 3 * 3, 128) # Hyperparameter of 128 - fully connected layers
        self.fc2 = nn.Linear(128, 10) # 10 is the output - 10 different outputs for KMNIST

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x