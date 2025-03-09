import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5, dropout_prob=0.2):
        super(CNN, self).__init__()
        # define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16, eps=2e-05, momentum=0.05)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32, eps=2e-05, momentum=0.05)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64, eps=2e-05, momentum=0.2)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Calculate the flattened size after convolutions
        # Assuming input size is (1, 96, 96)
        self.flattened_size = self._calculate_flattened_size((1, 96, 96))

        # Define fully connected layers with dropout
        self.fc1 = nn.Linear(self.flattened_size, 84)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(84, n_classes)

    def _calculate_flattened_size(self, shape):
        # Function to calculate the flattened size after convolutions
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # compute forward pass
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the output before passing it to fully connected layers
        x = x.view(-1, self.flattened_size)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
