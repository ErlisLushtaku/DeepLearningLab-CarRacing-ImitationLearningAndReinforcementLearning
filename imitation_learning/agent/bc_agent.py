import numpy as np
import torch
from agent.networks import CNN
import torch.optim as optim
import torch.nn.functional as F


class BCAgent:

    def __init__(self, lr=0.001, weight_decay=0.001):
        # Define network, loss function, optimizer
        self.net = CNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def update(self, X_batch, y_batch):
        # Zero the gradients
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_correct = 0

        for i in range(X_batch.shape[0]):  # Iterate over each image in the minibatch
            # Get the i-th image and label
            X_tensor = torch.tensor(X_batch[i:i + 1])  # Select the i-th image
            y_tensor = torch.tensor(y_batch[i:i + 1], dtype=torch.long)  # Convert label to float type

            # Forward pass
            outputs = self.net(X_tensor.unsqueeze(0))

            # Compute loss
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()

            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_tensor).item()

        # Optimize
        self.optimizer.step()

        # Return average loss and accuracy
        return total_loss / X_batch.shape[0], total_correct / len(X_batch)

    def predict(self, X):
        # Forward pass
        # Add batch size and channel dimensions
        X = np.expand_dims(X, axis=0)  # Add channel dimension (assuming grayscale image)
        X = np.expand_dims(X, axis=0)

        # Transform numpy array to PyTorch tensor
        X_tensor = torch.tensor(X)
        outputs = self.net(X_tensor)

        # Apply softmax activation to obtain class probabilities
        probs = F.softmax(outputs, dim=1)

        # Get the index of the class with the highest probability
        pred_class = torch.argmax(probs, dim=1)

        max_value, _ = torch.max(probs, 1)

        return pred_class.numpy(), max_value.item()

    def compute_loss(self, X_valid, y_valid):
        total_loss = 0.0
        total_correct = 0

        for i in range(X_valid.shape[0]):
            # Get the i-th image and label and transform input to tensors
            X_tensor = torch.tensor(X_valid[i:i + 1])
            y_tensor = torch.tensor(y_valid[i:i + 1], dtype=torch.long)

            # Forward pass
            outputs = self.net(X_tensor.unsqueeze(0))

            # Compute loss
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, y_tensor)

            # Accumulate total loss
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_tensor).item()

        # Compute average loss and accuracy
        return total_loss / X_valid.shape[0], total_correct / len(y_valid)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
