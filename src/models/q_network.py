import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """A simple fully connected neural network for Q-learning."""

    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim[0] * input_dim[1] * input_dim[2], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The output Q-values for each action.
        """
        x = state.view(state.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
