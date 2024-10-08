import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.models.q_network import QNetwork
from src.agents.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQNAgent implements a Deep Q-Network (DQN) agent with experience replay.
    """
    def __init__(
        self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
        epsilon_min=0.01, buffer_capacity=10000, batch_size=64, device="cpu", seed=42
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim (tuple): Dimensions of the input state.
            action_dim (int): Number of possible actions.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            epsilon (float, optional): Initial value for epsilon in epsilon-greedy action selection. Defaults to 1.0.
            epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.995.
            epsilon_min (float, optional): Minimum value for epsilon. Defaults to 0.01.
            buffer_capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
            batch_size (int, optional): Number of experiences to sample from the buffer for each training step. Defaults to 64.
            device (str, optional): Device to use for computations (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.seed = seed
        self.set_seed(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.device = device
        self.update_target_network()

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Seed value to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def update_target_network(self):
        """
        Update the target network with the current Q-network's parameters.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            int: The selected action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_network(state).argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (numpy.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train(self):
        """
        Train the Q-network using a batch of experiences from the replay buffer.
        """
        if self.replay_buffer.size() < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.int64)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32)

        q_values = self.q_network(state_batch)
        next_q_values = self.target_network(next_state_batch)

        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward_batch + (self.gamma * next_q_value * (1 - done_batch))

        loss = F.mse_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
