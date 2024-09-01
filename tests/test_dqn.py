import random
import numpy as np
import torch

from src.agents.dqn import DQNAgent

def test_dqn_initialization():
    state_dim = (4, 84, 84)
    action_dim = 6
    agent = DQNAgent(state_dim, action_dim, seed=42)

    assert agent.state_dim == state_dim
    assert agent.action_dim == action_dim
    assert agent.q_network is not None
    assert agent.target_network is not None
    assert agent.optimizer is not None
    assert agent.replay_buffer is not None
    assert agent.epsilon == 1.0

def test_set_seed():
    state_dim = (4, 84, 84)
    action_dim = 6
    agent = DQNAgent(state_dim, action_dim, seed=42)
    agent.set_seed(42)

    assert random.random() == 0.6394267984578837
    assert np.random.random() == 0.3745401188473625
    assert torch.rand(1).item() == 0.8822692632675171
