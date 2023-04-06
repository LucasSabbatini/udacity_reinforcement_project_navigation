import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)

        # Advantage values
        self.fc2_a = nn.Linear(64, 64)
        self.fc3_a = nn.Linear(64, action_size)

        # State value
        self.fc2_s = nn.Linear(64, 64)
        self.fc3_s = nn.Linear(64, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        advantage = F.relu(self.fc2_a(x))
        advantage = self.fc3_a(advantage)
        state_value = F.relu(self.fc2_s(x))
        state_value = self.fc3_s(state_value)
        q_values = state_value + advantage - advantage.mean()
        return q_values