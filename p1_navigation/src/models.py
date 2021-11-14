import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.V_fc1 = nn.Linear(state_size, 64)
        self.V_fc2 = nn.Linear(64, 64)
        self.V_fc3 = nn.Linear(64, 1)

        self.A_fc1 = nn.Linear(state_size, 64)
        self.A_fc2 = nn.Linear(64, 64)
        self.A_fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.V_fc1(state))
        x = F.relu(self.V_fc2(x))
        state_value = self.V_fc3(x)

        x = F.relu(self.A_fc1(state))
        x = F.relu(self.A_fc2(x))
        advantage_values = self.A_fc3(x)

        return state_value + (advantage_values - advantage_values.mean())