import torch
import torch.nn as nn
import torch.nn.functional as F

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