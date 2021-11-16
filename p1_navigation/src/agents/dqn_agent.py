import numpy as np
import random

from src.models import ModelFactory
from src.utils import get_seed, get_device
from hydra import compose, initialize
from enum import Enum

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.replay import ReplayFactory

device = get_device()

class DQNAgent():
    """Interacts with and learns from the environment."""


    def __init__(self, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.cfg = compose(config_name="agent")
        self.cfg_model = compose(config_name="model")

        self.state_size = state_size
        self.action_size = action_size

        seed = get_seed()
        self.seed = random.seed(seed)

        # Q-Network
        network_kind = ModelFactory[self.cfg.network_kind].value
        self.qnetwork_local = network_kind(state_size, action_size).to(device)
        self.qnetwork_target = network_kind(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.cfg_model.lr)

        # Replay memory
        self.replay_model = ReplayFactory[self.cfg.replay_kind]
        self.memory = (self.replay_model.value)(action_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Define next q state calculator
        NEXT_Q_STATE = {
            "ddqn": self.__dqn_q_next_state,
            "dqn": self.__ddqn_q_next_state
        }
        self.next_state_q_calculator = NEXT_Q_STATE[self.cfg.loss_kind]

        # Define loss calculator function
        LOSS_FUNCTION = {
            "prioritized": self.__loss_for_prioritized_replay,
            "uniform": self.__loss_for_uniform_replay
        }
        self.loss_calculator = LOSS_FUNCTION[self.cfg.replay_kind]
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.cfg.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.cfg_model.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.cfg.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.replay_model == ReplayFactory.uniform:
            states, actions, rewards, next_states, dones = experiences
            indices, weights = (None, None)
        else:
            states, actions, rewards, next_states, dones, indices, weights = experiences

        Q_targets_next_s = self.next_state_q_calculator(next_states)
        Q_targets = rewards + (gamma * Q_targets_next_s * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = self.loss_calculator(
            Q_expected,
            Q_targets,
            indices=indices,
            weights=weights
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.cfg.tau)
    
    def __loss_for_uniform_replay(self, Q_expected, Q_targets, **kwargs):
        return F.mse_loss(Q_expected, Q_targets)

    def __loss_for_prioritized_replay(self, Q_expected, Q_targets, **kwargs):
        indices = kwargs['indices']
        weights = kwargs['weights']

        elementwise_loss = F.mse_loss(Q_expected, Q_targets, reduction="none")

        loss_for_per = elementwise_loss.detach().cpu().flatten().tolist()
        self.memory.update_priorities(indices, loss_for_per)

        return torch.mean(weights * elementwise_loss)

    def __dqn_q_next_state(self, next_states):
        return self.qnetwork_target(next_states).detach().max(1).values.unsqueeze(1)

    def __ddqn_q_next_state(self, next_states):
        return self.qnetwork_target(next_states).gather(1, 
            self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
        )

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)