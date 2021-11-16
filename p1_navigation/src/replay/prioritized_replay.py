import random
import torch
import numpy as np

from collections import deque
from .base_replay import ReplayBuffer
from src.utils import get_device

device = get_device()

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size):
        super().__init__(action_size)

        self.max_error = 1.0 # this is handpicked, should be update when new losses are calculated and updated
        self.alpha = 0.6 # as described in the paper
        self.beta = 0.4 # as described in the paper
        self.eps = 1e-5
        self.losses = deque(self.buffer_size)

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)

        self.losses.append(self.max_error ** self.alpha)
    
    def update_priorities(self, indices, losses):
        for idx, loss in zip(indices, losses):
            if loss < 0.0:
                raise ValueError(f"Loss {loss} should not be less than 0")
            
            self.losses[idx] = (loss + self.eps) ** self.alpha

            self.max_error = max(self.max_error, loss)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences_with_indices = self.__get_samples()
        experiences, indices = zip(*experiences_with_indices)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = self.__calculate_weights(indices)
  
        return (states, actions, rewards, next_states, dones, indices, weights)

    def __get_samples(self):
        return random.choices(
            population=list(enumerate(self.memory)),
            k=self.batch_size,
            weights=self.losses
        )

    def __calculate_weights(self, indices):
        p_max = max(self.losses)
        max_weight = (1/(len(self) * p_max)) ** self.beta

        p_samples = [self.losses[i] for i in indices]
        def calculate_weight(p_val):
            return ((1/(len(self) * p_val)) ** self.beta) / max_weight

        return list(map(calculate_weight, p_samples))