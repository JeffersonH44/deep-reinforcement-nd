import torch
import numpy as np

from collections import deque
from .base_replay import ReplayBuffer
from src.utils import get_device, loop_choice

device = get_device()

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size):
        super().__init__(action_size)

        self.max_error = 1.0 # this is handpicked, should be update when new losses are calculated and updated
        self.alpha = self.cfg_replay.alpha # as described in the paper
        self.beta = self.cfg_replay.beta # as described in the paper
        self.eps = self.cfg_replay.epsilon
        self.losses = deque(maxlen=self.buffer_size)

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
        indices, experiences = zip(*experiences_with_indices)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.hstack(self.__calculate_weights(indices))).float().to(device)
  
        return (states, actions, rewards, next_states, dones, indices, weights)

    def __get_samples(self):
        indices = loop_choice(
            np.arange(len(self.memory)),
            self.losses,
            self.batch_size
        )

        return [(idx, self.memory[idx]) for idx in indices]

        """return random.choices(
            population=list(enumerate(self.memory)),
            k=self.batch_size,
            weights=self.losses
        )"""

    def __calculate_weights(self, indices):
        p_max = min(self.losses)
        max_weight = (1/(len(self) * p_max)) ** self.beta

        p_samples = [self.losses[i] for i in indices]
        def calculate_weight(p_val):
            return ((1/(len(self) * p_val)) ** self.beta) / max_weight

        return list(map(calculate_weight, p_samples))