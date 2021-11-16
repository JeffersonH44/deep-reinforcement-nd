import numpy as np
import random
import torch

from collections import namedtuple, deque
from hydra import compose
from src.utils import get_seed, get_device

device = get_device()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.cfg_replay = compose(config_name="replay")
        cfg_model = compose(config_name="model")

        buffer_size = self.cfg_replay.buffer_size
        batch_size = cfg_model.batch_size
        seed = get_seed()

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self.__get_samples()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __get_samples(self):
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)