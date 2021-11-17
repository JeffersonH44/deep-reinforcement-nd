import torch
import numpy as np

from .base_replay import ReplayBuffer
from hydra import compose
from src.utils import get_device, get_next_power_of_two, SumSegmentTree, MinSegmentTree

device = get_device()

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size):
        super().__init__(action_size)

        self.max_error = 1.0 # this is handpicked, should be update when new losses are calculated and updated
        self.max_error_updated = False

        self.alpha = self.cfg_replay.alpha # as described in the paper
        self.beta = self.cfg_replay.beta # as described in the paper
        self.eps = self.cfg_replay.epsilon

        real_size = get_next_power_of_two(self.buffer_size)
        self.losses = SumSegmentTree(real_size)#deque(maxlen=self.buffer_size)
        self.min_losses = MinSegmentTree(real_size)

        # variables for annealing beta
        self.curr_episode = 1
        self.total_episodes = compose(config_name="trainer").n_episodes

    def add(self, state, action, reward, next_state, done):
        # we do the update before the constructor, otherwise the index would not be the same
        loss = self.max_error ** self.alpha
        self.losses[self.idx_pos] = loss
        self.min_losses[self.idx_pos] = loss
        super().add(state, action, reward, next_state, done)
    
    def update_priorities(self, indices, losses):
        for idx, loss in zip(indices, losses):
            if loss < 0.0:
                raise ValueError(f"Loss {loss} should not be less than 0")
            
            self.losses[idx] = (loss + self.eps) ** self.alpha

            if self.max_error_updated:
                self.max_error = max(self.max_error, loss)
            else:
                self.max_error = loss
                self.max_error_updated = True

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

    def update_beta(self, percentage_complete):
        frac = min(percentage_complete, 1.0)
        self.beta = self.beta + frac * (1.0 - self.beta)

    def __get_samples(self):
        total_sum = self.losses.sum()
        step_size = total_sum / self.batch_size
        intervals = np.arange(0, total_sum, step=step_size)
        rand_values = np.random.uniform(0, step_size, size=self.batch_size)
        chosen_elems = intervals + rand_values
        indices = [self.losses.find_prefixsum_idx(elem) for elem in chosen_elems]
        experiences = [(idx, self.memory[idx]) for idx in indices]

        return experiences

    def __calculate_weights(self, indices):
        total_sum = self.losses.sum()
        def calc_weight(loss):
            p_val = loss / total_sum
            return ((p_val * len(self)) ** -self.beta)

        max_weight = calc_weight(self.min_losses.min())
        weights = [calc_weight(self.losses[i]) / max_weight for i in indices]

        return weights