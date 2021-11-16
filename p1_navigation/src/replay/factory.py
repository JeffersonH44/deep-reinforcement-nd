from enum import Enum
from random import uniform
from .base_replay import ReplayBuffer
from .prioritized_replay import PrioritizedReplayBuffer

class ReplayFactory(Enum):
    uniform = ReplayBuffer
    prioritized = PrioritizedReplayBuffer