from .dueling_qnetwork import DuelingQNetwork
from .qnetwork import QNetwork

from enum import Enum

class ModelFactory(Enum):
    qnetwork = QNetwork
    dueling_qnetwork = DuelingQNetwork