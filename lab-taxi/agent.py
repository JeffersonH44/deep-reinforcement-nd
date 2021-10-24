import numpy as np
import random

from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.episodes = 1
        self.gamma = 0.77
        self.alpha = 0.25
        self.epsilon = 0.01
        self.eps_decay = 0.9

    def get_epsilon_greedy_action(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.get_epsilon_greedy_action(state)
    
    def curr_func(self, state, action):
        # for sarsa learning
        #return self.Q[state][action]
        # for Q learning
        return max(self.Q[state])

    def __update(self, state, action, reward, next_state, next_action):
        Qsa_next = self.curr_func(next_state, next_action) if next_action is not None else 0.0
        Qsa_current = self.Q[state][action]
        target = reward + (self.gamma * Qsa_next)
        return Qsa_current + self.alpha*(target - Qsa_current)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.get_epsilon_greedy_action(next_state) if not done else None
        self.Q[state][action] = self.__update(state, action, reward, next_state, next_action)

        # after all updates, update episode
        if done:
            self.epsilon = self.epsilon*self.eps_decay