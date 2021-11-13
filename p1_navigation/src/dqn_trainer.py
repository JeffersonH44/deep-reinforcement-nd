import numpy as np
import torch

from src.dqn_agent import Agent
from collections import deque
from tqdm import trange

class DQNTrainer():
    def __init__(
        self,
        env,
        n_episodes=2000, 
        max_t=1000, 
        eps_start=1.0, 
        eps_end=0.01, 
        eps_decay=0.995,
        seed=555
    ) -> None:
        self.env = env
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.seed = seed

        self.create_agent()

    def create_agent(self):
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]

        env_info = self.env.reset(train_mode=True)[brain_name]

        action_size = brain.vector_action_space_size
        state_size = len(env_info.vector_observations[0])

        self.agent = Agent(
            state_size=state_size, 
            action_size=action_size, 
            seed=self.seed
        )

    def train(self):
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=300)  # last 100 scores
        eps = self.eps_start               # initialize epsilon
        env = self.env
        agent = self.agent
        t_bar = trange(self.n_episodes)
        for i_episode in t_bar:
            state = env.reset()
            score = 0
            for t in range(self.max_t):
                reward, done = self.train_step(agent, env, state, eps)
                score += reward
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            eps = max(self.eps_end, self.eps_decay*eps) 
            t_bar.set_description(f"Average Score: {np.mean(scores_window)}")
            if i_episode % 100 == 0:
                torch.save(agent.qnetwork_local.state_dict(), f"checpoint_{i_episode}.pth")
            
            
    def train_step(self, agent, env, state, eps) -> bool:
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        return reward, done