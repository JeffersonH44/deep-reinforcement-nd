import numpy as np
import torch

from src.dqn_agent import Agent
from collections import deque
from tqdm import trange
from src.models import DuelingQNetwork
import matplotlib.pyplot as plt

class DQNTrainer():
    def __init__(
        self,
        env,
        n_episodes=2000, 
        max_t=1000, 
        eps_start=1.0, 
        eps_end=0.01, 
        eps_decay=0.995,
        seed=0
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
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]

        env_info = self.env.reset(train_mode=True)[self.brain_name]

        action_size = brain.vector_action_space_size
        state_size = len(env_info.vector_observations[0])

        self.agent = Agent(
            state_size=state_size, 
            action_size=action_size, 
            seed=self.seed,
            network_kind=DuelingQNetwork
        )

    def train(self):
        self.scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = self.eps_start               # initialize epsilon
        env = self.env
        agent = self.agent
        t_bar = trange(self.n_episodes)
        for i_episode in t_bar:
            env_info = env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for _ in range(self.max_t):
                next_state, reward, done = self.train_step(agent, env, state, eps)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)
            self.scores.append(score)
            eps = max(self.eps_end, self.eps_decay*eps)
            scores_mean = np.mean(scores_window)
            if i_episode % 100 == 0:
                t_bar.set_description(f"Average Score: {scores_mean}")
            if i_episode % 100 == 0 and i_episode != 0:
                torch.save(agent.qnetwork_local.state_dict(), f"weights/checpoint_{i_episode}.pth")
            if scores_mean >= 13.0:
                t_bar.set_description(f"Problem solved with Score: {scores_mean}")
                torch.save(agent.qnetwork_local.state_dict(), f"weights/checkpoint_final.pth")
                break
            
            
    def train_step(self, agent, env, state, eps) -> bool:
        action = agent.act(state, eps)
        
        env_info = env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        
        agent.step(state, action, reward, next_state, done)
        return next_state, reward, done

    def plot_scores(self):
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        # Save rewards plot
        plt.savefig('Rewards.png')
        plt.show()