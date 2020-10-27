import os
import random

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from reinforcement_learning.policy import Policy
from reinforcement_learning.ppo.model import PolicyNetwork
from reinforcement_learning.ppo.replay_memory import Episode, ReplayBuffer

BUFFER_SIZE = 128_000
BATCH_SIZE = 8192
GAMMA = 0.95
LR = 0.5e-4
CLIP_FACTOR = .005
UPDATE_EVERY = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(Policy):
    def __init__(self, state_size, action_size, num_agents, env):
        self.action_size = action_size
        self.state_size = state_size
        self.num_agents = num_agents
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.old_policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.episodes = [Episode() for _ in range(num_agents)]
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0
        self.loss = 0
        self.env = env

    def reset(self):
        self.finished = [False] * len(self.episodes)
        self.tot_reward = [0] * self.num_agents

    # Decide on an action to take in the environment

    def act(self, handle, state, eps=None):
        if True:
            self.policy.eval()
            with torch.no_grad():
                output = self.policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                return Categorical(output).sample().item()

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.policy.eval()
            with torch.no_grad():
                output = self.policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                return Categorical(output).sample().item()
        else:
            return random.choice(np.arange(self.action_size))

    # Record the results of the agent's action and update the model
    def step(self, handle, state, action, reward, next_state, done):
        if not self.finished[handle]:
            # Push experience into Episode memory
            self.tot_reward[handle] += reward
            if done == 1:
                reward = 1  # self.tot_reward[handle]
            else:
                reward = 0

            self.episodes[handle].push(state, action, reward, next_state, done)

            # When we finish the episode, discount rewards and push the experience into replay memory
            if done:
                self.episodes[handle].discount_rewards(GAMMA)
                self.memory.push_episode(self.episodes[handle])
                self.episodes[handle].reset()
                self.finished[handle] = True

        # Perform a gradient update every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE * 4:
            self._learn(*self.memory.sample(BATCH_SIZE, device))

    def _clip_gradient(self, model, clip):

        for p in model.parameters():
            p.grad.data.clamp_(-clip, clip)
        return

        """Computes a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in model.parameters():
            if p.grad is not None:
                modulenorm = p.grad.data.norm()
                totalnorm += modulenorm ** 2
        totalnorm = np.sqrt(totalnorm)
        coeff = min(1, clip / (totalnorm + 1e-6))

        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(coeff)

    def _learn(self, states, actions, rewards, next_state, done):
        self.policy.train()

        responsible_outputs = torch.gather(self.policy(states), 1, actions)
        old_responsible_outputs = torch.gather(self.old_policy(states), 1, actions).detach()

        # rewards = rewards - rewards.mean()
        ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
        clamped_ratio = torch.clamp(ratio, 1. - CLIP_FACTOR, 1. + CLIP_FACTOR)
        loss = -torch.min(ratio * rewards, clamped_ratio * rewards).mean()
        self.loss = loss

        # Compute loss and perform a gradient step
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer.zero_grad()
        loss.backward()
        # self._clip_gradient(self.policy, 1.0)
        self.optimizer.step()

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.policy.state_dict(), filename + ".policy")
        torch.save(self.optimizer.state_dict(), filename + ".optimizer")

    def load(self, filename):
        print("load policy from file", filename)
        if os.path.exists(filename + ".policy"):
            print(' >> ', filename + ".policy")
            try:
                self.policy.load_state_dict(torch.load(filename + ".policy"))
            except:
                print(" >> failed!")
                pass
        if os.path.exists(filename + ".optimizer"):
            print(' >> ', filename + ".optimizer")
            try:
                self.optimizer.load_state_dict(torch.load(filename + ".optimizer"))
            except:
                print(" >> failed!")
                pass
