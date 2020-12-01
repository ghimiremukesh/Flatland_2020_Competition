import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
from reinforcement_learning.policy import Policy

LEARNING_RATE = 0.00001
GAMMA = 0.98
LMBDA = 0.95
EPS_CLIP = 0.1
K_EPOCH = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


class DataBuffers:
    def __init__(self, n_agents):
        self.memory = [[]] * n_agents

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PPOAgent(Policy):
    def __init__(self, state_size, action_size, n_agents):
        super(PPOAgent, self).__init__()
        self.n_agents = n_agents
        self.memory = DataBuffers(n_agents)
        self.loss = 0
        self.fc1 = nn.Linear(state_size, 256)
        self.fc_pi = nn.Linear(256, action_size)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def reset(self):
        pass

    def pi(self, x, softmax_dim=0):
        x = F.tanh(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.tanh(self.fc1(x))
        v = self.fc_v(x)
        return v

    def act(self, state, eps=None):
        prob = self.pi(torch.from_numpy(state).float())
        m = Categorical(prob)
        a = m.sample().item()
        return a

    # Record the results of the agent's action and update the model
    def step(self, handle, state, action, reward, next_state, done):
        prob = self.pi(torch.from_numpy(state).float())
        self.memory.memory[handle].append(((state, action, reward, next_state, prob[action].item(), done)))

    def make_batch(self, data_array):
        s_lst, a_lst, r_lst, s_next_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        total_reward = 0
        for transition in data_array:
            s, a, r, s_next, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            if True:
                total_reward += r
                if done:
                    r_lst.append([1])
                else:
                    r_lst.append([0])
            else:
                r_lst.append([r])
            s_next_lst.append(s_next)
            prob_a_lst.append([prob_a])
            done_mask = 1 - int(done)
            done_lst.append([done_mask])

        total_reward = max(1.0, total_reward)
        for i in range(len(r_lst)):
            r_lst[i][0] = r_lst[i][0] * total_reward

        s, a, r, s_next, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), \
                                             torch.tensor(a_lst), \
                                             torch.tensor(r_lst), \
                                             torch.tensor(s_next_lst, dtype=torch.float), \
                                             torch.tensor(done_lst, dtype=torch.float), \
                                             torch.tensor(prob_a_lst)

        return s, a, r, s_next, done_mask, prob_a

    def train_net(self):
        for handle in range(self.n_agents):
            if len(self.memory.memory[handle]) > 0:
                s, a, r, s_next, done_mask, prob_a = self.make_batch(self.memory.memory[handle])
                for i in range(K_EPOCH):
                    td_target = r + GAMMA * self.v(s_next) * done_mask
                    delta = td_target - self.v(s)
                    delta = delta.detach().numpy()

                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in delta[::-1]:
                        advantage = GAMMA * LMBDA * advantage + delta_t[0]
                        advantage_lst.append([advantage])
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float)

                    pi = self.pi(s, softmax_dim=1)
                    pi_a = pi.gather(1, a)
                    ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()
                    self.loss = loss.mean().detach().numpy()
        self.memory = DataBuffers(self.n_agents)

    def end_episode(self, train):
        if train:
            self.train_net()

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.fc1.state_dict(), filename + ".fc1")
        torch.save(self.fc_pi.state_dict(), filename + ".fc_pi")
        torch.save(self.fc_v.state_dict(), filename + ".fc_v")
        torch.save(self.optimizer.state_dict(), filename + ".optimizer")

    def _load(self, obj, filename):
        if os.path.exists(filename + ".policy"):
            print(' >> ', filename + ".policy")
            try:
                obj.load_state_dict(torch.load(filename + ".policy", map_location=device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        print("load policy from file", filename)
        self.fc1 = self._load(self.fc1, filename + ".fc1")
        self.fc_pi = self._load(self.fc_pi, filename + ".fc_pi")
        self.fc_v = self._load(self.fc_v, filename + ".fc_v")
        self.optimizer = self._load(self.optimizer, filename + ".optimizer")

    def clone(self):
        policy = PPOAgent(self.state_size, self.action_size, self.num_agents)
        policy.fc1 = copy.deepcopy(self.fc1)
        policy.fc_pi = copy.deepcopy(self.fc_pi)
        policy.fc_v = copy.deepcopy(self.fc_v)
        policy.optimizer = copy.deepcopy(self.optimizer)
        return self
