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
    def __init__(self):
        self.reset()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def reset(self):
        self.memory = {}

    def get_transitions(self, handle):
        return self.memory.get(handle, [])

    def push_transition(self, handle, transition):
        transitions = self.get_transitions(handle)
        transitions.append(transition)
        self.memory.update({handle: transitions})


class PPOAgent(Policy):
    def __init__(self, state_size, action_size):
        super(PPOAgent, self).__init__()
        self.memory = DataBuffers()
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

    def step(self, handle, state, action, reward, next_state, done):
        # Record the results of the agent's action as transition
        prob = self.pi(torch.from_numpy(state).float())
        transition = (state, action, reward, next_state, prob[action].item(), done)
        self.memory.push_transition(handle, transition)

    def _convert_transitions_to_torch(self, transitions_array):
        state_list, action_list, reward_list, state_next_list, prob_a_list, done_list = [], [], [], [], [], []
        total_reward = 0
        for transition in transitions_array:
            state_i, action_i, reward_i, state_next_i, prob_action_i, done_i = transition

            state_list.append(state_i)
            action_list.append([action_i])
            total_reward += reward_i
            if done_i:
                reward_list.append([max(1.0, total_reward)])
                done_list.append([1])
            else:
                reward_list.append([0])
                done_list.append([0])
            state_next_list.append(state_next_i)
            prob_a_list.append([prob_action_i])

        state, action, reward, s_next, done, prob_action = torch.tensor(state_list, dtype=torch.float), \
                                                           torch.tensor(action_list), \
                                                           torch.tensor(reward_list), \
                                                           torch.tensor(state_next_list, dtype=torch.float), \
                                                           torch.tensor(done_list, dtype=torch.float), \
                                                           torch.tensor(prob_a_list)

        return state, action, reward, s_next, done, prob_action

    def train_net(self):
        for handle in range(len(self.memory)):
            agent_episode_history = self.memory.get_transitions(handle)
            if len(agent_episode_history) > 0:
                # convert the replay buffer to torch tensors (arrays)
                state, action, reward, state_next, done, prob_action = \
                    self._convert_transitions_to_torch(agent_episode_history)

                # run K_EPOCH optimisation steps
                for i in range(K_EPOCH):
                    # temporal difference function / and prepare advantage function data
                    estimated_target_value = reward + GAMMA * self.v(state_next) * (1.0 - done)
                    difference_to_expected_value_deltas = estimated_target_value - self.v(state)
                    difference_to_expected_value_deltas = difference_to_expected_value_deltas.detach().numpy()

                    # build advantage function and convert it to torch tensor (array)
                    advantage_list = []
                    advantage_value = 0.0
                    for difference_to_expected_value_t in difference_to_expected_value_deltas[::-1]:
                        advantage_value = LMBDA * advantage_value + difference_to_expected_value_t[0]
                        advantage_list.append([advantage_value])
                    advantage_list.reverse()
                    advantage = torch.tensor(advantage_list, dtype=torch.float)

                    pi_action = self.pi(state, softmax_dim=1).gather(1, action)
                    ratio = torch.exp(torch.log(pi_action) - torch.log(prob_action))  # a/b == exp(log(a)-log(b))
                    # Normal Policy Gradient objective
                    surrogate_objective = ratio * advantage
                    # clipped version of Normal Policy Gradient objective
                    clipped_surrogate_objective = torch.clamp(ratio * advantage, 1 - EPS_CLIP, 1 + EPS_CLIP)
                    # value function loss
                    value_loss = F.mse_loss(self.v(state), estimated_target_value.detach())
                    # loss
                    loss = -torch.min(surrogate_objective, clipped_surrogate_objective) + value_loss

                    # update policy and actor networks
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()

                    # store current loss to the agent
                    self.loss = loss.mean().detach().numpy()
        self.memory.reset()

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
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=device))
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
        policy = PPOAgent(self.state_size, self.action_size)
        policy.fc1 = copy.deepcopy(self.fc1)
        policy.fc_pi = copy.deepcopy(self.fc_pi)
        policy.fc_v = copy.deepcopy(self.fc_v)
        policy.optimizer = copy.deepcopy(self.optimizer)
        return self
