import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
from reinforcement_learning.policy import Policy

LEARNING_RATE = 0.1e-4
GAMMA = 0.98
LMBDA = 0.9
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


class GlobalModel(nn.Module):
    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(GlobalModel, self).__init__()
        self._layer_1 = nn.Linear(state_size, hidsize1)
        self.global_network = nn.Linear(hidsize1, hidsize2)

    def get_model(self):
        return self.global_network

    def forward(self, x):
        val = F.relu(self._layer_1(x))
        val = F.relu(self.global_network(val))
        return val

    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.global_network.state_dict(), filename + ".global")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        self.global_network = self._load(self.global_network, filename + ".global")


class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, global_network, hidsize1=128, hidsize2=128):
        super(PolicyNetwork, self).__init__()
        self.global_network = global_network
        self.policy_network = nn.Linear(hidsize2, action_size)

    def forward(self, x, softmax_dim=0):
        x = F.tanh(self.global_network.forward(x))
        x = self.policy_network(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.policy_network.state_dict(), filename + ".policy")

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
        self.policy_network = self._load(self.policy_network, filename + ".policy")


class ValueNetwork(nn.Module):

    def __init__(self, state_size, action_size, global_network, hidsize1=128, hidsize2=128):
        super(ValueNetwork, self).__init__()
        self.global_network = global_network
        self.value_network = nn.Linear(hidsize2, 1)

    def forward(self, x):
        x = F.tanh(self.global_network.forward(x))
        v = self.value_network(x)
        return v

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.value_network.state_dict(), filename + ".value")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        self.value_network = self._load(self.value_network, filename + ".value")


class PPOAgent(Policy):
    def __init__(self, state_size, action_size):
        super(PPOAgent, self).__init__()
        # create the data buffer - collects all transitions (state, action, reward, next_state, action_prob, done)
        # each agent owns its own buffer
        self.memory = DataBuffers()
        # signal - stores the current loss
        self.loss = 0
        # create the global, shared deep neuronal network
        self.global_network = GlobalModel(state_size, action_size)
        # create the "critic" or value network
        self.value_network = ValueNetwork(state_size, action_size, self.global_network)
        # create the "actor" or policy network
        self.policy_network = PolicyNetwork(state_size, action_size, self.global_network)
        # create for each network a optimizer
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=LEARNING_RATE)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)


    def reset(self):
        pass

    def act(self, state, eps=None):
        prob = self.policy_network(torch.from_numpy(state).float())
        m = Categorical(prob)
        a = m.sample().item()
        return a

    def step(self, handle, state, action, reward, next_state, done):
        # Record the results of the agent's action as transition
        prob = self.policy_network(torch.from_numpy(state).float())
        transition = (state, action, reward, next_state, prob[action].item(), done)
        self.memory.push_transition(handle, transition)

    def _convert_transitions_to_torch_tensors(self, transitions_array):
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
                states, actions, rewards, states_next, dones, probs_action = \
                    self._convert_transitions_to_torch_tensors(agent_episode_history)

                # run K_EPOCH optimisation steps
                for i in range(K_EPOCH):
                    # temporal difference function / and prepare advantage function data
                    estimated_target_value = \
                        rewards + GAMMA * self.value_network(states_next) * (1.0 - dones)
                    difference_to_expected_value_deltas = \
                        estimated_target_value - self.value_network(states)
                    difference_to_expected_value_deltas = difference_to_expected_value_deltas.detach().numpy()

                    # build advantage function and convert it to torch tensor (array)
                    advantage_list = []
                    advantage_value = 0.0
                    for difference_to_expected_value_t in difference_to_expected_value_deltas[::-1]:
                        advantage_value = LMBDA * advantage_value + difference_to_expected_value_t[0]
                        advantage_list.append([advantage_value])
                    advantage_list.reverse()
                    advantages = torch.tensor(advantage_list, dtype=torch.float)

                    # estimate pi_action for all state
                    pi_actions = self.policy_network.forward(states, softmax_dim=1).gather(1, actions)
                    # calculate the ratios
                    ratios = torch.exp(torch.log(pi_actions) - torch.log(probs_action))
                    # Normal Policy Gradient objective
                    surrogate_objective = ratios * advantages
                    # clipped version of Normal Policy Gradient objective
                    clipped_surrogate_objective = torch.clamp(ratios * advantages, 1 - EPS_CLIP, 1 + EPS_CLIP)
                    # create value loss function
                    value_loss = F.mse_loss(self.value_network(states),
                                            estimated_target_value.detach())
                    # create final loss function
                    loss = -torch.min(surrogate_objective, clipped_surrogate_objective) + value_loss

                    # update policy ("actor") and value ("critic") networks
                    self.value_optimizer.zero_grad()
                    self.policy_optimizer.zero_grad()
                    loss.mean().backward()
                    self.value_optimizer.step()
                    self.policy_optimizer.step()

                    # store current loss
                    self.loss = loss.mean().detach().numpy()

        self.memory.reset()

    def end_episode(self, train):
        if train:
            self.train_net()

    def save(self, filename):
        self.global_network.save(filename)
        self.value_network.save(filename)
        self.policy_network.save(filename)
        torch.save(self.value_optimizer.state_dict(), filename + ".value_optimizer")
        torch.save(self.policy_optimizer.state_dict(), filename + ".policy_optimizer")

    def _load(self, obj, filename):
        if os.path.exists(filename):
            print(' >> ', filename)
            try:
                obj.load_state_dict(torch.load(filename, map_location=device))
            except:
                print(" >> failed!")
        return obj

    def load(self, filename):
        self.global_network.load(filename)
        self.value_network.load(filename)
        self.policy_network.load(filename)
        self.value_optimizer = self._load(self.value_optimizer, filename + ".value_optimizer")
        self.policy_optimizer = self._load(self.policy_optimizer, filename + ".policy_optimizer")

    def clone(self):
        policy = PPOAgent(self.state_size, self.action_size)
        policy.value_network = copy.deepcopy(self.value_network)
        policy.policy_network = copy.deepcopy(self.policy_network)
        policy.value_optimizer = copy.deepcopy(self.value_optimizer)
        policy.policy_optimizer = copy.deepcopy(self.policy_optimizer)
        return self
