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


class PPOModelNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(PPOModelNetwork, self).__init__()
        self.fc_layer_1_val = nn.Linear(state_size, hidsize1)
        self.shared_network = nn.Linear(hidsize1, hidsize2)
        self.fc_policy_pi = nn.Linear(hidsize2, action_size)
        self.fc_value = nn.Linear(hidsize2, 1)

    def forward(self, x):
        val = F.relu(self.fc_layer_1_val(x))
        val = F.relu(self.shared_network(val))
        return val

    def policy_pi_estimator(self, x, softmax_dim=0):
        x = F.tanh(self.forward(x))
        x = self.fc_policy_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def value_estimator(self, x):
        x = F.tanh(self.forward(x))
        v = self.fc_value(x)
        return v

    # Checkpointing methods
    def save(self, filename):
        # print("Saving model from checkpoint:", filename)
        torch.save(self.shared_network.state_dict(), filename + ".fc_shared")
        torch.save(self.fc_policy_pi.state_dict(), filename + ".fc_pi")
        torch.save(self.fc_value.state_dict(), filename + ".fc_v")

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
        self.shared_network = self._load(self.shared_network, filename + ".fc_shared")
        self.fc_policy_pi = self._load(self.fc_policy_pi, filename + ".fc_pi")
        self.fc_value = self._load(self.fc_value, filename + ".fc_v")


class PPOAgent(Policy):
    def __init__(self, state_size, action_size):
        super(PPOAgent, self).__init__()
        self.memory = DataBuffers()
        self.loss = 0
        self.value_model_network = PPOModelNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.value_model_network.parameters(), lr=LEARNING_RATE)

    def reset(self):
        pass

    def act(self, state, eps=None):
        prob = self.value_model_network.policy_pi_estimator(torch.from_numpy(state).float())
        m = Categorical(prob)
        a = m.sample().item()
        return a

    def step(self, handle, state, action, reward, next_state, done):
        # Record the results of the agent's action as transition
        prob = self.value_model_network.policy_pi_estimator(torch.from_numpy(state).float())
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
                    estimated_target_value = rewards + GAMMA * self.value_model_network.value_estimator(states_next) * (
                            1.0 - dones)
                    difference_to_expected_value_deltas = estimated_target_value - self.value_model_network.value_estimator(
                        states)
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
                    pi_actions = self.value_model_network.policy_pi_estimator(states, softmax_dim=1).gather(1, actions)
                    # calculate the ratios
                    ratios = torch.exp(torch.log(pi_actions) - torch.log(probs_action))
                    # Normal Policy Gradient objective
                    surrogate_objective = ratios * advantages
                    # clipped version of Normal Policy Gradient objective
                    clipped_surrogate_objective = torch.clamp(ratios * advantages, 1 - EPS_CLIP, 1 + EPS_CLIP)
                    # value function loss
                    value_loss = F.mse_loss(self.value_model_network.value_estimator(states),
                                            estimated_target_value.detach())
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
        self.value_model_network.save(filename)
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
        self.value_model_network.load(filename)
        print("load optimizer from file", filename)
        self.optimizer = self._load(self.optimizer, filename + ".optimizer")

    def clone(self):
        policy = PPOAgent(self.state_size, self.action_size)
        policy.fc1 = copy.deepcopy(self.fc1)
        policy.fc_pi = copy.deepcopy(self.fc_pi)
        policy.fc_v = copy.deepcopy(self.fc_v)
        policy.optimizer = copy.deepcopy(self.optimizer)
        return self
