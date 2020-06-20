import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 0.5e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works

# how often to update the network
UPDATE_EVERY = 20
UPDATE_EVERY_FINAL = 10
UPDATE_EVERY_AGENT_CANT_CHOOSE = 200


double_dqn = True  # If using double dqn algorithm
input_channels = 5  # Number of Input channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

USE_OPTIMIZER = optim.Adam
# USE_OPTIMIZER = optim.RMSprop
print(USE_OPTIMIZER)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, net_type, seed, double_dqn=True, input_channels=5):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.version = net_type
        self.double_dqn = double_dqn
        # Q-Network
        if self.version == "Conv":
            self.qnetwork_local = QNetwork2(state_size, action_size, seed, input_channels).to(device)
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        self.optimizer = USE_OPTIMIZER(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory_final = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory_agent_can_not_choose = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.final_step = {}

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step_final = 0
        self.t_step_agent_can_not_choose = 0

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
            print(filename + ".local -> ok")
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))
            print(filename + ".target -> ok")
        self.optimizer = USE_OPTIMIZER(self.qnetwork_local.parameters(), lr=LR)

    def _update_model(self, switch=0):
        # Learn every UPDATE_EVERY time steps.
        # If enough samples are available in memory, get random subset and learn
        if switch == 0:
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
        elif switch == 1:
            self.t_step_final = (self.t_step_final + 1) % UPDATE_EVERY_FINAL
            if self.t_step_final == 0:
                if len(self.memory_final) > BATCH_SIZE:
                    experiences = self.memory_final.sample()
                    self.learn(experiences, GAMMA)
        else:
            # If enough samples are available in memory_agent_can_not_choose, get random subset and learn
            self.t_step_agent_can_not_choose = (self.t_step_agent_can_not_choose + 1) % UPDATE_EVERY_AGENT_CANT_CHOOSE
            if self.t_step_agent_can_not_choose == 0:
                if len(self.memory_agent_can_not_choose) > BATCH_SIZE:
                    experiences = self.memory_agent_can_not_choose.sample()
                    self.learn(experiences, GAMMA)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self._update_model(0)

    def step_agent_can_not_choose(self, state, action, reward, next_state, done):
        # Save experience in replay memory_agent_can_not_choose
        self.memory_agent_can_not_choose.add(state, action, reward, next_state, done)
        self._update_model(2)

    def add_final_step(self, agent_handle, state, action, reward, next_state, done):
        if self.final_step.get(agent_handle) is None:
            self.final_step.update({agent_handle: [state, action, reward, next_state, done]})

    def make_final_step(self, additional_reward=0):
        for _, item in self.final_step.items():
            state = item[0]
            action = item[1]
            reward = item[2] + additional_reward
            next_state = item[3]
            done = item[4]
            self.memory_final.add(state, action, reward, next_state, done)
            self._update_model(1)
        self._reset_final_step()

    def _reset_final_step(self):
        self.final_step = {}

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            Q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

            # Compute Q targets for current states

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states


import copy
import os
import random
from collections import namedtuple, deque, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agent.model import QNetwork2, QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.95  # discount factor 0.99
TAU = 0.5e-4  # for soft update of target parameters
LR = 0.5e-3  # learning rate 0.5e-4 works

# how often to update the network
UPDATE_EVERY = 40
UPDATE_EVERY_FINAL = 1000
UPDATE_EVERY_AGENT_CANT_CHOOSE = 200

double_dqn = True  # If using double dqn algorithm
input_channels = 5  # Number of Input channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

USE_OPTIMIZER = optim.Adam
# USE_OPTIMIZER = optim.RMSprop
print(USE_OPTIMIZER)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, net_type, seed, double_dqn=True, input_channels=5):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.version = net_type
        self.double_dqn = double_dqn
        # Q-Network
        if self.version == "Conv":
            self.qnetwork_local = QNetwork2(state_size, action_size, seed, input_channels).to(device)
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        self.optimizer = USE_OPTIMIZER(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory_final = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory_agent_can_not_choose = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.final_step = {}

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_step_final = 0
        self.t_step_agent_can_not_choose = 0

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        print("try to load: " + filename)
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
            print(filename + ".local -> ok")
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))
            print(filename + ".target -> ok")
        self.optimizer = USE_OPTIMIZER(self.qnetwork_local.parameters(), lr=LR)

    def _update_model(self, switch=0):
        # Learn every UPDATE_EVERY time steps.
        # If enough samples are available in memory, get random subset and learn
        if switch == 0:
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
        elif switch == 1:
            self.t_step_final = (self.t_step_final + 1) % UPDATE_EVERY_FINAL
            if self.t_step_final == 0:
                if len(self.memory_final) > BATCH_SIZE:
                    experiences = self.memory_final.sample()
                    self.learn(experiences, GAMMA)
        else:
            # If enough samples are available in memory_agent_can_not_choose, get random subset and learn
            self.t_step_agent_can_not_choose = (self.t_step_agent_can_not_choose + 1) % UPDATE_EVERY_AGENT_CANT_CHOOSE
            if self.t_step_agent_can_not_choose == 0:
                if len(self.memory_agent_can_not_choose) > BATCH_SIZE:
                    experiences = self.memory_agent_can_not_choose.sample()
                    self.learn(experiences, GAMMA)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self._update_model(0)

    def step_agent_can_not_choose(self, state, action, reward, next_state, done):
        # Save experience in replay memory_agent_can_not_choose
        self.memory_agent_can_not_choose.add(state, action, reward, next_state, done)
        self._update_model(2)

    def add_final_step(self, agent_handle, state, action, reward, next_state, done):
        if self.final_step.get(agent_handle) is None:
            self.final_step.update({agent_handle: [state, action, reward, next_state, done]})
            return True
        else:
            return False

    def make_final_step(self, additional_reward=0):
        for _, item in self.final_step.items():
            state = item[0]
            action = item[1]
            reward = item[2] + additional_reward
            next_state = item[3]
            done = item[4]
            self.memory_final.add(state, action, reward, next_state, done)
            self._update_model(1)
        self._reset_final_step()

    def _reset_final_step(self):
        self.final_step = {}

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), False
        else:
            return random.choice(np.arange(self.action_size)), True

    def learn(self, experiences, gamma):

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            Q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

            # Compute Q targets for current states

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
