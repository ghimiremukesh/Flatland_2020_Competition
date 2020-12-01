import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    def step(self, handle, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps=0.):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def start_step(self,train):
        pass

    def end_step(self,train):
        pass

    def start_episode(self,train):
        pass

    def end_episode(self,train):
        pass

    def load_replay_buffer(self, filename):
        pass

    def test(self):
        pass

    def reset(self):
        pass

    def clone(self):
        return self