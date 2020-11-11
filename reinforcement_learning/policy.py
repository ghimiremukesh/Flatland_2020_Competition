class Policy:
    def step(self, handle, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps=0.):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def start_step(self):
        pass

    def end_step(self):
        pass

    def load_replay_buffer(self, filename):
        pass

    def test(self):
        pass

    def reset(self):
        pass

    def clone(self):
        return self