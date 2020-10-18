class Policy:
    def step(self, handle, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps=0.):
        raise NotImplementedError

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def test(self):
        pass

    def save_replay_buffer(self):
        pass

    def reset(self):
        pass

    def start_step(self):
        pass

    def end_step(self):
        pass
