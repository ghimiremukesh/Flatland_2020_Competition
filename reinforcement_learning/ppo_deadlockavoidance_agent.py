from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import Policy
from utils.agent_can_choose_helper import AgentCanChooseHelper
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent


class MultiDecisionAgent(Policy):

    def __init__(self, env: RailEnv, state_size, action_size, learning_agent):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.learning_agent = learning_agent
        self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(self.env, action_size, False)
        self.agent_can_choose_helper = AgentCanChooseHelper()
        self.memory = self.learning_agent.memory
        self.loss = self.learning_agent.loss

    def step(self, handle, state, action, reward, next_state, done):
        self.dead_lock_avoidance_agent.step(handle, state, action, reward, next_state, done)
        self.learning_agent.step(handle, state, action, reward, next_state, done)
        self.loss = self.learning_agent.loss

    def act(self, handle, state, eps=0.):
        agent = self.env.agents[handle]
        position = agent.position
        if position is None:
            position = agent.initial_position
        direction = agent.direction
        if agent.status < RailAgentStatus.DONE:
            agents_on_switch, agents_near_to_switch, _, _ = \
                self.agent_can_choose_helper.check_agent_decision(position, direction)
            if agents_on_switch or agents_near_to_switch:
                return self.learning_agent.act(handle, state, eps)
            else:
                act = self.dead_lock_avoidance_agent.act(handle, state, -1.0)
                if self.action_size == 4:
                    act = max(act - 1, 0)
                return act
        # Agent is still at target cell
        return RailEnvActions.DO_NOTHING

    def save(self, filename):
        self.dead_lock_avoidance_agent.save(filename)
        self.learning_agent.save(filename)

    def load(self, filename):
        self.dead_lock_avoidance_agent.load(filename)
        self.learning_agent.load(filename)

    def start_step(self, train):
        self.dead_lock_avoidance_agent.start_step(train)
        self.learning_agent.start_step(train)

    def end_step(self, train):
        self.dead_lock_avoidance_agent.end_step(train)
        self.learning_agent.end_step(train)

    def start_episode(self, train):
        self.dead_lock_avoidance_agent.start_episode(train)
        self.learning_agent.start_episode(train)

    def end_episode(self, train):
        self.dead_lock_avoidance_agent.end_episode(train)
        self.learning_agent.end_episode(train)

    def load_replay_buffer(self, filename):
        self.dead_lock_avoidance_agent.load_replay_buffer(filename)
        self.learning_agent.load_replay_buffer(filename)

    def test(self):
        self.dead_lock_avoidance_agent.test()
        self.learning_agent.test()

    def reset(self, env: RailEnv):
        self.env = env
        self.agent_can_choose_helper.build_data(env)
        self.dead_lock_avoidance_agent.reset(env)
        self.learning_agent.reset(env)

    def clone(self):
        return self
