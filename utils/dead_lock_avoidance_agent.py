import matplotlib.pyplot as plt
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions

from reinforcement_learning.policy import Policy
from utils.shortest_Distance_walker import ShortestDistanceWalker


class MyWalker(ShortestDistanceWalker):
    def __init__(self, env: RailEnv, agent_positions):
        super().__init__(env)
        self.shortest_distance_agent_map = np.zeros((self.env.get_num_agents(),
                                                     self.env.height,
                                                     self.env.width),
                                                    dtype=int) - 1

        self.agent_positions = agent_positions

        self.agent_map = {}

    def getData(self):
        return self.shortest_distance_agent_map

    def callback(self, handle, agent, position, direction, action):
        opp_a = self.agent_positions[position]
        if opp_a != -1 and opp_a != handle:
            d = self.agent_map.get(handle, [])
            d.append(opp_a)
            if self.env.agents[opp_a].direction != direction:
                self.agent_map.update({handle: d})
        self.shortest_distance_agent_map[(handle, position[0], position[1])] = direction


class DeadLockAvoidanceAgent(Policy):
    def __init__(self, env: RailEnv, state_size, action_size):
        self.env = env
        self.action_size = action_size
        self.state_size = state_size
        self.memory = []
        self.loss = 0
        self.agent_can_move = {}

    def step(self, handle, state, action, reward, next_state, done):
        pass

    def act(self, handle, state, eps=0.):
        agent = self.env.agents[handle]
        #if handle > self.env._elapsed_steps:
        #    return RailEnvActions.STOP_MOVING
        if agent.status == RailAgentStatus.ACTIVE:
            self.active_agent_cnt += 1
        #if agent.status > 20:
        #    return RailEnvActions.STOP_MOVING
        check = self.agent_can_move.get(handle, None)
        if check is None:
            # print(handle, RailEnvActions.STOP_MOVING)
            return RailEnvActions.STOP_MOVING

        return check[3]

    def reset(self):
        pass

    def start_step(self):
        self.active_agent_cnt = 0
        self.shortest_distance_mapper()

    def end_step(self):
        # print("#A:", self.active_agent_cnt, "/", self.env.get_num_agents(),self.env._elapsed_steps)
        pass

    def get_actions(self):
        pass

    def shortest_distance_mapper(self):

        # build map with agent positions (only active agents)
        agent_positions = np.zeros((self.env.height, self.env.width), dtype=int) - 1
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.status <= RailAgentStatus.ACTIVE:
                if agent.position is not None:
                    agent_positions[agent.position] = handle

        my_walker = MyWalker(self.env, agent_positions)
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.status <= RailAgentStatus.ACTIVE:
                my_walker.walk_to_target(handle)
        self.shortest_distance_agent_map = my_walker.getData()

        self.agent_can_move = {}
        agent_positions_map = np.clip(agent_positions + 1, 0, 1)
        for handle in range(self.env.get_num_agents()):
            opp_agents = my_walker.agent_map.get(handle, [])
            me = np.clip(self.shortest_distance_agent_map[handle] + 1, 0, 1)
            next_step_ok = True
            next_position, next_direction, action = my_walker.walk_one_step(handle)
            for opp_a in opp_agents:
                opp = np.clip(self.shortest_distance_agent_map[opp_a] + 1, 0, 1)
                delta = np.clip(me - opp - agent_positions_map, 0, 1)
                if (np.sum(delta) > 1):
                    next_step_ok = False
            if next_step_ok:
                self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if False:
            a = np.floor(np.sqrt(self.env.get_num_agents()))
            b = np.ceil(self.env.get_num_agents() / a)
            for handle in range(self.env.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(self.shortest_distance_agent_map[handle])
            # plt.colorbar()
            plt.show(block=False)
            plt.pause(0.001)
