# import matplotlib.pyplot as plt
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus, RailEnv

from reinforcement_learning.policy import Policy
from utils.shortest_Distance_walker import ShortestDistanceWalker


class ExtraPolicy(Policy):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.loss = 0

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def step(self, handle, state, action, reward, next_state, done):
        pass

    def act(self, handle, state, eps=0.):
        a = 0
        b = 4
        action = RailEnvActions.STOP_MOVING
        if state[2] == 1 and state[10 + a] == 0:
            action = RailEnvActions.MOVE_LEFT
        elif state[3] == 1 and state[11 + a] == 0:
            action = RailEnvActions.MOVE_FORWARD
        elif state[4] == 1 and state[12 + a] == 0:
            action = RailEnvActions.MOVE_RIGHT
        elif state[5] == 1 and state[13 + a] == 0:
            action = RailEnvActions.MOVE_FORWARD

        elif state[6] == 1 and state[10 + b] == 0:
            action = RailEnvActions.MOVE_LEFT
        elif state[7] == 1 and state[11 + b] == 0:
            action = RailEnvActions.MOVE_FORWARD
        elif state[8] == 1 and state[12 + b] == 0:
            action = RailEnvActions.MOVE_RIGHT
        elif state[9] == 1 and state[13 + b] == 0:
            action = RailEnvActions.MOVE_FORWARD

        return action

    def test(self):
        pass


def fast_argmax(possible_transitions: (int, int, int, int)) -> bool:
    if possible_transitions[0] == 1:
        return 0
    if possible_transitions[1] == 1:
        return 1
    if possible_transitions[2] == 1:
        return 2
    return 3


def fast_count_nonzero(possible_transitions: (int, int, int, int)):
    return possible_transitions[0] + possible_transitions[1] + possible_transitions[2] + possible_transitions[3]


class Extra(ObservationBuilder):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.observation_dim = 62

    def shortest_distance_mapper(self):

        class MyWalker(ShortestDistanceWalker):
            def __init__(self, env: RailEnv):
                super().__init__(env)
                self.shortest_distance_agent_counter = np.zeros((self.env.height, self.env.width), dtype=int)
                self.shortest_distance_agent_direction_counter = np.zeros((self.env.height, self.env.width, 4),
                                                                          dtype=int)

            def getData(self):
                return self.shortest_distance_agent_counter, self.shortest_distance_agent_direction_counter

            def callback(self, handle, agent, position, direction, action):
                self.shortest_distance_agent_counter[position] += 1
                self.shortest_distance_agent_direction_counter[(position[0], position[1], direction)] += 1

        my_walker = MyWalker(self.env)
        for handle in range(self.env.get_num_agents()):
            agent = self.env.agents[handle]
            if agent.status <= RailAgentStatus.ACTIVE:
                my_walker.walk_to_target(handle)

        self.shortest_distance_agent_counter, self.shortest_distance_agent_direction_counter = my_walker.getData()

        # plt.imshow(self.shortest_distance_agent_counter)
        # plt.colorbar()
        # plt.show()

    def build_data(self):
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self.debug_render_list = []
        self.debug_render_path_list = []
        if self.env is not None:
            self.find_all_cell_where_agent_can_choose()
            self.agent_positions = np.zeros((self.env.height, self.env.width), dtype=int) - 1
            self.history_direction = np.zeros((self.env.height, self.env.width), dtype=int) - 1
            self.history_same_direction_cnt = np.zeros((self.env.height, self.env.width), dtype=int)
            self.history_time = np.zeros((self.env.height, self.env.width), dtype=int) - 1

        self.shortest_distance_agent_counter = None
        self.shortest_distance_agent_direction_counter = None

    def find_all_cell_where_agent_can_choose(self):

        switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = fast_count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        if pos not in switches.keys():
                            switches.update({pos: [dir]})
                        else:
                            switches[pos].append(dir)

        switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                # look one step forward
                for dir in range(4):
                    pos = (h, w)
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    for d in range(4):
                        if possible_transitions[d] == 1:
                            new_cell = get_new_position(pos, d)
                            if new_cell in switches.keys() and pos not in switches.keys():
                                if pos not in switches_neighbours.keys():
                                    switches_neighbours.update({pos: [dir]})
                                else:
                                    switches_neighbours[pos].append(dir)

        self.switches = switches
        self.switches_neighbours = switches_neighbours

    def check_agent_descision(self, position, direction):
        switches = self.switches
        switches_neighbours = self.switches_neighbours
        agents_on_switch = False
        agents_on_switch_all = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False
        if position in switches.keys():
            agents_on_switch = direction in switches[position]
            agents_on_switch_all = True

        if position in switches_neighbours.keys():
            new_cell = get_new_position(position, direction)
            if new_cell in switches.keys():
                if not direction in switches[new_cell]:
                    agents_near_to_switch = direction in switches_neighbours[position]
            else:
                agents_near_to_switch = direction in switches_neighbours[position]

            agents_near_to_switch_all = direction in switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def required_agent_descision(self):
        agents_can_choose = {}
        agents_on_switch = {}
        agents_on_switch_all = {}
        agents_near_to_switch = {}
        agents_near_to_switch_all = {}
        for a in range(self.env.get_num_agents()):
            ret_agents_on_switch, ret_agents_near_to_switch, ret_agents_near_to_switch_all, ret_agents_on_switch_all = \
                self.check_agent_descision(
                    self.env.agents[a].position,
                    self.env.agents[a].direction)
            agents_on_switch.update({a: ret_agents_on_switch})
            agents_on_switch_all.update({a: ret_agents_on_switch_all})
            ready_to_depart = self.env.agents[a].status == RailAgentStatus.READY_TO_DEPART
            agents_near_to_switch.update({a: (ret_agents_near_to_switch and not ready_to_depart)})

            agents_can_choose.update({a: agents_on_switch[a] or agents_near_to_switch[a]})

            agents_near_to_switch_all.update({a: (ret_agents_near_to_switch_all and not ready_to_depart)})

        return agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def debug_render(self, env_renderer):
        agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = \
            self.required_agent_descision()
        self.env.dev_obs_dict = {}
        for a in range(max(3, self.env.get_num_agents())):
            self.env.dev_obs_dict.update({a: []})

        selected_agent = None
        if agents_can_choose[0]:
            if self.env.agents[0].position is not None:
                self.debug_render_list.append(self.env.agents[0].position)
            else:
                self.debug_render_list.append(self.env.agents[0].initial_position)

        if self.env.agents[0].position is not None:
            self.debug_render_path_list.append(self.env.agents[0].position)
        else:
            self.debug_render_path_list.append(self.env.agents[0].initial_position)

        env_renderer.gl.agent_colors[0] = env_renderer.gl.rgb_s2i("FF0000")
        env_renderer.gl.agent_colors[1] = env_renderer.gl.rgb_s2i("666600")
        env_renderer.gl.agent_colors[2] = env_renderer.gl.rgb_s2i("006666")
        env_renderer.gl.agent_colors[3] = env_renderer.gl.rgb_s2i("550000")

        self.env.dev_obs_dict[0] = self.debug_render_list
        self.env.dev_obs_dict[1] = self.switches.keys()
        self.env.dev_obs_dict[2] = self.switches_neighbours.keys()
        self.env.dev_obs_dict[3] = self.debug_render_path_list

    def reset(self):
        self.build_data()
        return

    def fast_argmax(self, array):
        if array[0] == 1:
            return 0
        if array[1] == 1:
            return 1
        if array[2] == 1:
            return 2
        return 3

    def _explore(self, handle, new_position, new_direction, distance_map, depth):

        may_has_opp_agent = 0
        has_opp_agent = -1
        has_other_target = 0
        has_target = 0
        visited = []

        new_cell_dist = np.inf

        # stop exploring (max_depth reached)
        if depth > self.max_depth:
            return has_opp_agent, may_has_opp_agent, has_other_target, has_target, visited, new_cell_dist

        # max_explore_steps = 100
        cnt = 0
        while cnt < 100:
            cnt += 1
            has_other_target = int(new_position in self.agent_targets)
            new_cell_dist = min(new_cell_dist, distance_map[handle,
                                                            new_position[0], new_position[1],
                                                            new_direction])

            visited.append(new_position)
            has_target = int(self.env.agents[handle].target == new_position)
            opp_a = self.agent_positions[new_position]
            if opp_a != -1 and opp_a != handle:
                possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
                if possible_transitions[self.env.agents[opp_a].direction] < 1:
                    # opp agent found
                    has_opp_agent = opp_a
                    may_has_opp_agent = 1
                    return has_opp_agent, may_has_opp_agent, has_other_target, has_target, visited, new_cell_dist

            # convert one-hot encoding to 0,1,2,3
            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all, \
            agents_on_switch_all = \
                self.check_agent_descision(new_position, new_direction)

            if agents_near_to_switch:
                return has_opp_agent, may_has_opp_agent, has_other_target, has_target, visited, new_cell_dist

            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            if fast_count_nonzero(possible_transitions) > 1:
                may_has_opp_agent_loop = 1
                for dir_loop in range(4):
                    if possible_transitions[dir_loop] == 1:
                        hoa, mhoa, hot, ht, v, min_cell_dist = self._explore(handle,
                                                                             get_new_position(new_position,
                                                                                              dir_loop),
                                                                             dir_loop,
                                                                             distance_map,
                                                                             depth + 1)

                        has_opp_agent = max(has_opp_agent, hoa)
                        may_has_opp_agent_loop = min(may_has_opp_agent_loop, mhoa)
                        has_other_target = max(has_other_target, hot)
                        has_target = max(has_target, ht)
                        visited.append(v)
                        new_cell_dist = min(min_cell_dist, new_cell_dist)
                return has_opp_agent, may_has_opp_agent_loop, has_other_target, has_target, visited, new_cell_dist
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

        return has_opp_agent, may_has_opp_agent, has_other_target, has_target, visited, new_cell_dist

    def get(self, handle):

        if (handle == 0):
            self.updateSharedData()

        # all values are [0,1]
        # observation[0]  : 1 path towards target (direction 0) / otherwise 0 -> path is longer or there is no path
        # observation[1]  : 1 path towards target (direction 1) / otherwise 0 -> path is longer or there is no path
        # observation[2]  : 1 path towards target (direction 2) / otherwise 0 -> path is longer or there is no path
        # observation[3]  : 1 path towards target (direction 3) / otherwise 0 -> path is longer or there is no path
        # observation[4]  : int(agent.status == RailAgentStatus.READY_TO_DEPART)
        # observation[5]  : int(agent.status == RailAgentStatus.ACTIVE)
        # observation[6] : If there is a path with step (direction 0) and there is a agent with opposite direction -> 1
        # observation[7] : If there is a path with step (direction 1) and there is a agent with opposite direction -> 1
        # observation[8] : If there is a path with step (direction 2) and there is a agent with opposite direction -> 1
        # observation[9] : If there is a path with step (direction 3) and there is a agent with opposite direction -> 1

        observation = np.zeros(self.observation_dim)
        visited = []
        agent = self.env.agents[handle]

        agent_done = False
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
            observation[0] = 1
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
            observation[1] = 1
        else:
            agent_virtual_position = (-1, -1)
            agent_done = True

        if not agent_done:
            visited.append(agent_virtual_position)
            distance_map = self.env.distance_map.get()
            current_cell_dist = distance_map[handle,
                                             agent_virtual_position[0], agent_virtual_position[1],
                                             agent.direction]
            possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
            orientation = agent.direction
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)
                    new_cell_dist = distance_map[handle,
                                                 new_position[0], new_position[1],
                                                 branch_direction]

                    has_opp_agent, \
                    may_has_opp_agent, \
                    has_other_target, \
                    has_target, \
                    v, \
                    min_cell_dist = self._explore(handle,
                                                  new_position,
                                                  branch_direction,
                                                  distance_map,
                                                  0)
                    if not (np.math.isinf(new_cell_dist) and np.math.isinf(current_cell_dist)):
                        observation[2 + dir_loop] = int(new_cell_dist < current_cell_dist)

                    new_cell_dist = min(min_cell_dist, new_cell_dist)
                    if not (np.math.isinf(new_cell_dist) and not np.math.isinf(current_cell_dist)):
                        observation[6 + dir_loop] = int(new_cell_dist < current_cell_dist)

                    visited.append(v)

                    observation[10 + dir_loop] = int(has_opp_agent > -1)
                    observation[14 + dir_loop] = may_has_opp_agent
                    observation[18 + dir_loop] = has_other_target
                    observation[22 + dir_loop] = has_target
                    observation[26 + dir_loop] = self.getHistorySameDirection(new_position, branch_direction)
                    observation[30 + dir_loop] = self.getHistoryOppositeDirection(new_position, branch_direction)
                    observation[34 + dir_loop] = self.getTemporalDistance(new_position)
                    observation[38 + dir_loop] = self.getFlowDensity(new_position)
                    observation[42 + dir_loop] = self.getDensitySameDirection(new_position, branch_direction)
                    observation[44 + dir_loop] = self.getDensity(new_position)
                    observation[48 + dir_loop] = int(not np.math.isinf(new_cell_dist))
                    observation[52 + dir_loop] = 1
                    observation[54 + dir_loop] = int(has_opp_agent > handle)

        self.env.dev_obs_dict.update({handle: visited})

        return observation

    def getDensitySameDirection(self, position, direction):
        val = self.shortest_distance_agent_direction_counter[(position[0], position[1], direction)]
        return val / self.env.get_num_agents()

    def getDensity(self, position):
        val = self.shortest_distance_agent_counter[position]
        return val / self.env.get_num_agents()

    def getHistorySameDirection(self, position, direction):
        val = self.history_direction[position]
        if val == -1:
            return -1
        if val == direction:
            return 1
        return 0

    def getHistoryOppositeDirection(self, position, direction):
        val = self.getHistorySameDirection(position, direction)
        if val == -1:
            return -1
        return 1 - val

    def getTemporalDistance(self, position):
        if self.history_time[position] == -1:
            return -1
        val = self.env._elapsed_steps - self.history_time[position]
        if val < 1:
            return 0
        return 1 + np.log(1 + val)

    def getFlowDensity(self, position):
        val = self.env._elapsed_steps - self.history_same_direction_cnt[position]
        return 1 + np.log(1 + val)

    def updateSharedData(self):
        self.shortest_distance_mapper()
        self.agent_positions = np.zeros((self.env.height, self.env.width), dtype=int) - 1
        self.agent_targets = []
        for a in np.arange(self.env.get_num_agents()):
            if self.env.agents[a].status == RailAgentStatus.ACTIVE:
                self.agent_targets.append(self.env.agents[a].target)
                if self.env.agents[a].position is not None:
                    self.agent_positions[self.env.agents[a].position] = a
                    if self.history_direction[self.env.agents[a].position] == self.env.agents[a].direction:
                        self.history_same_direction_cnt[self.env.agents[a].position] += 1
                    else:
                        self.history_same_direction_cnt[self.env.agents[a].position] = 0
                    self.history_direction[self.env.agents[a].position] = self.env.agents[a].direction
                    self.history_time[self.env.agents[a].position] = self.env._elapsed_steps
