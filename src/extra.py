#
# Author Adrian Egli
#
# This observation solves the FLATland challenge ROUND 1 - with agent's done 19.3%
#
# Training:
# For the training of the PPO RL agent I showed 10k episodes - The episodes used for the training
# consists of 1..20 agents on a 50x50 grid. Thus the RL agent has to learn to handle 1 upto 20 agents.
#
#   - https://github.com/mitchellgoffpc/flatland-training
# ./adrian_egli_ppo_training_done.png
#
# The key idea behind this observation is that agent's can not freely choose where they want.
#
# ./images/adrian_egli_decisions.png
# ./images/adrian_egli_info.png
# ./images/adrian_egli_start.png
# ./images/adrian_egli_target.png
#
# Private submission
# http://gitlab.aicrowd.com/adrian_egli/neurips2020-flatland-starter-kit/issues/8

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions

from src.ppo.agent import Agent


# ------------------------------------- USE FAST_METHOD from FLATland master ------------------------------------------
# Adrian Egli performance fix (the fast methods brings more than 50%)

def fast_isclose(a, b, rtol):
    return (a < (b + rtol)) or (a < (b - rtol))


def fast_clip(position: (int, int), min_value: (int, int), max_value: (int, int)) -> bool:
    return (
        max(min_value[0], min(position[0], max_value[0])),
        max(min_value[1], min(position[1], max_value[1]))
    )


def fast_argmax(possible_transitions: (int, int, int, int)) -> bool:
    if possible_transitions[0] == 1:
        return 0
    if possible_transitions[1] == 1:
        return 1
    if possible_transitions[2] == 1:
        return 2
    return 3


def fast_position_equal(pos_1: (int, int), pos_2: (int, int)) -> bool:
    return pos_1[0] == pos_2[0] and pos_1[1] == pos_2[1]


def fast_count_nonzero(possible_transitions: (int, int, int, int)):
    return possible_transitions[0] + possible_transitions[1] + possible_transitions[2] + possible_transitions[3]


# ------------------------------- END - USE FAST_METHOD from FLATland master ------------------------------------------

class Extra(ObservationBuilder):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.observation_dim = 26
        self.agent = None
        self.random_agent_starter = []

    def build_data(self):
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self.debug_render_list = []
        self.debug_render_path_list = []
        if self.env is not None:
            self.find_all_cell_where_agent_can_choose()

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
        agents_near_to_switch = False
        agents_near_to_switch_all = False
        if position in switches.keys():
            agents_on_switch = direction in switches[position]

        if position in switches_neighbours.keys():
            new_cell = get_new_position(position, direction)
            if new_cell in switches.keys():
                if not direction in switches[new_cell]:
                    agents_near_to_switch = direction in switches_neighbours[position]
            else:
                agents_near_to_switch = direction in switches_neighbours[position]

            agents_near_to_switch_all = direction in switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all

    def required_agent_descision(self):
        agents_can_choose = {}
        agents_on_switch = {}
        agents_near_to_switch = {}
        agents_near_to_switch_all = {}
        for a in range(self.env.get_num_agents()):
            ret_agents_on_switch, ret_agents_near_to_switch, ret_agents_near_to_switch_all = \
                self.check_agent_descision(
                    self.env.agents[a].position,
                    self.env.agents[a].direction)
            agents_on_switch.update({a: ret_agents_on_switch})
            ready_to_depart = self.env.agents[a].status == RailAgentStatus.READY_TO_DEPART
            agents_near_to_switch.update({a: (ret_agents_near_to_switch and not ready_to_depart)})

            agents_can_choose.update({a: agents_on_switch[a] or agents_near_to_switch[a]})

            agents_near_to_switch_all.update({a: (ret_agents_near_to_switch_all and not ready_to_depart)})

        return agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all

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

    def normalize_observation(self, obsData):
        return obsData

    def is_collision(self, obsData):
        return False

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

    def _explore(self, handle, new_position, new_direction, depth=0):
        has_opp_agent = 0
        has_same_agent = 0
        visited = []

        # stop exploring (max_depth reached)
        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, visited

        # max_explore_steps = 100
        cnt = 0
        while cnt < 100:
            cnt += 1

            visited.append(new_position)
            opp_a = self.env.agent_positions[new_position]
            if opp_a != -1 and opp_a != handle:
                if self.env.agents[opp_a].direction != new_direction:
                    # opp agent found
                    has_opp_agent = 1
                    return has_opp_agent, has_same_agent, visited
                else:
                    has_same_agent = 1
                    return has_opp_agent, has_same_agent, visited

            # convert one-hot encoding to 0,1,2,3
            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all = \
                self.check_agent_descision(new_position, new_direction)
            if agents_near_to_switch:
                return has_opp_agent, has_same_agent, visited

            if agents_on_switch:
                for dir_loop in range(4):
                    if possible_transitions[dir_loop] == 1:
                        hoa, hsa, v = self._explore(handle,
                                                    get_new_position(new_position, dir_loop),
                                                    dir_loop,
                                                    depth + 1)
                        visited.append(v)
                        has_opp_agent = 0.5 * (has_opp_agent + hoa)
                        has_same_agent = 0.5 * (has_same_agent + hsa)
                return has_opp_agent, has_same_agent, visited
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)
        return has_opp_agent, has_same_agent, visited

    def get(self, handle):
        # all values are [0,1]
        # observation[0]  : 1 path towards target (direction 0) / otherwise 0 -> path is longer or there is no path
        # observation[1]  : 1 path towards target (direction 1) / otherwise 0 -> path is longer or there is no path
        # observation[2]  : 1 path towards target (direction 2) / otherwise 0 -> path is longer or there is no path
        # observation[3]  : 1 path towards target (direction 3) / otherwise 0 -> path is longer or there is no path
        # observation[4]  : int(agent.status == RailAgentStatus.READY_TO_DEPART)
        # observation[5]  : int(agent.status == RailAgentStatus.ACTIVE)
        # observation[6]  : int(agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED)
        # observation[7]  : current agent is located at a switch, where it can take a routing decision
        # observation[8]  : current agent is located at a cell, where it has to take a stop-or-go decision
        # observation[9]  : current agent is located one step before/after a switch
        # observation[10] : 1 if there is a path (track/branch) otherwise 0 (direction 0)
        # observation[11] : 1 if there is a path (track/branch) otherwise 0 (direction 1)
        # observation[12] : 1 if there is a path (track/branch) otherwise 0 (direction 2)
        # observation[13] : 1 if there is a path (track/branch) otherwise 0 (direction 3)
        # observation[14] : If there is a path with step (direction 0) and there is a agent with opposite direction -> 1
        # observation[15] : If there is a path with step (direction 1) and there is a agent with opposite direction -> 1
        # observation[16] : If there is a path with step (direction 2) and there is a agent with opposite direction -> 1
        # observation[17] : If there is a path with step (direction 3) and there is a agent with opposite direction -> 1
        # observation[18] : If there is a path with step (direction 0) and there is a agent with same direction -> 1
        # observation[19] : If there is a path with step (direction 1) and there is a agent with same direction -> 1
        # observation[20] : If there is a path with step (direction 2) and there is a agent with same direction -> 1
        # observation[21] : If there is a path with step (direction 3) and there is a agent with same direction -> 1

        observation = np.zeros(self.observation_dim)
        visited = []
        agent = self.env.agents[handle]

        agent_done = False
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
            observation[4] = 1
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
            observation[5] = 1
        else:
            observation[6] = 1
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
                orientation = np.argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)

                    new_cell_dist = distance_map[handle,
                                                 new_position[0], new_position[1],
                                                 branch_direction]
                    if not (np.math.isinf(new_cell_dist) and np.math.isinf(current_cell_dist)):
                        observation[dir_loop] = int(new_cell_dist < current_cell_dist)

                    has_opp_agent, has_same_agent, v = self._explore(handle, new_position, branch_direction)
                    visited.append(v)

                    observation[10 + dir_loop] = 1
                    observation[14 + dir_loop] = has_opp_agent
                    observation[18 + dir_loop] = has_same_agent

                    opp_a = self.env.agent_positions[new_position]
                    if opp_a != -1 and opp_a != handle:
                        observation[22 + dir_loop] = 1

        agents_on_switch, \
        agents_near_to_switch, \
        agents_near_to_switch_all = \
            self.check_agent_descision(agent_virtual_position, agent.direction)
        observation[7] = int(agents_on_switch)
        observation[8] = int(agents_near_to_switch)
        observation[9] = int(agents_near_to_switch_all)

        self.env.dev_obs_dict.update({handle: visited})

        return observation

    def rl_agent_act_ADRIAN(self, observation, info, eps=0.0):
        self.loadAgent()
        action_dict = {}
        for a in range(self.env.get_num_agents()):
            if info['action_required'][a]:
                action_dict[a] = self.agent.act(observation[a], eps=eps)
                # action_dict[a] = np.random.randint(5)
            else:
                action_dict[a] = RailEnvActions.DO_NOTHING

        return action_dict

    def rl_agent_act(self, observation, info, eps=0.0):
        if len(self.random_agent_starter) != len(self.env.get_num_agents()):
            self.random_agent_starter = np.random.random(self.env.get_num_agents()) * 1000.0
            self.loadAgent()

        action_dict = {}
        for a in range(self.env.get_num_agents()):
            if self.random_agent_starter[a] > self.env._elapsed_steps:
                action_dict[a] = RailEnvActions.STOP_MOVING
            elif info['action_required'][a]:
                action_dict[a] = self.agent.act(observation[a], eps=eps)
                # action_dict[a] = np.random.randint(5)
            else:
                action_dict[a] = RailEnvActions.DO_NOTHING

        return action_dict

    def rl_agent_act_ADRIAN_01(self, observation, info, eps=0.0):
        self.loadAgent()
        action_dict = {}
        active_cnt = 0
        for a in range(self.env.get_num_agents()):
            if active_cnt < 10 or self.env.agents[a].status == RailAgentStatus.ACTIVE:
                if observation[a][6] == 1:
                    active_cnt += int(self.env.agents[a].status == RailAgentStatus.ACTIVE)
                    action_dict[a] = RailEnvActions.STOP_MOVING
                else:
                    active_cnt += int(self.env.agents[a].status < RailAgentStatus.DONE)
                    if (observation[a][7] + observation[a][8] + observation[a][9] > 0) or \
                            (self.env.agents[a].status < RailAgentStatus.ACTIVE):
                        if info['action_required'][a]:
                            action_dict[a] = self.agent.act(observation[a], eps=eps)
                            # action_dict[a] = np.random.randint(5)
                        else:
                            action_dict[a] = RailEnvActions.MOVE_FORWARD
                    else:
                        action_dict[a] = RailEnvActions.MOVE_FORWARD
            else:
                action_dict[a] = RailEnvActions.STOP_MOVING

        return action_dict

    def loadAgent(self):
        if self.agent is not None:
            return
        self.state_size = self.env.obs_builder.observation_dim
        self.action_size = 5
        print("action_size: ", self.action_size)
        print("state_size: ", self.state_size)
        self.agent = Agent(self.state_size, self.action_size, 0)
        self.agent.load('./checkpoints/', 0, 1.0)
