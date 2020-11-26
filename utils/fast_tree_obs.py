from typing import List, Optional

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import fast_count_nonzero, fast_argmax, RailEnvActions

from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent

"""
LICENCE for the FastTreeObs Observation Builder  

The observation can be used freely and reused for further submissions. Only the author needs to be referred to
/mentioned in any submissions - if the entire observation or parts, or the main idea is used.

Author: Adrian Egli (adrian.egli@gmail.com)

[Linkedin](https://www.researchgate.net/profile/Adrian_Egli2)
[Researchgate](https://www.linkedin.com/in/adrian-egli-733a9544/)
"""


class FastTreeObs(ObservationBuilder):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.observation_dim = 36

    def build_data(self):
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self.debug_render_list = []
        self.debug_render_path_list = []
        if self.env is not None:
            self.find_all_cell_where_agent_can_choose()
            self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(self.env, 5)
        else:
            self.dead_lock_avoidance_agent = None

    def find_all_switches(self):
        # Search the environment (rail grid) for all switch cells. A switch is a cell where more than one tranisation
        # exists and collect all direction where the switch is a switch.
        self.switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = fast_count_nonzero(possible_transitions)
                    if num_transitions > 1:
                        if pos not in self.switches.keys():
                            self.switches.update({pos: [dir]})
                        else:
                            self.switches[pos].append(dir)

    def find_all_switch_neighbours(self):
        # Collect all cells where is a neighbour to a switch cell. All cells are neighbour where the agent can make
        # just one step and he stands on a switch. A switch is a cell where the agents has more than one transition.
        self.switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                # look one step forward
                for dir in range(4):
                    pos = (h, w)
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    for d in range(4):
                        if possible_transitions[d] == 1:
                            new_cell = get_new_position(pos, d)
                            if new_cell in self.switches.keys() and pos not in self.switches.keys():
                                if pos not in self.switches_neighbours.keys():
                                    self.switches_neighbours.update({pos: [dir]})
                                else:
                                    self.switches_neighbours[pos].append(dir)

    def find_all_cell_where_agent_can_choose(self):
        # prepare the data - collect all cells where the agent can choose more than FORWARD/STOP.
        self.find_all_switches()
        self.find_all_switch_neighbours()

    def check_agent_decision(self, position, direction):
        # Decide whether the agent is
        # - on a switch
        # - at a switch neighbour (near to switch). The switch must be a switch where the agent has more option than
        #   FORWARD/STOP
        # - all switch : doesn't matter whether the agent has more options than FORWARD/STOP
        # - all switch neightbors : doesn't matter the agent has more then one options (transistion) when he reach the
        #   switch
        agents_on_switch = False
        agents_on_switch_all = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False
        if position in self.switches.keys():
            agents_on_switch = direction in self.switches[position]
            agents_on_switch_all = True

        if position in self.switches_neighbours.keys():
            new_cell = get_new_position(position, direction)
            if new_cell in self.switches.keys():
                if not direction in self.switches[new_cell]:
                    agents_near_to_switch = direction in self.switches_neighbours[position]
            else:
                agents_near_to_switch = direction in self.switches_neighbours[position]

            agents_near_to_switch_all = direction in self.switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all, agents_on_switch_all

    def required_agent_decision(self):
        agents_can_choose = {}
        agents_on_switch = {}
        agents_on_switch_all = {}
        agents_near_to_switch = {}
        agents_near_to_switch_all = {}
        for a in range(self.env.get_num_agents()):
            ret_agents_on_switch, ret_agents_near_to_switch, ret_agents_near_to_switch_all, ret_agents_on_switch_all = \
                self.check_agent_decision(
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
            self.required_agent_decision()
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

    def _explore(self, handle, new_position, new_direction, distance_map, depth=0):
        has_opp_agent = 0
        has_same_agent = 0
        has_target = 0
        visited = []
        min_dist = distance_map[handle, new_position[0], new_position[1], new_direction]

        # stop exploring (max_depth reached)
        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, has_target, visited, min_dist

        # max_explore_steps = 100 -> just to ensure that the exploration ends
        cnt = 0
        while cnt < 100:
            cnt += 1

            visited.append(new_position)
            opp_a = self.env.agent_positions[new_position]
            if opp_a != -1 and opp_a != handle:
                if self.env.agents[opp_a].direction != new_direction:
                    # opp agent found -> stop exploring. This would be a strong signal.
                    has_opp_agent = 1
                    return has_opp_agent, has_same_agent, has_target, visited, min_dist
                else:
                    # same agent found
                    # the agent can follow the agent, because this agent is still moving ahead and there shouldn't
                    # be any dead-lock nor other issue -> agent is just walking -> if other agent has a deadlock
                    # this should be avoided by other agents -> one edge case would be when other agent has it's
                    # target on this branch -> thus the agents should scan further whether there will be an opposite
                    # agent walking on same track
                    has_same_agent = 1
                    # !NOT stop exploring! return has_opp_agent, has_same_agent, has_switch, visited,min_dist

            # agents_on_switch == TRUE -> Current cell is a switch where the agent can decide (branch) in exploration
            # agent_near_to_switch == TRUE -> One cell before the switch, where the agent can decide
            #
            agents_on_switch, agents_near_to_switch, _, _ = \
                self.check_agent_decision(new_position, new_direction)

            if agents_near_to_switch:
                # The exploration was walking on a path where the agent can not decide
                # Best option would be MOVE_FORWARD -> Skip exploring - just walking
                return has_opp_agent, has_same_agent, has_target, visited, min_dist

            if self.env.agents[handle].target == new_position:
                has_target = 1

            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            if agents_on_switch:
                orientation = new_direction
                possible_transitions_nonzero = fast_count_nonzero(possible_transitions)
                if possible_transitions_nonzero == 1:
                    orientation = fast_argmax(possible_transitions)

                for dir_loop, branch_direction in enumerate(
                        [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                    # branch the exploration path and aggregate the found information
                    # --- OPEN RESEARCH QUESTION ---> is this good or shall we use full detailed information as
                    # we did in the TreeObservation (FLATLAND) ?
                    if possible_transitions[dir_loop] == 1:
                        hoa, hsa, ht, v, m_dist = self._explore(handle,
                                                                get_new_position(new_position, dir_loop),
                                                                dir_loop,
                                                                distance_map,
                                                                depth + 1)
                        visited.append(v)
                        has_opp_agent += max(hoa, has_opp_agent)
                        has_same_agent += max(hsa, has_same_agent)
                        has_target = max(has_target, ht)
                        min_dist = min(min_dist, m_dist)
                return has_opp_agent, has_same_agent, has_target, visited, min_dist
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

            min_dist = min(min_dist, distance_map[handle, new_position[0], new_position[1], new_direction])

        return has_opp_agent, has_same_agent, has_target, visited, min_dist

    def get_many(self, handles: Optional[List[int]] = None):
        self.dead_lock_avoidance_agent.start_step()
        observations = super().get_many(handles)
        self.dead_lock_avoidance_agent.end_step()
        return observations

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
        # observation[22] : If there is a switch on the path which agent can not use -> 1
        # observation[23] : If there is a switch on the path which agent can not use -> 1
        # observation[24] : If there is a switch on the path which agent can not use -> 1
        # observation[25] : If there is a switch on the path which agent can not use -> 1
        # observation[26] : If there the dead-lock avoidance agent predicts a deadlock -> 1
        # observation[27] : If there the agent can only walk forward or stop -> 1

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
                orientation = fast_argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate([(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)
                    new_cell_dist = distance_map[handle,
                                                 new_position[0], new_position[1],
                                                 branch_direction]
                    if not (np.math.isinf(new_cell_dist) and np.math.isinf(current_cell_dist)):
                        observation[dir_loop] = int(new_cell_dist < current_cell_dist)

                    has_opp_agent, has_same_agent, has_target, v, min_dist = self._explore(handle,
                                                                                           new_position,
                                                                                           branch_direction,
                                                                                           distance_map)
                    visited.append(v)

                    if not (np.math.isinf(min_dist) and np.math.isinf(current_cell_dist)):
                        observation[31 + dir_loop] = int(min_dist < current_cell_dist)
                    observation[11 + dir_loop] = int(not np.math.isinf(new_cell_dist))
                    observation[15 + dir_loop] = has_opp_agent
                    observation[19 + dir_loop] = has_same_agent
                    observation[23 + dir_loop] = has_target
                    observation[27 + dir_loop] = int(np.math.isinf(new_cell_dist))

            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all, \
            agents_on_switch_all = \
                self.check_agent_decision(agent_virtual_position, agent.direction)

            observation[7] = int(agents_on_switch)
            observation[8] = int(agents_on_switch_all)
            observation[9] = int(agents_near_to_switch)
            observation[10] = int(agents_near_to_switch_all)

            action = self.dead_lock_avoidance_agent.act([handle], 0.0)
            observation[31] = int(action == RailEnvActions.STOP_MOVING)

        self.env.dev_obs_dict.update({handle: visited})

        observation[np.isinf(observation)] = -1
        observation[np.isnan(observation)] = -1

        return observation
