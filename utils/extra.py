import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions, fast_argmax, fast_count_nonzero

from reinforcement_learning.policy import Policy
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent, DeadlockAvoidanceShortestDistanceWalker


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


class Extra(ObservationBuilder):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.observation_dim = 31

    def build_data(self):
        self.dead_lock_avoidance_agent = None
        if self.env is not None:
            self.env.dev_obs_dict = {}
            self.dead_lock_avoidance_agent = DeadLockAvoidanceAgent(self.env, 5, False)

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


    def _check_dead_lock_at_branching_position(self, handle, new_position, branch_direction):
        _, full_shortest_distance_agent_map = self.dead_lock_avoidance_agent.shortest_distance_walker.getData()
        opp_agents = self.dead_lock_avoidance_agent.shortest_distance_walker.opp_agent_map.get(handle, [])
        same_agents = self.dead_lock_avoidance_agent.shortest_distance_walker.same_agent_map.get(handle,[])
        local_walker = DeadlockAvoidanceShortestDistanceWalker(
            self.env,
            self.dead_lock_avoidance_agent.shortest_distance_walker.agent_positions,
            self.dead_lock_avoidance_agent.shortest_distance_walker.switches)
        local_walker.walk_to_target(handle, new_position, branch_direction)
        shortest_distance_agent_map, _ = self.dead_lock_avoidance_agent.shortest_distance_walker.getData()
        my_shortest_path_to_check = shortest_distance_agent_map[handle]
        next_step_ok = self.dead_lock_avoidance_agent.check_agent_can_move(handle,
                                                                           my_shortest_path_to_check,
                                                                           opp_agents,
                                                                           same_agents,
                                                                           full_shortest_distance_agent_map)
        return next_step_ok

    def _explore(self, handle, new_position, new_direction, depth=0):

        has_opp_agent = 0
        has_same_agent = 0
        has_switch = 0
        visited = []

        # stop exploring (max_depth reached)
        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, has_switch, visited

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
                    return has_opp_agent, has_same_agent, has_switch, visited
                else:
                    has_same_agent = 1
                    return has_opp_agent, has_same_agent, has_switch, visited

            # convert one-hot encoding to 0,1,2,3
            agents_on_switch, \
            agents_near_to_switch, \
            agents_near_to_switch_all, \
            agents_on_switch_all = \
                self.check_agent_descision(new_position, new_direction)
            if agents_near_to_switch:
                return has_opp_agent, has_same_agent, has_switch, visited

            possible_transitions = self.env.rail.get_transitions(*new_position, new_direction)
            if agents_on_switch:
                f = 0
                for dir_loop in range(4):
                    if possible_transitions[dir_loop] == 1:
                        f += 1
                        hoa, hsa, hs, v = self._explore(handle,
                                                        get_new_position(new_position, dir_loop),
                                                        dir_loop,
                                                        depth + 1)
                        visited.append(v)
                        has_opp_agent += hoa
                        has_same_agent += hsa
                        has_switch += hs
                f = max(f, 1.0)
                return has_opp_agent / f, has_same_agent / f, has_switch / f, visited
            else:
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

        return has_opp_agent, has_same_agent, has_switch, visited

    def get(self, handle):

        if handle == 0:
            self.dead_lock_avoidance_agent.start_step()

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
        # observation[26] : Is there a deadlock signal on shortest path walk(s) (direction 0)-> 1
        # observation[27] : Is there a deadlock signal on shortest path walk(s) (direction 1)-> 1
        # observation[28] : Is there a deadlock signal on shortest path walk(s) (direction 2)-> 1
        # observation[29] : Is there a deadlock signal on shortest path walk(s) (direction 3)-> 1
        # observation[30] : Is there a deadlock signal on shortest path walk(s) (current position check)-> 1

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

                    has_opp_agent, has_same_agent, has_switch, v = self._explore(handle, new_position, branch_direction)
                    visited.append(v)

                    observation[10 + dir_loop] = 1
                    observation[14 + dir_loop] = has_opp_agent
                    observation[18 + dir_loop] = has_same_agent
                    observation[22 + dir_loop] = has_switch

                    next_step_ok = self._check_dead_lock_at_branching_position(handle, new_position, branch_direction)
                    if next_step_ok:
                        observation[26 + dir_loop] = 1

        agents_on_switch, \
        agents_near_to_switch, \
        agents_near_to_switch_all, \
        agents_on_switch_all = \
            self.check_agent_descision(agent_virtual_position, agent.direction)
        observation[7] = int(agents_on_switch)
        observation[8] = int(agents_near_to_switch)
        observation[9] = int(agents_near_to_switch_all)

        observation[30] = int(self.dead_lock_avoidance_agent.act(handle, None, 0) == RailEnvActions.STOP_MOVING)

        self.env.dev_obs_dict.update({handle: visited})

        return observation

    @staticmethod
    def agent_can_choose(observation):
        return observation[7] == 1 or observation[8] == 1
