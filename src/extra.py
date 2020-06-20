import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from src.agent.dueling_double_dqn import Agent
from src.observations import normalize_observation

state_size = 179
action_size = 5
print("state_size: ", state_size)
print("action_size: ", action_size)
# Now we load a Double dueling DQN agent
global_rl_agent = Agent(state_size, action_size, "FC", 0)
global_rl_agent.load('./nets/training_best_0.626_agents_5276.pth')


class Extra:
    global_rl_agent = None

    def __init__(self, env: RailEnv):
        self.env = env
        self.rl_agent = global_rl_agent
        self.switches = {}
        self.switches_neighbours = {}
        self.find_all_cell_where_agent_can_choose()
        self.steps_counter = 0

        self.debug_render_list = []
        self.debug_render_path_list = []

    def rl_agent_act(self, observation, max_depth, eps=0.0):

        self.steps_counter += 1
        print(self.steps_counter, self.env.get_num_agents())

        agent_obs = [None] * self.env.get_num_agents()
        for a in range(self.env.get_num_agents()):
            if observation[a]:
                agent_obs[a] = self.generate_state(a, observation, max_depth)

        action_dict = {}
        # estimate whether the agent(s) can freely choose an action
        agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = \
            self.required_agent_descision()

        for a in range(self.env.get_num_agents()):
            if agent_obs[a] is not None:
                if agents_can_choose[a]:
                    act, agent_rnd = self.rl_agent.act(agent_obs[a], eps=eps)

                    l = len(agent_obs[a])
                    if agent_obs[a][l - 3] > 0 and agents_near_to_switch_all[a]:
                        act = RailEnvActions.STOP_MOVING

                    action_dict.update({a: act})
                else:
                    act = RailEnvActions.MOVE_FORWARD
                    action_dict.update({a: act})
            else:
                action_dict.update({a: RailEnvActions.DO_NOTHING})
        return action_dict

    def find_all_cell_where_agent_can_choose(self):
        switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions(*pos, dir)
                    num_transitions = np.count_nonzero(possible_transitions)
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

    def check_agent_descision(self, position, direction, switches, switches_neighbours):
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
                    self.env.agents[a].direction,
                    self.switches,
                    self.switches_neighbours)
            agents_on_switch.update({a: ret_agents_on_switch})
            ready_to_depart = self.env.agents[a].status == RailAgentStatus.READY_TO_DEPART
            agents_near_to_switch.update({a: (ret_agents_near_to_switch or ready_to_depart)})

            agents_can_choose.update({a: agents_on_switch[a] or agents_near_to_switch[a]})

            agents_near_to_switch_all.update({a: (ret_agents_near_to_switch_all or ready_to_depart)})

        return agents_can_choose, agents_on_switch, agents_near_to_switch, agents_near_to_switch_all

    def check_deadlock(self, only_next_cell_check=False, handle=None):
        agents_with_deadlock = []
        agents = range(self.env.get_num_agents())
        if handle is not None:
            agents = [handle]
        for a in agents:
            if self.env.agents[a].status < RailAgentStatus.DONE:
                position = self.env.agents[a].position
                first_step = True
                if position is None:
                    position = self.env.agents[a].initial_position
                    first_step = True
                direction = self.env.agents[a].direction
                while position is not None:  # and position != self.env.agents[a].target:
                    possible_transitions = self.env.rail.get_transitions(*position, direction)
                    # num_transitions = np.count_nonzero(possible_transitions)
                    agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = self.check_agent_descision(
                        position,
                        direction,
                        self.switches,
                        self.switches_neighbours)

                    if not agents_on_switch or first_step:
                        first_step = False
                        new_direction_me = np.argmax(possible_transitions)
                        new_cell_me = get_new_position(position, new_direction_me)
                        opp_agent = self.env.agent_positions[new_cell_me]
                        if opp_agent != -1:
                            opp_position = self.env.agents[opp_agent].position
                            opp_direction = self.env.agents[opp_agent].direction
                            opp_agents_on_switch, opp_agents_near_to_switch, agents_near_to_switch_all = \
                                self.check_agent_descision(opp_position,
                                                           opp_direction,
                                                           self.switches,
                                                           self.switches_neighbours)

                            # opp_possible_transitions = self.env.rail.get_transitions(*opp_position, opp_direction)
                            # opp_num_transitions = np.count_nonzero(opp_possible_transitions)
                            if not opp_agents_on_switch:
                                if opp_direction != direction:
                                    agents_with_deadlock.append(a)
                                    position = None
                                else:
                                    if only_next_cell_check:
                                        position = None
                                    else:
                                        position = new_cell_me
                                        direction = new_direction_me
                            else:
                                if only_next_cell_check:
                                    position = None
                                else:
                                    position = new_cell_me
                                    direction = new_direction_me
                        else:
                            if only_next_cell_check:
                                position = None
                            else:
                                position = new_cell_me
                                direction = new_direction_me
                    else:
                        position = None

        return agents_with_deadlock

    def generate_state(self, handle: int, root, max_depth: int):
        n_obs = normalize_observation(root[handle], max_depth)

        position = self.env.agents[handle].position
        direction = self.env.agents[handle].direction
        cell_free_4_first_step = -1
        deadlock_agents = []
        if self.env.agents[handle].status == RailAgentStatus.READY_TO_DEPART:
            if self.env.agent_positions[self.env.agents[handle].initial_position] == -1:
                cell_free_4_first_step = 1
            position = self.env.agents[handle].initial_position
        else:
            deadlock_agents = self.check_deadlock(only_next_cell_check=False, handle=handle)
        agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = self.check_agent_descision(position,
                                                                                                        direction,
                                                                                                        self.switches,
                                                                                                        self.switches_neighbours)

        append_obs = [self.env.agents[handle].status - RailAgentStatus.ACTIVE,
                      cell_free_4_first_step,
                      2.0 * int(len(deadlock_agents)) - 1.0,
                      2.0 * int(agents_on_switch) - 1.0,
                      2.0 * int(agents_near_to_switch) - 1.0]
        n_obs = np.append(n_obs, append_obs)

        return n_obs
