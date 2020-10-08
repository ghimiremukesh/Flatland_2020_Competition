import math
from typing import Dict, List, Optional, Tuple, Set
from typing import NamedTuple

import numpy as np
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from flatland.utils.ordered_set import OrderedSet

WalkingElement = NamedTuple('WalkingElement',
                            [('position', Tuple[int, int]), ('direction', int),
                             ('next_action_element', RailEnvActions)])


def get_valid_move_actions_(agent_direction: Grid4TransitionsEnum,
                            agent_position: Tuple[int, int],
                            rail: GridTransitionMap) -> Set[RailEnvNextAction]:
    """
    Get the valid move actions (forward, left, right) for an agent.

    Parameters
    ----------
    agent_direction : Grid4TransitionsEnum
    agent_position: Tuple[int,int]
    rail : GridTransitionMap


    Returns
    -------
    Set of `RailEnvNextAction` (tuples of (action,position,direction))
        Possible move actions (forward,left,right) and the next position/direction they lead to.
        It is not checked that the next cell is free.
    """
    valid_actions: Set[RailEnvNextAction] = OrderedSet()
    possible_transitions = rail.get_transitions(*agent_position, agent_direction)
    num_transitions = np.count_nonzero(possible_transitions)
    # Start from the current orientation, and see which transitions are available;
    # organize them as [left, forward, right], relative to the current orientation
    # If only one transition is possible, the forward branch is aligned with it.
    if rail.is_dead_end(agent_position):
        action = RailEnvActions.MOVE_FORWARD
        exit_direction = (agent_direction + 2) % 4
        if possible_transitions[exit_direction]:
            new_position = get_new_position(agent_position, exit_direction)
            valid_actions.add(RailEnvNextAction(action, new_position, exit_direction))
    elif num_transitions == 1:
        action = RailEnvActions.MOVE_FORWARD
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    else:
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                if new_direction == agent_direction:
                    action = RailEnvActions.MOVE_FORWARD
                elif new_direction == (agent_direction + 1) % 4:
                    action = RailEnvActions.MOVE_RIGHT
                elif new_direction == (agent_direction - 1) % 4:
                    action = RailEnvActions.MOVE_LEFT
                else:
                    raise Exception("Illegal state")

                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    return valid_actions


# N.B. get_shortest_paths is not part of distance_map since it refers to RailEnvActions (would lead to circularity!)
def get_paths(distance_map: DistanceMap, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) \
        -> Dict[int, Optional[List[WalkingElement]]]:
    """
    Computes the shortest path for each agent to its target and the action to be taken to do so.
    The paths are derived from a `DistanceMap`.

    If there is no path (rail disconnected), the path is given as None.
    The agent state (moving or not) and its speed are not taken into account

    example:
            agent_fixed_travel_paths = get_shortest_paths(env.distance_map, None, agent.handle)
            path = agent_fixed_travel_paths[agent.handle]

    Parameters
    ----------
    distance_map : reference to the distance_map
    max_depth : max path length, if the shortest path is longer, it will be cutted
    agent_handle : if set, the shortest for agent.handle will be returned , otherwise for all agents

    Returns
    -------
        Dict[int, Optional[List[WalkingElement]]]

    """
    shortest_paths = dict()

    def _shortest_path_for_agent(agent):
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            position = agent.target
        else:
            shortest_paths[agent.handle] = None
            return
        direction = agent.direction
        shortest_paths[agent.handle] = []
        distance = math.inf
        depth = 0
        cnt = 0
        while (position != agent.target and (max_depth is None or depth < max_depth)) and cnt < 1000:
            cnt = cnt + 1
            next_actions = get_valid_move_actions_(direction, position, distance_map.rail)
            best_next_action = None

            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    agent.handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            for next_action in next_actions:
                if next_action.action == RailEnvActions.MOVE_LEFT:
                    next_action_distance = distance_map.get()[
                        agent.handle, next_action.next_position[0], next_action.next_position[
                            1], next_action.next_direction]
                    if abs(next_action_distance - distance) < 5:
                        best_next_action = next_action
                        distance = next_action_distance

            shortest_paths[agent.handle].append(WalkingElement(position, direction, best_next_action))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_action is None:
                shortest_paths[agent.handle] = None
                return

            position = best_next_action.next_position
            direction = best_next_action.next_direction
        if max_depth is None or depth < max_depth:
            shortest_paths[agent.handle].append(
                WalkingElement(position, direction,
                               RailEnvNextAction(RailEnvActions.STOP_MOVING, position, direction)))

    if agent_handle is not None:
        _shortest_path_for_agent(distance_map.agents[agent_handle])
    else:
        for agent in distance_map.agents:
            _shortest_path_for_agent(agent)

    return shortest_paths


def agent_fake_position(agent):
    if agent.position is not None:
        return (agent.position[0], agent.position[1], 0)
    return (-agent.handle - 1, -1, None)


def compare_position_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return (a[0] == b[0] and a[1] == b[1])


def calc_conflict_matrix_next_step(env, paths, do_move, agent_position_matrix, agent_target_matrix,
                                   agent_next_position_matrix):
    # look step forward
    conflict_mat = np.zeros(shape=(env.get_num_agents(), env.get_num_agents())) - 1

    # calculate weighted (priority)
    priority = np.arange(env.get_num_agents()).astype(float)
    unique_ordered_priority = np.argsort(priority).astype(int)

    # build one-step away dead-lock matrix
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        path = paths[a]
        if path is None:
            continue

        conflict_mat[a][a] = unique_ordered_priority[a]
        for path_loop in range(len(path)):
            p_el = path[path_loop]
            p = p_el.position
            if compare_position_equal(agent.target, p):
                break
            else:
                a_loop = 0
                opp_a = (int)(agent_next_position_matrix[p[0]][p[1]][a_loop])

                cnt = 0
                while (opp_a > -1) and (cnt < 1000):
                    cnt = cnt + 1
                    opp_path = paths[opp_a]
                    if opp_path is not None:
                        opp_a_p1 = opp_path[0].next_action_element.next_position
                        if path_loop < len(path) - 1:
                            p1 = path[path_loop + 1].next_action_element.next_position
                            if not compare_position_equal(opp_a_p1, p1):
                                conflict_mat[a][opp_a] = unique_ordered_priority[opp_a]
                                conflict_mat[opp_a][a] = unique_ordered_priority[a]
                        a_loop += 1
                        opp_a = (int)(agent_next_position_matrix[p[0]][p[1]][a_loop])

    # update one-step away
    for a in range(env.get_num_agents()):
        if not do_move[a]:
            conflict_mat[conflict_mat == unique_ordered_priority[a]] = -1

    return conflict_mat


def avoid_dead_lock(env, a, paths, conflict_matrix, agent_position_matrix, agent_target_matrix,
                    agent_next_position_matrix):
    # performance optimisation
    if conflict_matrix is not None:
        if np.argmax(conflict_matrix[a]) == a:
            return True

    # dead lock algorithm
    agent = env.agents[a]
    agent_position = agent_fake_position(agent)
    if compare_position_equal(agent_position, agent.target):
        return True

    path = paths[a]
    if path is None:
        return True

    max_path_step_allowed = np.inf
    # iterate over agent a's travel path (fixed path)
    for path_loop in range(len(path)):
        p_el = path[path_loop]
        p = p_el.position
        if compare_position_equal(p, agent.target):
            break

        # iterate over all agents (opposite)
        # for opp_a in range(env.get_num_agents()):
        a_loop = 0
        opp_a = 0
        cnt = 0
        while (a_loop < env.get_num_agents() and opp_a > -1) and cnt < 1000:
            cnt = cnt + 1
            if conflict_matrix is not None:
                opp_a = (int)(agent_next_position_matrix[p[0]][p[1]][a_loop])
                a_loop += 1
            else:
                opp_a = (int)(agent_position_matrix[p[0]][p[1]])
                a_loop = env.get_num_agents()
            if opp_a > -1:
                if opp_a != a:
                    opp_agent = env.agents[opp_a]
                    opp_path = paths[opp_a]
                    if opp_path is not None:
                        opp_path_0 = opp_path[0]

                        # find all position in the opp.-path which are equal to current position.
                        # the method has to scan all path through
                        all_path_idx_offset_array = [0]
                        for opp_path_loop_itr in range(len(path)):
                            opp_p_el = opp_path[opp_path_loop_itr]
                            opp_p = opp_p_el.position
                            if compare_position_equal(opp_p, opp_agent.target):
                                break
                            opp_agent_position = agent_fake_position(opp_agent)
                            if compare_position_equal(opp_p, opp_agent_position):
                                all_path_idx_offset_array.extend([opp_path_loop_itr])
                            opp_p_next = opp_p_el.next_action_element.next_position
                            if compare_position_equal(opp_p_next, opp_agent_position):
                                all_path_idx_offset_array.extend([opp_path_loop_itr])

                        for all_path_idx_offset_loop in range(len(all_path_idx_offset_array)):
                            all_path_idx_offset = all_path_idx_offset_array[all_path_idx_offset_loop]
                            opp_path_0_el = opp_path[all_path_idx_offset]
                            opp_path_0 = opp_path_0_el.position
                            # if check_in_details is set to -1: no dead-lock candidate found
                            # if check_in_details is set to  0: dead-lock candidate are not yet visible (agents need one step to become visible)(case A)
                            # if check_in_details is set to  1: dead-lock candidate are visible, thus we have to collect them (case B)
                            check_in_detail = -1

                            # check mode, if conflict_matrix is set, then we are looking ..
                            if conflict_matrix is not None:
                                # Case A
                                if np.argmax(conflict_matrix[a]) != a:
                                    # avoid (parallel issue)
                                    if compare_position_equal(opp_path_0, p):
                                        check_in_detail = 0
                            else:
                                # Case B
                                # collect all dead-lock candidates and check
                                opp_agent_position = agent_fake_position(opp_agent)
                                if compare_position_equal(opp_agent_position, p):
                                    check_in_detail = 1

                            if check_in_detail > -1:
                                # print("Conflict risk found. My [", a, "] path is occupied by [", opp_a, "]")
                                opp_path_loop = all_path_idx_offset
                                back_path_loop = path_loop - check_in_detail
                                cnt = 0
                                while (opp_path_loop < len(opp_path) and back_path_loop > -1) and cnt < 1000:
                                    cnt = cnt + 1
                                    # retrieve position information
                                    opp_p_el = opp_path[opp_path_loop]
                                    opp_p = opp_p_el.position
                                    me_p_el = path[back_path_loop]
                                    me_p = me_p_el.next_action_element.next_position

                                    if not compare_position_equal(opp_p, me_p):
                                        # Case 1: The opposite train travels in same direction as the current train (agent a)
                                        # Case 2: The opposite train travels in opposite direction and the path divergent
                                        break

                                    # make one step backwards (agent a) and one step forward for opposite train (agent opp_a)
                                    # train a can no travel further than given position, because no divergent paths, this will cause a dead-lock
                                    max_path_step_allowed = min(max_path_step_allowed, back_path_loop)
                                    opp_path_loop += 1
                                    back_path_loop -= 1

                                    # check whether at least one step is allowed
                                    if max_path_step_allowed < 1:
                                        return False

                                if back_path_loop == -1:
                                    # No divergent path found, it cause a deadlock
                                    # print("conflict (stop): (", a, ",", opp_a, ")")
                                    return False

    # check whether at least one step is allowed
    return max_path_step_allowed > 0


def calculate_one_step(env):
    # can agent move array
    do_move = np.zeros(env.get_num_agents())
    if True:
        cnt = 0
        cnt_done = 0
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if agent.status < RailAgentStatus.DONE:
                cnt += 1
                if cnt < 30:
                    do_move[a] = True
            else:
                cnt_done += 1
        print("\r{}/{}\t".format(cnt_done, env.get_num_agents()), end="")
    else:
        agent_fixed_travel_paths = get_paths(env.distance_map, 1)
        # can agent move array
        do_move = np.zeros(env.get_num_agents())
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if agent.position is not None and not compare_position_equal(agent.position, agent.target):
                do_move[a] = True
                break

        if np.sum(do_move) == 0:
            for a in range(env.get_num_agents()):
                agent = env.agents[a]
                if agent_fixed_travel_paths[a] is not None:
                    if agent.position is None and compare_position_equal(agent.initial_position, agent.target):
                        do_move[a] = True
                        break
                    elif not compare_position_equal(agent.initial_position, agent.target):
                        do_move[a] = True
                        break

        initial_position = None
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if do_move[a]:
                initial_position = agent.initial_position

            if initial_position is not None:
                if compare_position_equal(agent.initial_position, initial_position):
                    do_move[a] = True

    # copy of agents fixed travel path (current path to follow) : only once : quite expensive
    # agent_fixed_travel_paths = get_shortest_paths(env.distance_map)
    agent_fixed_travel_paths = dict()
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if do_move[a]:
            agent_fixed_travel_paths[agent.handle] = get_paths(env.distance_map, None, agent.handle)[agent.handle]
        else:
            agent_fixed_travel_paths[agent.handle] = None

    # copy position, target and next position into cache (matrices)
    # (The cache idea increases the run-time performance)
    agent_position_matrix = np.zeros(shape=(env.height, env.width)) - 1.0
    agent_target_matrix = np.zeros(shape=(env.height, env.width)) - 1.0
    agent_next_position_matrix = np.zeros(shape=(env.height, env.width, env.get_num_agents() + 1)) - 1.0
    for a in range(env.get_num_agents()):
        if do_move[a] == False:
            continue
        agent = env.agents[a]
        agent_position = agent_fake_position(agent)
        if agent_position[2] is None:
            agent_position = agent.initial_position
        agent_position_matrix[agent_position[0]][agent_position[1]] = a
        agent_target_matrix[agent.target[0]][agent.target[1]] = a
        if not compare_position_equal(agent.target, agent_position):
            path = agent_fixed_travel_paths[a]
            if path is not None:
                p_el = path[0]
                p = p_el.position
                a_loop = 0
                cnt = 0
                while (agent_next_position_matrix[p[0]][p[1]][a_loop] > -1) and cnt < 1000:
                    cnt = cnt + 1
                    a_loop += 1
                agent_next_position_matrix[p[0]][p[1]][a_loop] = a

    # check which agents can move (see : avoid_dead_lock (case b))
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if not compare_position_equal(agent.position, agent.target) and do_move[a]:
            do_move[a] = avoid_dead_lock(env, a, agent_fixed_travel_paths, None, agent_position_matrix,
                                         agent_target_matrix,
                                         agent_next_position_matrix)

    # check which agents can move (see : avoid_dead_lock (case a))
    # calculate possible candidate for hidden one-step away dead-lock candidates
    conflict_matrix = calc_conflict_matrix_next_step(env, agent_fixed_travel_paths, do_move, agent_position_matrix,
                                                     agent_target_matrix,
                                                     agent_next_position_matrix)
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if not compare_position_equal(agent.position, agent.target):
            if do_move[a]:
                do_move[a] = avoid_dead_lock(env, a, agent_fixed_travel_paths, conflict_matrix, agent_position_matrix,
                                             agent_target_matrix,
                                             agent_next_position_matrix)

    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if agent.position is not None and compare_position_equal(agent.position, agent.target):
            do_move[a] = False

    # main loop (calculate actions for all agents)
    action_dict = {}
    is_moving_cnt = 0
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        action = RailEnvActions.MOVE_FORWARD

        if do_move[a] and is_moving_cnt < 10:
            is_moving_cnt += 1
            # check for deadlock:
            path = agent_fixed_travel_paths[a]
            if path is not None:
                action = path[0].next_action_element.action
        else:
            action = RailEnvActions.STOP_MOVING
        action_dict[a] = action

    return action_dict, do_move


def calculate_one_step_heuristics(env):
    # copy of agents fixed travel path (current path to follow)
    agent_fixed_travel_paths = get_paths(env.distance_map, 1)

    # main loop (calculate actions for all agents)
    action_dict = {}
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        action = RailEnvActions.MOVE_FORWARD

        # check for deadlock:
        path = agent_fixed_travel_paths[a]
        if path is not None:
            action = path[0].next_action_element.action
        action_dict[a] = action

    return action_dict, None


def calculate_one_step_primitive_implementation(env):
    # can agent move array
    do_move = np.zeros(env.get_num_agents())
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if agent.status > RailAgentStatus.ACTIVE:
            continue
        if (agent.status == RailAgentStatus.ACTIVE):
            do_move[a] = True
            break
        do_move[a] = True
        break

    # main loop (calculate actions for all agents)
    action_dict = {}
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        action = RailEnvActions.MOVE_FORWARD
        if do_move[a]:
            # check for deadlock:
            # copy of agents fixed travel path (current path to follow)
            agent_fixed_travel_paths = get_shortest_paths(env.distance_map, 1, agent.handle)
            path = agent_fixed_travel_paths[agent.handle]
            if path is not None:
                print("\rAgent:{:4d}/{:<4d} ".format(a + 1, env.get_num_agents()), end=" ")
                action = path[0].next_action_element.action
        else:
            action = RailEnvActions.STOP_MOVING
        action_dict[a] = action

    return action_dict, do_move


def calculate_one_step_package_implementation(env):
    # copy of agents fixed travel path (current path to follow)
    # agent_fixed_travel_paths = get_shortest_paths(env.distance_map,1)
    agent_fixed_travel_paths = get_paths(env.distance_map, 1)

    # can agent move array
    do_move = np.zeros(env.get_num_agents())
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if agent.position is not None and not compare_position_equal(agent.position, agent.target):
            do_move[a] = True
            break

    if np.sum(do_move) == 0:
        for a in range(env.get_num_agents()):
            agent = env.agents[a]
            if agent_fixed_travel_paths[a] is not None:
                if agent.position is None and compare_position_equal(agent.initial_position, agent.target):
                    do_move[a] = True
                    break
                elif not compare_position_equal(agent.initial_position, agent.target):
                    do_move[a] = True
                    break

    initial_position = None
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        if do_move[a]:
            initial_position = agent.initial_position

        if initial_position is not None:
            if compare_position_equal(agent.initial_position, initial_position):
                do_move[a] = True

    # main loop (calculate actions for all agents)
    action_dict = {}
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        action = RailEnvActions.MOVE_FORWARD

        if do_move[a]:
            # check for deadlock:
            path = agent_fixed_travel_paths[a]
            if path is not None:
                action = path[0].next_action_element.action
        else:
            action = RailEnvActions.STOP_MOVING
        action_dict[a] = action

    return action_dict, do_move
