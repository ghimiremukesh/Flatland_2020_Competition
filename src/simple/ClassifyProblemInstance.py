from enum import IntEnum

import numpy as np


class ProblemInstanceClass(IntEnum):
    SHORTEST_PATH_ONLY = 0
    SHORTEST_PATH_ORDERING_PROBLEM = 1
    REQUIRE_ALTERNATIVE_PATH = 2


def check_is_only_shortest_path_problem(env, project_path_matrix):
    x = project_path_matrix.copy()
    x[x < 2] = 0
    return np.sum(x) == 0


def check_is_shortest_path_and_ordering_problem(env, project_path_matrix):
    x = project_path_matrix.copy()
    for a in range(env.get_num_agents()):
        # loop over all path and project start position and target into the project_path_matrix
        agent = env.agents[a]
        if x[agent.position[0]][agent.position[1]] > 1:
            return False
        if x[agent.target[0]][agent.target[1]] > 1:
            return False
    return True


def check_is_require_alternative_path(env, project_path_matrix):
    paths = env.dev_pred_dict
    for a in range(env.get_num_agents()):
        agent = env.agents[a]
        path = paths[a]
        for path_loop in range(len(path)):
            p = path[path_loop]
            if p[0] == agent.target[0] and p[1] == agent.target[1]:
                break
            if project_path_matrix[p[0]][p[1]] > 1:
                # potential overlapping path found
                for opp_a in range(env.get_num_agents()):
                    opp_agent = env.agents[opp_a]
                    opp_path = paths[opp_a]
                    if p[0] == opp_agent.position[0] and p[1] == opp_agent.position[1]:
                        opp_path_loop = 0
                        tmp_path_loop = path_loop
                        while True:
                            if tmp_path_loop > len(path) - 1:
                                break
                            opp_p = opp_path[opp_path_loop]
                            tmp_p = path[tmp_path_loop + 1]
                            if opp_p[0] == opp_agent.target[0] and opp_p[1] == opp_agent.target[1]:
                                return True
                            if not (opp_p[0] == tmp_p[0] and opp_p[1] == tmp_p[1]):
                                break
                            if tmp_p[0] == agent.target[0] and tmp_p[1] == agent.target[1]:
                                break
                            opp_path_loop += 1
                            tmp_path_loop += 1

    return False


def classify_problem_instance(env):
    # shortest path from ShortesPathPredictorForRailEnv
    paths = env.dev_pred_dict

    project_path_matrix = np.zeros(shape=(env.height, env.width))
    for a in range(env.get_num_agents()):
        # loop over all path and project start position and target into the project_path_matrix
        agent = env.agents[a]
        project_path_matrix[agent.position[0]][agent.position[1]] += 1.0
        project_path_matrix[agent.target[0]][agent.target[1]] += 1.0

        if not (agent.target[0] == agent.position[0] and agent.target[1] == agent.position[1]):
            # project the whole path into
            path = paths[a]
            for path_loop in range(len(path)):
                p = path[path_loop]
                if p[0] == agent.target[0] and p[1] == agent.target[1]:
                    break
                else:
                    project_path_matrix[p[0]][p[1]] += 1.0

    return \
        {
            # analyse : SHORTEST_PATH_ONLY -> if conflict_mat does not contain any number > 1
            "SHORTEST_PATH_ONLY": check_is_only_shortest_path_problem(env, project_path_matrix),
            # analyse : SHORTEST_PATH_ORDERING_PROBLEM -> if agent_start and agent_target position does not contain any number > 1
            "SHORTEST_PATH_ORDERING_PROBLEM": check_is_shortest_path_and_ordering_problem(env, project_path_matrix),
            # analyse : REQUIRE_ALTERNATIVE_PATH -> if agent_start and agent_target position does not contain any number > 1
            "REQUIRE_ALTERNATIVE_PATH": check_is_require_alternative_path(env, project_path_matrix)

        }
