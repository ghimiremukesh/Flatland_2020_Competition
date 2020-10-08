import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions


class AdrianShortestPathPredictorForRailEnv(PredictionBuilder):
    """
    ShortestPathPredictorForRailEnv object.

    This object returns shortest-path predictions for agents in the RailEnv environment.
    The prediction acts as if no other agent is in the environment and always takes the forward action.
    """

    def __init__(self, max_depth=20):
        # Initialize with depth 20
        self.max_depth = max_depth

    def get(self, custom_args=None, handle=None):
        """
        Called whenever get_many in the observation build is called.
        Requires distance_map to extract the shortest path.

        Parameters
        -------
        custom_args: dict
            - distance_map : dict
        handle : int (optional)
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here
            The prediction at 0 is the current position, direction etc.
        """

        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]
        assert custom_args is not None
        distance_map = custom_args.get('distance_map')
        assert distance_map is not None

        prediction_dict = {}
        for agent in agents:
            _agent_initial_position = agent.position
            _agent_initial_direction = agent.direction
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *_agent_initial_position, _agent_initial_direction, 0]
            visited = []
            for index in range(1, self.max_depth + 1):
                # if we're at the target, stop moving...
                if agent.position == agent.target:
                    prediction[index] = [index, *agent.target, agent.direction, RailEnvActions.STOP_MOVING]
                    visited.append((agent.position[0], agent.position[1], agent.direction))
                    continue
                # Take shortest possible path
                cell_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)

                new_position = None
                new_direction = None
                if np.sum(cell_transitions) == 1:
                    new_direction = np.argmax(cell_transitions)
                    new_position = get_new_position(agent.position, new_direction)
                elif np.sum(cell_transitions) > 1:
                    min_dist = np.inf
                    no_dist_found = True
                    for direction in range(4):
                        if cell_transitions[direction] == 1:
                            neighbour_cell = get_new_position(agent.position, direction)
                            target_dist = distance_map[agent.handle, neighbour_cell[0], neighbour_cell[1], direction]
                            if target_dist < min_dist or no_dist_found:
                                min_dist = target_dist
                                new_direction = direction
                                no_dist_found = False
                    new_position = get_new_position(agent.position, new_direction)
                else:
                    print("--------------------")
                    print(agent.position, agent.direction, "valid:", self.env.rail.cell_neighbours_valid(
                          agent.position),
                          self.env.rail.get_full_transitions(agent.position[0],agent.position[1])
                          )
                    print("--------------------")
                    raise Exception("No transition possible {}".format(cell_transitions))

                # update the agent's position and direction
                agent.position = new_position
                agent.direction = new_direction

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.append((new_position[0], new_position[1], new_direction))
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

            # cleanup: reset initial position
            agent.position = _agent_initial_position
            agent.direction = _agent_initial_direction

        return prediction_dict
