from flatland.envs.rail_env import RailEnvActions


def get_flatland_full_action_size():
    # The action space of flatland is 5 discrete actions
    return 5


def get_action_size():
    # The agents (DDDQN, PPO, ... ) have this actions space
    return 4


def map_actions(actions):
    # Map the
    if get_action_size() == get_flatland_full_action_size():
        return actions
    for key in actions:
        value = actions.get(key, 0)
        actions.update({key: map_action(value)})
    return actions


def map_action(action):
    if get_action_size() == get_flatland_full_action_size():
        return action

    if action == 0:
        return RailEnvActions.MOVE_LEFT
    if action == 1:
        return RailEnvActions.MOVE_FORWARD
    if action == 2:
        return RailEnvActions.MOVE_RIGHT
    if action == 3:
        return RailEnvActions.STOP_MOVING


def map_rail_env_action(action):
    if get_action_size() == get_flatland_full_action_size():
        return action

    if action == RailEnvActions.MOVE_LEFT:
        return 0
    elif action == RailEnvActions.MOVE_FORWARD:
        return 1
    elif action == RailEnvActions.MOVE_RIGHT:
        return 2
    elif action == RailEnvActions.STOP_MOVING:
        return 3
    # action == RailEnvActions.DO_NOTHING:
    return 3
