
def get_flatland_full_action_size():
    # The action space of flatland is 5 discrete actions
    return 5


def get_action_size():
    # The agents (DDDQN, PPO, ... ) have this actions space
    return 4


def map_actions(actions, action_size):
    # Map the
    if action_size == get_flatland_full_action_size():
        return actions
    for key in actions:
        value = actions.get(key, 0)
        actions.update({key: (value + 1)})
    return actions


def map_action(action, action_size):
    if action_size == get_flatland_full_action_size():
        return action
    return action + 1
