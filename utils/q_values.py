import numpy as np

from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_shortest_paths
from four_room.constants import size, state_to_q
from rnd_exploration.dataset import State
from dqn.counter import MovingCountBasedUncertainty

def next_state(x, y, d, a, walls):
    orig_x, orig_y, orig_d = x, y, d
    if a == 0:
        d = (d - 1) % 4
    elif a == 1:
        d = (d + 1) % 4
    elif a == 2: # move forward
        if d == 0:
            x += 1
        elif d == 1:
            y += 1
        elif d == 2:
            x -= 1
        elif d == 3:
            y -= 1
    
    if (int(x), int(y)) in walls:
        return orig_x, orig_y, orig_d
    else:
        return x, y, d


def compute_mc(rewards: list, gamma: float = 0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)  
    return returns

def compute_q_value(obs, context, counter: MovingCountBasedUncertainty, gamma, action=None):
    state = obs_to_state(obs)

    paths = find_all_shortest_paths(state[:2], state[2], state[3:5], state[5:], size)
    path_index = np.random.randint(low=0, high=len(paths))
    path = paths[path_index]
    
    rewards = [counter[context, *path_state] for path_state in path]
    return compute_mc(rewards, gamma)[0]


def optimal_q_action(obs, context, walls, counter: MovingCountBasedUncertainty, gamma):
    state = obs_to_state(obs)
    actions = [0, 1, 2]
    q_values = []
    
    for action in actions:
        obs_prime = next_state(*state[0:3], action, walls)
        paths = find_all_shortest_paths(obs_prime[:2], obs_prime[2], state[3:5], state[5:], size)
        path_index = np.random.randint(low=0, high=len(paths))
        path = paths[path_index]
        rewards = [counter[context, *path_state] for path_state in path]
        q_values.append(compute_mc(rewards, gamma)[0])
        
    return np.array(q_values).argmax()