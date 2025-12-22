import gymnasium as gym
import numpy as np
import networkx as nx
import random

from four_room.shortest_path import find_all_shortest_paths, compute_actions, create_maze_graph
from four_room.constants import size

def sanitize_state(state):
    return (int(state[0]), int(state[1]), int(state[2]))
    
def sanitize_path(path: list):
    return [sanitize_state(state) for state in path]

def remove_goal(goal, path):
    path = sanitize_path(path)
    for i in range(4):
        goal_pos = (int(goal[0]), int(goal[1]), i)
        if goal_pos in path:
            path.remove(goal_pos)
    return path


def remove_blacklisted(blacklist, nodes):
    nodes = sanitize_path(nodes)
    blacklist = sanitize_path(blacklist)
    for node in blacklist:
        if node in nodes:
            nodes.remove(node)
    return nodes

def find_shortest_path(G, source, target):
        shortest_paths = []
        for p in nx.all_shortest_paths(G, source=source, target=target):
            shortest_paths.append(p)
        
        path_index = np.random.randint(low=0, high=len(shortest_paths))
        desc_actions = compute_actions(shortest_paths[path_index])
        return desc_actions, shortest_paths[path_index]

def explorego_exploration(state, env, goal_pos):
    max_k = len(env.get_wrapper_attr('valid_pos'))
    k = np.random.randint(low=0, high=max_k)
    aux_pos = env.get_wrapper_attr('valid_pos')[k]
    paths = find_all_shortest_paths(state[:2], state[2], aux_pos, state[5:], size)
    path_index = np.random.randint(low=0, high=len(paths))
    path = paths[path_index]
    path = remove_goal(goal_pos, path)
    path_k = np.random.randint(low=0, high=len(path))
    move_state = path[path_k]

    return move_state

def aux_pos_informed(state, env, counts=None):
    # NOTE: not working yet
    
    graph = create_maze_graph(state[5:], size)
    k = np.random.randint(low=0, high=50)
    
    full_path = []
    cur_pos = state[:3]
    for _ in range(k):
        next_state = None
        best_count = float('inf')
        descendants = nx.descendants_at_distance(graph, source=cur_pos, distance=1)
        descendants = remove_goal(state[3:5], descendants)
        descendants = remove_blacklisted(full_path, descendants)
        print(full_path)
        if counts:
            for desc in descendants:
                if counts[*desc] < best_count:
                    next_state = desc
                    best_count = counts[*desc]
                    
        else:
            next_state = random.choice(descendants)
        
        full_path.append(next_state)
        cur_pos = next_state
    
    actions = compute_actions(full_path)
            
    return actions, full_path

# def aux_pos_random_path(state, env):
#     graph = create_maze_graph(state[5:], size)
#     nodes = list(graph.nodes)
#     tel_pos = random.choice(nodes)
#     nodes.remove(tel_pos)
    

    
#     return actions, path

def aux_pos_multiple(state, env, num_jumps=4, distance=7):
    graph = create_maze_graph(state[5:], size)
    nodes = list(graph.nodes)
    nodes = remove_goal(state[3:5], nodes)
    k = np.random.randint(low=0, high=len(nodes))
    aux_pos = nodes[k]
    full_path = []
    
    actions, first_path = find_shortest_path(graph, state[:3], aux_pos)
    
    full_path.extend(first_path)

    aux_poses = [aux_pos]
    
    # compute other aux positions
    for jump in range(1, num_jumps+1):
        descendants = nx.descendants_at_distance(graph, source=aux_poses[jump-1], 
                                                 distance=distance)
        descendants = remove_goal(state[3:5], descendants)
        rand_desc = random.choice(list(descendants))
        aux_poses.append(rand_desc)
        desc_actions, path = find_shortest_path(graph, aux_poses[jump-1], rand_desc)
        actions.extend(desc_actions)
        full_path.extend(path)
        
    return actions, full_path
        

def explorego_multiple(state, env, num_jumps=4, distance=7):
    _, full_path = aux_pos_multiple(state, env, num_jumps, distance)
    k = np.random.randint(low=0, high=len(full_path))
    return full_path[k]