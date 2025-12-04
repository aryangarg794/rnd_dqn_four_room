import dill 
import torch 
import torch.nn as nn
import numpy as np
from four_room.shortest_path import find_all_action_values
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
from minigrid.wrappers import RGBImgObsWrapper
import numpy as np
import imageio
import argparse
import gymnasium as gym
from four_room.utils import obs_to_state
from four_room.constants import train_config, test_config, val_config, size
from collections import deque
from rnd_exploration.dataset import ExploreGoDataset, Transition
from rnd_exploration.utils import compare, RunningAverage

import warnings
warnings.filterwarnings(action='once')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from rnd_exploration.rnd import RNDNetwork

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    
env = gym_wrapper(gym.make(
        'MiniGrid-FourRooms-v1', 
        agent_pos= train_config['agent positions'],
        goal_pos = train_config['goal positions'],
        doors_pos = train_config['topologies'],
        agent_dir = train_config['agent directions'],
        size=size, 
        max_steps=1200, 
    ),
    original_obs=True
)


def sweep_maze(net):
    # for i in range(2 * len(train_config['topologies'])):
    for i in [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 50, 60, 150]:
        
        env.get_wrapper_attr('set_context')(i)
        obs, _ = env.reset()
        done = False 
        valid_pos = env.get_wrapper_attr('valid_pos')
        last_obs = obs
        for idx in range(len(valid_pos)):
            env.get_wrapper_attr('move_valid_pos')(idx)
            
            for _ in range(4): 
                obs, _, _, _, _ = env.step(1)
                per = compare(last_obs, obs)
                last_obs = obs
                state = obs_to_state(obs)
                agent_pos = state[:2]
                q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
                action = np.array([1])
                rnd_value = net.get_error(obs, action).item()
                net.observe(obs, action)
                
                print(f'Context is {(i)%200:04d} | agent x, y: {agent_pos} | agent dir: {state[2]} | %Diff Prev State: {per:.4f} | RND Val: {rnd_value:.5f}', end='\r')

def optimal_trajectories(net, window=250):
    # for i in range(2 * len(train_config['topologies'])):
    dataset = ExploreGoDataset()
    aux_dataset = ExploreGoDataset()
    
    falses_avg = RunningAverage(window)
    trues_avg = RunningAverage(window)
    all_avgs = RunningAverage(window)
    falses_alpha = RunningAverage(window)
    trues_alpha = RunningAverage(window)
    
    iters = 0 
    # for i in [1, 2, 3, 4]:
        
    #     env.get_wrapper_attr('set_context')(i)
    #     obs, _ = env.reset()
    #     done = False 
        
    #     while not done: 
            
    #         state = obs_to_state(obs)
    #         agent_pos = state[:2]
    #         rnd_value = net.get_error(obs)
    #         net.observe(obs)
    #         q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
    #         dataset.add_trans(np.array(obs), np.array([1]))
            
    #         action = np.array(q).argmax()
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
            
    #         print(f'Context is {(i)%200:04d} | agent x, y: {agent_pos} | agent dir: {state[2]} | RND Val: {rnd_value:.5f}', end='\r')

    # for i in range(len([train_config['topologies']])):
        
    for _ in range(10000):
        # context = np.random.randint(low=0, high=200)
        # env.get_wrapper_attr('set_context')(context)
        obs, _ = env.reset()
        done = False 
        valid_pos = env.get_wrapper_attr('valid_pos')
        idx = np.random.randint(low=0, high=len(valid_pos))
        env.get_wrapper_attr('move_valid_pos')(idx)
        context = env.get_wrapper_attr('context')
        
        while not done: 
            
            state = obs_to_state(obs)
            agent_pos = state[:2]
            rnd_value = net.get_error(obs).item()
            net.observe(obs)
            q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
            action = np.array(q).argmax()
            trans = Transition(np.array(obs), np.array(q))
            in_data = trans in dataset.unique_trans
            dataset.add_trans(np.array(obs), np.array(q))
            dataset.add(np.array(obs), q, np.array(state))
            curr_alpha = (rnd_value - all_avgs.avg)/all_avgs.std
            if in_data:
                trues_avg.update(rnd_value)
                trues_alpha.update(curr_alpha)
            else:
                falses_avg.update(rnd_value)
                falses_alpha.update(curr_alpha)
                
            if curr_alpha >= falses_alpha.avg: 
                aux_dataset.add_trans(np.array(obs), np.array(q))
                aux_dataset.add(np.array(obs), q, np.array(state))
            all_avgs.update(rnd_value)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if iters % 2 == 0 and len(dataset) > 512:
                xs, _ = dataset.sample(512)
                net.observe(xs.to(device))
            
            
            iters += 1
            
            # print(f'Context is {(context)%200:04d} | state in data: {in_data} | RND Val: {rnd_value:.5f} | uniqueness: {dataset.ratio_unique_trans:.4f} | rnd_trues: {trues_avg.avg:.4f} | rnd_falses: {falses_avg.avg:.4f} | all_avgs: {all_avgs.avg:.4f} | curr_alpha: {curr_alpha:.4f} | trues_alpha: {trues_alpha.avg:.4f} | falses_alpha: {falses_alpha.avg:.4f}' , end='\r')
            print(f'Len Aux: {len(aux_dataset):08d} | Uniqueness: {aux_dataset.ratio_unique_trans:.4f}' , end='\r')
            
        
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lr', '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-w', '--win', type=int, default=250, help='window')
            
    args = parser.parse_args()
    net = RNDNetwork(env, lr=args.lr, device=device)
    # sweep_maze(net)
    optimal_trajectories(net, window=args.win)

    
                
                
    