import gymnasium as gym
import numpy as np 
import torch 
import torch.nn as nn
import argparse
import random
import os
import imageio
import dill

from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque

from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.wrappers import gym_wrapper
from four_room.constants import train_config, val_config, test_config, size, state_to_q
from dqn_experiments.regression_exp_utils import run_experiment
from rnd_exploration.dataset import ReplayBuffer, State

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

@dataclass
class Args:
    env: gym.Env
    val_env: gym.Env
    seed: int = 0 
    dir: str = 'test'
    lr: float = 1e-5
    capacity: int = int(1e5)
    device: str = 'cuda'

def train_explorego(
    args: Args, 
    num_timesteps: int = int(2e5), 
    regression_freq: int = 50000,
    seed: int = 0,
    render: bool = False,
): 
    """
    """
    os.makedirs('dqn_results', exist_ok=True)
    imgs = deque(maxlen=2500)
    learning_curves = []
    scores = []
    uniqueness = []
    
    torch.backends.cudnn.deterministic = True
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    buffer = ReplayBuffer(
        args.env.observation_space.shape,
        args.env.action_space.n,
        capacity=args.capacity,
        device=args.device
    )
    
    env = deepcopy(args.env)
    items_added = 0
    
    obs, _ = env.reset()
    state = obs_to_state(obs)
    goal_pos = state[3:5]
    target_pos = state[3:5] # first phase is warmup
    
    max_k = len(env.get_wrapper_attr('valid_pos'))
    k = np.random.randint(low=0, high=max_k)
    aux_pos = env.get_wrapper_attr('valid_pos')[k]
    env.get_wrapper_attr('move_valid_pos')(k)
    
    ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
    ep_colors = np.empty_like(ep_highlight_mask, dtype=object)
    
    current_context = env.unwrapped.context
    past_pos = []
    visit_history = deque(maxlen=args.capacity+1)
    
    for step in (pbar := tqdm(range(1, num_timesteps+1))): 
        state = obs_to_state(obs)
    
        agent_pos = env.get_wrapper_attr('agent_pos')
        
        state_obj = State(state=obs)
        q = state_to_q[state_obj]
        action = q.argmax() if isinstance(q, np.ndarray) else np.array(q).argmax()
        
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.update(obs, action, reward, obs_prime, 0, int(done), q_value=q)
        if render: 
            ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
            ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
            past_pos.append(agent_pos)
            visit_history.append((current_context, *agent_pos))
            
            if buffer.size >= buffer.capacity:
                to_remove = visit_history.popleft()
                ep_highlight_mask[to_remove[0], to_remove[1], to_remove[2]] = False
                ep_colors[to_remove[0], to_remove[1], to_remove[2]] = None
                    
            # rnd_net.observe(obs)
            items_added += 1
            
        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
            agent_col = (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255) 
            
            imgs.append(env.unwrapped.render(highlight_mask=ep_highlight_mask[current_context], 
                                        colors=ep_colors[current_context], agent_col=agent_col))
            env.get_wrapper_attr('remove_aux')(aux_pos)
            
        obs = obs_prime
        
        if done:
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)
                
            past_pos = []
            
            obs, _ = env.reset()
            done = False
            state = obs_to_state(obs)
            goal_pos = state[3:5]
            
            max_k = len(env.get_wrapper_attr('valid_pos'))
            k = np.random.randint(low=0, high=max_k)
            aux_pos = env.get_wrapper_attr('valid_pos')[k]
        
            env.get_wrapper_attr('move_valid_pos')(k)
                
            current_context = env.unwrapped.context

            
        if step % regression_freq == 0 and buffer.size >= buffer.capacity:
            lc, test_score = run_experiment(buffer, device=args.device)
            learning_curves.append(lc)
            scores.append(test_score)
            
            results = {
                'lc_curves': learning_curves, 
                'reg_test_scores' : scores,
                'uniqueness': uniqueness, 
                'images': imgs, 
            } 
            
            with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_{step}.pl', 'wb') as file:
                dill.dump(results, file)
        
        uniqueness.append(buffer.ratio_unique_trans)
        # pbar.set_description(f"Training RND DQN | Uniqueness: {buffer.ratio_unique_trans:.4f} | Last Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Total Items added: {items_added} | Current Context: {current_context} | RND Val: {rnd_val:.4f} | Avg: {rms.avg:.4f} | STD: {rms.std:.4f}")
        pbar.set_description(f"Training RND DQN | Uniqueness: {buffer.ratio_unique_trans:.4f} | Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Items added: {items_added} | Context: {current_context}")
            
    return {
        'lc_curves': learning_curves, 
        'reg_test_scores' : scores,
        'uniqueness': uniqueness, 
        'images': imgs, 
    } 
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timesteps', type=int, default=int(1e6), help='timesteps')
    parser.add_argument('-f', '--dir', type=str, default='basic_rnd', help='save name')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-s', '--replaysize', type=int, default=int(1e5), help='size of replay buffer')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-fr', '--freq', type=int, default=int(1e5), help='freq of regression')
    
    args = parser.parse_args()
    
    env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos= train_config['agent positions'],
            goal_pos = train_config['goal positions'],
            doors_pos = train_config['topologies'],
            agent_dir = train_config['agent directions'],
            size=size, 
            render_mode='rgb_array',
            disable_env_checker=True
        ),
        original_obs=True
    )
    
    val_env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos= val_config['agent positions'],
            goal_pos = val_config['goal positions'],
            doors_pos = val_config['topologies'],
            agent_dir = val_config['agent directions'],
            size=size
        ),
        original_obs=True
    )
    
    test_env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos= test_config['agent positions'],
            goal_pos = test_config['goal positions'],
            doors_pos = test_config['topologies'],
            agent_dir = test_config['agent directions'],
            size=size
        ),
        original_obs=True
    )
    
    aux_args = Args(
       env=env, 
       val_env=val_env,
       seed=args.seed, 
       dir=args.dir,
       device=args.device,
       capacity=args.replaysize, 
    )
    
    
    results = train_explorego(
        args=aux_args,
        num_timesteps=args.timesteps,
        seed=args.seed,
        regression_freq=args.freq,
        render=args.render,
    )
    
    # with open(f'dqn_results/{args.dir}.pl', 'wb') as file:
    #     dill.dump(results, file)
    
    with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_{args.timesteps}.pl', 'wb') as file:
        dill.dump(results, file)
        
    if args.render:
        imgs = list(results['images'])
        imageio.mimsave(f'renders/rendered_{args.dir}_seed_{args.seed}.gif', [np.array(img) for i, img in enumerate(imgs[-500:]) if i%1 == 0], duration=150)