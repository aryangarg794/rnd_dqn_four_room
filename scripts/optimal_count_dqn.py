import gymnasium as gym
import numpy as np 
import torch 
import torch.nn as nn
import argparse
import random
import imageio
import os
import dill

from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque

from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.wrappers import gym_wrapper
from four_room.constants import state_to_q
from rnd_exploration.utils import RunningAverage
from four_room.constants import train_config, val_config, test_config, size
from rnd_exploration.dataset import State, Transition, ReplayBuffer
from dqn_experiments.regression_exp_utils import run_experiment
from dqn.model import DQN
from dqn.counter import CountBasedUncertainty, MovingCountBasedUncertainty
from utils.q_values import compute_q_value, optimal_q_action
from utils.exploration import aux_pos_random_path, aux_pos_multiple

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

@dataclass
class Args:
    env: gym.Env
    val_env: gym.Env 
    dir: str = 'test'
    seed: int = 0
    lr_agent: float = 5e-4
    use_cnn: bool = True
    capacity: int = int(1e5)
    tau: float = 0.005
    use_actions: bool = False
    device: str = 'cuda'
    
def train_dqn_count(
    args: Args, 
    gamma: float = 0.95, 
    num_timesteps: int = int(2e5), 
    regression_freq: int = 50000,
    seed: int = 0,
    alpha: float = 1.5, 
    window: int = 3500, 
    warmupsteps: int = 0,
    render: bool = False,
    debug: bool = False,
    return_ones: bool = False,
    alt_explore: bool = False, 
    eps: float = 0.0, 
): 
    rms_dqn = RunningAverage(window_size=window)
    rms_un = RunningAverage(window_size=window)
    rms_norms = RunningAverage(window_size=window)
    os.makedirs('results/dqn_exps', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
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
    
    counter_moving = MovingCountBasedUncertainty(capacity=args.capacity, return_ones=return_ones)
    counter_full = CountBasedUncertainty(capacity=args.capacity)
    
    buffer = ReplayBuffer(
        args.env.observation_space.shape,
        args.env.action_space.n,
        capacity=args.capacity,
        device=args.device
    )

    env = deepcopy(args.env)
    items_added = 0
    
    obs, _ = env.reset()
    record = False
    state = obs_to_state(obs)
    goal_pos = state[3:5]
    target_pos = state[3:5] # first phase is warmup
    aux_pos = None
    
    max_k = len(env.get_wrapper_attr('valid_pos'))
    k = np.random.randint(low=0, high=max_k)
    if warmupsteps > 0:
        env.get_wrapper_attr('move_valid_pos')(k)
    
    actions, path = aux_pos_random_path(state, env)
    aux_pos = (path[-1][0], path[-1][1])
    
    ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
    heatmap_swap = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=int)
    
    aux_heatmap = np.zeros_like(heatmap_swap)
    explore_heatmap = np.zeros_like(heatmap_swap)
    switch_state_history = []
    
    ep_colors = np.empty_like(ep_highlight_mask, dtype=object)
    
    current_context = env.get_wrapper_attr('context')
    past_pos = []
    visit_history = deque(maxlen=args.capacity+1)
    placeholder = np.array([1.0, 0.0, 0.0])
    
    switches = 0 
    trajs_added = 0
    contexts = []
    failed_to_add = False
    
    for step in (pbar := tqdm(range(1, num_timesteps+1), disable=debug)): 
    
        state = obs_to_state(obs)
        contexts.append(current_context)
        agent_pos = env.get_wrapper_attr('agent_pos')
        
        if len(actions) != 0 and not alt_explore and not record:
            action = actions.pop(0)
        elif np.random.random() < eps: 
            action = np.random.randint(low=0, high=3)
        else:
            q = state_to_q[State(state=obs)]
            action = q.argmax() if isinstance(q, np.ndarray) else np.array(q).argmax()
            failed_to_add = True
        
        with torch.no_grad():
            goal_action = state_to_q[State(obs)].argmax()
            dqn_val = compute_q_value(obs, current_context, counter_moving, gamma)
            obj_tuple = tuple([int(item) for item in state])
            obj_tuple = (*obj_tuple, current_context)
            obj_moving_tuple = (current_context, *agent_pos, state[2])
            uncertainty = counter_moving[*obj_moving_tuple]
            
        norm = (dqn_val - rms_dqn.avg)/rms_dqn.std
        
        if (dqn_val - rms_dqn.avg >= alpha * rms_dqn.std and not record) or np.random.random() < eps: # swap to record mode 
            switches += 1 
            heatmap_swap[current_context, agent_pos[0], agent_pos[1]] += 1
            record = True
            target_pos = goal_pos
            switch_state_history.append((step, current_context, *agent_pos))
            action = goal_action
        
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        
        if not record:
            explore_heatmap[current_context, agent_pos[0], agent_pos[1]] += 1
        
        # if counter_moving.counts[*obj_moving_tuple] > 0:
        rms_un.update(uncertainty)
        rms_dqn.update(dqn_val)
        rms_norms.update(norm)
        
        if step < warmupsteps or record:
            # print(f'Timestep: {step} | Context: {current_context} | State: {agent_pos} | Dir: {state[2]} | Switch Count: {heatmap_swap[current_context, *agent_pos]} | Uncert: {uncertainty:.4f} | Count: {counter_moving.counts[*obj_moving_tuple]}')
            q = state_to_q[State(state=obs)]
            q_next = state_to_q[State(state=obs_prime)] if not done else placeholder
            next_action = q_next.argmax()
            buffer.update(obs, action, reward, obs_prime, next_action, int(done), q_value=q)
            if render: 
                ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                past_pos.append(agent_pos)
                visit_history.append((current_context, *agent_pos))
                
                if buffer.size >= buffer.capacity:
                    to_remove = visit_history[0]
                    ep_highlight_mask[to_remove[0], to_remove[1], to_remove[2]] = False
                    ep_colors[to_remove[0], to_remove[1], to_remove[2]] = None
                    
            counter_moving.add(obj_moving_tuple, step)
            counter_full.add(obj_tuple)
            buffer.update_seen(obj_moving_tuple)
            items_added += 1

        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr('set_aux')(aux_pos) if aux_pos else None
            agent_col = (255, 0, 0) if record else (0, 0, 255)
            agent_col = (255, 165, 0) if failed_to_add else agent_col
        
            imgs.append(env.unwrapped.render(highlight_mask=ep_highlight_mask[current_context], 
                                        colors=ep_colors[current_context], agent_col=agent_col))
            env.get_wrapper_attr('remove_aux')(aux_pos) if aux_pos else None
            
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

            current_context = env.get_wrapper_attr('context')
            record = False
            trajs_added += 1
            
            actions, path = aux_pos_random_path(state, env)
            aux_pos = (path[-1][0], path[-1][1])
            failed_to_add = False
            
            max_k = len(env.get_wrapper_attr('valid_pos'))
            k = np.random.randint(low=0, high=max_k)
            if step < warmupsteps:
                target_pos = goal_pos # goal state
                env.get_wrapper_attr('move_valid_pos')(k)
            else: 
                target_pos = aux_pos

        
        if step % regression_freq == 0 and buffer.size >= buffer.capacity:
            lc, test_score = run_experiment(buffer, device=args.device)
            learning_curves.append(lc)
            scores.append(test_score)
            
            results = {
                'lc_curves': learning_curves, 
                'reg_test_scores' : scores,
                'uniqueness': uniqueness, 
                'images': imgs, 
                'heatmap': heatmap_swap,
                'counter_full': counter_full, 
                'counter_moving': counter_moving, 
                'aux_heatmap': aux_heatmap, 
                'explore_heatmap': explore_heatmap,
                'switch_states': switch_state_history,
                'context_history': contexts
            } 
            with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_intermediate.pl', 'wb') as file:
                dill.dump(results, file)
        
        uniqueness.append(buffer.ratio_unique_trans)
        value = (dqn_val - rms_dqn.avg)/rms_dqn.std  
        pbar.set_description(f"Training RND DQN | ")
        # pbar.set_description(f"Training RND Count | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Items added: {items_added} | Context: {current_context}")
    
    return {
        'lc_curves': learning_curves, 
        'reg_test_scores' : scores,
        'uniqueness': uniqueness, 
        'images': imgs, 
        'heatmap': heatmap_swap,
        'counter_full': counter_full, 
        'counter_moving': counter_moving, 
        'aux_heatmap': aux_heatmap, 
        'explore_heatmap': explore_heatmap,
        'switch_states': switch_state_history,
        'context_history': contexts,
        'running_mean': rms_dqn,
        'running_mean_un': rms_un,
        'running_mean_norms': rms_norms,
        'buffer': buffer
    }
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timesteps', type=int, default=int(3e5), help='timesteps')
    parser.add_argument('-f', '--dir', type=str, default='dqn_count_test', help='save name')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-s', '--replaysize', type=int, default=int(1e5), help='size of replay buffer')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-fr', '--freq', type=int, default=int(1e6), help='freq of regression')
    parser.add_argument('--window', type=int, default=3500, help='window size of rms_dqn')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--return_ones', action='store_true', help='return ones')
    parser.add_argument('--alt', action='store_true', help='alt_explore')
    
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
       dir=args.dir,
       seed=args.seed,
       val_env=val_env, 
       device=args.device,
       capacity=args.replaysize, 
    )
    
    
    results = train_dqn_count(
        args=aux_args,
        num_timesteps=args.timesteps,
        seed=args.seed,
        alpha=args.alpha,
        regression_freq=args.freq,
        render=args.render,
        debug=args.debug,
        window=args.window,
        return_ones=args.return_ones,
        alt_explore=args.alt
    )
    
    with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_{args.timesteps}.pl', 'wb') as file:
        dill.dump(results, file)
    
    if args.render:
        imgs = list(results['images'])
        imageio.mimsave(f'renders/rendered_{args.dir}_seed_{args.seed}.gif', [np.array(img) for i, img in enumerate(imgs[-500:]) if i%1 == 0], duration=150)