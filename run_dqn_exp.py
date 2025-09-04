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

from rnd_exploration.rnd import RNDNetwork
from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values, find_all_shortest_paths, compute_actions
from four_room.wrappers import gym_wrapper
from four_room.constants import state_to_q
from rnd_exploration.utils import RunningAverage
from four_room.constants import train_config, val_config, test_config, size
from rnd_exploration.dataset import State, Transition
from dqn_experiments.regression_exp_utils import run_experiment
from dqn.model import DQN
from line_profiler import profile

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
    lr_rnd: float = 1e-5
    use_actions: bool = False
    device: str = 'cuda'

def compute_mc(rewards: list, gamma: float = 0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)  
    return returns

@profile
def train_dqn_rnd(
    args: Args, 
    batch_size: int = 512, 
    gamma: float = 0.99, 
    num_timesteps: int = int(2e5), 
    grad_norm: float = 1.0,
    regression_freq: int = 50000,
    seed: int = 0,
    alpha: float = 1.5, 
    window: int = 10000, 
    update_freq: int = 1, 
    warmupsteps: int = 4000,
    render: bool = False,
    debug: bool = False,
    rnd_steps: int = 10
): 
    """
    """
    rms = RunningAverage(window_size=window)
    mse_loss = nn.MSELoss()
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
    
    agent = DQN(
        env=args.env,
        val_env=args.val_env,
        capacity=args.capacity,
        tau=args.tau,
        lr=args.lr_agent,
        device=args.device,
        use_cnn=args.use_cnn
    )
    
    rnd_net = RNDNetwork(
        env=args.env, 
        lr=args.lr_rnd,
        device=args.device
    )
    
    env = deepcopy(args.env)
    items_added = 0
    
    obs, _ = env.reset()
    record = False
    state = obs_to_state(obs)
    goal_pos = state[3:5]
    target_pos = state[3:5] # first phase is warmup
    
    max_k = len(env.get_wrapper_attr('valid_pos'))
    k = np.random.randint(low=0, high=max_k)
    aux_pos = env.get_wrapper_attr('valid_pos')[k]
    env.get_wrapper_attr('move_valid_pos')(k)
    
    paths = find_all_shortest_paths(state[:2], state[2], aux_pos, state[5:], size)
    path_index = np.random.randint(low=0, high=len(paths))
    actions = compute_actions(paths[path_index])
    
    ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
    ep_colors = np.empty_like(ep_highlight_mask, dtype=object)
    
    current_context = env.unwrapped.context
    past_pos = []
    visit_history = deque(maxlen=args.capacity+1)
    placeholder = np.array([1.0, 0.0, 0.0])
    
    dqn_vals = []
    rewards = []
    traj_in = []
    switches = 0 
    trajs_added = 0 
    rnd_seen_obs = torch.empty((rnd_steps, *args.env.observation_space.shape)).to(device=args.device)
    
    for step in (pbar := tqdm(range(1, num_timesteps+1), disable=debug)): 
        
        obs_torch = torch.from_numpy(obs).to(device=args.device).unsqueeze(dim=0)
        state = obs_to_state(obs)
    
        agent_pos = env.get_wrapper_attr('agent_pos')
        
        if np.array_equal(target_pos, aux_pos) and not np.array_equal(agent_pos, aux_pos):
            action = actions.pop(0)
        else:
            state_obj = State(state=obs)
            q = state_to_q[state_obj]
            action = q.argmax() if isinstance(q, np.ndarray) else np.array(q).argmax()
        
        with torch.no_grad():
            goal_action = state_to_q[State(obs)].argmax()
            dqn_val = agent(obs_torch).squeeze()[goal_action].item()
        
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent_pos_after = env.get_wrapper_attr('agent_pos')
        
        if step < warmupsteps or record:
            assert np.array_equal(target_pos, goal_pos) 
            
            if debug:
                reward_rnd = rnd_net.get_error(obs_torch)
                rewards.append(reward_rnd.item())
                dqn_vals.append(dqn_val)
                traj_in.append(
                    agent.buffer.unique_trans.has(Transition(obs, q))
                )
            
            q_next = state_to_q[State(state=obs_prime)] if not done else placeholder
            next_action = q_next.argmax()
            agent.buffer.update(obs, action, reward, obs_prime, next_action, int(done), q_value=q)
            if render: 
                ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                past_pos.append(agent_pos)
                visit_history.append((current_context, *agent_pos))
                
                if agent.buffer.size >= agent.buffer.capacity:
                    to_remove = visit_history[0]
                    ep_highlight_mask[to_remove[0], to_remove[1], to_remove[2]] = False
                    ep_colors[to_remove[0], to_remove[1], to_remove[2]] = None
            
            # rnd_net.observe(obs)
            rnd_seen_obs[step % rnd_steps] = obs_torch
            items_added += 1
        elif dqn_val - rms.avg >= alpha * rms.std or (np.array_equal(agent_pos_after, aux_pos) \
            or np.array_equal(agent_pos, aux_pos)) and not record: # swap to record mode 
            
            record = True
            target_pos = goal_pos
            if debug and dqn_val - rms.avg >= alpha * rms.std:
                switches += 1 
        
        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
            agent_col = (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255) 
            
            imgs.append(env.unwrapped.render(highlight_mask=ep_highlight_mask[current_context], 
                                        colors=ep_colors[current_context], agent_col=agent_col))
            env.get_wrapper_attr('remove_aux')(aux_pos)
            
        obs = obs_prime
        rms.update(dqn_val)
        
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
            
            paths = find_all_shortest_paths(state[:2], state[2], aux_pos, state[5:], size)
            path_index = np.random.randint(low=0, high=len(paths))
            actions = compute_actions(paths[path_index])
        
            if step < warmupsteps:
                target_pos = goal_pos # goal state
                env.get_wrapper_attr('move_valid_pos')(k)
            else: 
                target_pos = aux_pos
                
            record = False
            current_context = env.unwrapped.context
            trajs_added += 1
            
            if debug: 
                mc_vals = compute_mc(rewards, gamma=gamma)
                print(f"======Step:{step}========")
                print(f"Items added: {items_added}")
                print(f"Uniqueness: {agent.buffer.ratio_unique_trans:.4f}")
                print(f"Avg {rms.avg:.4f} | STD {rms.std:.4f}")
                print(f"DQN Vals: {dqn_vals}")
                print(f"MC Vals: {mc_vals}")
                print(f"Traj Uniqueness: {traj_in}")
                print(f"Switches: {switches}")
                print(f"Trajs Added: {trajs_added}")
            
            rewards = []
            dqn_vals = []
            traj_in = []
            
        if step % update_freq == 0: 
            batch_obs, batch_actions, _, batch_primes, batch_next_actions, batch_dones = agent.buffer.sample(batch_size=batch_size)
        
            with torch.no_grad():
                batch_rewards = rnd_net.get_error(batch_obs).detach().unsqueeze(dim=-1)
                target_vals = agent.target_net(batch_primes).gather(dim=1, index=batch_next_actions)
                targets = batch_rewards + gamma * target_vals * (1 - batch_dones)
                
            q_values = agent.net(batch_obs).gather(dim=1, index=batch_actions)
            loss = mse_loss(q_values, targets.detach())
            
            agent.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.net.parameters(), grad_norm)
            agent.optimizer.step()
            
        
        if step % rnd_steps == 0:
            rnd_batch_size = min(batch_size*(rnd_steps-1), 25)
            batch_rnd, _, _, _, _, _ = agent.buffer.sample(batch_size=rnd_batch_size)
            batch_rnd = torch.cat([batch_rnd, batch_obs, rnd_seen_obs], dim=0)
            rnd_net.observe(batch_rnd)
            rnd_seen_obs = torch.empty((rnd_steps, *args.env.observation_space.shape)).to(device=args.device)
            
        agent.soft_update()
        
        if step % regression_freq == 0 and agent.buffer.size >= agent.buffer.capacity:
            lc, test_score = run_experiment(agent.buffer, device=args.device)
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
        
        uniqueness.append(agent.buffer.ratio_unique_trans)
        # pbar.set_description(f"Training RND DQN | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Items added: {items_added} | Context: {current_context} | RND Val: {dqn_val:.4f} | Avg: {rms.avg:.4f} | STD: {rms.std:.4f}")
        pbar.set_description(f"Training RND DQN | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Items added: {items_added} | Context: {current_context}")
    
    return {
        'lc_curves': learning_curves, 
        'reg_test_scores' : scores,
        'uniqueness': uniqueness, 
        'images': imgs, 
    } 
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timesteps', type=int, default=int(1e6), help='timesteps')
    parser.add_argument('-f', '--dir', type=str, default='rnd_dqn', help='save name')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('-rnd', '--lr_rnd', type=float, default=1e-5, help='lr for rnd')
    parser.add_argument('-ag', '--lr_agent', type=float, default=1e-3, help='lr for dqn agent')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-s', '--replaysize', type=int, default=int(1e5), help='size of replay buffer')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('-fr', '--freq', type=int, default=int(1e5), help='freq of regression')
    parser.add_argument('--window', type=int, default=5000, help='window size of rms')
    parser.add_argument('--rndsteps', type=int, default=10, help='when to update rnd')
    parser.add_argument('-tau', '--tau', type=float, default=0.005, help='tau')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    
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
       lr_agent=args.lr_agent,
       device=args.device,
       capacity=args.replaysize, 
       tau=args.tau,
       lr_rnd=args.lr_rnd,
    )
    
    
    results = train_dqn_rnd(
        args=aux_args,
        batch_size=args.batch_size,
        num_timesteps=args.timesteps,
        seed=args.seed,
        alpha=args.alpha,
        regression_freq=args.freq,
        render=args.render,
        debug=args.debug,
        window=args.window,
        rnd_steps=args.rndsteps
    )
    
    with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_{args.timesteps}.pl', 'wb') as file:
        dill.dump(results, file)
    
    if args.render:
        imgs = list(results['images'])
        imageio.mimsave(f'renders/rendered_{args.dir}_seed_{args.seed}.gif', [np.array(img) for i, img in enumerate(imgs[-500:]) if i%1 == 0], duration=150)