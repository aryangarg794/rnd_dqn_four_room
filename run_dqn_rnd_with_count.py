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
from four_room.shortest_path import find_all_action_values, find_all_shortest_paths, compute_actions
from four_room.wrappers import gym_wrapper
from four_room.constants import state_to_q
from rnd_exploration.utils import RunningAverage
from four_room.constants import train_config, val_config, test_config, size
from rnd_exploration.dataset import State, Transition
from dqn_experiments.regression_exp_utils import run_experiment
from dqn.model import DQN
from dqn.counter import CountBasedUncertainty, MovingCountBasedUncertainty
from rnd_exploration.rnd import RNDNetwork


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
    
@torch.no_grad()
def record_dqn_scores(agent: DQN, current_env: int, env_range: int = 5, device: str = 'cuda'):
    env_ids = list(range(current_env-env_range, current_env+env_range+1))
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
    
    agent.net.eval()
    results = np.zeros((4, len(env_ids), env.get_wrapper_attr('width'), env.get_wrapper_attr('height')),
                       dtype=np.float32)
    for idx, env_id in enumerate(env_ids):
        obs, _ = env.reset()
        env.get_wrapper_attr('set_context')(env_id)
        valid_pos = env.get_wrapper_attr('valid_pos')

        for i, valid_state in enumerate(valid_pos):
            env.get_wrapper_attr('move_valid_pos')(i)
            
            for _ in range(4): # for each direction we want to store the state-q value pair
                obs, _, _, _, _ = env.step(1)
                state = obs_to_state(obs)
                agent_dir = state[2]
                
                obs_torch = torch.from_numpy(obs).to(device=device).unsqueeze(dim=0)
                goal_action = state_to_q[State(obs)].argmax()
                dqn_val = agent(obs_torch).squeeze()[goal_action].item()
                results[agent_dir, idx, *valid_state] = dqn_val
            
    agent.net.train()
    return results.max(axis=0), env_ids

def train_dqn_count(
    args: Args, 
    batch_size: int = 512, 
    gamma: float = 0.99, 
    num_timesteps: int = int(2e5), 
    grad_norm: float = 1.0,
    regression_freq: int = 50000,
    seed: int = 0,
    alpha: float = 1.5, 
    window: int = 2500, 
    warmupsteps: int = 3500,
    update_freq: int = 1,
    eval_size: int = 10, # how many times you want to evaluate the network
    render: bool = False,
    debug: bool = False,
    rnd_steps: int = 2
): 
    rms_dqn = RunningAverage(window_size=window)
    rms_rnd = RunningAverage(window_size=window)
    mse_loss = nn.MSELoss()
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
    
    counter_moving = MovingCountBasedUncertainty(capacity=args.capacity, return_ones=True)
    counter_full = CountBasedUncertainty(capacity=args.capacity)

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
    if warmupsteps > 0:
        env.get_wrapper_attr('move_valid_pos')(k)
    
    paths = find_all_shortest_paths(state[:2], state[2], aux_pos, state[5:], size)
    path_index = np.random.randint(low=0, high=len(paths))
    actions = compute_actions(paths[path_index])
    
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
    intervals = list(range(num_timesteps-eval_size, num_timesteps+1))
    dqn_scores_list = []
    
    rnd_step = 0 
    rnd_seen_obs = torch.randn((rnd_steps, *args.env.observation_space.shape)).to(device=args.device)
    
    for step in (pbar := tqdm(range(1, num_timesteps+1), disable=debug)): 
        
        obs_torch = torch.from_numpy(obs).to(device=args.device).unsqueeze(dim=0)
        state = obs_to_state(obs)
        contexts.append(current_context)
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
            rnd_val = rnd_net.get_error(obs).item()
            obj_tuple = tuple([int(item) for item in state])
        
        obj_moving_tuple = (current_context, *agent_pos)
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        uncertainty = counter_moving[*obj_moving_tuple]
        
        agent_pos_after = env.get_wrapper_attr('agent_pos')
        if not record:
            explore_heatmap[current_context, agent_pos[0], agent_pos[1]] += 1
        
        rms_rnd.update(rnd_val)
        rms_dqn.update(dqn_val)
            
        
        if step < warmupsteps or record:
            assert np.array_equal(target_pos, goal_pos) 
            
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
                    
            counter_moving.add(obj_moving_tuple, step)
            counter_full.add(obj_tuple, current_context)
            agent.buffer.update_seen(obj_tuple)
            items_added += 1
            
            rnd_seen_obs[rnd_step] = obs_torch
            rnd_step = (rnd_step + 1) % rnd_steps
            
        elif dqn_val - rms_dqn.avg >= alpha * rms_dqn.std and not record: # swap to record mode 
        # elif np.array_equal(agent_pos_after, aux_pos):
            switches += 1 
            heatmap_swap[current_context, agent_pos[0], agent_pos[1]] += 1
            record = True
            target_pos = goal_pos
            switch_state_history.append((step, current_context, *agent_pos))
            
            # add the state that was used for the switch
            agent.buffer.update(obs, action, reward, obs_prime, 0, int(done), q_value=q)
            if render: 
                ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                past_pos.append(agent_pos)
                visit_history.append((current_context, *agent_pos))
                
                if agent.buffer.size >= agent.buffer.capacity:
                    to_remove = visit_history.popleft()
                    ep_highlight_mask[to_remove[0], to_remove[1], to_remove[2]] = False
                    ep_colors[to_remove[0], to_remove[1], to_remove[2]] = None
                    
            counter_moving.add(obj_moving_tuple, step)
            counter_full.add(obj_tuple, current_context)
            agent.buffer.update_seen(obj_tuple)
            items_added += 1
            
            rnd_seen_obs[rnd_step] = obs_torch
            rnd_step = (rnd_step + 1) % rnd_steps
                 
        elif (np.array_equal(agent_pos_after, aux_pos) or np.array_equal(agent_pos, aux_pos)) and not record: 
            target_pos = goal_pos
        
        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
            agent_col = (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255) 
            
            imgs.append(env.unwrapped.render(highlight_mask=ep_highlight_mask[current_context], 
                                        colors=ep_colors[current_context], agent_col=agent_col))
            env.get_wrapper_attr('remove_aux')(aux_pos)
            
        obs = obs_prime
        
        if step % update_freq == 0: 
            batch_obs, batch_actions, _, batch_primes, batch_next_actions, batch_dones = agent.buffer.sample(batch_size=batch_size)
            
            batch_obs = torch.cat([batch_obs, obs_torch], dim=0)
            action_torch = torch.tensor(action, dtype=torch.int64, device=args.device).view(1, -1)
            batch_actions = torch.cat([batch_actions, action_torch], dim=0)
            obs_prime_torch = torch.from_numpy(obs_prime).to(device=args.device).unsqueeze(dim=0)
            batch_primes = torch.cat([batch_primes, obs_prime_torch], dim=0)
            next_action_torch = torch.tensor(next_action, dtype=torch.int64, device=args.device).view(1, -1)
            batch_next_actions = torch.cat([batch_next_actions, next_action_torch], dim=0)
            done_torch = torch.tensor(done, dtype=torch.int, device=args.device).view(1, -1)
            batch_dones = torch.cat([batch_dones, done_torch], dim=0)
            
            
            with torch.no_grad():
                batch_rewards = rnd_net.get_error(batch_obs)
                batch_rewards = (batch_rewards - rms_rnd.avg) / rms_rnd.std
                batch_rewards = batch_rewards.detach().unsqueeze(dim=-1)
                target_vals = agent.target_net(batch_primes).gather(dim=1, index=batch_next_actions)
                targets = batch_rewards + gamma * target_vals * (1 - batch_dones)
                
            q_values = agent.net(batch_obs).gather(dim=1, index=batch_actions)
            loss = mse_loss(q_values, targets.detach())
            
            agent.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.net.parameters(), grad_norm)
            agent.optimizer.step()
            
        if step % rnd_steps == 0:
            rnd_batch_size = batch_size
            batch_rnd, _, _, _, _, _ = agent.buffer.sample(batch_size=rnd_batch_size)
            batch_rnd = torch.cat([batch_rnd, batch_obs, rnd_seen_obs], dim=0)
            rnd_net.observe(batch_rnd)
            rnd_seen_obs = torch.zeros((rnd_steps, *args.env.observation_space.shape)).to(device=args.device)
            
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

            current_context = env.get_wrapper_attr('context')
            record = False
            trajs_added += 1
            
            if step < warmupsteps:
                target_pos = goal_pos # goal state
                env.get_wrapper_attr('move_valid_pos')(k)
            else: 
                target_pos = aux_pos
                
            
            
        agent.soft_update()
        
        # if step in intervals:
        #     dqn_scores, env_ids = record_dqn_scores(agent, current_context)
        #     dqn_scores_list.append((dqn_scores, env_ids))
        
        if step % regression_freq == 0 and agent.buffer.size >= agent.buffer.capacity:
            lc, test_score = run_experiment(agent.buffer, device=args.device)
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
                'context_history': contexts,
                'dqn_scores': dqn_scores_list
            } 
            with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_intermediate.pl', 'wb') as file:
                dill.dump(results, file)
        
        uniqueness.append(agent.buffer.ratio_unique_trans)
        value = (dqn_val - rms_dqn.avg)/rms_dqn.std  
        # pbar.set_description(f"Training RND DQN | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Last Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Total Items added: {items_added} | Current Context: {current_context} | RND Val: {dqn_val:.4f} | Avg: {rms_dqn.avg:.4f} | STD: {rms_dqn.std:.4f} | Switches: {switches} | Value: {value:.4f}")
        pbar.set_description(f"Training RND Count | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Items added: {items_added} | Context: {current_context} | Last RND {value:.4f}")
        
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
        'dqn_scores': dqn_scores_list,
        'running_mean': rms_dqn, 
        'running_mean_rnd': rms_rnd, 
    }, agent, rnd_net 
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timesteps', type=int, default=int(1e5), help='timesteps')
    parser.add_argument('-f', '--dir', type=str, default='dqn_count_rnd_test', help='save name')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('-rnd', '--lr_rnd', type=float, default=1e-5, help='lr for rnd')
    parser.add_argument('-ag', '--lr_agent', type=float, default=1e-3, help='lr for dqn agent')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-s', '--replaysize', type=int, default=int(5e4), help='size of replay buffer')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-fr', '--freq', type=int, default=int(1e6), help='freq of regression')
    parser.add_argument('--window', type=int, default=2500, help='window size of rms_dqn')
    parser.add_argument('--rndsteps', type=int, default=5, help='when to update rnd')
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
    )
    
    
    results, agent, rnd_net = train_dqn_count(
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
    
    torch.save(agent.net.state_dict(), f'results/models/{args.dir}_seed_{args.seed}_{args.timesteps}.pt')
    rnd_net.save(f'results/models/rnd/{args.dir}_seed_{args.seed}_{args.timesteps}.pt')
    
    with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_{args.timesteps}.pl', 'wb') as file:
        dill.dump(results, file)
    
    if args.render:
        imgs = list(results['images'])
        imageio.mimsave(f'renders/rendered_{args.dir}_seed_{args.seed}.gif', [np.array(img) for i, img in enumerate(imgs[-500:]) if i%1 == 0], duration=150)