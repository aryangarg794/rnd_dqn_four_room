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
from utils.exploration import aux_pos_multiple

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
    
class LastEpisode:
    
    def __init__(self, state_dim, capacity=5, device='cuda'):
        self.capacity = capacity
        
        self.device = device
        self.pointer = 0
        self.size = 0
        
        self.states = torch.zeros((self.capacity, *state_dim) ,dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.capacity, 1) ,dtype=torch.int64, device=self.device)
        self.next_states = torch.zeros((self.capacity, *state_dim) ,dtype=torch.float, device=self.device)
        self.next_actions = torch.zeros((self.capacity, 1) ,dtype=torch.int64, device=self.device)
        self.dones = torch.zeros((self.capacity, 1) ,dtype=torch.int, device=self.device)
        self.tuples = deque(maxlen=capacity)
    
    def update(self, state, action, next_state, next_action, done, obj_tuple):
        self.states[self.pointer] = torch.as_tensor(state).to(self.device)
        self.actions[self.pointer] = action
        self.next_states[self.pointer] = torch.as_tensor(next_state).to(self.device)
        self.next_actions[self.pointer] = next_action
        self.dones[self.pointer] = done
        self.tuples.append(obj_tuple)
        
        self.pointer = (self.pointer + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)
    
    def get(self, counter: MovingCountBasedUncertainty):
        rewards = [counter[*obj_tuple] for obj_tuple in self.tuples]
        return (
            self.states[:self.size], 
            self.actions[:self.size], 
            torch.tensor(rewards, device=self.device, dtype=torch.float32).view(-1, 1),
            self.next_states[:self.size], 
            self.next_actions[:self.size],
            self.dones[:self.size]
        )


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
    gradient_steps: int = 3,
    render: bool = False,
    debug: bool = False,
    return_ones: bool = True,
    last_episode_len: int = 30,
    eps: float = 0.05,
    inlcude_expl_state: bool = False
): 
    rms_dqn = RunningAverage(window_size=window)
    rms_un = RunningAverage(window_size=window)
    rms_norms = RunningAverage(window_size=window)
    mse_loss = nn.HuberLoss()
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
        use_cnn=args.use_cnn,
        hidden_layers=[128, 512, 512, 128]
    )
    
    counter_moving = MovingCountBasedUncertainty(capacity=args.capacity, return_ones=return_ones, device=args.device)
    counter_full = CountBasedUncertainty(capacity=args.capacity)

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
    
    actions, path = aux_pos_multiple(state, env)
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
    start_state, _, _ = env.get_wrapper_attr('context_info')(current_context)
    past_pos = []
    visit_history = deque(maxlen=args.capacity+1)
    placeholder = np.array([1.0, 0.0, 0.0])
    
    switches = 0 
    trajs_added = 0
    contexts = []
    
    
    for step in (pbar := tqdm(range(1, num_timesteps+1), disable=debug)): 
        
        obs_torch = torch.from_numpy(obs).to(device=args.device).unsqueeze(dim=0)
        state = obs_to_state(obs)
        contexts.append(current_context)
        agent_pos = env.get_wrapper_attr('agent_pos')
        
        if len(actions) != 0 and not record and step >= warmupsteps:
            action = actions.pop(0)
        elif np.random.random() < eps: 
            action = np.random.randint(low=0, high=3)
        else:
            q = state_to_q[State(state=obs)]
            action = q.argmax() if isinstance(q, np.ndarray) else np.array(q).argmax()
        
        with torch.no_grad():
            goal_action = state_to_q[State(obs)].argmax()
            dqn_val = agent(obs_torch).squeeze()[goal_action].item()
            obj_tuple = tuple([int(item) for item in state])
            obj_tuple = (*obj_tuple, current_context)
            obj_moving_tuple = (current_context, *agent_pos, state[2])
            uncertainty = counter_moving[*obj_moving_tuple]
            q = state_to_q[State(state=obs)]
            
        norm = (dqn_val - rms_dqn.avg)/rms_dqn.std
        
        if (dqn_val - rms_dqn.avg >= alpha * rms_dqn.std or np.random.random() < eps) and not record and step >= warmupsteps: # swap to record mode 
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
            assert np.array_equal(target_pos, goal_pos) 
            # print(f'Timestep: {step} | Context: {current_context} | State: {agent_pos} | Dir: {state[2]} | Switch Count: {heatmap_swap[current_context, *agent_pos]} | Uncert: {uncertainty:.4f} | Count: {counter_moving.counts[*obj_moving_tuple]}')
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
            counter_full.add(obj_tuple)
            agent.buffer.update_seen(obj_moving_tuple)
            items_added += 1
        
        
        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr('set_aux')(aux_pos) if aux_pos else None
            agent_col = (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255) 
            
            imgs.append(env.unwrapped.render(highlight_mask=ep_highlight_mask[current_context], 
                                        colors=ep_colors[current_context], agent_col=agent_col))
            env.get_wrapper_attr('remove_aux')(aux_pos) if aux_pos else None
            
        obs = obs_prime
        
        for _ in range(gradient_steps): 
            batch_rewards, ind = counter_moving.sample(batch_size=batch_size)
            batch_obs, batch_actions, _, batch_primes, batch_next_actions, batch_dones = agent.buffer.sample_index(ind)
            
            with torch.no_grad():
                batch_rewards = batch_rewards.detach()
                target_vals = agent.target_net(batch_primes).gather(dim=1, index=batch_next_actions)
                targets = batch_rewards + gamma * target_vals * (1 - batch_dones)
                
            q_values = agent.net(batch_obs).gather(dim=1, index=batch_actions)
            loss = mse_loss(q_values, targets.detach())
            
            agent.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.net.parameters(), grad_norm)
            agent.optimizer.step()
        
        if done:
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)
                
            past_pos = []
            
            obs, _ = env.reset()
            done = False
            state = obs_to_state(obs)
            goal_pos = state[3:5]
            
            actions, path = aux_pos_multiple(state, env)
            aux_pos = (path[-1][0], path[-1][1])

            current_context = env.get_wrapper_attr('context')
            start_state, _, _ = env.get_wrapper_attr('context_info')(current_context)
            record = False
            trajs_added += 1
            
            if step < warmupsteps:
                target_pos = goal_pos # goal state
                env.get_wrapper_attr('move_valid_pos')(k)
            else: 
                target_pos = aux_pos

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
        
        uniqueness.append(agent.buffer.ratio_unique_trans)
        value = (dqn_val - rms_dqn.avg)/rms_dqn.std  
        pbar.set_description(f"Training RND DQN | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Last Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Total Items added: {items_added} | Current Context: {current_context} | RND Val: {dqn_val:.4f} | Avg: {rms_dqn.avg:.4f} | STD: {rms_dqn.std:.4f} | Switches: {switches} | Value: {value:.4f}")
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
        'running_mean_norms': rms_norms
    }, agent 
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timesteps', type=int, default=int(3e5), help='timesteps')
    parser.add_argument('-f', '--dir', type=str, default='dqn_count_test', help='save name')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('-ag', '--lr_agent', type=float, default=1e-4, help='lr for dqn agent')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-s', '--replaysize', type=int, default=int(1e5), help='size of replay buffer')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-fr', '--freq', type=int, default=int(1e6), help='freq of regression')
    parser.add_argument('--window', type=int, default=3500, help='window size of rms_dqn')
    parser.add_argument('-tau', '--tau', type=float, default=0.005, help='tau')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--return_ones', action='store_true', help='return ones')
    parser.add_argument('-e', '--eps', type=float, default=0.05, help='eps')
    parser.add_argument('--last_ep', type=int, default=10, help='window size of last_ep')
    
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
    
    
    results, agent = train_dqn_count(
        args=aux_args,
        batch_size=args.batch_size,
        num_timesteps=args.timesteps,
        seed=args.seed,
        alpha=args.alpha,
        regression_freq=args.freq,
        render=args.render,
        debug=args.debug,
        window=args.window,
        return_ones=args.return_ones,
        eps=args.eps,
        last_episode_len=args.last_ep
    )
    
    torch.save(agent.net.state_dict(), f'results/models/{args.dir}_seed_{args.seed}_{args.timesteps}.pt')
    
    with open(f'results/dqn_exps/{args.dir}_seed_{args.seed}_{args.timesteps}.pl', 'wb') as file:
        dill.dump(results, file)
    
    if args.render:
        imgs = list(results['images'])
        imageio.mimsave(f'renders/rendered_{args.dir}_seed_{args.seed}.gif', [np.array(img) for i, img in enumerate(imgs[-500:]) if i%1 == 0], duration=150)