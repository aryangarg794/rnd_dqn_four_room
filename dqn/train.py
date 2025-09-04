import gymnasium as gym
import numpy as np 
import torch 
import torch.nn as nn

from tqdm import tqdm

from rnd_exploration.utils import RunningAverage
from rnd_exploration.rnd import RNDNetwork
from dqn.model import DQN

def train_eps_greedy(
    agent: DQN,
    env: gym.Env, 
    batch_size: int = 256, 
    gamma: float = 0.99, 
    num_timesteps: int = int(2e5), 
    eval_iter: int = 5000, 
    grad_norm: float = 1.0,
    device: str = 'cuda'
):
    metrics = RunningAverage(window_size=25)
    val_rewards = []
    mse_loss = nn.MSELoss()
    
    obs, _ = env.reset()
    for step in (pbar := tqdm(range(num_timesteps))): 
        obs_torch = torch.as_tensor(obs, device=device).view(1, *obs.shape)
        action = agent.epsilon_greedy(obs_torch).cpu().item()
        
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.update(obs, action, reward, obs_prime, int(done))
        
        obs = obs_prime
        
        if done:
            obs, _ = env.reset()
            done = False
            
        batch_obs, batch_actions, batch_rewards, batch_primes, batch_dones = agent.buffer.sample(batch_size=batch_size)
        with torch.no_grad():
            target_vals = agent.target_net(batch_primes).max(dim=1, keepdim=True)[0]
            targets = batch_rewards + gamma * target_vals * (1 - batch_dones)
            
        q_values = agent.net(batch_obs).gather(dim=1, index=batch_actions)
        loss = mse_loss(q_values, targets)
        
        agent.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.net.parameters(), grad_norm)
        agent.optimizer.step()
        
        agent.epsilon_decay(step)
        agent.soft_update()
        
        pbar.set_description(f"Training Agent | Average Reward: {metrics.avg:.4f} | Eps: {agent.epsilon:.4f} ")
        
        if step % eval_iter == 0: 
            avg_reward = agent.eval()
            val_rewards.append(avg_reward)
            metrics.update(avg_reward)
            
    return metrics, val_rewards
            
