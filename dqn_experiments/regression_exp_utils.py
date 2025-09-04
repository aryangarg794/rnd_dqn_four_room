import gymnasium as gym
import numpy as np 
import torch 
import time

from regression.experiment import RegressionModel
from rnd_exploration.dataset import ReplayBuffer
from four_room.wrappers import gym_wrapper
from four_room.env import FourRoomsEnv
from four_room.constants import val_config, test_config, size

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

        

def run_experiment(
    buffer: ReplayBuffer,
    timesteps: int = int(2e5), 
    val_freq: int = int(1e4),
    batch_size: int = 64, 
    device: str = 'cpu',
    print_freq: int = 5000
): 
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
    val_scores = []
    model = RegressionModel(val_env, val_env, device=device).to(device=device)
    
    X_train = buffer.states
    y_train = buffer.q_values
    n_samples = buffer.capacity
    
    start_time = time.time()
    for step in range(timesteps+1):
        batch_idx = torch.randint(0, n_samples, (batch_size,), device=device)
        batch_x = X_train[batch_idx]
        batch_y = y_train[batch_idx]
        
        preds = model(batch_x)
        loss = model.loss(preds, batch_y)
        
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if step % val_freq == 0:
            val_reward = model.validation(env=val_env)
            val_scores.append(val_reward)
            
        # if step % print_freq == 0:
        #     total_time = (time.time() - start_time) / print_freq 
        #     time_left = ((timesteps - step) * total_time) / 3600
        #     start_time = time.time() 
            
        #     print(f'Running Regression Step: {step} / {timesteps} | Loss: {loss.item():.3f} | Val Rewards: {val_reward:.3f} | Hours left: {time_left:.1f}',
        #       end='\r')
        
    test_result = model.validation(test_env, val_steps=200)
    return val_scores, test_result