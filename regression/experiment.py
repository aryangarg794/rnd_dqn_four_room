import time
import torch.nn as nn
import gymnasium as gym
import torch
import dill
import matplotlib.pyplot as plt
import numpy as np
import random

from four_room.arch import CNN
from four_room.wrappers import gym_wrapper
from rnd_exploration.dataset import Dataset
from four_room.constants import train_config, test_config, val_config, size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RegressionModel(nn.Module):
    
    def __init__(
        self, 
        env: gym.Env, 
        val_env: gym.Env,
        feature_dim: int = 64, 
        hidden_layers: list = list([32, 32]), 
        activation: nn.Module = nn.ReLU,
        lr: float = 1e-3, 
        device: str = 'cpu',
        *args, 
        **kwargs
    ):
        super(RegressionModel, self).__init__(*args, **kwargs)
        self.env = env
        
        self.feature_extractor = CNN(self.env.observation_space, feature_dim)
        self.layers = nn.Sequential()
        
        self.layers.extend(
            [nn.Linear(feature_dim, hidden_layers[0]),
             activation()]
        )
         
        # add more layers after the feature extractor
        for i, layer in enumerate(hidden_layers[:-1]):              
            self.layers.extend(
                [nn.Linear(layer, hidden_layers[i+1]),
                activation()]
            )
            
        # final layer for predicting values
        self.layers.append(
            nn.Linear(hidden_layers[-1], self.env.action_space.n)
        )
        
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        
        self.val_env = val_env
        
    def forward(self, obs):
        x = self.feature_extractor(obs)
        x = self.layers(x)
        return x 
    
    def validation(
        self, 
        env: gym.Env, 
        val_steps: int = 40 # hardcoded for number of val contexts 
    ): 
        self.eval()
        
        rewards = []
        with torch.no_grad():
            for _ in range(val_steps):
                obs, _ = env.reset()
                done = False
                ep_reward = 0 
                while not done: 
                    action = self(torch.as_tensor(
                        obs, 
                        dtype=torch.float32, 
                        device=self.device
                    ).view(1, *obs.shape)).cpu().numpy().argmax()
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated       

                rewards.append(ep_reward)
                
        self.train()
        
        return np.mean(rewards)
    
    def run(
        self, 
        dataset: Dataset, 
        label: str, 
        timesteps: int = int(1e5), 
        batch_size: int = 1024, 
        val_freq: int = 5000,
        print_freq: int = 1000
    ): 
        X_train = torch.as_tensor(np.array(dataset.X), dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(np.array(dataset.Y), dtype=torch.float32, device=self.device)
        
        n_samples = len(dataset)
        
        val_rewards = []
        
        start_time = time.time()
        for step in range(timesteps+1):
            batch_idx = torch.randint(0, n_samples, (batch_size,), device=self.device)
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            preds = self(batch_x)
            loss = self.loss(preds, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % val_freq == 0:
                val_reward = self.validation(env=self.val_env)
                val_rewards.append(val_reward)
                
            if step % print_freq == 0:
                total_time = (time.time() - start_time) / print_freq 
                time_left = ((timesteps - step) * total_time) / 3600
                start_time = time.time() 
                
                print(f'For dataset: {label} || Step: {step} / {timesteps} | Loss: {loss.item():.3f} | Val Rewards: {val_reward:.3f} | Hours left: {time_left:.1f}',
                  end='\r')
            
        return val_rewards
    
    
class Experiment:
    
    """Runs an experiment and then stores the results properly
    """
    
    def __init__(
        self, 
        exp_name: str,
        timesteps: int,
        val_freq: int = 40, 
        batch_size: int = 256, 
        seeds: list = list([0, 1, 2, 3, 4]), 
        save_dir = 'results', 
    ): 
        self.exp_name = exp_name
        self.seeds = seeds
        self.plot_dir = save_dir + '/plots/'
        self.results_dir = save_dir + '/pickles/'
        self.batch_size = batch_size
        
        self.timesteps = timesteps
        self.val_freq = val_freq
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos= train_config['agent positions'],
                goal_pos = train_config['goal positions'],
                doors_pos = train_config['topologies'],
                agent_dir = train_config['agent directions'],
                size=size, 
            ),
            original_obs=True
        )
        
        self.val_env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos= val_config['agent positions'],
                goal_pos = val_config['goal positions'],
                doors_pos = val_config['topologies'],
                agent_dir = val_config['agent directions'],
                size=size
            ),
            original_obs=True
        )
        
        self.test_env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos= test_config['agent positions'],
                goal_pos = test_config['goal positions'],
                doors_pos = test_config['topologies'],
                agent_dir = test_config['agent directions'],
                size=size
            ),
            original_obs=True
        )

    def run_experiment(
        self, 
        datasets: list, 
        labels: list 
    ):
        assert len(datasets) == len(labels)
        results = [[] for _ in range(len(datasets))]
        test_results = [[] for _ in range(len(datasets))]
        
        try: 
            torch.backends.cudnn.deterministic = True
            
            print(f'RUNNING EXPERIMENT {self.exp_name} ON DEVICE: {self.device}')
            for seed in self.seeds: 
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
                print(f'=============Seed {seed}===============\n')
                for i, dataset in enumerate(datasets):
                    model = RegressionModel(self.env, self.val_env, device=device).to(device=device)
                    val_rewards = model.run(dataset, timesteps=self.timesteps, 
                                            val_freq=self.val_freq, print_freq=self.val_freq, label=labels[i], 
                                            batch_size=self.batch_size)
                    results[i].append(val_rewards)
                    test_results[i].append(model.validation(self.test_env, val_steps=200))
                    
                print('\n')    
            self.plot_result(results, labels)
            self.save_results(results)
            
            for i, dataset in enumerate(datasets):
                mean = np.mean(test_results[i])
                print(f'Dataset: {labels[i]} | Test results: {mean:.4f}')
        except KeyboardInterrupt: 
            print('Experiment stopped prematurely')
            self.close()
    
    def save_results(
        self, 
        results: list
    ): 
        with open(f'{self.results_dir}/{self.exp_name}.pl', 'wb') as file: 
            dill.dump(results, file)
    
    def plot_result(
        self,
        results: list, 
        labels: list
    ): 
        def plot_with_ci(ax, data, label, q=5):
            data = np.array(data)
            mean = np.mean(data, axis=0)
            ci_upper = []
            ci_lower = []
            for i in range(data.shape[1]):
                ci_upper.append(np.percentile(data[:, i], q=100-q))
                ci_lower.append(np.percentile(data[:, i], q=q))

            ax.fill_between(np.arange(0, self.timesteps + self.val_freq, self.val_freq), 
                         ci_lower, ci_upper, alpha=0.2)
            ax.plot(np.arange(0, self.timesteps + self.val_freq, self.val_freq), mean, label=label)
        
        plt.style.use('ggplot')   
        fig, ax  = plt.subplots()
        for data, label in zip(results, labels):
            plot_with_ci(ax, data, label)
            
        ax.set_title(f"Train and Val Rewards for experiment {self.exp_name}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Rewards")
        ax.legend()
        ax.grid(True)

        fig.savefig(f'{self.plot_dir}/{self.exp_name}')
    
    def close(self): 
        self.env.close()
        self.val_env.close()