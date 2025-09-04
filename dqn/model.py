import torch 
import torch.nn as nn
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from copy import deepcopy

from rnd_exploration.dataset import ReplayBuffer
from rnd_exploration.utils import RunningAverage
from four_room.arch import CNN

class DQNModule(nn.Module):

    def __init__(
        self,
        env: gym.Env,
        use_cnn: bool = True, 
        cnn_features: int = 512, 
        hidden_layers: list = [256, 256],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_actions = env.action_space.n 

        self.layers = nn.Sequential()
        
        if use_cnn:
            self.layers.extend([
                CNN(observation_space=env.observation_space, features_dim=cnn_features),
                nn.ReLU(),
            ])
        else:
            self.layers.extend([
                nn.Linear(np.prod(env.observation_space.shape), cnn_features)
            ])
            
        self.layers.extend([
            nn.Linear(cnn_features, hidden_layers[0]),
            nn.ReLU()
        ])
        
        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([
                nn.Linear(layer1, layer2), 
                nn.ReLU()
            ])
            
        self.layers.append(nn.Linear(hidden_layers[-1], self.num_actions))

        self.apply(self._init)

    def _init(self, m):
      if isinstance(m, (nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

    
        
class DQN:
    
    def __init__(
        self,
        env: gym.Env, 
        val_env: gym.Env, 
        use_cnn: bool = True, 
        capacity: int = int(1e5),
        start_epsilon: float = 0.99,
        max_decay: float = 0.1,
        decay_steps: float = 10000,
        lr: float = 5e-4,
        tau: float = 0.005,
        device: str = 'cuda', 
        *args, 
        **kwargs
    ):
        self.net = DQNModule(
            env=env, 
            use_cnn=use_cnn
        ).to(device)
        
        self.target_net = deepcopy(self.net).to(device)
        
        self.env = env
        self.val_env = val_env
        self.start_epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon
        
        self.buffer = ReplayBuffer(state_dim=env.observation_space.shape, 
                                   capacity=capacity, num_actions=env.action_space.n)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.tau = tau
        self.device = device
        
    def soft_update(self):
        with torch.no_grad():
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            
    def eval(self, num_runs: int = 10, seed: int = 0):
        self.net.eval()
        rewards = []
        for _ in range(num_runs):
            obs, _ = self.val_env.reset(seed=seed)
            done = False
            ep_reward = 0 
            
            while not done:
                with torch.no_grad():
                    obs_torch = torch.as_tensor(obs, dtype=torch.float).view(1, -1).to(self.device)
                    action = self.net(obs_torch).view(-1).cpu().numpy().argmax()
                    
                    obs_prime, reward, terminated, truncated, _ = self.val_env.step(action)
                    ep_reward += reward
                    
                    obs = obs_prime
                    done = terminated or truncated

            rewards.append(ep_reward)

        self.net.train()
        return np.mean(rewards)
    
    def __call__(self, state: torch.Tensor, *args, **kwds):
        return self.net(state)
    
    # only need this for testing
    def epsilon_greedy(self, state, dim=1):
        rng = np.random.random()

        if rng < self.epsilon:
            action = self.env.action_space.sample()
            action = torch.tensor(action)
        else:
            with torch.no_grad():
                q_values = self.net(state)
            
            action = torch.argmax(q_values, dim=dim)

        return action

    def epsilon_decay(self, step):
        self.epsilon = self.max_decay + (self.start_epsilon - self.max_decay) * max(0, (self.decay_steps - step) / self.decay_steps)