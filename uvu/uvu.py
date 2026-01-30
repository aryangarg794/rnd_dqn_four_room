import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym

from copy import deepcopy

from rnd_exploration.dataset import ReplayBuffer
from four_room.arch import CNN
from utils.episode import LastEpisode

class L2Norm(nn.Module):
    
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) # (b, h) -> (b, 1)
        return x / norm

class UVUModule(nn.Module):
    
    def __init__(
        self,
        env: gym.Env,
        use_cnn: bool = True, 
        cnn_features: int = 512, 
        hidden_layers: list = [512, 512, 512],
        residual: bool = True, 
        init: str = 'orthogonal',
        act: nn.Module = nn.ReLU,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_actions = env.action_space.n 

        self.layers = nn.Sequential()
        
        if use_cnn:
            self.layers.extend([
                CNN(observation_space=env.observation_space, features_dim=cnn_features, residual=residual),
                act(),
            ])
        else:
            self.layers.extend([
                nn.Linear(np.prod(env.observation_space.shape), cnn_features)
            ])
            
        self.layers.extend([
            nn.Linear(cnn_features, hidden_layers[0]),
            L2Norm(),
        ])
        
        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([
                nn.Linear(layer1, layer2), 
                act()
            ])
            
        self.layers.extend([nn.Linear(hidden_layers[-1], self.num_actions), L2Norm()])

        self.apply(self.orthogonal_layer_init if init == 'orthogonal' else self._init)

    def _init(self, m):
      if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

    def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        if hasattr(layer, 'weight'):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
        
    
class UVU:
    
    def __init__(
        self,
        env: gym.Env, 
        val_env: gym.Env, 
        use_cnn: bool = True, 
        capacity: int = int(1e5),
        cnn_features: int = 512, 
        gamma: float = 0.99, 
        start_epsilon: float = 0.99,
        max_decay: float = 0.1,
        decay_steps: float = 10000,
        lr: float = 5e-4,
        tau: float = 0.005,
        hidden_layers: list = [512, 512, 512],
        hidden_layers_g: list = [128],
        device: str = 'cuda', 
        residual: bool = True, 
        grad_norm: float = 10.0,
        init: str = 'orthogonal', 
        *args, 
        **kwargs
    ):
        self.net = UVUModule(
            env=env, 
            use_cnn=use_cnn,
            hidden_layers=hidden_layers,
            cnn_features=cnn_features,
            residual=residual
        ).to(device)
        
        self.target_net = deepcopy(self.net).to(device)
        
        self.g = UVUModule(
            env=env, 
            use_cnn=use_cnn,
            hidden_layers=hidden_layers_g,
            cnn_features=cnn_features,
            residual=residual
        ).to(device)
        
        for param in self.g.parameters():
            param.requires_grad = False
        
        self.env = env
        self.val_env = val_env
        self.start_epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon
        
        self.buffer = ReplayBuffer(state_dim=env.observation_space.shape, 
                                   capacity=capacity, num_actions=env.action_space.n, device=device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.tau = tau
        self.gamma = gamma
        self.grad_norm = grad_norm
        self.loss = nn.HuberLoss()
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
    
    @torch.no_grad()
    def epistemic(self, state: torch.Tensor, action: torch.Tensor):
        u = self.net(state).gather(index=action, dim=-1) # (b, 1)
        g = self.g(state).gather(index=action, dim=-1)
        return (u - g).pow(2)
    
    @torch.no_grad()
    def reward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor, 
        next_act: torch.Tensor, 
        dones: torch.Tensor
    ):
        # g - y * g'
        g_cur = self.g(state).gather(index=action, dim=-1)
        g_next = self.g(next_state).gather(index=next_act, dim=-1)
        
        return g_cur - self.gamma * g_next * (1 - dones)
    
    def update_step(self, batch_size: int, last_ep: LastEpisode):
        batch_obs, batch_actions, _, batch_primes, batch_next_actions, batch_dones = self.buffer.sample(batch_size)
        last_obs, last_action, _, last_obs_primes, last_next_actions, last_dones = last_ep.get()            
        
        batch_obs = torch.cat([batch_obs, last_obs], dim=0)
        batch_actions = torch.cat([batch_actions, last_action], dim=0)
        batch_primes = torch.cat([batch_primes, last_obs_primes], dim=0)
        batch_next_actions = torch.cat([batch_next_actions, last_next_actions], dim=0)
        batch_dones = torch.cat([batch_dones, last_dones], dim=0)
        batch_rewards = self.reward(batch_obs, batch_actions, batch_primes, batch_next_actions, batch_dones)
        
        with torch.no_grad():
            batch_rewards = batch_rewards.detach()
            target_vals = self.target_net(batch_primes).gather(dim=1, index=batch_next_actions)
            targets = batch_rewards + self.gamma * target_vals * (1 - batch_dones)
            
        q_values = self.net(batch_obs).gather(dim=1, index=batch_actions)
        loss = self.loss(q_values, targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm)
        self.optimizer.step() 

    def __call__(self, state: torch.Tensor):
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