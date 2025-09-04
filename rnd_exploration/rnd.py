import gymnasium as gym
import dill
import torch 
import torch.nn as nn
import numpy as np
from torch import Tensor

from four_room.arch import CNN, kaiming_layer_init, orthogonal_layer_init
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper


class BaseNetwork(nn.Module):
    
    def __init__(
        self,
        use_action: bool, 
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        outdim: int = 1024, 
        feature_units: int = 2048, 
        hidden_layers: list = list([2048, 2048, 1024, 1024, 1024]), 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.cnn = CNN(obs_space, features_dim=feature_units)
        
        self.net = nn.Sequential()
        if use_action:
            self.net.append(nn.Linear(feature_units + np.prod(action_space.shape, dtype=np.int64), hidden_layers[0]))
        else:
            self.net.append(nn.Linear(feature_units, hidden_layers[0]))
        
        for next_dim, prev_dim in zip(hidden_layers[1:], hidden_layers[:-1]):
            self.net.extend([
                nn.Linear(prev_dim, next_dim),
                nn.LeakyReLU()
            ])
        self.net.append(nn.Linear(hidden_layers[-1], outdim))
        self.use_action = use_action
        self.net.apply(orthogonal_layer_init)
        
    def forward(self, state: Tensor, action: Tensor = None) -> Tensor:
        features = self.cnn(state)
        if self.use_action: 
            inp = torch.cat([features, action], dim=-1)
        else:
            inp = features
        return self.net(inp)
        

class RNDNetwork:
    
    def __init__(
        self,
        env: gym.Env,
        use_actions: bool = False,
        scale: float = 1, 
        lr: float = 1e-5,
        device: str = 'cpu',
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.target_net = BaseNetwork(
            use_actions, 
            env.observation_space, 
            env.action_space,
            hidden_layers=[128],
        ).to(device)
        self.rnd_net = BaseNetwork(use_actions, env.observation_space, env.action_space).to(device)
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.rnd_net.parameters(), lr=lr)
        self.device = device
        self.scale = scale
        self.env = env
        self.loss = nn.MSELoss(reduction='none')
        self.use_actions = use_actions
        
    def observe(self, states: Tensor, actions: Tensor = None) -> None:
        states = self.sanitize(states)
        if self.use_actions: 
            actions = self.sanitize(actions)
            
        preds = self.rnd_net(states, actions)
        targets = self.target_net(states, actions)
        loss = self.loss(preds, targets.detach()).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def get_error(self, state: Tensor, action: Tensor = None) -> float:
        states = self.sanitize(state)
        if self.use_actions: 
            action = self.sanitize(action)
            
        with torch.no_grad():
            return self.scale * self.loss(self.rnd_net(states, action), 
                                          self.target_net(states, action)).sum(dim=-1)
        
    
    def sanitize(self, tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            tensor = torch.as_tensor(tensor, device=self.device) 
        if len(tensor.shape) < 4:  # wont work for non-image inputs !!
            tensor = tensor.unsqueeze(dim=0)
        return tensor
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    size = 19
    with open('configs/train.pl', 'rb') as file:
        train_config = dill.load(file)

    with open('configs/test_reachable.pl', 'rb') as file:
        test_config = dill.load(file)

    with open('configs/validation_unreachable.pl', 'rb') as file:
        val_config = dill.load(file)
        
    env = gym_wrapper(gym.make(
        'MiniGrid-FourRooms-v1', 
        agent_pos= train_config['agent positions'],
        goal_pos = train_config['goal positions'],
        doors_pos = train_config['topologies'],
        agent_dir = train_config['agent directions'],
        size=size, 
    ),
    original_obs=True
)
    model = BaseNetwork(env.observation_space)
    print(model)