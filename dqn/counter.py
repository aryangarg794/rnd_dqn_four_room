import numpy as np 
import torch
import random

from collections import defaultdict, deque

class CountBasedUncertainty:
    
    def __init__(
        self,
        capacity: int,
        device: str = 'cuda'
    ):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.pointer = 0
        self.counts = defaultdict(int)
        self.device = device
        self.eps = 1e-6
        
    
    def __getitem__(self, key: tuple):
        return 1/self.counts[key] + self.eps
        
    def add(self, state_repr: tuple):
        self.counts[state_repr] += 1
        self.states.append(state_repr)
        
    def sample(self, batch_size: int = 256):
        ind = np.random.randint(low=0, high=len(self.states), size=(batch_size,))
        torch_ind = torch.tensor(ind, dtype=torch.int64)
        sampled_states = [self.states[i] for i in ind]
        sampled_counts = torch.tensor([self.counts[k] for k in sampled_states]).to(self.device) + self.eps
        batch_rewards = 1/sampled_counts.sqrt()
        
        return batch_rewards.unsqueeze(dim=-1), torch_ind
        