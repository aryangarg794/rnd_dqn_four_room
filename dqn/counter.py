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
        self.counts = defaultdict(int)
        self.device = device
        self.counts_matrix = np.zeros((200, 19, 19))
        self.eps = 1e-3
        
    
    def __getitem__(self, key: tuple):
        return 1/(np.sqrt(self.counts[key]) + self.eps)
        
    def add(self, state_repr: tuple, context: int = None):
        self.counts[state_repr] += 1
        self.states.append(state_repr)
        if context is not None:
            self.counts_matrix[context, state_repr[0], state_repr[1]] += 1
        
    def sample(self, batch_size: int = 256):
        ind = np.random.randint(low=0, high=len(self.states), size=(batch_size,))
        torch_ind = torch.tensor(ind, dtype=torch.int64)
        sampled_states = [self.states[i] for i in ind]
        batch_rewards = torch.tensor([self[k] for k in sampled_states]).to(self.device) + self.eps
        
        return batch_rewards.unsqueeze(dim=-1), torch_ind
    
class MovingCountBasedUncertainty:
    
    def __init__(
        self,
        capacity: int,
        device: str = 'cuda',
        return_ones: bool = True
    ):
        self.capacity = capacity
        self.states = np.empty((capacity,), dtype=object)
        self.all_states = []
        self.pointer = 0
        self.counts = np.zeros((200, 19, 19))
        self.device = device
        self.size = 0
        self.eps = 1e-3
        self.return_ones = return_ones
        
    
    def __getitem__(self, state_repr: tuple):
        if self.return_ones:
            return 1 if self.counts[*state_repr] > 0 else 0
        else:
            return 1/(np.sqrt(self.counts[*state_repr]) + self.eps)
        
    def add(self, state_repr: tuple, timestep: int = None):
        context, x, y = state_repr
        self.all_states.append((timestep, context, x, y))
        self.counts[context, x, y] += 1
        stale_repr = self.states[self.pointer] if self.size > 0 else None
        if stale_repr and self.counts[*stale_repr] > 0:
            self.counts[*stale_repr] -= 1
        
        self.states[self.pointer] = state_repr
        self.pointer = (self.pointer + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int = 256):
        ind = np.random.randint(low=0, high=self.size, size=(batch_size,))
        torch_ind = torch.tensor(ind, dtype=torch.int64)
        sampled_states = [self.states[i] for i in ind]
        batch_rewards = torch.tensor([self[*k] for k in sampled_states]).to(self.device) + self.eps
        
        return batch_rewards.unsqueeze(dim=-1), torch_ind
        