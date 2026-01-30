import torch

from collections import deque

from dqn.counter import MovingCountBasedUncertainty

class LastEpisode:
    
    def __init__(self, state_dim, capacity=5, device='cuda'):
        self.capacity = capacity
        
        self.device = device
        self.pointer = 0
        self.size = 0
        
        self.states = torch.zeros((self.capacity, *state_dim) ,dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.capacity, 1) ,dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1) ,dtype=torch.float, device=self.device)
        self.next_states = torch.zeros((self.capacity, *state_dim) ,dtype=torch.float, device=self.device)
        self.next_actions = torch.zeros((self.capacity, 1) ,dtype=torch.int64, device=self.device)
        self.dones = torch.zeros((self.capacity, 1) ,dtype=torch.int, device=self.device)
        self.tuples = deque(maxlen=capacity)
    
    def update(self, state, action, next_state, next_action, done, obj_tuple, reward=0.0):
        self.states[self.pointer] = torch.as_tensor(state).to(self.device)
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = torch.as_tensor(next_state).to(self.device)
        self.next_actions[self.pointer] = next_action
        self.dones[self.pointer] = done
        self.tuples.append(obj_tuple)
        
        self.pointer = (self.pointer + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)
    
    def get(self, counter: MovingCountBasedUncertainty = None):
        if counter: 
            rewards = [counter[*obj_tuple] for obj_tuple in self.tuples]
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).view(-1, 1)
        else:
            rewards = self.rewards[:self.size]
        
        return (
            self.states[:self.size], 
            self.actions[:self.size], 
            rewards,
            self.next_states[:self.size], 
            self.next_actions[:self.size],
            self.dones[:self.size]
        )