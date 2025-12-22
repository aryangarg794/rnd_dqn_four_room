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
        self.counts_matrix = np.zeros((200, 19, 19, 4))
        self.eps = 1e-1
        
    
    def __getitem__(self, key: tuple):
        return 1/(np.sqrt(self.counts[key]) + self.eps)
        
    def add(self, state_repr: tuple):
        self.counts[state_repr] += 1
        self.states.append(state_repr)
        self.counts_matrix[state_repr[-1], state_repr[0], state_repr[1], state_repr[2]] += 1
        
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
        return_ones: bool = True,
        dir: bool = True
    ):
        self.capacity = capacity
        self.states = np.empty((capacity,), dtype=object)
        self.all_states = []
        self.pointer = 0
        if dir:
            self.counts = np.zeros((200, 19, 19, 4)) #NOTE: Hard coded
        else:
            self.counts = np.zeros((200, 19, 19))
        self.device = device
        self.size = 0
        self.eps = 1e-1
        self.return_ones = return_ones
        self.dir = dir
        self.max_val = 1 if return_ones else 1/(np.sqrt(0) + self.eps)
    
    def __getitem__(self, state_repr: tuple):
        if self.return_ones:
            return 0 if self.counts[*state_repr] > 0 else 1
        else:
            return 1/(np.sqrt(self.counts[*state_repr]) + self.eps)
        
    def add(self, state_repr: tuple, timestep: int = None):
        self.all_states.append((timestep, *state_repr))
        if self.size == self.capacity:
            stale_repr = self.states[self.pointer]
            self.counts[*stale_repr] = max(0, self.counts[*stale_repr] - 1)
        
        self.states[self.pointer] = state_repr
        self.counts[*state_repr] += 1
        self.pointer = (self.pointer + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)
        
        if self.size >= self.capacity:
            assert self.counts.sum() == self.capacity
        
    def sample(self, batch_size: int = 256):
        ind = np.random.randint(low=0, high=self.size, size=(batch_size,))
        torch_ind = torch.tensor(ind, dtype=torch.int64)
        sampled_states = [self.states[i] for i in ind]
        batch_rewards = torch.tensor([self[*k] for k in sampled_states], dtype=torch.float32).to(self.device) + self.eps
        
        return batch_rewards.unsqueeze(dim=-1), torch_ind

    @property
    def counts_no_dir(self):
        return self.counts.sum(axis=-1) if self.dir else self.counts

if __name__ == "__main__":
    import numpy as np
    import torch

    def test_basic_add_no_overflow():
        print("Test: add <= capacity")
        cap = 5
        m = MovingCountBasedUncertainty(capacity=cap)

        state = (0, 0, 0, 0)

        for i in range(cap):
            m.add(state)

        assert m.counts[state] == cap, f"Expected count={cap}, got {m.counts[state]}"
        assert m.size == cap, f"Expected size={cap}, got {m.size}"
        assert m.counts.sum() == cap, f"Total counts mismatch: {m.counts.sum()}"
        print(" OK")

    def test_overflow_removal():
        print("Test: overflow removes oldest")
        cap = 3
        m = MovingCountBasedUncertainty(capacity=cap)

        s1 = (0, 0, 0, 0)
        s2 = (0, 0, 1, 0)
        s3 = (0, 0, 2, 0)
        s4 = (0, 0, 3, 0)

        m.add(s1)  # count(s1) = 1
        m.add(s2)  # count(s2) = 1
        m.add(s3)  # count(s3) = 1

        # Overflow: s1 should be removed
        m.add(s4)

        assert m.counts[s1] == 0, f"s1 should have been removed, got count={m.counts[s1]}"
        assert m.counts[s2] == 1, "s2 should still be present"
        assert m.counts[s3] == 1, "s3 should still be present"
        assert m.counts[s4] == 1, "s4 should have count=1"
        assert m.counts.sum() == cap, f"Counts should sum to {cap}"

        print(" OK")

    def test_getitem_behavior():
        print("Test: __getitem__ (return_ones=True)")
        m = MovingCountBasedUncertainty(capacity=5, return_ones=True)

        s = (1, 1, 1, 1)
        assert m[s] == 1, "__getitem__ should return 1 for unseen state"

        m.add(s)
        assert m[s] == 0, "__getitem__ should return 0 after first visit"
        print(" OK")

    def test_sample_shapes():
        print("Test: sample()")
        m = MovingCountBasedUncertainty(capacity=10)
        state = (0, 1, 2, 3)

        for _ in range(10):
            m.add(state)

        rewards, idx = m.sample(batch_size=4)

        assert rewards.shape == (4, 1), f"Rewards shape incorrect: {rewards.shape}"
        assert idx.shape == (4,), f"Indices shape incorrect: {idx.shape}"
        print(" OK")

    # Run tests
    test_basic_add_no_overflow()
    test_overflow_removal()
    test_getitem_behavior()
    test_sample_shapes()

    print("\nAll tests passed.\n")
