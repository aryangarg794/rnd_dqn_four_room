import numpy as np
import torch

from collections import deque


class RunningAverage:
    def __init__(self, window_size=250):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.means = []
        self.stds = []

    def update(self, value):
        self.values.append(value)
        self.means.append(self.avg)
        self.stds.append(self.std)

    @property
    def avg(self):
        if len(self.values) > 0:
            return float(np.mean(self.values))
        return 0.0

    @property
    def std(self):
        if len(self.values) > 1:
            std = float(np.std(self.values))
            return std if std > 0 else 1.0
        return 1.0

    def reset(self):
        self.values.clear()


class RunningAverageTorch:
    def __init__(self, num_envs: int, window_size: int = 250, device: str = "cuda"):
        self.window_size = window_size
        self.num_envs = num_envs
        self.device = device
        self.values = torch.zeros((window_size, num_envs), device=device)
        self.pos = 0
        self.size = 0

    def update(self, values: torch.Tensor):
        self.values[self.pos] = values.squeeze()
        self.pos = (self.pos + 1) % self.window_size
        self.size = min(self.size + 1, self.window_size)

    def check(self, alpha: float, values: torch.Tensor):
        values = values.squeeze()
        norm = (values - self.avg) / self.std
        return (norm >= alpha).cpu().numpy()

    @property
    def avg(self):
        if self.size < 2:
            return torch.zeros((50,), device=self.device)
        else:
            return self.values[:self.size].mean(dim=0)

    @property
    def std(self):
        if self.size < 2:
            return torch.ones((50,), device=self.device)
        else:
            return self.values[:self.size].std(dim=0)
            

    def reset(self):
        self.values = torch.zeros((self.window_size, self.num_envs), device=self.device)

def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    suffixes = ["", "k", "M", "B", "T"]
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), suffixes[magnitude]
    )