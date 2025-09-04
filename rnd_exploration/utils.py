import numpy as np
from collections import deque
import dill

def compare(a, b):

    els = np.sum(a != b)
    total = a.size
    per = 100 * els / total
    return els


class RunningAverage:
    def __init__(self, window_size=250):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)

    @property
    def avg(self):
        if len(self.values) > 0 :
            return float(np.mean(self.values))
        return 0.0
    
    @property
    def std(self):
        if len(self.values) > 0 :
            return float(np.std(self.values))
        return 1.0

    def reset(self):
        self.values.clear()
        