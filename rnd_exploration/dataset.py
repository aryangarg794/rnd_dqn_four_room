import numpy as np
import torch

from hashlib import sha1
from collections import defaultdict, OrderedDict


class Dataset:

    def __init__(self):
        self.X = []
        self.Y = []
        self.states = []

    def add(self, x, y, state):
        self.X.append(x)
        self.Y.append(y)
        self.states.append(state)

    def sort(self):
        binned_dict = defaultdict(list)

        X = np.array(self.X)
        Y = np.array(self.Y)

        for q in np.unique(Y, axis=0):
            mask = np.all(Y == q, axis=1)
            binned_dict[tuple(q)] = X[mask]
        return binned_dict

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return len(self.X)


class State:

    def __init__(self, state: np.ndarray):
        self.state = state

    def __eq__(self, other):
        return bool(np.all(self.state == other.state))

    def __hash__(self):
        state_bytes = self.state.tobytes()
        return int(sha1(state_bytes).hexdigest(), 16)


class Transition:

    def __init__(self, state: np.ndarray, q_value: np.ndarray):
        self.state = state
        self.q_value = q_value

    def __eq__(self, other):
        return bool(
            np.all(self.state == other.state) and np.all(self.q_value == other.q_value)
        )

    def __hash__(self):
        state_hash = int(sha1(self.state.flatten()).hexdigest(), 16)
        q_value_hash = int(sha1(self.q_value.flatten()).hexdigest(), 16)
        return hash((state_hash, q_value_hash))


class Trajectory:

    def __init__(self):
        self.transitions = []
        self.unique_transitions = set([])

    def add(self, transition: Transition):
        self.transitions.append(transition)
        self.unique_transitions.add(transition)

    def __eq__(self, other):
        if len(self.transitions) != len(other.transitions):
            return False
        return all(t1 == t2 for t1, t2 in zip(self.transitions, other.transitions))

    def __hash__(self):
        t_matrix_state = np.array([t.state for t in self.transitions]).flatten()
        t_matrix_q = np.array([t.state for t in self.transitions]).flatten()
        state_hash = int(sha1(t_matrix_state).hexdigest(), 16)
        q_value_hash = int(sha1(t_matrix_q).hexdigest(), 16)
        return hash((state_hash, q_value_hash))

    def uniqueness(self, other):
        intersection = self.unique_transitions.intersection(other.unique_transitions)
        return len(intersection) / len(
            self.transitions
        )  # compares how unique is our trajectory compared to other

    def __iter__(self):
        return iter([(t.state, t.q_value) for t in self.transitions])


class ExploreGoDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.trajectories = []
        self.unique_trajectories = set([])
        self.trans = []
        self.unique_trans = set([])
        self.current_traj = Trajectory()

    def wrap_trajectory(self):
        self.trajectories.append(self.current_traj)
        self.unique_trajectories.add(self.current_traj)
        for state, q in self.current_traj:
            self.add(state, q, None)  # not storing the state
        self.current_traj = Trajectory()

    def reset(self):
        self.current_traj = Trajectory()

    def add_traj(self, obs: np.ndarray, q_value: np.ndarray, state: np.ndarray):
        transition = Transition(obs, q_value)
        self.current_traj.add(transition)
        self.unique_trans.add(transition)

    def add_trans(self, obs: np.ndarray, q_value: np.ndarray):
        transition = Transition(obs, q_value)
        self.trans.append(transition)
        self.unique_trans.add(transition)

    def traj_uniqueness(self, trajectory: Trajectory):
        unique = 0
        for traj in self.unique_trajectories:
            unique = max(unique, trajectory.uniqueness(traj))
        return unique

    def sample(self, batch_size: int = 128):
        idxs = np.random.randint(low=0, high=len(self), size=(batch_size,))
        xs = np.array(self.X)[idxs]
        ys = np.array(self.Y)[idxs]
        return (
            torch.as_tensor(xs, dtype=torch.float32),
            torch.as_tensor(ys, dtype=torch.float32),
        )

    @property
    def ratio_unique_trans(self):
        return len(self.unique_trans) / len(self.trans) if len(self.trans) > 0 else 0.0


class MovingSet:

    def __init__(self, capacity: int = int(1e5)):
        self.size = 0
        self.capacity = capacity
        self.unique_set = OrderedDict()
        self._index = 0
        self._stop = self.size

    def add(self, obj: object):
        self.unique_set[obj] = self.unique_set.get(obj, 0) + 1
        if self.size + 1 > self.capacity:  # overflow
            pair = self.unique_set.popitem(last=False)
            if pair[1] - 1 > 0:
                self.unique_set[pair[0]] = pair[1] - 1
                self.unique_set.move_to_end(pair[0], last=False)

        self.size = min(self.size + 1, self.capacity)
        self._stop = self.size

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self._stop:
            self._keys = list(self.unique_set.keys())
            key = self._keys[self._index]
            self._index += 1
            return key
        else:
            raise StopIteration()

    def has(self, obj: object):
        return obj in self.unique_set.keys()

    @property
    def num_unique(self):
        return len(self.unique_set)

    def __repr__(self):
        return str(self.unique_set.items())
