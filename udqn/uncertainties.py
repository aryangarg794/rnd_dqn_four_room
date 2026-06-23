import torch as th
import numpy as np
from typing import Tuple
from functools import reduce
import hashlib


def get_hash(input):
    return hashlib.sha256(input).hexdigest()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class CountSAUncertainty:
    """
    Implements novelty with counts.
    Only works with discrete state spaces.
    """

    def __init__(self, total_state_action_space, obs_shape, device="cpu"):
        # this will be a dictionary keeping count of states encountered
        # self.state_action_counts_keys = HashTable(2*total_state_action_space,  ('u1', reduce(lambda x, y: x*y, obs_shape) + 1), almost_full=(0.8, 3.0))
        # self.state_action_counts_values = np.zeros(self.state_action_counts_keys.max, 'u4')
        self.state_action_counts = dict()
        self.eps = 1e-7
        self.device = device
        self.novelty_rms = RunningMeanStd(shape=())
        self.max_novelty = 0

    def observe(self, state, action, update_rms=False):
        if isinstance(state, th.Tensor):
            state = state.detach().cpu().numpy()
            action = action.detach().cpu().numpy()
        if isinstance(state, np.ndarray):
            if len(state.shape) == 3:
                np.expand_dims(state, axis=0)

        action = action.reshape((state.shape[0], 1)).astype("uint8")

        # self.state_action_counts_values[
        #     self.state_action_counts_keys.add(np.concatenate([state.reshape(state.shape[0], -1), action], axis=-1))
        #     ] += 1

        for i, s in enumerate(state):
            state_action_hash = get_hash(
                repr(
                    (get_hash(s.data.tobytes()), get_hash(action[i].data.tobytes()))
                ).encode()
            )
            if state_action_hash in self.state_action_counts:
                self.state_action_counts[state_action_hash] += 1
            else:
                self.state_action_counts[state_action_hash] = 1

        if update_rms:
            self.update_rms(state, action)

    def update_rms(self, state, action):
        if isinstance(state, th.Tensor):
            state = state.detach().cpu().numpy()
            action = action.detach().cpu().numpy()
        if isinstance(state, np.ndarray):
            if len(state.shape) == 3:
                np.expand_dims(state, axis=0)

        action = action.reshape((state.shape[0], 1)).astype("uint8")

        # n = np.array(self.state_action_counts_values[
        #     self.state_action_counts_keys.get(np.concatenate([state.reshape(state.shape[0], -1), action], axis=-1))
        # ])
        n = np.zeros(len(state))
        for i, s in enumerate(state):
            state_action_hash = get_hash(
                repr(
                    (get_hash(s.data.tobytes()), get_hash(action[i].data.tobytes()))
                ).encode()
            )
            n[i] = self.state_action_counts.get(state_action_hash, 0)

        novelty = 1.0 / np.sqrt(n + self.eps)
        self.novelty_rms.update(novelty)

    def __call__(self, state, action, binary=False, **kwargs):
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) ans Tensor.
        'state' can be either a Tuple, List, 1d Tensor or 2d Tensor (1d Tensors stacked in dim=0).
        Does not change the counters."""
        if isinstance(state, th.Tensor):
            state = state.detach().cpu().numpy()
            action = action.detach().cpu().numpy()
        if isinstance(state, np.ndarray):
            if len(state.shape) == 3:
                state = np.expand_dims(state, axis=0)

        action = action.reshape((state.shape[0], 1)).astype("uint8")

        # n = np.array(self.state_action_counts_values[
        #     self.state_action_counts_keys.get(np.concatenate([state.reshape(state.shape[0], -1), action], axis=-1))
        # ])
        n = np.zeros(len(state))
        for i, s in enumerate(state):
            state_action_hash = get_hash(
                repr(
                    (get_hash(s.data.tobytes()), get_hash(action[i].data.tobytes()))
                ).encode()
            )
            n[i] = self.state_action_counts.get(state_action_hash, 0)

        if binary:
            novelty = n == 0
        else:
            novelty = 1.0 / np.sqrt(n + self.eps)
            if 1.0 / np.sqrt(n.min() + self.eps) > self.max_novelty:
                self.max_novelty = 1.0 / np.sqrt(n.min() + self.eps)
            novelty = novelty / self.max_novelty
            # std = np.sqrt(self.novelty_rms.var + self.eps)
            # novelty = (novelty - self.novelty_rms.mean) / std
            # novelty = np.clip(novelty, a_min=2*std, a_max=None)

        return th.as_tensor(novelty, device=th.device(self.device)).float()


class EpisodicCountSAUncertainty:
    def __init__(
        self, n_envs, episode_timeout, obs_shape, device="cpu", global_uncertainty=None
    ):
        self.counters = []
        for _ in range(n_envs):
            self.counters.append(
                CountSAUncertainty(episode_timeout * 2, obs_shape, device=device)
            )
        self.episode_timeout = episode_timeout
        self.obs_shape = obs_shape
        self.device = device
        self.global_uncertainty = global_uncertainty
        self.n_envs = n_envs

    def observe(self, state, action, done, update_rms=False, indices=None):
        if indices is None:
            indices = np.arange(self.n_envs)

        novelty = []
        for i, s in enumerate(state):
            if done[i]:
                self.counters[indices[i]] = CountSAUncertainty(
                    self.episode_timeout * 2, self.obs_shape, device=self.device
                )
            novelty.append(
                self.counters[indices[i]](
                    np.expand_dims(s, axis=0),
                    np.expand_dims(action[i], axis=0),
                    binary=True,
                )
            )
            self.counters[indices[i]].observe(
                np.expand_dims(s, axis=0),
                np.expand_dims(action[i], axis=0),
                update_rms=update_rms,
            )

        if self.global_uncertainty is not None:
            self.global_uncertainty.observe(state, action, update_rms=update_rms)

        return th.concatenate(novelty, dim=0).detach().cpu().numpy()

    def get_episodic_bonus(self, state, action):
        novelty = []
        for i, s in enumerate(state):
            novelty.append(
                self.counters[i](
                    np.expand_dims(s, axis=0),
                    np.expand_dims(action[i], axis=0),
                    binary=True,
                )
            )
        return th.concatenate(novelty, dim=0).detach().cpu().numpy()

    def __call__(self, state, action, global_only=False, indices=None, **kwargs):
        if indices is None:
            indices = np.arange(self.n_envs)

        if isinstance(state, np.ndarray):
            state = th.as_tensor(state, device=self.device)
            action = th.as_tensor(action, device=self.device)
        if self.global_uncertainty is not None:
            bonus = self.global_uncertainty(state, action)
        else:
            bonus = th.ones((state.shape[0]), device=self.device).float()

        if not global_only:
            novelty = []
            for i, s in enumerate(state):
                if state.shape[0] == 3 * self.n_envs:
                    novelty.append(
                        self.counters[indices[i // 3]](
                            s.unsqueeze(0), action[i].unsqueeze(0), binary=True
                        )
                    )
                else:
                    novelty.append(
                        self.counters[indices[i]](
                            s.unsqueeze(0), action[i].unsqueeze(0), binary=True
                        )
                    )
            novelty = th.concatenate(novelty, dim=0)
            bonus = bonus * novelty

        return bonus
