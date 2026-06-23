from rnd_exploration.dataset import MovingSet, State, Transition

import gymnasium as gym
import numpy as np
import torch
import queue

from hashlib import sha1
from gymnasium import spaces
from torch import Tensor
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from stable_baselines3.common.buffers import ReplayBuffer
from collections import deque


class ReplayBufferBase:

    def __init__(
        self,
        state_dim: tuple,
        use_state: bool,
        num_actions: int = 3,
        capacity: int = int(1e5),
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.device = device
        self.pointer = 0
        self.size = 0

        state_space = (
            (self.capacity, *state_dim) if not use_state else (self.capacity, 13)
        )

        self.states = torch.zeros(state_space, dtype=torch.float, device=self.device)
        self.q_values = torch.zeros(
            (self.capacity, num_actions), dtype=torch.float, device=self.device
        )
        self.actions = torch.zeros(
            (self.capacity, 1), dtype=torch.int64, device=self.device
        )
        self.rewards = torch.zeros(
            (self.capacity, 1), dtype=torch.float, device=self.device
        )
        self.next_states = torch.zeros(
            state_space, dtype=torch.float, device=self.device
        )
        self.next_actions = torch.zeros(
            (self.capacity, 1), dtype=torch.int64, device=self.device
        )
        self.dones = torch.zeros(
            (self.capacity, 1), dtype=torch.int, device=self.device
        )
        self.counts = np.zeros((200, 19, 19, 4))

        self.trans = deque(maxlen=self.capacity)
        self.unique_trans = MovingSet(capacity=capacity)
        self.obs = MovingSet(capacity=capacity)
        self.seen_obs = deque(maxlen=self.capacity)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_action: np.ndarray,
        done: float | bool,
        *,
        q_value: np.ndarray | None = None,
    ) -> None:

        self.states[self.pointer] = torch.as_tensor(state).to(self.device)
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward if reward else 0
        self.next_states[self.pointer] = torch.as_tensor(next_state).to(self.device)
        self.next_actions[self.pointer] = next_action
        self.dones[self.pointer] = done

        if q_value is not None:
            self.q_values[self.pointer] = torch.as_tensor(q_value).to(self.device)
            state_obj = State(state=state)
            trans = Transition(state=state, q_value=q_value)
            self.trans.append(trans)
            self.unique_trans.add(trans)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 256):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        batch_torch = (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.next_actions[ind],
            self.dones[ind],
        )
        return batch_torch

    def sample_index(self, ind: Tensor):
        batch_torch = (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.next_actions[ind],
            self.dones[ind],
        )
        return batch_torch

    def __len__(self):
        return self.size

    def update_seen(self, obj: tuple):
        self.obs.add(obj)
        self.counts[*obj] += 1
        self.seen_obs.append(obj)

    def has(self, obj: tuple):
        if self.counts[*obj] > 0:
            return True
        return False

    @property
    def ratio_unique_trans(self):
        return (
            self.unique_trans.num_unique / len(self.trans)
            if len(self.trans) > 0
            else 0.0
        )


class ReplayBufferBoot(ReplayBufferBase):

    def __init__(
        self,
        state_dim,
        use_state,
        num_actions=3,
        num_heads=10,
        capacity=int(100000),
        device="cuda",
        bootstap_prob=0.8,
    ):
        super().__init__(state_dim, use_state, num_actions, capacity, device)

        self.num_heads = num_heads
        self.bootstrap_prob = bootstap_prob
        self.masks = torch.zeros((capacity, num_heads), device=self.device)

    def sample(self, batch_size: int = 256):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        batch_torch = (
            self.states[ind],
            self.actions[ind].unsqueeze(dim=1).repeat(1, self.num_heads, 1),
            self.rewards[ind],
            self.next_states[ind],
            self.next_actions[ind].unsqueeze(dim=1).repeat(1, self.num_heads, 1),
            self.dones[ind].unsqueeze(dim=1).repeat(1, self.num_heads, 1),
            self.masks[ind],
        )
        return batch_torch

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_action: np.ndarray,
        done: float | bool,
        *,
        q_value: np.ndarray | None = None,
    ) -> None:

        self.masks[self.pointer] = torch.bernoulli(
            torch.full((self.num_heads,), self.bootstrap_prob)
        )

        super().update(
            state, action, reward, next_state, next_action, done, q_value=q_value
        )


class ExploreGoReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        include_pure_experience: bool = False,
    ):
        assert optimize_memory_usage == False, "Optimize_memory_usage has to be False."
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.experience_queue = queue.Queue()
        self.include_pure_experience = include_pure_experience

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        normal_inds: np.ndarray,
    ) -> None:
        if self.include_pure_experience:
            super().add(obs, next_obs, action, reward, done, infos)
        else:
            for i in range(obs.shape[0]):
                # First add normal (non-pure) experience to the experience queue
                if normal_inds[i] == True:
                    experience_tuple = (
                        obs[i],
                        next_obs[i],
                        action[i],
                        reward[i],
                        done[i],
                        infos[i],
                    )
                    self.experience_queue.put(experience_tuple)

            # Add experience to the buffer once enough has been collected
            if self.experience_queue.qsize() >= self.n_envs:
                obs_list = []
                next_obs_list = []
                action_list = []
                reward_list = []
                done_list = []
                infos_list = []
                for _ in range(self.n_envs):
                    experience_tuple = self.experience_queue.get()
                    obs_list.append(experience_tuple[0])
                    next_obs_list.append(experience_tuple[1])
                    action_list.append(experience_tuple[2])
                    reward_list.append(experience_tuple[3])
                    done_list.append(experience_tuple[4])
                    infos_list.append(experience_tuple[5])
                obs_list = np.stack(obs_list, axis=0)
                next_obs_list = np.stack(next_obs_list, axis=0)
                action_list = np.stack(action_list, axis=0)
                reward_list = np.stack(reward_list, axis=0)
                done_list = np.stack(done_list, axis=0)

                super().add(
                    obs_list,
                    next_obs_list,
                    action_list,
                    reward_list,
                    done_list,
                    infos_list,
                )


class UvuGoReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        env: Optional[gym.Env] = None,
        n_envs: int = 1,
        state_action_bonus: bool = True,
        uncertainty: str = "egreedy",
        uncertainty_of_sampling: bool = False,
        episodic_discount: bool = True,
        split_uncertainty: bool = True,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        include_pure_experience: bool = False,
    ):
        assert optimize_memory_usage == False, "Optimize_memory_usage has to be False."
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.step_count = 0
        self.uncertainty = uncertainty
        self.env = env
        self.device = device
        self.recently_added_transitions = set()
        self.state_action_bonus = state_action_bonus
        self.uncertainty_of_sampling = uncertainty_of_sampling
        self.episodic_discount = episodic_discount
        self.split_uncertainty = split_uncertainty
        self.include_pure_experience = include_pure_experience
        assert not (
            self.episodic_discount and self.uncertainty_of_sampling
        ), "Episodic sampling and uncertainty of buffer sampling is not supported."
        assert not (
            self.split_uncertainty and not self.state_action_bonus
        ), "Split uncertainty can only be done with a state-action bonus."

        if self.episodic_discount:
            if self.split_uncertainty:
                self.rewards = np.zeros(
                    (self.buffer_size, self.n_envs, 1 + self.action_space.n),
                    dtype=np.float32,
                )
            else:
                self.rewards = np.zeros(
                    (self.buffer_size, self.n_envs, 2), dtype=np.float32
                )

        self.trans = deque(maxlen=buffer_size)
        self.unique_trans = MovingSet(capacity=buffer_size)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # add the transitions for uniquneess calc
        hash_obs = int(sha1(obs.tobytes()).hexdigest(), 16)
        hash_act = int(sha1(action.tobytes()).hexdigest(), 16)
        hash_next_obs = int(sha1(next_obs.tobytes()).hexdigest(), 16)

        self.trans.append((hash_obs, hash_act, hash_next_obs))
        self.unique_trans.add((hash_obs, hash_act, hash_next_obs))

        if self.step_count < 500_000:
            normalise = True
        else:
            normalise = False

        if not self.uncertainty == "egreedy":
            self.uncertainty.observe(obs, action, done, update_rms=normalise)

            actions = torch.as_tensor(range(self.action_space.n), device=self.device).repeat(obs.shape[0]).unsqueeze(1)
            obs_repeated = torch.repeat_interleave(torch.as_tensor(next_obs, device=self.device), self.action_space.n, dim=0)
            intrinsic_reward = self.uncertainty(obs_repeated, actions).reshape(obs.shape[0], -1).detach().cpu().numpy()
            reward = np.concatenate([np.expand_dims(reward, axis=-1), intrinsic_reward], axis=1)

        self.skip_add(obs, next_obs, action, reward, done, infos)
    
    def skip_add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        self.step_count += self.n_envs

    def sample(self, batch_size, env=None):
        if not self.optimize_memory_usage:
            sampled_batch = super().sample(batch_size=batch_size, env=env)
        else:
            # Do not sample the element with index `self.pos` as the transitions is invalid
            # (we use only one array to store `obs` and `next_obs`)
            if self.full:
                batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
            sampled_batch = super()._get_samples(batch_inds, env=env)

        real_rewards = sampled_batch.rewards
        if not self.uncertainty == "egreedy":
            with torch.no_grad():
                if self.state_action_bonus:
                    if self.episodic_discount:
                        if self.split_uncertainty:
                            intrinsic_rewards = []
                            for i in range(self.action_space.n):
                                intrinsic_rewards.append(sampled_batch.rewards.reshape(-1, 1 + self.action_space.n)[:, 1+i].unsqueeze(-1))
                            real_rewards = sampled_batch.rewards.reshape(-1, 1 + self.action_space.n)[:, 0].unsqueeze(-1)
                        else:
                            intrinsic_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 1].unsqueeze(-1)
                            intrinsic_rewards = intrinsic_rewards * self.uncertainty(sampled_batch.observations, sampled_batch.actions, global_only=True).unsqueeze(-1)
                            real_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 0].unsqueeze(-1)
                    else:
                        intrinsic_rewards = self.uncertainty(sampled_batch.observations, sampled_batch.actions).unsqueeze(dim=-1)
                else:
                    if self.episodic_discount:
                        intrinsic_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 1].unsqueeze(-1)
                        intrinsic_rewards = intrinsic_rewards * self.uncertainty(sampled_batch.next_observations).unsqueeze(-1)
                        real_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 0].unsqueeze(-1)
                    else:
                        intrinsic_rewards = self.uncertainty(sampled_batch.next_observations).unsqueeze(dim=-1)
        else:
            intrinsic_rewards = real_rewards
        if self.split_uncertainty:
            sampled_batch = sampled_batch._replace(rewards=torch.stack([real_rewards, *intrinsic_rewards], dim=0))
        else:
            sampled_batch = sampled_batch._replace(rewards=torch.stack([real_rewards, intrinsic_rewards], dim=0))

        if self.uncertainty_of_sampling and not self.uncertainty == "egreedy":
            # our uncertainty measures epistemic uncertainty with respect to what we have sampled for training
            if self.state_action_bonus:
                self.uncertainty.observe(sampled_batch.observations, sampled_batch.actions, update_rms=False)
            else:
                self.uncertainty.observe(sampled_batch.next_observations, update_rms=False)

        return sampled_batch

    @property
    def uniqueness(self):
        return (
            self.unique_trans.num_unique / len(self.trans)
            if len(self.trans) > 0
            else 0.0
        )
