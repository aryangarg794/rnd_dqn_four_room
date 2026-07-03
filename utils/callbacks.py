import dill
import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback
from four_room.constants import train_reachable_space, train_reachable_space_opt_actions, obs_to_q_values_map

from collections import deque
from rnd_exploration.dataset import MovingSet, State, Transition, TransitionSA, TransitionSAS


CONFIGS_DIR = "../configs/"


class UniquenesseCallback(BaseCallback):
    def __init__(self, log_freq, verbose=0):
        super(UniquenesseCallback, self).__init__(verbose)
        self.log_freq = log_freq

    def _on_training_start(self):
        self.total_added = 0
        self.last_buffer_obs = self.model.replay_buffer.observations.reshape(-1, *self.model.observation_space.shape).copy()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            buffer_obs = self.model.replay_buffer.observations.reshape(-1, *self.model.observation_space.shape)
            buffer_acts = self.model.replay_buffer.actions.reshape(-1, self.model.replay_buffer.action_dim)
            current_idx = buffer_obs.shape[0] if self.model.replay_buffer.full else self.model.replay_buffer.pos

            unique_trans = MovingSet(capacity=buffer_obs.shape[0])

            for obs, act in zip(buffer_obs[:current_idx], buffer_acts[:current_idx]):
                trans = TransitionSA(state=obs, action=act)
                unique_trans.add(trans)

            uniqueness = self.compute_uniqueness(unique_trans, current_idx)
            num_changed = np.unique((buffer_obs == self.last_buffer_obs).sum(axis=0))[0]
            self.total_added += num_changed
            self.last_buffer_obs = buffer_obs.copy()

            
            self.logger.record("train/uniqueness", uniqueness)
            self.logger.record("train/buffer_size", current_idx)
            self.logger.record("train/total_added", self.total_added)

        return True
    
    def compute_uniqueness(self, unique_trans: MovingSet, current_size: int):
        return (
            unique_trans.num_unique / current_size
            if current_size > 0
            else 0.0
        )


class ExplorationCoverageCallback(BaseCallback):
    def __init__(self, log_freq, total_states, num_actions, verbose=0):
        super(ExplorationCoverageCallback, self).__init__(verbose)
        self.state_action_coverage_set = set()
        self.log_freq = log_freq
        self.total_state_actions = total_states * num_actions

    def _on_step(self) -> bool:
        for i, obs in enumerate(self.locals["env"].buf_obs[None]):
            action = self.locals["actions"][i]
            self.state_action_coverage_set.add(
                hash((hash(obs.data.tobytes()), hash(action.data.tobytes())))
            )

        if self.num_timesteps % self.log_freq == 0:
            self.logger.record(
                "train/state_action_coverage_exploration",
                len(self.state_action_coverage_set) / self.total_state_actions,
            )

        return True


class BufferCoverageCallback(BaseCallback):
    """
    Custom callback for calculating the policy optimality and plotting it in tensorboard.
    """

    def __init__(self, freq, total_states, num_actions, verbose=0):
        super(BufferCoverageCallback, self).__init__(verbose)
        self.freq = freq
        self.num_actions = num_actions
        self.total_states = total_states
        self.total_state_actions = total_states * num_actions
        self.data = train_reachable_space

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            state_action_count = dict()
            for obs in self.data:
                state_action_count[hash(obs.data.tobytes())] = dict()

            if self.model.replay_buffer.full:
                for i, obs_stack in enumerate(self.model.replay_buffer.observations):
                    for j, obs in enumerate(obs_stack):
                        obs_hash = hash(obs.data.tobytes())
                        action_hash = self.model.replay_buffer.actions[
                            i, j
                        ].data.tobytes()
                        if obs_hash in state_action_count:
                            # if this is not the case this is because it is a terminal state which is just in the buffer for bootstrapping reasons
                            if action_hash in state_action_count[obs_hash]:
                                state_action_count[obs_hash][action_hash] += 1
                            else:
                                state_action_count[obs_hash][action_hash] = 1
                        else:
                            print(obs[0] == obs[-1])
            else:
                for i, obs_stack in enumerate(
                    self.model.replay_buffer.observations[
                        : self.model.replay_buffer.pos
                    ]
                ):
                    for j, obs in enumerate(obs_stack):
                        obs_hash = hash(obs.data.tobytes())
                        action_hash = self.model.replay_buffer.actions[
                            i, j
                        ].data.tobytes()
                        if obs_hash in state_action_count:
                            # if this is not the case this is because it is a terminal state which is just in the buffer for bootstrapping reasons
                            if action_hash in state_action_count[obs_hash]:
                                state_action_count[obs_hash][action_hash] += 1
                            else:
                                state_action_count[obs_hash][action_hash] = 1
                        else:
                            print(obs[0] == obs[-1])

            zero_count = 0
            state_actions_missing = 0

            for k, v in state_action_count.items():
                if len(v.keys()) == 0:
                    zero_count += 1
                if len(v.keys()) < self.num_actions:
                    state_actions_missing += self.num_actions - len(v.keys())

            self.logger.record(
                "train/state_coverage",
                (self.total_states - zero_count) / self.total_states,
            )
            self.logger.record(
                "train/state_action_coverage",
                (self.total_state_actions - state_actions_missing)
                / self.total_state_actions,
            )

        return True


class PolicyOptimalityCallback(BaseCallback):
    """
    Custom callback for calculating the policy optimality and plotting it in tensorboard.
    """

    def __init__(self, freq, num_training_levels, verbose=0, device='cpu'):
        super(PolicyOptimalityCallback, self).__init__(verbose)
        self.freq = freq
        self.num_training_levels = num_training_levels
        self.device = device

        self.reachable_batch = torch.as_tensor(train_reachable_space, device=device)
        self.optimal_actions = train_reachable_space_opt_actions
        self.obs_to_optimal_values = obs_to_q_values_map

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            max_action = []
            max_values = []
            num_batches = int(self.num_training_levels * 0.2)
            batch_size = self.reachable_batch.shape[0] // num_batches
            for start_ind in range(0, self.reachable_batch.shape[0], batch_size):
                with torch.no_grad():
                    values = self.model.q_net(
                        self.reachable_batch[start_ind : start_ind + batch_size]
                    )
                    max_action.append(values.max(dim=-1)[1].cpu().numpy())
                    max_values.append(values.max(dim=-1)[0].cpu().numpy())

            max_action = np.concatenate(max_action, axis=0)
            max_values = np.concatenate(max_values, axis=0)

            same_sum = 0
            value_diff = 0
            for i in range(self.reachable_batch.shape[0]):
                max_v = max_values[i].item()
                max_a = max_action[i].item()
                if max_a in self.optimal_actions[i]:
                    same_sum += 1

                opt_v = max(
                    self.obs_to_optimal_values[
                        self.reachable_batch[i].cpu().numpy().data.tobytes()
                    ]
                )
                value_diff += abs(max_v - opt_v)

            self.logger.record(
                "eval/policy_optimality", same_sum / self.reachable_batch.shape[0]
            )
            self.logger.record("eval/policy_optimality_values", value_diff)

            if self.model.replay_buffer.full:
                states_in_buffer = np.unique(
                    self.model.replay_buffer.observations.reshape(
                        -1, *self.reachable_batch[0].shape
                    ),
                    axis=0,
                )
                max_action = []
                max_values = []
                batch_size = 2048
                start_ind = 0
                while start_ind < states_in_buffer.shape[0]:
                    # for start_ind in range(0, states_in_buffer.shape[0], batch_size):
                    with torch.no_grad():
                        values = self.model.q_net(
                            torch.as_tensor(
                                states_in_buffer[start_ind : start_ind + batch_size],
                                device=self.device,
                            )
                        )
                        max_action.append(values.max(dim=-1)[1].cpu().numpy())
                        max_values.append(values.max(dim=-1)[0].cpu().numpy())
                    start_ind += batch_size

                max_action = np.concatenate(max_action, axis=0)
                max_values = np.concatenate(max_values, axis=0)

                same_sum = 0
                value_diff = 0
                for i in range(states_in_buffer.shape[0]):
                    max_v = max_values[i].item()
                    max_a = max_action[i].item()

                    opt_v = max(
                        self.obs_to_optimal_values[states_in_buffer[i].data.tobytes()]
                    )
                    opt_a = (
                        np.argwhere(
                            self.obs_to_optimal_values[
                                states_in_buffer[i].data.tobytes()
                            ]
                            == np.amax(
                                self.obs_to_optimal_values[
                                    states_in_buffer[i].data.tobytes()
                                ]
                            )
                        )
                        .flatten()
                        .tolist()
                    )
                    value_diff += abs(max_v - opt_v)
                    if max_a in opt_a:
                        same_sum += 1

                self.logger.record(
                    "eval/policy_optimality_buffer",
                    same_sum / states_in_buffer.shape[0],
                )
                self.logger.record(
                    "eval/policy_optimality_values_buffer",
                    value_diff / states_in_buffer.shape[0],
                )

        return True
