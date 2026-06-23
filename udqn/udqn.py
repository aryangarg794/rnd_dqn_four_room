import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from utils.statistics import RunningAverageTorch

from buffers.buffers import UvuGoReplayBuffer
from stable_baselines3.common.type_aliases import (
    GymEnv,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
    MaybeCallback,
)
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.dqn import DQN


class ExploreGoUVU(DQN):

    def __init__(
        self,
        policy: Union[str],
        env: Union[GymEnv, str],
        beta: float = 0.01,
        uncertainty=None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = int(5e5),
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        u_tau: float = 1.0,
        uvu_tau: float = 0.2,
        alpha: float = 0.75,
        uvu_grad_norm: float = 10.0,
        go_vs_uvu: float = 0.05,
        num_heads: int = 512, 
        window_size: int = 2500,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        uvu_gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[UvuGoReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        double_q: bool = False,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        max_pure_expl_steps: int = 0,
    ):
        self.double_q = double_q
        self.beta = beta
        self.uncertainty = uncertainty
        self.u_tau = u_tau
        self._uvu_updates = 0
        self.uvu_tau = uvu_tau
        self.uvu_gradient_steps = uvu_gradient_steps
        self.max_uvu_norm = uvu_grad_norm
        self.go_vs_uvu = go_vs_uvu
        self.alpha = alpha
        self.num_heads = num_heads

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            # double_q=double_q,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.betas = np.array([beta for _ in range(self.n_envs)])
        self.max_pure_expl_steps = max_pure_expl_steps
        self.num_pure_expl_steps = np.random.randint(
            0, max_pure_expl_steps + 1, size=env.num_envs
        )
        self.mode_per_env = np.random.random(size=env.num_envs) < self.go_vs_uvu
        self.running_means = RunningAverageTorch(
            num_envs=env.num_envs, window_size=window_size, device=self.device
        )

        if self.replay_buffer.include_pure_experience == False:
            self.record_per_env = np.zeros((env.num_envs,), dtype=bool)
        else:
            self.record_per_env = np.ones((env.num_envs,), dtype=bool)

        self.episode_steps = np.zeros(env.num_envs)
        self.num_normal_steps = 0

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.u_net = self.policy.u_net
        self.u_net_target = self.policy.u_net_target
        self.uvu_net = self.policy.uvu_net
        self.uvu_net_target = self.policy.uvu_net_target
        self.g_net = self.policy.g_net
        self.policy._set_uncertainty(self.uncertainty)

    def _on_step(self) -> None:
        super()._on_step()
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(
                self.u_net.parameters(), self.u_net_target.parameters(), self.u_tau
            )
            polyak_update(
                self.uvu_net.parameters(),
                self.uvu_net_target.parameters(),
                self.uvu_tau,
            )

        self.logger.record("rollout/beta", self.beta)

    @torch.no_grad()
    def uvu_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        actions = action.unsqueeze(dim=1).repeat(1, self.num_heads, 1)
        g_cur = self.g_net(obs).gather(index=actions, dim=-1)  # (b, m, 1)
        next_act = self.policy(next_obs).max(dim=1)[1].unsqueeze(dim=1)
        next_act = next_act.unsqueeze(dim=1).repeat(1, self.num_heads, 1)
        g_next = self.g_net(next_obs).gather(index=next_act, dim=-1)
        return g_cur - self.gamma * g_next * (1 - dones)

    @torch.no_grad()
    def epistemic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        actions = action.unsqueeze(dim=1).repeat(1, self.num_heads, 1)
        u = self.uvu_net(obs).gather(index=actions, dim=-1)  # (b, h, 1)
        g = self.g_net(obs).gather(index=actions, dim=-1)
        return (u - g).pow(2).mean(dim=1)

    def train(
        self, gradient_steps: int, uvu_gradient_steps: int, batch_size: int = 100
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        u_losses = []
        uvu_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                if self.double_q:
                    # Compute the next Q-values using the current network
                    next_q_values_current = self.q_net(replay_data.next_observations)
                    # Determine argmax based on the current network values
                    actions = next_q_values_current.max(dim=1)[1].unsqueeze(dim=1)
                    next_q_values = next_q_values.gather(dim=1, index=actions)
                else:
                    actions = next_q_values.max(dim=1)[1].unsqueeze(dim=1)
                    next_q_values = next_q_values.gather(dim=1, index=actions)

                    # 1-step TD target
                    target_q_values = (
                        replay_data.rewards[0]
                        + (1 - replay_data.dones) * self.gamma * next_q_values
                    )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())

            if not np.all(self.betas == 0):
                with torch.no_grad():
                    if self.uncertainty is not None:
                        next_obs_shape = replay_data.next_observations.shape
                        actions = (
                            torch.as_tensor(
                                range(self.action_space.n), device=self.device
                            )
                            .repeat(next_obs_shape[0])
                            .unsqueeze(1)
                        )
                        next_obs_repeated = torch.repeat_interleave(
                            replay_data.next_observations, self.action_space.n, dim=0
                        )
                        novelties = torch.concatenate(
                            [ir for ir in replay_data.rewards[1:]], dim=-1
                        ) * self.uncertainty(
                            next_obs_repeated, actions, global_only=True
                        ).reshape(
                            next_obs_shape[0], -1
                        )

                    # Compute the next uncertainties using the target network
                    next_u_values = self.u_net_target(replay_data.next_observations)
                    if self.double_q:
                        # Compute the next Q-values using the current network
                        next_u_values_current = self.u_net(
                            replay_data.next_observations
                        )
                        # Determine argmax based on the current network values
                        if self.uncertainty is not None:
                            actions = (
                                (next_u_values_current + novelties)
                                .max(dim=1)[1]
                                .unsqueeze(dim=1)
                            )
                        else:
                            actions = next_u_values_current.max(dim=1)[1].unsqueeze(
                                dim=1
                            )
                    else:
                        if self.uncertainty is not None:
                            actions = (
                                (next_u_values + novelties)
                                .max(dim=1)[1]
                                .unsqueeze(dim=1)
                            )
                        else:
                            actions = next_u_values.max(dim=1)[1].unsqueeze(dim=1)

                    if self.uncertainty is not None:
                        next_u_values = (next_u_values + novelties).gather(
                            dim=1, index=actions
                        )
                        # 1-step TD target
                        target_u_values = (
                            (1 - replay_data.dones) * self.gamma * next_u_values
                        )
                        # target_u_values = self.gamma * next_u_values
                    else:
                        next_u_values = next_u_values.gather(dim=1, index=actions)
                        # 1-step TD target
                        target_u_values = (
                            replay_data.rewards[1]
                            + (1 - replay_data.dones) * self.gamma * next_u_values
                        )
                        # target_u_values = replay_data.rewards[1] + self.gamma * next_u_values

                # Get current uncertainty estimates
                current_u_values = self.u_net(replay_data.observations)

                # Retrieve the uncertainties for the actions from the replay buffer
                current_u_values = torch.gather(
                    current_u_values, dim=1, index=replay_data.actions.long()
                )

                # Compute Huber loss (less sensitive to outliers)
                u_loss = F.smooth_l1_loss(current_u_values, target_u_values)

                # Optimize the policy
                self.policy.u_optimizer.zero_grad()
                u_loss.backward()
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.u_net.parameters(), self.max_grad_norm
                )
                self.policy.u_optimizer.step()
                u_losses.append(u_loss.item())
            else:
                u_losses.append(0)

        # uvu updating
        for _ in range(uvu_gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            with torch.no_grad():
                uvu_rewards = self.uvu_reward(
                    replay_data.observations,
                    replay_data.actions,
                    replay_data.next_observations,
                    replay_data.dones,
                ).detach()

                # NOTE: right now no option to use double q
                policy_next_acts = (
                    self.policy(replay_data.next_observations)
                    .max(dim=1)[1]
                    .unsqueeze(dim=1)
                )
                target_vals = self.uvu_net_target(replay_data.next_observations).gather(
                    dim=-1, index=policy_next_acts
                )
                targets = uvu_rewards + self.gamma * target_vals * (
                    1 - replay_data.dones
                )

            uvu_values = self.uvu_net(replay_data.observations).gather(
                dim=-1, index=replay_data.actions
            )
            loss_heads = F.smooth_l1_loss(uvu_values, targets, reduction="none")
            loss_uvu = loss_heads.squeeze(dim=-1).sum(dim=-1).mean()
            uvu_losses.append(loss_uvu.item())

            self.policy.uvu_optimizer.zero_grad()
            loss_uvu.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.uvu_net.parameters(), self.max_uvu_norm
            )
            self.policy.uvu_optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
        self._uvu_updates += uvu_gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/u_loss", np.mean(u_losses))
        self.logger.record("train/uvu_loss", np.mean(uvu_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(
                        batch_size=self.batch_size,
                        gradient_steps=gradient_steps,
                        uvu_gradient_steps=self.uvu_gradient_steps,
                    )

        callback.on_training_end()

        return self

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["replay_buffer_kwargs"]

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
            self.num_normal_steps += n_envs
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            with torch.no_grad():
                unscaled_action = self.policy._predict_pure(self._last_obs)

                # if the decision is to record then we sample the proper policy actions
                unscaled_action_normal, _ = self.predict(
                    self._last_obs, deterministic=False
                )

            unscaled_action[self.record_per_env] = unscaled_action_normal[
                self.record_per_env
            ]
            self.num_normal_steps += sum(self.record_per_env)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: UvuGoReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert (
                train_freq.unit == TrainFrequencyUnit.STEP
            ), "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # compute the uvu values
            with torch.no_grad():
                observation, _ = self.policy.obs_to_tensor(self._last_obs)
                policy_acts = self.policy(observation).max(dim=1)[1].unsqueeze(dim=1)
                uvu_vals = self.epistemic(observation, policy_acts)
                new_record = self.running_means.check(self.alpha, uvu_vals)

                self.running_means.update(uvu_vals)
                if self.num_timesteps < learning_starts:
                    new_record = np.ones((env.num_envs,), dtype=bool)
                else:
                    new_record[self.mode_per_env] = (
                        self.episode_steps >= self.num_pure_expl_steps
                    )[self.mode_per_env]

                self.record_per_env = self.record_per_env | new_record

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # If the last step of the pure exploration phase, set done to True
            buffer_dones = deepcopy(dones)
            if self.replay_buffer.include_pure_experience == False:
                last_pure_indices = self.episode_steps == (self.num_pure_expl_steps - 1)
                buffer_dones[last_pure_indices] = np.array(
                    [True for _ in range(last_pure_indices.sum())]
                )

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, buffer_dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            self.episode_steps += 1
            for idx, done in enumerate(dones):
                if done:
                    self.episode_steps[idx] = 0
                    self.num_pure_expl_steps[idx] = np.random.randint(
                        0, self.max_pure_expl_steps + 1
                    )
                    self.record_per_env[idx] = (
                        False if not replay_buffer.include_pure_experience else True
                    )
                    self.mode_per_env = np.random.random() < self.go_vs_uvu

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def _store_transition(
        self,
        replay_buffer: UvuGoReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :]
                        )

        # only add the transitions that are recorded
        obs_to_add = self._last_original_obs[self.record_per_env]
        next_obs_to_add = next_obs[self.record_per_env]
        actions_to_add = buffer_action[self.record_per_env]
        rewards_to_add = reward_[self.record_per_env]
        dones_to_add = dones[self.record_per_env]
        infos_to_add = []
        indices = np.where(self.record_per_env)[0]
        for idx in indices:
            infos_to_add.append(infos[idx])

        replay_buffer.add(
            obs_to_add,  # type: ignore[arg-type]
            next_obs_to_add,  # type: ignore[arg-type]
            actions_to_add,
            rewards_to_add,
            dones_to_add,
            infos_to_add,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
