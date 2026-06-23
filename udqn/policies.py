from typing import Any, Dict, List, Optional, Type

import gymnasium as gym
import torch
import numpy as np
from torch import nn

from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.dqn.policies import QNetwork

from stable_baselines3.common.torch_layers import (
    create_mlp,
)

from dqn.archs import *
from dqn.archs import _kaiming_init, _orthogonal_init
from four_room.arch import CNN


class UVUNetwork(BasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int = 256,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        norm: bool = True,
        init: str = "kaiming",
        num_heads: int = 1,
        *args,
        squash_output=False,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 512]
        assert net_arch[0] == features_dim

        self.num_heads = num_heads
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.num_actions = self.action_space.n
        action_dim = int(self.action_space.n)  # number of actions

        self.norm = L2Norm() if norm else nn.Identity()
        self.uvu_net = nn.Sequential(nn.Linear(net_arch[0] + action_dim, net_arch[1]))

        for layer1, layer2 in zip(net_arch[1:-1], net_arch[2:]):
            self.uvu_net.append(nn.Linear(layer1, layer2))
            self.uvu_net.append(activation_fn())

        self.uvu_net.append(L2Norm() if norm else nn.Identity())
        self.uvu_net.append(nn.Linear(net_arch[-1], num_heads))

        self.apply(_orthogonal_init if init == "orthogonal" else _kaiming_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.size(0)

        batched_input = obs.repeat(
            self.num_actions, *[1 for _ in range(len(self.observation_space.shape))]
        )
        batched_act = (
            torch.arange(0, self.num_actions, device=obs.device)
            .repeat_interleave(batch_size)
            .reshape(-1, 1)
        )
        act = nn.functional.one_hot(batched_act, self.num_actions).float().squeeze(1)

        features = self.extract_features(batched_input, self.features_extractor)
        features = self.norm(features)
        input_with_act = torch.cat([features, act], dim=-1)

        return (
            self.uvu_net(input_with_act)
            .view(self.num_actions, batch_size, self.num_heads)
            .permute(1, 0, 2)
            .reshape(batch_size, self.num_heads, self.num_actions)
        )

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        q_values = self(observation).mean(dim=1).squeeze(dim=1)
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class UVUGoPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        beta: float,
        u_lr: float,
        uvu_lr: float,
        n_envs: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        uvu_kwargs: Optional[Dict[str, Any]] = None,
        g_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.uvu_kwargs = uvu_kwargs
        self.g_kwargs = g_kwargs
        self.num_heads = uvu_kwargs["num_heads"]

        self.u_net, self.u_net_target = None, None
        u_lr_schedule = get_schedule_fn(u_lr)
        uvu_lr_schedule = get_schedule_fn(uvu_lr)
        self._build_unet(u_lr_schedule)
        self._build_uvu_net(uvu_lr_schedule)
        self.betas = torch.tensor([beta for _ in range(n_envs)])
        self.uncertainty = None

    def _set_uncertainty(self, uncertainty):
        self.uncertainty = uncertainty

    def make_uvu_net(
        self,
        net_kwargs: Optional[Dict[str, Any]] = None,
    ) -> UVUNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        net_args.update(net_kwargs)
        return UVUNetwork(**net_args).to(self.device)

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return QNetwork(**net_args).to(self.device)

    def _build_unet(self, lr_schedule: Schedule) -> None:
        self.u_net = self.make_q_net()
        self.u_net_target = self.make_q_net()
        self.u_net_target.load_state_dict(self.u_net.state_dict())
        self.u_net_target.set_training_mode(False)

        self.u_optimizer = self.optimizer_class(
            self.u_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_uvu_net(self, lr_schedule: Schedule) -> None:
        self.uvu_net = self.make_uvu_net(self.uvu_kwargs)
        self.uvu_net_target = self.make_uvu_net(self.uvu_kwargs)
        self.uvu_net_target.load_state_dict(self.uvu_net.state_dict())
        self.uvu_net_target.set_training_mode(False)

        self.g_net = self.make_uvu_net(self.g_kwargs)
        self.g_net.set_training_mode(False)

        self.uvu_optimizer = self.optimizer_class(
            self.uvu_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # this function is basically only used for sampling actions
        if not self.betas.device == self.device:
            self.betas = self.betas.to(self.device)
        q_values = self.q_net(obs)
        if torch.all(self.betas == 0):
            return q_values
        else:
            uncertainties = self.u_net(obs)

            if self.uncertainty is not None:
                if len(obs.shape) == 1 or len(obs.shape) == 3:
                    # torchere is no batch dimension
                    no_batch_dim = True
                    # novelties = torch.zeros((self.action_space.n), device=obs.device)
                    obs = obs.unsqueeze(0)
                else:
                    no_batch_dim = False

                actions = (
                    torch.as_tensor(range(self.action_space.n), device=self.device)
                    .repeat(obs.shape[0])
                    .unsqueeze(1)
                )
                obs_repeated = torch.repeat_interleave(obs, self.action_space.n, dim=0)
                novelties = self.uncertainty(obs_repeated, actions).reshape(
                    obs.shape[0], uncertainties.shape[-1]
                )

                if no_batch_dim:
                    novelties.squeeze(0)

                # assume torchat if obs.shape[0] is smaller torchan self.betas.shape[0], we are in a setting where beta is torche same everywhere
                if obs.shape[0] == self.betas.shape[0]:
                    return q_values + self.betas.unsqueeze(-1) * (
                        uncertainties + novelties
                    )
                else:
                    return q_values + self.betas[0] * (uncertainties + novelties)
            else:
                if obs.shape[0] == self.betas.shape[0]:
                    return q_values + self.betas.unsqueeze(-1) * uncertainties
                else:
                    return q_values + self.betas[0] * uncertainties

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if deterministic:
            # use only Q, not U
            values = self.q_net(obs)
        else:
            values = self(obs)
        # Greedy action
        action = values.argmax(dim=1).reshape(-1)
        return action

    def _predict_pure(self, obs: torch.Tensor) -> torch.Tensor:
        # Switch to eval mode (torchis affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake torchat torche user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do torchat check
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            raise ValueError(
                "You have passed a tuple to torche predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API witorch SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://gitorchub.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readtorchedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(obs)

        with torch.no_grad():
            if torch.all(self.betas == 0):
                # use only Q, not U
                values = self.q_net(obs_tensor)
            else:
                uncertainties = self.u_net(obs_tensor)
                if self.uncertainty is not None:
                    if len(obs_tensor.shape) == 1 or len(obs_tensor.shape) == 3:
                        # torchere is no batch dimension
                        no_batch_dim = True
                        # novelties = torch.zeros((self.action_space.n), device=obs.device)
                        obs_tensor = obs_tensor.unsqueeze(0)
                    else:
                        no_batch_dim = False

                    actions = (
                        torch.as_tensor(range(self.action_space.n), device=self.device)
                        .repeat(obs_tensor.shape[0])
                        .unsqueeze(1)
                    )
                    obs_repeated = torch.repeat_interleave(
                        obs_tensor, self.action_space.n, dim=0
                    )
                    novelties = self.uncertainty(obs_repeated, actions).reshape(
                        obs_tensor.shape[0], uncertainties.shape[-1]
                    )

                    if no_batch_dim:
                        novelties.squeeze(0)

                    values = uncertainties + novelties
                else:
                    values = uncertainties

            # Greedy pure exploration action
            pure_action = values.argmax(dim=1).reshape(-1)

        # Convert to numpy, and reshape to torche original action shape
        pure_action = pure_action.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                pure_action = self.unscale_action(pure_action)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip torche actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                pure_action = np.clip(pure_action, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(pure_action, np.ndarray)
            pure_action = pure_action.squeeze(axis=0)

        return pure_action
