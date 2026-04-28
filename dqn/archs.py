import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from four_room.arch import ConvSequence


@torch.no_grad()
def _kaiming_init(m):
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if hasattr(m, "bias"):
            nn.init.uniform_(m.bias, -1, 1)


@torch.no_grad()
def _orthogonal_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight"):
        nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.uniform_(layer.bias, -1, 1)


class L2Norm(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        norm = torch.sqrt(torch.clamp(norm_sq, min=self.eps))
        return x / norm


class IdentityExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=0):
        super().__init__(observation_space, features_dim)
        self.id = nn.Identity()

    def forward(self, x):
        return self.id(x)


class CustomQNetwork(BasePolicy):
    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor = IdentityExtractor,
        features_dim: int = 0,
        net_arch=None,
        activation_fn=nn.ReLU,
        normalize_images: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.q_net = nn.Sequential()

    def forward(self, obs: torch.Tensor):
        return self.q_net(obs)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True):
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


# just define all the archs manually
class DQNBase(CustomQNetwork):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        cnn_channels: int = 64,
        use_cnn: bool = True,
        hidden_layers: list = [256, 512],
        norm: bool = True,
        residual: bool = True,
        max_pool: bool = True,
        init: str = "kaiming",
        act: nn.Module = nn.ReLU,
        num_heads: int = 1,
    ) -> None:
        print("using DQNBase")
        self.use_cnn = use_cnn
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=IdentityExtractor,
            features_dim=0,
            normalize_images=False,
        )

        self.layers = nn.Sequential()
        self.image_normaliser = 10.0

        self.layers.extend(
            [
                (
                    ConvSequence(
                        observation_space.shape,
                        cnn_channels,
                        init_function=init,
                        residual=residual,
                        act=act,
                        max_pool=max_pool,
                    )
                    if use_cnn
                    else nn.Linear(13, hidden_layers[0])
                ),
                act(),
            ]
        )

        if use_cnn:
            with torch.no_grad():
                n_flatten = np.prod(
                    self.layers[0](
                        torch.as_tensor(observation_space.sample()[None]).float()
                    ).shape[1:]
                )
        else:
            n_flatten = hidden_layers[0]

        self.layers.extend(
            [
                nn.Flatten() if use_cnn else nn.Identity(),
                L2Norm() if norm else nn.Identity(),
                nn.Linear(n_flatten, hidden_layers[0]),
                act(),
            ]
        )

        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([nn.Linear(layer1, layer2), act()])

        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(hidden_layers[-1], num_heads * action_space.n),
            ]
        )

        self.apply(_orthogonal_init if init == "orthogonal" else _kaiming_init)

    def forward(self, obs):
        obs = obs.float()
        if self.use_cnn:
            obs = obs / self.image_normaliser
        return self.layers(obs)


class DQNBaseAction(CustomQNetwork):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        cnn_channels: int = 64,
        use_cnn: bool = True,
        embed_dim: int = 8,
        hidden_layers: list = [256, 512],
        norm: bool = True,
        modulation: str = 'concat',
        residual: bool = True,
        max_pool: bool = True,
        init: str = "kaiming",
        act: nn.Module = nn.ReLU,
        num_heads: int = 1,
    ) -> None:
        print("using DQNBaseAction")
        self.use_cnn = use_cnn
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=IdentityExtractor,
            features_dim=0,
            normalize_images=False,
        )

        self.obs_layers = nn.Sequential()
        self.act_layers = nn.Sequential()
        self.layers = nn.Sequential()
        self.image_normaliser = 10.0
        self.modulation = modulation

        self.obs_layers.extend(
            [
                (
                    ConvSequence(
                        observation_space.shape,
                        cnn_channels,
                        init_function=init,
                        residual=residual,
                        act=act,
                        max_pool=max_pool,
                    )
                    if use_cnn
                    else nn.Linear(13, hidden_layers[0])
                ),
                act(),
            ]
        )

        if use_cnn:
            with torch.no_grad():
                n_flatten = np.prod(
                    self.obs_layers[0](
                        torch.as_tensor(observation_space.sample()[None]).float()
                    ).shape[1:]
                )
        else:
            n_flatten = hidden_layers[0]

        self.obs_layers.extend(
            [
                nn.Flatten() if use_cnn else nn.Identity(),
                nn.Linear(n_flatten, hidden_layers[0]),
            ]
        )

        if modulation == 'one_hot':
            inp_dim = hidden_layers[0] + action_space.n
        elif modulation == 'concat':
            inp_dim = 2 * hidden_layers[0]
        else:
            inp_dim = hidden_layers[0]

        if not modulation == 'one_hot':
            self.act_embed = nn.Embedding(action_space.n, embed_dim)
            self.act_layers.extend(
                [nn.Linear(embed_dim, 64), act(), nn.Linear(64, hidden_layers[0])]
            )

        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(
                    inp_dim,
                    hidden_layers[0],
                ),
                act(),
            ]
        )

        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([nn.Linear(layer1, layer2), act()])

        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(hidden_layers[-1], num_heads),
            ]
        )
        self.num_heads = num_heads

        self.use_cnn = use_cnn
        self.apply(_orthogonal_init if init == "orthogonal" else _kaiming_init)

    def _forward_act(self, obs, act):
        obs = obs.float()
        if self.use_cnn:
            obs = obs / self.image_normaliser

        obs = self.obs_layers(obs)

        if self.modulation == 'one_hot':
            act = nn.functional.one_hot(act, 3).float().squeeze(1)
        else:
            act = self.act_embed(act).squeeze(dim=1)
            act = self.act_layers(act)

        if self.modulation == 'one_hot' or self.modulation == 'concat':
            inp = torch.cat([obs, act], dim=-1)
        elif self.modulation == 'mult':
            inp = obs * act
        elif self.modulation == 'add':
            inp = obs + act

        return self.layers(inp)

    def forward(self, obs):
        obs = obs.float()
        batch_size = obs.size(0)
        num_actions = self.action_space.n
        batched_input = obs.repeat(
            num_actions, *[1 for _ in range(len(self.observation_space.shape))]
        )
        batched_act = (
            torch.arange(0, num_actions, device=obs.device)
            .repeat_interleave(batch_size)
            .reshape(-1, 1)
        )
        q_vals = self._forward_act(batched_input, batched_act).view(num_actions, batch_size, self.num_heads)
        return q_vals.permute(1, 0, 2).reshape(batch_size, -1)


class DQNBaseDual(CustomQNetwork):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_layers: list = [256, 512],
        norm: bool = True,
        modulation: str = 'concat', 
        init: str = "kaiming",
        act: nn.Module = nn.ReLU,
        num_heads: int = 1,
    ) -> None:
        print("using DQNBaseDual")
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=IdentityExtractor,
            features_dim=0,
            normalize_images=False,
            use_action=False,
        )

        self.obs_layers = nn.Sequential()
        self.context_layers = nn.Sequential()
        self.layers = nn.Sequential()
        self.modulation = modulation

        self.obs_layers.extend(
            [nn.Linear(3, 128), act(), nn.Linear(128, hidden_layers[0])]
        )

        self.context_layers.extend(
            [nn.Linear(10, 128), act(), nn.Linear(128, hidden_layers[0])]
        )

        if self.modulation == 'concat':
            inp_dim = 2 * hidden_layers[0]
        else:
            inp_dim = hidden_layers[0]

        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(
                    inp_dim,
                    hidden_layers[0],
                ),
                act(),
            ]
        )

        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([nn.Linear(layer1, layer2), act()])

        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(hidden_layers[-1], num_heads * action_space.n),
            ]
        )
        self.num_heads = num_heads

        self.apply(_orthogonal_init if init == "orthogonal" else _kaiming_init)

    def forward(self, obs):
        obs = obs.float()
        agent_info = obs[:, :3]
        context = obs[:, 3:]
        agent_info = self.obs_layers(agent_info)
        context = self.context_layers(context)

        if self.modulation == 'concat':
            inp = torch.cat([obs, context], dim=-1)
        elif self.modulation == 'mult':
            inp = obs * context
        else:
            inp = obs + context

        return self.layers(inp)


class DQNBaseDualAction(CustomQNetwork):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        embed_dim: int = 8,
        hidden_layers: list = [256, 512],
        norm: bool = True,
        init: str = "kaiming",
        modulation: str = "concat", 
        act: nn.Module = nn.ReLU,
        num_heads: int = 1,
    ) -> None:
        print("using DQNDualAction")
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=IdentityExtractor,
            features_dim=0,
            normalize_images=False,
            use_action=True,
        )

        self.obs_layers = nn.Sequential()
        self.act_layers = nn.Sequential()
        self.context_layers = nn.Sequential()
        self.layers = nn.Sequential()
        self.modulation = modulation

        self.obs_layers.extend(
            [nn.Linear(3, 128), act(), nn.Linear(128, hidden_layers[0])]
        )

        self.context_layers.extend(
            [nn.Linear(10, 128), act(), nn.Linear(128, hidden_layers[0])]
        )

        self.act_embed = nn.Embedding(action_space.n, embed_dim)
        self.act_layers.extend(
            [nn.Linear(embed_dim, 128), act(), nn.Linear(128, hidden_layers[0])]
        )

        if modulation == 'one_hot':
            inp_dim = hidden_layers[0] + action_space.n
        elif modulation == 'concat_act':
            inp_dim = 2 * hidden_layers[0]
        elif modulation == 'concat_dual':
            inp_dim = 2 * hidden_layers[0]
        elif modulation == 'concat_act_add':
            inp_dim = 2 * hidden_layers[0]
        elif modulation == 'concat':
            inp_dim = 3 * hidden_layers[0]
        else:
            inp_dim = hidden_layers[0] 
        
        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(
                    inp_dim,
                    hidden_layers[0],
                ),
                act(),
            ]
        )

        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([nn.Linear(layer1, layer2), act()])

        self.layers.extend(
            [
                L2Norm() if norm else nn.Identity(),
                nn.Linear(hidden_layers[-1], num_heads),
            ]
        )
        self.num_heads = num_heads

        self.apply(_orthogonal_init if init == "orthogonal" else _kaiming_init)

    def _forward_act(self, obs, act):
        agent_info = obs[:, :3]
        context = obs[:, 3:]

        if self.modulation == 'one_hot':
            act = nn.functional.one_hot(act, 3).float().squeeze(1)
        else:
            act = self.act_embed(act).squeeze(dim=1)
            act = self.act_layers(act)
        
        agent_info = self.obs_layers(agent_info)
        context = self.context_layers(context)

        if self.modulation == 'one_hot':
            inp = torch.cat([agent_info * context, act], dim=-1)
        elif self.modulation == 'concat':
            inp = torch.cat([agent_info, act, context], dim=-1)
        elif self.modulation == 'concat_dual':
            inp = torch.cat([agent_info * act, context], dim=-1)
        elif self.modulation == 'concat_act_add':
            inp = torch.cat([agent_info + act, context], dim=-1)
        elif self.modulation == 'concat_act':
            inp = torch.cat([agent_info * context, act], dim=-1)
        elif self.modulation == 'mult':
            inp = agent_info * context * act
        elif self.modulation == 'add':
            inp = agent_info + context + act
        
        return self.layers(inp)

    def forward(self, obs):
        obs = obs.float()
        batch_size = obs.size(0)
        num_actions = self.action_space.n
        batched_input = obs.repeat(
            num_actions, *[1 for _ in range(len(self.observation_space.shape))]
        )
        batched_act = (
            torch.arange(0, num_actions, device=obs.device)
            .repeat_interleave(batch_size)
            .reshape(-1, 1)
        )
        q_vals = self._forward_act(batched_input, batched_act).view(num_actions, batch_size, self.num_heads)
        return q_vals.permute(1, 0, 2).reshape(batch_size, -1)


class DQNBasePolicy(DQNPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        use_cnn=False,
        use_dual=False,
        use_action=False,
        use_norm=True,
        modulation='concat',
        init_func="kaiming",
        activation_fn=nn.ReLU,
        features_extractor_class=...,
        features_extractor_kwargs=None,
        normalize_images=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        self.use_dual = use_dual
        self.use_cnn = use_cnn
        self.use_action = use_action
        self.use_norm = use_norm
        self.init_func = init_func
        self.modulation = modulation
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self):
        if self.use_dual:
            if self.use_action:
                model = DQNBaseDualAction(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    norm=self.use_norm,
                    init=self.init_func,
                    modulation=self.modulation
                )
            else:
                model = DQNBaseDual(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    norm=self.use_norm,
                    init=self.init_func,
                    modulation=self.modulation
                )
        else:
            if self.use_action:
                model = DQNBaseAction(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    use_cnn=self.use_cnn,
                    norm=self.use_norm,
                    init=self.init_func,
                    modulation=self.modulation,
                )
            else:
                model = DQNBase(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    use_cnn=self.use_cnn,
                    norm=self.use_norm,
                    init=self.init_func,
                )

        return model

    def forward(self, obs: torch.Tensor, deterministic: bool = True):
        obs = obs.float()
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True):
        return self.q_net._predict(obs, deterministic=deterministic)


class UVUBase(nn.Module):

    def __init__(
        self,
        observation_space,
        action_space,
        use_cnn=False,
        use_dual=False,
        use_action=False,
        use_norm=True,
        init_func="kaiming",
        modulation='concat',
        hidden_layers=list([256, 512]),
        activation_fn=nn.ReLU,
        num_heads=10,
        scale=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_dual = use_dual
        self.use_cnn = use_cnn
        self.use_action = use_action
        self.use_norm = use_norm
        self.init_func = init_func
        self.num_heads = num_heads
        self.num_actions = action_space.n
        self.modulation = modulation

        if self.use_dual:
            if self.use_action:
                model = DQNBaseDualAction(
                    observation_space=observation_space,
                    action_space=action_space,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    num_heads=num_heads,
                    hidden_layers=hidden_layers,
                    modulation=self.modulation,
                )
            else:
                model = DQNBaseDual(
                    observation_space=observation_space,
                    action_space=action_space,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    num_heads=num_heads,
                    hidden_layers=hidden_layers,
                    modulation=self.modulation,
                )
        else:
            if self.use_action:
                model = DQNBaseAction(
                    observation_space=observation_space,
                    action_space=action_space,
                    use_cnn=self.use_cnn,
                    norm=self.use_norm,
                    init=self.init_func,
                    one_hot=self.one_hot, 
                    act=activation_fn,
                    num_heads=num_heads,
                    hidden_layers=hidden_layers,
                    modulation=self.modulation,
                )
            else:
                model = DQNBase(
                    observation_space=observation_space,
                    action_space=action_space,
                    use_cnn=self.use_cnn,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    num_heads=num_heads,
                    hidden_layers=hidden_layers,
                )

        self.model = model
        self.scale = scale

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.num_heads, self.num_actions)

        return out * self.scale


class DQNModule(nn.Module):

    def __init__(
        self,
        observation_space,
        action_space,
        use_cnn=False,
        use_dual=False,
        use_action=False,
        use_norm=True,
        init_func="kaiming",
        concat={"action": True, "dual": True},
        hidden_layers=list([256, 512]),
        activation_fn=nn.ReLU,
        scale=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_dual = use_dual
        self.use_cnn = use_cnn
        self.use_action = use_action
        self.use_norm = use_norm
        self.init_func = init_func
        self.num_actions = action_space.n
        self.concat = concat 

        if self.use_dual:
            if self.use_action:
                model = DQNBaseDualAction(
                    observation_space=observation_space,
                    action_space=action_space,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    hidden_layers=hidden_layers,
                    concat=self.concat
                )
            else:
                model = DQNBaseDual(
                    observation_space=observation_space,
                    action_space=action_space,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    hidden_layers=hidden_layers,
                    concat=self.concat
                )
        else:
            if self.use_action:
                model = DQNBaseAction(
                    observation_space=observation_space,
                    action_space=action_space,
                    use_cnn=self.use_cnn,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    hidden_layers=hidden_layers,
                    concat=self.concat
                )
            else:
                model = DQNBase(
                    observation_space=observation_space,
                    action_space=action_space,
                    use_cnn=self.use_cnn,
                    norm=self.use_norm,
                    init=self.init_func,
                    act=activation_fn,
                    hidden_layers=hidden_layers,
                )

        self.model = model
        self.scale = scale

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.num_actions)

        return out * self.scale
