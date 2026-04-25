import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from copy import deepcopy

from rnd_exploration.dataset import ReplayBufferBoot
from four_room.arch import CNN
from utils.episode import LastEpisode
from four_room.utils import obs_to_state
from dqn.archs import UVUBase, L2Norm


class UVUModule(nn.Module):

    def __init__(
        self,
        env: gym.Env,
        use_cnn: bool = True,
        num_heads: int = 10,
        cnn_features: int = 512,
        hidden_layers: list = [512, 512, 512],
        residual: bool = True,
        init: str = "orthogonal",
        act: nn.Module = nn.ReLU,
        use_state: bool = False,
        stack_linear: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_actions = env.action_space.n

        self.layers = nn.Sequential()

        assert not (use_cnn and use_state), "Cant have both state and cnn"
        if use_cnn:
            self.layers.extend(
                [
                    CNN(
                        observation_space=env.observation_space,
                        features_dim=cnn_features,
                        residual=residual,
                    ),
                    act(),
                ]
            )
        elif use_state:
            self.layers.extend([nn.Linear(9, cnn_features), act()])
        else:
            self.layers.extend(
                [
                    nn.Flatten(),
                    nn.Linear(np.prod(env.observation_space.shape), cnn_features),
                    act(),
                ]
            )

        self.layers.extend([nn.Linear(cnn_features, hidden_layers[0]), L2Norm()])

        for layer1, layer2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.extend([nn.Linear(layer1, layer2), act()])

        self.num_heads = num_heads
        self.stack_linear = stack_linear

        if stack_linear:
            self.linears = nn.ModuleList(
                [
                    nn.Linear(hidden_layers[-1], self.num_actions)
                    for _ in range(num_heads)
                ]
            )
        else:
            self.layers.extend(
                [L2Norm(), nn.Linear(hidden_layers[-1], num_heads * self.num_actions)]
            )

        self.scale = scale

        self.apply(self.orthogonal_layer_init if init == "orthogonal" else self._init)

    def _init(self, m):
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.stack_linear:
            rep = self.layers(x)
            out = torch.stack([lin(rep) for lin in self.linears], dim=1)
            out = out.view(-1, self.num_heads, self.num_actions)
        else:
            out = self.layers(x)
            out = out.view(-1, self.num_heads, self.num_actions)
        return out * self.scale

    def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        if hasattr(layer, "weight"):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.uniform_(layer.bias, -1, 1)


class UVU:

    def __init__(
        self,
        env: gym.Env,
        val_env: gym.Env,
        use_cnn: bool = False,
        use_dual: bool = False,
        use_action: bool = False,
        use_norm: bool = True,
        boostrap_prob: float = 1.0,
        capacity: int = int(1e5),
        gamma: float = 0.99,
        start_epsilon: float = 0.99,
        max_decay: float = 0.1,
        decay_steps: float = 10000,
        lr: float = 5e-4,
        act: nn.Module = nn.ReLU,
        tau: float = 0.005,
        hidden_layers: list = [512, 512, 512],
        hidden_layers_g: list = [128],
        num_heads: int = 10,
        device: str = "cuda",
        grad_norm: float = 10.0,
        scale_params: bool = False,
        scale: float = 1.0,
        init_func: str = "kaiming",
        *args,
        **kwargs,
    ):
        # self.net = UVUModule(
        #     env=env,
        #     use_cnn=use_cnn,
        #     hidden_layers=hidden_layers,
        #     cnn_features=cnn_features,
        #     init=init,
        #     act=act,
        #     scale=scale,
        #     num_heads=num_heads,
        #     use_state=use_state,
        # ).to(device)

        # self.target_net = deepcopy(self.net).to(device)

        # self.g = UVUModule(
        #     env=env,
        #     use_cnn=use_cnn,
        #     hidden_layers=hidden_layers_g,
        #     cnn_features=cnn_features,
        #     init=init,
        #     act=act,
        #     scale=scale,
        #     num_heads=num_heads,
        #     use_state=use_state,
        # ).to(device)

        self.net = UVUBase(
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_cnn=use_cnn,
            use_dual=use_dual,
            use_norm=use_norm,
            use_action=use_action,
            init_func=init_func,
            hidden_layers=hidden_layers,
            activation_fn=act,
            num_heads=num_heads,
        ).to(device)

        self.target_net = deepcopy(self.net).to(device)

        self.g = UVUBase(
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_cnn=use_cnn,
            use_norm=use_norm,
            use_dual=use_dual,
            use_action=use_action,
            init_func=init_func,
            hidden_layers=hidden_layers_g,
            activation_fn=act,
            num_heads=num_heads,
        ).to(device)

        self.num_heads = num_heads

        if scale_params:
            for param in self.g.parameters():
                param.requires_grad = False
                param.data = param.data * 10
        else:
            for param in self.g.parameters():
                param.requires_grad = False

        self.env = env
        self.val_env = val_env
        self.start_epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon

        self.buffer = ReplayBufferBoot(
            state_dim=env.observation_space.shape,
            bootstap_prob=boostrap_prob,
            capacity=capacity,
            num_actions=env.action_space.n,
            device=device,
            num_heads=num_heads,
            use_state=not use_cnn,
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.tau = tau
        self.gamma = gamma
        self.grad_norm = grad_norm
        self.loss = nn.MSELoss()
        self.device = device
        self.scale = scale
        self.use_cnn = use_cnn

    def __call__(self, state: torch.Tensor):
        return self.net(state) * self.scale

    def soft_update(self):
        with torch.no_grad():
            for param, target_param in zip(
                self.net.parameters(), self.target_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def get_obs(self, obs: np.ndarray):
        if not self.use_cnn:
            np_state = np.array(list(obs))
            return (
                torch.from_numpy(np_state)
                .view(1, len(np_state))
                .to(self.device)
                .float()
            )
        else:
            return torch.from_numpy(obs).to(device=self.device).unsqueeze(dim=0).float()

    def eval(self, num_runs: int = 10, seed: int = 0):
        self.net.eval()
        rewards = []
        for _ in range(num_runs):
            obs, _ = self.val_env.reset(seed=seed)
            done = False
            ep_reward = 0

            while not done:
                with torch.no_grad():
                    obs_torch = (
                        torch.as_tensor(obs, dtype=torch.float)
                        .view(1, -1)
                        .to(self.device)
                    )
                    action = self.net(obs_torch).view(-1).cpu().numpy().argmax()

                    obs_prime, reward, terminated, truncated, _ = self.val_env.step(
                        action
                    )
                    ep_reward += reward

                    obs = obs_prime
                    done = terminated or truncated

            rewards.append(ep_reward)

        self.net.train()
        return np.mean(rewards)

    @torch.no_grad()
    def epistemic(self, state: torch.Tensor, action: torch.Tensor):
        actions = action.unsqueeze(dim=1).repeat(1, self.num_heads, 1)
        u = self.net(state).gather(index=actions, dim=-1)  # (b, 1)
        g = self.g(state).gather(index=actions, dim=-1)
        return (u - g).pow(2).mean(dim=1) * self.scale

    @torch.no_grad()
    def epistemic_no_act(self, state: torch.Tensor):
        u = self.net(state)
        g = self.g(state)
        return (u - g).pow(2).mean(dim=1) * self.scale

    @torch.no_grad()
    def reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_act: torch.Tensor,
        dones: torch.Tensor,
    ):
        # g - y * g'
        g_cur = self.g(state).gather(index=action, dim=-1)  # (b, m, 1)
        g_next = self.g(next_state).gather(index=next_act, dim=-1)

        return g_cur - self.gamma * g_next * (1 - dones)  # (b, m, 1)

    def update_step(self, batch_size: int):
        (
            batch_obs,
            batch_actions,
            _,
            batch_primes,
            batch_next_actions,
            batch_dones,
            batch_masks,
        ) = self.buffer.sample(batch_size)

        with torch.no_grad():
            batch_rewards = self.reward(
                batch_obs, batch_actions, batch_primes, batch_next_actions, batch_dones
            )
            batch_rewards = batch_rewards.detach()  # (b, m, 1)
            target_vals = self.target_net(batch_primes).gather(
                dim=-1, index=batch_next_actions
            )
            targets = batch_rewards + self.gamma * target_vals * (1 - batch_dones)

        q_values = self.net(batch_obs).gather(dim=-1, index=batch_actions)
        loss_heads = (targets - q_values) ** 2 * batch_masks.unsqueeze(
            dim=-1
        )  # (b, m, 1)
        loss_heads = loss_heads.squeeze(-1)
        # sum the heads * mask and then mean over the batch
        loss = (loss_heads.sum(dim=0) / batch_masks.sum(dim=0)).sum()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.grad_norm
        )
        self.optimizer.step()

    def save(self, path):
        torch.save(
            {"net": self.net.state_dict(), "g": self.g.state_dict()},
            f"results/models/{path}.pt",
        )

    def load(self, path):
        saved_model = torch.load(f"results/models/{path}.pt", weights_only=True)
        self.net.load_state_dict(saved_model["net"])
        self.g.load_state_dict(saved_model["g"])

    # only need this for testing
    def epsilon_greedy(self, state, dim=1):
        rng = np.random.random()

        if rng < self.epsilon:
            action = self.env.action_space.sample()
            action = torch.tensor(action)
        else:
            with torch.no_grad():
                q_values = self.net(state)

            action = torch.argmax(q_values, dim=dim)

        return action

    def epsilon_decay(self, step):
        self.epsilon = self.max_decay + (self.start_epsilon - self.max_decay) * max(
            0, (self.decay_steps - step) / self.decay_steps
        )
