import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from copy import deepcopy
from torch.nn.functional import mse_loss

from buffers.buffers import ReplayBufferBase
from utils.statistics import RunningAverage
from dqn.archs import DQNModule
from utils.episode import LastEpisode
from dqn.counter import MovingCountBasedUncertainty


class DQN:

    def __init__(
        self,
        env: gym.Env,
        val_env: gym.Env,
        use_cnn=False,
        use_dual=False,
        use_action=False,
        use_norm=True,
        capacity: int = int(1e5),
        gamma: float = 0.99,
        modulation: str = "concat",
        start_epsilon: float = 0.99,
        max_decay: float = 0.1,
        act: nn.Module = nn.ReLU,
        return_ones: bool = False,
        decay_steps: float = 10000,
        lr: float = 5e-4,
        tau: float = 0.005,
        hidden_layers: list = [512, 512, 512],
        device: str = "cuda",
        grad_norm: float = 10.0,
        init_func: str = "kaiming",
        *args,
        **kwargs
    ):
        self.net = DQNModule(
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_cnn=use_cnn,
            use_dual=use_dual,
            use_norm=use_norm,
            use_action=use_action,
            init_func=init_func,
            hidden_layers=hidden_layers,
            activation_fn=act,
            modulation=modulation,
        ).to(device)

        self.target_net = deepcopy(self.net).to(device)

        self.env = env
        self.val_env = val_env
        self.start_epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon
        self.use_cnn = use_cnn

        self.buffer = ReplayBufferBase(
            state_dim=env.observation_space.shape,
            capacity=capacity,
            num_actions=env.action_space.n,
            device=device,
            use_state=not use_cnn,
        )

        self.counter = MovingCountBasedUncertainty(
            device=device, capacity=capacity, return_ones=return_ones
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.tau = tau
        self.device = device
        self.grad_norm = grad_norm
        self.gamma = gamma

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

    def update_step(
        self,
        batch_size: int,
        last_ep: LastEpisode = None,
        last_expl: LastEpisode = None,
    ):
        batch_rewards, ind = self.counter.sample(batch_size=batch_size)
        (
            batch_obs,
            batch_actions,
            _,
            batch_primes,
            batch_next_actions,
            batch_dones,
        ) = self.buffer.sample_index(ind)

        obs, primes, acts, n_acts, dones, rewards = (
            [batch_obs],
            [batch_primes],
            [batch_actions],
            [batch_next_actions],
            [batch_dones],
            [batch_rewards],
        )

        if last_ep:
            (
                last_obs,
                last_action,
                last_rewards,
                last_obs_primes,
                last_next_actions,
                last_dones,
            ) = last_ep.get(self.counter)

            obs.append(last_obs)
            primes.append(last_obs_primes)
            acts.append(last_action)
            n_acts.append(last_next_actions)
            dones.append(last_dones)
            rewards.append(last_rewards)

        if last_expl:
            (
                last_obs_expl,
                last_action_expl,
                last_rewards_expl,
                last_obs_primes_expl,
                last_next_actions_expl,
                last_dones_expl,
            ) = last_ep.get(self.counter)

            obs.append(last_obs_expl)
            primes.append(last_obs_primes_expl)
            acts.append(last_action_expl)
            n_acts.append(last_next_actions_expl)
            dones.append(last_dones_expl)
            rewards.append(last_rewards_expl)

        batch_obs = torch.cat(obs, dim=0)
        batch_actions = torch.cat(acts, dim=0)
        batch_primes = torch.cat(primes, dim=0)
        batch_next_actions = torch.cat(n_acts, dim=0)
        batch_dones = torch.cat(dones, dim=0)
        batch_rewards = torch.cat(rewards, dim=0)

        with torch.no_grad():
            batch_rewards = batch_rewards.detach()
            target_vals = self.target_net(batch_primes).gather(
                dim=1, index=batch_next_actions
            )
            targets = batch_rewards + self.gamma * target_vals * (1 - batch_dones)

        q_values = self.net(batch_obs).gather(dim=1, index=batch_actions)
        loss = mse_loss(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.grad_norm
        )
        self.optimizer.step()

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

    def __call__(self, state: torch.Tensor):
        return self.net(state)

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
