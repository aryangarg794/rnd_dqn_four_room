import os
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import minigrid
import pickle
import tyro
import shutil
from copy import deepcopy

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "uvu"
    wandb_entity: str = None
    log_dir: str = "runs/debug"
    capture_video: bool = True
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "Custom-MultitaskGoToDoor"
    env_level: int = 10
    dataset_f: str = "datasets/gotodoor_med_10x10_SemiExpert.pkl"
    total_timesteps: int = 100_000
    n_task_rejections: int = 4
    learning_rate: float = 3e-4
    uvu_learning_rate: float = 1e-4
    network_width: int = 512
    uvu_sparsity: float = 0.0
    uvu_width: int = 512
    uvu_outs: int = 512
    eval_freq: int = 500
    eval_episodes: int = 10
    num_envs: int = 1
    gamma: float = 0.9
    tau: float = 1.0
    target_network_frequency: int = 256
    batch_size: int = 512
    epsilon: float = 0.001
    prior_scale: float = 1.0
    timeit: bool = False

def make_env(env_id, seed, idx, capture_video, run_name, args):
    def thunk():
        if env_id == "Custom-MultitaskGoToDoor":
            from envs.gotodoor import MultitaskGoToDoorEnv
            if capture_video and idx == 0:
                env = MultitaskGoToDoorEnv(size=args.env_level, render_mode="rgb_array", mode="medium", test_mode=False)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = MultitaskGoToDoorEnv(size=args.env_level, mode="medium")
        else:
            render_mode = "rgb_array" if capture_video and idx == 0 else None
            env = gym.make(env_id, render_mode=render_mode)
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

class UniversalMLPTorch(nn.Module):
    def __init__(self, input_dim, task_dim, n_hidden, n_feats, out_shape, modulation="product"):
        super().__init__()
        self.out_shape = out_shape
        self.modulation = modulation
        
        self.input_layer = nn.Linear(input_dim, n_feats)
        self.task_encoder = nn.Linear(task_dim, n_feats)
        
        layers = []
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_feats, n_feats))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        
        final_out = np.prod(out_shape)
        self.output_layer = nn.Linear(n_feats, final_out)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, task):
        x = nn.functional.layer_norm(self.input_layer(obs), (self.input_layer.out_features,))
        t = self.task_encoder(task)
        
        if self.modulation == "product":
            x = x * t
        
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.view(x.size(0), *self.out_shape)

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    
    obs_dim = obs_space["obs"].shape[0]
    task_dim = obs_space["task"].shape[0]
    
    q_net = UniversalMLPTorch(obs_dim, task_dim, 3, args.network_width, (act_space.n,)).to(device)
    q_target = deepcopy(q_net).to(device)
    q_optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)

    uvu_net = UniversalMLPTorch(obs_dim, task_dim, 3, args.uvu_width, (act_space.n, args.uvu_outs)).to(device)
    uvu_target = deepcopy(uvu_net).to(device)
    uvu_prior = UniversalMLPTorch(obs_dim, task_dim, 2, args.uvu_width, (act_space.n, args.uvu_outs)).to(device)
    for p in uvu_prior.parameters(): p.requires_grad = False
    
    uvu_optimizer = optim.Adam(uvu_net.parameters(), lr=args.uvu_learning_rate)

    with open(args.dataset_f, 'rb') as f:
        rb = pickle.load(f)

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        
        if global_step % args.eval_freq == 0:
            returns, ep_lens = [], []
            while len(returns) < args.eval_episodes:
                obs_t = torch.tensor(obs["obs"], dtype=torch.float32).to(device)
                task_t = torch.tensor(obs["task"], dtype=torch.float32).to(device)
                
                if random.random() < args.epsilon:
                    actions = np.array([act_space.sample()])
                else:
                    with torch.no_grad():
                        actions = q_net(obs_t, task_t).argmax(-1).cpu().numpy()
                
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                if "final_info" in infos:
                    for i, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            returns.append(info["episode"]["r"])
                            ep_lens.append(info["episode"]["l"])
                            
                            ts = envs.envs[i].unwrapped.get_task_selection()
                            t_obs = torch.tensor(next_obs["obs"][i], dtype=torch.float32).to(device).repeat(ts.shape[0], 1)
                            t_task = torch.tensor(ts, dtype=torch.float32).to(device)
                            
                            with torch.no_grad():
                                u_err = uvu_net(t_obs, t_task) - args.prior_scale * uvu_prior(t_obs, t_task)
                                u_std = torch.sqrt(u_err**2).sum(-1)
                                q_vals = q_net(t_obs, t_task)
                                sel_std = torch.gather(u_std, 1, q_vals.argmax(-1, keepdim=True)).squeeze(-1)
                                rejections = torch.argsort(sel_std)[-args.n_task_rejections:].cpu().numpy()
                            
                            next_obs, _ = envs.envs[i].unwrapped.reset(reject_tasks=rejections)
                obs = next_obs
            
            writer.add_scalar("charts/episodic_return", np.mean(returns), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(ep_lens), global_step)

        data = rb.sample(args.batch_size)
        b_obs = data.observations["obs"].to(device)
        b_task = data.observations["task"].to(device)
        b_next_obs = data.next_observations["obs"].to(device)
        b_actions = data.actions.to(device).long()
        b_rewards = data.rewards.to(device)
        b_dones = data.dones.to(device)

        with torch.no_grad():
            next_q_online = q_net(b_next_obs, b_task)
            next_acts = next_q_online.argmax(-1, keepdim=True)
            next_q_tar = q_target(b_next_obs, b_task).gather(-1, next_acts).squeeze(-1)
            target_q = b_rewards.flatten() + (1 - b_dones.flatten()) * args.gamma * next_q_tar

        q_values = q_net(b_obs, b_task).gather(-1, b_actions).squeeze(-1)
        q_loss = F.mse_loss(q_values, target_q)
        
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        with torch.no_grad():
            u_next = uvu_net(b_next_obs, b_task)
            p_next = uvu_prior(b_next_obs, b_task) * args.prior_scale
            u_err_next = u_next - p_next
            q_next = q_net(b_next_obs, b_task).unsqueeze(-1)
            uvu_next_acts = (u_err_next + q_next).argmax(1).unsqueeze(1)
            
            p_t = (uvu_prior(b_obs, b_task) * args.prior_scale).gather(1, b_actions.unsqueeze(-1).expand(-1, -1, args.uvu_outs)).squeeze(1)
            
            f_tp1_tar = uvu_target(b_next_obs, b_task).gather(1, uvu_next_acts.expand(-1, -1, args.uvu_outs)).squeeze(1)
            p_tp1 = (uvu_prior(b_next_obs, b_task) * args.prior_scale).gather(1, uvu_next_acts.expand(-1, -1, args.uvu_outs)).squeeze(1)
            uvu_target_val = args.gamma * (1.0 - b_dones) * (f_tp1_tar - p_tp1)

        f_t = uvu_net(b_obs, b_task).gather(1, b_actions.unsqueeze(-1).expand(-1, -1, args.uvu_outs)).squeeze(1)
        uvu_loss = 0.5 * F.mse_loss(f_t - p_t, uvu_target_val, reduction='sum') / args.batch_size
        
        uvu_optimizer.zero_grad()
        uvu_loss.backward()
        uvu_optimizer.step()

        if global_step % args.target_network_frequency == 0:
            for p, tp in zip(q_net.parameters(), q_target.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            for p, tp in zip(uvu_net.parameters(), uvu_target.parameters()):
                tp.data.copy_((args.tau/10.0) * p.data + (1 - (args.tau/10.0)) * tp.data)

        if global_step % 1000 == 0:
            writer.add_scalar("losses/td_loss", q_loss.item(), global_step)
            writer.add_scalar("losses/uvu_loss", uvu_loss.item(), global_step)
            print(f"Step: {global_step} | Q Loss: {q_loss.item():.4f} | UVU Loss: {uvu_loss.item():.4f}")

    envs.close()
    writer.close()