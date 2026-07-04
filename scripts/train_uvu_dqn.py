from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.uncertainties import CountSAUncertainty, EpisodicCountSAUncertainty
from udqn.uvu_dqn import ExploreGoUVU
from udqn.policies import UVUGoPolicy
from buffers.buffers import UvuGoReplayBuffer
from utils.callbacks import UniquenesseCallback, BufferCoverageCallback, PolicyOptimalityCallback, ExplorationCoverageCallback
from utils.statistics import human_format

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import uuid
import rich

from four_room.arch import *

from four_room.wrappers import gym_wrapper

import gymnasium as gym
from four_room.env import FourRoomsEnv
from four_room.constants import train_config, val_config

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=0,
    help="Provide the seeds for the agents to be trained",
)
parser.add_argument("--dir", type=str, default='uvugo')
parser.add_argument("--timesteps", type=int, default=8_000_000)
parser.add_argument("--exp_frac", type=float, default=0.125)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--uvu_gradient_steps", type=int, default=1)
parser.add_argument("--window_size", type=int, default=1500)
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--u_tau", type=float, default=0.005)
parser.add_argument("--uvu_tau", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--u_lr", type=float, default=0.001)
parser.add_argument("--uvu_lr", type=float, default=1e-4)
parser.add_argument("--go_vs_uvu", type=float, default=0.2)
parser.add_argument("--arch_size", type=str, default="large")
parser.add_argument("--max_pure_expl_steps", type=int, default=50)
parser.add_argument("--e_greedy", action="store_true")
parser.add_argument('--online', action='store_true')
parser.add_argument("--num_training_levels", type=int, default=200)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--alpha", type=float, default=0.75)

args = parser.parse_args()

LOGS_DIR = "logs/"


num_train_configs = len(train_config["topologies"])

config = {
    "exp_frac": args.exp_frac,
    "exploration_final_eps": 0.1,
    "learning_starts": 256,
    "buffer_size": 500_000,
    "batch_size": 256,
    "tau": args.tau,
    "u_tau": args.u_tau,
    "uvu_tau": args.uvu_tau,
    "gamma": 0.99,
    "gradient_steps": args.gradient_steps,
    "uvu_gradient_steps": args.uvu_gradient_steps,
    "target_update_interval": 50,
    "train_freq": 50,
    "learning_rate": args.lr,
    "u_learning_rate": args.u_lr,
    "uvu_learning_rate": args.uvu_lr,
    "go_vs_uvu": args.go_vs_uvu,
    "window_size": args.window_size,
    "n_envs": 50,
    "max_grad_norm": 1,
    "device": "cuda" if th.cuda.is_available() else "cpu",
    "max_pure_expl_steps": args.max_pure_expl_steps,
    "include_pure_experience": False,
    "double_q": False,
    "num_training_levels": args.num_training_levels,
    "split_uncertainty": True,
    "beta": args.beta,
    "alpha": args.alpha,
    "initialisation": "orthogonal",
    "dont_bootstrap_terminal": True,
}

for seed in args.seeds:
    config["seed"] = seed
    ipe = config["include_pure_experience"]
    mpes = config["max_pure_expl_steps"]
    eval_env = make_vec_env(
        "MiniGrid-FourRooms-v1",
        n_envs=1,
        seed=config["seed"],
        vec_env_cls=DummyVecEnv,
        wrapper_class=gym_wrapper,
        env_kwargs={
            "agent_pos": val_config["agent positions"],
            "goal_pos": val_config["goal positions"],
            "doors_pos": val_config["topologies"],
            "agent_dir": val_config["agent directions"],
            "size": 19,
            "max_steps": 100,
        },
        wrapper_kwargs={"original_obs": True},
    )

    train_env = make_vec_env(
        "MiniGrid-FourRooms-v1",
        n_envs=config["n_envs"],
        seed=config["seed"],
        vec_env_cls=DummyVecEnv,
        wrapper_class=gym_wrapper,
        env_kwargs={
            "agent_pos": train_config["agent positions"],
            "goal_pos": train_config["goal positions"],
            "doors_pos": train_config["topologies"],
            "agent_dir": train_config["agent directions"],
            "size": 19,
            "max_steps": 100,
        },
        wrapper_kwargs={"original_obs": True},
    )

    if not args.e_greedy:
        global_uncertainty = CountSAUncertainty(
            (8 * 8 * 4 + 3) * 4 * args.num_training_levels * 3, # roomx x roomy x nrooms + doors - goal x dirs x configs x actions
            train_env.observation_space.shape,
            device=config["device"],
        )
        uncertainty = EpisodicCountSAUncertainty(
            config["n_envs"],
            100,
            train_env.observation_space.shape,
            device=config["device"],
            global_uncertainty=global_uncertainty,
        )
        replay_buffer_kwargs = dict(
            uncertainty=uncertainty,
            state_action_bonus=True,
            uncertainty_of_sampling=False,
            episodic_discount=True,
            split_uncertainty=True,
            include_pure_experience=False,
        )
        config["replay_buffer_kwargs"] = replay_buffer_kwargs
    else:
        uncertainty = "egreedy"
        replay_buffer_kwargs = dict(
            uncertainty=uncertainty,
            state_action_bonus=True,
            uncertainty_of_sampling=False,
            episodic_discount=False,
            split_uncertainty=False,
        )
        config["replay_buffer_kwargs"] = replay_buffer_kwargs

    if args.arch_size == "small":
        net_arch = []
    elif args.arch_size == "large":
        net_arch = [256]
    policy_kwargs = dict(
        features_extractor_class=CNN,
        features_extractor_kwargs={
            "features_dim": 512,
            "init_function": "kaiming",
        },
        normalize_images=False,
        net_arch=net_arch,
        beta=config["beta"],
        uvu_lr=config["uvu_learning_rate"],
        uvu_kwargs={
            "feature_dims" : 512,
            "net_arch" : [512, 512, 512],
            "init": "kaiming",
            "norm" : True,
            "num_heads" : 512
        },
        g_kwargs={
            "feature_dims" : 512,
            "net_arch" : [512, 512],
            "init": "kaiming",
            "norm" : False,
            "num_heads" : 512
        },
        u_lr=config["u_learning_rate"],
        n_envs=config["n_envs"],
    )
    config["policy_kwargs"] = policy_kwargs

    callback = EvalCallback(
        eval_env,
        n_eval_episodes=len(val_config["topologies"]),
        eval_freq=max(100_000 // config["n_envs"], 1),
        verbose=0,
    )
    callback_list = [callback]

    save_path = (
        LOGS_DIR
        + f"UVU/{args.beta}_{args.max_pure_expl_steps}_{args.alpha}_{config['seed']}/"
    )
        
    checkpoint_callback = CheckpointCallback(
        save_freq=max(4_000_000 // config["n_envs"], 1),
        save_path=save_path,
        name_prefix="uvu",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callback_list.append(checkpoint_callback)

    unq_callback = UniquenesseCallback(log_freq=5_000)
    callback_list.append(unq_callback)    

    policy_callback = PolicyOptimalityCallback(100_000, config['num_training_levels'], device=config['device'])
    callback_list.append(policy_callback)

    exp_callback = ExplorationCoverageCallback(log_freq=100_000, total_states=(8*8*4+3)*4*args.num_training_levels, num_actions=3)
    callback_list.append(exp_callback)

    buffer_callback = BufferCoverageCallback(freq=100_000, total_states=(8*8*4+3)*4*args.num_training_levels, num_actions=3)
    callback_list.append(buffer_callback)


    # Delete the following lines if you don't want to use wandb for logging results
    import wandb
    from wandb.integration.sb3 import WandbCallback

    wandb_mode = 'online' if args.online else 'offline'
    time_name = human_format(args.timesteps)
    run_name = f"{args.dir}_{time_name}"
    with wandb.init(
        project="ExploreGo",
        name=f"seed_{seed}",
        group=run_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        tags=["FourRooms", "UVU"],
        config=config,
        mode=wandb_mode
    ):
        wandb_callback = WandbCallback()

        model = ExploreGoUVU(
            UVUGoPolicy,
            train_env,
            config["beta"],
            alpha=config["alpha"],
            uncertainty=uncertainty,
            learning_starts=config["learning_starts"],
            tensorboard_log=LOGS_DIR + "logging/",
            policy_kwargs=policy_kwargs,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            num_heads=512,
            u_tau=config["u_tau"],
            go_vs_uvu=config["go_vs_uvu"],
            uvu_tau=config["uvu_tau"],
            gamma=config["gamma"],
            train_freq=(config["train_freq"] // config["n_envs"], "step"),
            gradient_steps=config["gradient_steps"],
            uvu_gradient_steps=config["uvu_gradient_steps"],
            target_update_interval=config["target_update_interval"],
            exploration_final_eps=config["exploration_final_eps"],
            exploration_fraction=config["exp_frac"],
            max_grad_norm=config["max_grad_norm"],
            seed=config["seed"],
            device=config["device"],
            max_pure_expl_steps=config["max_pure_expl_steps"],
            replay_buffer_class=UvuGoReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            double_q=config["double_q"],
        )

        run_id = f"run_{uuid.uuid4()}_{config['seed']}"
        rich.print(config)
        model.learn(
            total_timesteps=args.timesteps, callback=callback_list, tb_log_name=run_id, progress_bar=True
        )
        train_env.close()
        eval_env.close()
