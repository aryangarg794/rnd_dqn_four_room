from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

import torch
from dqn.archs import DQNBasePolicy
from dqn.callbacks import EvalCallbackCustom

import gymnasium as gym
from four_room.env import FourRoomsEnv
from four_room.constants import train_config
from four_room.wrappers import gym_wrapper_state
from utils.statistics import human_format

import argparse
import matplotlib.pyplot as plt
from copy import deepcopy

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)

from stable_baselines3.common.callbacks import CallbackList


num_train_configs = len(train_config["topologies"])
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dir", type=str, default="dqn", help="save name")
    parser.add_argument(
        "-ag", "--lr", type=float, default=5e-4, help="lr for dqn agent"
    )
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed")
    parser.add_argument("-b", "--batch", type=int, default=256, help="batch_size")
    parser.add_argument(
        "-t", "--timesteps", type=int, default=6_000_000, help="batch_size"
    )
    parser.add_argument(
        "-g", "--gamma", type=float, default=0.99, help="discount for the agent"
    )
    parser.add_argument("--use_cnn", action="store_true", help="use cnn input")
    parser.add_argument("--use_action", action="store_true", help="use cnn input")
    parser.add_argument("--use_dual", action="store_true", help="use cnn input")
    parser.add_argument("--use_norm", action="store_true", help="use norm input")
    parser.add_argument("-i", "--init", type=str, default="kaiming", help="init func")
    parser.add_argument("-m", "--mod", type=str, default="concat", help="init func")

    args = parser.parse_args()

    exp_frac = 1.0
    buffer_size = 500_000
    batch_size = args.batch
    tau = 0.05
    gamma = args.gamma
    max_grad_norm = 1
    gradient_steps = 1
    target_update_interval = 50
    train_freq = 50
    exploration_final_eps = 0.1
    learning_rate = args.lr
    n_envs = 50

    train_env = make_vec_env(
        "MiniGrid-FourRooms-v1",
        n_envs=n_envs,
        seed=0,
        vec_env_cls=DummyVecEnv,
        wrapper_class=gym_wrapper_state,
        env_kwargs={
            "agent_pos": train_config["agent positions"],
            "goal_pos": train_config["goal positions"],
            "doors_pos": train_config["topologies"],
            "agent_dir": train_config["agent directions"],
        },
        wrapper_kwargs={"use_cnn": args.use_cnn},
    )

    eval_env = deepcopy(train_env)

    policy_kwargs = dict()
    policy_kwargs["use_cnn"] = args.use_cnn
    policy_kwargs["use_action"] = args.use_action
    policy_kwargs["use_dual"] = args.use_dual
    policy_kwargs["use_norm"] = args.use_norm
    policy_kwargs["init_func"] = args.init
    policy_kwargs["modulation"] = args.mod

    name_cnn = "_cnn" if args.use_cnn else "_mlp"
    name_norm = "_norm" if args.use_norm else ""
    name_act = "_act" if args.use_action else ""
    name_dual = "_dual" if args.use_dual else ""
    name_mod = f"_{args.mod}"
    name_init = "_" + args.init
    time_name = human_format(args.timesteps)

    group_name = f"{args.dir}{name_cnn}{name_act}{name_dual}{name_norm}{name_init}{name_mod}_{time_name}"
    save_file_name = f"{group_name}_seed_{args.seed}"

    eval_callback = EvalCallbackCustom(
        eval_env,
        save_file_name=save_file_name,
        n_eval_episodes=num_train_configs,
        eval_freq=max(100_000 // n_envs, 1),
        verbose=0,
        log_path=f"logging/",
    )

    callback = CallbackList([eval_callback])

    model = DQN(
        DQNBasePolicy,
        train_env,
        learning_starts=batch_size,
        tensorboard_log="logging/",
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        replay_buffer_class=ReplayBuffer,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=(train_freq // n_envs, "step"),
        gradient_steps=gradient_steps,
        max_grad_norm=max_grad_norm,
        target_update_interval=target_update_interval,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exp_frac,
        seed=args.seed,
        device=device,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    train_env.close()
    eval_env.close()
