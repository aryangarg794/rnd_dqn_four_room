import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import imageio
import os
import dill

from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque
from tabulate import tabulate

from dqn.counter import MovingCountBasedUncertainty, CountBasedUncertainty
from four_room.env import FourRoomsEnv
from four_room.utils import get_state
from four_room.wrappers import gym_wrapper_state
from four_room.constants import state_to_q_np
from rnd_exploration.utils import RunningAverage
from four_room.constants import train_config, val_config, test_config, size
from rnd_exploration.dataset import State
from dqn_experiments.regression_exp_utils import run_experiment
from uvu.uvu import UVU
from utils.exploration import aux_pos_multiple

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)
torch.set_num_interop_threads(1)
torch.set_num_threads(1)


@dataclass
class Args:
    env: gym.Env
    val_env: gym.Env
    dir: str = "test"
    seed: int = 0
    lr_agent: float = 5e-4
    use_cnn: bool = False
    use_dual: bool = False
    use_norm: bool = False
    use_action: bool = False
    mod: str = "one_hot"
    capacity: int = int(1e5)
    init: str = "kaiming"
    tau: float = 0.005
    device: str = "cuda"
    gamma: float = 0.99
    grad_norm: float = 10.0
    num_heads: int = 512


def train_uvu_count(
    args: Args,
    batch_size: int = 512,
    num_timesteps: int = int(2e5),
    regression_freq: int = 50000,
    seed: int = 0,
    alpha: float = 1.5,
    window: int = 2500,
    warmupsteps: int = 3500,
    gradient_steps: int = 5,
    render: bool = False,
    debug: bool = False,
    eps_mode: float = 0.05,
    eps_dqn: float = 0.05,
    gamma: float = 0.99,
):
    rms_uvu = RunningAverage(window_size=window)
    rms_norms = RunningAverage(window_size=window)

    os.makedirs("results/dqn_exps", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    imgs = deque(maxlen=2500)
    learning_curves = []
    scores = []
    uniqueness = []

    torch.backends.cudnn.deterministic = True

    at_end = True if regression_freq == -1 else False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    agent = UVU(
        env=args.env,
        val_env=args.val_env,
        capacity=args.capacity,
        tau=args.tau,
        lr=args.lr_agent,
        device=args.device,
        hidden_layers=[512, 512, 512],
        hidden_layers_g=[512, 512],
        use_cnn=args.use_cnn,
        use_action=args.use_action,
        use_dual=args.use_dual,
        use_norm=args.use_norm,
        init_func=args.init,
        modulation=args.mod, 
        num_heads=args.num_heads,
        gamma=gamma,
    )

    env = deepcopy(args.env)
    items_added = 0

    obs, _ = env.reset()
    record = False
    state = get_state(obs, args.use_cnn)
    goal_pos = state[3:5]
    target_pos = state[3:5]  # first phase is warmup
    aux_pos = None

    mode = False
    if np.random.random() < eps_mode or warmupsteps > 0:
        mode = True  # true = explorego, false = our heuristic version

    if warmupsteps > 0:
        max_k = len(env.get_wrapper_attr("valid_pos"))
        k = np.random.randint(low=0, high=max_k)
        env.get_wrapper_attr("move_valid_pos")(k)

    actions, path = aux_pos_multiple(state, env)
    aux_pos = (path[-1][0], path[-1][1])
    if mode:
        k = np.random.randint(low=0, high=len(path))
        rand_state = path[k]
        env.get_wrapper_attr("move_state")(rand_state)
        target_pos = goal_pos
        record = True

    ep_highlight_mask = np.zeros(
        (
            len(train_config["agent positions"]),
            env.get_wrapper_attr("width"),
            env.get_wrapper_attr("height"),
        ),
        dtype=bool,
    )
    heatmap_swap = np.zeros(
        (
            len(train_config["agent positions"]),
            env.get_wrapper_attr("width"),
            env.get_wrapper_attr("height"),
        ),
        dtype=int,
    )

    aux_heatmap = np.zeros_like(heatmap_swap)
    explore_heatmap = np.zeros_like(heatmap_swap)
    switch_state_history = []

    ep_colors = np.empty_like(ep_highlight_mask, dtype=object)

    current_context = env.get_wrapper_attr("context")
    start_state, _, _ = env.get_wrapper_attr("context_info")(current_context)
    past_pos = []
    visit_history = deque(maxlen=args.capacity + 1)
    placeholder = np.array([0.0, 0.0, 0.0])

    switches = 0
    trajs_added = 0
    contexts = []

    counter_moving = MovingCountBasedUncertainty(
        capacity=args.capacity, device=args.device
    )
    counter_full = CountBasedUncertainty(capacity=args.capacity)

    for step in (pbar := tqdm(range(1, num_timesteps + 1), disable=debug)):

        obs_torch = agent.get_obs(obs)
        state = get_state(obs, args.use_cnn)
        contexts.append(current_context)
        agent_pos = env.get_wrapper_attr("agent_pos")

        if len(actions) != 0 and not record and step >= warmupsteps:
            action = actions.pop(0)
        elif np.random.random() < eps_dqn:
            action = np.random.randint(low=0, high=3)
        else:
            q = state_to_q_np[current_context, state[0], state[1], state[2]]
            action = q.argmax() if isinstance(q, np.ndarray) else np.array(q).argmax()

        with torch.no_grad():
            goal_action = state_to_q_np[
                current_context, state[0], state[1], state[2]
            ].argmax()
            goal_action = torch.tensor([goal_action], device=args.device).view(1, 1)
            uvu_val = agent.epistemic(obs_torch, goal_action).item()
            obj_tuple = tuple([int(item) for item in state])
            obj_tuple = (*obj_tuple, current_context)
            obj_moving_tuple = (current_context, *agent_pos, state[2])
            q = state_to_q_np[current_context, state[0], state[1], state[2]]

        norm = (uvu_val - rms_uvu.avg) / rms_uvu.std

        if (
            uvu_val - rms_uvu.avg >= alpha * rms_uvu.std
            and not record
            and step >= warmupsteps
        ):  # swap to record mode
            switches += 1
            heatmap_swap[current_context, agent_pos[0], agent_pos[1]] += 1
            record = True
            target_pos = goal_pos
            switch_state_history.append((step, current_context, *agent_pos))
            action = goal_action

        obs_prime, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if not record:
            explore_heatmap[current_context, agent_pos[0], agent_pos[1]] += 1

        # if counter_moving.counts[*obj_moving_tuple] > 0:
        rms_uvu.update(uvu_val)
        rms_norms.update(norm)

        if step < warmupsteps or record:
            assert np.array_equal(target_pos, goal_pos)
            # print(f'Timestep: {step} | Context: {current_context} | State: {agent_pos} | Dir: {state[2]} | Switch Count: {heatmap_swap[current_context, *agent_pos]} | Uncert: {uncertainty:.4f} | Count: {counter_moving.counts[*obj_moving_tuple]}')
            state_prime = get_state(obs_prime, args.use_cnn)
            q_next = (
                state_to_q_np[
                    current_context, state_prime[0], state_prime[1], state_prime[2]
                ]
                if not done
                else placeholder
            )
            next_action = q_next.argmax()
            agent.buffer.update(
                obs, action, reward, obs_prime, next_action, int(done), q_value=q
            )
            if render:
                ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                past_pos.append(agent_pos)
                visit_history.append((current_context, *agent_pos))

                if agent.buffer.size >= agent.buffer.capacity:
                    to_remove = visit_history[0]
                    ep_highlight_mask[to_remove[0], to_remove[1], to_remove[2]] = False
                    ep_colors[to_remove[0], to_remove[1], to_remove[2]] = None

            counter_moving.add(obj_moving_tuple, step)
            counter_full.add(obj_tuple)
            agent.buffer.update_seen(obj_moving_tuple)
            items_added += 1
        else:
            state_prime = get_state(obs_prime, args.use_cnn)
            q_next = (
                state_to_q_np[
                    current_context, state_prime[0], state_prime[1], state_prime[2]
                ]
                if not done
                else placeholder
            )
            next_action = q_next.argmax()

        if render and step >= num_timesteps - 1000:
            env.get_wrapper_attr("set_aux")(aux_pos) if aux_pos else None
            agent_col = (
                (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255)
            )

            imgs.append(
                env.unwrapped.render(
                    highlight_mask=ep_highlight_mask[current_context],
                    colors=ep_colors[current_context],
                    agent_col=agent_col,
                )
            )
            env.get_wrapper_attr("remove_aux")(aux_pos) if aux_pos else None

        obs = obs_prime

        if agent.buffer.size >= batch_size:
            for _ in range(gradient_steps):
                agent.update_step(batch_size)

        if done:
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)

            past_pos = []

            obs, _ = env.reset()
            done = False
            state = get_state(obs, args.use_cnn)
            goal_pos = state[3:5]

            mode = False
            record = False
            if np.random.random() < eps_mode or step < warmupsteps:
                mode = True  # true = explorego, false = our heuristic version

            actions, path = aux_pos_multiple(state, env)
            aux_pos = (path[-1][0], path[-1][1])
            target_pos = aux_pos
            if mode:
                k = np.random.randint(low=0, high=len(path))
                rand_state = path[k]
                env.get_wrapper_attr("move_state")(rand_state)
                target_pos = goal_pos
                record = True

            current_context = env.get_wrapper_attr("context")
            start_state, _, _ = env.get_wrapper_attr("context_info")(current_context)
            trajs_added += 1

            if step < warmupsteps:
                max_k = len(env.get_wrapper_attr("valid_pos"))
                k = np.random.randint(low=0, high=max_k)
                target_pos = goal_pos  # goal state
                env.get_wrapper_attr("move_valid_pos")(k)

        agent.soft_update()

        if at_end:
            if step == num_timesteps:
                lc, test_score = run_experiment(agent.buffer, use_cnn=args.use_cnn, device=args.device, disable=True)
                learning_curves.append(lc)
                scores.append(test_score)
        else:
            if step % regression_freq == 0 and agent.buffer.size >= agent.buffer.capacity:
                lc, test_score = run_experiment(agent.buffer, use_cnn=args.use_cnn, device=args.device, disable=True)
                learning_curves.append(lc)
                scores.append(test_score)

                results = {
                    "lc_curves": learning_curves,
                    "reg_test_scores": scores,
                    "uniqueness": uniqueness,
                    "images": imgs,
                    "heatmap": heatmap_swap,
                    "counter_full": counter_full,
                    "counter_moving": counter_moving,
                    "aux_heatmap": aux_heatmap,
                    "explore_heatmap": explore_heatmap,
                    "switch_states": switch_state_history,
                    "context_history": contexts,
                }
                with open(
                    f"results/dqn_exps/{args.dir}_seed_{args.seed}_intermediate.pl", "wb"
                ) as file:
                    dill.dump(results, file)

        uniqueness.append(agent.buffer.ratio_unique_trans)
        value = (uvu_val - rms_uvu.avg) / rms_uvu.std
        # pbar.set_description(f"Training RND DQN | Uniqueness: {agent.buffer.ratio_unique_trans:.4f} | Last Regression Exp: {(scores[-1] if len(scores) > 0 else 0):.4f} | Total Items added: {items_added} | Current Context: {current_context} | RND Val: {dqn_val:.4f} | Avg: {rms_dqn.avg:.4f} | STD: {rms_dqn.std:.4f} | Switches: {switches} | Value: {value:.4f}")
        reg_exp = scores[-1] if len(scores) > 0 else 0
        pbar.set_description(f"Items added: {items_added} | Context: {current_context}")
        pbar.set_postfix(
            unq=agent.buffer.ratio_unique_trans,
            norm_avg=rms_norms.avg,
            reg=reg_exp,
            switches=switches,
            uvu_avg=rms_uvu.avg,
        )

    return {
        "lc_curves": learning_curves,
        "reg_test_scores": scores,
        "buffer": agent.buffer,
        "uniqueness": uniqueness,
        "images": imgs,
        "heatmap": heatmap_swap,
        "aux_heatmap": aux_heatmap,
        "counter_full": counter_full,
        "counter_moving": counter_moving,
        "explore_heatmap": explore_heatmap,
        "switch_states": switch_state_history,
        "context_history": contexts,
        "running_mean": rms_uvu,
        "running_mean_norms": rms_norms,
    }, agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--timesteps", type=int, default=int(3e5), help="timesteps"
    )
    parser.add_argument("-f", "--dir", type=str, default="uvu", help="save name")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument(
        "-ag", "--lr_agent", type=float, default=1e-4, help="lr for dqn agent"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device")
    parser.add_argument("-r", "--render", action="store_true", help="render mode")
    parser.add_argument(
        "-s", "--replaysize", type=int, default=int(1e5), help="size of replay buffer"
    )
    parser.add_argument("-seed", "--seed", type=int, default=0, help="seed")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "-fr", "--freq", type=int, default=-1, help="freq of regression"
    )
    parser.add_argument(
        "--window", type=int, default=2500, help="window size of rms_dqn"
    )
    parser.add_argument("-tau", "--tau", type=float, default=0.5, help="tau")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("-ed", "--eps_dqn", type=float, default=0.05, help="eps dqn")
    parser.add_argument("-em", "--eps_mode", type=float, default=0.05, help="eps dqn")
    parser.add_argument("--grad_steps", type=int, default=1, help="num of grad steps")
    parser.add_argument("--use_cnn", action="store_true", help="use cnn input")
    parser.add_argument("--use_action", action="store_true", help="use cnn input")
    parser.add_argument("--use_dual", action="store_true", help="use cnn input")
    parser.add_argument("--use_norm", action="store_true", help="use norm input")
    parser.add_argument("-m", "--mod", type=str, default="concat", help="init func")
    parser.add_argument(
        "-i", "--init", type=str, default="kaiming", help="init function"
    )

    args = parser.parse_args()

    env = gym_wrapper_state(
        gym.make(
            "MiniGrid-FourRooms-v1",
            agent_pos=train_config["agent positions"],
            goal_pos=train_config["goal positions"],
            doors_pos=train_config["topologies"],
            agent_dir=train_config["agent directions"],
            size=size,
            render_mode="rgb_array",
            disable_env_checker=True,
        ),
        use_cnn=args.use_cnn,
    )

    val_env = gym_wrapper_state(
        gym.make(
            "MiniGrid-FourRooms-v1",
            agent_pos=val_config["agent positions"],
            goal_pos=val_config["goal positions"],
            doors_pos=val_config["topologies"],
            agent_dir=val_config["agent directions"],
            size=size,
        ),
        use_cnn=args.use_cnn,
    )

    test_env = gym_wrapper_state(
        gym.make(
            "MiniGrid-FourRooms-v1",
            agent_pos=test_config["agent positions"],
            goal_pos=test_config["goal positions"],
            doors_pos=test_config["topologies"],
            agent_dir=test_config["agent directions"],
            size=size,
        ),
        use_cnn=args.use_cnn,
    )

    name_cnn = "_cnn" if args.use_cnn else "_mlp"
    name_norm = "_norm" if args.use_norm else ""
    name_act = "_act" if args.use_action else ""
    name_dual = "_dual" if args.use_dual else ""
    name_init = "_" + args.init
    name_alpha = "_alpha" + str(args.alpha).replace(".", "")
    name_grad = "_grad" + str(args.grad_steps)
    name_mod = f"_{args.mod}"
    name_lr = "_lr" + str(args.lr_agent).replace(".", "")
    name_bs = "_bs" + str(args.batch_size)
    name_tau = "_tau" + str(args.tau).replace(".", "")
    name_eps = "_eps" + str(args.eps_dqn).replace(".", "")

    group_name = (
        f"{args.dir}{name_alpha}{name_lr}{name_bs}{name_tau}{name_eps}"
    )
    save_file_name = f"{group_name}_seed_{args.seed}"

    aux_args = Args(
        env=env,
        dir=save_file_name,
        seed=args.seed,
        val_env=val_env,
        lr_agent=args.lr_agent,
        device=args.device,
        capacity=args.replaysize,
        tau=args.tau,
        use_cnn=args.use_cnn,
        use_action=args.use_action,
        use_dual=args.use_dual,
        use_norm=args.use_norm,
        init=args.init,
        mod=args.mod
    )

    results, agent = train_uvu_count(
        args=aux_args,
        batch_size=args.batch_size,
        num_timesteps=args.timesteps,
        seed=args.seed,
        alpha=args.alpha,
        regression_freq=args.freq,
        render=args.render,
        debug=args.debug,
        window=args.window,
        eps_dqn=args.eps_dqn,
        eps_mode=args.eps_mode,
        gradient_steps=args.grad_steps,
        gamma=args.gamma,
    )

    agent.save(f"{save_file_name}_{args.timesteps}")

    with open(f"results/dqn_exps/{save_file_name}_{args.timesteps}.pl", "wb") as file:
        dill.dump(results, file)
        file.close()

    data = [
        [results['uniqueness'][-1], np.mean(results['reg_test_scores']), agent.buffer.size]
    ]
    headers = ["Uniqueness", "Test Score", "Buffer Size"]
    print(tabulate(data, headers=headers, tablefmt="grid"))

    if args.render:
        imgs = list(results["images"])
        imageio.mimsave(
            f"renders/rendered_{save_file_name}.gif",
            [np.array(img) for i, img in enumerate(imgs[-500:]) if i % 1 == 0],
            duration=150,
        )
