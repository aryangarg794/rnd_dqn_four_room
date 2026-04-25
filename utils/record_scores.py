import gymnasium as gym
import numpy as np
import torch
import imageio
import cv2

from dqn.model import DQN
from uvu.uvu import UVU
from rnd_exploration.rnd import RNDNetwork
from rnd_exploration.dataset import State
from four_room.wrappers import gym_wrapper, gym_wrapper_state
from four_room.utils import obs_to_state
from four_room.constants import train_config, size, state_to_q, state_to_q_np
from utils.q_values import compute_q_value
from scripts.run_uvu import get_state


@torch.no_grad()
def get_rnd_scores(
    net: RNDNetwork, current_env: int, env_range: int = 5, device: str = "cuda"
):
    env_ids = list(range(current_env - env_range, current_env + env_range + 1))
    env = gym_wrapper(
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
        original_obs=True,
    )

    net.rnd_net.eval()
    results = np.zeros(
        (len(env_ids), env.get_wrapper_attr("width"), env.get_wrapper_attr("height")),
        dtype=np.float32,
    )

    for idx, env_id in enumerate(env_ids):
        env.get_wrapper_attr("set_context")(env_id)
        obs, _ = env.reset()
        valid_pos = env.get_wrapper_attr("valid_pos")

        for i, valid_state in enumerate(valid_pos):
            env.get_wrapper_attr("move_valid_pos")(i)

            obs, _, _, _, _ = env.step(1)
            rnd_val = net.get_error(obs).item()
            results[idx, *valid_state] = rnd_val

    net.rnd_net.train()
    return results, env_ids


def get_q_optimal(counter, current_env: int, env_range: int = 5, gamma: float = 0.99):
    env_ids = list(range(current_env - env_range, current_env + env_range + 1))
    env = gym_wrapper(
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
        original_obs=True,
    )

    results = np.zeros(
        (
            4,
            len(env_ids),
            env.get_wrapper_attr("width"),
            env.get_wrapper_attr("height"),
        ),
        dtype=np.float32,
    )
    start_state, _, _ = env.get_wrapper_attr("context_info")(current_env)
    for idx, env_id in enumerate(env_ids):
        env.get_wrapper_attr("set_context")(env_id)
        obs, _ = env.reset()
        valid_pos = env.get_wrapper_attr("valid_pos") + [start_state]

        for i, valid_state in enumerate(valid_pos):
            env.get_wrapper_attr("move_state")(valid_state)

            for _ in range(
                4
            ):  # for each direction we want to store the state-q value pair
                obs, _, _, _, _ = env.step(1)
                state = obs_to_state(obs)
                agent_dir = state[2]

                dqn_val = compute_q_value(obs, env_id, counter, gamma)
                results[agent_dir, idx, *valid_state] = dqn_val

    return results.max(axis=0), env_ids, results


def record_uncertainty_scores(counter, current_env: int, env_range: int = 5):
    env_ids = list(range(current_env - env_range, current_env + env_range + 1))
    env = gym_wrapper(
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
        original_obs=True,
    )

    results = np.zeros(
        (
            4,
            len(env_ids),
            env.get_wrapper_attr("width"),
            env.get_wrapper_attr("height"),
        ),
        dtype=np.float32,
    )
    start_state, _, _ = env.get_wrapper_attr("context_info")(current_env)
    for idx, env_id in enumerate(env_ids):
        env.get_wrapper_attr("set_context")(env_id)
        obs, _ = env.reset()
        valid_pos = env.get_wrapper_attr("valid_pos") + [start_state]

        for i, valid_state in enumerate(valid_pos):
            env.get_wrapper_attr("move_state")(valid_state)

            for _ in range(4):
                obs, _, _, _, _ = env.step(1)
                state = obs_to_state(obs)
                agent_dir = state[2]

                un_val = counter[env_id, *valid_state, agent_dir]
                results[agent_dir, idx, *valid_state] = un_val

    return results.min(axis=0), env_ids, results


@torch.no_grad()
def record_dqn_scores(
    agent: DQN,
    current_env: int,
    env_range: int = 5,
    device: str = "cuda",
    use_cnn: bool = False,
):
    env_ids = list(range(current_env - env_range, current_env + env_range + 1))
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
        use_cnn=use_cnn,
    )

    agent.net.eval()
    results = np.zeros(
        (
            4,
            len(env_ids),
            env.get_wrapper_attr("width"),
            env.get_wrapper_attr("height"),
        ),
        dtype=np.float32,
    )
    start_state, _, _ = env.get_wrapper_attr("context_info")(current_env)
    for idx, env_id in enumerate(env_ids):
        env.get_wrapper_attr("set_context")(env_id)
        obs, _ = env.reset()
        valid_pos = env.get_wrapper_attr("valid_pos") + [start_state]

        for i, valid_state in enumerate(valid_pos):
            env.get_wrapper_attr("move_state")(valid_state)

            for _ in range(
                4
            ):  # for each direction we want to store the state-q value pair
                obs, _, _, _, _ = env.step(1)
                state = get_state(obs, use_cnn)
                agent_dir = state[2]

                obs_torch = agent.get_obs(obs)
                goal_action = state_to_q_np[env_id, *state[:3]].argmax()
                dqn_val = agent(obs_torch).squeeze()[goal_action].item()
                results[agent_dir, idx, *valid_state] = dqn_val

    agent.net.train()
    return results.max(axis=0), env_ids, results


@torch.no_grad()
def record_uvu_scores(
    agent: UVU,
    current_env: int,
    env_range: int = 5,
    device: str = "cuda",
    render: bool = False,
    use_cnn: bool = False,
):
    env_ids = list(range(current_env - env_range, current_env + env_range + 1))
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
        use_cnn=use_cnn,
    )

    agent.net.eval()
    results = np.zeros(
        (
            4,
            len(env_ids),
            env.get_wrapper_attr("width"),
            env.get_wrapper_attr("height"),
        ),
        dtype=np.float32,
    )
    start_state, _, _ = env.get_wrapper_attr("context_info")(current_env)
    imgs = []
    for idx, env_id in enumerate(env_ids):
        env.get_wrapper_attr("set_context")(env_id)
        obs, _ = env.reset()
        valid_pos = env.get_wrapper_attr("valid_pos") + [start_state]

        for i, valid_state in enumerate(valid_pos):
            env.get_wrapper_attr("move_state")(valid_state)

            for _ in range(
                4
            ):  # for each direction we want to store the state-q value pair
                obs, _, _, _, _ = env.step(1)
                state = get_state(obs, use_cnn)
                agent_dir = state[2]

                obs_torch = agent.get_obs(obs)
                goal_action = state_to_q_np[env_id, *state[:3]].argmax()
                goal_action = (
                    torch.from_numpy(np.array([goal_action]))
                    .to(device=device)
                    .unsqueeze(dim=0)
                )
                dqn_val = agent.epistemic(obs_torch, goal_action)
                results[agent_dir, idx, *valid_state] = dqn_val
                if render:
                    imgs.append(cv2.transpose(env.unwrapped.render()))

    agent.net.train()
    if render:
        imageio.mimsave(
            f"renders/rendered_uvu_test.gif",
            [np.array(img) for i, img in enumerate(imgs) if i % 1 == 0],
            duration=150,
        )
    return results.max(axis=0), env_ids, results
