import numpy as np
import gymnasium as gym
import dill
import torch
import argparse

from tqdm import tqdm

from buffers.buffers import ReplayBufferBase
from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.constants import (
    train_config,
    val_config,
    test_config,
    size,
    state_to_q_np,
)
from four_room.wrappers import gym_wrapper_state
from four_room.utils import get_state
from utils.exploration import aux_pos_multiple

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)


class PseudoBuffer:

    def __init__(self):
        self.states = []
        self.q_values = []
        self.capacity = 0

    def add(self, x, y):
        self.states.append(torch.tensor(x))
        self.q_values.append(torch.tensor(y))
        self.capacity += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cnn", action="store_true", help="use cnn input")
    args = parser.parse_args()
    name_cnn = "cnn" if args.use_cnn else "mlp"

    print(f"Recording for full_dataset_{name_cnn}")

    env = gym_wrapper_state(
        gym.make(
            "MiniGrid-FourRooms-v1",
            agent_pos=train_config["agent positions"],
            goal_pos=train_config["goal positions"],
            doors_pos=train_config["topologies"],
            agent_dir=train_config["agent directions"],
            size=size,
            max_steps=1500,
        ),
        use_cnn=args.use_cnn,
    )

    # buffer = PseudoBuffer()
    buffer = ReplayBufferBase((3, 19, 19), False, capacity=207200)
    print(buffer.size)

    # for i in range(len(train_config["topologies"])):
    #     pairs_explored = []

    #     obs, _ = env.reset()
    #     done = False
    #     context = env.get_wrapper_attr("context")

    #     for idx in range(len(env.unwrapped.valid_pos)):
    #         env.get_wrapper_attr("move_valid_pos")(idx)
    #         agent_pos = env.get_wrapper_attr("valid_pos")[idx]

    #         for _ in range(4):
    #             obs, _, _, _, _ = env.step(1)
    #             state = get_state(obs, use_cnn=args.use_cnn)
    #             q = find_all_action_values(
    #                 state[:2], state[2], state[3:5], state[5:], 0.99, size
    #             )
    #             buffer.add(obs, q)

    #         print(f"Context is {context+1} | {state[:3]}", end="\r", flush=True)

    # buffer.X = torch.stack(buffer.X)
    # buffer.Y = torch.stack(buffer.Y)

    # with open(f'action_values/full_dataset_{name_cnn}.pl', 'wb') as file:
    #     dill.dump(buffer, file)
    #     file.close()

    # while buffer.capacity < 207200:
    #     for i in range(len(train_config["topologies"])):
    #         pairs_explored = []

    #         obs, _ = env.reset()
    #         done = False
    #         context = env.get_wrapper_attr("context")

    #         while not done:
    #             state = get_state(obs, use_cnn=args.use_cnn)
    #             q = state_to_q_np[context, *state[:3]]
    #             action = q.argmax()
    #             obs_next, reward, trunc, term, _ = env.step(action)
    #             buffer.add(obs, q)
    #             done = trunc or term
    #             obs = obs_next

    #             print(f"Context is {context+1} | {state[:3]} | Size: {buffer.capacity}", end="\r", flush=True)

    while buffer.size < 207200:
        for i in range(len(train_config["topologies"])):

            obs, _ = env.reset()
            done = False
            context = env.get_wrapper_attr("context")
            state = get_state(obs, use_cnn=args.use_cnn)
            actions, path = aux_pos_multiple(state, env)

            aux_pos = (path[-1][0], path[-1][1])

            k = np.random.randint(low=0, high=len(path))
            rand_state = path[k]
            env.get_wrapper_attr("move_state")(rand_state)
            obs, _, _, _, _ = env.step(1)

            while not done:
                state = get_state(obs, use_cnn=args.use_cnn)
                q = state_to_q_np[context, *state[:3]]
                action = q.argmax()
                obs_next, reward, trunc, term, _ = env.step(action)
                buffer.update(obs, action, 0.0, obs_next, 0, trunc or term, q_value=q)
                done = trunc or term
                obs = obs_next

                print(
                    f"Context is {context+1} | {state[:3]} | Size: {buffer.size}",
                    end="\r",
                    flush=True,
                )

    with open(f"action_values/full_dataset_{name_cnn}_expl.pl", "wb") as file:
        dill.dump(buffer, file)
        file.close()
