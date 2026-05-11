import numpy as np
import gymnasium as gym
import dill
import torch
import argparse

from rnd_exploration.dataset import ReplayBuffer
from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.constants import train_config, val_config, test_config, size
from four_room.wrappers import gym_wrapper_state
from four_room.utils import get_state 

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)

class PseudoBuffer:
    
    def __init__(self):
        self.X = []
        self.Y = []
        self.size = 0 
    
    def add(self, x, y):
        self.X.append(torch.tensor(x))
        self.Y.append(torch.tensor(y))
        self.size += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cnn", action="store_true", help="use cnn input")
    args = parser.parse_args()
    name_cnn = "cnn" if args.use_cnn else "mlp"

    print(f'Recording for full_dataset_{name_cnn}')

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

    buffer = PseudoBuffer()

    for i in range(len(train_config["topologies"])):
        pairs_explored = []

        obs, _ = env.reset()
        done = False
        context = env.get_wrapper_attr("context")

        for idx in range(len(env.unwrapped.valid_pos)):
            env.get_wrapper_attr("move_valid_pos")(idx)
            agent_pos = env.get_wrapper_attr("valid_pos")[idx]

            for _ in range(4):
                obs, _, _, _, _ = env.step(1)
                state = get_state(obs, use_cnn=args.use_cnn)
                q = find_all_action_values(
                    state[:2], state[2], state[3:5], state[5:], 0.99, size
                )
                buffer.add(obs, q)

            print(f"Context is {context+1} | {state[:3]}", end="\r", flush=True)

    buffer.X = torch.stack(buffer.X)
    buffer.Y = torch.stack(buffer.Y)

    with open(f'action_values/full_dataset_{name_cnn}.pl', 'wb') as file:
        dill.dump(buffer, file)
        file.close()
