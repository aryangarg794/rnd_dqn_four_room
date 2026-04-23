import numpy as np
import gymnasium as gym
import dill

from rnd_exploration.dataset import State
from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.constants import train_config, val_config, test_config, size
from four_room.wrappers import gym_wrapper

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)
env = gym_wrapper(
    gym.make(
        "MiniGrid-FourRooms-v1",
        agent_pos=train_config["agent positions"],
        goal_pos=train_config["goal positions"],
        doors_pos=train_config["topologies"],
        agent_dir=train_config["agent directions"],
        size=size,
        max_steps=1500,
    ),
    original_obs=True,
)

if __name__ == "__main__":
    # state_to_q = {}

    # for i in range(len(train_config['topologies'])):
    #     pairs_explored = []

    #     obs, _ = env.reset()
    #     done = False

    #     for idx in range(len(env.unwrapped.valid_pos)):
    #         env.get_wrapper_attr('move_valid_pos')(idx)
    #         agent_pos = env.get_wrapper_attr('valid_pos')[idx]

    #         for _ in range(4):
    #             obs, _, _, _, _ = env.step(1)
    #             state = obs_to_state(obs)
    #             q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
    #             ind = State(obs)
    #             state_to_q[ind] = np.array(q)

    #         print(f'Context is {i+1} | {agent_pos}', end='\r')

    # print(len(state_to_q))

    # with open('configs/state_to_q.pl', 'wb') as file:
    #     dill.dump(state_to_q, file)
    state_to_q = np.zeros((200, 19, 19, 4, 3))

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
                state = obs_to_state(obs)
                q = find_all_action_values(
                    state[:2], state[2], state[3:5], state[5:], 0.99, size
                )
                state_to_q[context, state[0], state[1], state[2]] = np.array(q)

            print(f"Context is {context+1} | {state[:3]}", end="\r", flush=True)

        goal_q = np.array([1, 1, 0.970299])
        state_to_q[context, state[3], state[4], 0] = goal_q
        state_to_q[context, state[3], state[4], 1] = goal_q
        state_to_q[context, state[3], state[4], 2] = goal_q
        state_to_q[context, state[3], state[4], 3] = goal_q

    print(state_to_q.size)

    np.save("configs/state_to_q_np.npy", state_to_q)

    # testing
    env = gym_wrapper(
        gym.make(
            "MiniGrid-FourRooms-v1",
            agent_pos=train_config["agent positions"],
            goal_pos=train_config["goal positions"],
            doors_pos=train_config["topologies"],
            agent_dir=train_config["agent directions"],
            size=size,
            max_steps=1200,
        ),
        original_obs=True,
    )
    for _ in range(len(train_config["topologies"])):
        obs, _ = env.reset()
        done = False
        context = env.get_wrapper_attr("context")

        while not done:
            state = obs_to_state(obs)
            action = state_to_q[context, *state[:3]].argmax()
            obs, reward, trunc, term, _ = env.step(action)
            print(f"Context is {context+1} | {state[:2]}", end="\r")
            done = trunc or term

    print("All environments checked")
