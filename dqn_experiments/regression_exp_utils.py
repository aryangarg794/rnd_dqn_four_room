import gymnasium as gym
import numpy as np
import torch
import time
import os
import datetime

from tqdm import tqdm

from regression.experiment import RegressionModel
from rnd_exploration.dataset import ReplayBuffer
from four_room.wrappers import gym_wrapper_state
from four_room.env import FourRoomsEnv
from four_room.constants import val_config, test_config, size

gym.register("MiniGrid-FourRooms-v1", FourRoomsEnv)


def run_experiment(
    buffer: ReplayBuffer,
    timesteps: int = int(5e5),
    val_freq: int = int(1e4),
    batch_size: int = 128,
    device: str = "cuda",
    use_cnn: bool = False, 
    print_freq: int = 5000,
):
    save_dir = "reg_models"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pt")
    val_env = gym_wrapper_state(
        gym.make(
            "MiniGrid-FourRooms-v1",
            agent_pos=val_config["agent positions"],
            goal_pos=val_config["goal positions"],
            doors_pos=val_config["topologies"],
            agent_dir=val_config["agent directions"],
            size=size,
        ),
        use_cnn=use_cnn,
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
        use_cnn=use_cnn,
    )
    val_scores = []
    model = RegressionModel(val_env, val_env, device=device, use_cnn=use_cnn).to(device=device)
    best_val_score = -float('inf')

    X_train = buffer.states.to(device).float()
    y_train = buffer.q_values.to(device).float()
    n_samples = buffer.capacity

    start_time = time.time()
    for step in (pbar := tqdm(range(1, timesteps + 1), disable=True)):
        batch_idx = torch.randint(0, n_samples, (batch_size,), device=device)
        batch_x = X_train[batch_idx]
        batch_y = y_train[batch_idx]

        preds = model(batch_x)
        loss = model.loss(preds, batch_y)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if step % val_freq == 0:
            val_reward = model.validation(env=val_env)
            val_scores.append(val_reward)

            if val_reward > best_val_score:
                torch.save(model.state_dict(), best_model_path)
                best_val_score = val_reward

        postfix = {}
        postfix['loss'] = loss.item()
        postfix['val_score'] = val_scores[-1] if len(val_scores) > 0 else 0.0
        pbar.set_postfix(postfix)

    if os.path.exists(best_model_path):
        print(f"Loading best model with score: {best_val_score}")
        model.load_state_dict(torch.load(best_model_path))

    test_result = model.validation(test_env, val_steps=200)
    return val_scores, test_result
