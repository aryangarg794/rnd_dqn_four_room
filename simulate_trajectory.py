import matplotlib.pyplot as plt
import gymnasium as gym
import dill
import torch
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image

from four_room.env import FourRoomsEnv
from four_room.constants import train_config, size, state_to_q
from four_room.wrappers import gym_wrapper
from four_room.shortest_path import find_all_shortest_paths, compute_actions
from four_room.utils import obs_to_state
from rnd_exploration.dataset import State
from run_dqn_count import record_dqn_scores
from plot_interactive import plot_env_heatmap, find_states
from dqn.model import DQN
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

@torch.no_grad()
def simulate_trajectory(file_name: str, alpha: float = 1.0, device: str = 'cuda'):
    
    env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos=train_config['agent positions'],
                goal_pos=train_config['goal positions'],
                doors_pos=train_config['topologies'],
                agent_dir=train_config['agent directions'],
                size=size, 
                render_mode='rgb_array',
                disable_env_checker=True
            ),
            original_obs=True
        )
    
    with open(f'results/dqn_exps/{file_name}.pl', 'rb') as file:
        results = dill.load(file)
    aux_states = results['aux_heatmap']
    
    counter = results['counter_moving']
    rms_dqn = results['running_mean']
    timesteps = len(results['context_history'])
        
    agent = DQN(env, deepcopy(env), device=device)
    agent.net.load_state_dict(torch.load(f'results/models/{file_name}.pt', weights_only=True))
    agent.net.eval()
    
    random_context = np.random.randint(low=0, high=200)
    env.get_wrapper_attr('set_context')(random_context-1)
    dqn_scores, _ = record_dqn_scores(agent, random_context, 0)
    context_info = env.get_wrapper_attr('context_info')(random_context)
    context_info = (*context_info, env.get_wrapper_attr('valid_pos'))
    normalized_scores = (dqn_scores - rms_dqn.avg)/rms_dqn.std

    obs, _ = env.reset()
    max_k = len(env.get_wrapper_attr('valid_pos'))
    k = np.random.randint(low=0, high=max_k)
    aux_pos = env.get_wrapper_attr('valid_pos')[k]
    
    state = obs_to_state(obs)
    goal_pos = state[3:5]
    paths = find_all_shortest_paths(state[:2], state[2], aux_pos, state[5:], size)
    path_index = np.random.randint(low=0, high=len(paths))
    actions = compute_actions(paths[path_index])
    done = False
    target_pos = aux_pos
    record = False
    step = timesteps
    
    images = []
    ep_highlight_mask = np.zeros((env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
    ep_colors = np.empty_like(ep_highlight_mask, dtype=object)
    
    relevant_buffer_states = find_states(counter.all_states, timesteps, counter.capacity)
    relevant_buffer = np.zeros_like(aux_states)
    for state in relevant_buffer_states:
        relevant_buffer[*state] += 1
        if state[0] == random_context:
            ep_highlight_mask[state[1], state[2]] = True
            ep_colors[state[1], state[2]] = (51, 0, 102)
            
    while not done:
        
        step += 1
        agent_pos = env.get_wrapper_attr('agent_pos')
        obs_torch = torch.from_numpy(obs).to(device=device).unsqueeze(dim=0)
        if np.array_equal(target_pos, aux_pos) and not np.array_equal(agent_pos, aux_pos):
            action = actions.pop(0)
        else:
            state_obj = State(state=obs)
            q = state_to_q[state_obj]
            action = q.argmax() if isinstance(q, np.ndarray) else np.array(q).argmax()

        goal_action = state_to_q[State(obs)].argmax()
        dqn_val = agent(obs_torch).squeeze()[goal_action].item()
        obj_tuple = tuple([int(item) for item in state])
            
        obj_moving_tuple = (random_context, *agent_pos)
        if record:
            relevant_buffer[*obj_moving_tuple] += 1
            agent.buffer.update_seen(obj_tuple)
            
        elif dqn_val - rms_dqn.avg >= alpha * rms_dqn.std and not record:
            relevant_buffer[*obj_moving_tuple] += 1
            agent.buffer.update_seen(obj_tuple)
            record = True
            target_pos = goal_pos
        
        elif np.array_equal(agent_pos, aux_pos) and not record: 
            target_pos = goal_pos
        
        
        # render logic
        env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
        agent_col = (255, 0, 0) if record else (0, 0, 255) 
        render = env.unwrapped.render(highlight_mask=ep_highlight_mask, colors=ep_colors, agent_col=agent_col)
        env.get_wrapper_attr('remove_aux')(aux_pos)
            
        fig, axes = plt.subplots(1, 4, figsize=(40, 10))
        
        axes[0].imshow(cv2.transpose(render))
        axes[0].set_title('Env Render')
        axes[0].axis('off')
        
        plot_env_heatmap(normalized_scores.squeeze(), context_info,
                     'Normalized DQN Scores', f'Normalized DQN Scores for context {random_context}', axes[1], intize=False)
        plot_env_heatmap(dqn_scores.squeeze(), context_info,
                     'DQN Scores', f'DQN Scores for context {random_context}', axes[2], intize=False) 
        plot_env_heatmap(relevant_buffer[random_context], context_info,
                     'Buffer Counts', f'Buffer Heatmap for context {random_context}', axes[3])
        
        fig.tight_layout()
        # fig.savefig(f'results/videos/image_{step}.png')
        fig.canvas.draw()
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        pil_img = Image.frombytes("RGBA", (w, h), buf, "raw", "ARGB")
        pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        images.append(image)
        
        obs_prime, _, terminated, truncated, _ = env.step(action)
        obs = obs_prime
        done = terminated or truncated
        plt.close(fig)

    first_frame = images[0]
    height, width, _ = first_frame.shape
    writer = cv2.VideoWriter(
        f'results/videos/{file_name}.mp4',
        cv2.VideoWriter_fourcc(*"mp4v"),
        1,
        (width, height)
    )
    for image in images:
        frame = cv2.resize(image, (width, height))
        writer.write(frame)

    writer.release()
    print(f"Saved video")
    
if __name__ == '__main__':
    simulate_trajectory('dqn_count_test_seed_0_100000')