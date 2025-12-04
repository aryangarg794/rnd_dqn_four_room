import matplotlib.pyplot as plt
import argparse
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
from plot_interactive import plot_env_heatmap, find_states
from dqn.model import DQN
from rnd_exploration.rnd import RNDNetwork
from utils.q_values import compute_q_value
from utils.record_scores import record_dqn_scores, get_rnd_scores, get_q_optimal, record_uncertainty_scores
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


@torch.no_grad()
def simulate_trajectory(file_name: str, alpha: float = 1.0, device: str = 'cuda', rnd: bool = False, optimal: bool = False):
    
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
    explore_map = results['explore_heatmap']
    switch_map = results['heatmap']
    
    agent = DQN(env, deepcopy(env), device=device)
    
    random_context = 199
    env.get_wrapper_attr('set_context')(random_context)
    obs, _ = env.reset()
    
    print(env.get_wrapper_attr('context'), random_context)
    
    if optimal:
        dqn_scores, _, dqn_scores_dirs = get_q_optimal(counter, random_context, 0, 0.99)
    else:
        agent.net.load_state_dict(torch.load(f'results/models/{file_name}.pt', weights_only=True))
        agent.net.eval()
        dqn_scores, _, dqn_scores_dirs = record_dqn_scores(agent, random_context, 0)
    context_info = env.get_wrapper_attr('context_info')(random_context)
    context_info = (*context_info, env.get_wrapper_attr('valid_pos'))
    normalized_scores = (dqn_scores - rms_dqn.avg)/rms_dqn.std
    normalized_scores_dirs = (dqn_scores_dirs - rms_dqn.avg)/rms_dqn.std
    uncertainty_map, _, uncertainty_map_dirs = record_uncertainty_scores(counter, random_context, 0)
    
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
            
    if rnd:
        rnd_net = RNDNetwork(env, device=device)
        rnd_net.load(f'results/models/rnd/{file_name}.pt')       
        rnd_scores, _ = get_rnd_scores(rnd_net, random_context, env_range=0, device=device)
        rms_rnd = results['running_mean_rnd']
        normalized_rnd = (rnd_scores - rms_rnd.avg) / rms_rnd.std
    
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

        if optimal:
            dqn_val = compute_q_value(obs, random_context, counter, 0.99)
            norm = (dqn_val - rms_dqn.avg)/rms_dqn.std 
        else:
            goal_action = state_to_q[State(obs)].argmax()
            dqn_val = agent(obs_torch).squeeze()[goal_action].item()
        obj_tuple = tuple([int(item) for item in state])
            
        obj_moving_tuple = (random_context, *agent_pos)

        if dqn_val - rms_dqn.avg >= alpha * rms_dqn.std and not record:
            relevant_buffer[*obj_moving_tuple] += 1
            agent.buffer.update_seen(obj_tuple)
            record = True
            target_pos = goal_pos
        
        elif np.array_equal(agent_pos, aux_pos) and not record: 
            target_pos = goal_pos
        
        if record:
            relevant_buffer[*obj_moving_tuple] += 1
            agent.buffer.update_seen(obj_tuple)
        
        
        # render logic
        env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
        agent_col = (255, 0, 0) if record else (0, 0, 255) 
        render = env.unwrapped.render(highlight_mask=ep_highlight_mask, colors=ep_colors, agent_col=agent_col)
        env.get_wrapper_attr('remove_aux')(aux_pos)
        
        fig, axes = plt.subplots(2, 4, figsize=(40, 20))
        
        axes[0, 0].imshow(cv2.transpose(render))
        axes[0, 0].set_title('Env Render')
        axes[0, 0].axis('off')
        
        plot_env_heatmap(
            relevant_buffer[random_context],
            context_info,
            'Buffer Counts',
            f'Buffer Heatmap for context {random_context}',
            axes[0, 3],
            agent_pos=agent_pos,
        )

        plot_env_heatmap(
            uncertainty_map.squeeze(),
            context_info,
            'Uncertainty Scores',
            f'Uncertainty for context {random_context}',
            axes[0, 1],
            agent_pos=agent_pos,
            intize=False
        )

        plot_env_heatmap(
            switch_map[random_context],
            context_info,
            'Switch Numbers',
            f'Switches (full) for context {random_context}',
            axes[0, 2],
            agent_pos=agent_pos,
        )
        
        # conversion due to transpose 0 -> 1, 1 -> 0, 2 -> 3, 3 -> 2
        plot_env_heatmap(
            normalized_scores_dirs[0].squeeze(),
            context_info,
            'DQN Scores',
            f'DQN Scores for context {random_context} for Down',
            axes[1, 0],
            agent_pos=agent_pos,
            intize=False
        )

        plot_env_heatmap(
            normalized_scores_dirs[1].squeeze(),
            context_info,
            'DQN Scores',
            f'DQN Scores for context {random_context} for Right',
            axes[1, 1],
            agent_pos=agent_pos,
            intize=False
        )

        plot_env_heatmap(
            normalized_scores_dirs[2].squeeze(),
            context_info,
            'DQN Scores',
            f'DQN Scores for context {random_context} for Up',
            axes[1, 2],
            agent_pos=agent_pos,
            intize=False
        )

        plot_env_heatmap(
            normalized_scores_dirs[3].squeeze(),
            context_info,
            'DQN Scores',
            f'DQN Scores for context {random_context} for Left',
            axes[1, 3],
            agent_pos=agent_pos,
            intize=False
        )
            
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rnd', action='store_true', help='rnd mode')
    parser.add_argument('-o', '--opt', action='store_true', help='optimal mode')
    parser.add_argument('-f', '--dir', type=str, default='dqn_count_test', help='save name')
    
    args = parser.parse_args()
    
    simulate_trajectory(args.dir, rnd=args.rnd, optimal=args.opt)