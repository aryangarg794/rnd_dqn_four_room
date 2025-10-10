import numpy as np 
import gymnasium as gym
import dill 
import argparse
import imageio

from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.wrappers import gym_wrapper
from rnd_exploration.dataset import ExploreGoDataset, Transition
from four_room.constants import train_config, test_config, size

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

def create_explogostar_dataset(
    dataset_size, 
    save_dir, 
    dropout=0.1, 
    render=False, 
):
    print(f'=============Dataset {save_dir} | Size {dataset_size} ====================')
    
    
    if render:
        env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos= train_config['agent positions'],
                goal_pos = train_config['goal positions'],
                doors_pos = train_config['topologies'],
                agent_dir = train_config['agent directions'],
                size=size, 
                render_mode="rgb_array",
                disable_env_checker=True
            ),
            original_obs=True
        )
    else:
        env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos= train_config['agent positions'],
                goal_pos = train_config['goal positions'],
                doors_pos = train_config['topologies'],
                agent_dir = train_config['agent directions'],
                size=size, 
                disable_env_checker=True
            ),
            original_obs=True
        )
        
        
    try:
        explorego = ExploreGoDataset()
        imgs = []
        iters = 0

        ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
        ep_colors = np.empty_like(ep_highlight_mask, dtype=object)

        while len(explorego) <= dataset_size:
            obs, _ = env.reset()
            done = False
            
            # emulate the (very good) pure exploration of explorego
            max_k = len(env.get_wrapper_attr('valid_pos'))
            k = np.random.randint(low=0, high=max_k)
            env.get_wrapper_attr('move_valid_pos')(k)
        
            current_context = env.unwrapped.context
            
            # find optimal trajectory
            past_pos = []
            while not done:
                agent_pos = env.get_wrapper_attr('agent_pos')

                state = obs_to_state(obs)
                q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, size)
                q = np.array(q)
                action = q.argmax()
                
                if len(explorego) < 5000:
                    
                    explorego.add_trans(np.array(obs), q)
                    explorego.add(np.array(obs), q, np.array(state))
                    if render: 
                        ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                        ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                        past_pos.append(agent_pos)
                elif np.random.random_sample() <= dropout:
                    explorego.add_trans(np.array(obs), q)
                    explorego.add(np.array(obs), q, np.array(state))
                    if render: 
                        ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                        ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                        past_pos.append(agent_pos)
                
                # env step logic
                obs_prime, _, terminated, truncated, _ = env.step(action)
                obs = obs_prime
                done = terminated or truncated
                
                if render and len(explorego) >= dataset_size - 1000: imgs.append(env.get_wrapper_attr('render')(highlight_mask=ep_highlight_mask[current_context], 
                                                colors=ep_colors[current_context]))
                iters += 1
                
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)
                    
                    
            print(f'Current size of dataset: {len(explorego):08d} | Current Context {current_context} | Current Uniqueness {explorego.ratio_unique_trans:.4f}', end='\r')

    except KeyboardInterrupt:
        with open(f'action_values/{save_dir}.pl', 'wb') as file:
            dill.dump(explorego, file)
        
    # save the obj       
    with open(f'action_values/{save_dir}.pl', 'wb') as file:
        dill.dump(explorego, file)
        
    return explorego, imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=25000, help='size of dataset')
    parser.add_argument('-f', '--dir', type=str, default='explore_go_random', help='name of dataset')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')

    args = parser.parse_args()

    dataset, imgs = create_explogostar_dataset(
        dataset_size=args.size, 
        save_dir=args.dir,  
        render=args.render,
    )
    print('\nDone')
   
    if args.render:
        imageio.mimsave(f'renders/rendered_{args.dir}.gif', [np.array(img) for i, img in enumerate(imgs[-1000:]) if i%1 == 0], duration=100)