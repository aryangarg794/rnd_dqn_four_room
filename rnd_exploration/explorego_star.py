import numpy as np 
import gymnasium as gym
import dill 
import argparse
import imageio

from gymnasium.wrappers.normalize import RunningMeanStd
from four_room.env import FourRoomsEnv
from four_room.utils import obs_to_state
from four_room.shortest_path import find_all_action_values
from four_room.wrappers import gym_wrapper
from rnd_exploration.dataset import ExploreGoDataset, Transition
from rnd_exploration.rnd import RNDNetwork
from rnd_exploration.utils import RunningAverage, train_config, test_config, size

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

def create_star_dataset(
    dataset_size, 
    save_dir, 
    alpha=1.5, 
    render=False, 
    device='cpu', 
    warmup=True, 
    warmupsteps=3500, 
    batch_size=512
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
        rnd_net = RNDNetwork(env, device=device, lr=1e-5)
        explorego = ExploreGoDataset()
        rms = RunningAverage()
        imgs = []
        iters = 0

        ep_highlight_mask = np.zeros((len(train_config['agent positions']), 
                                        env.get_wrapper_attr('width'), env.get_wrapper_attr('height')), dtype=bool)
        ep_colors = np.empty_like(ep_highlight_mask, dtype=object)

        while len(explorego) <= dataset_size:
            obs, _ = env.reset()
            done = False
            state = obs_to_state(obs)
            
            # emulate the (very good) pure exploration of explorego
            max_k = len(env.get_wrapper_attr('valid_pos'))
            k = np.random.randint(low=0, high=max_k)
            aux_pos = env.get_wrapper_attr('valid_pos')[k]
            

            if len(explorego) < warmupsteps:
                target_pos = state[3:5]
            else: 
                target_pos = aux_pos
                
            record = False
        
            current_context = env.unwrapped.context
            
            # find optimal trajectory
            past_pos = []
            while not done:
                agent_pos = env.get_wrapper_attr('agent_pos')
                
                state = obs_to_state(obs)
                goal_pos = state[3:5]
                q = find_all_action_values(state[:2], state[2], target_pos, state[5:], 0.99, size)
                q = np.array(q)
                action = q.argmax()
                rnd_val = rnd_net.get_error(obs).item()
                
                if record and len(explorego) >= warmupsteps:
                    assert np.array_equal(target_pos, goal_pos)
                    explorego.add_trans(np.array(obs), q)
                    explorego.add(np.array(obs), q, np.array(state))
                    if render: 
                        ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                        ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                        past_pos.append(agent_pos)
                    rnd_net.observe(obs)
                
                if len(explorego) < warmupsteps and warmup:
                    assert np.array_equal(target_pos, goal_pos)
                    explorego.add_trans(np.array(obs), q)
                    explorego.add(np.array(obs), q, np.array(state))
                    if render: 
                        ep_colors[current_context, agent_pos[0], agent_pos[1]] = (0, 0, 255)
                        ep_highlight_mask[current_context, agent_pos[0], agent_pos[1]] = True
                        past_pos.append(agent_pos)
                    rnd_net.observe(obs)
                    
                elif rnd_val - rms.avg >= alpha * rms.std or np.array_equal(agent_pos, aux_pos): # swap to record mode 
                    record = True
                    target_pos = goal_pos
                    
                elif rnd_val - rms.avg <= -alpha * rms.std and record: # swap to explore mode
                    k = np.random.randint(low=0, high=max_k)
                    aux_pos = env.get_wrapper_attr('valid_pos')[k]
                    record = False
                    target_pos = aux_pos
                
                rms.update(rnd_val)
                # env step logic
                obs_prime, _, terminated, truncated, _ = env.step(action)
                obs = obs_prime
                done = terminated or truncated
                if iters % 2 == 0 and len(explorego) > batch_size:
                    xs, _ = explorego.sample(batch_size)
                    rnd_net.observe(xs.to(device))
                aux_or_goal = 'goal' if np.array_equal(target_pos, goal_pos) else 'aux'
                if render and len(explorego) >= dataset_size - 1000:
                    env.get_wrapper_attr('set_aux')(aux_pos) # cannot add beforehand or else included in obs
                    agent_col = (255, 0, 0) if np.array_equal(target_pos, goal_pos) else (0, 0, 255) 
                    
                    imgs.append(env.get_wrapper_attr('render')(highlight_mask=ep_highlight_mask[current_context], 
                                                colors=ep_colors[current_context], agent_col=agent_col))
                    env.get_wrapper_attr('remove_aux')(aux_pos)
                    
                iters += 1
                # target = 'goal' if np.array_equal(target_pos, goal_pos) else 'auxg' 
                # print(f'Current size of dataset: {len(explorego):08d} | Current Context {current_context:03d} | Current Uniqueness {explorego.ratio_unique_trans:.4f} | Val {rnd_val:.4f} | Current Target: {target} | Recording: {record}')
                
            if render:
                for pos in past_pos:
                    ep_colors[current_context, pos[0], pos[1]] = (51, 0, 102)
                    
            print(f'Current size of dataset: {len(explorego):08d} | Current Context {current_context} | Current Uniqueness {explorego.ratio_unique_trans:.4f} | Val {rnd_val:.4f} | Avg {rms.avg:.4f} | Var {rms.std:.4f}', end='\r')
            

    except KeyboardInterrupt:
        with open(f'action_values/{save_dir}.pl', 'wb') as file:
            dill.dump(explorego, file)
        
    # save the obj       
    with open(f'action_values/{save_dir}.pl', 'wb') as file:
        dill.dump(explorego, file)
        
    return explorego, imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=3500, help='size of dataset')
    parser.add_argument('-f', '--dir', type=str, default='test', help='name of dataset')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='alpha')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-r', '--render', action='store_true', help='render mode')
    parser.add_argument('-w', '--warm', action='store_false', help='warmup mode')

    args = parser.parse_args()

    dataset, imgs = create_star_dataset(
        dataset_size=args.size, 
        save_dir=args.dir,  
        render=args.render,
        device=args.device,
        warmup=args.warm,
        warmupsteps=3500, 
        alpha=args.alpha, 
    )
    print('\nDone')
   
    if args.render:
        imageio.mimsave(f'renders/rendered_{args.dir}.gif', [np.array(img) for i, img in enumerate(imgs[-1000:]) if i%1 == 0], duration=200)