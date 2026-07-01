import dill
from four_room.env import FourRoomsEnv
import gymnasium as gym
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
from four_room.wrappers import gym_wrapper
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state
from four_room.constants import train_config
from tqdm import tqdm
import copy
import numpy as np

env_size = 19
original_obs = True
valid_agent_pos, valid_goal_pos, valid_doors_pos = FourRoomsEnv.valid_positions(env_size)
all_pos = set(valid_agent_pos)

reachable_obs = []
for i in tqdm(range(len(train_config['topologies']))):
	top = train_config['topologies'][i]
	extra_pos = (env_size // 2 , top[0]+1), (env_size // 2, top[1]+ (env_size // 2) + 1), (top[2]+1, env_size // 2), (top[3]+ (env_size // 2) + 1, env_size // 2)

	all_pos_copy = copy.deepcopy(all_pos)
	all_pos_copy.remove(train_config['goal positions'][i])

	for ep in extra_pos:
		all_pos_copy.add(ep)

	for p in all_pos_copy:
		for dir in range(4):
			env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
									agent_pos=[p], 
									goal_pos=[train_config['goal positions'][i]], 
									doors_pos=[train_config['topologies'][i]], 
									agent_dir=[dir],
									size=env_size,
									render_mode='rgb_array'), original_obs=original_obs)
			obs, _ = env.reset()
			reachable_obs.append(obs)

reachable_obs = np.array(reachable_obs)

print(reachable_obs.shape)
			
with open(f'configs/train_reachable_space.pl', 'wb') as file:
	dill.dump(reachable_obs, file)


optimal_actions = []
obs_to_optimal_values = dict()
for obs in tqdm(reachable_obs):
	state = obs_to_state(obs)
	q = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99, env_size)
	obs_to_optimal_values[obs.data.tobytes()] = tuple(q)
	optimal_actions.append(set([a.item() for a in np.argwhere(np.array(q) == np.array(q).max())]))

with open(f'configs/train_reachable_space_opt_actions.pl', 'wb') as file:
	dill.dump(optimal_actions, file)

with open(f'configs/obs_to_q_values_map.pl', 'wb') as file:
	dill.dump(obs_to_optimal_values, file)
