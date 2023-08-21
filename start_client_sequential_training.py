#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_client_sequential_training.py

# In this example, we perform sequential training for two environment sessions.
# the first session lasts for 3 episodes and the second session lasts for 1 episodes.
import numpy as np
import torch
import random
import sys
from utils.buffer import ReplayBuffer
from CleanRL_agents.sac import SACAgent

from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from gymnasium.wrappers import NormalizeObservation

sys.path.append('../')
sys.path.append('../../')

# def evaluate(model, env, n_episodes):
#     rewards = []
#     for i in range(n_episodes):
#         obs = env.reset()
#         done = False
#         total_reward = 0
#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, _ = env.step(action)
#             total_reward += reward
#         rewards.append(total_reward)

#     avg_reward = sum(rewards) / n_episodes
#     return avg_reward

MODEL_SAVE_FREQ = 1000
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 5


client_id = 0
env_name = "network_slicing"
config_json = load_config_file(env_name)
config_json["rl_config"]["agent"] = "SACAgent"
# Create the environment
env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id and configure file arguements.
normalized_env = NormalizeObservation(env) # normalize the observation

num_steps = 10000
breakpoint()
obs, info = normalized_env.reset()
obs = torch.Tensor(obs)
agent = SACAgent(state_dim=obs.shape[0], 
                     action_dim=env.action_space.shape[0], 
                     actor_lr=0.0003, 
                     critic_lr=0.003)
buffer = ReplayBuffer(max_size=100000, obs_shape=obs.shape[0], n_actions=env.action_space.shape[0])
epsilon = 1.0
for step in range(num_steps):
    
    
    if random.random() < epsilon:
        action = normalized_env.action_space.sample()
    else:
        try:
            action = agent.predict(obs)
        except:
            breakpoint()
    
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    nxt_obs, reward, terminated, truncated, info = normalized_env.step(action=action)
    buffer.store(obs, action, reward, nxt_obs, truncated)
    obs = nxt_obs
    obs = torch.Tensor(obs)
    
    
    if buffer.mem_cntr > 32:
        training_batch = buffer.sample(32)
        agent.learn(*training_batch)

    # If the environment is end, exit
    if terminated:
        break

    # If the epsiode is up (environment still running), then start another one
    if truncated:
        obs, info = normalized_env.reset()
        obs = torch.Tensor(obs)
        epsilon = max(epsilon*0.99, 0.01)


