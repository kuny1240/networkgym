#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_client_sequential_training.py

# In this example, we perform sequential training for two environment sessions.
# the first session lasts for 3 episodes and the second session lasts for 1 episodes.
import numpy as np
import torch
import random
import sys
import copy
from utils.buffer import ReplayBuffer
from CleanRL_agents.sac import SACAgent
from tqdm import tqdm
from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from gymnasium.wrappers import NormalizeObservation

sys.path.append('../')
sys.path.append('../../')

def evaluate(model, env, n_episodes):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        # breakpoint()
        while not done:
            
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = truncated
            total_reward += reward
        rewards.append(total_reward)
        print("Episode: {}, Reward: {}".format(i, total_reward))

    avg_reward = sum(rewards) / n_episodes
    return avg_reward

MODEL_SAVE_FREQ = 200
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 10


client_id = 1
env_name = "network_slicing"
config_json = load_config_file(env_name)
config_json["rl_config"]["agent"] = "SACAgent"
# Create the environment
env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id and configure file arguements.
normalized_env = NormalizeObservation(env) # normalize the observation

num_steps = 10000
# breakpoint()
obs, info = normalized_env.reset()
agent = SACAgent(state_dim=obs.shape[0], 
                     action_dim=env.action_space.shape[0], 
                     actor_lr=6e-5, 
                     critic_lr=3e-4,
                     action_high=1,
                     action_low=0,)
eva_agent = copy.deepcopy(agent)
buffer = ReplayBuffer(max_size=1000000, obs_shape=obs.shape[0], n_actions=env.action_space.shape[0])
epsilon = 1.0

num_episodes = 0
progress_bar = tqdm(range(num_steps))
# Training loop
# Evaluate every 500 steps, same as model saving frequency
for step in progress_bar:
    
    
    if random.random() < epsilon:
        action = normalized_env.action_space.sample()
    else:
            # breakpoint()
        action = agent.predict(obs)
    # action = np.exp(action)/np.sum(np.exp(action))  # softmax
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    nxt_obs, reward, terminated, truncated, info = normalized_env.step(action=action)
    buffer.store(obs, action, reward, nxt_obs, truncated)
    obs = nxt_obs
    
    
    if buffer.mem_cntr > 64:
        training_batch = buffer.sample(64)
        # breakpoint()
        agent.learn(*training_batch)

    # If the environment is end, exit
    if terminated:
        break

    # If the epsiode is up (environment still running), then start another one
    if truncated:
        obs, info = normalized_env.reset()
        obs = torch.Tensor(obs)
        epsilon = max(epsilon*0.99, 0.01)
        num_episodes += 1

    if (step + 1) % MODEL_SAVE_FREQ == 0:
        print("Step: {}, Saving model...".format(step))
        agent.save("./models/sac_model_{}_{}_{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], config_json["env_config"]["random_seed"]))
        eva_agent.load("./models/sac_model_{}_{}_{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], config_json["env_config"]["random_seed"]))
        eva_agent.actor.eval()
        avg_reward = evaluate(eva_agent, normalized_env, n_episodes=2)
        print("Step: {}, Eval Reward: {}".format(step, avg_reward))
        env.adapter.wandb.log({"eval_avg_reward": avg_reward})
    
    progress_bar.set_description("Step: {}, Reward: {}, Action: {}".format(step, reward, action))
    

