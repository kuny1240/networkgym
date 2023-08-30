#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_evaluation.py
import numpy as np
from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from gymnasium.wrappers import NormalizeObservation
from CleanRL_agents.sac import SACAgent
from tqdm import tqdm


client_id = 0
env_name = "network_slicing"
config_json = load_config_file(env_name)
config_json["rl_config"]["agent"] = "sac"
num_steps = 200
steps_per_episode = 100
episode_per_session = num_steps // steps_per_episode

config_json["env_config"]["episode_per_session"] = episode_per_session
config_json["env_config"]["steps_per_episode"] = steps_per_episode


eval_method = "sac"
# Create the environment
env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id and configure file arguements.
normalized_env = NormalizeObservation(env) # normalize the observation


obs, info = normalized_env.reset()


if eval_method == "sac":
    sac_agent = SACAgent(state_dim=obs.shape[0], 
                             action_dim=env.action_space.shape[0], 
                             actor_lr=0.003, 
                             critic_lr=0.03,
                             action_high=1,
                             action_low=0,)
    sac_agent.load("./models/sac_model_3.ckpt")
    sac_agent.actor.eval()
    
progress_bar = tqdm(range(num_steps))

for step in progress_bar:

    if eval_method == "random":
        action = normalized_env.action_space.sample()  # agent policy that uses the observation and info
    elif eval_method == "sac":
        action = sac_agent.predict(obs)
    else:
        action = [1/len(config_json["env_config"]["slice_list"])] * len(config_json["env_config"]["slice_list"])
        action = np.array(action)
    obs, reward, terminated, truncated, info = env.step(action)

    # If the environment is end, exit
    if terminated:
        break

    # If the epsiode is up (environment still running), then start another one
    if truncated:
        obs, info = env.reset()
        
    progress_bar.set_description("Step: {}, Reward: {}, Action: {}".format(step, reward, action))


