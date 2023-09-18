#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_client_sequential_training.py

# In this example, we perform sequential training for two environment sessions.
# the first session lasts for 3 episodes and the second session lasts for 1 episodes.
import numpy as np
import torch
import random
import sys
import fire
import copy
import wandb
from utils.buffer import ReplayBuffer
from CleanRL_agents import SACAgent
from CORL_agents import CQL
from tqdm import tqdm
from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from gymnasium.wrappers import NormalizeObservation

sys.path.append('../')
sys.path.append('../../')

def evaluate(model, env, n_episodes):
    rewards = []
    dict_slice = {}
    num_slices = 3
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        # breakpoint()
        pbar = tqdm(range(200))
        for _ in pbar:
            
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = truncated
            total_reward += reward
            pbar.set_description(f"Actions: {action}, reward: {reward}, total_reward: {total_reward}")
            if terminated:
                break
        rewards.append(total_reward)
        print("Episode: {}, Reward: {}".format(i, total_reward))

    avg_reward = sum(rewards) / n_episodes
    return avg_reward, dict_slice

MODEL_SAVE_FREQ = 500
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 10
EVAL_EPI_PER_SESSION = 1

def main(agent_type:str,
         env_name:str,
         num_steps = 6000,
         client_id = 0,
         hidden_dim = 64,
         steps_per_episode = 10,
         episode_per_session = 1,
         random_seed = 1,
         ):

    # client_id = 1
    # env_name = "network_slicing"
    
    storage_ver = 0
    config_json = load_config_file(env_name)
    config_json["env_config"]["random_seed"] = random_seed
    train_random_seed = random_seed
    config_json["rl_config"]["agent"] = agent_type
    config_json["env_config"]["steps_per_episode"] = steps_per_episode
    config_json["env_config"]["episodes_per_session"] = episode_per_session
    buffer = ReplayBuffer(max_size=1000000, obs_shape=15, n_actions=2)
    buffer.load_buffer("./dataset/offline_data_heavy_traffic.h5")
    buffer.nomarlize_states()
    # Create the environment
    target_entropy = -np.prod((2,)).item()
    # breakpoint()
    agent = CQL(state_dim=15, action_dim=2, hidden_dim=hidden_dim, target_entropy=target_entropy,
                q_n_hidden_layers=2, max_action=1, qf_lr=3e-3, policy_lr=6e-4,device="cuda:0")
    wandb.init(project="network-slicing-offline", 
                     config=config_json)

    num_episodes = 0
    progress_bar = tqdm(range(num_steps))
    best_eval_reward = -np.inf
    # Training loop
    # Evaluate every 500 steps, same as model saving frequency
    for step in progress_bar:
        
        batch = buffer.sample(256)
        # breakpoint()
        train_info = agent.learn(*batch)
        wandb.log(train_info)
        if (step + 1) % MODEL_SAVE_FREQ == 0:
            print("Step: {}, Saving model...".format(step))
            agent.save("./models/cql_model_ver{}.pt".format(storage_ver))
            eval_agent = copy.deepcopy(agent)
            eval_agent.actor.eval()
            config_json["env_config"]["steps_per_episode"] = 100
            config_json["env_config"]["episodes_per_session"] = EVAL_EPI_PER_SESSION
            random_seeds = [1, 21, 35]
            avg_reward = 0
            for random_seed in random_seeds:
                config_json["env_config"]["random_seed"] = random_seed
                eval_env = NetworkGymEnv(1, config_json, log=False)
                normalized_eval_env = NormalizeObservation(eval_env)
                env_reward, eval_info = evaluate(eval_agent, normalized_eval_env, n_episodes=1)
                avg_reward += env_reward
               
            avg_reward /= len(random_seeds)
            art = wandb.Artifact(f"{agent_type}-nn-{wandb.run.id}", type="model")
            art.add_file("./models/cql_model_ver{}.pt".format(storage_ver))
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                wandb.log_artifact(art, aliases=["latest", "best"])
            else:
                wandb.log_artifact(art)
            print("Step: {}, Eval Reward: {}".format(step, avg_reward))
            wandb.log({"eval_avg_reward": avg_reward})
            # buffer.save_buffer("./dataset/offline_data_heavy_traffic.h5")
            storage_ver += 1
            
        
            progress_bar.set_description("Step: {}, Eval Reward: {}".format(step, avg_reward))


if __name__ == "__main__":
    fire.Fire(main)


    

