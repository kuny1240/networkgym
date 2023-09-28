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
from utils.utils import *

sys.path.append('../')
sys.path.append('../../')



slice_lists = slice_lists = [
    [
        {"num_users":6,"dedicated_rbg":0,"prioritized_rbg":12,"shared_rbg":25},
        {"num_users":20,"dedicated_rbg":0,"prioritized_rbg":13,"shared_rbg":25},
        {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":25}
    ],
     [
        {"num_users":11,"dedicated_rbg":0,"prioritized_rbg":12,"shared_rbg":25},
        {"num_users":15,"dedicated_rbg":0,"prioritized_rbg":13,"shared_rbg":25},
        {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":25}
    ],
    [
        {"num_users":13,"dedicated_rbg":0,"prioritized_rbg":12,"shared_rbg":25},
        {"num_users":13,"dedicated_rbg":0,"prioritized_rbg":13,"shared_rbg":25},
        {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":25}
    ],
    [
        {"num_users":15,"dedicated_rbg":0,"prioritized_rbg":12,"shared_rbg":25},
        {"num_users":11,"dedicated_rbg":0,"prioritized_rbg":13,"shared_rbg":25},
        {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":25}
    ],
     [
        {"num_users":20,"dedicated_rbg":0,"prioritized_rbg":12,"shared_rbg":25},
        {"num_users":6,"dedicated_rbg":0,"prioritized_rbg":13,"shared_rbg":25},
        {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":25}
    ],
    
]



MODEL_SAVE_FREQ = 2000
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 10
EVAL_EPI_PER_SESSION = 1

def main(agent_type:str,
         env_name:str,
         dataset:str,
         num_steps = 60000,
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
    buffer.load_buffer(f"./dataset/{dataset}_buffer.h5")
    # buffer.nomarlize_states()
    # Create the environment
    target_entropy = -np.prod((2,)).item()
    # breakpoint()
    agent = CQL(state_dim=15, action_dim=2, hidden_dim=hidden_dim, target_entropy=target_entropy,
                q_n_hidden_layers=2, max_action=1, qf_lr=3e-3, policy_lr=6e-4,device="cuda:0")
    run = wandb.init(project="network-slicing-offline",
               name=f"{agent_type}-nn-{dataset}-ver{storage_ver}", 
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
            agent.save("./models/cql_dataset_{}_ver{}.pt".format(dataset, storage_ver))
            eval_agent = copy.deepcopy(agent)
            eval_agent.actor.eval()
            config_json["env_config"]["steps_per_episode"] = 52
            config_json["env_config"]["episodes_per_session"] = EVAL_EPI_PER_SESSION
            random_seed = 1
            avg_reward = 0
            for slice_list in slice_lists:
                print("evaluating env: {}".format(slice_list))
                config_json["env_config"]["slice_list"] = slice_list
                config_json["env_config"]["random_seed"] = random_seed
                eval_env = NetworkGymEnv(client_id, config_json, log=False)
                normalized_eval_env = NormalizeObservation(eval_env)
                env_reward, eval_dict = evaluate(eval_agent, normalized_eval_env, n_episodes=1)
                avg_reward += env_reward
                wandb.log(eval_dict)
                
               
            avg_reward /= len(slice_lists)
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


    

