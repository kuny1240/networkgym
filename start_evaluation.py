#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_evaluation.py
import numpy as np
import wandb
from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from gymnasium.wrappers import NormalizeObservation
from CleanRL_agents.sac import SACAgent
from CORL_agents import CQL
from tqdm import tqdm


client_id = 0
env_name = "network_slicing"
config_json = load_config_file(env_name)

num_steps = 100
steps_per_episode = 100
episode_per_session = num_steps // steps_per_episode

# Evaluation code, to test the performance of different methods, we need to test:
# 1. Different traffic patterns {User number, traffic type, traffic distribution}
# 2. Different agent {SAC, CQL, TD3, DDPG, PPO, A2C, etc.}
# 3. Different reward function {delay, delay + violation, delay + violation + cost, etc.}
# 4. Different network topology {number of slices, number of nodes, number of links, etc.}
# 5. Different random seeds

def baseline(loads):
    '''
    propotionally allocate rb resource to slices accordling to their traffic load
    '''
    total_load = sum(loads) +  1e-6
    loads = [load / total_load for load in loads]
    
    return np.array(loads)



config_json["env_config"]["episodes_per_session"] = episode_per_session
config_json["env_config"]["steps_per_episode"] = steps_per_episode
config_json["env_config"]["random_seed"] = 1


eval_method = "sac"
eval_method = "baseline"
eval_methods = ["sac", "cql", "baseline", "equal"]
# eval_method = "equal"
# eval_method = "cql"
# Create the environment
# breakpoint()
for eval_method in eval_methods:
    
    config_json["rl_config"]["agent"] = eval_method
    env = NetworkGymEnv(1, config_json) # make a network env using pass client id and configure file arguements.
    normalized_env = NormalizeObservation(env) # normalize the observation


    wandb.init(project = "network_gym_client", name = f"method_{eval_method}_evaluation", config = config_json)

    # breakpoint()
    obs, info = normalized_env.reset()


    if eval_method == "sac":
        sac_agent = SACAgent(state_dim=obs.shape[0], 
                                action_dim=env.action_space.shape[0], 
                                hidden_dim=64,
                                actor_lr=0.003, 
                                critic_lr=0.03,
                                action_high=1,
                                action_low=0,)
        sac_agent.load("./models/sac_model_3_weighted_80_best.ckpt")
        sac_agent.actor.eval()
    elif eval_method == "cql":
        agent = CQL(state_dim=15, action_dim=2, hidden_dim=64, target_entropy=-2,
                    q_n_hidden_layers=2, max_action=1, qf_lr=3e-4, policy_lr=6e-5,device="cuda:0")
        agent.load("./models/cql_model_best.pt")
        agent.actor.eval()
        
    progress_bar = tqdm(range(num_steps))

    for step in progress_bar:

        if eval_method == "baseline":
            df = info["network_stats"]
            loads =  np.array(df[df["name"] == "tx_rate"]["value"].to_list()[0])
            disturbed_user =  np.array(df[df['name'] == "slice_id"]["user"].to_list()[0])
            arg_sort_id = np.argsort(disturbed_user)
            slice_ids = np.array(df[df["name"] == "slice_id"]["value"].to_list()[0], dtype=np.int64)
            slice_ids = slice_ids[arg_sort_id]
            loads_per_slice = [np.sum(loads[slice_ids == i]) for i in range(2)]
            # breakpoint()
            action = baseline(loads_per_slice)  # agent policy that uses the observation and info
        elif eval_method == "sac":
            action = sac_agent.predict(obs)
        elif eval_method == "cql":
            action = agent.predict(obs)
        else:
            action = [0.5, 0.5]
            # action = [1/(len(config_json["env_config"]["slice_list"]) - 1)] * (len(config_json["env_config"]["slice_list"]) - 1)
            action = np.array(action)
        obs, reward, terminated, truncated, info = env.step(action)

        # If the environment is end, exit
        if terminated:
            break

        # If the epsiode is up (environment still running), then start another one
        if truncated:
            obs, info = env.reset()
            
        progress_bar.set_description("Step: {}, Reward: {:.2f}, Action: {}".format(step, reward, action))
        



