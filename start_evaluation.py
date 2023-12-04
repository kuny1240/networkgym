#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : start_evaluation.py
import numpy as np
import wandb
import pandas as pd
from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from gymnasium.wrappers import NormalizeObservation
from CleanRL_agents.sac import SACAgent
from CORL_agents import CQL
# from CQL_agent import ContinuousCQL as CQL
from tqdm import tqdm
from utils.utils import *
import torch
import json

client_id = 0
env_name = "network_slicing"
config_json = load_config_file(env_name)

num_steps = 200
steps_per_episode = 200
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


def baseline_delay(delays):
    '''
    propotionally allocate rb resource to slices accordling to their traffic load
    '''
    
    total_delay = sum(np.exp(delays))
    
    action = [np.exp(delay)/total_delay for delay in delays]

    return np.array(action)


slice_lists = [
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

# slice_lists = [
#     [
#         {"num_users":20,"dedicated_rbg":0,"prioritized_rbg":12,"shared_rbg":25},
#         {"num_users":5,"dedicated_rbg":0,"prioritized_rbg":13,"shared_rbg":25},
#         {"num_users":6,"dedicated_rbg":0,"prioritized_rbg":0,"shared_rbg":25}
#     ],
    
# ]


config_json["env_config"]["steps_per_episode"] = steps_per_episode
config_json["env_config"]["episodes_per_session"] = episode_per_session
labels = ["6_20","11_15","13_13","15_11","20_6"]
# labels = ["20_5"]
# eval_methods = ["cql", "cql_throughput", "cql_res"]
eval_methods = ["baseline_delay", "cql", "baseline", "sac"]
# eval_methods = ["sac"]
# eval_methods = ["no_scheduler"]
random_seeds = [233, 1240]
# random_seeds = [1240]

metrics_data = {method: {} for method in eval_methods}
for random_seed in random_seeds:
    config_json["env_config"]["random_seed"] = random_seed
    for i, slice_list in enumerate(slice_lists):
        config_json["env_config"]["slice_list"] = slice_list
        label = labels[i]
        for eval_method in eval_methods:
            config_json["rl_config"]["agent"] = eval_method
            env = NetworkGymEnv(0, config_json, log=False) # make a network env using pass client id and configure file arguements.
            # normalized_env = NormalizeObservation(env) # normalize the observation
            normalized_env = env
            metrics_df = pd.DataFrame()
            
            run = wandb.init(project="netgym_eval_diffsla", 
                            group=f"{env_name}_{label}", 
                            tags=[eval_method, label], 
                            name=f"{eval_method}_{label}_scen_1", 
                            config=config_json)

        
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
                sac_agent.load("./models/sac_model_3_weighted_1240_best.ckpt")
                sac_agent.actor.eval()
            elif eval_method == "cql":
                agent = CQL(state_dim=15, action_dim=2, hidden_dim=64, target_entropy=-2,
                            q_n_hidden_layers=1, max_action=1, qf_lr=3e-4, policy_lr=6e-5,device="cuda:0")
                agent.load("./models/cql_dataset_mixed_best.pt")
                agent.actor.eval()
            elif eval_method == "cql_throughput":
                agent = CQL(state_dim=15, action_dim=2, hidden_dim=64, target_entropy=-2,
                            q_n_hidden_layers=1, max_action=1, qf_lr=3e-4, policy_lr=6e-5,device="cuda:0")
                agent.load("./models/cql_dataset_mixed_throughput_best.pt")
                agent.actor.eval()
            elif eval_method == "cql_res":
                agent = CQL(state_dim=15, action_dim=2, hidden_dim=64, target_entropy=-2,
                            q_n_hidden_layers=1, max_action=1, qf_lr=3e-4, policy_lr=6e-5,device="cuda:0")
                agent.load("./models/cql_dataset_mixed_res_best.pt")
                agent.actor.eval()
                
            progress_bar = tqdm(range(num_steps))
            total_reward = 0
            eval_num = len(slice_lists)
            avg_dvr = np.zeros(3)
            avg_rbu = np.zeros(3)
            log_dict = dict()
            for step in progress_bar: 
                df = info["network_stats"]
                disturbed_user =  np.array(df[df['name'] == "slice_id"]["user"].to_list()[0])
                arg_sort_id = np.argsort(disturbed_user)
                loads =  np.array(df[df["name"] == "tx_rate"]["value"].to_list()[0])
                rates = np.array(df[(df["cid"] == "All") & (df["name"] == "rate")]["value"].to_list()[0])
                rb_usages = np.array(df[df["name"] == "rb_usage"]["value"].to_list()[0])
                rb_usages = rb_usages[arg_sort_id]
                delay_violation_rates = np.array(df[df["name"] == "delay_violation"]["value"].to_list()[0])
                slice_ids = np.array(df[df["name"] == "slice_id"]["value"].to_list()[0], dtype=np.int64)
                slice_ids = slice_ids[arg_sort_id]
                dvr_per_slice = [np.mean(delay_violation_rates[slice_ids == i]) for i in range(3)]
                rbu_per_slice = [np.sum(rb_usages[slice_ids == i]) for i in range(3)]
                # breakpoint()
                if eval_method == "baseline":
                    loads_per_slice = [np.sum(loads[slice_ids == i]) for i in range(2)]
                    dvr_loads = dvr_per_slice[:2]
                    action = baseline(loads_per_slice)  # agent policy that uses the observation and info
                elif eval_method == "baseline_delay":
                    dvr_loads = dvr_per_slice[:2]
                    action = baseline_delay(dvr_loads)

                elif eval_method == "sac":
                    action = sac_agent.predict(obs)
                    action = np.exp(action)/np.sum(np.exp(action))
                elif "cql" in eval_method:
                    action = agent.predict(obs)
                    action = np.exp(action)/np.sum(np.exp(action))
                else:
                    action = [0, 0]
                    action = np.array(action)
                    
                obs, reward, terminated, truncated, info = normalized_env.step(action)
                total_reward += reward
                avg_dvr += np.array(dvr_per_slice)
                avg_rbu += np.array(rbu_per_slice)
                logs = info_to_log(info)
                # print(logs)
                metrics_df = metrics_df.append(logs, ignore_index=True)
                logs.update({"step": step})
                
                
                run.log(logs)
                # If the environment is end, exit
                if terminated:
                    break

                # If the epsiode is up (environment still running), then start another one
                if truncated:
                    obs, info = env.reset()
                    
                progress_bar.set_description("Step: {}, Total_reward: {:.2f}, Action: {}".format(step, total_reward, action))
            metrics_data[eval_method][label] = metrics_df
            avg_dvr /= num_steps
            avg_rbu /= num_steps
            log_dict[f"eval/total_reward"] = total_reward
            for j in range(len(slice_list)):
                log_dict[f"eval/Mean_delay_violation_rate_slice_{j}"] = avg_dvr[j]
                log_dict[f"eval/Mean_resource_block_usage_slice_{j}"] = avg_rbu[j]
            run.log(log_dict)
            run.finish()

        
wandb.finish()