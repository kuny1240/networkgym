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
import wandb
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
            pbar.set_description(f"Actions: {action}, reward: {reward:.2f}, total_reward: {total_reward:.2f}")
            if terminated:
                break
        rewards.append(total_reward)
        print("Episode: {}, Reward: {}".format(i, total_reward))

    avg_reward = sum(rewards) / n_episodes
    return avg_reward



def info_to_log(info):
    log = dict()
    df = info["network_stats"]
    max_rates = np.array(df[df['name'] == "max_rate"]["value"].to_list()[0]) #keep the LTE rate.
    disturbed_user =  np.array(df[df['name'] == "slice_id"]["user"].to_list()[0])
    arg_sort_id = np.argsort(disturbed_user)
    loads = np.array(df[df["name"] == "tx_rate"]["value"].to_list()[0])
    rates = np.array(df[(df["cid"] == "All") & (df["name"] == "rate")]["value"].to_list()[0])
    rb_usages = np.array(df[df["name"] == "rb_usage"]["value"].to_list()[0])
    rb_usages = rb_usages[arg_sort_id]
    delay_violation_rates = np.array(df[df["name"] == "delay_violation"]["value"].to_list()[0])
    slice_ids = np.array(df[df["name"] == "slice_id"]["value"].to_list()[0], dtype=np.int64)
    slice_ids = slice_ids[arg_sort_id]
    owds = np.array(df[(df["cid"] == "LTE") & (df["name"] == "owd")]["value"].to_list()[0])
    max_owds = np.array(df[(df["cid"] == "LTE") & (df["name"] == "max_owd")]["value"].to_list()[0])
    try:
        assert len(max_rates) == len(loads) == len(rates) == len(rb_usages) == len(delay_violation_rates) == len(slice_ids) == len(owds) == len(max_owds)
    except:
        length = len(slice_ids)
        # Create a list of the arrays
        arrays = [rates, owds, max_owds]
        # Loop over the list of arrays
        for i in range(len(arrays)):
            # Get the last element
            last_element = arrays[i][-1]
            # Calculate the number of times to extend
            num_times = length - len(arrays[i])
            # Extend the array and assign the result back to the original variable
            arrays[i] = np.append(arrays[i], [last_element]*num_times)

            # Unpack the list back to the original variables
            rates, owds, max_owds = arrays
    for i in np.unique(slice_ids):
        log.update({"traffic/slice_{}_load".format(i): np.sum(loads[slice_ids == i])})
        log.update({"traffic/slice_{}_rate".format(i): np.sum(rates[slice_ids == i])})
        log.update({"resource/slice_{}_rb_usage".format(i): np.sum(rb_usages[slice_ids == i])})
        log.update ({"delay/slice_{}_delay_violation_rate".format(i): np.mean(delay_violation_rates[slice_ids == i])})
        log.update({"delay/slice_{}_mean_owd".format(i): np.mean(owds[slice_ids == i])})
        log.update({"delay/slice_{}_max_owd".format(i): np.mean(max_owds[slice_ids == i])})
    return log

MODEL_SAVE_FREQ = 500
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 10
EVAL_EPI_PER_SESSION = 1

#Each session will only have 10 episodes, and each episode will have 100 steps.
# When one of the client is terminated, the training program will sample a new client and continue training until the total number of steps reaches the num_steps
# For the candidate slice list, we can use the following slice list:

# 31 users is a boundary where the network is saturated, and we can observe dramatic delay difference between different slice list.
# we have different balanced slice list, and different unbalanced slice list.
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



def main(agent_type:str,
         env_name:str,
         client_id = 0,
         hidden_dim = 64,
         steps_per_episode = 100,
         episode_per_session = 5,
         num_steps = 12000,
         random_seed = 1
         ):

    # client_id = 1
    # env_name = "network_slicing"
    storage_ver = 0
    config_json = load_config_file(env_name)
    init_list = random.sample(slice_lists, 1)
    config_json["env_config"]["slice_list"] = init_list[0]
    config_json["env_config"]["random_seed"] = random_seed
    train_random_seed = random_seed
    config_json["rl_config"]["agent"] = agent_type
    config_json["env_config"]["steps_per_episode"] = steps_per_episode
    config_json["env_config"]["episodes_per_session"] = episode_per_session
    wandb.init(project = "network_gym_client", name = f"network_slicing_{agent_type}_training_{random_seed}", config = config_json)
    # Create the environment
    env = NetworkGymEnv(client_id, config_json, log=False) # make a network env using pass client id and configure file arguements.
    normalized_env = NormalizeObservation(env) # normalize the observation
    # normalized_env = env

    num_steps = num_steps
    # breakpoint()
    obs, info = normalized_env.reset()
    agent = SACAgent(state_dim=obs.shape[0], 
                        action_dim=env.action_space.shape[0], 
                        actor_lr=6e-4, 
                        critic_lr=3e-3,
                        action_high=1,
                        action_low=0,
                        hidden_dim=hidden_dim)
    eva_agent = copy.deepcopy(agent)
    buffer = ReplayBuffer(max_size=1000000, obs_shape=obs.shape[0], n_actions=env.action_space.shape[0])
    epsilon = 1.0

    num_episodes = 0
    progress_bar = tqdm(range(num_steps))
    best_eval_reward = -np.inf
    # Training loop
    # Evaluate every 500 steps, same as model saving frequency
    for step in progress_bar:
        
        
        if random.random() < epsilon:
            action = normalized_env.action_space.sample()
        else:
                # breakpoint()
            action = agent.predict(obs)
        if np.sum(action) >= 1: # Illegal action
            action = np.exp(action)/np.sum(np.exp(action))
        # action = np.exp(action)/np.sum(np.exp(action))  # softmax
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        nxt_obs, reward, terminated, truncated, info = normalized_env.step(action=action)
        buffer.store(obs, action, reward, nxt_obs, truncated)
        obs = nxt_obs
        log = info_to_log(info)
        wandb.log({"rewards/training_reward": reward})
        wandb.log(log)
        
        
        if buffer.mem_cntr > 256:
            training_batch = buffer.sample(256)
            # breakpoint()
            agent.learn(*training_batch)

        # If the environment is end, exit
        if terminated:
            print("Environment terminated, sampling a new environment...")
            if num_steps - step < steps_per_episode * episode_per_session:
                break
            else:
                print("Step: {}, Saving model...".format(step))
                agent.save("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
                eva_agent.load("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
                eva_agent.actor.eval()
                config_json["env_config"]["episodes_per_session"] = EVAL_EPI_PER_SESSION
                random_seed = 1
                avg_reward = 0
                for slice_list in slice_lists:
                    config_json["env_config"]["random_seed"] = random_seed
                    config_json["env_config"]["slice_list"] = slice_list
                    eval_env = NetworkGymEnv(client_id, config_json, log=False)
                    normalized_eval_env = NormalizeObservation(eval_env)
                    env_reward = evaluate(eva_agent, normalized_eval_env, n_episodes=1)
                    avg_reward += env_reward
                avg_reward /= len(slice_lists)
                art = wandb.Artifact(f"{agent_type}-nn-{wandb.run.id}", type="model")
                art.add_file("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.save("./models/sac_model_{}_{}_{}_best.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed))
                    wandb.log_artifact(art, aliases=["latest", "best"])
                else:
                    agent.save("./models/sac_model_{}_{}_{}_latest.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed))
                    wandb.log_artifact(art)
                print("Step: {}, Eval Reward: {}".format(step, avg_reward))
                wandb.log({"rewards/eval_reward": avg_reward})
                buffer.save_buffer("./dataset/offline_data_heavy_traffic_ver1.h5")
                storage_ver += 1
                slice_list = random.sample(slice_lists, 1)
                config_json["env_config"]["slice_list"] = slice_list[0]
                config_json["env_config"]["random_seed"] = random.randint(0, 1000)
                config_json["env_config"]["episodes_per_session"] = episode_per_session
                env = NetworkGymEnv(client_id, config_json, log=False) # make a network env using pass client id and configure file arguements.
                normalized_env = NormalizeObservation(env) # normalize the observation
                obs, info = normalized_env.reset()
                print(f"New environment created, slice list: {slice_list[0]}, random seed: {config_json['env_config']['random_seed']}")
                continue

        # If the epsiode is up (environment still running), then start another one
        if truncated:
            obs, info = normalized_env.reset()
            obs = torch.Tensor(obs)
            epsilon = max(epsilon*0.95, 0.01)
            num_episodes += 1

        # if (step + 1) % MODEL_SAVE_FREQ == 0:
        #     print("Step: {}, Saving model...".format(step))
        #     agent.save("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
        #     eva_agent.load("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
        #     eva_agent.actor.eval()
        #     config_json["env_config"]["episodes_per_session"] = EVAL_EPI_PER_SESSION
        #     random_seed = 1
        #     avg_reward = 0
        #     for slice_list in slice_lists:
        #         config_json["env_config"]["random_seed"] = random_seed
        #         config_json["env_config"]["slice_list"] = slice_list
        #         eval_env = NetworkGymEnv(1, config_json, log=False)
        #         normalized_eval_env = NormalizeObservation(eval_env)
        #         env_reward = evaluate(eva_agent, normalized_eval_env, n_episodes=1)
        #         avg_reward += env_reward
        #     avg_reward /= len(slice_lists)
        #     art = wandb.Artifact(f"{agent_type}-nn-{wandb.run.id}", type="model")
        #     art.add_file("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
        #     if avg_reward > best_eval_reward:
        #         best_eval_reward = avg_reward
        #         agent.save("./models/sac_model_{}_{}_{}_best.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed))
        #         wandb.log_artifact(art, aliases=["latest", "best"])
        #     else:
        #         agent.save("./models/sac_model_{}_{}_{}_latest.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed))
        #         wandb.log_artifact(art)
        #     print("Step: {}, Eval Reward: {}".format(step, avg_reward))
        #     wandb.log({"rewards/eval_reward": avg_reward})
        #     buffer.save_buffer("./dataset/offline_data_heavy_traffic_ver1.h5")
        #     storage_ver += 1
            
        
        progress_bar.set_description("Step: {}, Reward: {:.3f}, Action: {}".format(step, reward, action))


if __name__ == "__main__":
    fire.Fire(main)


    

