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

MODEL_SAVE_FREQ = 500
LOG_INTERVAL = 10
NUM_OF_EVALUATE_EPISODES = 10
EVAL_EPI_PER_SESSION = 1

def main(agent_type:str,
         env_name:str,
         client_id = 0,
         hidden_dim = 64,
         steps_per_episode = 100,
         episode_per_session = 60,
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
    # Create the environment
    env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id and configure file arguements.
    normalized_env = NormalizeObservation(env) # normalize the observation
   

    num_steps = steps_per_episode * episode_per_session
    # breakpoint()
    obs, info = normalized_env.reset()
    agent = SACAgent(state_dim=obs.shape[0], 
                        action_dim=env.action_space.shape[0], 
                        actor_lr=6e-5, 
                        critic_lr=3e-4,
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
            agent.save("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
            eva_agent.load("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
            eva_agent.actor.eval()
            config_json["env_config"]["episodes_per_session"] = EVAL_EPI_PER_SESSION
            random_seeds = [1, 21, 35]
            avg_reward = 0
            for random_seed in random_seeds:
                config_json["env_config"]["random_seed"] = random_seed
                eval_env = NetworkGymEnv(1, config_json, log=False)
                normalized_eval_env = NormalizeObservation(eval_env)
                env_reward = evaluate(eva_agent, normalized_eval_env, n_episodes=1)
                avg_reward += env_reward
            avg_reward /= len(random_seeds)
            art = normalized_env.adapter.wandb.Artifact(f"{agent_type}-nn-{env.adapter.wandb.run.id}", type="model")
            art.add_file("./models/sac_model_{}_{}_{}_ver{}.ckpt".format(len(config_json["env_config"]["slice_list"]), config_json["rl_config"]["reward_type"], train_random_seed, storage_ver))
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                normalized_env.adapter.wandb.log_artifact(art, aliases=["latest", "best"])
            else:
                normalized_env.adapter.wandb.log_artifact(art)
            print("Step: {}, Eval Reward: {}".format(step, avg_reward))
            normalized_env.adapter.wandb.log({"eval_avg_reward": avg_reward})
            buffer.save_buffer("./dataset/offline_data_heavy_traffic.h5")
            storage_ver += 1
            
        
        progress_bar.set_description("Step: {}, Reward: {}, Action: {}".format(step, reward, action))


if __name__ == "__main__":
    fire.Fire(main)


    

