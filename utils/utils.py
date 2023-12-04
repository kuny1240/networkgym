
import numpy as np
import torch
from tqdm import tqdm



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


def evaluate(model, env, n_episodes):
    rewards = []
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        # breakpoint()
        pbar = tqdm(range(200))
        for _ in pbar:
            action = model.predict(obs, device = "cuda:0")
            action = np.exp(action)/np.sum(np.exp(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = truncated
            total_reward += reward
            pbar.set_description(f"Actions: {action}, reward: {reward}, total_reward: {total_reward}")
            if terminated:
                break
        rewards.append(total_reward)
        print("Episode: {}, Reward: {}".format(i, total_reward))
    

    avg_reward = sum(rewards) / n_episodes
    log_dict = info_to_log(info)
    return avg_reward, log_dict


def dataset_to_obs(dataset):
    
    slice_num = len(dataset["max_rates"])
    obs = np.zeros((5, slice_num))
    max_rate = np.min(dataset["max_rates"])
    for i in range(slice_num):
            # breakpoint()
        obs_slice = np.array([dataset["rates"][i]/dataset["loads"][i],
                              dataset["loads"][i]/max_rate,
                              dataset["rb_usages"][i]/100,
                              dataset["delay_violation_rates"][i]/100,
                              dataset["one_way_delays"][i]/1000])
        obs[:, i] = obs_slice
        
    return obs.flatten()

def info_to_dataset(info):
    
    data = dict()
    df = info["network_stats"]
    max_rates = np.array(df[df['name'] == "max_rate"]["value"].to_list()[0]) #keep the LTE rate.
    disturbed_user =  np.array(df[df['name'] == "slice_id"]["user"].to_list()[0])
    arg_sort_id = np.argsort(disturbed_user)
    loads = np.array(df[df["name"] == "tx_rate"]["value"].to_list()[0])
    rates = np.array(df[(df["cid"] == "All") & (df["name"] == "rate")]["value"].to_list()[0])
    rb_usages = np.array(df[df["name"] == "rb_usage"]["value"].to_list()[0])
    rb_usages = rb_usages[arg_sort_id]
    dvr = np.array(df[df["name"] == "delay_violation"]["value"].to_list()[0])
    dvr_2 = np.array(df[df["name"] == "delay_test_1_violation"]["value"].to_list()[0])
    dvr_3 = np.array(df[df["name"] == "delay_test_2_violation"]["value"].to_list()[0])
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
            
            
    slice_num = len(np.unique(slice_ids))
    max_rates_per_slice = [np.min(max_rates[slice_ids == i]) for i in range(slice_num)]
    loads_per_slice = [np.sum(loads[slice_ids == i]) for i in range(slice_num)]
    rates_per_slice = [np.sum(rates[slice_ids == i]) for i in range(slice_num)]
    rb_usages_per_slice = [np.sum(rb_usages[slice_ids == i]) for i in range(slice_num)]
    dvr_per_slice = [np.mean(dvr[slice_ids == i]) for i in range(slice_num)]
    dvr_2_per_slice = [np.mean(dvr_2[slice_ids == i]) for i in range(slice_num)]
    dvr_3_per_slice = [np.mean(dvr_3[slice_ids == i]) for i in range(slice_num)]
    owds_per_slice = [np.mean(owds[slice_ids == i]) for i in range(slice_num)]
    max_owds_per_slice = [np.max(max_owds[slice_ids == i]) for i in range(slice_num)]

    
    data.update(
        max_rates = max_rates_per_slice,
        loads = loads_per_slice,
        rates = rates_per_slice,
        rb_usages = rb_usages_per_slice,
        delay_violation_rates = dvr_per_slice,
        delay_violation_rates_2 = dvr_2_per_slice,
        delay_violation_rates_3 = dvr_3_per_slice,
        one_way_delays = owds_per_slice,
        max_one_way_delays = max_owds_per_slice,
    )
    
    return data


def vary_rewards(observations, 
                 p, 
                 alpha, 
                 delta):
    
    
    new_rewards = np.zeros(observations.shape[0])
    
    for i in range(3):
    
        new_rewards += p[i] *(observations[:, i*5] - alpha * observations[:, 2+ i*5] - delta * observations[:, 3+i*5])

    return new_rewards