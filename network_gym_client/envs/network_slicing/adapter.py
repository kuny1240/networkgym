#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : adapter.py


import network_gym_client.adapter
import sys
import pandas as pd
from gymnasium import spaces
import numpy as np

class Adapter(network_gym_client.adapter.Adapter):
    """network_slicing environment adapter.

    Args:
        Adapter (network_gym_client.adapter.Adapter): the base class
    """
    def __init__(self, config_json):
        """Initilize adapter.
        """
        super().__init__(config_json)
        self.env = "network_slicing"
        self.num_slices = len(self.config_json['env_config']['slice_list'])
        self.num_features = 5 # {slice_rate/slice_load, slice_rb_usage, delay_violation_rate, max_delay, mean_delay}
        self.end_ts = 0

        self.num_users = 0
        for item in self.config_json['env_config']['slice_list']:
            self.num_users += item['num_users']
        self.config_json['env_config']['num_users'] = self.num_users
        
        rbg_size = self.get_rbg_size(self.config_json['env_config']['LTE']['resource_block_num'])
        self.rbg_num = self.config_json['env_config']['LTE']['resource_block_num']/rbg_size

        if config_json['env_config']['env'] != self.env:
            sys.exit("[ERROR] wrong environment helper. config file environment: " + str(config_json['env_config']['env']) + " helper environment: " + str(self.env))

    def get_action_space(self):
        """Get action space for network_slicing env.

        Returns:
            spaces: action spaces
        """
        if (self.env == self.config_json['env_config']['env']):
            return spaces.Box(low=0, high=1, shape=(len(self.config_json['env_config']['slice_list']),), dtype=np.float32)
        else:
            sys.exit("[ERROR] wrong environment or RL agent.")
    
    #consistent with the prepare_observation function.
    def get_observation_space(self):
        """Get observation space for network_slicing env.
        
        Returns:
            spaces: observation spaces 
        """
        
        # for network slicing, the user number is configured using the slice list. Cannot use the argument parser!
        # users are not directly linked with the slices thus cannot be directly used as in the observation space
        # TODO: Associate users with slices and have a more detailed observation space


       
        # num_users = 0
        # if (self.config_json['env_config'].get('num_users') is None):
        #     for item in self.config_json['env_config']['slice_list']:
        #         num_users += item['num_users']
        #         self.config_json['env_config']['num_users'] = num_users
        # else:
        #     print(self.config_json['env_config']['num_users'])
        #     num_users = self.config_json['env_config']['num_users']

        obs_space = None

        obs_space =  spaces.Box(low=0, high=1000,
                                            shape=(self.num_features,len(self.config_json['env_config']['slice_list']),), dtype=np.float32)
        return obs_space
    
    
    def df_to_observation(self, df):
        '''
        Convert the list of dataframes to a numpy array of observations
        
        0               ap_id
        1     delay_violation
        2             max_owd
        3             max_owd
        4            max_rate
        5      measurement_ok
        6       missed_action
        7                 owd
        8                 owd
        9         qos_marking
        10           qos_rate
        11           qos_rate
        12               rate
        13               rate
        14           rb_usage
        15           slice_id
        16      traffic_ratio
        17            tx_rate
        18              x_loc
        '''
        # Extract necessary dataframes from the list
        
        max_rates = np.array(df[df['name'] == "max_rate"]["value"].to_list()[0]) #keep the LTE rate.
        loads = np.array(df[df["name"] == "tx_rate"]["value"].to_list()[0])
        rates = np.array(df[(df["cid"] == "LTE") & (df["name"] == "rate")]["value"].to_list()[0])
        rb_usages = np.array(df[df["name"] == "rb_usage"]["value"].to_list()[0])
        delay_violation_rates = np.array(df[df["name"] == "delay_violation"]["value"].to_list()[0])
        slice_ids = np.array(df[df["name"] == "slice_id"]["value"].to_list()[0], dtype=np.int64)
        owds = np.array(df[(df["cid"] == "LTE") & (df["name"] == "owd")]["value"].to_list()[0])
        max_owds = np.array(df[(df["cid"] == "LTE") & (df["name"] == "max_owd")]["value"].to_list()[0])

        
        # First, ensure all arrays have the same length
        assert len(max_rates) == len(loads) == len(rates) == len(rb_usages) == len(delay_violation_rates) == len(slice_ids) == len(owds) == len(max_owds)

        # Create a DataFrame

        # Group by slice_id and compute the sum/mean
        obs = np.zeros((self.num_features, len(self.config_json['env_config']['slice_list'])))
        for i in np.unique(slice_ids):
            obs_slice = np.array([np.sum(rates[slice_ids == i])/np.sum(loads[slice_ids == i]), 
                                  np.sum(rb_usages[slice_ids == i])/100, 
                                  np.mean(delay_violation_rates[slice_ids == i])/100, 
                                  np.max(max_owds[slice_ids == i])/self.config_json['env_config']['qos_requirement']['delay_bound_ms'], 
                                  np.mean(owds[slice_ids == i])])/self.config_json['env_config']['qos_requirement']['delay_bound_ms']
            obs[:, i] = obs_slice
        
        
        # Convert the final dataframe to numpy array
        result = obs.reshape(-1)

        return result
    
    def get_observation(self, df):
        """Prepare observation for network_slicing env.

        This function should return the same number of features defined in the :meth:`get_observation_space`.

        Args:
            df (pandas.DataFrame): the network stats measurements

        Returns:
            spaces: observation spaces
        """

        observation = self.df_to_observation(df)
        
        return observation

    def prepare_policy(self, action):
        """Prepare the network policy for network_slicing env.

        Args:
            action (spaces): the action from RL agent

        Returns:
            json: the network policy
        """

        rbg_size = self.get_rbg_size(self.config_json['env_config']['LTE']['resource_block_num'])
        rbg_num = self.config_json['env_config']['LTE']['resource_block_num']/rbg_size
        
        if np.sum(action) > 1: # Illegal action
            action = np.exp(action)/sum(np.exp(action))
            
        scaled_action= np.interp(action, (0, 1), (0, rbg_num/len(self.config_json['env_config']['slice_list'])))
        #scaled_action= np.interp(action, (0, 1), (0, rbg_num))

        # Round the scaled subtracted action to integers
        rounded_scaled_action = np.round(scaled_action).astype(int)

        print("action --> " + str(rounded_scaled_action))
        action_list = []

        for slice_id in range(len(self.config_json['env_config']['slice_list'])):
            action_list.append({"slice":int(slice_id),"D":int(rounded_scaled_action[slice_id]),"P":int(0),"S":int(50)})

        # the unit of the action is resource block group number, not resource block!!!
        # please make sure the sum of the dedicated ("D") and priorititized ("P") resouce block group # is smaller than total resource block group number.
        return action_list
    

    def prepare_reward(self, df):
        """Prepare reward for the network_slicing env.

        Args:
            df (list[pandas.DataFrame]): network stats measurements

        Returns:
            spaces: reward spaces
        """

        
        observation = self.df_to_observation(df)
        alpha = .25
        gamma = 2

        # Pivot the DataFrame to extract "Wi-Fi" and "LTE" values
        # df_pivot = df_owd.pivot_table(index="user", columns="cid", values="value", aggfunc="first")[["Wi-Fi", "LTE"]]

        # Rename the columns to "wi-fi" and "lte"
        # df_pivot.columns = ["wi-fi", "lte"]

        # Sort the index in ascending order
        # df_pivot.sort_index(inplace=True)

        #check reward type, TODO: add reward combination of delay and throughput from network util function
        num_slices = len(self.config_json['env_config']['slice_list'])
        per_slice_achieved = observation[:num_slices]
        per_slice_rb_usage = observation[num_slices:2*num_slices]
        per_slice_delay_violation_rate = observation[2*num_slices:3*num_slices]
        per_slice_max_delay = observation[3*num_slices:4*num_slices]
        per_slice_mean_delay = observation[4*num_slices:5*num_slices]
        if self.config_json['env_config']['rl_config']['reward_type'] == 'default':
            reward = sum(per_slice_achieved - alpha * per_slice_rb_usage - gamma * per_slice_delay_violation_rate)
        elif self.config_json['env_config']['rl_config']['reward_type'] == 'delay':
            reward = -sum(per_slice_delay_violation_rate)
        elif self.config_json['env_config']['rl_config']['reward_type'] == 'throughput':
            reward = sum(per_slice_achieved)
        elif self.config_json['env_config']['rl_config']['reward_type'] == 'slice_wise':
            weight = np.ones_like(per_slice_achieved)
            reward = np.matmul(weight.T,(per_slice_achieved, per_slice_rb_usage, per_slice_delay_violation_rate))
        elif self.config_json['env_config']['rl_config']['reward_type'] == 'delay_threshold':
            thresholds = self.config_json['env_config']['delay_thresholds']
            for i in range(num_slices):
                reward += 3 if per_slice_mean_delay[i] <= thresholds[0] else 2 if per_slice_mean_delay[i] <= thresholds[1] else 1
        else:
            Warning("[WARNING] reward fucntion not defined yet")
            
        # TODO: Detailed reward function in the future
        # print("[WARNING] reward fucntion not defined yet")
        keys = [
            "rx/tx_rate_ratio",
            "rb_usage",
            "delay_violation_rate",
        ]
        slice_key = "slice_id"
        slice_ids = np.array(df[df["name"] == slice_key]["value"].to_list()[0], dtype=np.int64)
        
        for i,key in enumerate(keys):
            dict_slice = dict(zip([slice_key, key],[slice_ids, observation[num_slices*i:num_slices*(i+1)]]))
            if not self.wandb_log_info:
                self.wandb_log_info = dict_slice
            else:
                self.wandb_log_info.update(dict_slice)
        # if not self.wandb_log_info:
        #     self.wandb_log_info = dict_slice_load
        # else:
        #     self.wandb_log_info.update(dict_slice_load)
        # self.wandb_log_info.update(dict_owd)
        # self.wandb_log_info.update(dict_slice_lte_max_rate)
        # self.wandb_log_info.update(dict_lte_slice_rate)
        # self.wandb_log_info.update(dict_lte_qos_slice_rate)

        # self.wandb_log_info.update(dict_slice_delay_violation)
        # self.wandb_log_info.update(dict_slice_lte_rb_usage)
        
        self.wandb_log_info.update({"reward": reward, "avg_delay": per_slice_mean_delay.mean(), "max_delay": per_slice_max_delay.max()})

        return reward

    def slice_df_to_dict(self, df, description):
        """Convert the dataformat from dataframe to dict.

        Args:
            df (pandas.DataFrame): input dataframe
            description (str): description for the data

        Returns:
            dict: output data
        """
        df_cp = df.copy()
        df_cp['slice_id'] = df_cp['slice_id'].map(lambda u: f'slice_{int(u)}_'+description)
        # Set the index to the 'user' column
        df_cp = df_cp.set_index('slice_id')
        # Convert the DataFrame to a dictionary
        data = df_cp['slice_value'].to_dict()
        return data
    
    def get_rbg_size (self, bandwidth):
        """Compute the resource block group size based on the bandwith (RB number).

        This code is coppied from ns3.
        PF type 0 allocation RBG

        Args:
            bandwidth (int): the resouce block number

        Returns:
            int: the resouce block group size
        """
        # PF type 0 allocation RBG
        PfType0AllocationRbg = [10,26,63,110]      # see table 7.1.6.1-1 of 36.213

        for i in range(len(PfType0AllocationRbg)):
            if (bandwidth < PfType0AllocationRbg[i]):
                return (i + 1)
        return (-1)
