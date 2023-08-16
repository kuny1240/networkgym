#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : adapter.py


import network_gym_client.adapter
import sys
from gymnasium import spaces
import numpy as np

class Adapter(network_gym_client.adapter.Adapter):
    """network_slicing environment adapter.

    Args:
        Adapter (network_gym_client.adapter.Adapter): the base class
    """
    def __init__(self, config_json):
        """Initialize the adapter.

        Args:
            config_json (json): the configuration file
        """
        super().__init__(config_json)
        self.env = "network_slicing"
        self.num_slices = len(self.config_json['env_config']['slice_list'])
        self.num_features = 3
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

        return spaces.Box(low=0, high=1, shape=(self.num_slices,), dtype=np.float32)

    
    #consistent with the get_observation function.
    def get_observation_space(self):
        """Get observation space for network_slicing env.
        
        Returns:
            spaces: observation spaces
        """
        
        # for network slicing, the user number is configured using the slice list. Cannot use the argument parser!

        return spaces.Box(low=0, high=1000,
                                shape=(self.num_features, self.num_users), dtype=np.float32)
    
    def get_observation(self, df):
        """Prepare observation for network_slicing env.

        This function should return the same number of features defined in the :meth:`get_observation_space`.

        Args:
            df (pd.DataFrame): the network stats measurements

        Returns:
            spaces: observation spaces
        """
        print (df)
        if not df.empty:
            self.end_ts = int(df['end_ts'][0])
        #data_recv_flat = df.explode(column=['user', 'value'])
        #print(data_recv_flat)

        df_rate = df[df['name'] == 'rate'].reset_index(drop=True) # get the rate
        df_rate = df_rate[df_rate['cid'] == 'All'].reset_index(drop=True).explode(column=['user', 'value']) #keep the flow rate.
        #print(df_rate)

        df_max_rate = df[df['name'] == 'max_rate'].reset_index(drop=True)
        df_phy_lte_max_rate = df_max_rate[df_max_rate['cid'] == 'LTE'].reset_index(drop=True).explode(column=['user', 'value']) #get the LTE max_rate
        df_phy_wifi_max_rate = df_max_rate[df_max_rate['cid'] == 'Wi-Fi'].reset_index(drop=True).explode(column=['user', 'value']) # get the Wi-Fi max rate

        #print(df_phy_lte_max_rate)
        #print(df_phy_wifi_max_rate)

        df_phy_lte_slice_id = df[df['name'] == 'slice_id'].reset_index(drop=True).explode(column=['user', 'value'])

        df_phy_lte_rb_usage = df[df['name'] == 'rb_usage'].reset_index(drop=True).explode(column=['user', 'value'])

        # if not empty and send to wanDB database
        self.wandb_log_buffer_append(self.df_to_dict(df_phy_wifi_max_rate, "max-wifi-rate"))
    
        self.wandb_log_buffer_append(self.df_to_dict(df_phy_lte_max_rate, "max-lte-rate"))

        dict_rate = self.df_to_dict(df_rate, 'rate')
        dict_rate["sum_rate"] = df_rate[:]["value"].sum()
        self.wandb_log_buffer_append(dict_rate)

        self.wandb_log_buffer_append(self.df_to_dict(df_phy_lte_slice_id, "lte-slice-id"))

        self.wandb_log_buffer_append(self.df_to_dict(df_phy_lte_rb_usage, "lte-rb-usage"))

        df_x_loc = df[df['name'] == 'x_loc'].reset_index(drop=True).explode(column=['user', 'value'])
        self.wandb_log_buffer_append(self.df_to_dict(df_x_loc, "x_loc"))

        df_y_loc = df[df['name'] == 'y_loc'].reset_index(drop=True).explode(column=['user', 'value'])
        self.wandb_log_buffer_append(self.df_to_dict(df_y_loc, "y_loc"))
        
        # Fill the empy features with -1
        phy_lte_max_rate = self.fill_empty_feature(df_phy_lte_max_rate, -1)
        phy_wifi_max_rate = self.fill_empty_feature(df_phy_wifi_max_rate, -1)
        flow_rate = self.fill_empty_feature(df_rate, -1)

        observation = np.vstack([phy_lte_max_rate, phy_wifi_max_rate, flow_rate])

        # add a check that the size of observation equals the prepared observation space.
        if len(observation) != self.num_features:
            sys.exit("The size of the observation and self.num_features is not the same!!!")
        return observation

    def get_policy(self, action):
        """Prepare the network policy for network_slicing env.

        Args:
            action (spaces): the action from RL agent

        Returns:
            json: the network policy
        """

        if action.size != self.num_slices:
            sys.exit("The action size: " + str(action.size()) +" does not match with the number of slices:" + self.num_slices)
        # you may also check other constraints for action... e.g., min, max.
        
        # TODO: the sum of action should be smaller than 1!!!! Therefore the sum of scaled_action is smaller than the rbg_num
        scaled_action= np.interp(action, (0, 1), (0, self.rbg_num/self.num_slices))
        scaled_action = np.round(scaled_action).astype(int) # force it to be an interger.

        # you can add more tags
        tags = {}
        tags["end_ts"] = self.end_ts
        tags["downlink"] = self.config_json["env_config"]["downlink"]
        tags["cid"] = 'LTE'

        tags["rb_type"] = "D"# dedicated RBG
        # this function will convert the action to a nested json format
        policy1 = self.get_nested_json_policy('rb_allocation', tags, np.zeros(len(scaled_action)), 'slice')
        tags["rb_type"] = "P"# prioritized RBG
        policy2 = self.get_nested_json_policy('rb_allocation', tags, scaled_action, 'slice')
        
        tags["rb_type"] = "S"# shared RBG
        policy3 = self.get_nested_json_policy('rb_allocation', tags, np.ones(len(scaled_action))*self.rbg_num, 'slice')

        policy = policy1 + policy2 + policy3

        print('Action --> ' + str(policy))
        return policy

    def get_reward(self, df):
        """Prepare reward for the network_slicing env.

        Args:
            df (pd.DataFrame): network stats measurements

        Returns:
            spaces: reward spaces
        """

        df_tx_rate = df[df['name'] == 'tx_rate'].reset_index(drop=True).explode(column=['user', 'value']) # get the rate

        df_phy_lte_rb_usage = df[df['name'] == 'rb_usage'].reset_index(drop=True).explode(column=['user', 'value'])

        df_phy_lte_slice_id = df[df['name'] == 'slice_id'].reset_index(drop=True).explode(column=['user', 'value'])

        user_to_slice_id = np.zeros(len(df_phy_lte_slice_id))
        df_phy_lte_slice_id = df_phy_lte_slice_id.reset_index()  # make sure indexes pair with number of rows
        for index, row in df_phy_lte_slice_id.iterrows():
            user_to_slice_id[row['user']] = int(row['value'])
        #print (user_to_slice_id)

        
        df_tx_rate['slice_id']=user_to_slice_id[df_tx_rate['user'].astype(int)]
        df_tx_rate['slice_value']= df_tx_rate.groupby(['slice_id'])['value'].transform('sum')

        df_slice_tx_rate = df_tx_rate.drop_duplicates(subset=['slice_id'])
        df_slice_tx_rate = df_slice_tx_rate.drop(columns=['user'])
        df_slice_tx_rate = df_slice_tx_rate.drop(columns=['value'])

        #print (df_tx_rate)
        print (df_slice_tx_rate)

        df_phy_lte_rb_usage['slice_id']=user_to_slice_id[df_phy_lte_rb_usage['user'].astype(int)]
        df_phy_lte_rb_usage['slice_value']= df_phy_lte_rb_usage.groupby(['slice_id'])['value'].transform('sum')

        df_slice_lte_rb_usage = df_phy_lte_rb_usage.drop_duplicates(subset=['slice_id'])
        df_slice_lte_rb_usage = df_slice_lte_rb_usage.drop(columns=['user'])
        df_slice_lte_rb_usage = df_slice_lte_rb_usage.drop(columns=['value'])

        #print (df_phy_lte_rb_usage)
        print (df_slice_lte_rb_usage)

        
        df_slice_tx_rate = self.slice_df_to_dict(df_slice_tx_rate, 'tx_rate')
        #print(df_slice_tx_rate)

        dict_slice_lte_rb_usage = self.slice_df_to_dict(df_slice_lte_rb_usage, 'rb_usage')
        #print(dict_slice_lte_rb_usage)

        self.wandb_log_buffer_append(df_slice_tx_rate)
        self.wandb_log_buffer_append(dict_slice_lte_rb_usage)

        #TODO: add a reward function for you customized env
        reward = 0

        # send info to wandb
        self.wandb_log_buffer_append({"reward": reward})

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
