# A modified Replay buffer with a save option to save all state-action pairs inside the Buffer to a .h5 file
# The buffer needs to store data for RL purpose with observations, actions, rewards, dones and next_observations.


import numpy as np
import pandas as pd
import h5py
import os
import torch

class ReplayBuffer:
    def __init__(self, max_size, obs_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        # States for RL
        self.state_memory = np.zeros((self.mem_size, obs_shape))
        self.new_state_memory = np.zeros((self.mem_size, obs_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int64)
        # States for raw measurements
        slice_num = 3
        self.max_rates = np.zeros((self.mem_size, slice_num))
        self.loads = np.zeros((self.mem_size, slice_num))
        self.rates = np.zeros((self.mem_size, slice_num))
        self.rb_usages = np.zeros((self.mem_size, slice_num))
        self.delay_violation_rates = np.zeros((self.mem_size, slice_num))
        self.delay_violation_rates_2 = np.zeros((self.mem_size, slice_num))
        self.delay_violation_rates_3 = np.zeros((self.mem_size, slice_num))
        self.one_way_delays = np.zeros((self.mem_size, slice_num))
        self.max_one_way_delays = np.zeros((self.mem_size, slice_num))
        
        
    def store_raw_measurements(self, raw_data_dict):
        '''
        store per slice raw measurements to the buffer
        '''
        index = self.mem_cntr % self.mem_size
        self.max_rates[index] = raw_data_dict["max_rates"]
        self.loads[index] = raw_data_dict["loads"]
        self.rates[index] = raw_data_dict["rates"]
        self.rb_usages[index] = raw_data_dict["rb_usages"]
        self.delay_violation_rates[index] = raw_data_dict["delay_violation_rates"]
        self.delay_violation_rates_2[index] = raw_data_dict["delay_violation_rates_2"]
        self.delay_violation_rates_3[index] = raw_data_dict["delay_violation_rates_3"]
        self.one_way_delays[index] = raw_data_dict["one_way_delays"]
        self.max_one_way_delays[index] = raw_data_dict["max_one_way_delays"]

    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    
    def save_raw_data(self, file_name):
        
        min_size = min(self.mem_cntr, self.mem_size)
        if os.path.exists(file_name):
            with h5py.File(file_name, 'a') as f:
                for dataset_name, memory in [('max_rates', self.max_rates[:min_size]),
                                             ('loads', self.loads[:min_size]),
                                             ('rates', self.rates[:min_size]),
                                             ('rb_usages', self.rb_usages[:min_size]),
                                             ('delay_violation_rates', self.delay_violation_rates[:min_size]),
                                             ('delay_violation_rates_2', self.delay_violation_rates_2[:min_size]),
                                             ('delay_violation_rates_3', self.delay_violation_rates_3[:min_size]),
                                             ('one_way_delays', self.one_way_delays[:min_size]),
                                             ('max_one_way_delays', self.max_one_way_delays[:min_size])]:
                    f[dataset_name].resize((f[dataset_name].shape[0] + min_size), axis=0)
                    f[dataset_name][-min_size:] = memory[:min_size]
        else:
            with h5py.File(file_name, 'a') as f:
                for dataset_name, memory in  [('max_rates', self.max_rates[:min_size]),
                                                ('loads', self.loads[:min_size]),
                                                ('rates', self.rates[:min_size]),
                                                ('rb_usages', self.rb_usages[:min_size]),
                                                ('delay_violation_rates', self.delay_violation_rates[:min_size]),
                                                ('delay_violation_rates_2', self.delay_violation_rates_2[:min_size]),
                                                ('delay_violation_rates_3', self.delay_violation_rates_3[:min_size]),
                                                ('one_way_delays', self.one_way_delays[:min_size]),
                                                ('max_one_way_delays', self.max_one_way_delays[:min_size])]:
                    max_shape = (None,1e7)
                    chunks_shape = (1, memory.shape[1])  # This sets the chunk shape    
                    f.create_dataset(dataset_name, data=memory[:min_size],maxshape = max_shape,chunks=chunks_shape)
                
                
                
    def load_raw_data(self, file_name):
        
        
        with h5py.File(file_name, "r") as f:
             load_data_len = len(f['loads'][:])
             if self.mem_cntr + load_data_len > self.mem_size:
                 raise ValueError("The buffer is full, please create a new buffer.")
             else:  
                self.max_rates[self.mem_cntr:self.mem_cntr+load_data_len] = f['max_rates'][:]
                self.loads[self.mem_cntr:self.mem_cntr+load_data_len] = f['loads'][:]
                self.rates[self.mem_cntr:self.mem_cntr+load_data_len] = f['rates'][:]
                self.rb_usages[self.mem_cntr:self.mem_cntr+load_data_len] = f['rb_usages'][:]
                self.delay_violation_rates[self.mem_cntr:self.mem_cntr+load_data_len] = f['delay_violation_rates'][:]
                self.delay_violation_rates_2[self.mem_cntr:self.mem_cntr+load_data_len] = f['delay_violation_rates_2'][:]
                self.delay_violation_rates_3[self.mem_cntr:self.mem_cntr+load_data_len] = f['delay_violation_rates_3'][:]
                self.one_way_delays[self.mem_cntr:self.mem_cntr+load_data_len] = f['one_way_delays'][:]
                self.max_one_way_delays[self.mem_cntr:self.mem_cntr+load_data_len] = f['max_one_way_delays'][:]
            
        

    def save_buffer(self, file_name):
        '''
        If the file_name exists, then append the buffer to the file,
        otherwise, create a new file.
        '''
        min_size = min(self.mem_cntr, self.mem_size)
        if os.path.exists(file_name):
            with h5py.File(file_name, 'a') as f:
                for dataset_name, memory in [('states', self.state_memory[:min_size]), 
                                            ('actions', self.action_memory[:min_size]), 
                                            ('rewards', self.reward_memory[:min_size]), 
                                            ('next_states', self.new_state_memory[:min_size]), 
                                            ('dones', self.terminal_memory[:min_size])]:
                    f[dataset_name].resize((f[dataset_name].shape[0] + min_size), axis=0)
                    f[dataset_name][-min_size:] = memory[:min_size]
        else:
            with h5py.File(file_name, 'w') as f:
                # Assuming the datasets are 2D arrays. Adjust the shape, maxshape, and chunks as needed.
                
                for dataset_name, memory in [('states', self.state_memory[:min_size]), 
                                            ('actions', self.action_memory[:min_size]), 
                                            ('rewards', self.reward_memory[:min_size]), 
                                            ('next_states', self.new_state_memory)[:min_size], 
                                            ('dones', self.terminal_memory[:min_size])]:
                    
                    
                    if memory.ndim == 1:
                        max_shape = (None,)
                        chunks_shape = (1,)
                    else:
                        max_shape = (None,1e7)
                        chunks_shape = (1, memory.shape[1])  # This sets the chunk shape    
                    f.create_dataset(dataset_name, data=memory[:min_size],maxshape = max_shape,chunks=chunks_shape)



    def load_buffer(self, file_name):
        '''
        Load the buffer from the file_name,
        if the buffer is not empty, then append the loaded data to the buffer,
        otherwise, create a new buffer.
        '''
        
        with h5py.File(file_name, "r") as f:
             #Detect none zero data
             idx = (f['rewards'][:] != 0)
             load_data_len = idx.sum()
             if self.mem_cntr + load_data_len > self.mem_size:
                 raise ValueError("The buffer is full, please create a new buffer.")
             else:  
                self.state_memory[self.mem_cntr:self.mem_cntr+load_data_len] = f['states'][:load_data_len]
                self.action_memory[self.mem_cntr:self.mem_cntr+load_data_len] = f['actions'][:load_data_len]
                self.reward_memory[self.mem_cntr:self.mem_cntr+load_data_len] = f['rewards'][:load_data_len]
                self.new_state_memory[self.mem_cntr:self.mem_cntr+load_data_len] = f['next_states'][:load_data_len]
                self.terminal_memory[self.mem_cntr:self.mem_cntr+load_data_len] = f['dones'][:load_data_len]
             
             
             self.mem_cntr += load_data_len
        
        # with h5py.File(file_name, 'r') as f:
        #     self.state_memory = f['states'][:]
        #     self.action_memory = f['actions'][:]
        #     self.reward_memory = f['rewards'][:]
        #     self.new_state_memory = f['next_states'][:]
        #     self.terminal_memory = f['dones'][:]
        #     self.mem_cntr = min(self.mem_size, len(f['states']))
            
            
    def nomarlize_states(self):
        
        state_mean = np.mean(self.state_memory, axis=0)
        state_std = np.std(self.state_memory, axis=0)
        self.state_memory = (self.state_memory - state_mean) / (state_std + 1e-7)
        next_state_mean = np.mean(self.new_state_memory, axis=0)
        next_state_std = np.std(self.new_state_memory, axis=0)
        self.new_state_memory = (self.new_state_memory - next_state_mean) / (next_state_std + 1e-7)

