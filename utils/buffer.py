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
        self.state_memory = np.zeros((self.mem_size, obs_shape))
        self.new_state_memory = np.zeros((self.mem_size, obs_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int64)

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

    def save_buffer(self, file_name):
        '''
        If the file_name exists, then append the buffer to the file,
        otherwise, create a new file.
        '''
        min_size = min(self.mem_cntr, self.mem_size)
        if os.path.exists(file_name):
            with h5py.File(file_name, 'a') as f:
                for dataset_name, memory in [('states', self.state_memory), 
                                            ('actions', self.action_memory), 
                                            ('rewards', self.reward_memory), 
                                            ('next_states', self.new_state_memory), 
                                            ('dones', self.terminal_memory)]:
                    f[dataset_name].resize((f[dataset_name].shape[0] + min_size), axis=0)
                    f[dataset_name][-min_size:] = memory[:min_size]
        else:
            with h5py.File(file_name, 'w') as f:
                # Assuming the datasets are 2D arrays. Adjust the shape, maxshape, and chunks as needed.
                
                for dataset_name, memory in [('states', self.state_memory), 
                                            ('actions', self.action_memory), 
                                            ('rewards', self.reward_memory), 
                                            ('next_states', self.new_state_memory), 
                                            ('dones', self.terminal_memory)]:
                    
                    
                    if memory.ndim == 1:
                        max_shape = (None,)
                        chunks_shape = (1,)
                    else:
                        max_shape = (None,1e7)
                        chunks_shape = (1, memory.shape[1])  # This sets the chunk shape    
                    f.create_dataset(dataset_name, data=memory[:min_size],maxshape = max_shape,chunks=chunks_shape)



    def load_buffer(self, file_name):
        with h5py.File(file_name, 'r') as f:
            self.state_memory = f['states'][:]
            self.action_memory = f['actions'][:]
            self.reward_memory = f['rewards'][:]
            self.new_state_memory = f['next_states'][:]
            self.terminal_memory = f['dones'][:]
            self.mem_cntr = min(self.mem_size, len(f['states']))
            
            
    def nomarlize_states(self):
        
        state_mean = np.mean(self.state_memory, axis=0)
        state_std = np.std(self.state_memory, axis=0)
        self.state_memory = (self.state_memory - state_mean) / (state_std + 1e-7)
        next_state_mean = np.mean(self.new_state_memory, axis=0)
        next_state_std = np.std(self.new_state_memory, axis=0)
        self.new_state_memory = (self.new_state_memory - next_state_mean) / (next_state_std + 1e-7)

