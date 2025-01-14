import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import copy
import numpy as np
import sys

import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

from .base_agent import BaseAgent

policy_freq = 2
target_update_freq = 1

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(state_dim).prod() + np.prod(action_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, action_low, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(state_dim).prod(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, np.prod(action_dim))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_dim))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_high- action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        if not self.training:
            x_t = mean
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SACAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, action_high, action_low,device = "cuda:0", hidden_dim=256):
        
        self._device = device
        self.actor = Actor(state_dim, action_dim, action_high, action_low, hidden_dim).to(self._device)
        self.critic_1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self._device)
        self.critic_2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self._device)
        self.critic_target_1 = copy.deepcopy(self.critic_1).to(self._device)
        self.critic_target_2 = copy.deepcopy(self.critic_2).to(self._device)
        self.q_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=critic_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=actor_lr)
        self.global_step = 0
        self.gamma = 0.9
        self.tau = 0.05
        self.alpha = 0.2

    def predict(self, state, device="cuda:0"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _, _ = self.actor.get_action(state)
        # actions = torch.softmax(actions, dim=-1)
        return actions.cpu().data.numpy().flatten()

    def learn(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(states).float().to(self._device)
        next_states = torch.from_numpy(next_states).float().to(self._device)
        actions = torch.from_numpy(actions).float().to(self._device)
        rewards = torch.from_numpy(rewards).float().reshape((-1, 1)).to(self._device)
        dones = torch.from_numpy(dones).float().reshape((-1,1)).to(self._device)

        # Update Q-functions
        # breakpoint()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
            qf1_next_target = self.critic_target_1(next_states, next_state_actions)
            qf2_next_target = self.critic_target_2(next_states, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        q1 = self.critic_1(states, actions).view(-1)
        q2 = self.critic_2(states, actions).view(-1)

        q1_loss = nn.functional.mse_loss(q1, next_q_value)
        q2_loss = nn.functional.mse_loss(q2, next_q_value)
        qf_loss = q1_loss + q2_loss

        
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()


        # Update policy
        if self.global_step % policy_freq == 0:
            for _ in range(policy_freq):
                pi, log_pi, _ = self.actor.get_action(states)
                qf1_pi = self.critic_1(states, pi)
                qf2_pi = self.critic_2(states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
        log_dict = dict(
            qf1_pred=q1.mean().item(),
            qf2_pred=q2.mean().item(),
            average_target_qa=next_q_value.mean().item(),
            qf1_loss=q1_loss.item(),
            qf2_loss=q2_loss.item(),
            qf_loss=qf_loss.item(),
            actor_loss=actor_loss.item(),
        )

        # Update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return log_dict


    def save(self, filepath):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
