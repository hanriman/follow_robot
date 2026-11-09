"""
Rollout buffer for PPO with Generalized Advantage Estimation (GAE).
Stores trajectories and generates randomized mini-batches on device.
"""

import torch
import numpy as np


class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.buffer_size, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.buffer_size, self.act_dim), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)

        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)

        self.pos = 0
        self.full = False

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.pos] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.pos] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.logprobs[self.pos] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.rewards[self.pos] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.pos] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.values[self.pos] = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_val, last_done, gamma=0.99, gae_lambda=0.95):
        last_val = torch.as_tensor(last_val, dtype=torch.float32, device=self.device).squeeze()
        last_done = torch.as_tensor(last_done, dtype=torch.float32, device=self.device).squeeze()

        last_gae_lam = 0.0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        indices = np.random.permutation(self.buffer_size)
        start_idx = 0
        while start_idx < self.buffer_size:
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                self.obs[batch_indices],
                self.actions[batch_indices],
                self.logprobs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
                self.values[batch_indices],
            )
            start_idx += batch_size
