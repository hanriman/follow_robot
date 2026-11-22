"""
Recurrent Multi-Environment Rollout Buffer for PPO with Generalized Advantage Estimation (GAE).
Supports both single-env flat batches (backwards-compatible) and vectorized multi-env
sequence chunks for truncated BPTT.
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        obs_dim,
        act_dim,
        device="cpu",
        num_envs=1,
        gru_hidden_dim=128,
    ):
        self.buffer_size = int(buffer_size)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_envs = int(num_envs)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.device = torch.device(device)

        self.reset()

    def reset(self):
        shape_prefix = (self.buffer_size,) if self.num_envs == 1 else (self.buffer_size, self.num_envs)

        self.obs = torch.zeros(
            (*shape_prefix, self.obs_dim), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            (*shape_prefix, self.act_dim), dtype=torch.float32, device=self.device
        )
        self.logprobs = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        # `dones` stores collision `terminated` flags for GAE value-bootstrap masking
        self.dones = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        # `episode_dones` stores `terminated | truncated` for GRU hidden-state resets
        self.episode_dones = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        self.hiddens = torch.zeros(
            (*shape_prefix, self.gru_hidden_dim), dtype=torch.float32, device=self.device
        )

        self.advantages = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs,
        action,
        logprob,
        reward,
        done,
        value,
        hidden=None,
        episode_done=None,
    ):
        self.obs[self.pos] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.pos] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.logprobs[self.pos] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.rewards[self.pos] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.pos] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.values[self.pos] = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        if episode_done is not None:
            self.episode_dones[self.pos] = torch.as_tensor(
                episode_done, dtype=torch.float32, device=self.device
            )
        else:
            self.episode_dones[self.pos] = self.dones[self.pos]

        if hidden is not None:
            h_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device)
            if h_t.dim() == 3 and h_t.shape[0] == 1:
                h_t = h_t.squeeze(0)
            if self.num_envs == 1 and h_t.dim() == 2 and h_t.shape[0] == 1:
                h_t = h_t.squeeze(0)
            self.hiddens[self.pos] = h_t

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_val, last_done, gamma=0.99, gae_lambda=0.95):
        last_val = torch.as_tensor(last_val, dtype=torch.float32, device=self.device)
        if self.num_envs == 1:
            last_val = last_val.squeeze()
            last_gae_lam = 0.0
        else:
            last_val = last_val.reshape(self.num_envs)
            last_gae_lam = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.buffer_size)):
            next_values = last_val if t == self.buffer_size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yield randomized flat mini-batches (preserved for single-env compatibility & tests)."""
        if self.num_envs == 1:
            total = self.buffer_size
            obs = self.obs
            actions = self.actions
            logprobs = self.logprobs
            advantages = self.advantages
            returns = self.returns
            values = self.values
        else:
            total = self.buffer_size * self.num_envs
            obs = self.obs.reshape(total, self.obs_dim)
            actions = self.actions.reshape(total, self.act_dim)
            logprobs = self.logprobs.reshape(total)
            advantages = self.advantages.reshape(total)
            returns = self.returns.reshape(total)
            values = self.values.reshape(total)

        indices = np.random.permutation(total)
        start_idx = 0
        while start_idx < total:
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                obs[batch_indices],
                actions[batch_indices],
                logprobs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
                values[batch_indices],
            )
            start_idx += batch_size

    def get_recurrent_batches(self, num_mini_batches, seq_len=32):
        """
        Slice each environment trajectory into contiguous chunks of length `seq_len`,
        shuffle chunks, and yield sequence mini-batches shaped (seq_len, mb_chunks, ...).
        """
        if self.buffer_size % seq_len != 0:
            raise ValueError(
                f"buffer_size ({self.buffer_size}) must be divisible by seq_len ({seq_len})"
            )

        chunks_per_env = self.buffer_size // seq_len
        if self.num_envs == 1:
            obs = self.obs.unsqueeze(1)
            actions = self.actions.unsqueeze(1)
            logprobs = self.logprobs.unsqueeze(1)
            advantages = self.advantages.unsqueeze(1)
            returns = self.returns.unsqueeze(1)
            values = self.values.unsqueeze(1)
            hiddens = self.hiddens.unsqueeze(1)
            ep_dones = self.episode_dones.unsqueeze(1)
        else:
            obs = self.obs
            actions = self.actions
            logprobs = self.logprobs
            advantages = self.advantages
            returns = self.returns
            values = self.values
            hiddens = self.hiddens
            ep_dones = self.episode_dones

        # Reshape (T, E, ...) -> (chunks_per_env, seq_len, E, ...) -> permute -> (total_chunks, seq_len, ...)
        def to_chunks(tensor, tail_shape=()):
            reshaped = tensor.reshape(chunks_per_env, seq_len, self.num_envs, *tail_shape)
            # (chunks_per_env, E, seq_len, ...) -> (total_chunks, seq_len, ...)
            return reshaped.permute(0, 2, 1, *range(3, reshaped.dim())).reshape(
                chunks_per_env * self.num_envs, seq_len, *tail_shape
            )

        c_obs = to_chunks(obs, (self.obs_dim,))
        c_act = to_chunks(actions, (self.act_dim,))
        c_logp = to_chunks(logprobs)
        c_adv = to_chunks(advantages)
        c_ret = to_chunks(returns)
        c_val = to_chunks(values)
        c_ep_dones = to_chunks(ep_dones)
        c_hx = to_chunks(hiddens, (self.gru_hidden_dim,))[:, 0, :]  # (total_chunks, H)

        total_chunks = chunks_per_env * self.num_envs
        mb_chunks = max(1, total_chunks // int(num_mini_batches))
        perm = np.random.permutation(total_chunks)

        for start in range(0, total_chunks, mb_chunks):
            idx = perm[start : start + mb_chunks]
            # Transpose back to (seq_len, mb_chunks, ...) for GRU sequence execution
            yield (
                c_obs[idx].transpose(0, 1),
                c_act[idx].transpose(0, 1),
                c_logp[idx].transpose(0, 1),
                c_adv[idx].transpose(0, 1),
                c_ret[idx].transpose(0, 1),
                c_val[idx].transpose(0, 1),
                c_hx[idx].unsqueeze(0),  # (1, mb_chunks, H)
                c_ep_dones[idx].transpose(0, 1),
            )
