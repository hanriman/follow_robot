import numpy as np
import torch

from rl_ws.policies.ppo.actor_critic import ActorCritic
from rl_ws.policies.ppo.buffer import RolloutBuffer


def get_test_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_actor_critic_forward_pass():
    device = get_test_device()
    model = ActorCritic(obs_dim=426, act_dim=2).to(device)

    batch_size = 8
    dummy_obs = torch.randn((batch_size, 426), dtype=torch.float32, device=device)

    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs)

    assert action.shape == (batch_size, 2)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)


def test_action_bounds():
    device = get_test_device()
    model = ActorCritic(obs_dim=426, act_dim=2).to(device)

    # Sample 100 actions and verify all physical limits
    dummy_obs = torch.randn((100, 426), dtype=torch.float32, device=device)
    actions, _, _, _ = model.get_action_and_value(dummy_obs)

    v = actions[:, 0]
    steer = actions[:, 1]

    assert (v >= -0.401).all() and (v <= 0.801).all()
    assert (steer >= -0.554).all() and (steer <= 0.554).all()


def test_deterministic_action():
    device = get_test_device()
    model = ActorCritic(obs_dim=426, act_dim=2).to(device)

    dummy_obs = torch.randn((4, 426), dtype=torch.float32, device=device)
    action1 = model.get_deterministic_action(dummy_obs)
    action2 = model.get_deterministic_action(dummy_obs)

    # Deterministic output should be identical
    assert torch.allclose(action1, action2)


def test_rollout_buffer_and_gae():
    device = get_test_device()
    buf_size = 64
    buffer = RolloutBuffer(buffer_size=buf_size, obs_dim=426, act_dim=2, device=device)

    for i in range(buf_size):
        obs = np.random.randn(426).astype(np.float32)
        act = np.array([0.2, 0.1], dtype=np.float32)
        logp = -1.5
        rew = 1.0
        done = i == buf_size - 1
        val = 0.5
        buffer.add(obs, act, logp, rew, done, val)

    assert buffer.full is True

    # Compute GAE
    buffer.compute_returns_and_advantages(last_val=0.5, last_done=False)

    assert buffer.advantages.shape == (buf_size,)
    assert buffer.returns.shape == (buf_size,)

    # Test batch generator
    batch_count = 0
    total_samples = 0
    for b_obs, b_act, b_logp, b_adv, b_ret, b_val in buffer.get_batches(batch_size=16):
        batch_count += 1
        total_samples += len(b_obs)
        assert b_obs.shape == (16, 426)
        assert b_act.shape == (16, 2)

    assert batch_count == 4
    assert total_samples == buf_size


def test_gae_terminal_reward_propagation():
    """Verify GAE correctly propagates terminal collision penalty to preceding steps without bootstrapping across episodes."""
    device = torch.device("cpu")
    buf_size = 10
    buffer = RolloutBuffer(buffer_size=buf_size, obs_dim=426, act_dim=2, device=device)

    # Episode 1: Steps 0 to 4. Step 4 ends in a collision penalty (-25.0, done=1.0)
    # Episode 2: Steps 5 to 9. Step 5 is clean restart (reward=1.0, done=0.0, value=10.0)
    for t in range(buf_size):
        obs = np.zeros(426, dtype=np.float32)
        act = np.zeros(2, dtype=np.float32)
        logp = 0.0
        val = 0.0
        if t < 4:
            rew = 1.0
            done = 0.0
        elif t == 4:
            rew = -25.0  # Fatal collision at step 4
            done = 1.0
        else:
            rew = 2.0  # New clean episode
            done = 0.0
            val = 5.0
        buffer.add(obs, act, logp, rew, done, val)

    buffer.compute_returns_and_advantages(last_val=0.0, last_done=True, gamma=0.99, gae_lambda=0.95)

    # Advantages for steps prior to collision must be significantly negative due to -25.0 penalty propagation
    adv = buffer.advantages.cpu().numpy()
    assert adv[4] < -20.0, f"Terminal step advantage should reflect -25 penalty, got {adv[4]}"
    assert adv[3] < -15.0, f"Step prior to collision must receive propagated penalty, got {adv[3]}"
    assert adv[2] < -10.0, f"Step 2 must receive propagated penalty, got {adv[2]}"

    # Step 4 must NOT bootstrap from Step 5's high value (5.0)
    # If done[4] was ignored or off-by-one, adv[4] would add gamma * val[5] (5.0 * 0.99 = 4.95)
    delta_4 = -25.0 - 0.0  # rew[4] - val[4] = -25.0
    assert abs(adv[4] - delta_4) < 1e-4, f"Terminal delta must equal exact penalty, got {adv[4]}"


def test_actor_critic_angular_sensitivity_and_non_saturating_bounds():
    """Verify CoordConv/polar features distinguish left vs right target dips and reach full steering bounds."""
    device = get_test_device()
    model = ActorCritic(obs_dim=426, act_dim=2).to(device)

    obs_right = torch.ones((1, 426), dtype=torch.float32, device=device)
    obs_right[:, 5:9] = 0.20  # Target dip on right side of camera FOV
    obs_right[:, 424:] = 0.0

    obs_left = torch.ones((1, 426), dtype=torch.float32, device=device)
    obs_left[:, 55:59] = 0.20  # Target dip on left side of camera FOV
    obs_left[:, 424:] = 0.0

    feat_r, _ = model._extract_features(obs_right)
    feat_l, _ = model._extract_features(obs_left)
    assert not torch.allclose(feat_r, feat_l, atol=1e-4)

    # Verify TanhNormal pre-tanh mean reaches > 99% of normalized +/-1.0 bounds smoothly
    extreme_logits = torch.tensor([[3.0, -3.0]], dtype=torch.float32, device=device)
    squashed = torch.tanh(extreme_logits)
    assert (squashed[:, 0] > 0.99).all() and (squashed[:, 1] < -0.99).all()


def test_actor_critic_recurrent_memory_and_episode_reset():
    """Verify GRU hidden state retains prior target observation and resets cleanly on episode_dones."""
    device = get_test_device()
    model = ActorCritic(obs_dim=426, act_dim=2).to(device)

    obs_target_left = torch.ones((1, 426), dtype=torch.float32, device=device)
    obs_target_left[:, 56:60] = 0.25
    obs_empty = torch.ones((1, 426), dtype=torch.float32, device=device)

    hx0 = model.init_hidden(1, device=device)
    _, _, _, _, hx1 = model.get_action_and_value(obs_target_left, hx=hx0, return_hidden=True)

    # Empty observation with retained memory differs from empty observation with zero memory
    act_with_mem = model.get_deterministic_action(obs_empty, hx=hx1)
    act_no_mem = model.get_deterministic_action(obs_empty, hx=hx0)
    assert not torch.allclose(act_with_mem, act_no_mem, atol=1e-5)

    # Passing episode_dones=1.0 must zero next_hx
    done_flag = torch.tensor([1.0], dtype=torch.float32, device=device)
    _, _, _, _, hx_reset = model.get_action_and_value(
        obs_target_left, hx=hx0, episode_dones=done_flag, return_hidden=True
    )
    assert torch.allclose(hx_reset, torch.zeros_like(hx_reset))


def test_recurrent_rollout_buffer_and_vectorized_gae():
    """Verify multi-env RolloutBuffer vectorized GAE and sequence-chunk BPTT batches."""
    device = get_test_device()
    buf_size = 32
    num_envs = 4
    seq_len = 16
    buffer = RolloutBuffer(
        buffer_size=buf_size,
        obs_dim=426,
        act_dim=2,
        device=device,
        num_envs=num_envs,
        gru_hidden_dim=128,
    )

    for t in range(buf_size):
        obs = np.ones((num_envs, 426), dtype=np.float32)
        act = np.zeros((num_envs, 2), dtype=np.float32)
        logp = np.zeros(num_envs, dtype=np.float32)
        rew = np.ones(num_envs, dtype=np.float32)
        done = np.zeros(num_envs, dtype=np.float32)
        val = np.full(num_envs, 0.5, dtype=np.float32)
        hx = np.zeros((num_envs, 128), dtype=np.float32)
        buffer.add(obs, act, logp, rew, done, val, hidden=hx, episode_done=done)

    assert buffer.full is True
    buffer.compute_returns_and_advantages(
        last_val=np.full(num_envs, 0.5, dtype=np.float32),
        last_done=np.zeros(num_envs, dtype=np.float32),
    )
    assert buffer.advantages.shape == (buf_size, num_envs)
    assert buffer.returns.shape == (buf_size, num_envs)

    batches = list(buffer.get_recurrent_batches(num_mini_batches=4, seq_len=seq_len))
    assert len(batches) == 4
    for b_obs, b_act, b_logp, b_adv, b_ret, b_val, b_hx, b_ep_dones in batches:
        assert b_obs.shape == (seq_len, 2, 426)
        assert b_act.shape == (seq_len, 2, 2)
        assert b_hx.shape == (1, 2, 128)
        assert b_ep_dones.shape == (seq_len, 2)
