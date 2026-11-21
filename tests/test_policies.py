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
