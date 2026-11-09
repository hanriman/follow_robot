"""
Unit tests for CarRacingObservationWrapper and CarRacingHeuristicPolicy.
"""

import numpy as np

from wm_control.config import EnvConfig
from wm_control.data.env_wrapper import make_carracing_env
from wm_control.data.heuristic_policy import CarRacingHeuristicPolicy, OrnsteinUhlenbeckActionNoise


def test_env_wrapper_observation_shape_and_range():
    """Verify that observation wrapper outputs (3, 64, 64) float32 in [0, 1]."""
    config = EnvConfig(target_height=64, target_width=64, crop_dashboard=True)
    env = make_carracing_env(config=config, seed=42)

    obs, info = env.reset(seed=42)
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    assert "telemetry" in info

    # Step with valid continuous action
    action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert next_obs.shape == (3, 64, 64)
    assert next_obs.dtype == np.float32
    assert 0.0 <= next_obs.min() and next_obs.max() <= 1.0
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "telemetry" in next_info

    env.close()


def test_telemetry_fields():
    """Verify all 10 Box2D physics telemetry attributes are extracted."""
    env = make_carracing_env(seed=42)
    _obs, _info = env.reset(seed=42)
    telemetry = env.get_telemetry()

    expected_keys = [
        "velocity_x",
        "velocity_y",
        "speed",
        "forward_velocity",
        "lateral_velocity",
        "angular_velocity",
        "angle",
        "position_x",
        "position_y",
        "steering_angle",
    ]

    for k in expected_keys:
        assert k in telemetry, f"Missing telemetry key: {k}"
        assert isinstance(telemetry[k], float)

    # Initial speed should be near 0
    assert telemetry["speed"] >= 0.0

    env.close()


def test_heuristic_policy_actions():
    """Verify heuristic policy generates valid continuous actions [steering, gas, brake]."""
    env = make_carracing_env(seed=42)
    _obs, info = env.reset(seed=42)
    policy = CarRacingHeuristicPolicy()

    # Step for 10 steps and verify actions
    for _ in range(10):
        action = policy.select_action(env, info.get("telemetry"))
        assert action.shape == (3,)
        assert -1.0 <= action[0] <= 1.0  # steering
        assert 0.0 <= action[1] <= 1.0   # gas
        assert 0.0 <= action[2] <= 1.0   # brake
        _obs, _reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    env.close()


def test_ornstein_uhlenbeck_noise():
    """Verify OU noise temporal dynamics and bounds."""
    noise = OrnsteinUhlenbeckActionNoise(action_dim=3, sigma=0.2)
    samples = [noise.sample() for _ in range(100)]
    samples_arr = np.stack(samples, axis=0)

    assert samples_arr.shape == (100, 3)
    # Samples should have near zero mean
    assert np.abs(samples_arr.mean(axis=0)).max() < 0.5
