import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from rl_ws.gym_ackbot.envs.ackbot_follow_env import AckBotFollowEnv

def test_gymnasium_api_compliance():
    env = AckBotFollowEnv()
    # Official Gymnasium environment validation suite
    check_env(env.unwrapped)
    env.close()

def test_gym_make_registration():
    env = gym.make("AckBotFollow-v0")
    obs, info = env.reset(seed=42)
    assert obs.shape == (426,)
    assert "target_dist" in info
    env.close()

def test_reset_and_step_shapes():
    env = AckBotFollowEnv()
    obs, info = env.reset(seed=123)
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (426,)
    assert obs.dtype == np.float32
    assert -180.0 <= info["rel_heading_deg"] <= 180.0
    
    # Take random action
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    
    assert next_obs.shape == (426,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)
    env.close()

def test_action_bounds_and_ackermann_limits():
    env = AckBotFollowEnv()
    env.reset()
    
    # Test maximum forward speed and right steer
    action = np.array([10.0, 5.0], dtype=np.float32)  # Exceeds bounds
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Internal robot velocity should be bounded by actuator dynamics and max limits
    assert env.robot_v <= env.v_max
    assert abs(env.robot_steer) <= env.max_steering_angle
    env.close()

def test_collision_detection():
    env = AckBotFollowEnv()
    env.reset()
    
    # Force robot outside arena wall
    env.robot_x = 6.0  # Outside wall at 5.0m
    is_collision, c_type = env._check_collision()
    assert is_collision is True
    assert c_type == "wall"
    
    # Step should terminate
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert reward < -10.0
    env.close()
