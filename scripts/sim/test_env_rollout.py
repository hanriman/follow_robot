#!/usr/bin/env python3
"""
Diagnostic runner for AckBotFollow-v0 Gymnasium environment.
Runs rollouts, measures physics/LiDAR steps per second, and prints state transitions.
"""

import math
import time

import gymnasium as gym
import numpy as np


def main():
    print("=" * 68)
    print("       ACKBOT GYMNASIUM ENVIRONMENT BENCHMARK & DIAGNOSTIC          ")
    print("=" * 68)

    env = gym.make("AckBotFollow-v0")
    print("Environment ID      : AckBotFollow-v0")
    print(
        f"Observation Space   : {env.observation_space.shape} (64 Camera Depth + 360 LiDAR + 2 State)"
    )
    print(f"Action Space        : {env.action_space}")
    print(f"Speed Limits        : [{env.unwrapped.v_min:.2f}, {env.unwrapped.v_max:.2f}] m/s")
    print(f"Max Steering Angle  : ±{math.degrees(env.unwrapped.max_steering_angle):.2f}°")
    print(f"Min Turning Radius  : {env.unwrapped.min_turning_radius:.3f} m")
    print(
        f"LiDAR Rays          : {env.unwrapped.lidar_samples} rays @ max {env.unwrapped.lidar_range_max} m"
    )
    print("-" * 68)

    # 1. Test Single Episode with Step-by-Step Reporting
    print("Running sample episode with proportional target-seeking heuristic:")
    obs, info = env.reset(seed=42)
    print(
        f"Spawn Robot : ({env.unwrapped.robot_x:.2f}, {env.unwrapped.robot_y:.2f}) | Heading: {math.degrees(env.unwrapped.robot_yaw):.1f}°"
    )
    print(
        f"Spawn Target: ({env.unwrapped.target_x:.2f}, {env.unwrapped.target_y:.2f}) | Distance: {info['target_dist']:.2f} m"
    )
    print("-" * 68)
    print(
        f"{'Step':>4} | {'Robot (X, Y)':>16} | {'Target (X, Y)':>16} | {'Dist':>6} | {'Speed':>6} | {'Steer':>7} | {'Reward':>7}"
    )
    print("-" * 68)

    total_reward = 0.0
    for step in range(1, 16):
        # Heuristic: Steer towards target angle, throttle proportional to distance
        rel_x, rel_y, rel_heading = env.unwrapped._get_relative_target_state()
        target_dist = math.hypot(rel_x, rel_y)

        # Proportional controller
        desired_v = np.clip(0.5 * (target_dist - 0.8), -0.2, 0.8)
        desired_steer = np.clip(
            1.2 * rel_heading, -env.unwrapped.max_steering_angle, env.unwrapped.max_steering_angle
        )
        action = np.array([desired_v, desired_steer], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"{step:4d} | "
            f"({env.unwrapped.robot_x:6.2f}, {env.unwrapped.robot_y:6.2f}) | "
            f"({env.unwrapped.target_x:6.2f}, {env.unwrapped.target_y:6.2f}) | "
            f"{info['target_dist']:5.2f}m | "
            f"{info['robot_v']:5.2f} | "
            f"{info['robot_steer_deg']:+6.1f}° | "
            f"{reward:7.3f}"
        )

        if terminated or truncated:
            break

    print("-" * 68)
    print(f"Sample rollout completed. Total reward: {total_reward:.2f}")

    # 2. High-Throughput Speed Benchmark
    print("\nBenchmarking raw simulation throughput on your M4 Pro (1,000 steps)...")
    env.reset(seed=100)
    start_time = time.perf_counter()
    n_steps = 1000

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()

    elapsed = time.perf_counter() - start_time
    fps = n_steps / elapsed

    print(f"Completed {n_steps:,} simulation steps in {elapsed:.3f} seconds.")
    print(f"==> Simulation Speed: {fps:,.0f} steps/second (FPS)!")
    print(f"==> 1 Million Steps will take ~{1_000_000 / fps / 60:.1f} minutes.")
    print("=" * 68)
    env.close()


if __name__ == "__main__":
    main()
