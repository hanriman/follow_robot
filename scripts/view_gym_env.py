#!/usr/bin/env python3
"""
Interactive visual viewer for the actual AckBotFollow-v0 Gymnasium Environment.
Renders the real mathematical simulation (Ackermann steering, 360 LiDAR rays,
bipedal human target, curriculum scenarios, multi-pedestrian Re-ID tracking) directly in a native desktop window.

Supports selectable policy controllers:
- 'model'     : Trained PPO Actor-Critic neural network
- 'lewm'      : Teacher-Free LeWM CEM-MPC with live latent unrolling
- 'diffusion' : Action-Conditioned Diffusion CEM-MPC
- 'heuristic' : Proportional pursuit controller
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_ws.policies.ppo.actor_critic import ActorCritic
from rl_ws.policies.vla.diffusion_wm import ActionConditionedDiffusionWorldModel
from rl_ws.policies.world_model.cem_planner import CEMPlanner
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.vae import MultiModalSensorVAE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PPO = (
    PROJECT_ROOT
    / "outputs/experiments/exp_20261108_154753_curriculum_ppo_active_follow/checkpoints/best_model.pt"
)
DEFAULT_LEWM = PROJECT_ROOT / "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
DEFAULT_REW_LEWM = (
    PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"
)
DEFAULT_VAE = PROJECT_ROOT / "outputs/experiments/vae_curriculum/checkpoints/best_model.pt"
DEFAULT_DIFF = PROJECT_ROOT / "outputs/experiments/diffusion_curriculum/checkpoints/best_model.pt"
DEFAULT_REW_VAE = PROJECT_ROOT / "outputs/experiments/mpc_curriculum/checkpoints/reward_head_vae.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="AckBot Gymnasium Environment Viewer")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Curriculum stage (1: Open, 2: Corner, 3: U-Turn, 4: Clutter)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="model",
        choices=["model", "lewm", "diffusion", "heuristic"],
        help="Controller policy: 'model' (PPO), 'lewm' (LeWM MPC), 'diffusion' (AC-LDM), 'heuristic'",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="Number of distractor pedestrians (e.g. 0, 1, 2, 3)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional custom checkpoint path for selected policy",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="Run in headless mode for N steps and exit (no GUI window)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="3d",
        choices=["3d", "2d"],
        help="Visualization mode: '3d' (4-panel composite dashboard) or '2d' (classic top-down)",
    )
    parser.add_argument(
        "--save-screenshot",
        type=str,
        default=None,
        help="Path to save frame snapshot image (.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    stage_names = {
        1: "Stage 1: Open Pursuit (Clean arena, no obstacles)",
        2: "Stage 2: Blind Corner Occlusion (L-wall divider, target disappears from camera FOV)",
        3: "Stage 3: Dead-End Corridor & U-Turn Encounter (Target turns 180 deg and walks past robot)",
        4: "Stage 4: Complex Clutter (6 static pillars + dynamic multi-pedestrian crowd)",
    }

    print("=" * 72)
    print("       ACKBOT GYMNASIUM LIVE CURRICULUM ENVIRONMENT VIEWER          ")
    print("=" * 72)
    print(f"Active Scenario : {stage_names[args.stage]}")
    print(f"Active Policy   : {args.policy.upper()}")
    print(f"Distractors     : {args.distractors} pedestrians")
    print(
        f"View Mode       : {args.view.upper()} ({'4-Panel 3D Multi-View Dashboard' if args.view == '3d' else 'Classic 2D Top-Down'})"
    )
    print(f"Device          : {device}")
    print("Features & Visuals:")
    print("  • Bipedal Human Walker: Animated 2 feet stepping at 1.6 Hz cadence")
    print("  • Dynamic Appearance  : Randomized clothing color palette per episode")
    print("  • Target Registration : EKF tracking + Mahalanobis & Color gating")
    print("  • Ackermann Robot     : Genuine front wheel steering & 360 LiDAR")
    if args.headless_frames == 0:
        print("Interactive Key Controls:")
        print("  • [V]     : Toggle View Mode (3D Multi-View <-> 2D Top-Down)")
        print("  • [Space] : Pause / Resume simulation")
        print("  • [R]     : Reset episode")
        print("  • [S]     : Save instant frame screenshot")
        print("  • [Q/Esc] : Exit viewer")
    print("=" * 72)

    # Initialize policy controller
    policy_fn = None
    reset_fn = None

    if args.policy == "model":
        ckpt_p = Path(args.checkpoint) if args.checkpoint else DEFAULT_PPO
        if ckpt_p.exists():
            print(f"[+] Loading PPO Policy: {ckpt_p.name}")
            ppo = ActorCritic(obs_dim=426, act_dim=2).to(device)
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            ppo.load_state_dict(ckpt["model_state_dict"])
            ppo.eval()

            def ppo_act(obs):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    return ppo.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

            policy_fn = ppo_act
        else:
            print(f"[!] Warning: PPO checkpoint not found at {ckpt_p}. Using heuristic controller.")
            args.policy = "heuristic"

    elif args.policy == "lewm":
        ckpt_lewm = Path(args.checkpoint) if args.checkpoint else DEFAULT_LEWM
        if ckpt_lewm.exists() and DEFAULT_REW_LEWM.exists():
            print(f"[+] Loading LeWM World Model: {ckpt_lewm.name}")
            lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
            ckpt = torch.load(ckpt_lewm, map_location=device, weights_only=False)
            lewm.load_state_dict(ckpt["model_state_dict"])
            lewm.eval()

            rew_head = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
            rew_ckpt = torch.load(DEFAULT_REW_LEWM, map_location=device, weights_only=False)
            rew_head.load_state_dict(rew_ckpt["model_state_dict"])
            rew_head.eval()

            planner = CEMPlanner(horizon=5, pop_size=35, num_elites=5, num_iters=3, device=device)
            reset_fn = planner.reset

            def lewm_act(obs):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    z_0 = lewm.encode(obs_t)
                    act_t = planner.plan(
                        initial_z=z_0,
                        rollout_fn=lambda z, acts: lewm.predictor.rollout(z, acts),
                        score_fn=lambda unrolled: rew_head.compute_control_score(unrolled),
                    )
                    return act_t.cpu().numpy()

            policy_fn = lewm_act
        else:
            print("[!] Warning: LeWM checkpoint not found. Using heuristic controller.")
            args.policy = "heuristic"

    elif args.policy == "diffusion":
        if DEFAULT_DIFF.exists() and DEFAULT_VAE.exists() and DEFAULT_REW_VAE.exists():
            print("[+] Loading AC-LDM Diffusion World Model & Sensor VAE...")
            vae = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(device)
            vae.load_state_dict(
                torch.load(DEFAULT_VAE, map_location=device, weights_only=False)["model_state_dict"]
            )
            vae.eval()

            diff_wm = ActionConditionedDiffusionWorldModel(
                latent_channels=4,
                latent_length=32,
                action_dim=2,
                base_channels=64,
                num_timesteps=1000,
                vae=vae,
            ).to(device)
            diff_wm.load_state_dict(
                torch.load(DEFAULT_DIFF, map_location=device, weights_only=False)[
                    "model_state_dict"
                ]
            )
            diff_wm.eval()

            rew_vae = LatentRewardHead(
                in_dim=128, is_spatial=True, spatial_channels=4, spatial_length=32
            ).to(device)
            rew_vae.load_state_dict(
                torch.load(DEFAULT_REW_VAE, map_location=device, weights_only=False)[
                    "model_state_dict"
                ]
            )
            rew_vae.eval()

            planner_diff = CEMPlanner(
                horizon=3, pop_size=15, num_elites=3, num_iters=2, device=device
            )
            reset_fn = planner_diff.reset

            def diff_act(obs):
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    z_0, _ = vae.encode(obs_t)
                    act_t = planner_diff.plan(
                        initial_z=z_0,
                        rollout_fn=lambda z, acts: diff_wm.rollout_latents(z, acts, ddim_steps=4),
                        score_fn=lambda unrolled: rew_vae.compute_control_score(unrolled),
                    )
                    return act_t.cpu().numpy()

            policy_fn = diff_act
        else:
            print("[!] Warning: Diffusion checkpoints not found. Using heuristic controller.")
            args.policy = "heuristic"

    is_headless = args.headless_frames > 0
    render_mode = "rgb_array" if is_headless else "human"
    env = gym.make(
        "AckBotFollow-v0",
        render_mode=render_mode,
        stage=args.stage,
        num_distractors=args.distractors,
        view_mode=args.view,
    )
    obs, info = env.reset(seed=42)
    if reset_fn:
        reset_fn()

    # Initial frame render to initialize display and surface
    captured_frame = env.render()

    episode = 1
    episode_reward = 0.0
    step_count = 0
    running = True
    paused = False

    try:
        while running:
            if not is_headless:
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                            break
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            state_str = "PAUSED" if paused else "RESUMED"
                            print(f"[Controls] Simulation {state_str}")
                        elif event.key == pygame.K_r:
                            print("[Controls] Resetting episode...")
                            obs, info = env.reset()
                            if reset_fn:
                                reset_fn()
                            episode_reward = 0.0
                            step_count = 0
                        elif event.key == pygame.K_v:
                            curr_vm = env.unwrapped.view_mode
                            new_vm = "2d" if curr_vm == "3d" else "3d"
                            env.unwrapped.view_mode = new_vm
                            print(f"[Controls] View Mode switched to: {new_vm.upper()}")
                        elif event.key == pygame.K_s:
                            ts = int(time.time())
                            ss_dir = PROJECT_ROOT / "outputs/reports/media"
                            ss_dir.mkdir(parents=True, exist_ok=True)
                            ss_path = ss_dir / f"ackbot_{env.unwrapped.view_mode}_{ts}.png"
                            if env.unwrapped.window is not None:
                                pygame.image.save(env.unwrapped.window, str(ss_path))
                                print(f"[+] Snapshot saved to: {ss_path.resolve()}")

                if not running:
                    break

            if paused:
                env.render()
                time.sleep(0.04)
                continue

            if policy_fn is not None:
                action = policy_fn(obs)
            else:
                # Heuristic controller
                rel_x, rel_y, rel_heading = env.unwrapped._get_relative_target_state()
                target_dist = math.hypot(rel_x, rel_y)

                if (
                    args.stage == 3
                    and env.unwrapped.scenario_state == "WALK_BACK"
                    and env.unwrapped.target_x > env.unwrapped.robot_x
                ):
                    desired_v = -0.25
                    desired_steer = 0.0
                else:
                    desired_v = float(np.clip(0.6 * (target_dist - 0.80), -0.2, 0.80))
                    desired_steer = float(
                        np.clip(
                            1.4 * rel_heading,
                            -env.unwrapped.max_steering_angle,
                            env.unwrapped.max_steering_angle,
                        )
                    )

                action = np.array([desired_v, desired_steer], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            frame = env.render()
            if frame is not None:
                captured_frame = frame

            if is_headless and step_count >= args.headless_frames:
                print(f"[+] Headless run completed {step_count} steps successfully.")
                break

            if terminated or truncated:
                status = "COLLISION!" if info.get("collision") else "DONE"
                occ_str = " (OCCLUSION EVENT)" if info.get("target_occluded") else ""
                print(
                    f"Episode {episode:02d} [{status}{occ_str}] | "
                    f"Stage: {info['stage']} | Steps: {info['step']:03d} | Total Reward: {episode_reward:+7.1f}"
                )
                if not is_headless:
                    time.sleep(0.3)
                obs, info = env.reset()
                if reset_fn:
                    reset_fn()
                episode += 1
                episode_reward = 0.0

    except KeyboardInterrupt:
        print("\nExiting viewer...")
    finally:
        if args.save_screenshot:
            save_path = Path(args.save_screenshot)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not is_headless and env.unwrapped.window is not None:
                import pygame

                pygame.image.save(env.unwrapped.window, str(save_path))
                print(f"[+] Saved visual screenshot to: {save_path.resolve()}")
            elif captured_frame is not None:
                plt.imsave(save_path, captured_frame)
                print(f"[+] Saved visual screenshot to: {save_path.resolve()}")
        env.close()


if __name__ == "__main__":
    main()
