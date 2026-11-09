#!/usr/bin/env python3
"""
Diagnostic script to test LeWM CEM-MPC planning behavior on canonical observations.
"""

import math
import torch
import numpy as np

from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.cem_planner import CEMPlanner


def main():
    device = torch.device("cpu")
    torch.set_num_threads(2)

    wm_path = "outputs/experiments/sigreg_curriculum/checkpoints/best_model.pt"
    rew_path = "outputs/experiments/mpc_curriculum/checkpoints/reward_head_lewm.pt"

    lewm = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(device)
    ckpt = torch.load(wm_path, map_location=device, weights_only=False)
    lewm.load_state_dict(ckpt["model_state_dict"])
    lewm.eval()

    reward_head = LatentRewardHead(in_dim=128, is_spatial=False).to(device)
    rew_ckpt = torch.load(rew_path, map_location=device, weights_only=False)
    reward_head.load_state_dict(rew_ckpt["model_state_dict"])
    reward_head.eval()

    planner = CEMPlanner(
        horizon=5,
        pop_size=35,
        num_elites=5,
        num_iters=3,
        v_min=-0.40,
        v_max=0.80,
        steer_max=0.553953,
        device=device,
    )

    # Let's construct a synthetic observation where the target is straight ahead at 1.4m:
    # 64 camera depth: all 5.0m (normalized 1.0) except center beams (beams 30-34) at 1.4m
    cam_depth = np.ones(64, dtype=np.float32)
    # 1.4m normalized: (1.4 - 0.1) / (5.0 - 0.1) = 1.3 / 4.9 = 0.265
    cam_depth[30:35] = 0.265

    # 360 LiDAR: all 3.0m (normalized 1.0) except forward beams (0, 1, 359) at 1.4m
    # (1.4 - 0.1) / (3.0 - 0.1) = 1.3 / 2.9 = 0.448
    lidar_ranges = np.ones(360, dtype=np.float32)
    lidar_ranges[0:3] = 0.448
    lidar_ranges[358:360] = 0.448
    # Left wall at 1.2m (beam 90: (1.2-0.1)/2.9 = 0.379)
    lidar_ranges[85:96] = 0.379

    state = np.array([0.0, 0.0], dtype=np.float32)
    obs = np.concatenate([cam_depth, lidar_ranges, state], dtype=np.float32)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        z_0 = lewm.encode(obs_t)
        print("Encoded z_0 norm:", float(torch.norm(z_0)))

        # Evaluate scores for different actions directly:
        # e.g., straight (steer=0), steer left (+0.4), steer right (-0.4)
        for steer_val in [-0.45, -0.30, -0.15, 0.0, +0.15, +0.30, +0.45]:
            acts = torch.tensor([[[0.35, steer_val]] * 5], device=device, dtype=torch.float32)
            unrolled = lewm.predictor.rollout(z_0, acts)
            score = reward_head.compute_control_score(unrolled)
            preds = reward_head(unrolled)
            print(
                f"Steer={steer_val:+5.2f} rad ({math.degrees(steer_val):+5.1f}°) -> "
                f"Total Score={float(score.sum()):+7.3f} | "
                f"r_env={float(preds[..., 0].sum()):+6.3f}, "
                f"dist_err={float(preds[..., 1].sum()):+6.3f}, "
                f"vis_lock={float(preds[..., 2].sum()):+6.3f}"
            )

        # Now test what CEM planner outputs over 5 trials
        print("\nTesting CEM Planner 5 runs:")
        for trial in range(5):
            planner.reset()
            action = planner.plan(
                initial_z=z_0,
                rollout_fn=lambda z, acts: lewm.predictor.rollout(z, acts),
                score_fn=lambda unrolled: reward_head.compute_control_score(unrolled),
            )
            v_cmd = action[0].item()
            steer_cmd = action[1].item()
            print(f"Trial {trial+1}: v={v_cmd:+.3f} m/s, steer={steer_cmd:+.3f} rad ({math.degrees(steer_cmd):+5.1f}°)")


if __name__ == "__main__":
    main()
