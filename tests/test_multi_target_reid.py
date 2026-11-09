"""
Unit tests for Multi-Pedestrian Distractor Simulation and Gated Re-ID Tracking.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from rl_ws.gym_ackbot.envs.ackbot_follow_env import AckBotFollowEnv
from rl_ws.gym_ackbot.envs.human_target import BipedalHumanTarget
from rl_ws.policies.tracking.target_tracker import TargetRegistrationTracker


def test_multi_pedestrian_spawning_and_colors():
    """Verify environment initializes target and multiple distractors with distinct clothing colors."""
    env = AckBotFollowEnv(num_distractors=3)
    obs, info = env.reset(seed=42)

    assert len(env.distractors) == 3
    assert obs.shape == (426,)

    # Distractor colors should be valid RGB tuples
    target_color = env.human.color_rgb
    distractor_colors = [d.color_rgb for d in env.distractors]

    # Verify distractors have distinct colors from the registered target
    for d_col in distractor_colors:
        assert d_col != target_color
        assert len(d_col) == 3

    # Step environment and verify all pedestrians advance their gait
    initial_phases = [d.gait_phase for d in env.distractors]
    env.step(np.array([0.0, 0.0], dtype=np.float32))

    for idx, d in enumerate(env.distractors):
        assert d.gait_phase != initial_phases[idx]

    env.close()


def test_sensor_raytracing_with_multiple_pedestrians():
    """Verify depth camera and LiDAR detect both target and distractor feet."""
    env = AckBotFollowEnv(num_distractors=1)
    env.reset(seed=42)

    # Place target at (2.0, 0.0) and distractor at (1.5, 0.3) inside camera FOV
    env.robot_x = 0.0
    env.robot_y = 0.0
    env.robot_yaw = 0.0

    env.human.x = 2.5
    env.human.y = 0.0
    env.distractors[0].x = 1.2
    env.distractors[0].y = 0.0  # Directly in front of camera at 1.2m

    cam_ranges = env._simulate_camera_depth()
    # Ray passing through (0, 0) towards +X should hit the closer distractor at ~1.2m, not the target at 2.5m
    center_idx = len(cam_ranges) // 2
    assert cam_ranges[center_idx] < 1.40
    assert math.isclose(cam_ranges[center_idx], 1.2 - (env.chassis_length / 2.0) - 0.09, abs_tol=0.15)
    env.close()


def test_gated_reid_association_rejects_distractor():
    """Verify tracker rejects crossing distractor with different clothing color and stays locked on target."""
    tracker = TargetRegistrationTracker(dt=0.05)
    crimson_red = (0.80, 0.20, 0.20)
    navy_blue = (0.15, 0.30, 0.65)

    tracker.register_target(x=2.0, y=0.0, vx=0.25, vy=0.0, color_rgb=crimson_red, target_id="target_0")

    # Candidate A: Navy blue distractor walking close to the target
    cand_distractor = {
        "x": 2.08,
        "y": 0.15,
        "color_rgb": navy_blue,
        "id": "distractor_1",
        "is_target": False,
    }
    # Candidate B: True target with crimson red clothing
    cand_target = {
        "x": 2.05,
        "y": 0.01,
        "color_rgb": crimson_red,
        "id": "target",
        "is_target": True,
    }

    # Data association update with both candidates present
    result = tracker.update_with_candidates([cand_distractor, cand_target])

    assert result["associated"] is True
    assert result["selected_candidate"]["is_target"] is True
    assert result["selected_candidate"]["id"] == "target"
    assert tracker.is_occluded is False
    assert tracker.occlusion_steps == 0


def test_spatial_gating_rejects_distant_same_color_human():
    """Verify Mahalanobis gating rejects distant human even if clothing color matches perfectly."""
    tracker = TargetRegistrationTracker(dt=0.05)
    emerald_green = (0.15, 0.60, 0.35)
    tracker.register_target(x=1.5, y=0.0, vx=0.0, vy=0.0, color_rgb=emerald_green)

    # Distant human 4.5m away with the EXACT same clothing color
    distant_human = {
        "x": 6.0,
        "y": 3.0,
        "color_rgb": emerald_green,
        "id": "distractor_far",
        "is_target": False,
    }

    result = tracker.update_with_candidates([distant_human])

    # Must be rejected by Mahalanobis validation gate (d^2 > 9.21)
    assert result["associated"] is False
    assert result["selected_candidate"] is None
    assert tracker.is_occluded is True
    assert tracker.occlusion_steps == 1


def test_crossing_pedestrian_occlusion_and_reid_recovery():
    """Verify target state projects forward during distractor occlusion and recovers upon reappearance."""
    tracker = TargetRegistrationTracker(dt=0.05)
    target_color = (0.85, 0.55, 0.15)  # Amber orange
    distractor_color = (0.20, 0.65, 0.70)  # Cyan

    tracker.register_target(x=2.0, y=0.0, vx=0.30, vy=0.0, color_rgb=target_color)

    # Initial 5 steps tracking cleanly
    for s in range(1, 6):
        cand = {"x": 2.0 + 0.30 * (s * 0.05), "y": 0.0, "color_rgb": target_color, "id": "target", "is_target": True}
        res = tracker.update_with_candidates([cand])
        assert res["associated"] is True

    # Next 15 steps (0.75 seconds): Distractor walks in front, true target is occluded
    for s in range(6, 21):
        crossing_distractor = {
            "x": 2.0 + 0.30 * 0.3,
            "y": -1.0 + 0.15 * (s - 6),
            "color_rgb": distractor_color,
            "id": "distractor",
            "is_target": False,
        }
        res = tracker.update_with_candidates([crossing_distractor])
        # Tracker must reject distractor and project target forward
        assert res["associated"] is False
        assert tracker.is_occluded is True

    assert tracker.occlusion_steps == 15
    est_x, est_y = tracker.get_status()["estimated_pos"]
    expected_x = 2.0 + 0.30 * (20 * 0.05)
    assert math.isclose(est_x, expected_x, abs_tol=0.08)

    # Step 21: Target re-emerges from occlusion
    reemerged_target = {
        "x": expected_x + 0.01,
        "y": 0.0,
        "color_rgb": target_color,
        "id": "target",
        "is_target": True,
    }
    reid_res = tracker.update_with_candidates([reemerged_target])

    assert reid_res["associated"] is True
    assert reid_res["selected_candidate"]["is_target"] is True
    assert tracker.is_occluded is False
    assert tracker.occlusion_steps == 0
