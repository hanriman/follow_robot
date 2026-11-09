"""
Comprehensive unit tests for the AckBot Curriculum Progression, Bipedal Human Walker,
and Target Registration / EKF Occlusion Tracking Architecture.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from rl_ws.gym_ackbot.envs.ackbot_follow_env import AckBotFollowEnv
from rl_ws.gym_ackbot.envs.human_target import BipedalHumanTarget, CLOTHING_PALETTE
from rl_ws.policies.tracking.target_tracker import TargetRegistrationTracker


# ---------------------------------------------------------------------------
# 1. Bipedal Walker Kinematics and Gait Dynamics
# ---------------------------------------------------------------------------
def test_bipedal_target_stride_and_gait():
    """Verify 2-foot bipedal walker geometry, stride oscillation, and stance width."""
    walker = BipedalHumanTarget(
        x=0.0,
        y=0.0,
        yaw=0.0,  # Facing +X
        speed=0.28,
        cadence_hz=1.60,
        stride_length=0.45,
        stance_width=0.22,
        leg_radius=0.09,
    )
    assert len(CLOTHING_PALETTE) >= 6
    assert walker.color_rgb in CLOTHING_PALETTE

    # Stance width check: left foot is at +y, right foot is at -y
    walker.gait_phase = 0.0
    left_leg, right_leg = walker.get_feet_positions()
    assert left_leg.radius == 0.09
    assert right_leg.radius == 0.09
    assert math.isclose(left_leg.y, 0.11, abs_tol=1e-4)
    assert math.isclose(right_leg.y, -0.11, abs_tol=1e-4)

    # At gait_phase = pi/2: Left foot reaches max forward stride (+0.225m), Right foot reaches min (-0.225m)
    walker.gait_phase = math.pi / 2.0
    left_leg, right_leg = walker.get_feet_positions()
    assert math.isclose(left_leg.x - walker.x, 0.225, abs_tol=1e-3)
    assert math.isclose(right_leg.x - walker.x, -0.225, abs_tol=1e-3)

    # At gait_phase = 3*pi/2: Right foot reaches max forward stride (+0.225m), Left foot reaches min (-0.225m)
    walker.gait_phase = 3.0 * math.pi / 2.0
    left_leg, right_leg = walker.get_feet_positions()
    assert math.isclose(left_leg.x - walker.x, -0.225, abs_tol=1e-3)
    assert math.isclose(right_leg.x - walker.x, 0.225, abs_tol=1e-3)

    # Advance walker over multiple steps and verify phase advancement
    initial_phase = walker.gait_phase
    walker.step(dt=0.05)
    assert walker.gait_phase != initial_phase
    assert walker.x > 0.0  # Moved forward


# ---------------------------------------------------------------------------
# 2. Dual-Leg Raytracing Mechanics
# ---------------------------------------------------------------------------
def test_sensor_dual_leg_raytracing():
    """Verify that ray intersection tests detect both left and right legs independently."""
    walker = BipedalHumanTarget(
        x=2.0,
        y=0.0,
        yaw=math.pi / 2.0,  # Facing +Y (so stance is separated along X axis: Left at +X, Right at -X)
        stance_width=0.40,  # Wide stance to test gap
        stride_length=0.0,  # No stride swing for clean test
        leg_radius=0.08,
    )
    walker.gait_phase = 0.0
    left_leg, right_leg = walker.get_feet_positions()

    # Left leg should be at x = 2.0 - 0.20 = 1.80, y = 0.0
    # Right leg should be at x = 2.0 + 0.20 = 2.20, y = 0.0
    ray_origin_x, ray_origin_y = 2.0, -3.0

    # Ray 1: Directed straight up along x=2.0 (through the gap between the feet)
    # Cos=0, Sin=1 (heading north)
    hit_gap = walker.ray_intersection(
        ray_origin_x=2.0,
        ray_origin_y=-3.0,
        cos_angle=0.0,
        sin_angle=1.0,
        max_range=10.0,
    )
    # Gap is 0.40 - 2*0.08 = 0.24m wide; center ray passes cleanly through
    assert hit_gap == 10.0

    # Ray 2: Directed directly at left foot (x=1.80)
    angle_to_left = math.atan2(left_leg.y - (-3.0), left_leg.x - 2.0)
    hit_left = walker.ray_intersection(
        ray_origin_x=2.0,
        ray_origin_y=-3.0,
        cos_angle=math.cos(angle_to_left),
        sin_angle=math.sin(angle_to_left),
        max_range=10.0,
    )
    assert hit_left < 10.0
    assert math.isclose(hit_left, math.hypot(left_leg.x - 2.0, left_leg.y - (-3.0)) - 0.08, abs_tol=0.05)


# ---------------------------------------------------------------------------
# 3. Curriculum Stage Topologies & Gymnasium Compliance
# ---------------------------------------------------------------------------
def test_curriculum_stage_topologies():
    """Verify that Stages 1 to 4 correctly configure their respective arena geometry."""
    # Stage 1: Open arena (no obstacles or internal walls)
    env1 = AckBotFollowEnv(stage=1)
    obs1, info1 = env1.reset(seed=10)
    assert env1.stage == 1
    assert len(env1.static_obstacles) == 0
    assert len(env1.internal_walls) == 0
    assert info1["stage"] == 1
    assert obs1.shape == (426,)
    env1.close()

    # Stage 2: Blind Corner Occlusion (L-Wall divider)
    env2 = AckBotFollowEnv(stage=2)
    obs2, info2 = env2.reset(seed=10)
    assert env2.stage == 2
    assert len(env2.static_obstacles) == 0
    assert len(env2.internal_walls) == 2  # L-corner walls
    assert env2.scenario_state == "APPROACH_CORNER"
    assert info2["stage"] == 2
    env2.close()

    # Stage 3: Dead-End Corridor & U-Turn Encounter
    env3 = AckBotFollowEnv(stage=3)
    obs3, info3 = env3.reset(seed=10)
    assert env3.stage == 3
    assert len(env3.static_obstacles) == 0
    assert len(env3.internal_walls) == 3  # North wall, South wall, East end wall
    assert env3.scenario_state == "TO_DEADEND"
    assert info3["stage"] == 3
    env3.close()

    # Stage 4: Full Complex Clutter (6 cylindrical pillars)
    env4 = AckBotFollowEnv(stage=4)
    obs4, info4 = env4.reset(seed=10)
    assert env4.stage == 4
    assert len(env4.static_obstacles) == 6  # 6 pillars
    assert len(env4.internal_walls) == 0
    assert env4.scenario_state == "CLUTTER"
    assert info4["stage"] == 4
    env4.close()


def test_runtime_stage_switching():
    """Verify reset(options={'stage': N}) transitions environment stage dynamically."""
    env = AckBotFollowEnv(stage=1)
    env.reset(seed=42)
    assert env.stage == 1

    env.reset(options={"stage": 3})
    assert env.stage == 3
    assert len(env.internal_walls) == 3

    env.reset(options={"stage": 2})
    assert env.stage == 2
    assert len(env.internal_walls) == 2
    env.close()


# ---------------------------------------------------------------------------
# 4. Blind Corner Visual Occlusion Event (Stage 2)
# ---------------------------------------------------------------------------
def test_blind_corner_occlusion_event():
    """Verify that Stage 2 L-wall corner blocks camera line-of-sight and raises occlusion flag."""
    env = AckBotFollowEnv(stage=2)
    obs, info = env.reset(seed=42)

    # Robot at (1.2, -1.2), target at (1.2, 0.8) -> clean line of sight in east hallway
    assert info["target_visible"] is True
    assert info["is_occluded"] is False

    # Move target behind the L-wall divider at (-1.5, 2.6) while robot stays at (1.2, -1.2)
    env.human.x = -1.5
    env.human.y = 2.6
    env.human.yaw = math.pi
    env.target_x = env.human.x
    env.target_y = env.human.y

    # Execute a neutral step
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    # Line of sight between (1.2, -1.2) and (-1.5, 2.6) intersects the wall (0, -3) to (0, 2)
    # Therefore, camera must report target not visible and tracker must be occluded
    assert info["target_visible"] is False
    assert info["is_occluded"] is True
    assert info["tracker_status"]["occlusion_steps"] >= 1
    env.close()


# ---------------------------------------------------------------------------
# 5. Dead-End U-Turn Sequence (Stage 3)
# ---------------------------------------------------------------------------
def test_deadend_uturn_trajectory():
    """Verify Stage 3 target halts at dead-end, executes 180° turn, and walks back."""
    env = AckBotFollowEnv(stage=3)
    env.reset(seed=42)

    # Fast-forward target near the dead-end wall at x=2.88
    env.human.x = 2.88
    env.human.y = 0.0
    env.human.yaw = 0.0
    env.scenario_state = "TO_DEADEND"

    # Step until target reaches halt condition
    for _ in range(5):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        if env.scenario_state == "UTURN_HALT":
            break

    assert env.scenario_state == "UTURN_HALT"
    assert env.human.speed == 0.0

    # Step through halt timer into pivot
    for _ in range(25):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        if env.scenario_state in ("UTURN_PIVOT", "WALK_BACK"):
            break

    assert env.scenario_state in ("UTURN_PIVOT", "WALK_BACK")

    # Step until pivot completes into WALK_BACK heading west
    for _ in range(60):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        if env.scenario_state == "WALK_BACK":
            break

    assert env.scenario_state == "WALK_BACK"
    assert math.isclose(abs(env.human.yaw), math.pi, abs_tol=0.2)
    assert env.human.vx < 0.0  # Walking back towards robot
    env.close()


# ---------------------------------------------------------------------------
# 6. Target Registration Tracker & EKF Occlusion Projection
# ---------------------------------------------------------------------------
def test_target_tracker_ekf_occlusion_prediction():
    """Verify EKF propagates track state during complete visual occlusion and recovers upon re-detection."""
    tracker = TargetRegistrationTracker(dt=0.05)
    target_id = tracker.register_target(x=2.0, y=0.0, vx=0.30, vy=0.0, color_rgb=(0.8, 0.2, 0.2))

    assert tracker.is_registered is True
    assert tracker.is_occluded is False
    assert tracker.target_id == target_id

    # Simulate 5 visible measurement updates along line y=0 moving at 0.30 m/s
    for step in range(1, 6):
        meas_x = 2.0 + 0.30 * (step * 0.05)
        meas_y = 0.0
        success = tracker.update(meas_x, meas_y)
        assert success is True

    status_before_occ = tracker.get_status()
    assert status_before_occ["is_occluded"] is False
    assert status_before_occ["confidence"] > 0.85

    # Simulate 1.0 second (20 steps) of complete visual occlusion
    for _ in range(20):
        tracker.predict()

    status_during_occ = tracker.get_status()
    assert status_during_occ["is_occluded"] is True
    assert status_during_occ["occlusion_steps"] == 20
    # Position should have projected forward by ~0.30 * 20 * 0.05 = 0.30 m
    est_x, est_y = status_during_occ["estimated_pos"]
    expected_x = (2.0 + 0.30 * 0.25) + 0.30 * (20 * 0.05)
    assert math.isclose(est_x, expected_x, abs_tol=0.05)
    # Confidence should have decayed
    assert status_during_occ["confidence"] < status_before_occ["confidence"]

    # Re-acquire target: observation passes Mahalanobis gate and updates filter
    reacq_success = tracker.update(expected_x + 0.02, 0.01)
    assert reacq_success is True

    status_recovered = tracker.get_status()
    assert status_recovered["is_occluded"] is False
    assert status_recovered["occlusion_steps"] == 0
    assert status_recovered["confidence"] > status_during_occ["confidence"]


def test_tracker_mahalanobis_gating_rejects_spurious_hits():
    """Verify Mahalanobis gating rejects distant obstacle detections (> 9.21 gate)."""
    tracker = TargetRegistrationTracker(dt=0.05)
    tracker.register_target(x=2.0, y=0.0, vx=0.0, vy=0.0)

    # Distant measurement 5.0m away (spurious wall reflection)
    rejected = tracker.update(7.0, 5.0)
    assert rejected is False  # Rejected by gating!


# ---------------------------------------------------------------------------
# 7. Stage 3 Reverse & Yield Encounter Reward
# ---------------------------------------------------------------------------
def test_stage3_reverse_and_yield_reward():
    """Verify Stage 3 yields bonus reward when reversing away from oncoming walker."""
    env = AckBotFollowEnv(stage=3)
    env.reset(seed=42)

    # Set up oncoming encounter: human at x=1.0 walking west, robot at x=-0.5
    env.scenario_state = "WALK_BACK"
    env.human.x = 1.0
    env.human.y = 0.0
    env.human.yaw = math.pi
    env.human.vx = -0.25
    env.human.vy = 0.0
    env.target_x = 1.0
    env.target_y = 0.0
    env.robot_x = -0.5
    env.robot_y = 0.0
    env.robot_yaw = 0.0

    # Action 1: Reverse cleanly (v = -0.30 m/s)
    rev_action = np.array([-0.30, 0.0], dtype=np.float32)
    _, rev_reward, _, _, _ = env.step(rev_action)

    # Reset same state
    env.human.x = 1.0
    env.human.y = 0.0
    env.human.yaw = math.pi
    env.human.vx = -0.25
    env.target_x = 1.0
    env.target_y = 0.0
    env.robot_x = -0.5
    env.robot_y = 0.0
    env.robot_yaw = 0.0
    env.robot_v = 0.0

    # Action 2: Charge forward into oncoming walker (v = +0.30 m/s)
    fwd_action = np.array([+0.30, 0.0], dtype=np.float32)
    _, fwd_reward, _, _, _ = env.step(fwd_action)

    # Reversing when human approaches should yield significantly higher reward than charging forward
    assert rev_reward > fwd_reward + 1.5
    env.close()


# ---------------------------------------------------------------------------
# 8. Auto-Curriculum Progression & Replay Mixing
# ---------------------------------------------------------------------------
def test_auto_curriculum_accumulates_full_episode_return():
    """Verify auto_curriculum accumulates true episode return across steps, not just terminal step reward."""
    env = AckBotFollowEnv(auto_curriculum=True, stage=1, max_episode_steps=5)
    env.reset(seed=42)

    total_stepped_reward = 0.0
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.array([0.1, 0.0], dtype=np.float32))
        total_stepped_reward += reward
        if terminated or truncated:
            break

    assert len(env.curriculum_history) == 1
    # curriculum_history must store the accumulated episode return
    assert math.isclose(env.curriculum_history[0], total_stepped_reward, rel_tol=1e-4)
    env.close()


def test_stage_graduation_thresholds():
    """Verify agent graduates when rolling episode return exceeds stage threshold."""
    env = AckBotFollowEnv(auto_curriculum=True, stage=1, max_episode_steps=5)
    env.reset(seed=42)
    assert env.unlocked_stage == 1

    # Simulate 20 high-performing episodes with mean return > 380
    env.curriculum_history = [400.0] * 19
    env.episode_return = 410.0
    # Step into truncation
    env.current_step = env.max_episode_steps - 1
    obs, reward, terminated, truncated, info = env.step(np.array([0.1, 0.0], dtype=np.float32))

    assert truncated is True
    assert info["stage_graduated"] is True
    assert env.unlocked_stage == 2
    assert env.stage == 2
    assert len(env.curriculum_history) == 0  # Cleared upon graduation
    env.close()


def test_replay_stage_sampling():
    """Verify resets sample previous stages with the configured replay ratio once unlocked."""
    env = AckBotFollowEnv(auto_curriculum=True, stage=1)
    env.reset(seed=42)
    env.unlocked_stage = 3  # Unlocked up to stage 3
    env.replay_previous_ratio = 0.5  # 50% replay for strong statistical signal

    sampled_stages = []
    for s in range(60):
        env.reset(seed=100 + s)
        sampled_stages.append(env.stage)

    # Must contain stage 3 (the unlocked frontier) and earlier stages (1 or 2)
    assert 3 in sampled_stages
    assert (1 in sampled_stages) or (2 in sampled_stages)
    # No stages beyond unlocked
    assert max(sampled_stages) <= 3
    env.close()


