"""
Gymnasium environment for 4-wheel Ackermann robot ("AckBot") target following.
Conforms strictly to physical specifications in spec.txt and robot_params.yaml.

Features:
- 4-Stage Curriculum Progression:
    Stage 1: Open Pursuit (No obstacles, pure tracking)
    Stage 2: Blind Corner Occlusion (L-wall divider, target disappears from camera FOV)
    Stage 3: Dead-End Corridor & U-Turn Encounter (Target turns 180 deg and walks past robot)
    Stage 4: Full Complex Clutter (6 static pillars + arena boundary)
- Realistic Bipedal Human Walker with alternating 2-foot gait dynamics (cadence 1.6 Hz)
- Dynamic Clothing and Appearance RGB color variation randomized per episode
- Target Registration Tracker (Extended Kalman Filter tracking with blind-corner occlusion prediction)
- Pure 426-D sensor observation space (64 depth + 360 LiDAR + 2 state)
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_ws.common.config import get_robot_config
from rl_ws.gym_ackbot.envs.human_target import BipedalHumanTarget
from rl_ws.policies.tracking.target_tracker import TargetRegistrationTracker

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "robot_params.yaml"


def intersect_ray_segment(
    x0: float,
    y0: float,
    cos_a: float,
    sin_a: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Compute intersection distance of 2D ray (origin x0, y0, unit dir cos_a, sin_a)
    against a line segment from (x1, y1) to (x2, y2).
    Returns distance t > 0 if intersection occurs, else infinity.
    """
    sx = x2 - x1
    sy = y2 - y1
    denom = cos_a * sy - sin_a * sx
    if abs(denom) < 1e-6:
        return float("inf")
    dx = x1 - x0
    dy = y1 - y0
    t = (dx * sy - dy * sx) / denom
    u = (dx * sin_a - dy * cos_a) / denom
    if t > 0.0 and 0.0 <= u <= 1.0:
        return t
    return float("inf")


class AckBotFollowEnv(gym.Env):
    """
    AckBot target-following environment with Ackermann steering,
    360-degree 2D LiDAR, and relative target tracking.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        arena_size: float = 10.0,
        max_episode_steps: int = 500,
        stage: int = 1,
        auto_curriculum: bool = False,
        num_distractors: int = 0,
        view_mode: str = "3d",
        min_episodes_per_stage: int = 20,
        stage_thresholds: dict[int, float] | None = None,
        replay_previous_ratio: float = 0.30,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.view_mode = str(view_mode).lower()
        self.radar_sweep_angle = 0.0
        self.arena_size = float(arena_size)  # -5m to +5m
        self.max_episode_steps = max_episode_steps
        self.stage = int(stage)
        self.unlocked_stage = int(stage)
        self.auto_curriculum = bool(auto_curriculum)
        self.num_distractors = int(num_distractors)
        self.distractors: list[BipedalHumanTarget] = []
        self.min_episodes_per_stage = int(min_episodes_per_stage)
        self.stage_thresholds = (
            {int(k): float(v) for k, v in stage_thresholds.items()}
            if stage_thresholds is not None
            else {1: 380.0, 2: 320.0, 3: 300.0}
        )
        self.replay_previous_ratio = float(replay_previous_ratio)
        self.curriculum_history: list[float] = []
        self.episode_return = 0.0
        self.stage_graduated = False

        # Load canonical physical parameters
        self._load_robot_params()

        # Simulation time step: 20 Hz
        self.dt = self.cfg.dt

        # Desired following distance
        self.target_follow_distance = self.cfg.target_follow_distance  # meters

        # --- Front RGB-D Camera Parameters ---
        self.cam_pos_x = self.cfg.cam_pos_x
        self.cam_fov_deg = self.cfg.cam_fov_deg
        self.cam_fov_rad = self.cfg.cam_fov_rad
        self.cam_samples = self.cfg.cam_samples
        self.cam_range_min = self.cfg.cam_range_min  # 0.15 m from spec.txt
        self.cam_range_max = self.cfg.cam_range_max
        self.cam_ray_angles = np.linspace(
            -self.cam_fov_rad / 2.0, self.cam_fov_rad / 2.0, self.cam_samples, endpoint=True
        )

        # --- Action Space ---
        # Action: [target_velocity, target_steering_angle]
        self.action_space = spaces.Box(
            low=np.array([self.v_min, -self.max_steering_angle], dtype=np.float32),
            high=np.array([self.v_max, self.max_steering_angle], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Pure Sensor Observation Space ---
        # 64 camera depth + 360 LiDAR + 2 vehicle state = 426 features
        self.obs_dim = self.cam_samples + self.lidar_samples + 2
        obs_low = np.full(self.obs_dim, 0.0, dtype=np.float32)
        obs_high = np.full(self.obs_dim, 1.0, dtype=np.float32)
        obs_low[-2:] = -1.0
        obs_high[-2:] = 1.0

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.ray_angles = np.linspace(0.0, 2.0 * np.pi, self.lidar_samples, endpoint=False)

        # Internal geometry containers
        self.static_obstacles: list[tuple[float, float, float]] = []
        self.internal_walls: list[tuple[float, float, float, float]] = []

        # Bipedal Human Walker and Target Registration Tracker
        self.human = BipedalHumanTarget(cadence_hz=1.60, stride_length=0.45, stance_width=0.22)
        self.tracker = TargetRegistrationTracker(dt=self.dt)

        # Robot Internal State
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_steer = 0.0

        # Legacy convenience properties for target state
        self.target_x = 2.0
        self.target_y = 0.0
        self.target_vx = 0.25
        self.target_vy = 0.0

        # Scenario state variables
        self.scenario_state = "NORMAL"
        self.scenario_timer = 0
        self.prev_target_dist = 2.0
        self.prev_angle_to_cam = 0.0

        self.current_step = 0
        self.last_action = np.zeros(2, dtype=np.float32)

        # Rendering
        self._renderer = None
        self.window = None
        self.clock = None
        self.window_size = 700
        self.last_collision = False

    def _load_robot_params(self):
        self.cfg = get_robot_config(CONFIG_PATH if CONFIG_PATH.exists() else None)
        self.wheelbase = self.cfg.wheelbase
        self.chassis_length = self.cfg.chassis_length
        self.chassis_width = self.cfg.chassis_width
        self.overall_width = self.cfg.overall_width
        self.min_turning_radius = self.cfg.min_turning_radius
        self.max_steering_angle = self.cfg.max_steering_angle
        self.v_max = self.cfg.v_max
        self.v_min = self.cfg.v_min
        self.max_steer_rate = self.cfg.max_steering_rate

        self.lidar_pos_x = self.cfg.lidar_pos_x
        self.lidar_range_max = self.cfg.lidar_range_max
        self.lidar_range_min = self.cfg.lidar_range_min
        self.lidar_samples = self.cfg.lidar_samples

        self.max_accel = 1.5  # m/s^2

    def set_unlocked_stage(self, stage: int) -> None:
        """Synchronize unlocked curriculum frontier across vectorized workers."""
        new_unlocked = max(1, min(4, int(stage)))
        if new_unlocked != self.unlocked_stage:
            self.unlocked_stage = new_unlocked
            self.curriculum_history.clear()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_collision = False
        self.episode_return = 0.0
        self.stage_graduated = False

        if options and "stage" in options:
            self.stage = int(options["stage"])
            self.unlocked_stage = max(self.unlocked_stage, self.stage)
        elif self.auto_curriculum and self.unlocked_stage > 1:
            # Replay mixing: 70% current unlocked stage, 30% uniform over previous stages
            if self.np_random.random() < self.replay_previous_ratio:
                self.stage = int(self.np_random.integers(1, self.unlocked_stage))
            else:
                self.stage = self.unlocked_stage
        else:
            self.stage = self.unlocked_stage

        if options and "num_distractors" in options:
            self.num_distractors = int(options["num_distractors"])

        # Configure geometry & spawns according to Curriculum Stage
        if self.stage == 1:
            # Stage 1: Open Pursuit (Clean arena, no obstacles)
            self.static_obstacles = []
            self.internal_walls = []

            self.robot_x = float(self.np_random.uniform(-1.0, 1.0))
            self.robot_y = float(self.np_random.uniform(-1.0, 1.0))
            self.robot_yaw = float(self.np_random.uniform(-np.pi, np.pi))
            self.robot_v = 0.0
            self.robot_steer = 0.0

            target_dist = float(self.np_random.uniform(1.5, 2.2))
            target_angle = self.robot_yaw + float(self.np_random.uniform(-0.25, 0.25))
            tx = self.robot_x + target_dist * math.cos(target_angle)
            ty = self.robot_y + target_dist * math.sin(target_angle)
            tyaw = self.robot_yaw + float(self.np_random.uniform(-0.2, 0.2))

            self.human.reset(
                x=tx,
                y=ty,
                yaw=tyaw,
                speed=float(self.np_random.uniform(0.20, 0.30)),
                rng=self.np_random,
            )
            self.scenario_state = "OPEN"

        elif self.stage == 2:
            # Stage 2: Blind Corner Occlusion (L-Wall divider)
            # Wall 1: vertical divider x=0.0, y in [-3.0, 2.0]
            # Wall 2: horizontal corner x in [-3.0, 0.0], y=2.0
            self.static_obstacles = []
            self.internal_walls = [
                (0.0, -3.0, 0.0, 2.0),
                (0.0, 2.0, -3.0, 2.0),
            ]

            # Robot starts in east hallway facing north
            self.robot_x = 1.2
            self.robot_y = -1.2
            self.robot_yaw = math.pi / 2.0  # Facing +Y
            self.robot_v = 0.0
            self.robot_steer = 0.0

            # Target starts ahead in hallway facing north
            tx = 1.2
            ty = 0.8
            tyaw = math.pi / 2.0
            self.human.reset(x=tx, y=ty, yaw=tyaw, speed=0.26, rng=self.np_random)
            self.scenario_state = "APPROACH_CORNER"

        elif self.stage == 3:
            # Stage 3: Dead-End Corridor & U-Turn Encounter
            # Corridor width: 2.4m (y in [-1.2, 1.2]) extending x in [-3.5, 3.5], dead-end at x=3.5
            self.static_obstacles = []
            self.internal_walls = [
                (-3.5, 1.2, 3.5, 1.2),
                (-3.5, -1.2, 3.5, -1.2),
                (3.5, -1.2, 3.5, 1.2),  # Dead-end wall
            ]

            # Robot starts in corridor facing east (+X)
            self.robot_x = -1.2
            self.robot_y = 0.0
            self.robot_yaw = 0.0
            self.robot_v = 0.0
            self.robot_steer = 0.0

            # Target starts ahead facing east (+X) towards dead-end
            tx = 1.0
            ty = 0.0
            tyaw = 0.0
            self.human.reset(x=tx, y=ty, yaw=tyaw, speed=0.25, rng=self.np_random)
            self.scenario_state = "TO_DEADEND"
            self.scenario_timer = 0

        else:
            # Stage 4: Full Complex Clutter (6 static cylindrical pillars)
            self.static_obstacles = [
                (1.8, 1.8, 0.35),
                (-2.0, 1.5, 0.40),
                (2.0, -2.0, 0.45),
                (-2.2, -1.8, 0.35),
                (0.0, 3.0, 0.40),
                (0.0, -3.0, 0.40),
            ]
            self.internal_walls = []

            self.robot_x = float(self.np_random.uniform(-1.0, 1.0))
            self.robot_y = float(self.np_random.uniform(-1.0, 1.0))
            self.robot_yaw = float(self.np_random.uniform(-np.pi, np.pi))
            self.robot_v = 0.0
            self.robot_steer = 0.0

            target_dist = float(self.np_random.uniform(1.5, 2.5))
            target_angle = self.robot_yaw + float(self.np_random.uniform(-0.35, 0.35))
            tx = self.robot_x + target_dist * math.cos(target_angle)
            ty = self.robot_y + target_dist * math.sin(target_angle)
            bound = (self.arena_size / 2.0) - 0.6
            tx = float(np.clip(tx, -bound, bound))
            ty = float(np.clip(ty, -bound, bound))
            tyaw = self.robot_yaw + float(self.np_random.uniform(-0.3, 0.3))

            self.human.reset(
                x=tx,
                y=ty,
                yaw=tyaw,
                speed=float(self.np_random.uniform(0.18, 0.30)),
                rng=self.np_random,
            )
            self.scenario_state = "CLUTTER"

        # Synchronize target coordinates
        self.target_x = self.human.x
        self.target_y = self.human.y
        self.target_vx = self.human.vx
        self.target_vy = self.human.vy
        self.prev_target_dist = math.hypot(
            self.target_x - self.robot_x, self.target_y - self.robot_y
        )
        half_l = self.chassis_length / 2.0
        cam_x = self.robot_x + half_l * math.cos(self.robot_yaw)
        cam_y = self.robot_y + half_l * math.sin(self.robot_yaw)
        init_angle = math.atan2(self.target_y - cam_y, self.target_x - cam_x) - self.robot_yaw
        init_angle = (init_angle + np.pi) % (2.0 * np.pi) - np.pi
        self.prev_angle_to_cam = abs(init_angle)

        # Spawn distractors if enabled
        self.distractors = []
        if self.num_distractors > 0:
            from rl_ws.gym_ackbot.envs.human_target import CLOTHING_PALETTE

            available_colors = [c for c in CLOTHING_PALETTE if c != self.human.color_rgb]
            if not available_colors:
                available_colors = CLOTHING_PALETTE

            bound = (self.arena_size / 2.0) - 1.2
            for d_idx in range(self.num_distractors):
                dx = float(self.np_random.uniform(-bound, bound))
                dy = float(self.np_random.uniform(-bound, bound))
                if math.hypot(dx - self.robot_x, dy - self.robot_y) < 1.5:
                    dx += math.copysign(2.0, dx if abs(dx) > 0.1 else 1.0)
                    dx = float(np.clip(dx, -bound, bound))
                dyaw = float(self.np_random.uniform(-np.pi, np.pi))
                dspeed = float(self.np_random.uniform(0.18, 0.28))
                dcolor = available_colors[d_idx % len(available_colors)]
                d_target = BipedalHumanTarget(
                    x=dx,
                    y=dy,
                    yaw=dyaw,
                    speed=dspeed,
                    cadence_hz=1.55,
                    stride_length=0.42,
                    stance_width=0.22,
                    color_rgb=dcolor,
                    rng=self.np_random,
                )
                self.distractors.append(d_target)

        # Register target in tracker
        self.tracker.register_target(
            x=self.target_x,
            y=self.target_y,
            vx=self.target_vx,
            vy=self.target_vy,
            color_rgb=self.human.color_rgb,
            target_id="target_human",
        )

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_v, target_steer = float(action[0]), float(action[1])

        # 1. Apply physical actuator dynamics
        delta_v = target_v - self.robot_v
        max_delta_v = self.max_accel * self.dt
        self.robot_v += math.copysign(min(abs(delta_v), max_delta_v), delta_v)

        delta_steer = target_steer - self.robot_steer
        max_delta_steer = self.max_steer_rate * self.dt
        self.robot_steer += math.copysign(min(abs(delta_steer), max_delta_steer), delta_steer)

        # 2. Ackermann Kinematics
        yaw_rate = (self.robot_v / self.wheelbase) * math.tan(self.robot_steer)
        self.robot_yaw += yaw_rate * self.dt
        self.robot_yaw = (self.robot_yaw + np.pi) % (2.0 * np.pi) - np.pi

        self.robot_x += self.robot_v * math.cos(self.robot_yaw) * self.dt
        self.robot_y += self.robot_v * math.sin(self.robot_yaw) * self.dt

        # 3. Update Target Motion & Bipedal Gait
        self._update_target_motion()
        self.human.step(self.dt)
        self.target_x = self.human.x
        self.target_y = self.human.y
        self.target_vx = self.human.vx
        self.target_vy = self.human.vy

        # 4. Check Collisions (Arena boundaries, obstacles, internal walls, human body)
        is_collision, collision_type = self._check_collision()
        self.last_collision = is_collision

        # 5. Compute Observation & Info
        obs = self._get_observation()
        info = self._get_info()
        info["collision"] = is_collision
        info["collision_type"] = collision_type

        # 6. Target Registration Tracker Update / Multi-Person Candidate Association
        half_l = self.chassis_length / 2.0
        cam_x = self.robot_x + half_l * math.cos(self.robot_yaw)
        cam_y = self.robot_y + half_l * math.sin(self.robot_yaw)
        tdx = self.target_x - cam_x
        tdy = self.target_y - cam_y
        angle_to_cam = math.atan2(tdy, tdx) - self.robot_yaw
        angle_to_cam = (angle_to_cam + np.pi) % (2.0 * np.pi) - np.pi
        target_cam_dist = math.hypot(tdx, tdy)

        # Check line-of-sight visibility against internal walls for registered target
        is_los_blocked = False
        if abs(angle_to_cam) <= (self.cam_fov_rad / 2.0) and target_cam_dist <= self.cam_range_max:
            cos_los = math.cos(self.robot_yaw + angle_to_cam)
            sin_los = math.sin(self.robot_yaw + angle_to_cam)
            for x1, y1, x2, y2 in self.internal_walls:
                dist_wall = intersect_ray_segment(cam_x, cam_y, cos_los, sin_los, x1, y1, x2, y2)
                if dist_wall < (target_cam_dist - 0.2):
                    is_los_blocked = True
                    break

        in_fov = (
            abs(angle_to_cam) <= (self.cam_fov_rad / 2.0)
            and (target_cam_dist <= self.cam_range_max)
            and not is_los_blocked
        )
        info["target_in_fov"] = in_fov
        info["line_of_sight_blocked"] = is_los_blocked

        # Collect candidate detections from all humans in camera FOV
        candidates = []
        all_humans = [self.human] + self.distractors
        for h_idx, person in enumerate(all_humans):
            p_dx = person.x - cam_x
            p_dy = person.y - cam_y
            p_ang = math.atan2(p_dy, p_dx) - self.robot_yaw
            p_ang = (p_ang + np.pi) % (2.0 * np.pi) - np.pi
            p_dist = math.hypot(p_dx, p_dy)

            p_los_blocked = False
            if abs(p_ang) <= (self.cam_fov_rad / 2.0) and p_dist <= self.cam_range_max:
                p_cos = math.cos(self.robot_yaw + p_ang)
                p_sin = math.sin(self.robot_yaw + p_ang)
                for x1, y1, x2, y2 in self.internal_walls:
                    d_w = intersect_ray_segment(cam_x, cam_y, p_cos, p_sin, x1, y1, x2, y2)
                    if d_w < (p_dist - 0.2):
                        p_los_blocked = True
                        break

            if (
                abs(p_ang) <= (self.cam_fov_rad / 2.0)
                and p_dist <= self.cam_range_max
                and not p_los_blocked
            ):
                candidates.append(
                    {
                        "x": person.x,
                        "y": person.y,
                        "color_rgb": person.color_rgb,
                        "id": "target" if person is self.human else f"distractor_{h_idx}",
                        "is_target": bool(person is self.human),
                    }
                )

        assoc = self.tracker.update_with_candidates(candidates)
        tracker_stat = self.tracker.get_status(self.robot_x, self.robot_y, self.robot_yaw)

        info["target_visible"] = in_fov
        info["is_occluded"] = tracker_stat["is_occluded"]
        info["target_registered"] = tracker_stat["registered"]
        info["target_occluded"] = tracker_stat["is_occluded"]
        info["tracker_estimated_dist"] = tracker_stat["estimated_dist"]
        info["tracker_confidence"] = tracker_stat["confidence"]
        info["tracker_status"] = tracker_stat
        info["human_color"] = self.human.color_rgb
        info["stage"] = self.stage
        info["association_result"] = assoc
        info["num_candidates_in_fov"] = len(candidates)
        info["target_locked"] = bool(
            assoc["associated"]
            and assoc["selected_candidate"]
            and assoc["selected_candidate"]["is_target"]
        )
        info["distractor_false_lock"] = bool(
            assoc["associated"]
            and assoc["selected_candidate"]
            and not assoc["selected_candidate"]["is_target"]
        )

        # 7. Compute Reward with Continuous 360° Bearing & Kinematic Recovery Shaping
        target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        dist_change = self.prev_target_dist - target_dist
        self.prev_target_dist = target_dist

        abs_angle = abs(angle_to_cam)
        angle_change = self.prev_angle_to_cam - abs_angle
        self.prev_angle_to_cam = abs_angle

        reward_progress = 4.0 * dist_change
        raw_reward_dist = 1.0 - min(1.0, abs(target_dist - self.target_follow_distance) / 3.0)

        # Tailgating Slot Reward
        target_heading = math.atan2(self.target_vy, self.target_vx)
        slot_x = self.target_x - self.target_follow_distance * math.cos(target_heading)
        slot_y = self.target_y - self.target_follow_distance * math.sin(target_heading)
        dist_to_slot = math.hypot(self.robot_x - slot_x, self.robot_y - slot_y)
        raw_reward_slot = 1.0 - min(1.0, dist_to_slot / 3.0)

        # Gate distance/slot rewards strictly by camera alignment when LOS is unobstructed
        # (exempt Stage 2 blind corner occlusion and Stage 3 WALK_BACK oncoming yield)
        if is_los_blocked or (self.stage == 3 and self.scenario_state == "WALK_BACK"):
            orient_gate = 1.0
        else:
            orient_gate = max(0.0, math.cos(angle_to_cam))

        reward_dist = orient_gate * raw_reward_dist
        reward_slot = orient_gate * raw_reward_slot

        # Continuous Visual / Bearing Lock Reward across full [-pi, pi]
        if in_fov:
            reward_visual = 0.60 * math.cos(angle_to_cam)
        elif is_los_blocked:
            reward_visual = -0.10
        else:
            reward_visual = 0.45 * math.cos(angle_to_cam) - 0.15

        # Potential-based angular tracking reward (rewards reducing |angle_to_cam|)
        reward_turn = 1.5 * angle_change

        if target_dist > self.target_follow_distance and in_fov:
            reward_forward = (
                0.5 * max(0.0, self.robot_v / self.v_max) * max(0.0, math.cos(angle_to_cam))
            )
        else:
            reward_forward = 0.0

        # Blend heading alignment from target bearing (far) to target velocity heading (in slot)
        bearing_align = max(0.0, math.cos(angle_to_cam))
        vel_align = max(0.0, math.cos(self.robot_yaw - target_heading))
        slot_blend = max(0.0, min(1.0, (target_dist - 0.9) / 1.0))
        reward_heading = 0.3 * (slot_blend * bearing_align + (1.0 - slot_blend) * vel_align)

        # Ackermann kinematic recovery when target is out of FOV + close-range overshoot guard
        reward_recovery = 0.0
        if not in_fov and not is_los_blocked:
            if yaw_rate * math.copysign(1.0, angle_to_cam) > 0.05:
                reward_recovery += 0.20
            if abs(self.robot_v) < 0.05 and abs_angle > 0.4:
                reward_recovery -= 0.20
            if self.robot_v > 0.45:
                reward_recovery -= 0.30 * (
                    (self.robot_v - 0.45) / max(1e-3, self.v_max - 0.45)
                )
        elif (
            in_fov
            and target_dist < 0.65
            and not (self.stage == 3 and self.scenario_state == "WALK_BACK")
        ):
            if self.robot_v > 0.10:
                reward_recovery -= 0.50 * (self.robot_v / self.v_max)

        action_diff = action - self.last_action
        reward_smooth = -0.05 * float(np.sum(action_diff**2))
        reward_collision = -25.0 if is_collision else 0.0
        reward_proximity = -2.0 * max(0.0, 0.40 - target_dist)
        reward_step = -0.01

        # Stage 3 Specific: U-Turn Reverse & Yield Encounter Reward
        reward_yield = 0.0
        if self.stage == 3 and self.scenario_state == "WALK_BACK":
            # Human is walking west (-X) towards/past robot
            # If robot is in front of oncoming human:
            if self.target_x > self.robot_x:
                # Reward reversing to maintain clearance buffer
                if self.robot_v < -0.05 and target_dist >= 0.70:
                    reward_yield = +1.5 * abs(self.robot_v / self.v_min)
                elif self.robot_v > 0.05:
                    reward_yield = -1.5  # Penalize charging forward into oncoming human

        reward = (
            reward_progress
            + reward_dist
            + reward_slot
            + reward_forward
            + reward_visual
            + reward_turn
            + reward_heading
            + reward_recovery
            + reward_smooth
            + reward_collision
            + reward_proximity
            + reward_yield
            + reward_step
        )

        self.last_action = action.copy()

        # Accumulate episodic return
        self.episode_return += float(reward)

        terminated = bool(is_collision)
        truncated = bool(self.current_step >= self.max_episode_steps)

        # Automatic Curriculum Progression (frontier episodes only)
        self.stage_graduated = False
        if (terminated or truncated) and self.auto_curriculum:
            if self.stage == self.unlocked_stage:
                self.curriculum_history.append(float(self.episode_return))
                if len(self.curriculum_history) >= self.min_episodes_per_stage:
                    mean_ret = float(
                        np.mean(self.curriculum_history[-self.min_episodes_per_stage :])
                    )
                    threshold = self.stage_thresholds.get(self.unlocked_stage)
                    if (
                        threshold is not None
                        and mean_ret >= threshold
                        and self.unlocked_stage < 4
                    ):
                        self.unlocked_stage += 1
                        self.stage = self.unlocked_stage
                        self.stage_graduated = True
                        self.curriculum_history.clear()

        info["unlocked_stage"] = self.unlocked_stage
        info["stage_graduated"] = self.stage_graduated
        info["episode_return"] = self.episode_return

        return obs, float(reward), terminated, truncated, info

    def _steer_towards(self, person: BipedalHumanTarget, desired_yaw: float, max_turn: float = 0.08) -> None:
        """Smoothly steer a walker toward desired_yaw without unphysical 1-frame snaps."""
        yaw_err = (desired_yaw - person.yaw + math.pi) % (2.0 * math.pi) - math.pi
        person.yaw = (
            person.yaw + max(-max_turn, min(max_turn, yaw_err)) + math.pi
        ) % (2.0 * math.pi) - math.pi
        person.omega = 0.0

    def _update_target_motion(self):
        """Update target trajectory according to active curriculum scenario."""
        if self.stage == 1:
            # Stage 1: Smooth wander with bounded angular velocity & smooth boundary steering
            boundary = (self.arena_size / 2.0) - 0.8
            if abs(self.human.x) > boundary or abs(self.human.y) > boundary:
                self._steer_towards(self.human, math.atan2(-self.human.y, -self.human.x))
            else:
                self.human.omega = float(
                    np.clip(
                        0.85 * self.human.omega + self.np_random.uniform(-0.12, 0.12),
                        -0.50,
                        0.50,
                    )
                )

        elif self.stage == 2:
            # Stage 2: Hallway north -> L-corner turn around (0, 2) -> West traversal
            if self.scenario_state == "APPROACH_CORNER":
                # Walking north along x=1.2
                self.human.yaw = math.pi / 2.0
                if self.human.y >= 2.6:
                    self.scenario_state = "TURN_LEFT"
            elif self.scenario_state == "TURN_LEFT":
                # Pivot left towards -X (yaw -> pi)
                self.human.omega = 0.8
                if abs(self.human.yaw - math.pi) < 0.1 or self.human.yaw < 0:
                    self.human.yaw = math.pi
                    self.human.omega = 0.0
                    self.scenario_state = "WALK_WEST"
            elif self.scenario_state == "WALK_WEST":
                self.human.yaw = math.pi
                if self.human.x <= -3.8:
                    self.human.yaw = 0.0  # Bounce back if hitting arena boundary

        elif self.stage == 3:
            # Stage 3: Corridor east -> Dead-end at x=3.5 -> U-turn 180 deg -> Walk west
            if self.scenario_state == "TO_DEADEND":
                self.human.yaw = 0.0
                if self.human.x >= 2.9:
                    self.scenario_state = "UTURN_HALT"
                    self.scenario_timer = 15  # Pause for ~0.75 seconds
                    self.human.speed = 0.0
            elif self.scenario_state == "UTURN_HALT":
                self.scenario_timer -= 1
                if self.scenario_timer <= 0:
                    self.scenario_state = "UTURN_PIVOT"
                    self.human.omega = 1.2
            elif self.scenario_state == "UTURN_PIVOT":
                # Smoothly rotate 180 degrees
                if abs(self.human.yaw - math.pi) < 0.15 or self.human.yaw < -3.0:
                    self.human.yaw = math.pi
                    self.human.omega = 0.0
                    self.human.speed = 0.25
                    self.scenario_state = "WALK_BACK"
            elif self.scenario_state == "WALK_BACK":
                self.human.yaw = math.pi
                if self.human.x <= -3.2:
                    self.human.speed = 0.0  # Corridor exit reached

        else:
            # Stage 4: Smooth obstacle avoidance and bounded-omega wander
            boundary = (self.arena_size / 2.0) - 0.8
            if abs(self.human.x) > boundary or abs(self.human.y) > boundary:
                self._steer_towards(self.human, math.atan2(-self.human.y, -self.human.x))
            else:
                avoiding = False
                for ox, oy, obs_r in self.static_obstacles:
                    dx = self.human.x - ox
                    dy = self.human.y - oy
                    if math.hypot(dx, dy) < (obs_r + 0.35):
                        self._steer_towards(self.human, math.atan2(dy, dx), max_turn=0.12)
                        avoiding = True
                        break
                if not avoiding:
                    self.human.omega = float(
                        np.clip(
                            0.85 * self.human.omega + self.np_random.uniform(-0.15, 0.15),
                            -0.60,
                            0.60,
                        )
                    )

        # Update distractors motion & periodic bipedal gait
        boundary = (self.arena_size / 2.0) - 0.8
        for d in self.distractors:
            if abs(d.x) > boundary or abs(d.y) > boundary:
                d.yaw = math.atan2(-d.y, -d.x)
            else:
                for ox, oy, obs_r in self.static_obstacles:
                    dx = d.x - ox
                    dy = d.y - oy
                    if math.hypot(dx, dy) < (obs_r + 0.35):
                        d.yaw = math.atan2(dy, dx)
                        break
                d.yaw += float(self.np_random.uniform(-0.06, 0.06))
            d.step(self.dt)

    def _check_collision(self):
        half_len = self.chassis_length / 2.0
        half_wid = self.overall_width / 2.0
        max_bound = (self.arena_size / 2.0) - 0.1

        corners = [
            (half_len, half_wid),
            (half_len, -half_wid),
            (-half_len, half_wid),
            (-half_len, -half_wid),
        ]
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)

        # 1. Outer Arena Boundary Collision
        for cx, cy in corners:
            wx = self.robot_x + cx * cos_y - cy * sin_y
            wy = self.robot_y + cx * sin_y + cy * cos_y
            if abs(wx) >= max_bound or abs(wy) >= max_bound:
                return True, "wall"

        # 2. Internal Walls Collision
        for x1, y1, x2, y2 in self.internal_walls:
            sx = x2 - x1
            sy = y2 - y1
            seg_len_sq = max(sx * sx + sy * sy, 1e-6)
            for cx, cy in corners:
                wx = self.robot_x + cx * cos_y - cy * sin_y
                wy = self.robot_y + cx * sin_y + cy * cos_y
                u = max(0.0, min(1.0, ((wx - x1) * sx + (wy - y1) * sy) / seg_len_sq))
                proj_x = x1 + u * sx
                proj_y = y1 + u * sy
                if math.hypot(wx - proj_x, wy - proj_y) < 0.08:
                    return True, "internal_wall"

        # 3. Static Obstacles Collision (Spheres)
        spheres = [
            (
                self.robot_x + (self.wheelbase / 2.0) * cos_y,
                self.robot_y + (self.wheelbase / 2.0) * sin_y,
                0.20,
            ),
            (
                self.robot_x - (self.wheelbase / 2.0) * cos_y,
                self.robot_y - (self.wheelbase / 2.0) * sin_y,
                0.20,
            ),
        ]
        for ox, oy, obs_r in self.static_obstacles:
            for sx, sy, sr in spheres:
                if math.hypot(ox - sx, oy - sy) < (obs_r + sr):
                    return True, "obstacle"

        # 4. Human Body / Leg Collision (Fatal bump with target or distractors)
        for person in [self.human] + self.distractors:
            left_leg, right_leg = person.get_feet_positions()
            for leg in (left_leg, right_leg):
                for sx, sy, sr in spheres:
                    if math.hypot(leg.x - sx, leg.y - sy) < (leg.radius + sr - 0.05):
                        return True, "human"

        return False, "none"

    def _cast_rays(
        self,
        origin_x: float,
        origin_y: float,
        ray_angles: np.ndarray,
        r_min: float,
        r_max: float,
    ) -> np.ndarray:
        """Vectorized 2D ray casting against arena boundaries, internal walls, circular obstacles, and humans."""
        global_angles = self.robot_yaw + ray_angles
        cos_a = np.cos(global_angles)
        sin_a = np.sin(global_angles)

        r = np.full(len(ray_angles), float(r_max), dtype=np.float64)
        half_arena = self.arena_size / 2.0

        # 1. Outer boundary walls
        pos_x = cos_a > 1e-4
        neg_x = cos_a < -1e-4
        if np.any(pos_x):
            r[pos_x] = np.minimum(r[pos_x], (half_arena - origin_x) / cos_a[pos_x])
        if np.any(neg_x):
            r[neg_x] = np.minimum(r[neg_x], (-half_arena - origin_x) / cos_a[neg_x])

        pos_y = sin_a > 1e-4
        neg_y = sin_a < -1e-4
        if np.any(pos_y):
            r[pos_y] = np.minimum(r[pos_y], (half_arena - origin_y) / sin_a[pos_y])
        if np.any(neg_y):
            r[neg_y] = np.minimum(r[neg_y], (-half_arena - origin_y) / sin_a[neg_y])

        # 2. Internal walls
        for x1, y1, x2, y2 in self.internal_walls:
            sx = x2 - x1
            sy = y2 - y1
            denom = cos_a * sy - sin_a * sx
            valid = np.abs(denom) >= 1e-6
            safe_denom = np.where(valid, denom, 1.0)
            dx = x1 - origin_x
            dy = y1 - origin_y
            t = (dx * sy - dy * sx) / safe_denom
            u = (dx * sin_a - dy * cos_a) / safe_denom
            hit = valid & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
            if np.any(hit):
                r[hit] = np.minimum(r[hit], t[hit])

        # 3. Circular obstacles + Bipedal Human Walker Dual-Leg cylinders
        circles: list[tuple[float, float, float]] = list(self.static_obstacles)
        for person in [self.human] + self.distractors:
            left_leg, right_leg = person.get_feet_positions()
            circles.append((left_leg.x, left_leg.y, left_leg.radius))
            circles.append((right_leg.x, right_leg.y, right_leg.radius))

        if circles:
            arr = np.asarray(circles, dtype=np.float64)  # (C, 3)
            dx = arr[:, 0:1] - origin_x  # (C, 1)
            dy = arr[:, 1:2] - origin_y  # (C, 1)
            r_sq = (arr[:, 2:3]) ** 2  # (C, 1)
            proj = dx * cos_a[None, :] + dy * sin_a[None, :]  # (C, N)
            perp_sq = (dx * dx + dy * dy) - (proj * proj)
            hit = (proj > 0.0) & (perp_sq < r_sq)
            if np.any(hit):
                sqrt_term = np.sqrt(np.maximum(0.0, r_sq - perp_sq))
                dist_hit = np.where(hit, proj - sqrt_term, float(r_max))
                dist_hit = np.where(dist_hit > 0.0, dist_hit, float(r_max))
                r = np.minimum(r, np.min(dist_hit, axis=0))

        return np.clip(r, r_min, r_max).astype(np.float32)

    def _simulate_lidar(self) -> np.ndarray:
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)
        lidar_x = self.robot_x + self.lidar_pos_x * cos_y
        lidar_y = self.robot_y + self.lidar_pos_x * sin_y
        return self._cast_rays(
            lidar_x, lidar_y, self.ray_angles, self.lidar_range_min, self.lidar_range_max
        )

    def _simulate_camera_depth(self) -> np.ndarray:
        half_l = self.chassis_length / 2.0
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)
        cam_x = self.robot_x + half_l * cos_y
        cam_y = self.robot_y + half_l * sin_y
        return self._cast_rays(
            cam_x, cam_y, self.cam_ray_angles, self.cam_range_min, self.cam_range_max
        )

    def _get_observation(self):
        cam_ranges = self._simulate_camera_depth()
        cam_norm = (cam_ranges - self.cam_range_min) / (self.cam_range_max - self.cam_range_min)

        lidar_ranges = self._simulate_lidar()
        lidar_norm = (lidar_ranges - self.lidar_range_min) / (
            self.lidar_range_max - self.lidar_range_min
        )

        state_features = np.array(
            [
                self.robot_v / self.v_max,
                self.robot_steer / self.max_steering_angle,
            ],
            dtype=np.float32,
        )

        return np.concatenate([cam_norm, lidar_norm, state_features], dtype=np.float32)

    def _get_info(self):
        target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        rel_x, rel_y, rel_heading = self._get_relative_target_state()
        half_l = self.chassis_length / 2.0
        cam_x = self.robot_x + half_l * math.cos(self.robot_yaw)
        cam_y = self.robot_y + half_l * math.sin(self.robot_yaw)
        tdx = self.target_x - cam_x
        tdy = self.target_y - cam_y
        angle_to_cam = math.atan2(tdy, tdx) - self.robot_yaw
        angle_to_cam = (angle_to_cam + np.pi) % (2.0 * np.pi) - np.pi
        target_cam_dist = math.hypot(tdx, tdy)

        # Check line-of-sight visibility against internal walls
        is_los_blocked = False
        if abs(angle_to_cam) <= (self.cam_fov_rad / 2.0) and target_cam_dist <= self.cam_range_max:
            cos_los = math.cos(self.robot_yaw + angle_to_cam)
            sin_los = math.sin(self.robot_yaw + angle_to_cam)
            for x1, y1, x2, y2 in self.internal_walls:
                dist_wall = intersect_ray_segment(cam_x, cam_y, cos_los, sin_los, x1, y1, x2, y2)
                if dist_wall < (target_cam_dist - 0.2):
                    is_los_blocked = True
                    break

        in_fov = (
            abs(angle_to_cam) <= (self.cam_fov_rad / 2.0)
            and (target_cam_dist <= self.cam_range_max)
            and not is_los_blocked
        )
        tracker_stat = self.tracker.get_status(self.robot_x, self.robot_y, self.robot_yaw)

        return {
            "target_dist": target_dist,
            "rel_heading_deg": math.degrees(rel_heading),
            "angle_to_cam_deg": math.degrees(angle_to_cam),
            "robot_v": self.robot_v,
            "robot_steer_deg": math.degrees(self.robot_steer),
            "step": self.current_step,
            "stage": self.stage,
            "unlocked_stage": self.unlocked_stage,
            "stage_graduated": self.stage_graduated,
            "target_in_fov": in_fov,
            "target_visible": in_fov,
            "line_of_sight_blocked": is_los_blocked,
            "is_occluded": tracker_stat["is_occluded"],
            "target_registered": tracker_stat["registered"],
            "target_occluded": tracker_stat["is_occluded"],
            "tracker_estimated_dist": tracker_stat["estimated_dist"],
            "tracker_confidence": tracker_stat["confidence"],
            "tracker_status": tracker_stat,
            "human_color": self.human.color_rgb,
        }

    def _get_relative_target_state(self):
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        cos_y = math.cos(-self.robot_yaw)
        sin_y = math.sin(-self.robot_yaw)
        rel_x = dx * cos_y - dy * sin_y
        rel_y = dx * sin_y + dy * cos_y
        rel_heading = math.atan2(rel_y, rel_x)
        return rel_x, rel_y, rel_heading

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from rl_ws.gym_ackbot.envs.renderer import AckBotRenderer

            self._renderer = AckBotRenderer(
                window_size=self.window_size, render_fps=self.metadata["render_fps"]
            )
        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.window = None
        self.clock = None
