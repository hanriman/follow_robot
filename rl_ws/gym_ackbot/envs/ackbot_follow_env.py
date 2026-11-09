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
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
        self.replay_previous_ratio = 0.30
        self.curriculum_history: list[float] = []
        self.episode_return = 0.0
        self.stage_graduated = False

        # Load physical parameters
        self._load_robot_params()

        # Simulation time step: 20 Hz
        self.dt = 0.05

        # Desired following distance
        self.target_follow_distance = 0.80  # meters

        # --- Front RGB-D Camera Parameters ---
        self.cam_pos_x = self.chassis_length / 2.0  # +0.15 m
        self.cam_fov_deg = 85.0
        self.cam_fov_rad = math.radians(self.cam_fov_deg)  # ~1.48353 rad
        self.cam_samples = 64
        self.cam_range_min = 0.10
        self.cam_range_max = 5.00
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

        self.current_step = 0
        self.last_action = np.zeros(2, dtype=np.float32)

        # Rendering
        self.window = None
        self.clock = None
        self.window_size = 700
        self.last_collision = False

    def _load_robot_params(self):
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                params = yaml.safe_load(f)
            chassis = params["chassis"]
            wheels = params["wheels"]
            kin = params["kinematics"]
            lid = params["lidar_2d"]

            self.wheelbase = float(wheels["wheelbase"])  # 0.30 m
            self.chassis_length = float(chassis["length_x"])  # 0.30 m
            self.chassis_width = float(chassis["width_y"])  # 0.20 m
            self.overall_width = float(wheels["overall_width"])  # 0.36 m
            self.min_turning_radius = float(kin["min_turning_radius"])  # 0.485 m
            self.max_steering_angle = float(kin["max_steering_angle"])  # 0.5540 rad (~31.74 deg)
            self.v_max = float(kin["max_linear_speed"])  # 0.80 m/s
            self.v_min = float(kin["min_linear_speed"])  # -0.40 m/s
            self.max_steer_rate = float(kin["max_steering_rate"])  # 1.047 rad/s (60 deg/s)

            self.lidar_pos_x = float(lid["pos_x"])  # -0.15 m (rear axle)
            self.lidar_range_max = float(lid["range_max"])  # 3.00 m
            self.lidar_range_min = float(lid["range_min"])  # 0.10 m
            self.lidar_samples = int(lid["samples"])  # 360
        else:
            self.wheelbase = 0.30
            self.chassis_length = 0.30
            self.chassis_width = 0.20
            self.overall_width = 0.36
            self.min_turning_radius = 0.485
            self.max_steering_angle = 0.553953
            self.v_max = 0.80
            self.v_min = -0.40
            self.max_steer_rate = 1.047198
            self.lidar_pos_x = -0.15
            self.lidar_range_max = 3.00
            self.lidar_range_min = 0.10
            self.lidar_samples = 360

        self.max_accel = 1.5  # m/s^2

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
        self.prev_target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)

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

            if abs(p_ang) <= (self.cam_fov_rad / 2.0) and p_dist <= self.cam_range_max and not p_los_blocked:
                candidates.append({
                    "x": person.x,
                    "y": person.y,
                    "color_rgb": person.color_rgb,
                    "id": "target" if person is self.human else f"distractor_{h_idx}",
                    "is_target": bool(person is self.human),
                })

        assoc = self.tracker.update_with_candidates(candidates)
        tracker_stat = self.tracker.get_status()

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
        info["target_locked"] = bool(assoc["associated"] and assoc["selected_candidate"] and assoc["selected_candidate"]["is_target"])
        info["distractor_false_lock"] = bool(assoc["associated"] and assoc["selected_candidate"] and not assoc["selected_candidate"]["is_target"])

        # 7. Compute Reward with Stage-Specific Shaping
        target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        dist_change = self.prev_target_dist - target_dist
        self.prev_target_dist = target_dist

        reward_progress = 4.0 * dist_change
        reward_dist = 1.0 - min(1.0, abs(target_dist - self.target_follow_distance) / 3.0)

        # Tailgating Slot Reward
        target_heading = math.atan2(self.target_vy, self.target_vx)
        slot_x = self.target_x - self.target_follow_distance * math.cos(target_heading)
        slot_y = self.target_y - self.target_follow_distance * math.sin(target_heading)
        dist_to_slot = math.hypot(self.robot_x - slot_x, self.robot_y - slot_y)
        reward_slot = 1.0 - min(1.0, dist_to_slot / 3.0)

        # Visual Lock Reward
        if in_fov:
            reward_visual = 0.5 * math.cos(angle_to_cam)
        else:
            reward_visual = -0.5

        if target_dist > self.target_follow_distance and in_fov:
            reward_forward = 0.5 * max(0.0, self.robot_v / self.v_max) * max(0.0, math.cos(angle_to_cam))
        else:
            reward_forward = 0.0

        reward_heading = 0.3 * max(0.0, math.cos(self.robot_yaw - target_heading))
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
            + reward_heading
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

        # Automatic Curriculum Progression
        self.stage_graduated = False
        if (terminated or truncated) and self.auto_curriculum:
            self.curriculum_history.append(float(self.episode_return))
            if len(self.curriculum_history) >= 20:
                mean_ret = float(np.mean(self.curriculum_history[-20:]))
                should_graduate = False
                if self.unlocked_stage == 1 and mean_ret >= 380.0:
                    should_graduate = True
                elif self.unlocked_stage == 2 and mean_ret >= 320.0:
                    should_graduate = True
                elif self.unlocked_stage == 3 and mean_ret >= 300.0:
                    should_graduate = True

                if should_graduate and self.unlocked_stage < 4:
                    self.unlocked_stage += 1
                    self.stage = self.unlocked_stage
                    self.stage_graduated = True
                    self.curriculum_history.clear()

        info["unlocked_stage"] = self.unlocked_stage
        info["stage_graduated"] = self.stage_graduated
        info["episode_return"] = self.episode_return

        return obs, float(reward), terminated, truncated, info

    def _update_target_motion(self):
        """Update target trajectory according to active curriculum scenario."""
        if self.stage == 1:
            # Stage 1: Smooth wander with outer boundary bounce
            boundary = (self.arena_size / 2.0) - 0.8
            if abs(self.human.x) > boundary:
                self.human.yaw = math.atan2(-self.human.y, -self.human.x)
            elif abs(self.human.y) > boundary:
                self.human.yaw = math.atan2(-self.human.y, -self.human.x)
            else:
                self.human.yaw += float(self.np_random.uniform(-0.06, 0.06))

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
            # Stage 4: Obstacle bounce and wander
            boundary = (self.arena_size / 2.0) - 0.8
            if abs(self.human.x) > boundary or abs(self.human.y) > boundary:
                self.human.yaw = math.atan2(-self.human.y, -self.human.x)
            else:
                for ox, oy, obs_r in self.static_obstacles:
                    dx = self.human.x - ox
                    dy = self.human.y - oy
                    if math.hypot(dx, dy) < (obs_r + 0.35):
                        self.human.yaw = math.atan2(dy, dx)
                        break
                self.human.yaw += float(self.np_random.uniform(-0.08, 0.08))

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
            (self.robot_x + (self.wheelbase / 2.0) * cos_y, self.robot_y + (self.wheelbase / 2.0) * sin_y, 0.20),
            (self.robot_x - (self.wheelbase / 2.0) * cos_y, self.robot_y - (self.wheelbase / 2.0) * sin_y, 0.20),
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

    def _simulate_lidar(self):
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)
        lidar_x = self.robot_x + self.lidar_pos_x * cos_y
        lidar_y = self.robot_y + self.lidar_pos_x * sin_y

        ranges = np.full(self.lidar_samples, self.lidar_range_max, dtype=np.float32)
        half_arena = self.arena_size / 2.0

        for i, angle_offset in enumerate(self.ray_angles):
            global_angle = self.robot_yaw + angle_offset
            cos_a = math.cos(global_angle)
            sin_a = math.sin(global_angle)

            r = self.lidar_range_max

            # Outer boundary walls
            if cos_a > 1e-4:
                r = min(r, (half_arena - lidar_x) / cos_a)
            elif cos_a < -1e-4:
                r = min(r, (-half_arena - lidar_x) / cos_a)

            if sin_a > 1e-4:
                r = min(r, (half_arena - lidar_y) / sin_a)
            elif sin_a < -1e-4:
                r = min(r, (-half_arena - lidar_y) / sin_a)

            # Internal walls
            for x1, y1, x2, y2 in self.internal_walls:
                d_wall = intersect_ray_segment(lidar_x, lidar_y, cos_a, sin_a, x1, y1, x2, y2)
                if d_wall < r:
                    r = d_wall

            # Circular obstacles
            for ox, oy, obs_r in self.static_obstacles:
                dx = ox - lidar_x
                dy = oy - lidar_y
                proj = dx * cos_a + dy * sin_a
                if proj > 0:
                    perp_sq = (dx * dx + dy * dy) - (proj * proj)
                    if perp_sq < obs_r * obs_r:
                        dist_hit = proj - math.sqrt(max(0.0, obs_r * obs_r - perp_sq))
                        if dist_hit > 0:
                            r = min(r, dist_hit)

            # Bipedal Human Walker Dual-Leg Intersections (Target + Distractors)
            for person in [self.human] + self.distractors:
                hit_p = person.ray_intersection(lidar_x, lidar_y, cos_a, sin_a, self.lidar_range_max)
                if hit_p < r:
                    r = hit_p

            ranges[i] = np.clip(r, self.lidar_range_min, self.lidar_range_max)

        return ranges

    def _simulate_camera_depth(self):
        half_l = self.chassis_length / 2.0
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)
        cam_x = self.robot_x + half_l * cos_y
        cam_y = self.robot_y + half_l * sin_y

        ranges = np.full(self.cam_samples, self.cam_range_max, dtype=np.float32)
        half_arena = self.arena_size / 2.0

        for i, angle_offset in enumerate(self.cam_ray_angles):
            global_angle = self.robot_yaw + angle_offset
            cos_a = math.cos(global_angle)
            sin_a = math.sin(global_angle)

            r = self.cam_range_max

            # Outer boundary walls
            if cos_a > 1e-4:
                r = min(r, (half_arena - cam_x) / cos_a)
            elif cos_a < -1e-4:
                r = min(r, (-half_arena - cam_x) / cos_a)

            if sin_a > 1e-4:
                r = min(r, (half_arena - cam_y) / sin_a)
            elif sin_a < -1e-4:
                r = min(r, (-half_arena - cam_y) / sin_a)

            # Internal walls
            for x1, y1, x2, y2 in self.internal_walls:
                d_wall = intersect_ray_segment(cam_x, cam_y, cos_a, sin_a, x1, y1, x2, y2)
                if d_wall < r:
                    r = d_wall

            # Circular obstacles
            for ox, oy, obs_r in self.static_obstacles:
                dx = ox - cam_x
                dy = oy - cam_y
                proj = dx * cos_a + dy * sin_a
                if proj > 0:
                    perp_sq = (dx * dx + dy * dy) - (proj * proj)
                    if perp_sq < obs_r * obs_r:
                        dist_hit = proj - math.sqrt(max(0.0, obs_r * obs_r - perp_sq))
                        if dist_hit > 0:
                            r = min(r, dist_hit)

            # Bipedal Human Walker Dual-Leg Intersections (Target + Distractors)
            for person in [self.human] + self.distractors:
                hit_p = person.ray_intersection(cam_x, cam_y, cos_a, sin_a, self.cam_range_max)
                if hit_p < r:
                    r = hit_p

            ranges[i] = np.clip(r, self.cam_range_min, self.cam_range_max)

        return ranges

    def _get_observation(self):
        cam_ranges = self._simulate_camera_depth()
        cam_norm = (cam_ranges - self.cam_range_min) / (self.cam_range_max - self.cam_range_min)

        lidar_ranges = self._simulate_lidar()
        lidar_norm = (lidar_ranges - self.lidar_range_min) / (self.lidar_range_max - self.lidar_range_min)

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
        tracker_stat = self.tracker.get_status()

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

        import pygame

        target_w = 1200 if self.view_mode == "3d" else self.window_size
        target_h = 720 if self.view_mode == "3d" else self.window_size

        if self.window is None or self.window.get_size() != (target_w, target_h):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((target_w, target_h))
                mode_str = "3D Multi-View Dashboard" if self.view_mode == "3d" else "Classic 2D View"
                pygame.display.set_caption(f"AckBot Digital Twin - {mode_str} (Stage {self.stage})")
            else:
                self.window = pygame.Surface((target_w, target_h))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((target_w, target_h))

        if self.view_mode == "3d":
            self._render_3d_multiview(surface)
        else:
            self._render_2d(surface)

        if self.render_mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), (1, 0, 2))

    def _render_2d(self, surface):
        import pygame
        surface.fill((24, 26, 31))

        scale = self.window_size / self.arena_size
        half_arena = self.arena_size / 2.0

        def to_screen(x, y):
            sx = int((x + half_arena) * scale)
            sy = int((half_arena - y) * scale)
            return sx, sy

        # 1. Grid
        for i in range(-int(half_arena), int(half_arena) + 1):
            color = (36, 40, 48) if i != 0 else (48, 54, 66)
            pygame.draw.line(surface, color, to_screen(i, -half_arena), to_screen(i, half_arena), 1)
            pygame.draw.line(surface, color, to_screen(-half_arena, i), to_screen(half_arena, i), 1)

        # 2. Outer Walls
        wall_thick = max(2, int(0.1 * scale))
        top_left = to_screen(-half_arena, half_arena)
        bottom_right = to_screen(half_arena, -half_arena)
        pygame.draw.rect(
            surface,
            (70, 75, 87),
            (top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]),
            wall_thick,
        )

        # 3. Internal Corridor / Corner Walls
        for x1, y1, x2, y2 in self.internal_walls:
            p1 = to_screen(x1, y1)
            p2 = to_screen(x2, y2)
            pygame.draw.line(surface, (160, 165, 178), p1, p2, max(3, int(0.12 * scale)))

        # 4. Static Obstacles
        for ox, oy, obs_r in self.static_obstacles:
            pos = to_screen(ox, oy)
            r_px = int(obs_r * scale)
            pygame.draw.circle(surface, (68, 73, 85), pos, r_px)
            pygame.draw.circle(surface, (120, 128, 145), pos, r_px, 2)

        # 5. Draw Bipedal Human Walker (Animated Dual Feet + Torso)
        left_leg, right_leg = self.human.get_feet_positions()
        leg_r_px = max(2, int(left_leg.radius * scale))

        cr, cg, cb = [int(c * 255) for c in self.human.color_rgb]
        human_color = (cr, cg, cb)

        p_left = to_screen(left_leg.x, left_leg.y)
        p_right = to_screen(right_leg.x, right_leg.y)
        pygame.draw.circle(surface, (max(0, cr - 40), max(0, cg - 40), max(0, cb - 40)), p_left, leg_r_px)
        pygame.draw.circle(surface, (max(0, cr - 40), max(0, cg - 40), max(0, cb - 40)), p_right, leg_r_px)

        target_pos = to_screen(self.target_x, self.target_y)
        torso_r_px = max(4, int(self.human.torso_radius * scale))
        pygame.draw.circle(surface, human_color, target_pos, torso_r_px)
        pygame.draw.circle(surface, (255, 255, 255), target_pos, torso_r_px, 1)

        cos_th = math.cos(self.human.yaw)
        sin_th = math.sin(self.human.yaw)
        tgt_arrow = to_screen(self.target_x + 0.35 * cos_th, self.target_y + 0.35 * sin_th)
        pygame.draw.line(surface, (255, 255, 255), target_pos, tgt_arrow, 2)

        # Distractor Walkers
        for d in self.distractors:
            d_l, d_r = d.get_feet_positions()
            d_cr, d_cg, d_cb = [int(c * 255) for c in d.color_rgb]
            d_color = (d_cr, d_cg, d_cb)
            dp_l = to_screen(d_l.x, d_l.y)
            dp_r = to_screen(d_r.x, d_r.y)
            pygame.draw.circle(surface, (max(0, d_cr - 40), max(0, d_cg - 40), max(0, d_cb - 40)), dp_l, leg_r_px)
            pygame.draw.circle(surface, (max(0, d_cr - 40), max(0, d_cg - 40), max(0, d_cb - 40)), dp_r, leg_r_px)

            d_pos = to_screen(d.x, d.y)
            pygame.draw.circle(surface, d_color, d_pos, torso_r_px)
            pygame.draw.circle(surface, (140, 140, 145), d_pos, torso_r_px, 1)

            d_cos = math.cos(d.yaw)
            d_sin = math.sin(d.yaw)
            d_arr = to_screen(d.x + 0.30 * d_cos, d.y + 0.30 * d_sin)
            pygame.draw.line(surface, (140, 140, 145), d_pos, d_arr, 1)

        tracker_stat = self.tracker.get_status()
        if tracker_stat["is_occluded"]:
            est_x, est_y = tracker_stat["estimated_pos"]
            est_pos = to_screen(est_x, est_y)
            pygame.draw.circle(surface, (224, 108, 117), est_pos, torso_r_px + 3, 2)

        # 6. Draw AckBot Robot
        half_l = self.chassis_length / 2.0
        half_w = self.chassis_width / 2.0
        cos_y = math.cos(self.robot_yaw)
        sin_y = math.sin(self.robot_yaw)

        corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]
        screen_corners = []
        for cx, cy in corners:
            wx = self.robot_x + cx * cos_y - cy * sin_y
            wy = self.robot_y + cx * sin_y + cy * cos_y
            screen_corners.append(to_screen(wx, wy))

        robot_col = (224, 108, 117) if self.last_collision else (97, 175, 239)
        pygame.draw.polygon(surface, robot_col, screen_corners)
        pygame.draw.polygon(surface, (255, 255, 255), screen_corners, 2)

        front_pos = to_screen(self.robot_x + half_l * cos_y, self.robot_y + half_l * sin_y)
        center_pos = to_screen(self.robot_x, self.robot_y)
        pygame.draw.line(surface, (255, 255, 255), center_pos, front_pos, 2)

        steer_yaw = self.robot_yaw + self.robot_steer
        for wy_off in [self.overall_width / 2.0 - 0.03, -self.overall_width / 2.0 + 0.03]:
            r_wx = self.robot_x - (self.wheelbase / 2.0) * cos_y - wy_off * sin_y
            r_wy = self.robot_y - (self.wheelbase / 2.0) * sin_y + wy_off * cos_y
            pygame.draw.circle(surface, (30, 32, 38), to_screen(r_wx, r_wy), int(0.06 * scale))

            f_wx = self.robot_x + (self.wheelbase / 2.0) * cos_y - wy_off * sin_y
            f_wy = self.robot_y + (self.wheelbase / 2.0) * sin_y + wy_off * cos_y
            sp1 = (f_wx + 0.06 * math.cos(steer_yaw), f_wy + 0.06 * math.sin(steer_yaw))
            sp2 = (f_wx - 0.06 * math.cos(steer_yaw), f_wy - 0.06 * math.sin(steer_yaw))
            pygame.draw.line(surface, (10, 10, 10), to_screen(sp1[0], sp1[1]), to_screen(sp2[0], sp2[1]), 4)

        pygame.draw.line(surface, (229, 192, 123), center_pos, target_pos, 1)

        font = pygame.font.SysFont("monospace", 13)
        target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        track_status = "OCCLUDED (EKF)" if tracker_stat["is_occluded"] else "LOCKED"
        track_col = (224, 108, 117) if tracker_stat["is_occluded"] else (152, 195, 121)

        hud_lines = [
            (f"AckBot Follow - Stage {self.stage} ({self.scenario_state})", (97, 175, 239)),
            (f"Tracking : {track_status}", track_col),
            (f"Dist     : {target_dist:.2f} m", (255, 255, 255)),
            (f"Speed    : {self.robot_v:.2f} m/s", (229, 192, 123)),
            (f"Steer    : {math.degrees(self.robot_steer):+.1f}°", (229, 192, 123)),
            (f"Target V : {self.human.speed:.2f} m/s", (171, 178, 191)),
            (f"Step     : {self.current_step}/{self.max_episode_steps}", (171, 178, 191)),
            ("[V] Toggle 3D Dashboard", (100, 140, 190)),
        ]
        if self.num_distractors > 0:
            hud_lines.append((f"Crowd    : 1 Target + {len(self.distractors)} Distractors", (152, 195, 121)))

        for idx, (text, color) in enumerate(hud_lines):
            surface.blit(font.render(text, True, color), (15, 15 + idx * 18))

    def _render_3d_multiview(self, surface):
        """Render 4-panel composite dashboard (1200x720)."""
        import pygame

        surface.fill((16, 18, 24))

        # Panel 1: Left 720x720 (3D Isometric Arena)
        self._render_3d_arena(surface)

        # Panel 2: Top-Right 480x320 (Front Camera Viewport)
        self._render_front_camera_panel(surface)

        # Panel 3: Middle-Right 480x240 (Polar LiDAR Radar)
        self._render_polar_lidar_panel(surface)

        # Panel 4: Bottom-Right 480x160 (MPC Telemetry HUD)
        self._render_telemetry_hud(surface)

        # Dividing Borders
        pygame.draw.line(surface, (45, 55, 70), (720, 0), (720, 720), 2)
        pygame.draw.line(surface, (45, 55, 70), (720, 320), (1200, 320), 2)
        pygame.draw.line(surface, (45, 55, 70), (720, 560), (1200, 560), 2)

    def _render_3d_arena(self, surface):
        """Render 3D isometric arena projection with height extrusions."""
        import pygame

        cx = 360
        cy = 380
        s = 35.0  # ground scale
        sz = 46.0  # vertical height scale
        cos30 = 0.866025
        sin30 = 0.500000

        def to_iso(x, y, z=0.0):
            u = int(cx + (x - y) * s * cos30)
            v = int(cy + (x + y) * s * sin30 - z * sz)
            return u, v

        half_arena = self.arena_size / 2.0  # 5.0m

        # 1. Ground Grid
        for i in range(-int(half_arena), int(half_arena) + 1):
            col = (30, 34, 44) if i != 0 else (45, 55, 72)
            pygame.draw.line(surface, col, to_iso(i, -half_arena), to_iso(i, half_arena), 1)
            pygame.draw.line(surface, col, to_iso(-half_arena, i), to_iso(half_arena, i), 1)

        # 2. Outer Boundary Walls (Height = 0.8m)
        h_wall = 0.80
        corners_2d = [
            (-half_arena, -half_arena),
            (half_arena, -half_arena),
            (half_arena, half_arena),
            (-half_arena, half_arena),
        ]
        for idx in range(4):
            x1, y1 = corners_2d[idx]
            x2, y2 = corners_2d[(idx + 1) % 4]
            poly = [to_iso(x1, y1, 0), to_iso(x2, y2, 0), to_iso(x2, y2, h_wall), to_iso(x1, y1, h_wall)]
            pygame.draw.polygon(surface, (50, 56, 70), poly)
            pygame.draw.polygon(surface, (90, 100, 120), poly, 1)
            pygame.draw.line(surface, (130, 142, 165), to_iso(x1, y1, h_wall), to_iso(x2, y2, h_wall), 2)

        # 3. Internal Walls
        for x1, y1, x2, y2 in self.internal_walls:
            poly = [to_iso(x1, y1, 0), to_iso(x2, y2, 0), to_iso(x2, y2, h_wall), to_iso(x1, y1, h_wall)]
            pygame.draw.polygon(surface, (75, 85, 105), poly)
            pygame.draw.polygon(surface, (140, 150, 175), poly, 1)
            pygame.draw.line(surface, (190, 200, 225), to_iso(x1, y1, h_wall), to_iso(x2, y2, h_wall), 2)

        # 4. Obstacle Pillars
        for ox, oy, obs_r in self.static_obstacles:
            h_pillar = 1.20
            num_pts = 10
            base_pts = []
            top_pts = []
            for k in range(num_pts):
                ang = k * (2.0 * math.pi / num_pts)
                px = ox + obs_r * math.cos(ang)
                py = oy + obs_r * math.sin(ang)
                base_pts.append(to_iso(px, py, 0.0))
                top_pts.append(to_iso(px, py, h_pillar))

            # Pillar vertical sides
            for k in range(num_pts):
                k_next = (k + 1) % num_pts
                side_poly = [base_pts[k], base_pts[k_next], top_pts[k_next], top_pts[k]]
                shading = 55 + int(25 * math.sin(k * 0.6))
                pygame.draw.polygon(surface, (shading, shading + 6, shading + 15), side_poly)

            # Top lid
            pygame.draw.polygon(surface, (110, 120, 140), top_pts)
            pygame.draw.polygon(surface, (160, 170, 195), top_pts, 1)

        # 5. Bipedal Human Walker in 3D
        left_leg, right_leg = self.human.get_feet_positions()
        cr, cg, cb = [int(c * 255) for c in self.human.color_rgb]

        # Left leg (z=0 to 0.70m)
        p_foot_l = to_iso(left_leg.x, left_leg.y, 0.0)
        p_hip_l = to_iso(self.target_x + 0.08 * math.cos(self.human.yaw + math.pi / 2), self.target_y + 0.08 * math.sin(self.human.yaw + math.pi / 2), 0.70)
        pygame.draw.line(surface, (50, 60, 80), p_foot_l, p_hip_l, 5)
        pygame.draw.circle(surface, (35, 42, 58), p_foot_l, 4)

        # Right leg (z=0 to 0.70m)
        p_foot_r = to_iso(right_leg.x, right_leg.y, 0.0)
        p_hip_r = to_iso(self.target_x - 0.08 * math.cos(self.human.yaw + math.pi / 2), self.target_y - 0.08 * math.sin(self.human.yaw + math.pi / 2), 0.70)
        pygame.draw.line(surface, (50, 60, 80), p_foot_r, p_hip_r, 5)
        pygame.draw.circle(surface, (35, 42, 58), p_foot_r, 4)

        # Torso cylinder (z=0.70 to 1.45m)
        p_waist = to_iso(self.target_x, self.target_y, 0.70)
        p_neck = to_iso(self.target_x, self.target_y, 1.45)
        pygame.draw.line(surface, (cr, cg, cb), p_waist, p_neck, 9)

        # Head sphere (z=1.60m)
        p_head = to_iso(self.target_x, self.target_y, 1.58)
        pygame.draw.circle(surface, (235, 195, 140), p_head, 6)

        # Target forward heading arrow
        tgt_fwd = to_iso(self.target_x + 0.40 * math.cos(self.human.yaw), self.target_y + 0.40 * math.sin(self.human.yaw), 0.70)
        pygame.draw.line(surface, (255, 255, 255), p_waist, tgt_fwd, 2)

        # Distractors in 3D
        for d in self.distractors:
            d_l, d_r = d.get_feet_positions()
            d_cr, d_cg, d_cb = [int(c * 255) for c in d.color_rgb]
            dp_foot_l = to_iso(d_l.x, d_l.y, 0.0)
            dp_foot_r = to_iso(d_r.x, d_r.y, 0.0)
            dp_waist = to_iso(d.x, d.y, 0.70)
            dp_neck = to_iso(d.x, d.y, 1.45)
            pygame.draw.line(surface, (45, 52, 65), dp_foot_l, dp_waist, 4)
            pygame.draw.line(surface, (45, 52, 65), dp_foot_r, dp_waist, 4)
            pygame.draw.line(surface, (d_cr, d_cg, d_cb), dp_waist, dp_neck, 8)
            pygame.draw.circle(surface, (215, 185, 135), to_iso(d.x, d.y, 1.58), 5)

        # Occlusion EKF phantom ring
        tracker_stat = self.tracker.get_status()
        if tracker_stat["is_occluded"]:
            est_x, est_y = tracker_stat["estimated_pos"]
            p_est = to_iso(est_x, est_y, 0.85)
            pygame.draw.circle(surface, (224, 108, 117), p_est, 12, 2)

        # 6. AckBot 3D Model
        rx, ry = self.robot_x, self.robot_y
        yaw = self.robot_yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        steer_yaw = yaw + self.robot_steer

        # 4 Wheels
        for wy_off, is_front in [(0.15, False), (-0.15, False), (0.15, True), (-0.15, True)]:
            dx_w = (self.wheelbase / 2.0) if is_front else (-self.wheelbase / 2.0)
            wx = rx + dx_w * cos_y - wy_off * sin_y
            wy = ry + dx_w * sin_y + wy_off * cos_y
            w_yaw = steer_yaw if is_front else yaw
            wp1 = to_iso(wx - 0.06 * math.cos(w_yaw), wy - 0.06 * math.sin(w_yaw), 0.06)
            wp2 = to_iso(wx + 0.06 * math.cos(w_yaw), wy + 0.06 * math.sin(w_yaw), 0.06)
            pygame.draw.line(surface, (22, 25, 32), wp1, wp2, 6)
            pygame.draw.line(surface, (160, 172, 190), wp1, wp2, 2)

        # 3-Tier Chassis Decks
        def render_deck(length, width, z_elev, col_fill, col_border):
            hl = length / 2.0
            hw = width / 2.0
            d_corners = [
                to_iso(rx + hl * cos_y - hw * sin_y, ry + hl * sin_y + hw * cos_y, z_elev),
                to_iso(rx + hl * cos_y + hw * sin_y, ry + hl * sin_y - hw * cos_y, z_elev),
                to_iso(rx - hl * cos_y + hw * sin_y, ry - hl * sin_y - hw * cos_y, z_elev),
                to_iso(rx - hl * cos_y - hw * sin_y, ry - hl * sin_y + hw * cos_y, z_elev),
            ]
            pygame.draw.polygon(surface, col_fill, d_corners)
            pygame.draw.polygon(surface, col_border, d_corners, 1)

        # Level 1 Base: 30cm x 20cm at z=0.06m
        render_deck(0.30, 0.20, 0.06, (35, 65, 105), (60, 100, 150))
        # Level 2 Electronics Deck: 26cm x 18cm at z=0.16m
        render_deck(0.26, 0.18, 0.16, (97, 175, 239), (180, 220, 255))
        # Level 3 Sensor Deck: 20cm x 16cm at z=0.26m
        render_deck(0.20, 0.16, 0.26, (65, 115, 170), (140, 190, 240))

        # Rear Elevated LiDAR Turret (x=-0.15m, z=0.30m)
        lx = rx - 0.15 * cos_y
        ly = ry - 0.15 * sin_y
        p_lidar_base = to_iso(lx, ly, 0.26)
        p_lidar_top = to_iso(lx, ly, 0.32)
        pygame.draw.line(surface, (20, 24, 32), p_lidar_base, p_lidar_top, 8)
        pygame.draw.circle(surface, (0, 220, 255), p_lidar_top, 4)

        # Front Camera Sensor Block (x=+0.15m, z=0.25m)
        fx = rx + 0.15 * cos_y
        fy = ry + 0.15 * sin_y
        pygame.draw.circle(surface, (240, 180, 50), to_iso(fx, fy, 0.25), 3)

        # Robot Forward Heading Line
        p_robot_ctr = to_iso(rx, ry, 0.16)
        p_robot_fwd = to_iso(rx + 0.50 * cos_y, ry + 0.50 * sin_y, 0.16)
        pygame.draw.line(surface, (255, 255, 255), p_robot_ctr, p_robot_fwd, 2)

        # Tether tracking line to human
        pygame.draw.line(surface, (229, 192, 123), p_robot_ctr, p_waist, 1)

        # 7. Header & Compass
        font_h1 = pygame.font.SysFont("monospace", 14, bold=True)
        font_sub = pygame.font.SysFont("monospace", 11)
        surface.blit(font_h1.render("ACKBOT 3D DIGITAL TWIN (ISOMETRIC)", True, (255, 255, 255)), (16, 14))
        surface.blit(font_sub.render(f"Curriculum Stage {self.stage}  |  3-Tier Chassis & Steerable Ackermann", True, (0, 210, 255)), (16, 32))

        # Compass in bottom-left
        comp_cx, comp_cy = 50, 670
        pygame.draw.circle(surface, (28, 35, 48), (comp_cx, comp_cy), 22)
        pygame.draw.circle(surface, (60, 75, 98), (comp_cx, comp_cy), 22, 1)
        p_north = to_iso(0.0, 1.2, 0.0)
        p_zero = to_iso(0.0, 0.0, 0.0)
        ndx = p_north[0] - p_zero[0]
        ndy = p_north[1] - p_zero[1]
        nlen = math.hypot(ndx, ndy) or 1.0
        n_end = (int(comp_cx + (ndx / nlen) * 16), int(comp_cy + (ndy / nlen) * 16))
        pygame.draw.line(surface, (0, 255, 128), (comp_cx, comp_cy), n_end, 2)
        surface.blit(font_sub.render("N", True, (0, 255, 128)), (n_end[0] - 4, n_end[1] - 14))

        # Keybind tips in bottom
        surface.blit(font_sub.render("[V] Toggle 3D/2D  |  [Space] Pause  |  [R] Reset", True, (120, 135, 155)), (120, 690))

    def _render_front_camera_panel(self, surface):
        """Render Top-Right Panel (480x320): Front Camera Viewport."""
        import pygame

        ox, oy = 720, 0
        w, h = 480, 320

        # Sky and floor perspective background
        pygame.draw.rect(surface, (16, 20, 28), (ox, oy, w, 160))
        pygame.draw.rect(surface, (22, 26, 34), (ox, oy + 160, w, 160))

        # Converging floor perspective lines
        for offset_x in range(0, w + 1, 48):
            pygame.draw.line(surface, (30, 36, 48), (ox + 240, oy + 160), (ox + offset_x, oy + 320), 1)

        # 64 Depth rays
        cam_ranges = self._simulate_camera_depth()
        col_w = w / 64.0
        for i in range(64):
            r = float(cam_ranges[i])
            if r < 4.90:
                h_slice = int(np.clip((1.60 / r) * 160.0, 18, 270))
                y1 = oy + 160 - h_slice // 2
                y2 = oy + 160 + h_slice // 2
                x1 = int(ox + i * col_w)
                x2 = int(ox + (i + 1) * col_w) + 1

                if r < 0.90:
                    col = (250, 100, 40)
                elif r < 1.60:
                    col = (245, 175, 45)
                elif r < 2.50:
                    col = (55, 205, 120)
                elif r < 3.50:
                    col = (40, 180, 220)
                else:
                    col = (45, 75, 125)

                pygame.draw.rect(surface, col, (x1, y1, max(1, x2 - x1), y2 - y1))

        # Camera Header Banner
        pygame.draw.rect(surface, (20, 26, 36), (ox, oy, w, 28))
        font = pygame.font.SysFont("monospace", 11, bold=True)
        surface.blit(font.render("Robot Front RealSense Camera (64 Depth Rays, 85° FOV)", True, (255, 255, 255)), (ox + 12, oy + 7))

        # Central Targeting Reticle
        rx, ry = ox + 240, oy + 160
        tracker_stat = self.tracker.get_status()
        is_occ = tracker_stat["is_occluded"]
        reticle_col = (224, 108, 117) if is_occ else (0, 255, 120)

        pygame.draw.circle(surface, reticle_col, (rx, ry), 15, 1)
        pygame.draw.line(surface, reticle_col, (rx - 28, ry), (rx - 10, ry), 1)
        pygame.draw.line(surface, reticle_col, (rx + 10, ry), (rx + 28, ry), 1)
        pygame.draw.line(surface, reticle_col, (rx, ry - 28), (rx, ry - 10), 1)
        pygame.draw.line(surface, reticle_col, (rx, ry + 10), (rx, ry + 28), 1)

        # Status Tag
        status_txt = "TARGET: OCCLUDED (EKF)" if is_occ else "TARGET: LOCKED"
        target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        rel_x, rel_y, rel_head = self._get_relative_target_state()
        surface.blit(font.render(status_txt, True, reticle_col), (rx - 65, ry + 24))
        surface.blit(font.render(f"Range: {target_dist:.2f}m | Bearing: {math.degrees(rel_head):+.1f}°", True, (200, 215, 235)), (rx - 85, ry + 40))

    def _render_polar_lidar_panel(self, surface):
        """Render Middle-Right Panel (480x240): Polar 2D LiDAR Radar Scope."""
        import pygame

        ox, oy = 720, 320
        w, h = 480, 240

        pygame.draw.rect(surface, (14, 18, 24), (ox, oy, w, h))

        # Header
        pygame.draw.rect(surface, (20, 26, 36), (ox, oy, w, 24))
        font = pygame.font.SysFont("monospace", 11, bold=True)
        surface.blit(font.render("360° 2D LiDAR Polar Radar Scope (3.0m max)", True, (0, 220, 255)), (ox + 12, oy + 5))

        # Radar origin
        rcx = ox + 240
        rcy = oy + 205
        scale_r = 56.0  # px per meter (3m = 168px)

        # Concentric range rings
        font_sub = pygame.font.SysFont("monospace", 9)
        for rm in [1.0, 2.0, 3.0]:
            r_px = int(rm * scale_r)
            pygame.draw.circle(surface, (35, 45, 60), (rcx, rcy), r_px, 1)
            surface.blit(font_sub.render(f"{rm:.0f}m", True, (70, 90, 120)), (rcx + r_px - 14, rcy + 4))

        # Camera 85 deg FOV cone lines
        fov_half = math.radians(42.5)
        flen = int(3.0 * scale_r)
        p1 = (int(rcx - flen * math.sin(fov_half)), int(rcy - flen * math.cos(fov_half)))
        p2 = (int(rcx + flen * math.sin(fov_half)), int(rcy - flen * math.cos(fov_half)))
        pygame.draw.line(surface, (40, 55, 75), (rcx, rcy), p1, 1)
        pygame.draw.line(surface, (40, 55, 75), (rcx, rcy), p2, 1)

        # Plot 360 LiDAR ranges
        lidar_ranges = self._simulate_lidar()
        for idx in range(360):
            r = float(lidar_ranges[idx])
            if 0.10 <= r <= 3.00:
                ang = idx * (2.0 * math.pi / 360.0)
                px = int(rcx - r * math.sin(ang) * scale_r)
                py = int(rcy - r * math.cos(ang) * scale_r)
                if ox <= px < ox + w and oy <= py < oy + h:
                    if r < 0.60:
                        c = (255, 75, 50)
                    elif r < 1.20:
                        c = (255, 180, 40)
                    else:
                        c = (40, 230, 190)
                    pygame.draw.circle(surface, c, (px, py), 1)

        # Animated sweep line
        self.radar_sweep_angle = (self.radar_sweep_angle + 0.12) % (2.0 * math.pi)
        sw_x = int(rcx + flen * math.cos(self.radar_sweep_angle))
        sw_y = int(rcy - flen * math.sin(self.radar_sweep_angle))
        pygame.draw.line(surface, (0, 255, 140), (rcx, rcy), (sw_x, sw_y), 1)

        # Robot footprint icon at center
        pygame.draw.rect(surface, (0, 190, 255), (rcx - 6, rcy - 9, 12, 14))
        pygame.draw.line(surface, (255, 255, 255), (rcx, rcy + 3), (rcx, rcy - 12), 1)

    def _render_telemetry_hud(self, surface):
        """Render Bottom-Right Panel (480x160): Live MPC Telemetry HUD."""
        import pygame

        ox, oy = 720, 560
        w, h = 480, 160

        pygame.draw.rect(surface, (18, 22, 30), (ox, oy, w, h))

        # Header
        pygame.draw.rect(surface, (24, 32, 45), (ox, oy, w, 22))
        font_h = pygame.font.SysFont("monospace", 11, bold=True)
        surface.blit(font_h.render("LeWM CEM-MPC Telemetry & Vehicle Kinematics", True, (255, 255, 255)), (ox + 12, oy + 4))

        font = pygame.font.SysFont("monospace", 11)
        font_bold = pygame.font.SysFont("monospace", 11, bold=True)

        # Row 1: Tracking Status
        y1 = oy + 32
        tracker_stat = self.tracker.get_status()
        is_occ = tracker_stat["is_occluded"]
        bg_pill = (24, 80, 40) if not is_occ else (110, 35, 25)
        border_pill = (40, 180, 80) if not is_occ else (220, 70, 50)
        pygame.draw.rect(surface, bg_pill, (ox + 14, y1 - 4, 100, 20), border_radius=4)
        pygame.draw.rect(surface, border_pill, (ox + 14, y1 - 4, 100, 20), 1, border_radius=4)
        txt = "● LOCKED" if not is_occ else "● OCCLUDED"
        surface.blit(font_bold.render(txt, True, (255, 255, 255)), (ox + 20, y1))

        target_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        conf = tracker_stat.get("confidence", 1.0)
        surface.blit(font.render(f"Range: {target_dist:04.2f}m   Conf: {conf:04.2f}", True, (210, 225, 245)), (ox + 130, y1))

        # Row 2: Speedometer Bar
        y2 = oy + 62
        surface.blit(font.render(f"Velocity: {self.robot_v:+5.2f} m/s", True, (229, 192, 123)), (ox + 14, y2))
        bar_x = ox + 180
        bar_w = 270
        bar_h = 10
        pygame.draw.rect(surface, (32, 40, 54), (bar_x, y2 + 2, bar_w, bar_h))
        # Center notch for zero velocity
        zero_x = bar_x + int(bar_w * (0.40 / 1.20))
        pygame.draw.line(surface, (120, 135, 160), (zero_x, y2 - 1), (zero_x, y2 + bar_h + 1), 2)
        v_ratio = (self.robot_v - self.v_min) / (self.v_max - self.v_min)
        vx_pos = bar_x + int(np.clip(v_ratio, 0.0, 1.0) * bar_w)
        if vx_pos >= zero_x:
            pygame.draw.rect(surface, (97, 175, 239), (zero_x, y2 + 2, vx_pos - zero_x, bar_h))
        else:
            pygame.draw.rect(surface, (224, 108, 117), (vx_pos, y2 + 2, zero_x - vx_pos, bar_h))

        # Row 3: Steering Angle Dial
        y3 = oy + 92
        steer_deg = math.degrees(self.robot_steer)
        surface.blit(font.render(f"Steering: {steer_deg:+5.1f}°", True, (229, 192, 123)), (ox + 14, y3))
        pygame.draw.rect(surface, (32, 40, 54), (bar_x, y3 + 2, bar_w, bar_h))
        steer_zero = bar_x + bar_w // 2
        pygame.draw.line(surface, (120, 135, 160), (steer_zero, y3 - 1), (steer_zero, y3 + bar_h + 1), 2)
        s_ratio = (self.robot_steer + self.max_steering_angle) / (2.0 * self.max_steering_angle)
        sx_pos = bar_x + int(np.clip(s_ratio, 0.0, 1.0) * bar_w)
        pygame.draw.circle(surface, (0, 220, 255), (sx_pos, y3 + 7), 5)

        # Row 4: Headway Distance & Stage Indicator
        y4 = oy + 122
        surface.blit(font.render(f"Headway : {target_dist:04.2f}m", True, (152, 195, 121)), (ox + 14, y4))
        pygame.draw.rect(surface, (32, 40, 54), (bar_x, y4 + 2, bar_w, bar_h))
        # Draw safe zone (1.0m to 1.4m) in green
        z1 = bar_x + int((1.0 / 3.5) * bar_w)
        z2 = bar_x + int((1.4 / 3.5) * bar_w)
        pygame.draw.rect(surface, (30, 80, 45), (z1, y4 + 2, z2 - z1, bar_h))
        d_pos = bar_x + int(np.clip(target_dist / 3.5, 0.0, 1.0) * bar_w)
        pygame.draw.circle(surface, (255, 210, 60), (d_pos, y4 + 7), 4)
        surface.blit(font.render(f"Stage {self.stage} ({self.scenario_state})", True, (97, 175, 239)), (ox + 340, y1))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
