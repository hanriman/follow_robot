"""Pygame 2D and 3D isometric dashboard renderer for AckBotFollowEnv."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rl_ws.gym_ackbot.envs.ackbot_follow_env import AckBotFollowEnv


class AckBotRenderer:
    """Encapsulates Pygame rendering surfaces, HUD, isometric 3D viewport, and camera/lidar displays."""

    def __init__(self, window_size: int = 720, render_fps: int = 20) -> None:
        self.window_size = window_size
        self.render_fps = render_fps
        self.window = None
        self.clock = None
        self.radar_sweep_angle = 0.0

    def render(self, env: AckBotFollowEnv) -> np.ndarray | None:
        if env.render_mode is None:
            return None

        import pygame

        target_w = 1200 if env.view_mode == "3d" else self.window_size
        target_h = 720 if env.view_mode == "3d" else self.window_size

        if self.window is None or self.window.get_size() != (target_w, target_h):
            pygame.init()
            if env.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((target_w, target_h))
                mode_str = "3D Multi-View Dashboard" if env.view_mode == "3d" else "Classic 2D View"
                pygame.display.set_caption(f"AckBot Digital Twin - {mode_str} (Stage {env.stage})")
            else:
                self.window = pygame.Surface((target_w, target_h))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((target_w, target_h))

        if env.view_mode == "3d":
            self._render_3d_multiview(env, surface)
        else:
            self._render_2d(env, surface)

        if env.render_mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            return None
        elif env.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), (1, 0, 2))
        return None

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _render_2d(self, env: AckBotFollowEnv, surface) -> None:
        import pygame

        surface.fill((24, 26, 31))

        scale = self.window_size / env.arena_size
        half_arena = env.arena_size / 2.0

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
            (
                top_left[0],
                top_left[1],
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
            ),
            wall_thick,
        )

        # 3. Internal Corridor / Corner Walls
        for x1, y1, x2, y2 in env.internal_walls:
            p1 = to_screen(x1, y1)
            p2 = to_screen(x2, y2)
            pygame.draw.line(surface, (160, 165, 178), p1, p2, max(3, int(0.12 * scale)))

        # 4. Static Obstacles
        for ox, oy, obs_r in env.static_obstacles:
            pos = to_screen(ox, oy)
            r_px = int(obs_r * scale)
            pygame.draw.circle(surface, (68, 73, 85), pos, r_px)
            pygame.draw.circle(surface, (120, 128, 145), pos, r_px, 2)

        # 5. Draw Bipedal Human Walker
        left_leg, right_leg = env.human.get_feet_positions()
        leg_r_px = max(2, int(left_leg.radius * scale))

        cr, cg, cb = [int(c * 255) for c in env.human.color_rgb]
        human_color = (cr, cg, cb)

        p_left = to_screen(left_leg.x, left_leg.y)
        p_right = to_screen(right_leg.x, right_leg.y)
        pygame.draw.circle(
            surface, (max(0, cr - 40), max(0, cg - 40), max(0, cb - 40)), p_left, leg_r_px
        )
        pygame.draw.circle(
            surface, (max(0, cr - 40), max(0, cg - 40), max(0, cb - 40)), p_right, leg_r_px
        )

        target_pos = to_screen(env.target_x, env.target_y)
        torso_r_px = max(4, int(env.human.torso_radius * scale))
        pygame.draw.circle(surface, human_color, target_pos, torso_r_px)
        pygame.draw.circle(surface, (255, 255, 255), target_pos, torso_r_px, 1)

        cos_th = math.cos(env.human.yaw)
        sin_th = math.sin(env.human.yaw)
        tgt_arrow = to_screen(env.target_x + 0.35 * cos_th, env.target_y + 0.35 * sin_th)
        pygame.draw.line(surface, (255, 255, 255), target_pos, tgt_arrow, 2)

        # Distractor Walkers
        for d in env.distractors:
            d_l, d_r = d.get_feet_positions()
            d_cr, d_cg, d_cb = [int(c * 255) for c in d.color_rgb]
            d_color = (d_cr, d_cg, d_cb)
            dp_l = to_screen(d_l.x, d_l.y)
            dp_r = to_screen(d_r.x, d_r.y)
            pygame.draw.circle(
                surface, (max(0, d_cr - 40), max(0, d_cg - 40), max(0, d_cb - 40)), dp_l, leg_r_px
            )
            pygame.draw.circle(
                surface, (max(0, d_cr - 40), max(0, d_cg - 40), max(0, d_cb - 40)), dp_r, leg_r_px
            )

            d_pos = to_screen(d.x, d.y)
            pygame.draw.circle(surface, d_color, d_pos, torso_r_px)
            pygame.draw.circle(surface, (140, 140, 145), d_pos, torso_r_px, 1)

            d_cos = math.cos(d.yaw)
            d_sin = math.sin(d.yaw)
            d_arr = to_screen(d.x + 0.30 * d_cos, d.y + 0.30 * d_sin)
            pygame.draw.line(surface, (140, 140, 145), d_pos, d_arr, 1)

        tracker_stat = env.tracker.get_status(env.robot_x, env.robot_y, env.robot_yaw)
        if tracker_stat["is_occluded"]:
            est_x, est_y = tracker_stat["estimated_pos"]
            est_pos = to_screen(est_x, est_y)
            pygame.draw.circle(surface, (224, 108, 117), est_pos, torso_r_px + 3, 2)

        # 6. Draw AckBot Robot
        half_l = env.chassis_length / 2.0
        half_w = env.chassis_width / 2.0
        cos_y = math.cos(env.robot_yaw)
        sin_y = math.sin(env.robot_yaw)

        corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]
        screen_corners = []
        for cx, cy in corners:
            wx = env.robot_x + cx * cos_y - cy * sin_y
            wy = env.robot_y + cx * sin_y + cy * cos_y
            screen_corners.append(to_screen(wx, wy))

        robot_col = (224, 108, 117) if env.last_collision else (97, 175, 239)
        pygame.draw.polygon(surface, robot_col, screen_corners)
        pygame.draw.polygon(surface, (255, 255, 255), screen_corners, 2)

        front_pos = to_screen(env.robot_x + half_l * cos_y, env.robot_y + half_l * sin_y)
        center_pos = to_screen(env.robot_x, env.robot_y)
        pygame.draw.line(surface, (255, 255, 255), center_pos, front_pos, 2)

        steer_yaw = env.robot_yaw + env.robot_steer
        for wy_off in [env.overall_width / 2.0 - 0.03, -env.overall_width / 2.0 + 0.03]:
            r_wx = env.robot_x - (env.wheelbase / 2.0) * cos_y - wy_off * sin_y
            r_wy = env.robot_y - (env.wheelbase / 2.0) * sin_y + wy_off * cos_y
            pygame.draw.circle(surface, (30, 32, 38), to_screen(r_wx, r_wy), int(0.06 * scale))

            f_wx = env.robot_x + (env.wheelbase / 2.0) * cos_y - wy_off * sin_y
            f_wy = env.robot_y + (env.wheelbase / 2.0) * sin_y + wy_off * cos_y
            sp1 = (f_wx + 0.06 * math.cos(steer_yaw), f_wy + 0.06 * math.sin(steer_yaw))
            sp2 = (f_wx - 0.06 * math.cos(steer_yaw), f_wy - 0.06 * math.sin(steer_yaw))
            pygame.draw.line(
                surface, (10, 10, 10), to_screen(sp1[0], sp1[1]), to_screen(sp2[0], sp2[1]), 4
            )

        pygame.draw.line(surface, (229, 192, 123), center_pos, target_pos, 1)

        font = pygame.font.SysFont("monospace", 13)
        target_dist = math.hypot(env.target_x - env.robot_x, env.target_y - env.robot_y)
        track_status = "OCCLUDED (EKF)" if tracker_stat["is_occluded"] else "LOCKED"
        track_col = (224, 108, 117) if tracker_stat["is_occluded"] else (152, 195, 121)

        hud_lines = [
            (f"AckBot Follow - Stage {env.stage} ({env.scenario_state})", (97, 175, 239)),
            (f"Tracking : {track_status}", track_col),
            (f"Dist     : {target_dist:.2f} m", (255, 255, 255)),
            (f"Speed    : {env.robot_v:.2f} m/s", (229, 192, 123)),
            (f"Steer    : {math.degrees(env.robot_steer):+.1f}°", (229, 192, 123)),
            (f"Target V : {env.human.speed:.2f} m/s", (171, 178, 191)),
            (f"Step     : {env.current_step}/{env.max_episode_steps}", (171, 178, 191)),
            ("[V] Toggle 3D Dashboard", (100, 140, 190)),
        ]
        if env.num_distractors > 0:
            hud_lines.append(
                (f"Crowd    : 1 Target + {len(env.distractors)} Distractors", (152, 195, 121))
            )

        for idx, (text, color) in enumerate(hud_lines):
            surface.blit(font.render(text, True, color), (15, 15 + idx * 18))

    def _render_3d_multiview(self, env: AckBotFollowEnv, surface) -> None:
        import pygame

        surface.fill((16, 18, 24))

        # Panel 1: Left 720x720 (3D Isometric Arena)
        self._render_3d_arena(env, surface)

        # Panel 2: Top-Right 480x320 (Front Camera Viewport)
        self._render_front_camera_panel(env, surface)

        # Panel 3: Middle-Right 480x240 (Polar LiDAR Radar)
        self._render_polar_lidar_panel(env, surface)

        # Panel 4: Bottom-Right 480x160 (MPC Telemetry HUD)
        self._render_telemetry_hud(env, surface)

        # Dividing Borders
        pygame.draw.line(surface, (45, 55, 70), (720, 0), (720, 720), 2)
        pygame.draw.line(surface, (45, 55, 70), (720, 320), (1200, 320), 2)
        pygame.draw.line(surface, (45, 55, 70), (720, 560), (1200, 560), 2)

    def _render_3d_arena(self, env: AckBotFollowEnv, surface) -> None:
        import pygame

        cx = 360
        cy = 380
        s = 35.0
        sz = 46.0
        cos30 = 0.866025
        sin30 = 0.500000

        def to_iso(x, y, z=0.0):
            u = int(cx + (x - y) * s * cos30)
            v = int(cy + (x + y) * s * sin30 - z * sz)
            return u, v

        half_arena = env.arena_size / 2.0

        # Ground Grid
        for i in range(-int(half_arena), int(half_arena) + 1):
            col = (30, 34, 44) if i != 0 else (45, 55, 72)
            pygame.draw.line(surface, col, to_iso(i, -half_arena), to_iso(i, half_arena), 1)
            pygame.draw.line(surface, col, to_iso(-half_arena, i), to_iso(half_arena, i), 1)

        # Outer Boundary Walls
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
            poly = [
                to_iso(x1, y1, 0),
                to_iso(x2, y2, 0),
                to_iso(x2, y2, h_wall),
                to_iso(x1, y1, h_wall),
            ]
            pygame.draw.polygon(surface, (50, 56, 70), poly)
            pygame.draw.polygon(surface, (90, 100, 120), poly, 1)
            pygame.draw.line(
                surface, (130, 142, 165), to_iso(x1, y1, h_wall), to_iso(x2, y2, h_wall), 2
            )

        # Internal Walls
        for x1, y1, x2, y2 in env.internal_walls:
            poly = [
                to_iso(x1, y1, 0),
                to_iso(x2, y2, 0),
                to_iso(x2, y2, h_wall),
                to_iso(x1, y1, h_wall),
            ]
            pygame.draw.polygon(surface, (75, 85, 105), poly)
            pygame.draw.polygon(surface, (140, 150, 175), poly, 1)
            pygame.draw.line(
                surface, (190, 200, 225), to_iso(x1, y1, h_wall), to_iso(x2, y2, h_wall), 2
            )

        # Obstacle Pillars
        for ox, oy, obs_r in env.static_obstacles:
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

            for k in range(num_pts):
                k_next = (k + 1) % num_pts
                side_poly = [base_pts[k], base_pts[k_next], top_pts[k_next], top_pts[k]]
                shading = 55 + int(25 * math.sin(k * 0.6))
                pygame.draw.polygon(surface, (shading, shading + 6, shading + 15), side_poly)

            pygame.draw.polygon(surface, (110, 120, 140), top_pts)
            pygame.draw.polygon(surface, (160, 170, 195), top_pts, 1)

        # Bipedal Human Walker in 3D
        left_leg, right_leg = env.human.get_feet_positions()
        cr, cg, cb = [int(c * 255) for c in env.human.color_rgb]

        p_foot_l = to_iso(left_leg.x, left_leg.y, 0.0)
        p_hip_l = to_iso(
            env.target_x + 0.08 * math.cos(env.human.yaw + math.pi / 2),
            env.target_y + 0.08 * math.sin(env.human.yaw + math.pi / 2),
            0.70,
        )
        pygame.draw.line(surface, (50, 60, 80), p_foot_l, p_hip_l, 5)
        pygame.draw.circle(surface, (35, 42, 58), p_foot_l, 4)

        p_foot_r = to_iso(right_leg.x, right_leg.y, 0.0)
        p_hip_r = to_iso(
            env.target_x - 0.08 * math.cos(env.human.yaw + math.pi / 2),
            env.target_y - 0.08 * math.sin(env.human.yaw + math.pi / 2),
            0.70,
        )
        pygame.draw.line(surface, (50, 60, 80), p_foot_r, p_hip_r, 5)
        pygame.draw.circle(surface, (35, 42, 58), p_foot_r, 4)

        p_waist = to_iso(env.target_x, env.target_y, 0.70)
        p_neck = to_iso(env.target_x, env.target_y, 1.45)
        pygame.draw.line(surface, (cr, cg, cb), p_waist, p_neck, 9)

        p_head = to_iso(env.target_x, env.target_y, 1.58)
        pygame.draw.circle(surface, (235, 195, 140), p_head, 6)

        tgt_fwd = to_iso(
            env.target_x + 0.40 * math.cos(env.human.yaw),
            env.target_y + 0.40 * math.sin(env.human.yaw),
            0.70,
        )
        pygame.draw.line(surface, (255, 255, 255), p_waist, tgt_fwd, 2)

        # Distractors in 3D
        for d in env.distractors:
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

        tracker_stat = env.tracker.get_status(env.robot_x, env.robot_y, env.robot_yaw)
        if tracker_stat["is_occluded"]:
            est_x, est_y = tracker_stat["estimated_pos"]
            p_est = to_iso(est_x, est_y, 0.85)
            pygame.draw.circle(surface, (224, 108, 117), p_est, 12, 2)

        # AckBot 3D Model
        rx, ry = env.robot_x, env.robot_y
        yaw = env.robot_yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        steer_yaw = yaw + env.robot_steer

        # 4 Wheels
        for wy_off, is_front in [(0.15, False), (-0.15, False), (0.15, True), (-0.15, True)]:
            dx_w = (env.wheelbase / 2.0) if is_front else (-env.wheelbase / 2.0)
            wx = rx + dx_w * cos_y - wy_off * sin_y
            wy = ry + dx_w * sin_y + wy_off * cos_y
            w_yaw = steer_yaw if is_front else yaw
            wp1 = to_iso(wx - 0.06 * math.cos(w_yaw), wy - 0.06 * math.sin(w_yaw), 0.06)
            wp2 = to_iso(wx + 0.06 * math.cos(w_yaw), wy + 0.06 * math.sin(w_yaw), 0.06)
            pygame.draw.line(surface, (22, 25, 32), wp1, wp2, 6)
            pygame.draw.line(surface, (160, 172, 190), wp1, wp2, 2)

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

        render_deck(0.30, 0.20, 0.06, (35, 65, 105), (60, 100, 150))
        render_deck(0.26, 0.18, 0.16, (97, 175, 239), (180, 220, 255))
        render_deck(0.20, 0.16, 0.26, (65, 115, 170), (140, 190, 240))

        # LiDAR Turret
        lx = rx - 0.15 * cos_y
        ly = ry - 0.15 * sin_y
        p_lidar_base = to_iso(lx, ly, 0.26)
        p_lidar_top = to_iso(lx, ly, 0.32)
        pygame.draw.line(surface, (20, 24, 32), p_lidar_base, p_lidar_top, 8)
        pygame.draw.circle(surface, (0, 220, 255), p_lidar_top, 4)

        # Camera Block
        fx = rx + 0.15 * cos_y
        fy = ry + 0.15 * sin_y
        pygame.draw.circle(surface, (240, 180, 50), to_iso(fx, fy, 0.25), 3)

        p_robot_ctr = to_iso(rx, ry, 0.16)
        p_robot_fwd = to_iso(rx + 0.50 * cos_y, ry + 0.50 * sin_y, 0.16)
        pygame.draw.line(surface, (255, 255, 255), p_robot_ctr, p_robot_fwd, 2)
        pygame.draw.line(surface, (229, 192, 123), p_robot_ctr, p_waist, 1)

        font_h1 = pygame.font.SysFont("monospace", 14, bold=True)
        font_sub = pygame.font.SysFont("monospace", 11)
        surface.blit(
            font_h1.render("ACKBOT 3D DIGITAL TWIN (ISOMETRIC)", True, (255, 255, 255)), (16, 14)
        )
        surface.blit(
            font_sub.render(
                f"Curriculum Stage {env.stage}  |  3-Tier Chassis & Steerable Ackermann",
                True,
                (0, 210, 255),
            ),
            (16, 32),
        )

        # Compass
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

        surface.blit(
            font_sub.render(
                "[V] Toggle 3D/2D  |  [Space] Pause  |  [R] Reset", True, (120, 135, 155)
            ),
            (120, 690),
        )

    def _render_front_camera_panel(self, env: AckBotFollowEnv, surface) -> None:
        import pygame

        ox, oy = 720, 0
        w = 480

        pygame.draw.rect(surface, (16, 20, 28), (ox, oy, w, 160))
        pygame.draw.rect(surface, (22, 26, 34), (ox, oy + 160, w, 160))

        for offset_x in range(0, w + 1, 48):
            pygame.draw.line(
                surface, (30, 36, 48), (ox + 240, oy + 160), (ox + offset_x, oy + 320), 1
            )

        cam_ranges = env._simulate_camera_depth()
        col_w = w / float(env.cam_samples)
        for i in range(env.cam_samples):
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

        pygame.draw.rect(surface, (20, 26, 36), (ox, oy, w, 28))
        font = pygame.font.SysFont("monospace", 11, bold=True)
        surface.blit(
            font.render(
                "Robot Front RealSense Camera (64 Depth Rays, 85° FOV)", True, (255, 255, 255)
            ),
            (ox + 12, oy + 7),
        )

        rx, ry = ox + 240, oy + 160
        tracker_stat = env.tracker.get_status(env.robot_x, env.robot_y, env.robot_yaw)
        is_occ = tracker_stat["is_occluded"]
        reticle_col = (224, 108, 117) if is_occ else (0, 255, 120)

        pygame.draw.circle(surface, reticle_col, (rx, ry), 15, 1)
        pygame.draw.line(surface, reticle_col, (rx - 28, ry), (rx - 10, ry), 1)
        pygame.draw.line(surface, reticle_col, (rx + 10, ry), (rx + 28, ry), 1)
        pygame.draw.line(surface, reticle_col, (rx, ry - 28), (rx, ry - 10), 1)
        pygame.draw.line(surface, reticle_col, (rx, ry + 10), (rx, ry + 28), 1)

        status_txt = "TARGET: OCCLUDED (EKF)" if is_occ else "TARGET: LOCKED"
        target_dist = math.hypot(env.target_x - env.robot_x, env.target_y - env.robot_y)
        rel_x, rel_y, rel_head = env._get_relative_target_state()
        surface.blit(font.render(status_txt, True, reticle_col), (rx - 65, ry + 24))
        surface.blit(
            font.render(
                f"Range: {target_dist:.2f}m | Bearing: {math.degrees(rel_head):+.1f}°",
                True,
                (200, 215, 235),
            ),
            (rx - 85, ry + 40),
        )

    def _render_polar_lidar_panel(self, env: AckBotFollowEnv, surface) -> None:
        import pygame

        ox, oy = 720, 320
        w, h = 480, 240

        pygame.draw.rect(surface, (14, 18, 24), (ox, oy, w, h))

        pygame.draw.rect(surface, (20, 26, 36), (ox, oy, w, 24))
        font = pygame.font.SysFont("monospace", 11, bold=True)
        surface.blit(
            font.render("360° 2D LiDAR Polar Radar Scope (3.0m max)", True, (0, 220, 255)),
            (ox + 12, oy + 5),
        )

        rcx = ox + 240
        rcy = oy + 205
        scale_r = 56.0

        font_sub = pygame.font.SysFont("monospace", 9)
        for rm in [1.0, 2.0, 3.0]:
            r_px = int(rm * scale_r)
            pygame.draw.circle(surface, (35, 45, 60), (rcx, rcy), r_px, 1)
            surface.blit(
                font_sub.render(f"{rm:.0f}m", True, (70, 90, 120)), (rcx + r_px - 14, rcy + 4)
            )

        fov_half = math.radians(42.5)
        flen = int(3.0 * scale_r)
        p1 = (int(rcx - flen * math.sin(fov_half)), int(rcy - flen * math.cos(fov_half)))
        p2 = (int(rcx + flen * math.sin(fov_half)), int(rcy - flen * math.cos(fov_half)))
        pygame.draw.line(surface, (40, 55, 75), (rcx, rcy), p1, 1)
        pygame.draw.line(surface, (40, 55, 75), (rcx, rcy), p2, 1)

        lidar_ranges = env._simulate_lidar()
        for idx in range(env.lidar_samples):
            r = float(lidar_ranges[idx])
            if env.lidar_range_min <= r <= env.lidar_range_max:
                ang = idx * (2.0 * math.pi / float(env.lidar_samples))
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

        self.radar_sweep_angle = (self.radar_sweep_angle + 0.12) % (2.0 * math.pi)
        sw_x = int(rcx + flen * math.cos(self.radar_sweep_angle))
        sw_y = int(rcy - flen * math.sin(self.radar_sweep_angle))
        pygame.draw.line(surface, (0, 255, 140), (rcx, rcy), (sw_x, sw_y), 1)

        pygame.draw.rect(surface, (0, 190, 255), (rcx - 6, rcy - 9, 12, 14))
        pygame.draw.line(surface, (255, 255, 255), (rcx, rcy + 3), (rcx, rcy - 12), 1)

    def _render_telemetry_hud(self, env: AckBotFollowEnv, surface) -> None:
        import pygame

        ox, oy = 720, 560
        w, h = 480, 160

        pygame.draw.rect(surface, (18, 22, 30), (ox, oy, w, h))

        pygame.draw.rect(surface, (24, 32, 45), (ox, oy, w, 22))
        font_h = pygame.font.SysFont("monospace", 11, bold=True)
        surface.blit(
            font_h.render("LeWM CEM-MPC Telemetry & Vehicle Kinematics", True, (255, 255, 255)),
            (ox + 12, oy + 4),
        )

        font = pygame.font.SysFont("monospace", 11)
        font_bold = pygame.font.SysFont("monospace", 11, bold=True)

        y1 = oy + 32
        tracker_stat = env.tracker.get_status(env.robot_x, env.robot_y, env.robot_yaw)
        is_occ = tracker_stat["is_occluded"]
        bg_pill = (24, 80, 40) if not is_occ else (110, 35, 25)
        border_pill = (40, 180, 80) if not is_occ else (220, 70, 50)
        pygame.draw.rect(surface, bg_pill, (ox + 14, y1 - 4, 100, 20), border_radius=4)
        pygame.draw.rect(surface, border_pill, (ox + 14, y1 - 4, 100, 20), 1, border_radius=4)
        txt = "● LOCKED" if not is_occ else "● OCCLUDED"
        surface.blit(font_bold.render(txt, True, (255, 255, 255)), (ox + 20, y1))

        target_dist = math.hypot(env.target_x - env.robot_x, env.target_y - env.robot_y)
        conf = tracker_stat.get("confidence", 1.0)
        surface.blit(
            font.render(f"Range: {target_dist:04.2f}m   Conf: {conf:04.2f}", True, (210, 225, 245)),
            (ox + 130, y1),
        )

        y2 = oy + 62
        surface.blit(
            font.render(f"Velocity: {env.robot_v:+5.2f} m/s", True, (229, 192, 123)), (ox + 14, y2)
        )
        bar_x = ox + 180
        bar_w = 270
        bar_h = 10
        pygame.draw.rect(surface, (32, 40, 54), (bar_x, y2 + 2, bar_w, bar_h))
        zero_x = bar_x + int(bar_w * (0.40 / 1.20))
        pygame.draw.line(surface, (120, 135, 160), (zero_x, y2 - 1), (zero_x, y2 + bar_h + 1), 2)
        v_ratio = (env.robot_v - env.v_min) / (env.v_max - env.v_min)
        vx_pos = bar_x + int(np.clip(v_ratio, 0.0, 1.0) * bar_w)
        if vx_pos >= zero_x:
            pygame.draw.rect(surface, (97, 175, 239), (zero_x, y2 + 2, vx_pos - zero_x, bar_h))
        else:
            pygame.draw.rect(surface, (224, 108, 117), (vx_pos, y2 + 2, zero_x - vx_pos, bar_h))

        y3 = oy + 92
        steer_deg = math.degrees(env.robot_steer)
        surface.blit(
            font.render(f"Steering: {steer_deg:+5.1f}°", True, (229, 192, 123)), (ox + 14, y3)
        )
        pygame.draw.rect(surface, (32, 40, 54), (bar_x, y3 + 2, bar_w, bar_h))
        steer_zero = bar_x + bar_w // 2
        pygame.draw.line(
            surface, (120, 135, 160), (steer_zero, y3 - 1), (steer_zero, y3 + bar_h + 1), 2
        )
        s_ratio = (env.robot_steer + env.max_steering_angle) / (2.0 * env.max_steering_angle)
        sx_pos = bar_x + int(np.clip(s_ratio, 0.0, 1.0) * bar_w)
        pygame.draw.circle(surface, (0, 220, 255), (sx_pos, y3 + 7), 5)

        y4 = oy + 122
        surface.blit(
            font.render(f"Headway : {target_dist:04.2f}m", True, (152, 195, 121)), (ox + 14, y4)
        )
        pygame.draw.rect(surface, (32, 40, 54), (bar_x, y4 + 2, bar_w, bar_h))
        z1 = bar_x + int((1.0 / 3.5) * bar_w)
        z2 = bar_x + int((1.4 / 3.5) * bar_w)
        pygame.draw.rect(surface, (30, 80, 45), (z1, y4 + 2, z2 - z1, bar_h))
        d_pos = bar_x + int(np.clip(target_dist / 3.5, 0.0, 1.0) * bar_w)
        pygame.draw.circle(surface, (255, 210, 60), (d_pos, y4 + 7), 4)
        surface.blit(
            font.render(f"Stage {env.stage} ({env.scenario_state})", True, (97, 175, 239)),
            (ox + 340, y1),
        )
