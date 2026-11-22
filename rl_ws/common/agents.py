"""Unified Agent Interfaces and Model Loaders for AckBot Following."""

from __future__ import annotations

import abc
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_ws.common.config import RobotConfig, get_robot_config, select_device
from rl_ws.policies.ppo.actor_critic import ActorCritic
from rl_ws.policies.vla.diffusion_wm import ActionConditionedDiffusionWorldModel
from rl_ws.policies.world_model.cem_planner import CEMPlanner
from rl_ws.policies.world_model.reward_head import LatentRewardHead
from rl_ws.policies.world_model.sigreg_wm import TeacherFreeLeWorldModel
from rl_ws.policies.world_model.vae import MultiModalSensorVAE


class BaseAgent(abc.ABC):
    """Abstract base agent for AckBot person following policies."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal episodic state or planner history."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        """Compute physical action [v (m/s), steer (rad)]."""


class RandomAgent(BaseAgent):
    """Uniform random exploration baseline within canonical physical bounds."""

    def __init__(self, cfg: RobotConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or get_robot_config()
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        v = self.rng.uniform(self.cfg.v_min, self.cfg.v_max)
        steer = self.rng.uniform(-self.cfg.max_steering_angle, self.cfg.max_steering_angle)
        return np.array([v, steer], dtype=np.float32)


class HeuristicPursuitAgent(BaseAgent):
    """Proportional visual pursuit policy with LiDAR obstacle avoidance reflex."""

    def __init__(self, cfg: RobotConfig | None = None) -> None:
        self.cfg = cfg or get_robot_config()

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        cfg = self.cfg
        # Extract visual target or tracker state
        target_dist = 0.80
        rel_heading = 0.0

        if info is not None and "tracker_status" in info:
            stat = info["tracker_status"]
            target_dist = float(stat.get("estimated_dist", 0.80))
            rel_heading = float(stat.get("estimated_bearing", 0.0))
        elif info is not None and "target_dist" in info:
            target_dist = float(info["target_dist"])
            rel_heading = math.radians(float(info.get("rel_heading_deg", 0.0)))
        else:
            # Estimate from front camera depth (first 64 rays)
            cam_m = cfg.denormalize_cam(obs[:64])
            min_idx = int(np.argmin(cam_m))
            target_dist = float(cam_m[min_idx])
            fov = cfg.cam_fov_rad
            rel_heading = float(-fov / 2.0 + (min_idx / 63.0) * fov)

        # Proportional pursuit steering
        steer_gain = 1.2
        target_steer = np.clip(
            steer_gain * rel_heading, -cfg.max_steering_angle, cfg.max_steering_angle
        )

        # Velocity profile relative to target follow distance
        dist_err = target_dist - cfg.target_follow_distance
        if dist_err > 0.10:
            target_v = np.clip(0.30 + 0.50 * dist_err, 0.0, cfg.v_max)
        elif dist_err < -0.20:
            target_v = cfg.v_min * 0.50
        else:
            target_v = 0.25

        # LiDAR obstacle avoidance reflex: rays 330 to 30 (front 60 deg)
        lidar_m = cfg.denormalize_lidar(obs[64:424])
        front_rays = np.concatenate([lidar_m[:30], lidar_m[330:]])
        min_front_dist = np.min(front_rays)

        if min_front_dist < 0.45:
            left_dist = np.mean(lidar_m[10:50])
            right_dist = np.mean(lidar_m[310:350])
            avoid_steer = (
                cfg.max_steering_angle if left_dist > right_dist else -cfg.max_steering_angle
            )
            target_steer = 0.70 * avoid_steer + 0.30 * target_steer
            target_v = min(target_v, 0.15)

        return np.array([target_v, target_steer], dtype=np.float32)


class PPOAgent(BaseAgent):
    """Recurrent PPO Neural Policy Agent."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str = "auto",
        deterministic: bool = True,
    ) -> None:
        self.device = select_device(device) if isinstance(device, str) else device
        self.deterministic = deterministic

        self.model = ActorCritic(obs_dim=426, act_dim=2).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.hx = self.model.init_hidden(1, device=self.device)

    def reset(self) -> None:
        self.hx = self.model.init_hidden(1, device=self.device)

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.deterministic:
                action, self.hx = self.model.get_deterministic_action(
                    obs_t, hx=self.hx, return_hidden=True
                )
            else:
                action, _, _, _, self.hx = self.model.get_action_and_value(
                    obs_t, hx=self.hx, return_hidden=True
                )
            return action.squeeze(0).cpu().numpy()


class LatentMPCAgent(BaseAgent):
    """World-Model Guided Latent Action Optimization (CEM-MPC) Agent."""

    def __init__(
        self,
        world_model: torch.nn.Module,
        reward_head: LatentRewardHead,
        model_type: str = "lewm",
        horizon: int = 5,
        pop_size: int = 50,
        num_elites: int = 5,
        num_iters: int = 3,
        device: torch.device | str = "auto",
        cfg: RobotConfig | None = None,
    ) -> None:
        self.device = select_device(device) if isinstance(device, str) else device
        self.world_model = world_model.to(self.device).eval()
        self.reward_head = reward_head.to(self.device).eval()
        self.model_type = model_type.lower()
        self.cfg = cfg or get_robot_config()

        self.planner = CEMPlanner(
            horizon=horizon,
            pop_size=pop_size,
            num_elites=num_elites,
            num_iters=num_iters,
            v_min=self.cfg.v_min,
            v_max=self.cfg.v_max,
            steer_max=self.cfg.max_steering_angle,
            device=self.device,
        )

    def reset(self) -> None:
        self.planner.reset()

    def act(self, obs: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            if self.model_type == "lewm":
                z_0 = self.world_model.encode(obs_t)
                action_t = self.planner.plan(
                    initial_z=z_0,
                    rollout_fn=lambda z, acts: self.world_model.predictor.rollout(z, acts),
                    score_fn=lambda unrolled: self.reward_head.compute_control_score(unrolled),
                    obs_t=obs,
                    tracker_status=info.get("tracker_status") if info else None,
                )
            elif self.model_type in ("diffusion", "ac_ldm"):
                z_0 = self.world_model.encode(obs_t)
                action_t = self.planner.plan(
                    initial_z=z_0,
                    rollout_fn=lambda z, acts: self.world_model.rollout(z, acts, num_steps=5),
                    score_fn=lambda unrolled: self.reward_head.compute_control_score(unrolled),
                    obs_t=obs,
                    tracker_status=info.get("tracker_status") if info else None,
                )
            elif self.model_type == "vae":
                # For baseline VAE without learned transition dynamics: 1-step latent projection
                z_0 = self.world_model.encode(obs_t)
                action_t = self.planner.plan(
                    initial_z=z_0,
                    rollout_fn=lambda z, acts: z.unsqueeze(1).repeat(1, acts.size(1), 1, 1),
                    score_fn=lambda unrolled: self.reward_head.compute_control_score(unrolled),
                    obs_t=obs,
                    tracker_status=info.get("tracker_status") if info else None,
                )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            return action_t.cpu().numpy()


def load_world_model(
    model_type: str,
    checkpoint_path: str | Path,
    device: torch.device | str = "auto",
    vae_path: str | Path | None = None,
) -> torch.nn.Module:
    """Load and return world model evaluated in eval mode."""
    dev = select_device(device) if isinstance(device, str) else device
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    mtype = model_type.lower()

    if mtype == "lewm":
        model = TeacherFreeLeWorldModel(latent_dim=128, action_dim=2).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
    elif mtype == "vae":
        model = MultiModalSensorVAE(in_dim=426, latent_channels=4, latent_length=32).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
    elif mtype in ("diffusion", "ac_ldm"):
        if vae_path is None:
            raise ValueError(
                "vae_path is required to instantiate ActionConditionedDiffusionWorldModel"
            )
        vae = load_world_model("vae", vae_path, device=dev)
        model = ActionConditionedDiffusionWorldModel(
            latent_channels=4,
            latent_length=32,
            action_dim=2,
            base_channels=64,
            num_timesteps=1000,
            vae=vae,
        ).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Unknown world model type: {model_type}")

    model.eval()
    return model


def load_reward_head(
    checkpoint_path: str | Path,
    device: torch.device | str = "auto",
    is_spatial: bool = False,
) -> LatentRewardHead:
    """Load and return LatentRewardHead evaluated in eval mode."""
    dev = select_device(device) if isinstance(device, str) else device
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    head = LatentRewardHead(
        in_dim=128,
        is_spatial=is_spatial,
        spatial_channels=4,
        spatial_length=32,
    ).to(dev)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()
    return head
