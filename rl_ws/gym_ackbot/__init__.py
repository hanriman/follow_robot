"""Gymnasium environments for 4-wheel Ackermann robot (AckBot)."""

from rl_ws.gym_ackbot import envs as envs  # noqa: F401 - registers AckBotFollow-v0
from rl_ws.gym_ackbot.envs.ackbot_follow_env import AckBotFollowEnv

__version__ = "0.1.0"
__all__ = ["AckBotFollowEnv"]
