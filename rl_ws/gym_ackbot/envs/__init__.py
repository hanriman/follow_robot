"""Environment registrations for gym_ackbot."""

from gymnasium.envs.registration import register
from rl_ws.gym_ackbot.envs.ackbot_follow_env import AckBotFollowEnv

register(
    id="AckBotFollow-v0",
    entry_point="rl_ws.gym_ackbot.envs.ackbot_follow_env:AckBotFollowEnv",
    max_episode_steps=500,
)

__all__ = ["AckBotFollowEnv"]
