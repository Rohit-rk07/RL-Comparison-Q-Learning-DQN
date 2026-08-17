"""Environment construction with consistent seeding."""
from __future__ import annotations

import gymnasium as gym


def make_frozen_lake(seed: int | None = None) -> gym.Env:
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_cart_pole(seed: int | None = None) -> gym.Env:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
