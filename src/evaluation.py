"""Evaluation routines that never explore or update model parameters."""
from __future__ import annotations

from typing import Protocol

import gymnasium as gym
import numpy as np


class EvaluatableAgent(Protocol):
    def select_action(self, state: object, epsilon: float = 0.0) -> int: ...


def evaluate_agent(agent: EvaluatableAgent, env: gym.Env, episodes: int, seed: int, max_steps: int) -> dict[str, float]:
    rewards, lengths = [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(total_reward)
        lengths.append(step)
    return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)), "success_rate": float(np.mean(np.asarray(rewards) > 0)), "mean_length": float(np.mean(lengths))}
