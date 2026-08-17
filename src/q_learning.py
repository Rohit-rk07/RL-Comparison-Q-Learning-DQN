"""NumPy tabular Q-Learning agent."""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from src.config import QLearningConfig


@dataclass
class QLearningAgent:
    state_count: int
    action_count: int
    config: QLearningConfig
    seed: int = 0

    def __post_init__(self) -> None:
        self.q_table = np.zeros((self.state_count, self.action_count), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def epsilon_for_episode(self, episode: int) -> float:
        if self.config.epsilon_strategy == "constant":
            return self.config.epsilon_start
        return max(self.config.epsilon_min, self.config.epsilon_start * self.config.epsilon_decay**episode)

    def select_action(self, state: int, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        best_actions = np.flatnonzero(self.q_table[state] == self.q_table[state].max())
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        bootstrap = 0.0 if done else self.config.gamma * float(self.q_table[next_state].max())
        td_target = reward + bootstrap
        self.q_table[state, action] += self.config.alpha * (td_target - self.q_table[state, action])

    def train(self, env: gym.Env, seed: int) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for episode in range(self.config.episodes):
            state, _ = env.reset(seed=seed + episode)
            epsilon = self.epsilon_for_episode(episode)
            total_reward, steps = 0.0, 0
            for steps in range(1, self.config.max_steps + 1):
                action = self.select_action(int(state), epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(int(state), action, reward, int(next_state), done)
                total_reward += float(reward)
                state = next_state
                if done:
                    break
            history.append({"episode": episode + 1, "reward": total_reward, "length": steps, "epsilon": epsilon})
        return history
