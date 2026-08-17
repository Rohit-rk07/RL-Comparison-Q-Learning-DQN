"""Central experiment configuration objects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class QLearningConfig:
    environment: str = "FrozenLake-v1"
    episodes: int = 2_000
    alpha: float = 0.25
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.997
    epsilon_strategy: Literal["exponential", "constant"] = "exponential"
    max_steps: int = 100

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DQNConfig:
    environment: str = "CartPole-v1"
    episodes: int = 300
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 10_000
    min_replay_size: int = 500
    target_update_frequency: int = 100
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.985
    hidden_size: int = 64
    max_steps: int = 500

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (42, 123, 2026)
    evaluation_episodes: int = 100
    moving_average_window: int = 25
