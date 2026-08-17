"""Fixed-capacity experience replay for DQN."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self._storage: deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._storage.append(Transition(np.asarray(state, dtype=np.float32).copy(), action, float(reward), np.asarray(next_state, dtype=np.float32).copy(), bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self):
            raise ValueError(f"Cannot sample {batch_size} transitions from a buffer of {len(self)}.")
        indices = self._rng.choice(len(self), size=batch_size, replace=False)
        transitions: Iterable[Transition] = (self._storage[i] for i in indices)
        batch = list(transitions)
        return (
            np.stack([item.state for item in batch]),
            np.asarray([item.action for item in batch], dtype=np.int64),
            np.asarray([item.reward for item in batch], dtype=np.float32),
            np.stack([item.next_state for item in batch]),
            np.asarray([item.done for item in batch], dtype=np.float32),
        )
