import numpy as np
import pytest

from src.replay_buffer import ReplayBuffer


def test_capacity_and_sampling_dimensions() -> None:
    buffer = ReplayBuffer(2, seed=1)
    for index in range(3):
        buffer.push(np.array([index, index + 1]), index % 2, float(index), np.array([index + 1, index + 2]), False)
    assert len(buffer) == 2
    states, actions, rewards, next_states, dones = buffer.sample(2)
    assert states.shape == (2, 2)
    assert actions.shape == rewards.shape == dones.shape == (2,)
    assert next_states.shape == (2, 2)


def test_sampling_before_minimum_size_raises() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(3).sample(1)
