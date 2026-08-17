import numpy as np

from src.config import QLearningConfig
from src.q_learning import QLearningAgent


def make_agent() -> QLearningAgent:
    return QLearningAgent(4, 2, QLearningConfig(alpha=0.5, gamma=0.9), seed=7)


def test_q_table_dimensions() -> None:
    assert make_agent().q_table.shape == (4, 2)


def test_greedy_action_selection() -> None:
    agent = make_agent()
    agent.q_table[0] = [0.0, 3.0]
    assert agent.select_action(0, epsilon=0.0) == 1


def test_q_value_update_and_terminal_handling() -> None:
    agent = make_agent()
    agent.q_table[1] = [10.0, 20.0]
    agent.update(0, 1, reward=2.0, next_state=1, done=True)
    assert agent.q_table[0, 1] == 1.0  # no terminal bootstrap
    agent.update(0, 1, reward=2.0, next_state=1, done=False)
    assert np.isclose(agent.q_table[0, 1], 10.5)
