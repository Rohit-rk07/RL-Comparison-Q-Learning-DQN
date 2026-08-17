import numpy as np
import torch

from src.config import DQNConfig
from src.dqn import DQNAgent, QNetwork


def test_network_output_dimensions() -> None:
    network = QNetwork(4, 2, 8)
    assert network(torch.zeros((3, 4))).shape == (3, 2)


def test_parameter_update_and_target_synchronization() -> None:
    config = DQNConfig(batch_size=2, min_replay_size=2, replay_capacity=10, hidden_size=8)
    agent = DQNAgent(4, 2, config, seed=2)
    for _ in range(2):
        agent.replay_buffer.push(np.zeros(4), 0, 1.0, np.ones(4), False)
    before = [parameter.detach().clone() for parameter in agent.policy_network.parameters()]
    assert agent.learn() is not None
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.policy_network.parameters()))
    agent.sync_target_network()
    assert all(torch.equal(policy, target) for policy, target in zip(agent.policy_network.parameters(), agent.target_network.parameters()))
