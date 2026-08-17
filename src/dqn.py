"""Small PyTorch DQN implementation with replay and a target network."""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from src.config import DQNConfig
from src.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


@dataclass
class DQNAgent:
    state_size: int
    action_count: int
    config: DQNConfig
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.policy_network = QNetwork(self.state_size, self.action_count, self.config.hidden_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_count, self.config.hidden_size).to(self.device)
        self.sync_target_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity, self.seed)
        self.training_steps = 0

    def epsilon_for_episode(self, episode: int) -> float:
        return max(self.config.epsilon_min, self.config.epsilon_start * self.config.epsilon_decay**episode)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_network(state_tensor).argmax(dim=1).item())

    def learn(self) -> tuple[float, float] | None:
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.min_replay_size):
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        q_values = self.policy_network(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1, keepdim=True).values
            targets = rewards_t + self.config.gamma * (1 - dones_t) * next_q_values
        loss = self.loss_function(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.config.target_update_frequency == 0:
            self.sync_target_network()
        return float(loss.item()), float(q_values.mean().item())

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, env: gym.Env, seed: int) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for episode in range(self.config.episodes):
            state, _ = env.reset(seed=seed + episode)
            epsilon, total_reward, losses, q_values = self.epsilon_for_episode(episode), 0.0, [], []
            for step in range(1, self.config.max_steps + 1):
                action = self.select_action(np.asarray(state), epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, reward, next_state, done)
                learned = self.learn()
                if learned is not None:
                    loss, mean_q = learned
                    losses.append(loss)
                    q_values.append(mean_q)
                total_reward += float(reward)
                state = next_state
                if done:
                    break
            history.append({"episode": episode + 1, "reward": total_reward, "length": step, "epsilon": epsilon, "loss": float(np.mean(losses)) if losses else float("nan"), "mean_q": float(np.mean(q_values)) if q_values else float("nan")})
        return history
