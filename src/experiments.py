"""Training orchestration, result persistence, summaries, and plots."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DQNConfig, ExperimentConfig, QLearningConfig
from src.dqn import DQNAgent
from src.environments import make_cart_pole, make_frozen_lake
from src.evaluation import evaluate_agent
from src.q_learning import QLearningAgent
from src.utils import PROCESSED_RESULTS_DIR, PLOTS_DIR, RAW_RESULTS_DIR, ensure_directories, save_json, set_global_seed, timestamp

Algorithm = Literal["q_learning", "dqn"]


def train_once(algorithm: Algorithm, seed: int, config: QLearningConfig | DQNConfig, evaluation_episodes: int = 100) -> tuple[pd.DataFrame, dict[str, object]]:
    set_global_seed(seed)
    started = perf_counter()
    if algorithm == "q_learning":
        assert isinstance(config, QLearningConfig)
        env = make_frozen_lake(seed)
        agent = QLearningAgent(env.observation_space.n, env.action_space.n, config, seed)
    else:
        assert isinstance(config, DQNConfig)
        env = make_cart_pole(seed)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config, seed)
    try:
        history = pd.DataFrame(agent.train(env, seed))
        evaluation = evaluate_agent(agent, env, evaluation_episodes, seed + 10_000, config.max_steps)
    finally:
        env.close()
    history["seed"] = seed
    summary: dict[str, object] = {"algorithm": algorithm, "environment": config.environment, "seed": seed, "timestamp_utc": timestamp(), "duration_seconds": perf_counter() - started, "final_25_episode_mean": float(history["reward"].tail(min(25, len(history))).mean()), "best_25_episode_mean": float(history["reward"].rolling(25, min_periods=1).mean().max()), "hyperparameters": config.to_dict(), "evaluation": evaluation}
    return history, summary


def run_configuration(name: str, algorithm: Algorithm, config: QLearningConfig | DQNConfig, seeds: tuple[int, ...] = (42, 123, 2026), evaluation_episodes: int = 100) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    ensure_directories()
    histories, summaries = [], []
    for seed in seeds:
        history, summary = train_once(algorithm, seed, config, evaluation_episodes)
        history["experiment"] = name
        histories.append(history)
        summaries.append(summary)
        save_json(RAW_RESULTS_DIR / f"{name}_{algorithm}_seed-{seed}_summary.json", summary)
    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv(RAW_RESULTS_DIR / f"{name}_{algorithm}_episodes.csv", index=False)
    return combined, summaries


def summarize(name: str, algorithm: Algorithm, histories: pd.DataFrame, summaries: list[dict[str, object]]) -> pd.DataFrame:
    grouped = histories.groupby("episode")["reward"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "mean_reward", "std": "std_reward"})
    grouped["std_reward"] = grouped["std_reward"].fillna(0.0)
    grouped["moving_mean_reward"] = grouped["mean_reward"].rolling(25, min_periods=1).mean()
    grouped.to_csv(PROCESSED_RESULTS_DIR / f"{name}_{algorithm}_aggregate.csv", index=False)
    evaluations = [item["evaluation"] for item in summaries]
    summary_row = pd.DataFrame([{"experiment": name, "algorithm": algorithm, "seeds": len(summaries), "final_25_episode_mean": np.mean([item["final_25_episode_mean"] for item in summaries]), "best_25_episode_mean": np.mean([item["best_25_episode_mean"] for item in summaries]), "evaluation_mean_reward": np.mean([item["mean_reward"] for item in evaluations]), "evaluation_std_across_seeds": np.std([item["mean_reward"] for item in evaluations]), "mean_duration_seconds": np.mean([item["duration_seconds"] for item in summaries])}])
    summary_path = PROCESSED_RESULTS_DIR / "experiment_summary.csv"
    summary_row.to_csv(summary_path, mode="a", index=False, header=not summary_path.exists())
    return grouped


def plot_learning_curve(aggregate: pd.DataFrame, title: str, output_name: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    episodes = aggregate["episode"].to_numpy()
    mean = aggregate["mean_reward"].to_numpy()
    std = aggregate["std_reward"].to_numpy()
    axis.plot(episodes, mean, alpha=0.35, color="#4c78a8", label="Mean episode reward")
    axis.plot(episodes, aggregate["moving_mean_reward"], linewidth=2.2, color="#f58518", label="25-episode moving mean")
    axis.fill_between(episodes, mean - std, mean + std, color="#4c78a8", alpha=0.18, label="±1 seed standard deviation")
    axis.set(title=title, xlabel="Episode", ylabel="Reward")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close(fig)


def plot_comparison(aggregates: dict[str, pd.DataFrame], title: str, output_name: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, aggregate in aggregates.items():
        episodes, mean, std = (aggregate[column].to_numpy() for column in ("episode", "mean_reward", "std_reward"))
        axis.plot(episodes, mean, linewidth=2, label=label)
        axis.fill_between(episodes, mean - std, mean + std, alpha=0.12)
    axis.set(title=title, xlabel="Episode", ylabel="Reward")
    axis.legend(title="Configuration")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close(fig)


def plot_normalized_algorithm_perspective(q_learning: pd.DataFrame, dqn: pd.DataFrame) -> None:
    """Contrast task-relative learning progress without equating raw rewards."""
    fig, axis = plt.subplots(figsize=(9, 5))
    q_episode = q_learning["episode"].to_numpy()
    dqn_episode = dqn["episode"].to_numpy()
    # FrozenLake's maximum episodic return is 1; CartPole's is capped at 500.
    axis.plot(q_episode, q_learning["mean_reward"], label="Q-Learning: FrozenLake success rate (0–1)", linewidth=2)
    axis.plot(dqn_episode, dqn["mean_reward"] / 500.0, label="DQN: CartPole reward / 500 (0–1)", linewidth=2)
    axis.set(title="Task-normalized learning progress (not a raw-reward ranking)", xlabel="Episode", ylabel="Environment-specific normalized return")
    axis.set_ylim(-0.05, 1.05)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "algorithm_perspective_normalized.png", dpi=160)
    plt.close(fig)


def baseline_experiments(experiment_config: ExperimentConfig = ExperimentConfig()) -> None:
    baseline_aggregates: dict[str, pd.DataFrame] = {}
    for name, algorithm, configuration, title, filename in [
        ("baseline", "q_learning", QLearningConfig(), "Q-Learning on deterministic FrozenLake (4×4)", "q_learning_baseline.png"),
        ("baseline", "dqn", DQNConfig(), "DQN on CartPole-v1", "dqn_baseline.png"),
    ]:
        histories, summaries = run_configuration(name, algorithm, configuration, experiment_config.seeds, experiment_config.evaluation_episodes)
        baseline_aggregates[algorithm] = summarize(name, algorithm, histories, summaries)
        plot_learning_curve(baseline_aggregates[algorithm], title, filename)
    plot_normalized_algorithm_perspective(baseline_aggregates["q_learning"], baseline_aggregates["dqn"])


def hyperparameter_experiments(experiment_config: ExperimentConfig = ExperimentConfig()) -> None:
    # One major variable changes per family; all other baseline values are held fixed.
    study_specs: list[tuple[str, Algorithm, list[tuple[str, QLearningConfig | DQNConfig]], str, str]] = [
        ("exploration_q_learning", "q_learning", [("constant ε=0.30", replace(QLearningConfig(), epsilon_strategy="constant", epsilon_start=0.30)), ("standard decay", QLearningConfig()), ("fast decay", replace(QLearningConfig(), epsilon_decay=0.985))], "Q-Learning exploration-strategy sensitivity", "exploration_q_learning.png"),
        ("learning_rate_q_learning", "q_learning", [("α=0.05", replace(QLearningConfig(), alpha=0.05)), ("α=0.25", QLearningConfig()), ("α=0.80", replace(QLearningConfig(), alpha=0.80))], "Q-Learning learning-rate sensitivity", "learning_rate_q_learning.png"),
        ("learning_rate_dqn", "dqn", [("lr=0.0001", replace(DQNConfig(), learning_rate=1e-4)), ("lr=0.001", DQNConfig()), ("lr=0.01", replace(DQNConfig(), learning_rate=1e-2))], "DQN learning-rate sensitivity", "learning_rate_dqn.png"),
        ("replay_buffer_dqn", "dqn", [("capacity=1,000", replace(DQNConfig(), replay_capacity=1_000)), ("capacity=10,000", DQNConfig()), ("capacity=50,000", replace(DQNConfig(), replay_capacity=50_000))], "DQN replay-buffer capacity sensitivity", "replay_buffer_dqn.png"),
        ("target_update_dqn", "dqn", [("every 25 steps", replace(DQNConfig(), target_update_frequency=25)), ("every 100 steps", DQNConfig()), ("every 500 steps", replace(DQNConfig(), target_update_frequency=500))], "DQN target-network update sensitivity", "target_update_dqn.png"),
    ]
    for family, algorithm, variants, title, output in study_specs:
        aggregates: dict[str, pd.DataFrame] = {}
        for label, configuration in variants:
            safe_label = label.replace("=", "_").replace(" ", "_").replace(",", "").replace("ε", "epsilon")
            name = f"{family}_{safe_label}"
            histories, summaries = run_configuration(name, algorithm, configuration, experiment_config.seeds, experiment_config.evaluation_episodes)
            aggregates[label] = summarize(name, algorithm, histories, summaries)
        plot_comparison(aggregates, title, output)


def seed_experiments(experiment_config: ExperimentConfig = ExperimentConfig()) -> None:
    for algorithm, configuration, title, output in [("q_learning", QLearningConfig(), "Q-Learning reproducibility across random seeds", "q_learning_seeds.png"), ("dqn", DQNConfig(), "DQN reproducibility across random seeds", "dqn_seeds.png")]:
        aggregates: dict[str, pd.DataFrame] = {}
        for seed in experiment_config.seeds:
            histories, summaries = run_configuration(f"seed_study_{seed}", algorithm, configuration, (seed,), experiment_config.evaluation_episodes)
            aggregates[f"seed {seed}"] = summarize(f"seed_study_{seed}", algorithm, histories, summaries)
        plot_comparison(aggregates, title, output)
