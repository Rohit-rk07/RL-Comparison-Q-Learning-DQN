"""Filesystem, serialization, and determinism helpers."""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    for directory in (RAW_RESULTS_DIR, PROCESSED_RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Tiny networks and batches are substantially faster and more reproducible
    # with a single CPU worker than with thread-pool scheduling overhead.
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
