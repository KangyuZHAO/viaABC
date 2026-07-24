from __future__ import annotations

import numpy as np


def compute_stopping_statistic(previous_particles: np.ndarray, current_particles: np.ndarray) -> float:
    prev = np.asarray(previous_particles, dtype=np.float64)
    curr = np.asarray(current_particles, dtype=np.float64)
    if prev.size == 0 or curr.size == 0:
        return 0.0
    prev_mean = prev.mean(axis=0)
    curr_mean = curr.mean(axis=0)
    denom = np.linalg.norm(prev_mean) + 1e-8
    return float(1.0 - min(1.0, np.linalg.norm(curr_mean - prev_mean) / denom))


def should_stop(generation: int, qt: float | None, q_threshold: float, max_generations: int) -> bool:
    if generation >= max_generations:
        return True
    return qt is not None and qt >= q_threshold
