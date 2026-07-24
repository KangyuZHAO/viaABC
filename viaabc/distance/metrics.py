from __future__ import annotations

from collections.abc import Callable

import numpy as np


def l1(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(np.asarray(x) - np.asarray(y)).mean())


def l2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(x) - np.asarray(y)).reshape(-1)))


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = np.asarray(x).reshape(-1)
    y_flat = np.asarray(y).reshape(-1)
    denom = np.linalg.norm(x_flat) * np.linalg.norm(y_flat)
    return float(1.0 - np.dot(x_flat, y_flat) / (denom + 1e-8))


REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "l1": l1,
    "l2": l2,
    "cosine": cosine,
}


def get_distance(name_or_fn: str | Callable[[np.ndarray, np.ndarray], float]):
    if callable(name_or_fn):
        return name_or_fn
    try:
        return REGISTRY[name_or_fn]
    except KeyError as exc:
        raise ValueError(f"Unknown distance metric {name_or_fn!r}.") from exc
