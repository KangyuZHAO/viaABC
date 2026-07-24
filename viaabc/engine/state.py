from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GenerationState:
    t: int
    particles: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    epsilon: float
    cov: np.ndarray
    simulations: int = 0
    qt: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, Any]:
        data = {
            "t": self.t,
            "particles": self.particles,
            "weights": self.weights,
            "distances": self.distances,
            "epsilon": self.epsilon,
            "cov": self.cov,
            "simulations": self.simulations,
            "meta": self.meta,
        }
        if self.qt is not None:
            data["qt"] = self.qt
        return data
