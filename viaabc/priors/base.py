from __future__ import annotations

from typing import Protocol

import numpy as np


class Prior(Protocol):
    names: tuple[str, ...]

    @property
    def ndim(self) -> int:
        ...

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ...

    def log_prob(self, theta: np.ndarray) -> float:
        ...

    def supports(self, theta: np.ndarray) -> bool:
        return np.isfinite(self.log_prob(theta))
