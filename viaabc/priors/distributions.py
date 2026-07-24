from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm, uniform


@dataclass(frozen=True)
class UniformPrior:
    low: float
    high: float
    name: str = "theta"

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name,)

    @property
    def ndim(self) -> int:
        return 1

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.asarray([rng.uniform(self.low, self.high)], dtype=np.float64)

    def log_prob(self, theta: np.ndarray) -> float:
        return float(uniform.logpdf(float(np.asarray(theta)[0]), loc=self.low, scale=self.high - self.low))

    def supports(self, theta: np.ndarray) -> bool:
        value = float(np.asarray(theta)[0])
        return self.low <= value <= self.high


@dataclass(frozen=True)
class LogNormalPrior:
    mean: float
    sigma: float
    name: str = "theta"

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name,)

    @property
    def ndim(self) -> int:
        return 1

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.asarray([rng.lognormal(self.mean, self.sigma)], dtype=np.float64)

    def log_prob(self, theta: np.ndarray) -> float:
        return float(lognorm.logpdf(float(np.asarray(theta)[0]), s=self.sigma, scale=np.exp(self.mean)))

    def supports(self, theta: np.ndarray) -> bool:
        return float(np.asarray(theta)[0]) > 0
