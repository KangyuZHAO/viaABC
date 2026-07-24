from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from viaabc.priors.base import Prior
from viaabc.priors.distributions import LogNormalPrior, UniformPrior


class JointPrior:
    """Independent joint prior over named scalar parameters."""

    def __init__(self, priors: Mapping[str, Prior] | list[Prior] | tuple[Prior, ...]) -> None:
        if isinstance(priors, Mapping):
            self._priors = tuple(_with_name(name, prior) for name, prior in priors.items())
        else:
            self._priors = tuple(priors)
        self.names = tuple(prior.names[0] for prior in self._priors)

    @property
    def ndim(self) -> int:
        return len(self._priors)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([prior.sample(rng).reshape(-1) for prior in self._priors])

    def log_prob(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != self.ndim:
            return float("-inf")
        return float(sum(prior.log_prob(theta[i : i + 1]) for i, prior in enumerate(self._priors)))

    def supports(self, theta: np.ndarray) -> bool:
        return np.isfinite(self.log_prob(theta))

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, float | str]]) -> "JointPrior":
        priors = {}
        for name, spec in config.items():
            prior_type = str(spec.get("type", "uniform")).lower()
            if prior_type == "uniform":
                priors[name] = UniformPrior(float(spec["low"]), float(spec["high"]), name=name)
            elif prior_type in {"lognormal", "log_norm", "log-normal"}:
                priors[name] = LogNormalPrior(float(spec["mean"]), float(spec["sigma"]), name=name)
            else:
                raise ValueError(f"Unsupported prior type {prior_type!r} for parameter {name!r}.")
        return cls(priors)


def _with_name(name: str, prior: Prior) -> Prior:
    if hasattr(prior, "name"):
        try:
            object.__setattr__(prior, "name", name)
        except Exception:
            pass
    return prior
