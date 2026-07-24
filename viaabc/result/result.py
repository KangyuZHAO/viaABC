from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ViaABCResult:
    generations: list[dict[str, Any]]
    config: Any = None
    spec: Any = None

    @property
    def final_generation(self) -> dict[str, Any]:
        if not self.generations:
            raise ValueError("No generations are available.")
        return self.generations[-1]

    @property
    def particles(self) -> np.ndarray:
        return np.asarray(self.final_generation["particles"])

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.final_generation["weights"])

    def posterior_mean(self) -> np.ndarray:
        return np.average(self.particles, weights=self.weights, axis=0)

    def posterior_cov(self) -> np.ndarray:
        mean = self.posterior_mean()
        centered = self.particles - mean
        return (centered * self.weights[:, None]).T @ centered

    def credible_interval(self, level: float = 0.95) -> np.ndarray:
        alpha = (1.0 - level) / 2.0
        return np.quantile(self.particles, [alpha, 1.0 - alpha], axis=0)

    def to_dataframe(self):
        import pandas as pd

        data = {f"theta_{i}": self.particles[:, i] for i in range(self.particles.shape[1])}
        data["weight"] = self.weights
        return pd.DataFrame(data)

    def summary(self) -> dict[str, Any]:
        return {
            "num_generations": len(self.generations),
            "epsilon": float(self.final_generation.get("epsilon", np.nan)),
            "posterior_mean": self.posterior_mean().tolist(),
            "credible_interval_95": self.credible_interval(0.95).tolist(),
        }

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "generations.npy", self.generations, allow_pickle=True)
        posterior = np.column_stack([self.particles, self.weights])
        header = ",".join([f"theta_{i}" for i in range(self.particles.shape[1])] + ["weight"])
        np.savetxt(output_dir / "posterior.csv", posterior, delimiter=",", header=header, comments="")
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(self.summary(), stream, indent=2)

    @classmethod
    def load(cls, output_dir: str | Path) -> "ViaABCResult":
        generations = np.load(Path(output_dir) / "generations.npy", allow_pickle=True).tolist()
        return cls(generations=generations)
