from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np


class ABCSystem(Protocol):
    """User-defined scientific system consumed by viaABC.

    A system owns domain logic only: parameter semantics, simulation, and raw
    data preprocessing. Population updates, tolerance updates, and stopping
    criteria live in the engine.
    """

    parameter_space: Mapping[str, Any]

    def simulate(self, theta: np.ndarray) -> tuple[np.ndarray, int]:
        """Generate one simulated sample. Status 0 means success."""

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Convert raw observed/simulated data into encoder-ready data."""


class LegacyViaABCSystem(Protocol):
    """Compatibility protocol for the existing src.viaABC engine subclasses."""

    generations: list[dict[str, Any]]

    def run(
        self,
        num_particles: int,
        q_threshold: float = 0.99,
        max_generations: int = 20,
        k: int = 5,
        num_workers: int | None = None,
        simulation_batch_size: int | None = None,
        max_pending_simulations: int | None = None,
    ) -> None:
        ...


def is_legacy_viaabc_system(system: Any) -> bool:
    return callable(getattr(system, "run", None)) and hasattr(system, "generations")
