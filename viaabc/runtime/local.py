from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from viaabc.runtime.base import Runtime


class LocalRuntime(Runtime):
    def __init__(self, num_workers: int | None = None, simulation_batch_size: int | None = None, **_: object) -> None:
        self.num_workers = num_workers
        self.simulation_batch_size = simulation_batch_size

    def simulate_batch(self, system, theta_batch: np.ndarray) -> list[tuple[np.ndarray, int]]:
        theta_batch = list(theta_batch)
        if not theta_batch:
            return []
        max_workers = self.num_workers or min(32, len(theta_batch))
        max_workers = max(1, min(max_workers, len(theta_batch)))
        if max_workers == 1:
            return [system.simulate(theta) for theta in theta_batch]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(system.simulate, theta_batch))
