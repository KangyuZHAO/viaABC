from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from viaabc.system import CallableSystem


class SpatialSIR3DSystem(CallableSystem):
    """Framework-conforming example component matching legacy SpatialSIR3D."""

    parameter_space = {
        "beta": {"lower": 0.2, "upper": 4.5, "description": "infection rate"},
        "tau_I": {"lower": 0.2, "upper": 4.5, "description": "infection duration"},
    }

    def __init__(
        self,
        grid_size: int = 80,
        initial_infected: int = 5,
        radius: int = 5,
        t0: int = 0,
        tmax: int = 16,
        interval: int = 1,
        time_space: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        del t0, tmax, interval, initial_infected
        self.grid_size = int(grid_size)
        self.radius = int(radius)
        self.time_space = np.asarray(time_space if time_space is not None else np.arange(1, 16, 1), dtype=np.float64)
        self.seed = seed
        self.centers = _default_centers(self.grid_size)
        super().__init__(
            simulate=self._simulate,
            preprocess=preprocess_spatial_sir3d,
            parameter_space=self.parameter_space,
        )

    def _simulate(self, theta: np.ndarray) -> tuple[np.ndarray, int]:
        return simulate_spatial_sir3d(
            theta=theta,
            grid_size=self.grid_size,
            radius=self.radius,
            time_space=self.time_space,
            centers=self.centers,
            seed=self.seed,
        )


def simulate_spatial_sir3d(
    theta: np.ndarray,
    grid_size: int,
    radius: int,
    time_space: np.ndarray,
    centers: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, int]:
    susceptible, infected, recovered = 0, 1, 2
    beta, tau_i = np.asarray(theta, dtype=np.float64)
    rng = np.random.default_rng(seed)

    dt = 0.05
    resistance_duration = 1.0
    steps = int(np.round(float(np.max(time_space)) / dt))

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    infection_timer = np.zeros_like(grid, dtype=np.float32)
    recovery_timer = np.zeros_like(grid, dtype=np.float32)

    for x, y in centers:
        dx, dy = rng.integers(-radius, radius + 1, size=2)
        xi, yi = np.clip([x + dx, y + dy], 0, grid_size - 1)
        grid[xi, yi] = infected

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    target_steps = set((time_space / dt).astype(int).tolist())
    frames: list[np.ndarray] = []

    for step in range(1, steps + 1):
        infected_neighbors = convolve((grid == infected).astype(np.uint8), kernel, mode="constant")
        p_inf = 1.0 - np.exp(-beta * infected_neighbors * dt)

        new_infections = (grid == susceptible) & (rng.random(grid.shape) < p_inf)
        grid[new_infections] = infected
        infection_timer[new_infections] = 0.0
        recovery_timer[new_infections] = 0.0

        infection_timer[grid == infected] += dt
        to_recover = (grid == infected) & (infection_timer >= tau_i)
        grid[to_recover] = recovered
        infection_timer[to_recover] = 0.0
        recovery_timer[to_recover] = 0.0

        recovery_timer[grid == recovered] += dt
        to_reset = (grid == recovered) & (recovery_timer >= resistance_duration)
        grid[to_reset] = susceptible
        recovery_timer[to_reset] = 0.0

        if step in target_steps:
            frames.append(grid.copy())

    return np.asarray(frames, dtype=np.uint8), 0


def preprocess_spatial_sir3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        return x.astype(np.float32)
    if x.ndim == 4 and x.shape[0] == 3:
        return np.argmax(x, axis=0).astype(np.float32)
    raise ValueError(f"SpatialSIR3D input must be [T,H,W] labels or [3,T,H,W] one-hot, got {x.shape}.")


def labels_to_onehot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    return np.eye(3, dtype=np.float32)[y].transpose(3, 0, 1, 2)


def _default_centers(grid_size: int) -> np.ndarray:
    centers = np.array([[44, 67], [24, 67], [64, 73], [3, 55], [12, 20]], dtype=np.int64)
    if grid_size != 80:
        centers = np.clip(centers, 0, grid_size - 1)
    return centers
