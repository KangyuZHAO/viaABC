from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class LotkaSystem:
    parameter_space = {
        "alpha": {"lower": 0.0, "upper": 10.0},
        "delta": {"lower": 0.0, "upper": 10.0},
    }

    def __init__(
        self,
        state0: list[float] | None = None,
        t0: float = 0.0,
        tmax: float = 15.0,
        time_space: list[float] | None = None,
    ) -> None:
        self.state0 = np.asarray(state0 or [1.0, 0.5], dtype=np.float64)
        self.t0 = float(t0)
        self.tmax = float(tmax)
        self.time_space = np.asarray(
            time_space or [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4],
            dtype=np.float64,
        )

    def ode(self, t: float, state: np.ndarray, theta: np.ndarray) -> list[float]:
        alpha, delta = theta
        beta, gamma = 1.0, 1.0
        prey, predator = state
        return [
            prey * (alpha - beta * predator),
            predator * (-gamma + delta * prey),
        ]

    def simulate(self, theta: np.ndarray) -> tuple[np.ndarray, int]:
        solution = solve_ivp(
            self.ode,
            [self.t0, self.tmax],
            y0=self.state0,
            t_eval=self.time_space,
            args=(theta,),
        )
        return solution.y.T, solution.status

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)
