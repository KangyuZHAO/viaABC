from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

import numpy as np


class CallableSystem:
    """Small adapter for function-based user systems."""

    def __init__(
        self,
        simulate: Callable[[np.ndarray], np.ndarray | tuple[np.ndarray, int]],
        preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        parameter_space: Mapping[str, Any] | None = None,
    ) -> None:
        self._simulate = simulate
        self._preprocess = preprocess or (lambda x: x)
        self.parameter_space = dict(parameter_space or {})

    def simulate(self, theta: np.ndarray) -> tuple[np.ndarray, int]:
        output = self._simulate(theta)
        if isinstance(output, tuple) and len(output) == 2:
            return output
        return output, 0

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        return self._preprocess(x)
