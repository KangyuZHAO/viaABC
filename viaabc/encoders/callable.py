from __future__ import annotations

from collections.abc import Callable

import numpy as np


class CallableEncoder:
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.function = function

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.function(x))
