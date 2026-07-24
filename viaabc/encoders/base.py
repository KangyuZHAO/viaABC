from __future__ import annotations

from typing import Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, x: np.ndarray) -> np.ndarray:
        ...
