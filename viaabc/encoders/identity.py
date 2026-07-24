from __future__ import annotations

import numpy as np


class IdentityEncoder:
    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)
