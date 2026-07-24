from __future__ import annotations

import numpy as np


def update_tolerance(distances: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.asarray(distances, dtype=np.float64), quantile))
