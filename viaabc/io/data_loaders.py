from __future__ import annotations

from pathlib import Path

import numpy as np


def load_observed(path: str | Path, loader: str = "numpy"):
    path = Path(path)
    loader = loader.lower()
    if loader in {"numpy", "npy"}:
        return np.load(path, allow_pickle=True)
    if loader in {"txt", "text"}:
        return np.loadtxt(path)
    if loader == "csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported observed data loader {loader!r}.")
