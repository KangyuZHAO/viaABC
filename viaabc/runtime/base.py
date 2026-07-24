from __future__ import annotations

from typing import Any

import numpy as np


class Runtime:
    """Execution strategy boundary for performance-sensitive work."""

    @classmethod
    def auto(cls, **kwargs: Any) -> "Runtime":
        try:
            import torch

            if torch.cuda.is_available():
                from viaabc.runtime.gpu import GPURuntime

                return GPURuntime(**kwargs)
        except Exception:
            pass
        from viaabc.runtime.local import LocalRuntime

        return LocalRuntime(**kwargs)

    @classmethod
    def cpu(cls, **kwargs: Any) -> "Runtime":
        from viaabc.runtime.local import LocalRuntime

        return LocalRuntime(**kwargs)

    @classmethod
    def gpu(cls, **kwargs: Any) -> "Runtime":
        from viaabc.runtime.gpu import GPURuntime

        return GPURuntime(**kwargs)

    def simulate_batch(self, system, theta_batch: np.ndarray) -> list[tuple[np.ndarray, int]]:
        raise NotImplementedError

    def encode_batch(self, system, encoder, simulated_batch: list[np.ndarray]) -> np.ndarray:
        processed = [system.preprocess(sample) for sample in simulated_batch]
        return np.asarray([encoder.encode(sample) for sample in processed])
