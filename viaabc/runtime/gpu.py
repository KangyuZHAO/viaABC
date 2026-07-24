from __future__ import annotations

import numpy as np

from viaabc.runtime.local import LocalRuntime


class GPURuntime(LocalRuntime):
    """GPU-aware runtime.

    Simulation remains CPU/thread based; encoder implementations can use their
    own model device. This keeps GPU optimization independent from simulator
    code and mirrors the existing high-throughput inference path.
    """

    def encode_batch(self, system, encoder, simulated_batch: list[np.ndarray]) -> np.ndarray:
        processed = [system.preprocess(sample) for sample in simulated_batch]
        try:
            return np.asarray(encoder.encode(np.stack(processed, axis=0)))
        except Exception:
            return super().encode_batch(system, encoder, simulated_batch)
