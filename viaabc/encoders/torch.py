from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TorchEncoder:
    """Adapter for models exposing get_latent(x, pooling_method)."""

    def __init__(self, model: Any, pooling_method: str = "no_cls") -> None:
        self.model = model.eval() if hasattr(model, "eval") else model
        self.pooling_method = pooling_method

    @property
    def device(self):
        import torch

        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def encode(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            if hasattr(self.model, "get_latent"):
                encoded = self.model.get_latent(tensor, self.pooling_method)
            else:
                encoded = self.model(tensor)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.detach().cpu().numpy()
            return np.asarray(encoded)


class TorchCheckpointEncoder(TorchEncoder):
    """Minimal checkpoint encoder hook.

    Loading arbitrary project checkpoints needs a model factory, which keeps the
    core package independent from Hydra and project-specific Lightning modules.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_factory,
        pooling_method: str = "no_cls",
        map_location: str = "cpu",
    ) -> None:
        import torch

        model = model_factory()
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(state, strict=False)
        super().__init__(model=model, pooling_method=pooling_method)
        self.checkpoint_path = Path(checkpoint_path)
