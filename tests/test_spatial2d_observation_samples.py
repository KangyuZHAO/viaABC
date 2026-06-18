import numpy as np

from src.viaABC.systems import Spatial2D


def test_spatial2d_uses_injected_observation_samples(monkeypatch):
    fallback_samples = {
        "sample_1": {
            "txt": "initial_grid1_cppn.txt",
            "image": None,
        }
    }
    injected_samples = {
        "sample_1": {
            "txt": "initial_grid1_cpp.txt",
            "image": None,
        }
    }
    used_txt_paths = []

    monkeypatch.setattr(
        Spatial2D,
        "_load_spatial2d_samples",
        staticmethod(lambda: fallback_samples),
    )

    def fake_load_sample_grids(self, sample_paths, sample_id):
        used_txt_paths.append(sample_paths[sample_id]["txt"])
        return np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)

    monkeypatch.setattr(Spatial2D, "_load_sample_grids", fake_load_sample_grids)

    Spatial2D(
        model=None,
        t0=0,
        tmax=1,
        sample_id="sample_1",
        observation_samples=injected_samples,
    )

    assert used_txt_paths == ["initial_grid1_cpp.txt"]
