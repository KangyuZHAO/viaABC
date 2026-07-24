from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from viaabc import ABCConfig, IdentityEncoder, infer
from viaabc.priors import JointPrior, UniformPrior
from viaabc.runtime import Runtime

from examples.viaabc_ioc.lotka_system import LotkaSystem
from examples.viaabc_ioc.spatial_sir3d_system import SpatialSIR3DSystem


OUTPUT_DIR = Path("outputs/viaabc_ioc_smoke")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_lotka()
    run_spatial_sir3d()


def run_lotka() -> None:
    system = LotkaSystem()
    observed, _ = system.simulate(np.array([1.0, 1.0]))
    prior = JointPrior(
        {
            "alpha": UniformPrior(0.0, 2.0),
            "delta": UniformPrior(0.0, 2.0),
        }
    )
    result = infer(
        system=system,
        prior=prior,
        encoder=IdentityEncoder(),
        observed=observed,
        config=ABCConfig(
            num_particles=24,
            k=2,
            max_generations=2,
            q_threshold=0.999,
            epsilon_quantile=0.5,
            num_workers=4,
            simulation_batch_size=8,
            seed=7,
            distance="l2",
        ),
        runtime=Runtime.cpu(num_workers=4),
    )
    result.save(OUTPUT_DIR / "lotka")

    posterior_mean = result.posterior_mean()
    predicted, _ = system.simulate(posterior_mean)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(system.time_space, observed[:, 0], label="observed prey", color="tab:blue")
    axes[0].plot(system.time_space, predicted[:, 0], "--", label="posterior prey", color="tab:blue")
    axes[0].plot(system.time_space, observed[:, 1], label="observed predator", color="tab:red")
    axes[0].plot(system.time_space, predicted[:, 1], "--", label="posterior predator", color="tab:red")
    axes[0].set_title("Lotka trajectory")
    axes[0].legend(fontsize=7)

    axes[1].scatter(result.particles[:, 0], result.particles[:, 1], c=result.weights, cmap="viridis")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("delta")
    axes[1].set_title("Posterior particles")

    eps = [generation["epsilon"] for generation in result.generations]
    axes[2].plot(range(len(eps)), eps, marker="o")
    axes[2].set_xlabel("generation")
    axes[2].set_ylabel("epsilon")
    axes[2].set_title("Tolerance trace")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lotka_visualization.png", dpi=160)
    plt.close(fig)


def run_spatial_sir3d() -> None:
    system = SpatialSIR3DSystem(grid_size=24, time_space=[1.0, 2.0, 3.0, 4.0], radius=3, seed=11)
    observed, _ = system.simulate(np.array([1.1, 1.2]))
    np.save(OUTPUT_DIR / "spatial_sir3d_observed.npy", observed)
    prior = JointPrior(
        {
            "beta": UniformPrior(0.2, 2.0),
            "tau_I": UniformPrior(0.5, 2.2),
        }
    )
    result = infer(
        system=system,
        prior=prior,
        encoder=IdentityEncoder(),
        observed=observed,
        config=ABCConfig(
            num_particles=20,
            k=2,
            max_generations=2,
            q_threshold=0.999,
            epsilon_quantile=0.5,
            num_workers=4,
            simulation_batch_size=8,
            seed=13,
            distance="l2",
        ),
        runtime=Runtime.cpu(num_workers=4),
    )
    result.save(OUTPUT_DIR / "spatial_sir3d")

    predicted, _ = system.simulate(result.posterior_mean())

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    cmap = "viridis"
    axes[0].imshow(_sir_frame_labels(observed), vmin=0, vmax=2, cmap=cmap)
    axes[0].set_title("Observed final")
    axes[1].imshow(_sir_frame_labels(predicted), vmin=0, vmax=2, cmap=cmap)
    axes[1].set_title("Posterior final")
    axes[2].scatter(result.particles[:, 0], result.particles[:, 1], c=result.weights, cmap="magma")
    axes[2].set_xlabel("beta")
    axes[2].set_ylabel("tau_I")
    axes[2].set_title("Posterior particles")
    axes[3].plot([generation["epsilon"] for generation in result.generations], marker="o")
    axes[3].set_xlabel("generation")
    axes[3].set_ylabel("epsilon")
    axes[3].set_title("Tolerance trace")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spatial_sir3d_visualization.png", dpi=160)
    plt.close(fig)


def _sir_frame_labels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[0] == 3:
        return np.argmax(x[:, -1], axis=0)
    if x.ndim == 3:
        return x[-1]
    raise ValueError(f"Unsupported SIR frame shape: {x.shape}")


if __name__ == "__main__":
    main()
