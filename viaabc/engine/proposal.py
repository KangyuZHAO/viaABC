from __future__ import annotations

import numpy as np


def stabilize_covariance(cov: np.ndarray, min_variance: float = 1e-8) -> np.ndarray:
    cov = np.asarray(np.atleast_2d(cov), dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag_indices_from(cov)
    cov[diag] = np.maximum(cov[diag], min_variance)
    jitter = min_variance
    for _ in range(8):
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            cov[diag] += jitter
            jitter *= 10.0
    return cov


def diagonal_population_covariance(particles: np.ndarray) -> np.ndarray:
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim == 1:
        particles = particles.reshape(-1, 1)
    cov = np.cov(particles, rowvar=False)
    cov = np.asarray(np.atleast_2d(cov), dtype=np.float64)
    return stabilize_covariance(2.0 * np.diag(np.diag(cov)))


def weighted_gaussian_proposal(
    particles: np.ndarray,
    weights: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    index = int(rng.choice(len(particles), p=weights / weights.sum()))
    return rng.multivariate_normal(np.asarray(particles[index], dtype=np.float64), stabilize_covariance(cov))
