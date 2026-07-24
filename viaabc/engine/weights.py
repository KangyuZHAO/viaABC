from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from viaabc.engine.proposal import stabilize_covariance


def initial_weights(num_particles: int) -> np.ndarray:
    return np.ones(num_particles, dtype=np.float64) / float(num_particles)


def update_weights(
    particles: np.ndarray,
    previous_particles: np.ndarray,
    previous_weights: np.ndarray,
    prior,
    cov: np.ndarray,
) -> np.ndarray:
    cov = stabilize_covariance(cov)
    weights = np.empty(len(particles), dtype=np.float64)
    for i, theta in enumerate(particles):
        numerator = np.exp(prior.log_prob(theta))
        kernel_values = np.asarray(
            [
                multivariate_normal.pdf(theta, mean=previous_theta, cov=cov)
                for previous_theta in previous_particles
            ],
            dtype=np.float64,
        )
        denominator = float(np.sum(previous_weights * kernel_values))
        weights[i] = numerator / (denominator + 1e-300)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return initial_weights(len(particles))
    return weights / total
