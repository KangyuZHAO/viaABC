from __future__ import annotations

import logging
from typing import Any

import numpy as np

from viaabc.config.schema import ABCConfig
from viaabc.distance import get_distance
from viaabc.engine.proposal import diagonal_population_covariance, weighted_gaussian_proposal
from viaabc.engine.state import GenerationState
from viaabc.engine.stopping import compute_stopping_statistic, should_stop
from viaabc.engine.tolerance import update_tolerance
from viaabc.engine.weights import initial_weights, update_weights
from viaabc.system.base import is_legacy_viaabc_system

log = logging.getLogger(__name__)


class ViaABCEngine:
    """Default viaABC population-based inference engine.

    Legacy `src.viaABC` systems are delegated to their existing optimized run
    method. New lightweight systems use the implementation in this module.
    """

    def __init__(self, system: Any, prior: Any = None, encoder: Any = None, distance: str | Any = "l2", runtime: Any = None) -> None:
        self.system = system
        self.prior = prior
        self.encoder = encoder
        self.distance = get_distance(distance)
        self.runtime = runtime

    def run(self, observed: np.ndarray | None, config: ABCConfig) -> list[dict[str, Any]]:
        if is_legacy_viaabc_system(self.system):
            return self._run_legacy(config)
        return [state.to_legacy_dict() for state in self._run_generic(observed, config)]

    def _run_legacy(self, config: ABCConfig) -> list[dict[str, Any]]:
        log.info("Running legacy optimized viaABC engine")
        self.system.run(
            num_particles=config.num_particles,
            k=config.k,
            q_threshold=config.q_threshold,
            max_generations=config.max_generations,
            num_workers=config.num_workers,
            simulation_batch_size=config.simulation_batch_size,
            max_pending_simulations=config.max_pending_simulations,
        )
        return list(self.system.generations)

    def _run_generic(self, observed: np.ndarray | None, config: ABCConfig) -> list[GenerationState]:
        if self.prior is None:
            raise ValueError("A Prior is required for non-legacy systems.")
        if self.encoder is None:
            raise ValueError("An Encoder is required for non-legacy systems.")
        if observed is None:
            raise ValueError("Observed data is required for non-legacy systems.")
        if self.runtime is None:
            from viaabc.runtime import Runtime

            self.runtime = Runtime.auto(num_workers=config.num_workers, simulation_batch_size=config.simulation_batch_size)

        rng = np.random.default_rng(config.seed)
        observed_repr = self.encoder.encode(self.system.preprocess(observed))

        states = [self._initialize_population(observed_repr, config, rng)]
        for generation in range(1, config.max_generations + 1):
            current = self._sample_generation(states[-1], observed_repr, generation, config, rng)
            current.qt = compute_stopping_statistic(states[-1].particles, current.particles)
            states.append(current)
            if should_stop(generation, current.qt, config.q_threshold, config.max_generations):
                break
        return states

    def _initialize_population(self, observed_repr: np.ndarray, config: ABCConfig, rng: np.random.Generator) -> GenerationState:
        target = config.k * config.num_particles
        particles, distances = self._sample_from_prior(target, observed_repr, config, rng)
        order = np.argsort(distances)
        particles = particles[order][: config.num_particles]
        distances = distances[order][: config.num_particles]
        weights = initial_weights(config.num_particles)
        cov = diagonal_population_covariance(particles)
        epsilon = float(distances[-1])
        return GenerationState(0, particles, weights, distances, epsilon, cov, simulations=target)

    def _sample_from_prior(
        self,
        target: int,
        observed_repr: np.ndarray,
        config: ABCConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        particles = []
        distances = []
        simulations = 0
        while len(particles) < target:
            theta_batch = np.asarray([self.prior.sample(rng) for _ in range(max(1, config.simulation_batch_size or 32))])
            theta_batch = np.asarray([theta for theta in theta_batch if self.prior.supports(theta)])
            results = self.runtime.simulate_batch(self.system, theta_batch)
            simulations += len(results)
            successful_theta = []
            successful_data = []
            for theta, (sample, status) in zip(theta_batch, results):
                if status == 0:
                    successful_theta.append(theta)
                    successful_data.append(sample)
            if not successful_data:
                continue
            encoded = self.runtime.encode_batch(self.system, self.encoder, successful_data)
            for theta, representation in zip(successful_theta, encoded):
                particles.append(theta)
                distances.append(self.distance(observed_repr, representation))
                if len(particles) >= target:
                    break
        log.info("Initialization used %d simulations", simulations)
        return np.asarray(particles, dtype=np.float64), np.asarray(distances, dtype=np.float64)

    def _sample_generation(
        self,
        previous: GenerationState,
        observed_repr: np.ndarray,
        generation: int,
        config: ABCConfig,
        rng: np.random.Generator,
    ) -> GenerationState:
        accepted_particles = []
        accepted_distances = []
        simulations = 0
        while len(accepted_particles) < config.num_particles:
            batch_size = max(1, config.simulation_batch_size or 32)
            proposals = []
            while len(proposals) < batch_size:
                theta = weighted_gaussian_proposal(previous.particles, previous.weights, previous.cov, rng)
                if self.prior.supports(theta):
                    proposals.append(theta)
            theta_batch = np.asarray(proposals, dtype=np.float64)
            results = self.runtime.simulate_batch(self.system, theta_batch)
            simulations += len(results)
            successful_theta = []
            successful_data = []
            for theta, (sample, status) in zip(theta_batch, results):
                if status == 0:
                    successful_theta.append(theta)
                    successful_data.append(sample)
            if not successful_data:
                continue
            encoded = self.runtime.encode_batch(self.system, self.encoder, successful_data)
            for theta, representation in zip(successful_theta, encoded):
                distance = self.distance(observed_repr, representation)
                if distance <= previous.epsilon:
                    accepted_particles.append(theta)
                    accepted_distances.append(distance)
                    if len(accepted_particles) >= config.num_particles:
                        break

        particles = np.asarray(accepted_particles, dtype=np.float64)
        distances = np.asarray(accepted_distances, dtype=np.float64)
        cov = diagonal_population_covariance(particles)
        weights = update_weights(particles, previous.particles, previous.weights, self.prior, previous.cov)
        epsilon = update_tolerance(distances, config.epsilon_quantile)
        return GenerationState(generation, particles, weights, distances, epsilon, cov, simulations=simulations)
