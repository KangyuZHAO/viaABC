from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ABCConfig:
    num_particles: int = 1000
    k: int = 5
    q_threshold: float = 0.99
    max_generations: int = 20
    epsilon_quantile: float = 0.5
    distance: str = "l2"
    num_workers: int | None = None
    simulation_batch_size: int | None = None
    max_pending_simulations: int | None = None
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ABCConfig":
        return cls(**(data or {}))


@dataclass
class ComponentSpec:
    target: str | None = None
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    @property
    def locator(self) -> str:
        locator = self.target or self.name
        if not locator:
            raise ValueError("ComponentSpec requires either `target` or `name`.")
        return locator

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str | None) -> "ComponentSpec":
        if data is None:
            return cls()
        if isinstance(data, str):
            return cls(target=data)
        return cls(
            target=data.get("target"),
            name=data.get("name"),
            params=dict(data.get("params", {})),
            version=data.get("version"),
        )


@dataclass
class DataSpec:
    path: str | None = None
    loader: str = "numpy"
    version: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DataSpec":
        return cls(**(data or {}))


@dataclass
class ExperimentSpec:
    name: str = "viaabc_experiment"
    seed: int | None = None
    system: ComponentSpec = field(default_factory=ComponentSpec)
    prior: ComponentSpec | dict[str, Any] | None = None
    encoder: ComponentSpec = field(default_factory=lambda: ComponentSpec(name="encoder.identity"))
    distance: str | ComponentSpec = "l2"
    runtime: ComponentSpec = field(default_factory=lambda: ComponentSpec(name="runtime.auto"))
    tracking: dict[str, Any] = field(default_factory=dict)
    observed_data: DataSpec = field(default_factory=DataSpec)
    abc: ABCConfig = field(default_factory=ABCConfig)
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSpec":
        experiment = data.get("experiment", {})
        abc = ABCConfig.from_dict(data.get("abc"))
        if experiment.get("seed") is not None and abc.seed is None:
            abc.seed = experiment["seed"]

        distance_data = data.get("distance", abc.distance)
        distance: str | ComponentSpec
        if isinstance(distance_data, dict):
            distance = ComponentSpec.from_dict(distance_data)
            if distance.name:
                abc.distance = distance.name
        else:
            distance = str(distance_data)
            abc.distance = distance

        return cls(
            name=experiment.get("name", data.get("name", "viaabc_experiment")),
            seed=experiment.get("seed", abc.seed),
            system=ComponentSpec.from_dict(data.get("system")),
            prior=data.get("prior"),
            encoder=ComponentSpec.from_dict(data.get("encoder", {"name": "encoder.identity"})),
            distance=distance,
            runtime=ComponentSpec.from_dict(data.get("runtime", {"name": "runtime.auto"})),
            tracking=dict(data.get("tracking", {})),
            observed_data=DataSpec.from_dict(data.get("observed_data")),
            abc=abc,
            output=dict(data.get("output", {})),
        )
