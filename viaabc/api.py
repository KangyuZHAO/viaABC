from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from viaabc.config import ABCConfig, apply_overrides, read_dict, read_yaml
from viaabc.engine import ViaABCEngine
from viaabc.result import ViaABCResult
from viaabc.runtime import Runtime
from viaabc.tracking import NullTracker


def infer(
    system: Any,
    prior: Any = None,
    encoder: Any = None,
    observed: Any = None,
    config: ABCConfig | None = None,
    runtime: Any = None,
    tracker: Any = None,
    spec: Any = None,
) -> ViaABCResult:
    """Run viaABC inference while keeping algorithm steps in the engine layer."""

    config = config or ABCConfig()
    runtime = runtime or Runtime.auto(num_workers=config.num_workers, simulation_batch_size=config.simulation_batch_size)
    tracker = tracker or NullTracker()
    tracker.log_config(config)

    engine = ViaABCEngine(
        system=system,
        prior=prior,
        encoder=encoder,
        distance=config.distance,
        runtime=runtime,
    )
    generations = engine.run(observed=observed, config=config)
    for generation in generations:
        tracker.log_generation(generation)
    result = ViaABCResult(generations=generations, config=config, spec=spec)
    tracker.log_result(result)
    return result


def infer_from_config(filepath: str | Path, overrides: dict[str, Any] | list[str] | None = None) -> ViaABCResult:
    from viaabc.experiment.runner import infer_from_spec

    if overrides:
        with Path(filepath).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        spec = read_dict(apply_overrides(data, overrides))
    else:
        spec = read_yaml(filepath)
    return infer_from_spec(spec)
