from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from viaabc.api import infer
from viaabc.config.schema import ComponentSpec, ExperimentSpec
from viaabc.encoders import IdentityEncoder
from viaabc.experiment.registry import registry
from viaabc.io import load_observed
from viaabc.priors import JointPrior
from viaabc.runtime import Runtime
from viaabc.tracking import MLflowTracker, NullTracker


def build_component(spec: ComponentSpec, default: Any = None) -> Any:
    if not spec.target and not spec.name:
        if default is not None:
            return default
        raise ValueError("Missing component spec.")
    cls_or_factory = registry.resolve(spec.locator)
    return cls_or_factory(**spec.params)


def build_prior(prior_spec: Any):
    if prior_spec is None:
        return None
    if isinstance(prior_spec, dict) and "target" not in prior_spec and "name" not in prior_spec:
        return JointPrior.from_config(prior_spec.get("params", prior_spec))
    return build_component(ComponentSpec.from_dict(prior_spec))


def build_runtime(spec: ComponentSpec, abc_config):
    if spec.name == "runtime.auto" or spec.target == "runtime.auto":
        return Runtime.auto(num_workers=abc_config.num_workers, simulation_batch_size=abc_config.simulation_batch_size)
    if spec.name == "runtime.cpu":
        return Runtime.cpu(**spec.params)
    if spec.name == "runtime.gpu":
        return Runtime.gpu(**spec.params)
    return build_component(spec)


def build_tracker(spec: ExperimentSpec):
    tracking = spec.tracking or {}
    backend = str(tracking.get("backend", "null")).lower()
    if backend == "mlflow":
        return MLflowTracker(
            experiment_name=tracking.get("experiment_name", "viaabc"),
            run_name=tracking.get("run_name", spec.name),
            tracking_uri=tracking.get("tracking_uri"),
        )
    return NullTracker()


def infer_from_spec(spec: ExperimentSpec):
    system = build_component(spec.system)
    prior = build_prior(spec.prior)
    encoder = build_component(spec.encoder, default=IdentityEncoder())
    runtime = build_runtime(spec.runtime, spec.abc)
    tracker = build_tracker(spec)

    observed = None
    if spec.observed_data.path:
        observed = load_observed(spec.observed_data.path, spec.observed_data.loader)

    result = infer(
        system=system,
        prior=prior,
        encoder=encoder,
        observed=observed,
        config=spec.abc,
        runtime=runtime,
        tracker=tracker,
        spec=spec,
    )

    output_dir = spec.output.get("dir") if spec.output else None
    if output_dir:
        result.save(output_dir)
        _save_resolved_spec(spec, output_dir)
    if hasattr(tracker, "end"):
        tracker.end()
    return result


def _save_resolved_spec(spec: ExperimentSpec, output_dir: str | Path) -> None:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_spec.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(spec), stream, indent=2, default=str)
