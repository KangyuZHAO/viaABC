# viaABC IoC examples

This folder contains user-facing component templates for the new viaABC API.
Core package code should not be edited for a new scientific problem. Instead,
define a system class here or in your own project and reference it from YAML.

Minimal workflow:

```python
from viaabc import infer_from_config

result = infer_from_config("examples/viaabc_ioc/infer_lotka.yaml")
print(result.posterior_mean())
```

Command-line workflow:

```bash
viaabc infer --config examples/viaabc_ioc/infer_lotka.yaml abc.num_particles=500
```

The observed data file in the YAML is intentionally external. Generate it with
your simulator or replace the path with your own `.npy` observation.
