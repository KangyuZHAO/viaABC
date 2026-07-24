from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from viaabc.config.overrides import apply_overrides
from viaabc.config.schema import ExperimentSpec


def expand_matrix_sweep(base_data: dict[str, Any]) -> list[ExperimentSpec]:
    sweep = base_data.get("sweep", {}).get("matrix")
    if not sweep:
        return [ExperimentSpec.from_dict(base_data)]
    keys = list(sweep)
    specs = []
    for values in product(*(sweep[key] for key in keys)):
        overrides = dict(zip(keys, values))
        data = apply_overrides(deepcopy(base_data), overrides)
        data.pop("sweep", None)
        specs.append(ExperimentSpec.from_dict(data))
    return specs
