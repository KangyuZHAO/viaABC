from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from viaabc.config.schema import ExperimentSpec


def read_yaml(filepath: str | Path) -> ExperimentSpec:
    with Path(filepath).open("r", encoding="utf-8") as stream:
        return read_dict(yaml.safe_load(stream) or {})


def read_json(filepath: str | Path) -> ExperimentSpec:
    with Path(filepath).open("r", encoding="utf-8") as stream:
        return read_dict(json.load(stream))


def read_dict(data: dict[str, Any]) -> ExperimentSpec:
    return ExperimentSpec.from_dict(data)
