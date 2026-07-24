from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any] | list[str] | None) -> dict[str, Any]:
    if not overrides:
        return deepcopy(data)
    result = deepcopy(data)
    items = overrides.items() if isinstance(overrides, dict) else (_split_override(item) for item in overrides)
    for dotted_key, value in items:
        _set_dotted(result, dotted_key, _coerce_value(value))
    return result


def _split_override(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Override must be KEY=VALUE, got {text!r}.")
    key, value = text.split("=", 1)
    return key, value


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _coerce_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
