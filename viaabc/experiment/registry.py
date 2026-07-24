from __future__ import annotations

import importlib
from typing import Any


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        self._items[name] = obj

    def resolve(self, name_or_target: str) -> Any:
        if name_or_target in self._items:
            return self._items[name_or_target]
        return import_object(name_or_target)


def import_object(path: str) -> Any:
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


registry = ComponentRegistry()
