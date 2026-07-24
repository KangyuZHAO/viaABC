from __future__ import annotations

from typing import Protocol


class Tracker(Protocol):
    def log_config(self, config) -> None:
        ...

    def log_generation(self, generation: dict) -> None:
        ...

    def log_result(self, result) -> None:
        ...
