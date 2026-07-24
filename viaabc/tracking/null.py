from __future__ import annotations


class NullTracker:
    def log_config(self, config) -> None:
        return None

    def log_generation(self, generation: dict) -> None:
        return None

    def log_result(self, result) -> None:
        return None
