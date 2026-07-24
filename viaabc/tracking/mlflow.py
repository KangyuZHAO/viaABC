from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class MLflowTracker:
    def __init__(self, experiment_name: str = "viaabc", run_name: str | None = None, tracking_uri: str | None = None) -> None:
        import viaabc.tracking.mlflow as mlflow

        self.mlflow = mlflow
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)

    def log_config(self, config: Any) -> None:
        params = vars(config) if hasattr(config, "__dict__") else {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                self.mlflow.log_param(key, value)

    def log_generation(self, generation: dict) -> None:
        step = int(generation.get("t", len(generation)))
        if "epsilon" in generation:
            self.mlflow.log_metric("epsilon", float(generation["epsilon"]), step=step)
        if "qt" in generation and generation["qt"] is not None:
            self.mlflow.log_metric("qt", float(generation["qt"]), step=step)
        if "simulations" in generation:
            self.mlflow.log_metric("simulations", float(generation["simulations"]), step=step)

    def log_result(self, result) -> None:
        summary = result.summary()
        if "epsilon" in summary:
            self.mlflow.log_metric("final_epsilon", float(summary["epsilon"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "viaabc_result"
            result.save(path)
            self.mlflow.log_artifacts(str(path))

    def end(self) -> None:
        self.mlflow.end_run()
