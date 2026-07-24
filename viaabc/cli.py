from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from viaabc.api import infer_from_config
from viaabc.config import apply_overrides, read_dict
from viaabc.experiment.planner import expand_matrix_sweep
from viaabc.experiment.runner import infer_from_spec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="viaabc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run one viaABC experiment")
    infer_parser.add_argument("--config", required=True, help="YAML experiment config")
    infer_parser.add_argument("overrides", nargs="*", help="Dotted KEY=VALUE overrides")

    sweep_parser = subparsers.add_parser("sweep", help="Run matrix sweep from YAML")
    sweep_parser.add_argument("--config", required=True, help="YAML experiment config")
    sweep_parser.add_argument("overrides", nargs="*", help="Dotted KEY=VALUE overrides")

    args = parser.parse_args(argv)

    if args.command == "infer":
        infer_from_config(args.config, overrides=args.overrides)
        return

    if args.command == "sweep":
        with Path(args.config).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        data = apply_overrides(data, args.overrides)
        for spec in expand_matrix_sweep(data):
            infer_from_spec(spec)
        return


if __name__ == "__main__":
    main()
