from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

from viaabc.io import canonical_hash, sha256_file


def build_fingerprint(spec: Any) -> dict[str, Any]:
    data = {
        "viaabc_version": _package_version(),
        "git_commit": _git_commit(),
        "spec_hash": canonical_hash(spec),
    }
    observed_path = getattr(getattr(spec, "observed_data", None), "path", None)
    if observed_path and Path(observed_path).exists():
        data["observed_data_sha256"] = sha256_file(observed_path)
    return data


def _package_version() -> str:
    try:
        return importlib.metadata.version("viaabc")
    except importlib.metadata.PackageNotFoundError:
        return "editable"


def _git_commit() -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return output.strip()
    except Exception:
        return None
