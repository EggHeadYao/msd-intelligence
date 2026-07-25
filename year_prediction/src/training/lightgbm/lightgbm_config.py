from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PATH_ARGUMENTS = ("input", "manifest", "output")


def read_config(path: Path, allowed: set[str]) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("LightGBM configuration must be a JSON object")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown LightGBM configuration fields: {unknown}")
    for name in PATH_ARGUMENTS:
        if name in payload:
            payload[name] = Path(payload[name])
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
