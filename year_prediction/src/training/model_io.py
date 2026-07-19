from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    output = path.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output
