from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any


C1_MANIFEST_NAME = "c1_manifest.json"


def write_json_atomic(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def staging_directory(output: Path, run_id: str) -> Path:
    absolute = output.absolute()
    if not absolute.name:
        raise ValueError("C1 output must be a named directory")
    staging = absolute.parent / f".{absolute.name}.staging-{run_id}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"C1 staging path already exists: {staging}")
    staging.mkdir(parents=True)
    return staging


def publish_directory(staging: Path, output: Path, run_id: str) -> None:
    absolute = output.absolute()
    backup = absolute.parent / f".{absolute.name}.backup-{run_id}"
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"C1 backup path already exists: {backup}")
    had_output = absolute.exists() or absolute.is_symlink()
    if had_output:
        absolute.replace(backup)
    try:
        staging.replace(absolute)
    except BaseException:
        if had_output and backup.exists() and not absolute.exists():
            backup.replace(absolute)
        raise
    if had_output:
        try:
            remove_path(backup)
        except OSError as error:
            warnings.warn(f"failed to remove old C1 backup {backup}: {error}", RuntimeWarning)
