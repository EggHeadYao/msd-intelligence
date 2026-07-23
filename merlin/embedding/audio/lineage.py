from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from columns import CONTRACT_VERSION, PREPARED_AUDIO_COLUMNS, SHARED_FEATURE_COLUMNS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_path(path: str | Path) -> str:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"artifact does not exist: {root}")
    files = (root,) if root.is_file() else tuple(
        sorted(item for item in root.rglob("*") if item.is_file())
    )
    digest = hashlib.sha256()
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def feature_order_sha256(columns: Sequence[str]) -> str:
    payload = json.dumps(list(columns), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_provenance(repo_root: Path, pathspec: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    require(bool(git("ls-files", "--", pathspec)), f"no tracked source files match {pathspec}")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_pathspec": pathspec,
    }
