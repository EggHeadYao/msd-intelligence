from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Sequence

from columns import CONTRACT_VERSION, PREPARED_AUDIO_COLUMNS, SHARED_FEATURE_COLUMNS

C1_MANIFEST_NAME = "c1_manifest.json"
C1_OUTPUTS = {
    "embeddings": "song_embeddings_audio.parquet",
    "scaler_model": "scaler_model",
    "pca_model": "pca_model",
    "encoder_metadata": "audio_encoder_metadata.json",
}
C1_SUCCESS_MARKERS = {
    "embeddings": ("_SUCCESS",),
    "scaler_model": ("data/_SUCCESS", "metadata/_SUCCESS"),
    "pca_model": ("data/_SUCCESS", "metadata/_SUCCESS"),
    "encoder_metadata": (),
}


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


def replace_artifact(source: Path, target: Path, run_id: str) -> None:
    backup = target.parent / f".{target.name}.backup-{run_id}"
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"artifact backup already exists: {backup}")
    had_target = target.exists() or target.is_symlink()
    if had_target:
        target.replace(backup)
    try:
        source.replace(target)
    except BaseException:
        if had_target and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    if had_target:
        try:
            remove_path(backup)
        except OSError as error:
            warnings.warn(f"failed to remove artifact backup {backup}: {error}", RuntimeWarning)


def build_c1_manifest(staging: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    outputs = {name: {"path": relative} for name, relative in C1_OUTPUTS.items()}
    return {
        "artifact_type": "c1_audio_encoder",
        "artifact_version": "v2",
        "status": "valid",
        "run_id": metadata["run_id"],
        "created_at_utc": metadata["created_at_utc"],
        "producer": metadata["producer"],
        "parent_prepared_manifest": metadata["parent_prepared_manifest"],
        "input": {
            "path": metadata["input_path"],
            "schema_sha256": metadata["input_schema_sha256"],
            "row_count": metadata["row_count"],
        },
        "outputs": outputs,
    }


def validate_c1_manifest(output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output / C1_MANIFEST_NAME
    if not manifest_path.is_file():
        raise AssertionError(f"missing C1 manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "artifact_type": "c1_audio_encoder",
        "artifact_version": "v2",
        "status": "valid",
        "run_id": metadata["run_id"],
        "producer": metadata["producer"],
        "parent_prepared_manifest": metadata["parent_prepared_manifest"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise AssertionError(f"C1 manifest {key} mismatch")
    input_artifact = manifest.get("input", {})
    input_expected = {
        "path": metadata["input_path"],
        "schema_sha256": metadata["input_schema_sha256"],
        "row_count": metadata["row_count"],
    }
    if input_artifact != input_expected:
        raise AssertionError("C1 manifest input mismatch")
    outputs = manifest.get("outputs", {})
    if set(outputs) != set(C1_OUTPUTS):
        raise AssertionError("C1 manifest outputs mismatch")
    for name, relative in C1_OUTPUTS.items():
        item = outputs[name]
        if item != {"path": relative}:
            raise AssertionError(f"C1 manifest {name} path mismatch")
        artifact = output / relative
        if not artifact.exists():
            raise AssertionError(f"C1 output {name} is missing")
        for marker in C1_SUCCESS_MARKERS[name]:
            if not (artifact / marker).is_file():
                raise AssertionError(f"C1 output {name} is incomplete")
    return manifest


def _require(condition: bool, message: str) -> None:
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

    _require(bool(git("ls-files", "--", pathspec)), f"no tracked source files match {pathspec}")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_pathspec": pathspec,
    }


