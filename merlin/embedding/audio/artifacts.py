from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any

from lineage import sha256_path


C1_MANIFEST_NAME = "c1_manifest.json"
C1_OUTPUTS = {
    "embeddings": "song_embeddings_audio.parquet",
    "scaler_model": "scaler_model",
    "pca_model": "pca_model",
    "encoder_metadata": "audio_encoder_metadata.json",
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


def build_c1_manifest(staging: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    outputs = {
        name: {"path": relative, "sha256": sha256_path(staging / relative)}
        for name, relative in C1_OUTPUTS.items()
    }
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
            "data_sha256": metadata["input_data_sha256"],
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
        "data_sha256": metadata["input_data_sha256"],
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
        if item.get("path") != relative or item.get("sha256") != sha256_path(output / relative):
            raise AssertionError(f"C1 manifest {name} hash mismatch")
    return manifest
