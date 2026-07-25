from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from merlin.embedding.audio.columns import (
    CONTRACT_VERSION,
    PREPARED_AUDIO_COLUMNS,
    SHARED_FEATURE_COLUMNS,
)

C1_MANIFEST_NAME = "c1_manifest.json"
ENCODER_METADATA_NAME = "audio_encoder_metadata.json"
CANONICAL_EMBEDDING_DIMENSION = 128
AUDIO_INDEX_NAME = "index_audio.faiss"
AUDIO_MAPPING_NAME = "index_audio_track_ids.parquet"
FAISS_MANIFEST_NAME = "index_audio_manifest.json"
FAISS_MANIFEST_VERSION = "merlin_faiss_index_v1"
C1_OUTPUTS = {
    "embeddings": "song_embeddings_audio.parquet",
    "scaler_model": "scaler_model",
    "pca_model": "pca_model",
    "encoder_metadata": ENCODER_METADATA_NAME,
}
C1_SUCCESS_MARKERS = {
    "embeddings": ("_SUCCESS",),
    "scaler_model": ("data/_SUCCESS", "metadata/_SUCCESS"),
    "pca_model": ("data/_SUCCESS", "metadata/_SUCCESS"),
    "encoder_metadata": (),
}


@dataclass(frozen=True, slots=True)
class EncoderContract:
    run_id: str
    row_count: int
    selected_k: int
    fitted_k: int

    @property
    def canonical_dimension(self) -> bool:
        return (
            self.selected_k == CANONICAL_EMBEDDING_DIMENSION
            and self.fitted_k == CANONICAL_EMBEDDING_DIMENSION
        )


def _encoder_dimensions(metadata: dict[str, Any]) -> EncoderContract:
    required = {
        "run_id",
        "shared_audio_contract_version",
        "c1_feature_version",
        "embedding_format",
        "row_count",
        "target_variance",
        "fixed_k",
        "selected_k",
        "max_components",
        "explained_variance",
        "cumulative_explained_variance",
    }
    missing = sorted(required - set(metadata))
    _require(not missing, f"C1 encoder metadata missing keys: {missing}")
    _require(
        metadata["shared_audio_contract_version"] == CONTRACT_VERSION,
        "wrong audio contract",
    )
    _require(int(metadata["c1_feature_version"]) == 2, "wrong C1 feature version")
    _require(metadata["embedding_format"] == "array<float32>", "wrong embedding format")
    contract = EncoderContract(
        str(metadata["run_id"]),
        int(metadata["row_count"]),
        int(metadata["selected_k"]),
        int(metadata["max_components"]),
    )
    _require(bool(contract.run_id), "C1 encoder run_id must be non-empty")
    _require(contract.row_count > 0, "C1 metadata row count must be positive")
    _require(contract.selected_k > 0, "C1 selected PCA dimension must be positive")
    _require(
        contract.fitted_k >= contract.selected_k,
        "C1 selected dimension exceeds fitted PCA dimension",
    )
    return contract


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


def load_prepared_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Prepared parent manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    _require(isinstance(manifest, dict), "Prepared parent manifest must be an object")
    _require(manifest.get("artifact_type") == "prepared_tables", "wrong parent artifact type")
    _require(manifest.get("artifact_version") == "v2", "wrong parent artifact version")
    _require(manifest.get("status") == "valid", "Prepared parent manifest is not valid")
    _require(manifest.get("validation", {}).get("passed") is True, "Prepared validation failed")

    config = manifest.get("config", {})
    expected_config = {
        "shared_audio_contract_version": CONTRACT_VERSION,
        "shared_audio_feature_count": 628,
        "merlin_audio_feature_count": 552,
        "merlin_raw_feature_count": 563,
    }
    for key, expected in expected_config.items():
        _require(config.get(key) == expected, f"Prepared parent {key} mismatch")

    contracts = [
        item
        for item in manifest.get("inputs", [])
        if item.get("name") == "audio_feature_contract"
    ]
    _require(len(contracts) == 1, "Prepared parent must bind one audio feature contract")
    contract = contracts[0]
    _require(
        contract.get("contract_version") == CONTRACT_VERSION,
        "parent audio contract mismatch",
    )
    _require(
        contract.get("feature_order_sha256") == feature_order_sha256(SHARED_FEATURE_COLUMNS),
        "parent audio feature order hash mismatch",
    )

    outputs = [
        item
        for item in manifest.get("outputs", [])
        if item.get("name") == "song_audio_features"
    ]
    _require(len(outputs) == 1, "Prepared parent must describe one audio output")
    _require(
        outputs[0].get("columns") == list(PREPARED_AUDIO_COLUMNS),
        "Prepared audio column order mismatch",
    )
    _require(
        manifest.get("statistics", {}).get("output_row_counts", {}).get(
            "song_audio_features_raw"
        )
        == 1_000_000,
        "Prepared audio row count mismatch",
    )
    return manifest, sha256_path(path)


def parent_lineage(manifest: dict[str, Any], path: Path, digest: str) -> dict[str, Any]:
    code = manifest.get("code", {})
    return {
        "artifact_type": manifest["artifact_type"],
        "artifact_version": manifest["artifact_version"],
        "manifest_contract_version": manifest.get("manifest_contract_version"),
        "path": str(path.resolve()),
        "run_id": manifest.get("run_id"),
        "sha256": digest,
        "producer_commit": code.get("commit"),
        "producer_dirty": code.get("dirty"),
        "shared_audio_contract_version": manifest["config"][
            "shared_audio_contract_version"
        ],
    }
