"""Hash and fail-closed integrity validation for MERLIN artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FAISS_MANIFEST_VERSION = "merlin_faiss_index_v1"
FAISS_ARTIFACT_TYPE = "merlin_faiss_index"


def artifact_size_bytes(path: str | Path) -> int:
    """Return the total payload bytes for one file or artifact directory."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"artifact does not exist: {root}")
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory using relative names and file contents."""
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


def load_faiss_manifest(
    path: str | Path,
    *,
    index_path: str | Path,
    mapping_path: str | Path,
    encoder_metadata_path: str | Path,
    expected_space: str,
    expected_contract_key: str,
    expected_contract: str,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FAISS manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = {
        "artifact_type",
        "manifest_version",
        "embedding_space",
        expected_contract_key,
        "index_type",
        "metric",
        "dimension",
        "row_count",
        "index_file",
        "mapping_path",
        "index_sha256",
        "mapping_sha256",
        "encoder_metadata_sha256",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"FAISS manifest missing keys: {missing}")
    if manifest[expected_contract_key] != expected_contract:
        raise ValueError("FAISS manifest contract version mismatch")
    if manifest["artifact_type"] != FAISS_ARTIFACT_TYPE:
        raise ValueError("FAISS artifact type mismatch")
    if manifest["manifest_version"] != FAISS_MANIFEST_VERSION:
        raise ValueError("FAISS manifest version mismatch")
    if manifest["embedding_space"] != expected_space:
        raise ValueError("FAISS embedding space mismatch")
    if manifest["index_file"] != Path(index_path).name:
        raise ValueError("FAISS manifest index path mismatch")
    if manifest["mapping_path"] != Path(mapping_path).name:
        raise ValueError("FAISS manifest mapping path mismatch")
    if (
        manifest["index_type"] != "IndexFlatIP"
        or manifest["metric"] != "inner_product"
        or int(manifest["dimension"]) != 128
    ):
        raise ValueError("FAISS manifest must describe a 128D IndexFlatIP")
    if manifest["index_sha256"] != sha256_path(index_path):
        raise ValueError("FAISS index hash mismatch")
    if manifest["mapping_sha256"] != sha256_path(mapping_path):
        raise ValueError("FAISS mapping hash mismatch")
    if manifest["encoder_metadata_sha256"] != sha256_path(encoder_metadata_path):
        raise ValueError("FAISS encoder metadata hash mismatch")
    return manifest
