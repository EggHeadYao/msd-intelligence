from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from columns import PREPARED_AUDIO_COLUMNS
from shared_contract import CONTRACT_VERSION, SHARED_FEATURE_COLUMNS


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

    files = tuple(filter(None, git("ls-files", "--", pathspec).splitlines()))
    require(bool(files), f"no tracked source files match {pathspec}")
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with (repo_root / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_pathspec": pathspec,
        "source_sha256": digest.hexdigest(),
    }


def load_prepared_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Prepared parent manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    require(isinstance(manifest, dict), "Prepared parent manifest must be an object")
    require(manifest.get("artifact_type") == "prepared_tables", "wrong parent artifact type")
    require(manifest.get("artifact_version") == "v2", "wrong parent artifact version")
    require(manifest.get("status") == "valid", "Prepared parent manifest is not valid")
    require(manifest.get("validation", {}).get("passed") is True, "Prepared validation failed")

    config = manifest.get("config", {})
    expected_config = {
        "shared_audio_contract_version": CONTRACT_VERSION,
        "shared_audio_feature_count": 628,
        "merlin_audio_feature_count": 552,
        "merlin_raw_feature_count": 563,
    }
    for key, expected in expected_config.items():
        require(config.get(key) == expected, f"Prepared parent {key} mismatch")

    contracts = [item for item in manifest.get("inputs", []) if item.get("name") == "audio_feature_contract"]
    require(len(contracts) == 1, "Prepared parent must bind one audio feature contract")
    contract = contracts[0]
    require(contract.get("contract_version") == CONTRACT_VERSION, "parent audio contract mismatch")
    require(
        contract.get("feature_order_sha256") == feature_order_sha256(SHARED_FEATURE_COLUMNS),
        "parent audio feature order hash mismatch",
    )

    outputs = [item for item in manifest.get("outputs", []) if item.get("name") == "song_audio_features"]
    require(len(outputs) == 1, "Prepared parent must describe one audio output")
    require(outputs[0].get("columns") == list(PREPARED_AUDIO_COLUMNS), "Prepared audio column order mismatch")
    require(
        manifest.get("statistics", {}).get("output_row_counts", {}).get("song_audio_features_raw") == 1_000_000,
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
        "shared_audio_contract_version": manifest["config"]["shared_audio_contract_version"],
    }
