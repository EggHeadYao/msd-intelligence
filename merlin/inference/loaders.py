"""Production artifact loaders with frozen MERLIN contract defaults."""

from __future__ import annotations

from pathlib import Path

from .artifact_paths import AUDIO_INDEX_PATH, AUDIO_MANIFEST_PATH, AUDIO_MAPPING_PATH
from .faiss_index import FaissTrackIndex


AUDIO_CONTRACT_VERSION = "shared_audio_628_v1"


def load_audio_index(
    index_path: str | Path = AUDIO_INDEX_PATH,
    mapping_path: str | Path = AUDIO_MAPPING_PATH,
    manifest_path: str | Path = AUDIO_MANIFEST_PATH,
) -> FaissTrackIndex:
    """Load the final C1 v2 index and reject any historical lineage."""
    return FaissTrackIndex.from_files(
        index_path,
        mapping_path,
        manifest_path,
        expected_contract_key="shared_audio_contract_version",
        expected_contract=AUDIO_CONTRACT_VERSION,
    )
