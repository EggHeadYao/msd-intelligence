"""Validated artifact bundle used to construct production inference."""

from __future__ import annotations

from dataclasses import dataclass

from ..artifact_lineage import sha256_path
from ..artifact_paths import InferenceArtifactPaths
from ..candidate_policy import load_candidate_policy
from ..faiss_index import FaissTrackIndex
from ..features_v2 import FeatureFillValues, TrackMetadataV2, load_track_metadata_v2
from ..loaders import load_audio_index
from ..ranker import LogisticRanker
from ..ranker_lineage import load_ranker_bundle
from ..track_identity import SameSongFilter, load_same_song_filter


@dataclass(frozen=True, slots=True)
class InferenceArtifacts:
    paths: InferenceArtifactPaths
    audio_index: FaissTrackIndex
    graph_index: FaissTrackIndex
    tracks: dict[str, TrackMetadataV2]
    same_song: SameSongFilter
    ranker: LogisticRanker
    fills: FeatureFillValues
    candidate_policy: dict[str, object]


def load_inference_artifacts(
    paths: InferenceArtifactPaths = InferenceArtifactPaths(),
    *,
    graph_contract_key: str,
    graph_contract_version: str,
) -> InferenceArtifacts:
    """Load the lineage-bound inputs shared by the full C3 pipeline."""
    policy = load_candidate_policy(paths.candidate_policy)
    audio = load_audio_index(
        paths.audio_index,
        paths.audio_mapping,
        paths.audio_manifest,
        paths.audio_encoder_metadata,
    )
    graph = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=graph_contract_key,
        expected_contract=graph_contract_version,
    )
    parents = {
        "audio_index_manifest": sha256_path(paths.audio_manifest),
        "graph_index_manifest": sha256_path(paths.graph_manifest),
        "candidate_policy_manifest": sha256_path(paths.candidate_policy),
        "tag_idf": sha256_path(paths.tag_idf),
    }
    ranker = load_ranker_bundle(
        paths.ranker_schema,
        paths.ranker_scaler,
        paths.ranker_coefficients,
        paths.ranker_training_manifest,
        expected_parent_hashes=parents,
    )
    return InferenceArtifacts(
        paths=paths,
        audio_index=audio,
        graph_index=graph,
        tracks=load_track_metadata_v2(paths.songs_metadata),
        same_song=load_same_song_filter(paths.songs_metadata),
        ranker=ranker,
        fills=FeatureFillValues.from_artifact(paths.ranker_scaler, ranker.feature_order),
        candidate_policy=policy,
    )
