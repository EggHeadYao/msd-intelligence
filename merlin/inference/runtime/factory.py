"""Production construction of the canonical four-source C3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ..artifacts.integrity import sha256_path
from ..artifacts.paths import InferenceArtifactPaths
from ..recall.policy import CANONICAL_RETRIEVER_LIMITS, load_candidate_policy
from ..data.catalog import SameSongFilter, load_same_song_filter
from ..retrieval.faiss import FaissTrackIndex, load_audio_index
from ..ranking.features import (
    FeatureFillValues,
    PairSignalLookups,
    RankerFeatureComputer,
    TrackMetadata,
    load_track_metadata,
)
from ..ranking.model import LogisticRanker, load_ranker_bundle
from ..recall.factory import build_canonical_retrievers
from .pipeline import MerlinPipeline


@dataclass(frozen=True, slots=True)
class InferenceArtifacts:
    paths: InferenceArtifactPaths
    audio_index: FaissTrackIndex
    graph_index: FaissTrackIndex
    tracks: dict[str, TrackMetadata]
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
        tracks=load_track_metadata(paths.songs_metadata),
        same_song=load_same_song_filter(paths.songs_metadata),
        ranker=ranker,
        fills=FeatureFillValues.from_artifact(paths.ranker_scaler, ranker.feature_order),
        candidate_policy=policy,
    )


def load_inference_pipeline(
    paths: InferenceArtifactPaths = InferenceArtifactPaths(),
    *,
    graph_contract_key: str,
    graph_contract_version: str,
) -> MerlinPipeline:
    artifacts = load_inference_artifacts(
        paths,
        graph_contract_key=graph_contract_key,
        graph_contract_version=graph_contract_version,
    )
    return build_inference_pipeline(artifacts)


def build_inference_pipeline(artifacts: InferenceArtifacts) -> MerlinPipeline:
    """Construct a canonical pipeline only from a validated artifact bundle."""
    audio, graph, bfs, tag = build_canonical_retrievers(
        artifacts.audio_index,
        artifacts.graph_index,
        artifacts.paths,
        artifacts.same_song,
    )
    signals = PairSignalLookups(
        audio=artifacts.audio_index.similarity,
        graph=artifacts.graph_index.similarity,
        bfs=bfs.pair_score,
        tags=tag.pair_score,
        audio_batch=artifacts.audio_index.similarities,
        graph_batch=artifacts.graph_index.similarities,
        bfs_batch=lambda query_id, candidate_ids: bfs.pair_scores(
            [(query_id, candidate_id) for candidate_id in candidate_ids]
        ),
        tags_batch=lambda query_id, candidate_ids: tag.pair_scores(
            [(query_id, candidate_id) for candidate_id in candidate_ids]
        ),
        audio_pairs=artifacts.audio_index.pair_similarities,
        graph_pairs=artifacts.graph_index.pair_similarities,
        bfs_pairs=bfs.pair_scores,
        tags_pairs=tag.pair_scores,
    )
    features = RankerFeatureComputer(
        tracks=artifacts.tracks,
        signals=signals,
        fills=artifacts.fills,
    )
    return MerlinPipeline(
        retrievers=(audio, graph, bfs, tag),
        retriever_limits=CANONICAL_RETRIEVER_LIMITS,
        feature_computer=features,
        ranker=artifacts.ranker,
        canonical=True,
    )
