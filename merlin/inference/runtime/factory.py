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
    )
    features = RankerV2FeatureComputer(
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
