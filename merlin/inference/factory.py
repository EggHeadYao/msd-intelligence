"""Production construction of the canonical four-source C3 pipeline."""

from __future__ import annotations

from .artifacts import InferenceArtifacts
from .candidate_policy import CANONICAL_RETRIEVER_LIMITS
from .features_v2 import PairSignalLookups, RankerV2FeatureComputer
from .pipeline import MerlinPipeline
from .retrieval import BfsRetriever, TagRetriever, VectorRetriever


def build_inference_pipeline(artifacts: InferenceArtifacts) -> MerlinPipeline:
    """Construct a canonical pipeline only from a validated artifact bundle."""
    same_song = artifacts.same_song
    audio = VectorRetriever("audio", artifacts.audio_index.search, same_song=same_song)
    graph = VectorRetriever("graph", artifacts.graph_index.search, same_song=same_song)
    bfs = BfsRetriever.from_parquet(
        str(artifacts.paths.songs_metadata),
        str(artifacts.paths.graph_edges),
        same_song=same_song,
    )
    tag = TagRetriever.from_parquet(
        str(artifacts.paths.songs_metadata),
        str(artifacts.paths.graph_edges),
        tag_idf_path=str(artifacts.paths.tag_idf),
        same_song=same_song,
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
