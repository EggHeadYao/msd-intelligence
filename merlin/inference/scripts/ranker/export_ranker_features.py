"""CLI to export pre-fill Ranker-v2 features for labeled pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...artifact_paths import InferenceArtifactPaths
from ...catalog_data import load_catalog_context
from ...faiss_index import FaissTrackIndex
from ...features_v2 import PairSignalLookups, RankerV2FeatureComputer
from ...loaders import load_audio_index
from ...ranker_features import export_raw_pair_features
from ...training_pairs import load_training_pair_manifest
from ...validation_groups import load_validation_group_manifest
from ...recall_factory import build_canonical_retrievers
from ...retrieval import TagRetriever
from ...tag_data import load_tag_idf


def parse_args() -> argparse.Namespace:
    defaults = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-kind", choices=("training", "validation"), default="training")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--pairs-manifest", type=Path)
    parser.add_argument("--validation-positives", type=Path, default=defaults.validation_group_positives)
    parser.add_argument("--validation-thresholds", type=Path, default=defaults.validation_group_thresholds)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--stage", choices=("tuning", "final_retrain"), default="tuning")
    parser.add_argument("--graph-contract-key", default=GRAPH_CONTRACT_KEY)
    parser.add_argument("--graph-contract-version", default=GRAPH_CONTRACT_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = InferenceArtifactPaths()
    pairs = args.pairs or (
        paths.training_pairs if args.pair_kind == "training" else paths.validation_pairs
    )
    pairs_manifest = args.pairs_manifest or (
        paths.training_pairs_manifest
        if args.pair_kind == "training"
        else paths.validation_groups_manifest
    )
    output = args.output or (
        paths.raw_pair_features
        if args.pair_kind == "training"
        else paths.validation_raw_features
    )
    output_manifest = args.manifest or (
        paths.raw_pair_features_manifest
        if args.pair_kind == "training"
        else paths.validation_raw_features_manifest
    )
    if args.pair_kind == "training":
        load_training_pair_manifest(
            pairs_manifest,
            pairs,
            expected_scope=args.scope,
            expected_stage=args.stage,
        )
    else:
        if args.stage != "tuning":
            raise ValueError("validation features are only defined for tuning")
        load_validation_group_manifest(
            pairs_manifest,
            thresholds_path=args.validation_thresholds,
            positives_path=args.validation_positives,
            validation_pairs_path=pairs,
            expected_scope=args.scope,
        )
    audio = load_audio_index()
    graph = FaissTrackIndex.from_files(
        paths.graph_index,
        paths.graph_mapping,
        paths.graph_manifest,
        paths.graph_encoder_metadata,
        expected_space="graph",
        expected_contract_key=args.graph_contract_key,
        expected_contract=args.graph_contract_version,
    )
    catalog = load_catalog_context(
        paths.songs_metadata,
        paths.graph_edges,
        include_ranker_metadata=True,
    )
    same_song = catalog.same_song
    tag = TagRetriever.from_data(
        catalog.tag_data,
        idf_values=load_tag_idf(
            paths.tag_idf,
            expected_graph_edges_path=paths.graph_edges,
        ),
        same_song=same_song,
    )
    _audio, _graph, bfs, tag = build_canonical_retrievers(
        audio, graph, paths, same_song, tag
    )
    computer = RankerV2FeatureComputer(
        tracks=catalog.ranker_tracks,
        signals=PairSignalLookups(
            audio=audio.similarity,
            graph=graph.similarity,
            bfs=bfs.pair_score,
            tags=tag.pair_score,
            audio_batch=audio.similarities,
            graph_batch=graph.similarities,
        ),
    )
    manifest = export_raw_pair_features(
        pairs,
        computer,
        output,
        output_manifest,
        parent_paths={
            f"{args.pair_kind}_pairs": pairs,
            f"{args.pair_kind}_pairs_manifest": pairs_manifest,
            "audio_index_manifest": paths.audio_manifest,
            "graph_index_manifest": paths.graph_manifest,
            "tag_idf": paths.tag_idf,
            "songs_metadata": paths.songs_metadata,
            "graph_edges": paths.graph_edges,
        },
        scope=args.scope,
        pair_kind=args.pair_kind,
        stage=args.stage,
    )
    print(
        "ranker_features_ready "
        f"scope={args.scope} kind={args.pair_kind} "
        f"rows={manifest['row_count']} output={output}",
    )


if __name__ == "__main__":
    main()
