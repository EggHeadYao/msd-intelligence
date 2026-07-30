"""CLI to export pre-fill Ranker features for labeled pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from merlin.embedding.graph.config import GRAPH_CONTRACT_KEY, GRAPH_CONTRACT_VERSION

from ...artifacts.integrity import artifact_size_bytes, sha256_path
from ...artifacts.paths import InferenceArtifactPaths
from ...data.catalog import load_catalog_context
from ...retrieval.faiss import FaissTrackIndex
from ...evaluation.protocol import load_development_protocol
from ...retrieval.faiss import load_audio_index
from ...ranking.features import (
    PairSignalLookups,
    RankerFeatureComputer,
    export_raw_pair_features,
)
from ...training.pairs import load_training_pair_manifest
from ...training.validation_groups import load_validation_group_manifest
from ...recall.factory import build_canonical_retrievers
from ...retrieval import TagRetriever
from ..support.scratch import prepare_scratch_root
from ...data.tags import load_tag_idf


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
    parser.add_argument(
        "--stage",
        choices=("tuning", "final_retrain", "development_evaluation"),
        default="tuning",
    )
    parser.add_argument("--development-protocol", type=Path)
    parser.add_argument("--graph-contract-key", default=GRAPH_CONTRACT_KEY)
    parser.add_argument("--graph-contract-version", default=GRAPH_CONTRACT_VERSION)
    parser.add_argument("--min-free-gb", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = InferenceArtifactPaths()
    is_development = args.stage == "development_evaluation"
    is_tuning = args.stage == "tuning"
    if is_development and args.pair_kind != "validation":
        raise ValueError("evaluation requires validation pairs")
    if is_development and args.development_protocol is None:
        raise ValueError("evaluation requires a development protocol")
    if is_development and args.validation_positives == paths.validation_group_positives:
        args.validation_positives = paths.development_positives
    if args.pair_kind == "training":
        default_pairs = (
            paths.tuning_training_pairs if is_tuning else paths.training_pairs
        )
        default_pairs_manifest = (
            paths.tuning_training_pairs_manifest
            if is_tuning
            else paths.training_pairs_manifest
        )
        default_output = (
            paths.tuning_raw_pair_features if is_tuning else paths.raw_pair_features
        )
        default_output_manifest = (
            paths.tuning_raw_pair_features_manifest
            if is_tuning
            else paths.raw_pair_features_manifest
        )
    else:
        default_pairs = (
            paths.development_validation_pairs
            if is_development
            else paths.validation_pairs
        )
        default_pairs_manifest = (
            paths.development_groups_manifest
            if is_development
            else paths.validation_groups_manifest
        )
        default_output = (
            paths.development_raw_features
            if is_development
            else paths.validation_raw_features
        )
        default_output_manifest = (
            paths.development_raw_features_manifest
            if is_development
            else paths.validation_raw_features_manifest
        )
    pairs = args.pairs or default_pairs
    pairs_manifest = args.pairs_manifest or default_pairs_manifest
    output = args.output or default_output
    output_manifest = args.manifest or default_output_manifest
    projected_gb = artifact_size_bytes(pairs) * 2 / (1024 ** 3)
    prepare_scratch_root(
        output.parent,
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_gb,
    )
    if args.pair_kind == "training":
        load_training_pair_manifest(
            pairs_manifest,
            pairs,
            expected_scope=args.scope,
            expected_stage=args.stage,
        )
    else:
        if args.stage not in {"tuning", "development_evaluation"}:
            raise ValueError(
                "validation features require tuning or development_evaluation"
            )
        if is_development:
            load_development_protocol(
                args.development_protocol,
                expected_scope=args.scope,
                expected_split="set_c",
                expected_parent_hashes={
                    "split_manifest": sha256_path(paths.split_manifest),
                    "split_assignments": sha256_path(paths.split_assignments),
                    "candidate_policy_manifest": sha256_path(paths.candidate_policy),
                    "validation_group_thresholds": sha256_path(
                        paths.validation_group_thresholds
                    ),
                    "ranker_training_manifest": sha256_path(
                        paths.ranker_training_manifest
                    ),
                    "no_hard_neg_training_manifest": sha256_path(
                        paths.no_hard_neg_training_manifest
                    ),
                    "audio_index_manifest": sha256_path(paths.audio_manifest),
                    "graph_index_manifest": sha256_path(paths.graph_manifest),
                    "tag_idf": sha256_path(paths.tag_idf),
                    "songs_metadata": sha256_path(paths.songs_metadata),
                    "graph_edges": sha256_path(paths.graph_edges),
                },
            )
        load_validation_group_manifest(
            pairs_manifest,
            thresholds_path=args.validation_thresholds,
            positives_path=args.validation_positives,
            validation_pairs_path=pairs,
            expected_scope=args.scope,
            expected_apply_split="set_c" if is_development else "set_b",
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
    computer = RankerFeatureComputer(
        tracks=catalog.ranker_tracks,
        signals=PairSignalLookups(
            audio=audio.similarity,
            graph=graph.similarity,
            bfs=bfs.pair_score,
            tags=tag.pair_score,
            audio_batch=audio.similarities,
            graph_batch=graph.similarities,
            bfs_batch=lambda query_id, candidate_ids: bfs.pair_scores(
                [(query_id, candidate_id) for candidate_id in candidate_ids]
            ),
            tags_batch=lambda query_id, candidate_ids: tag.pair_scores(
                [(query_id, candidate_id) for candidate_id in candidate_ids]
            ),
            audio_pairs=audio.pair_similarities,
            graph_pairs=graph.pair_similarities,
            bfs_pairs=bfs.pair_scores,
            tags_pairs=tag.pair_scores,
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
            **(
                {"evaluation_protocol": args.development_protocol}
                if is_development
                else {}
            ),
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
