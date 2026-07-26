"""Export and validate pre-fill Ranker feature artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
import json
import math
from pathlib import Path
from typing import Iterator, Mapping

from ...artifacts.integrity import artifact_size_bytes, sha256_path
from ...artifacts.io import (
    parquet_rows,
    read_row_artifact,
    write_json_atomic,
    write_row_artifact,
)
from ...types import Candidate
from .compute import FEATURE_SCHEMA, RankerFeatureComputer


RAW_FEATURE_VERSION = "merlin_ranker_raw_features_v3"
SAMPLE_WEIGHT_COLUMN = "sample_weight"
RAW_BASE_FEATURES = (
    "cos_audio",
    "cos_graph",
    "has_graph",
    "bfs_score",
    "has_bfs",
    "tag_tfidf_cosine",
    "has_tags",
    "same_release",
    "has_release",
    "year_gap",
    "has_year",
)
FILL_FEATURES = (
    "cos_audio",
    "cos_graph",
    "bfs_score",
    "tag_tfidf_cosine",
    "year_gap",
)


def raw_feature_parquet_schema(pair_kind: str):
    if pair_kind not in {"training", "validation"}:
        raise ValueError("raw-feature pair kind must be training or validation")
    import pyarrow as pa

    fields = [
        pa.field("query_track_id", pa.string(), nullable=False),
        pa.field("candidate_track_id", pa.string(), nullable=False),
    ]
    if pair_kind == "training":
        fields.extend((
            pa.field("label", pa.int64(), nullable=False),
            pa.field(SAMPLE_WEIGHT_COLUMN, pa.float32(), nullable=False),
        ))
    else:
        fields.extend((
            pa.field("recall_sources", pa.list_(pa.string()), nullable=False),
            pa.field("validation_groups", pa.list_(pa.struct((
                pa.field("query_group", pa.string(), nullable=False),
                pa.field("label", pa.int64(), nullable=False),
                pa.field("eligible_positive_count", pa.int64(), nullable=False),
            ))), nullable=False),
        ))
    fields.extend(pa.field(name, pa.float32()) for name in RAW_BASE_FEATURES)
    return pa.schema(fields)


def materialize_raw_features(
    raw: Mapping[str, object],
    fill_values: Mapping[str, float],
) -> dict[str, float]:
    values = {
        name: (
            float(raw[name])
            if raw.get(name) is not None
            else float(fill_values[name])
        )
        for name in FILL_FEATURES
    }
    return {
        "cos_audio": values["cos_audio"],
        "cos_graph": values["cos_graph"],
        "has_graph": float(raw["has_graph"]),
        "bfs_score": values["bfs_score"],
        "has_bfs": float(raw["has_bfs"]),
        "tag_tfidf_cosine": values["tag_tfidf_cosine"],
        "has_tags": float(raw["has_tags"]),
        "same_release": float(raw["same_release"]),
        "has_release": float(raw["has_release"]),
        "year_gap": values["year_gap"],
        "has_year": float(raw["has_year"]),
        "audio_tag_interaction": values["cos_audio"]
        * values["tag_tfidf_cosine"],
        "graph_bfs_interaction": values["cos_graph"] * values["bfs_score"],
    }


def export_raw_pair_features(
    pair_path: str | Path,
    computer: RankerFeatureComputer,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    parent_paths: Mapping[str, str | Path],
    scope: str,
    pair_kind: str = "training",
    stage: str = "tuning",
) -> dict[str, object]:
    if scope not in {"formal", "smoke"}:
        raise ValueError("raw-feature scope must be formal or smoke")
    if pair_kind not in {"training", "validation"}:
        raise ValueError("raw-feature pair kind must be training or validation")
    if stage not in {"tuning", "final_retrain", "final_evaluation"}:
        raise ValueError(
            "raw-feature stage must be tuning, final_retrain, or final_evaluation"
        )
    if stage == "final_evaluation" and pair_kind != "validation":
        raise ValueError("evaluation features require validation pairs")
    counts: Counter[str] = Counter()
    loss_weight_sums: Counter[str] = Counter()

    def pair_rows() -> Iterator[Mapping[str, object]]:
        if pair_kind == "training":
            yield from read_row_artifact(pair_path)
            return
        for values in parquet_rows(
            pair_path,
            (
                "query_track_id",
                "candidate_track_id",
                "recall_sources",
                "validation_groups",
            ),
            engine="pyarrow",
        ):
            query_id, candidate_id, recall_sources, validation_groups = values
            yield {
                "query_track_id": query_id,
                "candidate_track_id": candidate_id,
                "recall_sources": list(recall_sources or ()),
                "validation_groups": list(validation_groups or ()),
            }

    def compute(query_id: str, pairs: list[Mapping[str, object]]):
        candidate_ids = [str(pair["candidate_track_id"]) for pair in pairs]
        compute_ids = getattr(computer, "compute_raw_ids", None)
        raw_rows = (
            compute_ids(query_id, candidate_ids)
            if compute_ids is not None
            else [
                computer.compute_raw(query_id, Candidate(candidate_id))
                for candidate_id in candidate_ids
            ]
        )
        return zip(pairs, candidate_ids, raw_rows, strict=True)

    def training_rows() -> Iterator[dict[str, object]]:
        grouped_rows = groupby(pair_rows(), key=lambda pair: str(pair["query_track_id"]))
        for query_id, grouped in grouped_rows:
            pairs = list(grouped)
            for pair, candidate_id, raw in compute(query_id, pairs):
                label = int(pair["label"])
                if label not in {0, 1}:
                    raise ValueError("training pair label must be binary")
                sample_weight = float(pair[SAMPLE_WEIGHT_COLUMN])
                if not math.isfinite(sample_weight) or sample_weight < 0.0:
                    raise ValueError("training pair sample weight must be finite and non-negative")
                counts["rows"] += 1
                counts[f"label_{label}"] += 1
                source = str(pair.get("negative_source") or "positive")
                loss_weight_sums[source] += sample_weight
                yield {
                    "query_track_id": query_id,
                    "candidate_track_id": candidate_id,
                    "label": label,
                    SAMPLE_WEIGHT_COLUMN: sample_weight,
                    **raw,
                }

    def validation_rows() -> Iterator[dict[str, object]]:
        grouped_rows = groupby(pair_rows(), key=lambda pair: str(pair["query_track_id"]))
        seen_queries: set[str] = set()
        for query_id, grouped in grouped_rows:
            if query_id in seen_queries:
                raise ValueError("validation pairs are not clustered by query")
            seen_queries.add(query_id)
            pairs = list(grouped)
            for pair, candidate_id, raw in compute(query_id, pairs):
                groups = []
                for group in pair["validation_groups"]:
                    label = int(group["label"])
                    if label not in {0, 1}:
                        raise ValueError("validation pair label must be binary")
                    groups.append({
                        "query_group": str(group["query_group"]),
                        "label": label,
                        "eligible_positive_count": int(group["eligible_positive_count"]),
                    })
                    counts["group_rows"] += 1
                    counts[f"label_{label}"] += 1
                counts["rows"] += 1
                yield {
                    "query_track_id": query_id,
                    "candidate_track_id": candidate_id,
                    "recall_sources": list(pair.get("recall_sources", [])),
                    "validation_groups": groups,
                    **raw,
                }

    def rows() -> Iterator[dict[str, object]]:
        if pair_kind == "training":
            yield from training_rows()
        else:
            yield from validation_rows()

    output = Path(output_path)
    parquet_schema = None
    if output.suffix == ".parquet":
        parquet_schema = raw_feature_parquet_schema(pair_kind)
    row_count = write_row_artifact(rows(), output, parquet_schema=parquet_schema)
    if row_count == 0:
        raise ValueError("raw feature artifact must not be empty")
    manifest = {
        "artifact_type": "ranker_raw_pair_features",
        "artifact_version": RAW_FEATURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "pair_kind": pair_kind,
        "stage": stage,
        "row_layout": (
            "one_row_per_training_pair"
            if pair_kind == "training"
            else "one_feature_row_per_pair_with_nested_validation_groups"
        ),
        "feature_schema_version": FEATURE_SCHEMA,
        "raw_feature_order": list(RAW_BASE_FEATURES),
        "row_count": row_count,
        "counts": dict(sorted(counts.items())),
        **({
            "loss_weighting": {
                "column": SAMPLE_WEIGHT_COLUMN,
                "positive_weight_sum": float(loss_weight_sums["positive"]),
                "candidate_aware_weight_sum": float(
                    loss_weight_sums["candidate_aware"]
                ),
                "random_weight_sum": float(loss_weight_sums["random"]),
            }
        } if pair_kind == "training" else {}),
        "output_file": output.name,
        "storage_format": "parquet" if output.suffix == ".parquet" else "jsonl_gzip",
        "output_sha256": sha256_path(output),
        "output_size_bytes": artifact_size_bytes(output),
        "parent_hashes": {
            name: sha256_path(path) for name, path in sorted(parent_paths.items())
        },
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def load_raw_feature_manifest(
    manifest_path: str | Path,
    feature_path: str | Path,
    *,
    expected_scope: str,
    expected_pair_kind: str | None = None,
    expected_stage: str | None = None,
) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_type") != "ranker_raw_pair_features":
        raise ValueError("raw-feature artifact type mismatch")
    if manifest.get("artifact_version") != RAW_FEATURE_VERSION:
        raise ValueError("raw-feature artifact version mismatch")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA:
        raise ValueError("raw-feature schema version mismatch")
    if manifest.get("raw_feature_order") != list(RAW_BASE_FEATURES):
        raise ValueError("raw-feature order mismatch")
    if manifest.get("scope") != expected_scope:
        raise ValueError("raw-feature scope mismatch")
    if expected_pair_kind is not None and manifest.get("pair_kind") != expected_pair_kind:
        raise ValueError("raw-feature pair kind mismatch")
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        raise ValueError("raw-feature stage mismatch")
    features = Path(feature_path)
    if manifest.get("output_file") != features.name:
        raise ValueError("raw-feature output path mismatch")
    if manifest.get("output_sha256") != sha256_path(features):
        raise ValueError("raw-feature output hash mismatch")
    return manifest
