"""Build the typed C2 vocabulary and canonical paired adjacency tables."""

from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.types import BinaryType

from merlin.embedding.graph.config import ADJACENCY_NAMES, DIRECTED_EDGES, EDGE_SCHEMA

VOCAB_VERSION = "c2_typed_vocab_v1"
GRAPH_COLUMNS: tuple[str, ...] = (
    "src_type",
    "src_id",
    "dst_type",
    "dst_id",
    "directed",
    "edge_type",
)


def _local_path(path: str) -> Path:
    """Convert a plain path or file URI to a local Path."""
    parsed = urlparse(path)
    if parsed.scheme not in {"", "file"}:
        msg = f"C2 local persistence requires a local path, got: {path}"
        raise ValueError(msg)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(path)


def encode_typed_key(node_type: str, raw_id: str) -> str:
    """Encode a typed node key without delimiter collision."""
    return json.dumps(
        [node_type, raw_id],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def decode_typed_key(key: str) -> tuple[str, str]:
    """Decode and validate a serialized typed node key."""
    value: Any = json.loads(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"Invalid typed node key: {key!r}")
    return value[0], value[1]


def persist_vocabulary(index_dir: str, output_dir: str) -> str:
    """Copy the vocabulary next to the durable walk output."""
    source = _local_path(index_dir) / "vocab.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing C2 vocabulary: {source}")

    destination_dir = _local_path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "vocab.json"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination)


def validate_graph_edges(edges: DataFrame) -> None:
    """Fail early when the prepared graph is not the canonical A3 graph."""
    actual_columns = tuple(edges.columns)
    if actual_columns != GRAPH_COLUMNS:
        raise ValueError(
            "C2 graph schema mismatch: "
            f"expected={GRAPH_COLUMNS}, actual={actual_columns}",
        )

    expected_types: dict[str, type[T.DataType]] = {
        "src_type": T.StringType,
        "src_id": T.StringType,
        "dst_type": T.StringType,
        "dst_id": T.StringType,
        "directed": T.BooleanType,
        "edge_type": T.StringType,
    }
    for field in edges.schema.fields:
        if not isinstance(field.dataType, expected_types[field.name]):
            raise ValueError(
                f"C2 graph type mismatch for {field.name}: {field.dataType}",
            )

    actual_edge_types = {
        row["edge_type"] for row in edges.select("edge_type").distinct().collect()
    }
    expected_edge_types = set(EDGE_SCHEMA)
    if actual_edge_types != expected_edge_types:
        raise ValueError(
            "C2 edge types mismatch: "
            f"expected={sorted(expected_edge_types)}, "
            f"actual={sorted(actual_edge_types)}",
        )

    invalid = (
        F.col("src_type").isNull()
        | (F.col("src_type") == "")
        | F.col("src_id").isNull()
        | (F.col("src_id") == "")
        | F.col("dst_type").isNull()
        | (F.col("dst_type") == "")
        | F.col("dst_id").isNull()
        | (F.col("dst_id") == "")
        | F.col("directed").isNull()
        | F.col("edge_type").isNull()
        | (F.col("edge_type") == "")
    )
    endpoint_rule = F.lit(False)
    for edge_type, (src_type, dst_type) in EDGE_SCHEMA.items():
        endpoint_rule = endpoint_rule | (
            (F.col("edge_type") == edge_type)
            & (
                (F.col("src_type") != src_type)
                | (F.col("dst_type") != dst_type)
                | (F.col("directed") != (edge_type in DIRECTED_EDGES))
            )
        )
    if edges.where(invalid | endpoint_rule).limit(1).count():
        raise ValueError("C2 graph contains invalid IDs, endpoint types, or directions")


def build_node_vocabulary(
    spark: SparkSession,
    input_path: str,
) -> tuple[dict[str, int], dict[int, str], dict[int, str]]:
    """Build a deterministic integer vocabulary over typed graph nodes."""
    edges = spark.read.parquet(input_path)
    validate_graph_edges(edges)

    src_nodes = edges.select(
        F.col("src_type").alias("node_type"),
        F.col("src_id").alias("raw_id"),
    )
    dst_nodes = edges.select(
        F.col("dst_type").alias("node_type"),
        F.col("dst_id").alias("raw_id"),
    )
    rows = (
        src_nodes.unionByName(dst_nodes)
        .distinct()
        .sort("node_type", "raw_id")
        .collect()
    )

    node_to_int: dict[str, int] = {}
    int_to_node: dict[int, str] = {}
    int_to_type: dict[int, str] = {}
    for node_int, row in enumerate(rows):
        node_type = str(row["node_type"])
        raw_id = str(row["raw_id"])
        node_to_int[encode_typed_key(node_type, raw_id)] = node_int
        int_to_node[node_int] = raw_id
        int_to_type[node_int] = node_type
    return node_to_int, int_to_node, int_to_type


def _pairs_to_binary(
    pairs: list[Row],
    vocab: dict[str, int],
) -> tuple[bytes, bytes]:
    """Encode sorted neighbor/weight structs as aligned binary arrays."""
    neighbor_ids: list[int] = []
    weights: list[float] = []
    for pair in pairs:
        key = encode_typed_key(
            str(pair["neighbor_type"]),
            str(pair["neighbor_raw_id"]),
        )
        if key not in vocab:
            raise ValueError(f"Adjacency neighbor is missing from vocabulary: {key}")
        weight = float(pair["weight"])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Invalid adjacency weight for {key}: {weight}")
        neighbor_ids.append(vocab[key])
        weights.append(weight)

    return (
        struct.pack(f"<{len(neighbor_ids)}i", *neighbor_ids),
        struct.pack(f"<{len(weights)}f", *weights),
    )


def _save_adjacency(
    frame: DataFrame,
    output_dir: str,
    output_name: str,
    vocab_bc: Any,
    *,
    group_type_col: str,
    group_id_col: str,
    neighbor_type_col: str,
    neighbor_id_col: str,
    weight_col: str | None = None,
) -> int:
    """Aggregate sorted neighbor/weight pairs and write one adjacency table."""
    if output_name not in ADJACENCY_NAMES:
        raise ValueError(f"Unknown adjacency output: {output_name}")

    weight = F.col(weight_col).cast("double") if weight_col else F.lit(1.0)
    pairs = F.sort_array(
        F.collect_list(
            F.struct(
                F.col(neighbor_type_col).alias("neighbor_type"),
                F.col(neighbor_id_col).alias("neighbor_raw_id"),
                weight.alias("weight"),
            ),
        ),
    )
    grouped = frame.groupBy(group_type_col, group_id_col).agg(
        pairs.alias("neighbor_pairs"),
    )

    binary_schema = T.StructType(
        (
            T.StructField("neighbor_ids", T.BinaryType(), False),
            T.StructField("weights", T.BinaryType(), False),
        ),
    )
    encode_pairs = F.udf(
        lambda values: _pairs_to_binary(values, vocab_bc.value),
        binary_schema,
    )
    encode_node = F.udf(
        lambda node_type, raw_id: vocab_bc.value[
            encode_typed_key(str(node_type), str(raw_id))
        ],
        T.IntegerType(),
    )

    encoded = grouped.withColumn("encoded", encode_pairs("neighbor_pairs"))
    output = encoded.select(
        encode_node(group_type_col, group_id_col).alias("node_id"),
        F.col("encoded.neighbor_ids").alias("neighbor_ids"),
        F.col("encoded.weights").alias("weights"),
    )
    output_path = f"{output_dir}/{output_name}.parquet"
    output.write.mode("errorifexists").parquet(output_path)
    count = frame.sparkSession.read.parquet(output_path).count()
    print(f"  {output_name}: {count} nodes")
    return count


def _strs_to_int_binary(strs: list, vocab: dict[str, int]) -> bytes:
    """Convert a list of string node IDs to int32 numpy binary."""
    return np.array(
        [vocab[s] for s in strs if s in vocab], dtype=np.int32,
    ).tobytes()


def _floats_to_binary(vals: list) -> bytes:
    """Convert a list of floats to float32 numpy binary."""
    return np.array(vals, dtype=np.float32).tobytes()


def save_adjacency_parquet(
    edges: DataFrame,
    output_dir: str,
    vocab_bc,
    specs: list[tuple[str, str, str, str]],
) -> None:
    """Build adjacency via DataFrame groupBy and write directly to Parquet.

    Executors write Parquet files directly -- no data collected to
    driver, avoiding maxResultSize / driver OOM on 100M+ edges.

    Args:
        edges: graph_edges DataFrame (all edge types).
        output_dir: directory for output Parquet files.
        vocab_bc: broadcast variable containing node_to_int dict.
        specs: list of (edge_type, group_col, value_col, output_name).
    """
    to_bin_int_udf = F.udf(
        lambda strs: _strs_to_int_binary(strs, vocab_bc.value), BinaryType(),
    )
    to_bin_udf = F.udf(_floats_to_binary, BinaryType())

    for et, group_col, value_col, out_name in specs:
        part: DataFrame = edges.filter(F.col("edge_type") == et)

        grouped: DataFrame = part.groupBy(group_col).agg(
            F.collect_list(value_col).alias("neighbor_strs"),
            F.collect_list("weight").alias("weights_list"),
        )

        out: DataFrame = grouped.select(
            F.col(group_col).alias("node_str"),
            to_bin_int_udf(F.col("neighbor_strs")).alias("neighbor_ids"),
            to_bin_udf(F.col("weights_list")).alias("weights"),
        )

        out_path: str = f"{output_dir}/{out_name}.parquet"
        out.write.mode("overwrite").parquet(out_path)

        cnt: int = out.count()
        print(f"  {out_name}: {cnt} nodes")


def _build_p3_edges(edges: DataFrame) -> tuple[DataFrame, float]:
    """Attach smoothed capped-IDF to eligible artist-term edges."""
    active_artists = (
        edges.where(F.col("edge_type") == "track_artist")
        .select(F.col("dst_id").alias("artist_id"))
        .distinct()
    )
    artist_count = active_artists.count()
    if artist_count == 0:
        raise ValueError("Cannot build P3 adjacency without active artists")

    artist_terms = (
        edges.where(F.col("edge_type") == "artist_term")
        .join(active_artists, F.col("src_id") == F.col("artist_id"), "inner")
        .drop("artist_id")
    )
    term_stats = (
        artist_terms.groupBy("dst_id")
        .agg(F.countDistinct("src_id").alias("artist_df"))
        .where(F.col("artist_df") >= 2)
        .withColumn(
            "idf",
            F.log(
                (F.lit(float(artist_count)) + F.lit(1.0))
                / (F.col("artist_df").cast("double") + F.lit(1.0)),
            )
            + F.lit(1.0),
        )
    )
    quantiles = term_stats.approxQuantile("idf", [0.99], 0.0)
    if not quantiles:
        raise ValueError("No artist term connects at least two active artists")
    idf_cap = float(quantiles[0])
    weighted = (
        artist_terms.join(term_stats.select("dst_id", "idf"), "dst_id", "inner")
        .withColumn("p3_weight", F.least(F.col("idf"), F.lit(idf_cap)))
        .drop("idf")
    )
    return weighted, idf_cap


def _write_vocabulary(
    output_dir: str,
    node_to_int: dict[str, int],
    int_to_node: dict[int, str],
    int_to_type: dict[int, str],
) -> Path:
    output_path = _local_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vocab_path = output_path / "vocab.json"
    payload = {
        "vocab_version": VOCAB_VERSION,
        "node_to_int": node_to_int,
        "int_to_node": {str(key): value for key, value in int_to_node.items()},
        "int_to_type": {str(key): value for key, value in int_to_type.items()},
    }
    vocab_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="ascii",
    )
    return vocab_path


def _save_uniform_adjacencies(
    edges: DataFrame,
    output_dir: str,
    vocab_bc: Any,
) -> None:
    specs = (
        (
            "track_artist",
            "track_to_artist",
            ("src_type", "src_id"),
            ("dst_type", "dst_id"),
        ),
        (
            "track_artist",
            "artist_to_tracks",
            ("dst_type", "dst_id"),
            ("src_type", "src_id"),
        ),
        (
            "track_release",
            "track_to_release",
            ("src_type", "src_id"),
            ("dst_type", "dst_id"),
        ),
        (
            "track_release",
            "release_to_tracks",
            ("dst_type", "dst_id"),
            ("src_type", "src_id"),
        ),
        (
            "artist_similarity",
            "artist_to_similar_artists",
            ("src_type", "src_id"),
            ("dst_type", "dst_id"),
        ),
    )
    for edge_type, output_name, group, neighbor in specs:
        _save_adjacency(
            edges.where(F.col("edge_type") == edge_type),
            output_dir,
            output_name,
            vocab_bc,
            group_type_col=group[0],
            group_id_col=group[1],
            neighbor_type_col=neighbor[0],
            neighbor_id_col=neighbor[1],
        )


def _save_p3_adjacencies(
    edges: DataFrame,
    output_dir: str,
    vocab_bc: Any,
) -> float:
    """Write both directions of the eligible artist-term relation."""
    weighted_edges, idf_cap = _build_p3_edges(edges)
    _save_adjacency(
        weighted_edges,
        output_dir,
        "artist_to_terms",
        vocab_bc,
        group_type_col="src_type",
        group_id_col="src_id",
        neighbor_type_col="dst_type",
        neighbor_id_col="dst_id",
        weight_col="p3_weight",
    )
    _save_adjacency(
        weighted_edges,
        output_dir,
        "term_to_artists",
        vocab_bc,
        group_type_col="dst_type",
        group_id_col="dst_id",
        neighbor_type_col="src_type",
        neighbor_id_col="src_id",
    )
    return idf_cap


def load_and_build_index(
    spark: SparkSession,
    input_path: str,
    output_dir: str,
) -> tuple[
    dict[str, int],
    dict[int, str],
    dict[int, str],
]:
    """Build adjacency index and save as intermediate Parquet files.

    Writes one Parquet per edge-type grouping to *output_dir*.
    Returns vocab mappings for downstream walk generation.

    Returns:
        node_to_int: string node ID -> integer index.
        int_to_node: integer index -> string node ID.
        int_to_type: integer index -> node type string.
    """
    node_to_int, int_to_node, int_to_type = build_node_vocabulary(
        spark, input_path,
    )

    edges: DataFrame = spark.read.parquet(input_path)
    bc_vocab = spark.sparkContext.broadcast(node_to_int)

    # --- forward adjacency specs ---
    fwd_specs: list[tuple[str, str, str, str]] = [
        ("song_artist", "src_id", "dst_id", "fwd_song_artist"),
        ("song_album", "src_id", "dst_id", "fwd_song_album"),
        ("song_tag", "src_id", "dst_id", "fwd_song_tag"),
        ("song_similar_artist", "src_id", "dst_id", "fwd_song_similar_artist"),
        ("song_year", "src_id", "dst_id", "fwd_song_year"),
    ]
    save_adjacency_parquet(edges, output_dir, bc_vocab, fwd_specs)

    # --- reverse adjacency specs ---
    rev_specs: list[tuple[str, str, str, str]] = [
        ("song_artist", "dst_id", "src_id", "rev_song_artist"),
        ("song_album", "dst_id", "src_id", "rev_song_album"),
        ("song_tag", "dst_id", "src_id", "rev_song_tag"),
        ("song_year", "dst_id", "src_id", "rev_song_year"),
        ("artist_tag", "src_id", "dst_id", "rev_artist_tag_fwd"),
        ("artist_tag", "dst_id", "src_id", "rev_artist_tag_rev"),
        ("artist_similarity", "src_id", "dst_id", "rev_artist_similarity"),
    ]
    save_adjacency_parquet(edges, output_dir, bc_vocab, rev_specs)

    # Persist vocab so walk generation can load it without Spark.
    import json
    vocab_path: Path = _local_path(output_dir) / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({
            "node_to_int": node_to_int,
            "int_to_node": {str(k): v for k, v in int_to_node.items()},
            "int_to_type": {str(k): v for k, v in int_to_type.items()},
        }, f)
    print(f"Vocab saved to {vocab_path}")

    print(
        f"Index saved to {output_dir}: "
        f"{len(node_to_int)} nodes, "
        f"{len(fwd_specs) + len(rev_specs)} adjacency files",
    )

    return node_to_int, int_to_node, int_to_type
