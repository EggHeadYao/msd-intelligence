"""C2 adjacency index builder: node vocabulary + forward/reverse adjacency."""

from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType


def build_node_vocabulary(
    spark: SparkSession, input_path: str
) -> tuple[dict[str, int], dict[int, str], dict[int, str]]:
    """Build string<->int mappings for all unique graph nodes.

    Reads graph_edges.parquet and assigns a sequential integer ID to
    every unique node (song, artist, album, tag, year).  Nodes are
    sorted by their string ID so the mapping is deterministic.

    Returns:
        node_to_int: string node ID -> integer index
        int_to_node: integer index -> string node ID
        int_to_type: integer index -> node type (song|artist|album|tag|year)
    """
    edges: DataFrame = spark.read.parquet(input_path)

    src_nodes: DataFrame = edges.select(
        F.col("src_id").alias("node_id"),
        F.col("src_type").alias("node_type"),
    ).distinct()

    dst_nodes: DataFrame = edges.select(
        F.col("dst_id").alias("node_id"),
        F.col("dst_type").alias("node_type"),
    ).distinct()

    all_nodes: DataFrame = (
        src_nodes.unionByName(dst_nodes)
        .distinct()
        .sort("node_id")
    )

    rows: list = all_nodes.collect()

    node_to_int: dict[str, int] = {}
    int_to_node: dict[int, str] = {}
    int_to_type: dict[int, str] = {}

    for idx, row in enumerate(rows):
        node_id: str = row["node_id"]
        node_type: str = row["node_type"]
        node_to_int[node_id] = idx
        int_to_node[idx] = node_id
        int_to_type[idx] = node_type

    return node_to_int, int_to_node, int_to_type


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
    local_dir: str = output_dir.replace("file://", "")
    vocab_path: str = f"{local_dir}/vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
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
