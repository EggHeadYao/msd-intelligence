"""C2 adjacency index builder: node vocabulary + forward/reverse adjacency."""

from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


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


def build_forward_adjacency(
    edges: DataFrame, node_to_int: dict[str, int]
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Build per-song forward adjacency from graph edges.

    Processes each edge_type partition separately to avoid a single
    huge shuffle.  Returns a dict mapping song_int_id -> {edge_type:
    (neighbor_int_ids, weights)} where both arrays are numpy int32 /
    float32 contiguous arrays.

    Only forward direction is stored here.  Reverse (intermediate
    node -> song) is built separately by build_reverse_adjacency.
    """
    fwd_edge_types: list[str] = [
        "song_artist",
        "song_album",
        "song_tag",
        "song_similar_artist",
        "song_year",
    ]

    fwd_adj: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for et in fwd_edge_types:
        part: DataFrame = edges.filter(F.col("edge_type") == et).select(
            F.col("src_id"),
            F.col("dst_id"),
            F.col("weight"),
        )

        grouped: DataFrame = part.groupBy("src_id").agg(
            F.collect_list("dst_id").alias("neighbor_strs"),
            F.collect_list("weight").alias("weights_list"),
        )

        for row in grouped.collect():
            src_str: str = row["src_id"]
            if src_str not in node_to_int:
                continue
            src_int: int = node_to_int[src_str]

            neighbor_ints: np.ndarray = np.array(
                [node_to_int[n] for n in row["neighbor_strs"]],
                dtype=np.int32,
            )
            weights_arr: np.ndarray = np.array(
                row["weights_list"], dtype=np.float32
            )

            fwd_adj.setdefault(src_int, {})[et] = (neighbor_ints, weights_arr)

    return fwd_adj
