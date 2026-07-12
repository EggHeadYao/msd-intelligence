"""C2 adjacency index builder: node vocabulary + forward/reverse adjacency."""

from __future__ import annotations

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
