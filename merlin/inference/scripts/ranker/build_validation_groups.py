"""Build frozen Audio/Relation/Mixed groups for Set B or Set C."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from merlin.inference.artifacts.paths import InferenceArtifactPaths
from merlin.inference.artifacts.integrity import sha256_path
from merlin.inference.recall.pool import load_candidate_pool_manifest
from merlin.inference.evaluation.protocol import load_set_c_protocol
from merlin.inference.artifacts.io import write_json_atomic
from merlin.inference.scripts.support.scratch import prepare_scratch_root
from merlin.inference.training.split import load_split_manifest
from merlin.inference.data.tags import load_tag_idf
from merlin.inference.training.validation_groups import (
    VALIDATION_GROUP_SEED,
    VALIDATION_QUERY_GROUPS,
    build_nested_validation_pairs,
    collect_normalized_vector_matrix,
    estimate_validation_scratch_gb,
    load_selected_artist_terms,
    sampled_pair_cosine_quantiles,
    write_audio_threshold_pairs_numpy,
    write_high_tag_pairs_sparse,
    write_validation_group_manifest,
)


MAX_THRESHOLD_PAIRS = 1_000_000


def parse_args() -> argparse.Namespace:
    defaults = InferenceArtifactPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-audio", type=Path, default=defaults.raw_audio_features)
    parser.add_argument("--prepared-manifest", type=Path, default=defaults.prepared_manifest)
    parser.add_argument("--songs-metadata", type=Path, default=defaults.songs_metadata)
    parser.add_argument("--audio-root", type=Path, default=defaults.audio_encoder_metadata.parent)
    parser.add_argument("--graph-edges", type=Path, default=defaults.graph_edges)
    parser.add_argument("--tag-idf", type=Path, default=defaults.tag_idf)
    parser.add_argument("--weak-thresholds", type=Path, default=defaults.weak_label_thresholds)
    parser.add_argument("--split-assignments", type=Path, default=defaults.split_assignments)
    parser.add_argument("--split-manifest", type=Path, default=defaults.split_manifest)
    parser.add_argument("--candidate-pool", type=Path, default=defaults.set_b_candidate_pool)
    parser.add_argument(
        "--candidate-pool-manifest",
        type=Path,
        default=defaults.set_b_candidate_pool_manifest,
    )
    parser.add_argument("--thresholds", type=Path, default=defaults.validation_group_thresholds)
    parser.add_argument("--positives", type=Path, default=defaults.validation_group_positives)
    parser.add_argument("--validation-pairs", type=Path, default=defaults.validation_pairs)
    parser.add_argument("--manifest", type=Path, default=defaults.validation_groups_manifest)
    parser.add_argument("--max-threshold-pairs", type=int, default=MAX_THRESHOLD_PAIRS)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--audio-pair-engine", choices=("numpy", "spark"), default="numpy")
    parser.add_argument("--audio-block-size", type=int, default=256)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--min-free-gb", type=float)
    return parser.parse_args()


def _uri(path: Path) -> str:
    # Hadoop's local filesystem does not decode an escaped Hive partition '='.
    return path.resolve().as_uri().replace("%3D", "=")


def _require_new_outputs(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"validation-group outputs already exist: {existing}")


def main() -> None:
    args = parse_args()
    if args.max_threshold_pairs <= 0 or args.max_threshold_pairs > MAX_THRESHOLD_PAIRS:
        raise ValueError("max-threshold-pairs must be in [1, 1000000]")
    if args.shuffle_partitions <= 0 or args.audio_block_size <= 0:
        raise ValueError("shuffle partitions and audio block size must be positive")
    _require_new_outputs((args.thresholds, args.positives, args.validation_pairs, args.manifest))
    split_manifest = load_split_manifest(args.split_manifest, args.split_assignments)
    if args.scope == "formal" and split_manifest.get("scope") != "formal":
        raise ValueError("formal validation groups require a formal split artifact")
    candidate_manifest = load_candidate_pool_manifest(
        args.candidate_pool_manifest,
        args.candidate_pool,
        expected_scope=args.scope,
    )
    with args.weak_thresholds.open("r", encoding="utf-8") as stream:
        weak_thresholds = json.load(stream)
    if weak_thresholds.get("artifact_type") != "weak_label_thresholds":
        raise ValueError("weak-label threshold artifact type mismatch")
    tag_positive_threshold = float(weak_thresholds["tag_tfidf_cosine_p90"])
    if not math.isfinite(tag_positive_threshold):
        raise ValueError("tag positive threshold must be finite")
    tag_idf = load_tag_idf(args.tag_idf, expected_graph_edges_path=args.graph_edges)

    encoder_metadata_path = args.audio_root / "audio_encoder_metadata.json"
    c1_manifest_path = args.audio_root / "c1_manifest.json"
    scaler_model_path = args.audio_root / "scaler_model"
    with encoder_metadata_path.open("r", encoding="utf-8") as stream:
        encoder_metadata = json.load(stream)
    with c1_manifest_path.open("r", encoding="utf-8") as stream:
        c1_manifest = json.load(stream)
    if c1_manifest.get("artifact_type") != "c1_audio_encoder" or c1_manifest.get("status") != "valid":
        raise ValueError("C1 manifest is not a valid audio encoder")
    if encoder_metadata.get("run_id") != c1_manifest.get("run_id"):
        raise ValueError("C1 encoder metadata run does not match its manifest")
    if encoder_metadata.get("model_ready_schema_version") != "c1_model_ready_v2":
        raise ValueError("unsupported C1 model-ready schema")
    feature_columns = tuple(str(value) for value in encoder_metadata["feature_columns"])
    split_counts = split_manifest.get("track_counts", {})
    candidate_totals = candidate_manifest.get("totals", {})
    projected_scratch_gb = 0.0
    if args.scope == "formal":
        projected_scratch_gb = estimate_validation_scratch_gb(
            set_a_tracks=int(split_counts.get("set_a", 0)),
            set_b_tracks=int(split_counts.get("set_b", 0)),
            feature_dimension=len(feature_columns),
            unique_candidates=int(candidate_totals.get("unique_candidates", 0)),
            max_sample_pairs=args.max_threshold_pairs,
        )
    prepare_scratch_root(
        args.validation_pairs.parent,
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_scratch_gb,
    )
    scratch_root = prepare_scratch_root(
        args.scratch_root or args.validation_pairs.parent / ".c3-scratch",
        scope=args.scope,
        min_free_gb=args.min_free_gb,
        projected_gb=projected_scratch_gb,
    )

    from pyspark import StorageLevel
    from pyspark.ml.feature import StandardScalerModel, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    audio_module = Path(__file__).resolve().parents[3] / "embedding" / "audio"
    sys.path.insert(0, str(audio_module))
    try:
        from columns import TRACK_ID_COLUMN
        from preprocess import apply_frozen_preprocess
        from train_pca import FEATURES_COLUMN, SCALED_FEATURES_COLUMN
    finally:
        sys.path.pop(0)

    def array_dot(left, right):
        products = F.zip_with(left, right, lambda x, y: x.cast("double") * y.cast("double"))
        return F.aggregate(products, F.lit(0.0), lambda total, value: total + value)

    def cosine(left, right, left_norm, right_norm):
        return array_dot(left, right) / (left_norm * right_norm)

    def not_same_song(prefix_left: str, prefix_right: str):
        left = F.col(f"{prefix_left}_song_id")
        right = F.col(f"{prefix_right}_song_id")
        return ~(left.isNotNull() & right.isNotNull() & (left == right))

    spark_local_temporary = TemporaryDirectory(prefix="merlin-c3-spark-", dir=scratch_root)
    spark = (
        SparkSession.builder.appName("MerlinBuildSetBValidationGroups")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.local.dir", spark_local_temporary.name)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    cached = []
    audio_pair_temporary = None
    tag_pair_temporary = None
    try:
        def release(frame) -> None:
            frame.unpersist(blocking=True)
            cached[:] = [item for item in cached if item is not frame]

        def read_rows(path: Path):
            return (
                spark.read.parquet(_uri(path))
                if path.suffix == ".parquet"
                else spark.read.json(_uri(path))
            )

        assignments = read_rows(args.split_assignments).select(
            F.col("track_id").cast("string"), F.col("split").cast("string")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(assignments)
        pool_queries = read_rows(args.candidate_pool).select(
            F.col("query_track_id").cast("string").alias("track_id")
        ).distinct().persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(pool_queries)
        invalid_pool_queries = (
            pool_queries.join(assignments, "track_id", "left")
            .where(F.col("split").isNull() | (F.col("split") != "set_b"))
            .limit(1)
            .count()
        )
        if invalid_pool_queries:
            raise ValueError("candidate pool contains a query outside Set B")

        selected_ids = assignments.where(F.col("split").isin("set_a", "set_b")).select(
            TRACK_ID_COLUMN
        )
        raw = spark.read.parquet(_uri(args.raw_audio)).join(
            F.broadcast(selected_ids), TRACK_ID_COLUMN, "inner"
        )
        processed = apply_frozen_preprocess(raw, feature_columns, encoder_metadata["preprocess"])
        assembled = VectorAssembler(
            inputCols=list(feature_columns), outputCol=FEATURES_COLUMN
        ).transform(processed)
        scaler = StandardScalerModel.load(_uri(scaler_model_path))
        scaler_checks = (
            (
                tuple(float(value) for value in scaler.mean),
                tuple(float(value) for value in encoder_metadata["scaler_mean"]),
            ),
            (
                tuple(float(value) for value in scaler.std),
                tuple(float(value) for value in encoder_metadata["scaler_std"]),
            ),
        )
        if any(
            len(actual) != len(expected)
            or max(
                (abs(left - right) for left, right in zip(actual, expected)),
                default=0.0,
            )
            > 1e-12
            for actual, expected in scaler_checks
        ):
            raise ValueError("C1 scaler model does not match encoder metadata")
        pre_pca = scaler.transform(assembled).select(
            TRACK_ID_COLUMN,
            vector_to_array(F.col(SCALED_FEATURES_COLUMN)).alias("pre_pca_vector"),
        ).withColumn(
            "pre_pca_norm",
            F.sqrt(array_dot(F.col("pre_pca_vector"), F.col("pre_pca_vector"))),
        )
        metadata = spark.read.parquet(_uri(args.songs_metadata)).select(
            TRACK_ID_COLUMN,
            F.col("song_id").cast("string"),
            F.col("artist_id").cast("string"),
            F.col("release_7digitalid").cast("long").alias("release_id"),
        )
        vectors = pre_pca.join(metadata, TRACK_ID_COLUMN, "inner").join(
            assignments, TRACK_ID_COLUMN, "inner"
        ).localCheckpoint(eager=True)
        cached.append(vectors)

        set_a = vectors.where(F.col("split") == "set_a").drop("split")
        set_a_count = set_a.count()
        if set_a_count < 2:
            raise ValueError("Set A has too few C1 vectors for threshold fitting")
        slots = max(1, math.ceil(2 * args.max_threshold_pairs / set_a_count) + 4)
        indexed_source = set_a.select(
            TRACK_ID_COLUMN, "song_id", "artist_id", "pre_pca_norm"
        )
        indexed_schema = indexed_source.schema.add("sample_row_id", "long", nullable=False)
        indexed = spark.createDataFrame(
            indexed_source.orderBy(TRACK_ID_COLUMN).rdd.zipWithIndex().map(
                lambda row_and_index: (*row_and_index[0], int(row_and_index[1]))
            ),
            schema=indexed_schema,
        ).persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(indexed)
        queries = indexed.select(
            F.col(TRACK_ID_COLUMN).alias("q_track_id"),
            F.col("song_id").alias("q_song_id"),
            F.col("artist_id").alias("q_artist_id"),
            F.col("pre_pca_norm").alias("q_norm"),
        ).withColumn("sample_slot", F.explode(F.sequence(F.lit(0), F.lit(slots - 1))))
        queries = queries.withColumn(
            "candidate_row_id",
            F.pmod(
                F.xxhash64("q_track_id", "sample_slot", F.lit(VALIDATION_GROUP_SEED)),
                F.lit(set_a_count),
            ),
        )
        candidates = indexed.select(
            F.col("sample_row_id").alias("candidate_row_id"),
            F.col(TRACK_ID_COLUMN).alias("c_track_id"),
            F.col("song_id").alias("c_song_id"),
            F.col("artist_id").alias("c_artist_id"),
            F.col("pre_pca_norm").alias("c_norm"),
        )
        threshold_pair_ids = (
            queries.join(candidates, "candidate_row_id", "inner")
            .where(F.col("q_track_id") < F.col("c_track_id"))
            .where(not_same_song("q", "c"))
            .where(
                F.col("q_artist_id").isNotNull()
                & F.col("c_artist_id").isNotNull()
                & (F.length("q_artist_id") > 0)
                & (F.length("c_artist_id") > 0)
                & (F.col("q_artist_id") != F.col("c_artist_id"))
            )
            .where(
                F.col("q_norm").isNotNull()
                & F.col("c_norm").isNotNull()
                & ~F.isnan("q_norm")
                & ~F.isnan("c_norm")
                & (F.col("q_norm") > 0.0)
                & (F.col("c_norm") > 0.0)
            )
            .dropDuplicates(["q_track_id", "c_track_id"])
            .select(
                "q_track_id",
                "c_track_id",
                F.xxhash64(
                    "q_track_id", "c_track_id", F.lit(VALIDATION_GROUP_SEED)
                ).alias("sample_hash"),
            )
            .orderBy("sample_hash")
            .limit(args.max_threshold_pairs)
            .select("q_track_id", "c_track_id")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(threshold_pair_ids)
        threshold_sample_count = threshold_pair_ids.count()
        if threshold_sample_count == 0:
            raise ValueError("Set-A pre-PCA threshold sample is empty")
        vector_positions, normalized_vectors = collect_normalized_vector_matrix(
            set_a.select(
                TRACK_ID_COLUMN, "pre_pca_vector", "pre_pca_norm"
            ).toLocalIterator(),
            capacity=set_a_count,
            dimension=len(feature_columns),
        )
        threshold_sample_count, acoustic_p50, acoustic_p90 = sampled_pair_cosine_quantiles(
            threshold_pair_ids.toLocalIterator(),
            vector_positions,
            normalized_vectors,
            expected_pairs=threshold_sample_count,
        )
        del normalized_vectors, vector_positions
        if not math.isfinite(acoustic_p50) or not math.isfinite(acoustic_p90):
            raise ValueError("Set-A pre-PCA thresholds are not finite")
        if acoustic_p90 < acoustic_p50:
            raise ValueError("Set-A pre-PCA thresholds are not monotonic")
        release(threshold_pair_ids)
        release(indexed)

        set_b = vectors.where(F.col("split") == "set_b").drop("split").persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(set_b)
        if set_b.count() == 0:
            raise ValueError("split has no Set-B C1 vectors")
        release(vectors)
        release(assignments)
        query_b = set_b.join(F.broadcast(pool_queries), TRACK_ID_COLUMN, "inner")
        if query_b.limit(1).count() == 0:
            raise ValueError("candidate pool has no Set-B query with a C1 vector")

        b_artists = set_b.where(
            F.col("artist_id").isNotNull() & (F.length("artist_id") > 0)
        ).select("artist_id").distinct()
        artist_terms = load_selected_artist_terms(
            args.graph_edges,
            (row["artist_id"] for row in b_artists.collect()),
            tag_idf,
        )
        if not artist_terms:
            raise ValueError("Set B has no artist-term vectors")
        tag_pair_temporary = TemporaryDirectory(
            prefix="merlin-setb-tag-pairs-", dir=scratch_root
        )
        high_tag_pairs_path = Path(tag_pair_temporary.name) / "pairs.parquet"
        write_high_tag_pairs_sparse(
            artist_terms,
            tag_idf,
            high_tag_pairs_path,
            threshold=tag_positive_threshold,
            block_size=args.audio_block_size,
        )
        high_tag_pairs = spark.read.parquet(_uri(high_tag_pairs_path))
        tagged_artists = spark.createDataFrame(
            ((artist_id,) for artist_id in sorted(artist_terms)),
            ("artist_id",),
        )

        def q_columns(frame):
            return frame.select(
                F.col(TRACK_ID_COLUMN).alias("query_track_id"),
                F.col("song_id").alias("q_song_id"),
                F.col("artist_id").alias("q_artist_id"),
                F.col("release_id").alias("q_release_id"),
                F.col("pre_pca_vector").alias("q_vector"),
                F.col("pre_pca_norm").alias("q_norm"),
            )

        def c_columns(frame):
            return frame.select(
                F.col(TRACK_ID_COLUMN).alias("candidate_track_id"),
                F.col("song_id").alias("c_song_id"),
                F.col("artist_id").alias("c_artist_id"),
                F.col("release_id").alias("c_release_id"),
                F.col("pre_pca_vector").alias("c_vector"),
                F.col("pre_pca_norm").alias("c_norm"),
            )

        q_tracks = q_columns(query_b)
        c_tracks = c_columns(set_b)
        valid_q_audio = (
            q_tracks.where(
                F.col("q_artist_id").isNotNull()
                & (F.length("q_artist_id") > 0)
                & F.col("q_release_id").isNotNull()
                & (F.col("q_release_id") > 0)
                & (F.col("q_norm") > 0.0)
            )
            .join(
                F.broadcast(tagged_artists.select(F.col("artist_id").alias("q_artist_id"))),
                "q_artist_id",
                "inner",
            )
        )
        valid_c_audio = (
            c_tracks.where(
                F.col("c_artist_id").isNotNull()
                & (F.length("c_artist_id") > 0)
                & F.col("c_release_id").isNotNull()
                & (F.col("c_release_id") > 0)
                & (F.col("c_norm") > 0.0)
            )
            .join(
                F.broadcast(tagged_artists.select(F.col("artist_id").alias("c_artist_id"))),
                "c_artist_id",
                "inner",
            )
        )
        if args.audio_pair_engine == "numpy":
            audio_pair_temporary = TemporaryDirectory(
                prefix="merlin-setb-audio-pairs-", dir=scratch_root
            )
            raw_audio_pairs_path = Path(audio_pair_temporary.name) / "pairs.parquet"
            write_audio_threshold_pairs_numpy(
                [row.asDict(recursive=True) for row in valid_q_audio.collect()],
                [row.asDict(recursive=True) for row in valid_c_audio.collect()],
                raw_audio_pairs_path,
                threshold=acoustic_p90,
                block_size=args.audio_block_size,
            )
            raw_audio_pairs = spark.read.parquet(_uri(raw_audio_pairs_path))
        else:
            raw_audio_pairs = (
                valid_q_audio.crossJoin(valid_c_audio)
                .where(F.col("query_track_id") != F.col("candidate_track_id"))
                .where(not_same_song("q", "c"))
                .where(
                    (F.col("q_artist_id") != F.col("c_artist_id"))
                    & (F.col("q_release_id") != F.col("c_release_id"))
                )
                .withColumn(
                    "pre_pca_cosine",
                    cosine(
                        F.col("q_vector"),
                        F.col("c_vector"),
                        F.col("q_norm"),
                        F.col("c_norm"),
                    ),
                )
                .where(F.col("pre_pca_cosine") >= F.lit(acoustic_p90))
                .select("query_track_id", "candidate_track_id", "q_artist_id", "c_artist_id")
            )
        audio_pairs = (
            raw_audio_pairs.join(
                high_tag_pairs, ["q_artist_id", "c_artist_id"], "left_anti"
            )
            .select(
                "query_track_id",
                "candidate_track_id",
                F.lit("audio_dominant").alias("query_group"),
                F.array(F.lit("pre_pca_audio")).alias("positive_sources"),
            ).persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(audio_pairs)
        audio_pairs.count()
        if audio_pair_temporary is not None:
            audio_pair_temporary.cleanup()
            audio_pair_temporary = None

        q_meta = q_tracks.drop("q_vector")
        c_meta = c_tracks.drop("c_vector")
        same_artist = q_meta.join(c_meta, F.col("q_artist_id") == F.col("c_artist_id"), "inner").select(
            "query_track_id", "candidate_track_id", F.lit("same_artist").alias("relation_source")
        )
        same_release = q_meta.where(F.col("q_release_id").isNotNull() & (F.col("q_release_id") > 0)).join(
            c_meta.where(F.col("c_release_id").isNotNull() & (F.col("c_release_id") > 0)),
            F.col("q_release_id") == F.col("c_release_id"),
            "inner",
        ).select("query_track_id", "candidate_track_id", F.lit("same_release").alias("relation_source"))
        directed_edges = spark.read.parquet(
            _uri(args.graph_edges / "edge_type=artist_similarity")
        ).select(
            F.col("src_id").cast("string").alias("q_artist_id"),
            F.col("dst_id").cast("string").alias("c_artist_id"),
        ).distinct()
        directed = q_meta.join(directed_edges, "q_artist_id", "inner").join(
            c_meta, "c_artist_id", "inner"
        ).select("query_track_id", "candidate_track_id", F.lit("directed_artist_similarity").alias("relation_source"))
        high_tag = high_tag_pairs.join(
            q_meta, "q_artist_id", "inner"
        ).join(c_meta, "c_artist_id", "inner").select(
            "query_track_id", "candidate_track_id", F.lit("high_artist_term").alias("relation_source")
        )
        relation_sources = (
            same_artist.unionByName(same_release).unionByName(directed).unionByName(high_tag)
            .where(F.col("query_track_id") != F.col("candidate_track_id"))
        )
        relation_pairs = (
            relation_sources.join(F.broadcast(q_tracks), "query_track_id", "inner")
            .join(F.broadcast(c_tracks), "candidate_track_id", "inner")
            .where(not_same_song("q", "c"))
            .where((F.col("q_norm") > 0.0) & (F.col("c_norm") > 0.0))
            .withColumn(
                "pre_pca_cosine",
                cosine(F.col("q_vector"), F.col("c_vector"), F.col("q_norm"), F.col("c_norm")),
            )
            .where(F.col("pre_pca_cosine") < F.lit(acoustic_p50))
            .groupBy("query_track_id", "candidate_track_id")
            .agg(F.sort_array(F.collect_set("relation_source")).alias("positive_sources"))
            .select(
                "query_track_id",
                "candidate_track_id",
                F.lit("relation_dominant").alias("query_group"),
                "positive_sources",
            ).persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(relation_pairs)
        relation_pairs.count()
        tag_pair_temporary.cleanup()
        tag_pair_temporary = None
        release(set_b)

        counts = audio_pairs.groupBy("query_track_id").count().withColumnRenamed("count", "audio_count").join(
            relation_pairs.groupBy("query_track_id").count().withColumnRenamed("count", "relation_count"),
            "query_track_id",
            "inner",
        ).withColumn("balanced_count", F.least("audio_count", "relation_count")).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        cached.append(counts)
        audio_window = Window.partitionBy("query_track_id").orderBy(
            F.xxhash64("query_track_id", "candidate_track_id", F.lit("audio"), F.lit(VALIDATION_GROUP_SEED)),
            "candidate_track_id",
        )
        relation_window = Window.partitionBy("query_track_id").orderBy(
            F.xxhash64("query_track_id", "candidate_track_id", F.lit("relation"), F.lit(VALIDATION_GROUP_SEED)),
            "candidate_track_id",
        )
        mixed_audio = audio_pairs.withColumn("side_rank", F.row_number().over(audio_window)).join(
            counts.select("query_track_id", "balanced_count"), "query_track_id"
        ).where(F.col("side_rank") <= F.col("balanced_count")).select(
            "query_track_id",
            "candidate_track_id",
            F.lit("mixed").alias("query_group"),
            F.array(F.lit("audio_dominant_side")).alias("positive_sources"),
        )
        mixed_relation = relation_pairs.withColumn("side_rank", F.row_number().over(relation_window)).join(
            counts.select("query_track_id", "balanced_count"), "query_track_id"
        ).where(F.col("side_rank") <= F.col("balanced_count")).select(
            "query_track_id",
            "candidate_track_id",
            F.lit("mixed").alias("query_group"),
            F.array(F.lit("relation_dominant_side")).alias("positive_sources"),
        )
        positives = (
            audio_pairs.unionByName(relation_pairs).unionByName(mixed_audio).unionByName(mixed_relation)
            .dropDuplicates(["query_track_id", "candidate_track_id", "query_group"])
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(positives)
        positives.count()
        release(audio_pairs)
        release(relation_pairs)
        release(counts)
        pool = read_rows(args.candidate_pool).select(
            F.col("query_track_id").cast("string"), "candidates"
        )
        candidate_rows = pool.select(
            "query_track_id", F.explode("candidates").alias("candidate")
        ).select(
            "query_track_id",
            F.col("candidate.track_id").cast("string").alias("candidate_track_id"),
            F.col("candidate.recall_sources").alias("recall_sources"),
        )
        validation_pairs, eligible, recalled_positives = build_nested_validation_pairs(
            candidate_rows, positives
        )
        eligible = eligible.persist(StorageLevel.MEMORY_AND_DISK)
        recalled_positives = recalled_positives.persist(StorageLevel.MEMORY_AND_DISK)
        cached.extend((eligible, recalled_positives))
        validation_pairs = validation_pairs.persist(StorageLevel.MEMORY_AND_DISK)
        cached.append(validation_pairs)
        layout_counts = validation_pairs.agg(
            F.count("*").alias("pair_count"),
            F.sum(F.size("validation_groups")).alias("group_row_count"),
        ).first()
        validation_pair_count = int(layout_counts["pair_count"])
        validation_group_row_count = int(layout_counts["group_row_count"] or 0)
        if validation_pair_count == 0:
            raise ValueError("Set-B validation pairs are empty")
        missing_candidate_queries = eligible.select("query_track_id").distinct().join(
            pool_queries.select(F.col("track_id").alias("query_track_id")),
            "query_track_id",
            "left_anti",
        ).limit(1).count()
        if missing_candidate_queries:
            raise ValueError("an eligible Set-B validation query has no canonical candidates")

        positive_stats = {
            row["query_group"]: row.asDict()
            for row in positives.groupBy("query_group").agg(
                F.count("*").alias("positive_count"),
                F.countDistinct("query_track_id").alias("eligible_query_count"),
            ).collect()
        }
        hit_stats = {
            row["query_group"]: row.asDict()
            for row in recalled_positives.groupBy("query_group").agg(
                F.count("*").alias("candidate_hits"),
                F.countDistinct("query_track_id").alias("covered_queries"),
            ).collect()
        }
        group_stats = {}
        for group in VALIDATION_QUERY_GROUPS:
            positive_count = int(positive_stats.get(group, {}).get("positive_count", 0))
            eligible_query_count = int(
                positive_stats.get(group, {}).get("eligible_query_count", 0)
            )
            candidate_hits = int(hit_stats.get(group, {}).get("candidate_hits", 0))
            covered_queries = int(hit_stats.get(group, {}).get("covered_queries", 0))
            group_stats[group] = {
                "eligible_query_count": eligible_query_count,
                "positive_count": positive_count,
                "candidate_positive_hits": candidate_hits,
                "candidate_recall": candidate_hits / positive_count if positive_count else 0.0,
                "zero_coverage_query_count": eligible_query_count - covered_queries,
            }
        if args.scope == "formal" and any(
            int(group_stats[group]["eligible_query_count"]) == 0 for group in VALIDATION_QUERY_GROUPS
        ):
            raise ValueError("formal Set-B validation is missing an eligible query group")
        release(recalled_positives)
        release(eligible)
        release(pool_queries)

        threshold_payload = {
            "artifact_type": "set_b_validation_thresholds",
            "artifact_version": "merlin_validation_groups_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fit_split": "set_a",
            "seed": VALIDATION_GROUP_SEED,
            "sample_method": "deterministic_hash_sampled_cross_artist_pairs",
            "quantile_method": "numpy_linear_exact",
            "max_sample_pairs": args.max_threshold_pairs,
            "sampled_cross_artist_pairs": threshold_sample_count,
            "pre_pca_acoustic_cosine_p50": acoustic_p50,
            "pre_pca_acoustic_cosine_p90": acoustic_p90,
            "tag_positive_threshold": tag_positive_threshold,
            "tag_positive_threshold_source": str(args.weak_thresholds),
            "audio_pair_engine": args.audio_pair_engine,
            "audio_block_size": args.audio_block_size,
        }
        write_json_atomic(threshold_payload, args.thresholds)
        positives.write.mode("errorifexists").parquet(_uri(args.positives))
        release(positives)
        validation_pairs.repartition(
            args.shuffle_partitions, "query_track_id"
        ).sortWithinPartitions("query_track_id", "candidate_track_id").write.mode(
            "errorifexists"
        ).parquet(_uri(args.validation_pairs))
        release(validation_pairs)
        manifest = write_validation_group_manifest(
            args.manifest,
            thresholds_path=args.thresholds,
            positives_path=args.positives,
            validation_pairs_path=args.validation_pairs,
            parent_paths={
                "prepared_manifest": args.prepared_manifest,
                "c1_manifest": c1_manifest_path,
                "audio_encoder_metadata": encoder_metadata_path,
                "audio_scaler_model": scaler_model_path,
                "split_manifest": args.split_manifest,
                "split_assignments": args.split_assignments,
                "candidate_pool_manifest": args.candidate_pool_manifest,
                "candidate_pool": args.candidate_pool,
                "weak_label_thresholds": args.weak_thresholds,
                "tag_idf": args.tag_idf,
            },
            scope=args.scope,
            threshold_sample_count=threshold_sample_count,
            pair_count=validation_pair_count,
            group_row_count=validation_group_row_count,
            group_stats=group_stats,
        )
        print(
            "validation_groups_ready "
            f"scope={manifest['scope']} set_a_sample={threshold_sample_count} "
            f"candidate_queries={candidate_manifest['query_count']} output={args.validation_pairs}"
        )
    finally:
        for frame in reversed(cached):
            try:
                frame.unpersist(blocking=False)
            except Exception:
                pass
        spark.stop()
        if audio_pair_temporary is not None:
            audio_pair_temporary.cleanup()
        if tag_pair_temporary is not None:
            tag_pair_temporary.cleanup()
        spark_local_temporary.cleanup()


if __name__ == "__main__":
    main()
