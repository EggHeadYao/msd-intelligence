from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from contract import (
    ARTIST_ID,
    AUDIT_COLUMNS,
    AUDIO_FEATURE_ORDER_SHA256,
    BINARY_FEATURE_COLUMNS,
    DERIVED_SCALAR_COLUMNS,
    EXPECTED_GROUP_COUNTS,
    EXPECTED_LABELED_TRACKS,
    EXPECTED_TRACKS,
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_PREDICTOR_COLUMNS,
    GLOBAL_SCALAR_COLUMNS,
    SPLIT,
    T90_COLUMNS,
    TRACK_ID,
    YEAR,
    YEAR_EXCLUDED_COLUMNS,
    column_missing_rule,
    column_source,
    column_unit,
    full_predictor_columns,
    load_audio_contract,
    order_sha256,
    ordered_feature_groups,
    year_shared_columns,
)

FADE_TOLERANCE_SECONDS = 0.001
SPLIT_VALUES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate year-prediction feature views.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("parquets/year_prediction/features"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("parquets/year_prediction/raw/audio_features"),
    )
    parser.add_argument(
        "--scalar",
        type=Path,
        default=Path("parquets/year_prediction/raw/songs_scalar.parquet"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("parquets/year_prediction/dataset"),
    )
    parser.add_argument("--hdf5-root", type=Path)
    parser.add_argument("--hdf5-samples", type=int, default=16)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def spark_path(path: str | Path) -> str:
    text = str(path)
    return text if "://" in text else Path(text).resolve().as_uri()


def audio_paths(path: Path) -> list[str]:
    return [spark_path(item) for item in sorted(path.glob("features_*.parquet"))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def schema_types(frame: DataFrame) -> dict[str, str]:
    return {field.name: field.dataType.simpleString() for field in frame.schema.fields}


def require_schema(
    frame: DataFrame,
    columns: tuple[str, ...],
    types: dict[str, str],
    label: str,
) -> None:
    require(tuple(frame.columns) == columns, f"{label} column order differs")
    actual = schema_types(frame)
    require(actual == types, f"{label} schema differs")


def require_same(left: DataFrame, right: DataFrame, columns: tuple[str, ...], label: str) -> None:
    left_view = left.select(*columns)
    right_view = right.select(*columns)
    require(left_view.exceptAll(right_view).limit(1).count() == 0, f"{label}: unexpected rows")
    require(right_view.exceptAll(left_view).limit(1).count() == 0, f"{label}: missing rows")


def row_digest(frame: DataFrame, columns: tuple[str, ...], name: str) -> DataFrame:
    payload = F.to_json(
        F.struct(*(F.col(column) for column in columns)),
        options={"ignoreNullFields": "false"},
    )
    return frame.select(TRACK_ID, F.sha2(payload, 256).alias(name))


def require_same_values(
    left: DataFrame,
    right: DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    left_hash = row_digest(left, columns, "_left_hash")
    right_hash = row_digest(right, columns, "_right_hash")
    mismatch = (
        left_hash.join(right_hash, TRACK_ID, "inner")
        .where(F.col("_left_hash") != F.col("_right_hash"))
        .limit(1)
        .count()
    )
    require(mismatch == 0, f"{label}: values differ")


def finite(column: Column) -> Column:
    return column.isNotNull() & ~F.isnan(column) & (F.abs(column) != F.lit(float("inf")))


def clip_ratio(numerator: Column, denominator: Column) -> Column:
    return F.greatest(F.lit(0.0), F.least(F.lit(1.0), numerator / denominator))


def expected_globals(scalar: DataFrame) -> DataFrame:
    cleaned = (
        scalar.withColumn(
            "tempo",
            F.when(finite(F.col("tempo")) & (F.col("tempo") > 0), F.col("tempo")).cast(
                "double"
            ),
        )
        .withColumn("key", F.when(F.col("key").between(0, 11), F.col("key")).cast("int"))
        .withColumn("mode", F.when(F.col("mode").isin(0, 1), F.col("mode")).cast("int"))
        .withColumn(
            "time_signature",
            F.when(F.col("time_signature") > 0, F.col("time_signature")).cast("int"),
        )
    )
    duration = F.col("duration")
    fade_in = F.col("end_of_fade_in")
    fade_out = F.col("start_of_fade_out")
    valid = (
        finite(duration)
        & (duration > 0.0)
        & finite(fade_in)
        & finite(fade_out)
        & (fade_in >= 0.0)
        & (fade_out >= 0.0)
        & (fade_in <= fade_out)
        & (fade_in <= duration + FADE_TOLERANCE_SECONDS)
        & (fade_out <= duration + FADE_TOLERANCE_SECONDS)
    )
    clipped_in = F.least(fade_in, duration)
    clipped_out = F.least(fade_out, duration)
    missing = F.lit(None).cast("double")
    return (
        cleaned.withColumn(
            "fade_in_ratio",
            F.when(valid, clip_ratio(clipped_in, duration)).otherwise(missing),
        )
        .withColumn(
            "fade_out_ratio",
            F.when(valid, clip_ratio(duration - clipped_out, duration)).otherwise(missing),
        )
        .withColumn(
            "active_audio_ratio",
            F.when(valid, clip_ratio(clipped_out - clipped_in, duration)).otherwise(missing),
        )
        .select(TRACK_ID, *GLOBAL_SCALAR_COLUMNS, *DERIVED_SCALAR_COLUMNS)
    )


def expected_audit(scalar: DataFrame, labels: DataFrame) -> DataFrame:
    split = labels.select(TRACK_ID, SPLIT)
    return scalar.select(TRACK_ID, ARTIST_ID, YEAR).join(split, TRACK_ID, "left").select(
        *AUDIT_COLUMNS
    )


def require_output_counts(frame: DataFrame, label: str, split_counts: dict[str, Any]) -> None:
    summary = frame.agg(
        F.count("*").alias("rows"),
        F.countDistinct(TRACK_ID).alias("tracks"),
        F.sum(F.when(F.col(YEAR).isNotNull(), 1).otherwise(0)).alias("labeled"),
        F.sum(
            F.when(F.col(YEAR).isNotNull() != F.col(SPLIT).isNotNull(), 1).otherwise(0)
        ).alias("label_split_mismatch"),
        F.sum(
            F.when(
                F.col(TRACK_ID).isNull()
                | (F.col(TRACK_ID) == "")
                | F.col(ARTIST_ID).isNull()
                | (F.col(ARTIST_ID) == ""),
                1,
            ).otherwise(0)
        ).alias("invalid_ids"),
    ).first()
    require(int(summary["rows"]) == EXPECTED_TRACKS, f"{label} row count differs")
    require(int(summary["tracks"]) == EXPECTED_TRACKS, f"{label} track IDs are duplicated")
    require(int(summary["labeled"]) == EXPECTED_LABELED_TRACKS, f"{label} label count differs")
    require(int(summary["label_split_mismatch"]) == 0, f"{label} label/split relation differs")
    require(int(summary["invalid_ids"]) == 0, f"{label} contains invalid IDs")
    actual_splits = {
        row[SPLIT]: int(row["tracks"])
        for row in frame.where(F.col(SPLIT).isNotNull()).groupBy(SPLIT).count().withColumnRenamed(
            "count", "tracks"
        ).collect()
    }
    expected_splits = {name: int(values["tracks"]) for name, values in split_counts.items()}
    require(actual_splits == expected_splits, f"{label} split counts differ")


def require_no_infinity(frame: DataFrame, numeric_columns: tuple[str, ...]) -> None:
    values = F.array(*(F.col(column).cast("double") for column in numeric_columns))
    invalid = F.exists(values, lambda value: F.isnan(value) | (F.abs(value) == float("inf")))
    require(frame.where(invalid).limit(1).count() == 0, "full view contains NaN or Inf")


def require_binary_flags(frame: DataFrame) -> None:
    masks = tuple(column for column in BINARY_FEATURE_COLUMNS if column != "mode")
    invalid_masks = reduce(
        lambda left, right: left | right,
        (F.col(column).isNull() | ~F.col(column).isin(0.0, 1.0) for column in masks),
    )
    invalid_mode = F.col("mode").isNotNull() & ~F.col("mode").isin(0, 1)
    require(frame.where(invalid_masks | invalid_mode).limit(1).count() == 0, "binary flags differ")


def require_categories(frame: DataFrame) -> None:
    invalid = (
        (F.col("key").isNotNull() & ~F.col("key").between(0, 11))
        | (F.col("mode").isNotNull() & ~F.col("mode").isin(0, 1))
        | (F.col("time_signature").isNotNull() & (F.col("time_signature") <= 0))
    )
    require(frame.where(invalid).limit(1).count() == 0, "categorical values differ")


def require_mask_nulls(frame: DataFrame) -> None:
    dependencies = {
        "has_beat_intervals": tuple(
            column
            for column in frame.columns
            if column.startswith("beat_interval_") or column.startswith("beat_local_bpm_")
        ),
        "has_bar_intervals": tuple(
            column for column in frame.columns if column.startswith("bar_interval_")
        ),
        "has_tatum_intervals": tuple(
            column for column in frame.columns if column.startswith("tatum_interval_")
        ),
        "has_t90": T90_COLUMNS,
        "has_pitch_profile": ("pitch_profile_entropy", "pitch_profile_concentration"),
        "has_key_relative_pitch": tuple(
            column for column in frame.columns if column.startswith("key_relative_")
        ),
    }
    for part in range(4):
        dependencies[f"has_quarter_{part}"] = tuple(
            column
            for column in frame.columns
            if column.startswith(f"quarter_{part}_")
            or column.startswith(f"key_relative_quarter_{part}_")
        )
    for part in range(2):
        dependencies[f"has_half_{part}"] = tuple(
            column
            for column in frame.columns
            if column.startswith(f"half_{part}_")
            or column.startswith(f"key_relative_half_{part}_")
        )
    for mask, columns in dependencies.items():
        present_when_missing = reduce(
            lambda left, right: left | right,
            (F.col(column).isNotNull() for column in columns),
        )
        require(
            frame.where((F.col(mask) == 0.0) & present_when_missing).limit(1).count() == 0,
            f"{mask} does not agree with dependent nulls",
        )


def require_ratio_identity(frame: DataFrame) -> None:
    ratios = [F.col(column) for column in DERIVED_SCALAR_COLUMNS]
    all_present = reduce(lambda left, right: left & right, (column.isNotNull() for column in ratios))
    invalid = all_present & (F.abs(sum(ratios[1:], ratios[0]) - 1.0) > 1e-12)
    require(frame.where(invalid).limit(1).count() == 0, "fade ratios do not sum to one")


def validate_hdf5_samples(
    root: Path,
    sample_count: int,
    audio: DataFrame,
    audio_columns: tuple[str, ...],
) -> None:
    require(sample_count > 0, "HDF5 sample count must be positive")
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    try:
        import numpy as np
        from tools.hdf5.audio_features import FEATURE_COLUMNS
        from tools.hdf5.extract_musics import process_one_file
    except ImportError as error:
        raise ValueError("The shared 628-dimensional HDF5 extractor branch is not available") from error

    require(tuple(FEATURE_COLUMNS) == audio_columns, "HDF5 extractor contract differs")
    rows = audio.orderBy(TRACK_ID).limit(sample_count).collect()
    for row in rows:
        track_id = row[TRACK_ID]
        path = root / track_id[2] / track_id[3] / track_id[4] / f"{track_id}.h5"
        extracted_id, extracted = process_one_file(path)
        require(extracted_id == track_id, f"HDF5 track ID differs: {track_id}")
        stored = np.asarray(
            [np.nan if row[column] is None else row[column] for column in audio_columns],
            dtype=np.float64,
        )
        require(
            bool(np.allclose(stored, extracted, rtol=1e-10, atol=1e-10, equal_nan=True)),
            f"HDF5 recomputation differs: {track_id}",
        )


def validate(args: argparse.Namespace, spark: SparkSession) -> None:
    manifest_path = args.features / "manifest.json"
    manifest = load_json(manifest_path)
    audio_contract_path = args.audio / "feature_contract.json"
    audio_contract = load_audio_contract(audio_contract_path)
    dataset_manifest_path = args.dataset / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    require(manifest["contract_version"] == FEATURE_CONTRACT_VERSION, "feature contract differs")
    require(manifest["format_version"] == 1, "feature manifest format differs")
    require(tuple(manifest["audit_columns"]) == AUDIT_COLUMNS, "manifest audit columns differ")
    require(manifest["sources"]["scalar"]["sha256"] == sha256_file(args.scalar), "scalar checksum differs")
    require(
        manifest["sources"]["audio"]["contract_sha256"] == sha256_file(audio_contract_path),
        "audio contract checksum differs",
    )
    require(
        manifest["sources"]["dataset"]["manifest_sha256"] == sha256_file(dataset_manifest_path),
        "dataset manifest checksum differs",
    )
    require(
        manifest["sources"]["audio"]["feature_order_sha256"]
        == AUDIO_FEATURE_ORDER_SHA256,
        "audio order hash differs",
    )

    paths = audio_paths(args.audio)
    require(len(paths) == 100, "audio batch count differs")
    audio = spark.read.parquet(*paths)
    scalar = spark.read.parquet(spark_path(args.scalar))
    labels = spark.read.parquet(spark_path(args.dataset / "labelled_tracks.parquet"))
    t90 = spark.read.parquet(spark_path(args.features / "t90.parquet"))
    full = spark.read.parquet(spark_path(args.features / "full_tabular.parquet"))

    shared_columns = year_shared_columns(audio_contract)
    predictor_columns = full_predictor_columns(audio_contract)
    audit_types = {TRACK_ID: "string", ARTIST_ID: "string", YEAR: "int", SPLIT: "string"}
    t90_types = {**audit_types, **{column: "double" for column in T90_COLUMNS}}
    full_types = {
        **audit_types,
        **{column: "double" for column in shared_columns},
        **{column: "double" for column in GLOBAL_SCALAR_COLUMNS},
        "key": "int",
        "mode": "int",
        "time_signature": "int",
        **{column: "double" for column in DERIVED_SCALAR_COLUMNS},
    }
    require_schema(t90, AUDIT_COLUMNS + T90_COLUMNS, t90_types, "T90")
    require_schema(full, AUDIT_COLUMNS + predictor_columns, full_types, "full tabular")
    require(manifest["counts"]["tracks"] == EXPECTED_TRACKS, "manifest track count differs")
    require(
        manifest["counts"]["labeled_tracks"] == EXPECTED_LABELED_TRACKS,
        "manifest label count differs",
    )
    require(
        tuple(manifest["views"]["t90"]["predictor_columns"]) == T90_COLUMNS,
        "manifest T90 columns differ",
    )
    require(
        tuple(manifest["views"]["full_tabular"]["predictor_columns"])
        == predictor_columns,
        "manifest full columns differ",
    )
    require(
        manifest["views"]["t90"]["predictor_count"] == len(T90_COLUMNS),
        "manifest T90 dimension differs",
    )
    require(
        manifest["views"]["full_tabular"]["predictor_count"]
        == len(predictor_columns),
        "manifest full dimension differs",
    )
    for view_name, frame in (("t90", t90), ("full_tabular", full)):
        manifest_schema = {
            field["name"]: field["type"] for field in manifest["views"][view_name]["schema"]
        }
        require(manifest_schema == schema_types(frame), f"manifest {view_name} schema differs")
    require(not set(YEAR_EXCLUDED_COLUMNS) & set(full.columns), "excluded audio columns found")
    require(
        not (set(full.columns) & set(FORBIDDEN_PREDICTOR_COLUMNS)) - set(AUDIT_COLUMNS),
        "forbidden predictors found",
    )
    require(manifest["views"]["t90"]["predictor_order_sha256"] == order_sha256(T90_COLUMNS), "T90 order hash differs")
    require(
        manifest["views"]["full_tabular"]["predictor_order_sha256"]
        == order_sha256(predictor_columns),
        "full order hash differs",
    )
