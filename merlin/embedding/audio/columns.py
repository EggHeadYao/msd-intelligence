from __future__ import annotations

from shared_contract import (
    CONTRACT_VERSION,
    MERLIN_ARRAY_FEATURE_COLUMNS,
    SHARED_FEATURE_COUNT,
)

TRACK_ID_COLUMN = "track_id"  # Stable song key across MERLIN artifacts.
KEY_COLUMN = "key"  # Raw 0-11 musical key; circular-encoded later.
MODE_COLUMN = "mode"  # Raw major/minor mode flag.
TIME_SIGNATURE_COLUMN = "time_signature"  # Raw meter category; one-hot later.
HAS_SEGMENTS_COLUMN = "has_segments"  # Raw mask for available segment arrays.

PITCH_DIMS = 12  # Pitch-class dimensions per Echo Nest segment.
TIMBRE_DIMS = 12  # Timbre coefficient dimensions per Echo Nest segment.
SEGMENT_STATS = ("mean", "std", "min", "max")  # Stats for pitch/timbre columns.
LOUDNESS_STATS = ("mean", "std", "min", "max")  # Stats for segment loudness.

RAW_CONTINUOUS_COLUMNS = (
    "loudness", "tempo", "duration", "key_confidence", "mode_confidence",
    "time_signature_confidence", "end_of_fade_in", "start_of_fade_out",
)  # Raw numeric scalars before clipping, log transforms, and scaling.
RAW_CATEGORICAL_COLUMNS = (KEY_COLUMN, TIME_SIGNATURE_COLUMN)  # Raw categorical scalars.
RAW_BINARY_COLUMNS = (MODE_COLUMN, HAS_SEGMENTS_COLUMN)  # Raw binary scalars.
RAW_SCALAR_COLUMNS = (
    TRACK_ID_COLUMN, *RAW_CONTINUOUS_COLUMNS, *RAW_CATEGORICAL_COLUMNS, MODE_COLUMN,
)  # Raw non-segment columns in song_audio_features_raw.parquet.

KEY_CIRCULAR_COLUMNS = ("key_sin", "key_cos")  # Derived circular key columns.
LOG_CONTINUOUS_COLUMNS = ("tempo_log", "duration_log")  # Derived clipped-log columns.
CLIPPED_CONTINUOUS_COLUMNS = ("loudness_clipped",)  # Derived clipped loudness column.
PASSTHROUGH_CONTINUOUS_COLUMNS = (
    "key_confidence", "mode_confidence", "time_signature_confidence",
    "end_of_fade_in", "start_of_fade_out",
)
PASSTHROUGH_BINARY_COLUMNS = (MODE_COLUMN,)
TIME_SIGNATURE_ONE_HOT_PREFIX = "time_signature_"  # Prefix for meter one-hot columns.
TIME_SIGNATURE_VALUES = (3, 4, 5, 6, 7)
TIME_SIGNATURE_UNKNOWN_COLUMN = "time_signature_unknown"
SCALAR_AVAILABILITY_COLUMNS = (
    "has_loudness", "has_tempo", "has_duration", "has_key", "has_mode",
    "has_time_signature",
)


def indexed_feature_columns(
    prefix: str,
    dimensions: int,
    stats: tuple[str, ...] = SEGMENT_STATS,
) -> tuple[str, ...]:
    return tuple(f"{prefix}_{stat}_{index}" for stat in stats for index in range(dimensions))


def pitch_feature_columns() -> tuple[str, ...]:
    return indexed_feature_columns("pitch", PITCH_DIMS)


def timbre_feature_columns() -> tuple[str, ...]:
    return indexed_feature_columns("timbre", TIMBRE_DIMS)


def loudness_feature_columns() -> tuple[str, ...]:
    return tuple(f"loudness_{stat}" for stat in LOUDNESS_STATS)


def time_signature_one_hot_column(value: int) -> str:
    return f"{TIME_SIGNATURE_ONE_HOT_PREFIX}{value}"


PITCH_FEATURE_COLUMNS = pitch_feature_columns()  # Expanded 48 pitch aggregate columns.
TIMBRE_FEATURE_COLUMNS = timbre_feature_columns()  # Expanded 48 timbre aggregate columns.
LOUDNESS_FEATURE_COLUMNS = loudness_feature_columns()  # Expanded 4 loudness columns.
SEGMENT_FEATURE_COLUMNS = MERLIN_ARRAY_FEATURE_COLUMNS
RAW_AUDIO_COLUMNS = (*RAW_SCALAR_COLUMNS, *SEGMENT_FEATURE_COLUMNS)
SEGMENT_FEATURE_COLUMN_SET = frozenset(SEGMENT_FEATURE_COLUMNS)  # Fast segment lookup set.


def build_feature_columns(time_signature_columns: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *KEY_CIRCULAR_COLUMNS,
        *LOG_CONTINUOUS_COLUMNS,
        *CLIPPED_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_BINARY_COLUMNS,
        *time_signature_columns,
        *SCALAR_AVAILABILITY_COLUMNS,
        *SEGMENT_FEATURE_COLUMNS,
    )


MERLIN_ARRAY_FEATURE_COUNT = len(MERLIN_ARRAY_FEATURE_COLUMNS)
MERLIN_RAW_VIEW_COUNT = MERLIN_ARRAY_FEATURE_COUNT + 11

assert SHARED_FEATURE_COUNT == 628
assert MERLIN_ARRAY_FEATURE_COUNT == 552
assert MERLIN_RAW_VIEW_COUNT == 563
