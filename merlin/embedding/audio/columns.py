from __future__ import annotations

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
    "danceability", "energy", "loudness", "tempo", "duration",
)  # Raw numeric scalars before clipping, log transforms, and scaling.
RAW_CATEGORICAL_COLUMNS = (KEY_COLUMN, TIME_SIGNATURE_COLUMN)  # Raw categorical scalars.
RAW_BINARY_COLUMNS = (MODE_COLUMN, HAS_SEGMENTS_COLUMN)  # Raw binary scalars.
RAW_SCALAR_COLUMNS = (
    TRACK_ID_COLUMN, *RAW_CONTINUOUS_COLUMNS, *RAW_CATEGORICAL_COLUMNS, MODE_COLUMN,
)  # Raw non-segment columns in song_audio_features_raw.parquet.

KEY_CIRCULAR_COLUMNS = ("key_sin", "key_cos")  # Derived circular key columns.
LOG_CONTINUOUS_COLUMNS = ("tempo_log", "duration_log")  # Derived clipped-log columns.
CLIPPED_CONTINUOUS_COLUMNS = ("loudness_clipped",)  # Derived clipped loudness column.
PASSTHROUGH_CONTINUOUS_COLUMNS = ("danceability", "energy")  # Raw numeric keep-candidates.
PASSTHROUGH_BINARY_COLUMNS = (MODE_COLUMN, HAS_SEGMENTS_COLUMN)  # Raw binary keep-candidates.
TIME_SIGNATURE_ONE_HOT_PREFIX = "time_signature_"  # Prefix for meter one-hot columns.


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
SEGMENT_FEATURE_COLUMNS = (
    *PITCH_FEATURE_COLUMNS, *TIMBRE_FEATURE_COLUMNS, *LOUDNESS_FEATURE_COLUMNS,
)  # All 100 segment aggregate columns used by the audio encoder.
RAW_AUDIO_COLUMNS = (*RAW_SCALAR_COLUMNS, *SEGMENT_FEATURE_COLUMNS, HAS_SEGMENTS_COLUMN)  # Full raw input schema.
SEGMENT_FEATURE_COLUMN_SET = frozenset(SEGMENT_FEATURE_COLUMNS)  # Fast segment lookup set.


def build_feature_columns(time_signature_columns: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *KEY_CIRCULAR_COLUMNS,
        *LOG_CONTINUOUS_COLUMNS,
        *CLIPPED_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_BINARY_COLUMNS,
        *time_signature_columns,
        *SEGMENT_FEATURE_COLUMNS,
    )
