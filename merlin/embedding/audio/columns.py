from __future__ import annotations

TRACK_ID_COLUMN = "track_id"
KEY_COLUMN = "key"
MODE_COLUMN = "mode"
TIME_SIGNATURE_COLUMN = "time_signature"
HAS_SEGMENTS_COLUMN = "has_segments"

PITCH_DIMS = 12
TIMBRE_DIMS = 12
SEGMENT_STATS = ("mean", "std", "min", "max")
LOUDNESS_STATS = ("mean", "std", "min", "max")

RAW_CONTINUOUS_COLUMNS = (
    "danceability", "energy", "loudness", "tempo", "duration",
)
RAW_CATEGORICAL_COLUMNS = (KEY_COLUMN, TIME_SIGNATURE_COLUMN)
RAW_BINARY_COLUMNS = (MODE_COLUMN, HAS_SEGMENTS_COLUMN)
RAW_SCALAR_COLUMNS = (
    TRACK_ID_COLUMN, *RAW_CONTINUOUS_COLUMNS, *RAW_CATEGORICAL_COLUMNS, MODE_COLUMN,
)

KEY_CIRCULAR_COLUMNS = ("key_sin", "key_cos")
LOG_CONTINUOUS_COLUMNS = ("tempo_log", "duration_log")
CLIPPED_CONTINUOUS_COLUMNS = ("loudness_clipped",)
PASSTHROUGH_CONTINUOUS_COLUMNS = ("danceability", "energy")
PASSTHROUGH_BINARY_COLUMNS = (MODE_COLUMN, HAS_SEGMENTS_COLUMN)
TIME_SIGNATURE_ONE_HOT_PREFIX = "time_signature_"


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


PITCH_FEATURE_COLUMNS = pitch_feature_columns()
TIMBRE_FEATURE_COLUMNS = timbre_feature_columns()
LOUDNESS_FEATURE_COLUMNS = loudness_feature_columns()
SEGMENT_FEATURE_COLUMNS = (
    *PITCH_FEATURE_COLUMNS, *TIMBRE_FEATURE_COLUMNS, *LOUDNESS_FEATURE_COLUMNS,
)
RAW_AUDIO_COLUMNS = (*RAW_SCALAR_COLUMNS, *SEGMENT_FEATURE_COLUMNS, HAS_SEGMENTS_COLUMN)
SEGMENT_FEATURE_COLUMN_SET = frozenset(SEGMENT_FEATURE_COLUMNS)
