from __future__ import annotations

TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
NORMALIZED_YEAR = "normalized_year"
SPLIT = "split"
TRAIN = "train"

MIN_YEAR = 1922
MAX_YEAR = 2011
YEAR_SPAN = MAX_YEAR - MIN_YEAR

RAW_CONTINUOUS_COLUMNS = (
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "duration",
)
KEY_COLUMN = "key"
MODE_COLUMN = "mode"
TIME_SIGNATURE_COLUMN = "time_signature"
HAS_SEGMENTS_COLUMN = "has_segments"


def indexed_columns(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_{statistic}_{index}"
        for statistic in ("mean", "std", "min", "max")
        for index in range(12)
    )


PITCH_COLUMNS = indexed_columns("pitch")
TIMBRE_COLUMNS = indexed_columns("timbre")
SEGMENT_LOUDNESS_COLUMNS = (
    "loudness_mean",
    "loudness_std",
    "loudness_min",
    "loudness_max",
)
SEGMENT_COLUMNS = (*PITCH_COLUMNS, *TIMBRE_COLUMNS, *SEGMENT_LOUDNESS_COLUMNS)

RAW_PREDICTOR_COLUMNS = (
    *RAW_CONTINUOUS_COLUMNS,
    KEY_COLUMN,
    MODE_COLUMN,
    TIME_SIGNATURE_COLUMN,
    *SEGMENT_COLUMNS,
    HAS_SEGMENTS_COLUMN,
)
INPUT_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, *RAW_PREDICTOR_COLUMNS, SPLIT)

TRANSFORMED_CONTINUOUS_COLUMNS = (
    "danceability",
    "energy",
    "loudness_clipped",
    "tempo_log",
    "duration_log",
    *SEGMENT_COLUMNS,
)
KEY_ENCODED_COLUMNS = ("key_sin", "key_cos", "key_unknown")
TIME_SIGNATURE_UNKNOWN_COLUMN = "time_signature_unknown"


def time_signature_column(value: int) -> str:
    return f"time_signature_{value}"


def candidate_columns(time_signature_values: tuple[int, ...]) -> tuple[str, ...]:
    return (
        *TRANSFORMED_CONTINUOUS_COLUMNS,
        *KEY_ENCODED_COLUMNS,
        MODE_COLUMN,
        HAS_SEGMENTS_COLUMN,
        *(time_signature_column(value) for value in time_signature_values),
        TIME_SIGNATURE_UNKNOWN_COLUMN,
    )


IDENTIFIER_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, NORMALIZED_YEAR)
AUDIT_CATEGORY_COLUMNS = (KEY_COLUMN, TIME_SIGNATURE_COLUMN)

