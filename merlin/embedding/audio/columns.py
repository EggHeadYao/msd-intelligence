"""Frozen audio feature contract and C1 model-input column definitions."""

from __future__ import annotations

from shared_contract import (
    CONTRACT_VERSION,
    MERLIN_ARRAY_FEATURE_COLUMNS,
    SHARED_FEATURE_COUNT,
)

CONTRACT_VERSION = "shared_audio_628_v1"
LEGACY_FEATURE_COUNT = 308
SHARED_FEATURE_COUNT = 628
PITCH_DIMS = 12
TIMBRE_DIMS = 12
ROBUST_STATS = ("mean", "std", "q10", "q90")


def _indexed(prefix: str, stats: tuple[str, ...], dims: int = 12) -> tuple[str, ...]:
    return tuple(f"{prefix}_{stat}_{index}" for stat in stats for index in range(dims))


GLOBAL_PITCH_COLUMNS = _indexed("pitch", ROBUST_STATS)
GLOBAL_TIMBRE_COLUMNS = _indexed("timbre", ROBUST_STATS)
LOUDNESS_SIGNALS = (
    "loudness_start", "loudness_max", "loudness_gain", "loudness_peak_time_ratio",
)
GLOBAL_LOUDNESS_COLUMNS = tuple(
    f"{signal}_{stat}" for signal in LOUDNESS_SIGNALS for stat in ROBUST_STATS
)
QUARTER_COLUMNS = tuple(
    column
    for quarter in range(4)
    for column in (
        *(f"quarter_{quarter}_pitch_mean_{index}" for index in range(12)),
        *(f"quarter_{quarter}_timbre_mean_{index}" for index in range(12)),
        f"quarter_{quarter}_loudness_max_mean",
    )
)
SEGMENT_TIMING_COLUMNS = (
    "segment_count", "segment_density",
    *(f"segment_duration_{stat}" for stat in ("mean", "std", "q10", "q50", "q90", "cv")),
    *(f"pitch_delta_cosine_{stat}" for stat in ("mean", "std", "q50", "q90")),
    *_indexed("timbre_delta_abs", ("mean", "std")),
    *(f"loudness_delta_abs_{stat}" for stat in ("mean", "std", "q50", "q90")),
)
EVENTS = ("beat", "bar", "tatum", "section")
RHYTHM_STRUCTURE_COLUMNS = (
    *(f"{event}_{stat}" for event in EVENTS for stat in ("count", "density")),
    *(f"{event}_interval_{stat}" for event in EVENTS[:3] for stat in ("median", "iqr", "cv")),
    "beat_local_bpm_median", "beat_local_bpm_cv", "global_local_tempo_deviation",
    "beats_per_bar_median", "tatums_per_beat_median",
    *(f"{event}_{suffix}" for event in ("segments", "beats", "bars", "tatums", "sections")
      for suffix in ("confidence_mean", "low_confidence_fraction")),
    *(f"section_duration_{stat}" for stat in ("mean", "std", "q10", "q50", "q90", "cv")),
    "section_longest_ratio",
    *(f"section_{signal}_change_{stat}" for signal in ("pitch", "timbre", "loudness")
      for stat in ("mean", "std")),
)
LEGACY_MASK_COLUMNS = (
    "has_segments", "has_beats", "has_bars", "has_tatums", "has_sections",
    "invalid_segment_duration_fraction", "valid_analysis_duration_ratio",
    "has_quarter_0", "has_quarter_1", "has_quarter_2", "has_quarter_3",
)
LEGACY_FEATURE_COLUMNS = (
    *GLOBAL_PITCH_COLUMNS, *GLOBAL_TIMBRE_COLUMNS, *GLOBAL_LOUDNESS_COLUMNS,
    *QUARTER_COLUMNS, *SEGMENT_TIMING_COLUMNS, *RHYTHM_STRUCTURE_COLUMNS,
    *LEGACY_MASK_COLUMNS,
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
PREPARED_SCALAR_COLUMNS = (
    TRACK_ID_COLUMN, "loudness", "tempo", "duration", KEY_COLUMN, "key_confidence",
    MODE_COLUMN, "mode_confidence", TIME_SIGNATURE_COLUMN, "time_signature_confidence",
    "end_of_fade_in", "start_of_fade_out",
)

KEY_CIRCULAR_COLUMNS = ("key_sin", "key_cos")  # Derived circular key columns.
LOG_CONTINUOUS_COLUMNS = ("tempo_log", "duration_log")  # Derived clipped-log columns.
CLIPPED_CONTINUOUS_COLUMNS = ("loudness_clipped",)  # Derived clipped loudness column.
PASSTHROUGH_CONTINUOUS_COLUMNS = (
    "key_confidence", "mode_confidence", "time_signature_confidence",
)
FADE_RATIO_COLUMNS = ("fade_in_ratio", "fade_out_ratio")
PASSTHROUGH_BINARY_COLUMNS = (MODE_COLUMN,)
TIME_SIGNATURE_ONE_HOT_PREFIX = "time_signature_"  # Prefix for meter one-hot columns.
TIME_SIGNATURE_VALUES = (3, 4, 5, 6, 7)
TIME_SIGNATURE_UNKNOWN_COLUMN = "time_signature_unknown"
SCALAR_AVAILABILITY_COLUMNS = (
    "has_loudness", "has_tempo", "has_duration", "has_key", "has_mode",
    "has_time_signature", "has_fade_in_ratio", "has_fade_out_ratio",
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
PREPARED_AUDIO_COLUMNS = (*PREPARED_SCALAR_COLUMNS, *SEGMENT_FEATURE_COLUMNS)
SEGMENT_FEATURE_COLUMN_SET = frozenset(SEGMENT_FEATURE_COLUMNS)  # Fast segment lookup set.


def build_feature_columns(time_signature_columns: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *KEY_CIRCULAR_COLUMNS,
        *LOG_CONTINUOUS_COLUMNS,
        *CLIPPED_CONTINUOUS_COLUMNS,
        *PASSTHROUGH_CONTINUOUS_COLUMNS,
        *FADE_RATIO_COLUMNS,
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
