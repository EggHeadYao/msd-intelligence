from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TRACK_ID = "track_id"
ARTIST_ID = "artist_id"
YEAR = "year"
SPLIT = "split"
AUDIT_COLUMNS = (TRACK_ID, ARTIST_ID, YEAR, SPLIT)

AUDIO_CONTRACT_VERSION = "shared_audio_628_v1"
AUDIO_FEATURE_COUNT = 628
AUDIO_FEATURE_ORDER_SHA256 = (
    "70a34615d2a0c4734df885b08fcb752a83f4c757a1bd6339ad9cc6601aa5f0ec"
)
FEATURE_CONTRACT_VERSION = "year_prediction_features_v1"
EXPECTED_TRACKS = 1_000_000
EXPECTED_LABELED_TRACKS = 515_576


def indexed(prefix: str, count: int = 12) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(count))


def make_t90_columns() -> tuple[str, ...]:
    means = indexed("t90_timbre_mean")
    covariance = tuple(
        f"t90_timbre_cov_{index}_{index + offset}"
        for offset in range(12)
        for index in range(12 - offset)
    )
    return means + covariance


T90_COLUMNS = make_t90_columns()
YEAR_EXCLUDED_COLUMNS = tuple(
    f"key_relative_pitch_{stat}_{index}"
    for stat in ("std", "q10", "q50", "q90")
    for index in range(12)
)
GLOBAL_SCALAR_COLUMNS = (
    "loudness",
    "tempo",
    "duration",
    "key",
    "key_confidence",
    "mode",
    "mode_confidence",
    "time_signature",
    "time_signature_confidence",
    "end_of_fade_in",
    "start_of_fade_out",
)
DERIVED_SCALAR_COLUMNS = ("fade_in_ratio", "fade_out_ratio", "active_audio_ratio")

ROBUST_COLUMNS = tuple(
    column
    for prefix in ("pitch", "timbre")
    for stat in ("mean", "std", "q10", "q50", "q90")
    for column in indexed(f"{prefix}_{stat}")
) + tuple(
    f"{signal}_{stat}"
    for signal in (
        "loudness_start",
        "loudness_max",
        "loudness_gain",
        "loudness_peak_time_ratio",
    )
    for stat in ("mean", "std", "q10", "q50", "q90")
)
TEMPORAL_COLUMNS = tuple(
    column
    for part in range(4)
    for column in (
        *indexed(f"quarter_{part}_pitch_mean"),
        *indexed(f"quarter_{part}_timbre_mean"),
        f"quarter_{part}_loudness_max_mean",
    )
) + tuple(
    column
    for part in range(2)
    for column in (
        *indexed(f"half_{part}_pitch_mean"),
        *indexed(f"half_{part}_timbre_mean"),
        f"half_{part}_loudness_max_mean",
    )
) + tuple(f"has_quarter_{part}" for part in range(4)) + ("has_half_0", "has_half_1")
TONALITY_COLUMNS = (
    indexed("key_relative_pitch_mean")
    + tuple(
        column
        for part in range(2)
        for column in indexed(f"key_relative_half_{part}_pitch_mean")
    )
    + tuple(
        column
        for part in range(4)
        for column in indexed(f"key_relative_quarter_{part}_pitch_mean")
    )
    + (
        "pitch_profile_entropy",
        "pitch_profile_concentration",
        "has_pitch_profile",
        "has_key_relative_pitch",
    )
)
DYNAMICS_COLUMNS = (
    "segment_count",
    "segment_density",
    *(f"segment_duration_{stat}" for stat in ("mean", "std", "q10", "q50", "q90", "cv")),
    *(f"pitch_delta_cosine_{stat}" for stat in ("mean", "std", "q50", "q90")),
    *(f"timbre_delta_abs_{stat}_{index}" for stat in ("mean", "std") for index in range(12)),
    *(f"loudness_delta_abs_{stat}" for stat in ("mean", "std", "q50", "q90")),
    "section_pitch_change_mean",
    "section_pitch_change_std",
    "section_timbre_change_mean",
    "section_timbre_change_std",
    "section_loudness_change_mean",
    "section_loudness_change_std",
)
RHYTHM_COLUMNS = (
    *(f"{event}_{suffix}" for event in ("beat", "bar", "tatum", "section") for suffix in ("count", "density")),
    *(f"{event}_interval_{stat}" for event in ("beat", "bar", "tatum") for stat in ("median", "iqr", "cv")),
    "beat_local_bpm_median",
    "beat_local_bpm_cv",
    "global_local_tempo_deviation",
    "beats_per_bar_median",
    "tatums_per_beat_median",
    *(f"{event}_{suffix}" for event in ("segments", "beats", "bars", "tatums", "sections") for suffix in ("confidence_mean", "low_confidence_fraction")),
    *(f"section_duration_{stat}" for stat in ("mean", "std", "q10", "q50", "q90", "cv")),
    "section_longest_ratio",
    *(f"{event}_interval_{quantile}" for event in ("beat", "bar", "tatum") for quantile in ("q10", "q90")),
    "beat_local_bpm_q10",
    "beat_local_bpm_q90",
    "beat_local_bpm_iqr",
    "section_duration_iqr",
    "has_beat_intervals",
    "has_bar_intervals",
    "has_tatum_intervals",
)
QUALITY_COLUMNS = (
    "has_segments",
    "has_beats",
    "has_bars",
    "has_tatums",
    "has_sections",
    "invalid_segment_duration_fraction",
    "valid_analysis_duration_ratio",
    "has_t90",
)
FEATURE_GROUPS = {
    "robust": ROBUST_COLUMNS,
    "temporal": TEMPORAL_COLUMNS,
    "t90": T90_COLUMNS,
    "tonality": TONALITY_COLUMNS,
    "dynamics": DYNAMICS_COLUMNS,
    "rhythm": RHYTHM_COLUMNS,
    "quality": QUALITY_COLUMNS,
}
EXPECTED_GROUP_COUNTS = {
    "robust": 140,
    "temporal": 156,
    "t90": 90,
    "tonality": 88,
    "dynamics": 46,
    "rhythm": 52,
    "quality": 8,
}
YEAR_SHARED_FEATURE_SET = frozenset(
    column for columns in FEATURE_GROUPS.values() for column in columns
)
BINARY_FEATURE_COLUMNS = tuple(
    column
    for columns in FEATURE_GROUPS.values()
    for column in columns
    if column.startswith("has_")
) + ("mode",)
CATEGORICAL_COLUMNS = ("key", "mode", "time_signature")
FORBIDDEN_PREDICTOR_COLUMNS = (
    "track_id",
    "song_id",
    "artist_id",
    "artist_name",
    "title",
    "release",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
    "year",
    "split",
)


