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
