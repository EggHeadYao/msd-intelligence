"""Frozen shared_audio_628_v1 column order and MERLIN projection."""

CONTRACT_VERSION = "shared_audio_628_v1"
LEGACY_FEATURE_COUNT = 308
SHARED_FEATURE_COUNT = 628
PITCH_DIMS = TIMBRE_DIMS = 12
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
    *(f"{event}_confidence_{stat}" for event in ("segment", *EVENTS)
      for stat in ("mean", "low_confidence_fraction")),
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

GLOBAL_Q50_COLUMNS = (
    *(f"pitch_q50_{index}" for index in range(12)),
    *(f"timbre_q50_{index}" for index in range(12)),
    *(f"{signal}_q50" for signal in LOUDNESS_SIGNALS),
)
HALF_COLUMNS = tuple(
    column
    for half in range(2)
    for column in (
        *(f"half_{half}_pitch_mean_{index}" for index in range(12)),
        *(f"half_{half}_timbre_mean_{index}" for index in range(12)),
        f"half_{half}_loudness_max_mean",
    )
)
T90_COLUMNS = (
    *(f"t90_timbre_mean_{index}" for index in range(12)),
    *(f"t90_timbre_cov_{index}_{index + offset}"
      for offset in range(12) for index in range(12 - offset)),
)
KEY_RELATIVE_GLOBAL_COLUMNS = _indexed(
    "key_relative_pitch", ("mean", "std", "q10", "q50", "q90"),
)
KEY_RELATIVE_HALF_COLUMNS = tuple(
    f"key_relative_half_{half}_pitch_mean_{index}"
    for half in range(2) for index in range(12)
)
KEY_RELATIVE_QUARTER_COLUMNS = tuple(
    f"key_relative_quarter_{quarter}_pitch_mean_{index}"
    for quarter in range(4) for index in range(12)
)
PITCH_PROFILE_COLUMNS = ("pitch_profile_entropy", "pitch_profile_concentration")
NEW_MASK_COLUMNS = (
    "has_half_0", "has_half_1", "has_t90", "has_pitch_profile",
    "has_key_relative_pitch",
)
INTERVAL_EXTENSION_COLUMNS = (
    *(f"{event}_interval_{quantile}" for event in ("beat", "bar", "tatum")
      for quantile in ("q10", "q90")),
    "beat_local_bpm_q10", "beat_local_bpm_q90", "beat_local_bpm_iqr",
    "section_duration_iqr",
    "has_beat_intervals", "has_bar_intervals", "has_tatum_intervals",
)
APPENDED_FEATURE_COLUMNS = (
    *GLOBAL_Q50_COLUMNS, *HALF_COLUMNS, *T90_COLUMNS,
    *KEY_RELATIVE_GLOBAL_COLUMNS, *KEY_RELATIVE_HALF_COLUMNS,
    *KEY_RELATIVE_QUARTER_COLUMNS, *PITCH_PROFILE_COLUMNS, *NEW_MASK_COLUMNS,
    *INTERVAL_EXTENSION_COLUMNS,
)
SHARED_FEATURE_COLUMNS = (*LEGACY_FEATURE_COLUMNS, *APPENDED_FEATURE_COLUMNS)
MERLIN_EXCLUDED_COLUMNS = frozenset((
    *HALF_COLUMNS, *KEY_RELATIVE_HALF_COLUMNS, "has_half_0", "has_half_1",
))
MERLIN_ARRAY_FEATURE_COLUMNS = tuple(
    column for column in SHARED_FEATURE_COLUMNS if column not in MERLIN_EXCLUDED_COLUMNS
)

assert len(LEGACY_FEATURE_COLUMNS) == LEGACY_FEATURE_COUNT
assert len(APPENDED_FEATURE_COLUMNS) == 320
assert len(SHARED_FEATURE_COLUMNS) == SHARED_FEATURE_COUNT
assert len(MERLIN_EXCLUDED_COLUMNS) == 76
assert len(MERLIN_ARRAY_FEATURE_COLUMNS) == 552
assert len(set(SHARED_FEATURE_COLUMNS)) == SHARED_FEATURE_COUNT
