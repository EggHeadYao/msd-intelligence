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
