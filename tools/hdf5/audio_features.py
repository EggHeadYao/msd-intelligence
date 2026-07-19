"""Pure NumPy feature aggregation for MSD per-track audio arrays."""

from __future__ import annotations

import numpy as np

LOW_CONFIDENCE_THRESHOLD = 0.5
EPSILON = 1e-12
EVENT_NAMES = ("segments", "beats", "bars", "tatums", "sections")
LEGACY_FEATURE_COUNT = 308
SHARED_FEATURE_COUNT = 628
FEATURE_COUNT = SHARED_FEATURE_COUNT
CONTRACT_VERSION = "shared_audio_628_v1"


def _make_feature_columns() -> list[str]:
    columns: list[str] = []
    for prefix in ("pitch", "timbre"):
        for stat in ("mean", "std", "q10", "q90"):
            columns.extend(f"{prefix}_{stat}_{index}" for index in range(12))
    for signal in (
        "loudness_start",
        "loudness_max",
        "loudness_gain",
        "loudness_peak_time_ratio",
    ):
        columns.extend(f"{signal}_{stat}" for stat in ("mean", "std", "q10", "q90"))

    for quarter in range(4):
        columns.extend(f"quarter_{quarter}_pitch_mean_{index}" for index in range(12))
        columns.extend(f"quarter_{quarter}_timbre_mean_{index}" for index in range(12))
        columns.append(f"quarter_{quarter}_loudness_max_mean")

    columns.extend(("segment_count", "segment_density"))
    columns.extend(
        f"segment_duration_{stat}"
        for stat in ("mean", "std", "q10", "q50", "q90", "cv")
    )
    columns.extend(
        f"pitch_delta_cosine_{stat}" for stat in ("mean", "std", "q50", "q90")
    )
    for stat in ("mean", "std"):
        columns.extend(f"timbre_delta_abs_{stat}_{index}" for index in range(12))
    columns.extend(
        f"loudness_delta_abs_{stat}" for stat in ("mean", "std", "q50", "q90")
    )

    for event in ("beat", "bar", "tatum", "section"):
        columns.extend((f"{event}_count", f"{event}_density"))
    for event in ("beat", "bar", "tatum"):
        columns.extend(f"{event}_interval_{stat}" for stat in ("median", "iqr", "cv"))
    columns.extend(
        (
            "beat_local_bpm_median",
            "beat_local_bpm_cv",
            "global_local_tempo_deviation",
            "beats_per_bar_median",
            "tatums_per_beat_median",
        ),
    )
    for event in EVENT_NAMES:
        columns.extend((f"{event}_confidence_mean", f"{event}_low_confidence_fraction"))
    columns.extend(
        f"section_duration_{stat}"
        for stat in ("mean", "std", "q10", "q50", "q90", "cv")
    )
    columns.append("section_longest_ratio")
    columns.extend(
        (
            "section_pitch_change_mean",
            "section_pitch_change_std",
            "section_timbre_change_mean",
            "section_timbre_change_std",
            "section_loudness_change_mean",
            "section_loudness_change_std",
        ),
    )

    columns.extend(f"has_{event}" for event in EVENT_NAMES)
    columns.extend(
        (
            "invalid_segment_duration_fraction",
            "valid_analysis_duration_ratio",
        ),
    )
    columns.extend(f"has_quarter_{quarter}" for quarter in range(4))

    columns.extend(f"pitch_q50_{index}" for index in range(12))
    columns.extend(f"timbre_q50_{index}" for index in range(12))
    columns.extend(
        (
            "loudness_start_q50",
            "loudness_max_q50",
            "loudness_gain_q50",
            "loudness_peak_time_ratio_q50",
        ),
    )

    for half in range(2):
        columns.extend(f"half_{half}_pitch_mean_{index}" for index in range(12))
        columns.extend(f"half_{half}_timbre_mean_{index}" for index in range(12))
        columns.append(f"half_{half}_loudness_max_mean")

    columns.extend(f"t90_timbre_mean_{index}" for index in range(12))
    for offset in range(12):
        columns.extend(
            f"t90_timbre_cov_{index}_{index + offset}" for index in range(12 - offset)
        )

    for stat in ("mean", "std", "q10", "q50", "q90"):
        columns.extend(f"key_relative_pitch_{stat}_{index}" for index in range(12))
    for half in range(2):
        columns.extend(
            f"key_relative_half_{half}_pitch_mean_{index}" for index in range(12)
        )
    for quarter in range(4):
        columns.extend(
            f"key_relative_quarter_{quarter}_pitch_mean_{index}" for index in range(12)
        )
    columns.extend(
        (
            "pitch_profile_entropy",
            "pitch_profile_concentration",
            "has_half_0",
            "has_half_1",
            "has_t90",
            "has_pitch_profile",
            "has_key_relative_pitch",
        ),
    )
    for event in ("beat", "bar", "tatum"):
        columns.extend((f"{event}_interval_q10", f"{event}_interval_q90"))
    columns.extend(
        (
            "beat_local_bpm_q10",
            "beat_local_bpm_q90",
            "beat_local_bpm_iqr",
            "section_duration_iqr",
            "has_beat_intervals",
            "has_bar_intervals",
            "has_tatum_intervals",
        ),
    )
    return columns


FEATURE_COLUMNS = tuple(_make_feature_columns())
if len(FEATURE_COLUMNS) != FEATURE_COUNT:
    raise RuntimeError(
        f"Expected {FEATURE_COUNT} audio features, got {len(FEATURE_COLUMNS)}"
    )
if len(set(FEATURE_COLUMNS)) != FEATURE_COUNT:
    raise RuntimeError("Audio feature columns are not unique")

LEGACY_FEATURE_COLUMNS = FEATURE_COLUMNS[:LEGACY_FEATURE_COUNT]
MERLIN_EXCLUDED_COLUMNS = tuple(
    column
    for column in FEATURE_COLUMNS
    if column.startswith("half_")
    or column.startswith("key_relative_half_")
    or column.startswith("has_half_")
)
MERLIN_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in MERLIN_EXCLUDED_COLUMNS
)
if len(MERLIN_EXCLUDED_COLUMNS) != 76 or len(MERLIN_FEATURE_COLUMNS) != 552:
    raise RuntimeError("Invalid MERLIN audio feature projection")


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    selected_values = values[mask]
    selected_weights = weights[mask]
    order = np.argsort(selected_values, kind="stable")
    selected_values = selected_values[order]
    selected_weights = selected_weights[order]
    positions = (np.cumsum(selected_weights) - 0.5 * selected_weights) / np.sum(
        selected_weights,
    )
    return float(
        np.interp(
            quantile,
            positions,
            selected_values,
            left=selected_values[0],
            right=selected_values[-1],
        ),
    )


def _weighted_summary(values: np.ndarray, weights: np.ndarray) -> list[float]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    summaries: list[np.ndarray] = []
    means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    stds = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    q10 = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    q90 = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        mask = np.isfinite(column) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            continue
        selected = column[mask]
        selected_weights = weights[mask]
        mean = float(np.average(selected, weights=selected_weights))
        means[index] = mean
        stds[index] = float(
            np.sqrt(np.average((selected - mean) ** 2, weights=selected_weights)),
        )
        q10[index] = _weighted_quantile(column, weights, 0.10)
        q90[index] = _weighted_quantile(column, weights, 0.90)
    summaries.extend((means, stds, q10, q90))
    return np.concatenate(summaries).tolist()


def _weighted_q50(values: np.ndarray, weights: np.ndarray) -> list[float]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return [
        _weighted_quantile(matrix[:, index], weights, 0.50)
        for index in range(matrix.shape[1])
    ]


def _weighted_summary_five(values: np.ndarray, weights: np.ndarray) -> list[float]:
    four = np.asarray(_weighted_summary(values, weights), dtype=np.float64)
    width = 1 if np.asarray(values).ndim == 1 else np.asarray(values).shape[1]
    mean, std, q10, q90 = four.reshape(4, width)
    q50 = np.asarray(_weighted_q50(values, weights), dtype=np.float64)
    return np.concatenate((mean, std, q10, q50, q90)).tolist()


def _unweighted_stats(values: np.ndarray) -> list[float]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return [float("nan")] * 6
    mean = float(np.mean(selected))
    std = float(np.std(selected))
    return [
        mean,
        std,
        float(np.quantile(selected, 0.10)),
        float(np.quantile(selected, 0.50)),
        float(np.quantile(selected, 0.90)),
        std / mean if abs(mean) > EPSILON else 0.0,
    ]


def _four_stats(values: np.ndarray) -> list[float]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return [float("nan")] * 4
    return [
        float(np.mean(selected)),
        float(np.std(selected)),
        float(np.quantile(selected, 0.50)),
        float(np.quantile(selected, 0.90)),
    ]


def _quantile(values: np.ndarray, quantile: float) -> float:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    return float(np.quantile(selected, quantile)) if selected.size else float("nan")


def _iqr(values: np.ndarray) -> float:
    return _quantile(values, 0.75) - _quantile(values, 0.25)


def _interval_stats(starts: np.ndarray) -> tuple[list[float], np.ndarray]:
    selected = np.asarray(starts, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    intervals = np.diff(selected)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0:
        return [float("nan")] * 3, intervals
    mean = float(np.mean(intervals))
    return [
        float(np.median(intervals)),
        float(np.quantile(intervals, 0.75) - np.quantile(intervals, 0.25)),
        float(np.std(intervals)) / mean if mean > EPSILON else 0.0,
    ], intervals


def _event_intervals(
    starts: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=np.float64)
    if starts.size == 0:
        return starts, np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)
    ends = np.concatenate((starts[1:], np.array([duration], dtype=np.float64)))
    lengths = ends - starts
    valid = (
        np.isfinite(starts)
        & np.isfinite(lengths)
        & (starts >= 0)
        & (lengths > 0)
        & (ends <= duration + 0.001)
    )
    return ends, lengths, valid


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        mask = np.isfinite(column) & np.isfinite(weights) & (weights > 0)
        if np.any(mask):
            result[index] = np.average(column[mask], weights=weights[mask])
    return result


def _pool_interval(
    interval_start: float,
    interval_end: float,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_valid: np.ndarray,
    pitches: np.ndarray,
    timbre: np.ndarray,
    loudness_max: np.ndarray,
) -> tuple[np.ndarray, bool]:
    overlap = np.maximum(
        0.0,
        np.minimum(segment_ends, interval_end)
        - np.maximum(segment_starts, interval_start),
    )
    overlap = np.where(segment_valid, overlap, 0.0)
    available = bool(np.any(overlap > 0))
    pooled = np.concatenate(
        (
            _weighted_mean(pitches, overlap),
            _weighted_mean(timbre, overlap),
            _weighted_mean(loudness_max, overlap),
        ),
    )
    return pooled, available


def _confidence_stats(values: np.ndarray) -> list[float]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return [float("nan")] * 2
    return [
        float(np.mean(selected)),
        float(np.mean(selected < LOW_CONFIDENCE_THRESHOLD)),
    ]


def _column_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    stds = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for index in range(matrix.shape[1]):
        selected = matrix[:, index]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            means[index] = np.mean(selected)
            stds[index] = np.std(selected)
    return means, stds


def _t90_features(timbre: np.ndarray) -> tuple[list[float], float]:
    matrix = np.asarray(timbre, dtype=np.float64)
    selected = matrix[np.all(np.isfinite(matrix), axis=1)]
    if selected.shape[0] < 3:
        return [float("nan")] * 90, 0.0
    mean = np.mean(selected, axis=0)
    covariance = np.cov(selected, rowvar=False, ddof=1)
    flattened = np.concatenate(
        tuple(np.diag(covariance, offset) for offset in range(12)),
    )
    result = np.concatenate((mean, flattened))
    if result.shape != (90,) or not np.all(np.isfinite(result)):
        return [float("nan")] * 90, 0.0
    return result.tolist(), 1.0


def _key_index(song: np.void) -> int | None:
    key = float(song["key"])
    confidence = float(song["key_confidence"])
    rounded = int(round(key)) if np.isfinite(key) else -1
    if (
        not np.isfinite(key)
        or not np.isclose(key, rounded)
        or rounded not in range(12)
        or not np.isfinite(confidence)
        or confidence <= 0
    ):
        return None
    return rounded


def _pitch_profile(
    pitches: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[float], float]:
    matrix = np.asarray(pitches, dtype=np.float64)
    mask = np.all(np.isfinite(matrix), axis=1) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return [float("nan")] * 2, 0.0
    mean = np.average(matrix[mask], axis=0, weights=weights[mask])
    mean = np.clip(mean, 0.0, None)
    total = float(np.sum(mean))
    if not np.isfinite(total) or total <= EPSILON:
        return [float("nan")] * 2, 0.0
    probabilities = mean / total
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    entropy /= float(np.log(12.0))
    return [entropy, float(np.max(probabilities))], 1.0


def _pool_pitch_interval(
    interval_start: float,
    interval_end: float,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    segment_valid: np.ndarray,
    pitches: np.ndarray,
) -> np.ndarray:
    overlap = np.maximum(
        0.0,
        np.minimum(segment_ends, interval_end)
        - np.maximum(segment_starts, interval_start),
    )
    overlap = np.where(segment_valid, overlap, 0.0)
    return _weighted_mean(pitches, overlap)


def aggregate_audio_features(
    song: np.void,
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    """Aggregate one track's variable-length arrays into 628 nullable features."""
    duration = float(song["duration"])
    tempo = float(song["tempo"])
    duration_valid = bool(np.isfinite(duration) and duration > 0)
    if not duration_valid:
        duration = float("nan")

    starts = np.asarray(arrays["segments_start"], dtype=np.float64)
    pitches = np.asarray(arrays["segments_pitches"], dtype=np.float64)
    timbre = np.asarray(arrays["segments_timbre"], dtype=np.float64)
    loudness_start = np.asarray(arrays["segments_loudness_start"], dtype=np.float64)
    loudness_max = np.asarray(arrays["segments_loudness_max"], dtype=np.float64)
    loudness_max_time = np.asarray(
        arrays["segments_loudness_max_time"],
        dtype=np.float64,
    )
    segment_count = starts.size
    expected_shapes = {
        "segments_pitches": (segment_count, 12),
        "segments_timbre": (segment_count, 12),
        "segments_loudness_start": (segment_count,),
        "segments_loudness_max": (segment_count,),
        "segments_loudness_max_time": (segment_count,),
        "segments_confidence": (segment_count,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"{name} has shape {arrays[name].shape}, expected {expected}"
            )
    for event in EVENT_NAMES:
        start_shape = arrays[f"{event}_start"].shape
        confidence_shape = arrays[f"{event}_confidence"].shape
        if start_shape != confidence_shape:
            raise ValueError(
                f"{event} start/confidence shapes differ: "
                f"{start_shape} != {confidence_shape}",
            )

    segment_ends, segment_lengths, segment_valid = _event_intervals(starts, duration)
    weights = np.where(segment_valid, segment_lengths, 0.0)
    loudness_gain = loudness_max - loudness_start
    peak_ratio = np.divide(
        loudness_max_time,
        segment_lengths,
        out=np.full_like(loudness_max_time, np.nan),
        where=segment_valid & np.isfinite(loudness_max_time),
    )
    peak_ratio = np.clip(peak_ratio, 0.0, 1.0)

    features: list[float] = []
    features.extend(_weighted_summary(pitches, weights))
    features.extend(_weighted_summary(timbre, weights))
    loudness_matrix = np.column_stack(
        (loudness_start, loudness_max, loudness_gain, peak_ratio),
    )
    loudness_summary = np.asarray(_weighted_summary(loudness_matrix, weights)).reshape(
        4, 4
    )
    features.extend(loudness_summary.T.reshape(-1).tolist())

    quarter_masks: list[float] = []
    for quarter in range(4):
        quarter_start = duration * quarter / 4.0
        quarter_end = duration * (quarter + 1) / 4.0
        pooled, available = _pool_interval(
            quarter_start,
            quarter_end,
            starts,
            segment_ends,
            segment_valid,
            pitches,
            timbre,
            loudness_max,
        )
        features.extend(pooled.tolist())
        quarter_masks.append(float(available))

    valid_segment_count = int(np.count_nonzero(segment_valid))
    features.extend(
        (
            float(valid_segment_count) if duration_valid else float("nan"),
            valid_segment_count / duration if duration_valid else float("nan"),
        ),
    )
    features.extend(_unweighted_stats(segment_lengths[segment_valid]))

    pair_valid = segment_valid[:-1] & segment_valid[1:]
    pitch_deltas: list[float] = []
    for left, right, valid in zip(pitches[:-1], pitches[1:], pair_valid, strict=True):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if (
            valid
            and denominator > EPSILON
            and np.all(np.isfinite(left))
            and np.all(np.isfinite(right))
        ):
            cosine = float(np.dot(left, right) / denominator)
            pitch_deltas.append(1.0 - float(np.clip(cosine, -1.0, 1.0)))
    features.extend(_four_stats(np.asarray(pitch_deltas)))

    timbre_deltas = np.abs(np.diff(timbre, axis=0))
    valid_timbre_deltas = timbre_deltas[pair_valid]
    timbre_delta_mean, timbre_delta_std = _column_mean_std(valid_timbre_deltas)
    features.extend(timbre_delta_mean.tolist())
    features.extend(timbre_delta_std.tolist())
    loudness_deltas = np.abs(np.diff(loudness_max))
    features.extend(_four_stats(loudness_deltas[pair_valid]))

    valid_event_starts: dict[str, np.ndarray] = {}
    event_counts: dict[str, int] = {}
    for event in EVENT_NAMES:
        event_starts = np.asarray(arrays[f"{event}_start"], dtype=np.float64)
        valid_starts = event_starts[
            np.isfinite(event_starts)
            & (event_starts >= 0)
            & ((event_starts <= duration) if duration > 0 else False)
        ]
        valid_event_starts[event] = valid_starts
        event_counts[event] = int(valid_starts.size)

    for event in ("beats", "bars", "tatums", "sections"):
        count = event_counts[event]
        features.extend(
            (
                float(count) if duration_valid else float("nan"),
                count / duration if duration_valid else float("nan"),
            ),
        )

    interval_values: dict[str, np.ndarray] = {}
    for event in ("beats", "bars", "tatums"):
        stats, intervals = _interval_stats(valid_event_starts[event])
        features.extend(stats)
        interval_values[event] = intervals

    beat_intervals = interval_values["beats"]
    local_bpm = 60.0 / beat_intervals if beat_intervals.size else np.empty(0)
    local_bpm_median = float(np.median(local_bpm)) if local_bpm.size else float("nan")
    local_bpm_mean = float(np.mean(local_bpm)) if local_bpm.size else float("nan")
    local_bpm_cv = (
        float(np.std(local_bpm)) / local_bpm_mean
        if local_bpm.size and local_bpm_mean > EPSILON
        else float("nan")
    )
    tempo_deviation = (
        abs(local_bpm_median - tempo) / tempo
        if np.isfinite(tempo) and tempo > 0 and local_bpm.size
        else float("nan")
    )
    beat_median = (
        float(np.median(beat_intervals)) if beat_intervals.size else float("nan")
    )
    bar_intervals = interval_values["bars"]
    tatum_intervals = interval_values["tatums"]
    bar_median = float(np.median(bar_intervals)) if bar_intervals.size else float("nan")
    tatum_median = (
        float(np.median(tatum_intervals)) if tatum_intervals.size else float("nan")
    )
    features.extend(
        (
            local_bpm_median,
            local_bpm_cv,
            tempo_deviation,
            bar_median / beat_median if beat_median > EPSILON else float("nan"),
            beat_median / tatum_median if tatum_median > EPSILON else float("nan"),
        ),
    )

    for event in EVENT_NAMES:
        features.extend(_confidence_stats(arrays[f"{event}_confidence"]))

    section_starts = np.asarray(arrays["sections_start"], dtype=np.float64)
    section_ends, section_lengths, section_valid = _event_intervals(
        section_starts, duration
    )
    valid_section_lengths = section_lengths[section_valid]
    features.extend(_unweighted_stats(valid_section_lengths))
    features.append(
        float(np.max(valid_section_lengths) / duration)
        if valid_section_lengths.size and duration_valid
        else float("nan"),
    )

    section_pools: list[np.ndarray] = []
    for index in np.flatnonzero(section_valid):
        pooled, available = _pool_interval(
            float(section_starts[index]),
            float(section_ends[index]),
            starts,
            segment_ends,
            segment_valid,
            pitches,
            timbre,
            loudness_max,
        )
        if available:
            section_pools.append(pooled)
    section_pitch_changes: list[float] = []
    section_timbre_changes: list[float] = []
    section_loudness_changes: list[float] = []
    for left, right in zip(section_pools[:-1], section_pools[1:], strict=True):
        denominator = float(np.linalg.norm(left[:12]) * np.linalg.norm(right[:12]))
        if denominator > EPSILON:
            cosine = float(np.dot(left[:12], right[:12]) / denominator)
            section_pitch_changes.append(1.0 - float(np.clip(cosine, -1.0, 1.0)))
        section_timbre_changes.append(
            float(np.mean(np.abs(right[12:24] - left[12:24])))
        )
        section_loudness_changes.append(float(abs(right[24] - left[24])))
    for values in (
        section_pitch_changes,
        section_timbre_changes,
        section_loudness_changes,
    ):
        array = np.asarray(values, dtype=np.float64)
        features.extend(
            (
                float(np.mean(array)) if array.size else float("nan"),
                float(np.std(array)) if array.size else float("nan"),
            ),
        )

    features.extend(float(event_counts[event] > 0) for event in EVENT_NAMES)
    features.extend(
        (
            (
                1.0 - valid_segment_count / segment_count
                if duration_valid and segment_count
                else (0.0 if duration_valid else float("nan"))
            ),
            (
                float(np.clip(np.sum(weights) / duration, 0.0, 1.0))
                if duration_valid
                else float("nan")
            ),
        ),
    )
    features.extend(quarter_masks)

    features.extend(_weighted_q50(pitches, weights))
    features.extend(_weighted_q50(timbre, weights))
    features.extend(_weighted_q50(loudness_matrix, weights))

    half_masks: list[float] = []
    for half in range(2):
        half_start = duration * half / 2.0
        half_end = duration * (half + 1) / 2.0
        pooled, available = _pool_interval(
            half_start,
            half_end,
            starts,
            segment_ends,
            segment_valid,
            pitches,
            timbre,
            loudness_max,
        )
        features.extend(pooled.tolist())
        half_masks.append(float(available))

    t90, has_t90 = _t90_features(timbre)
    features.extend(t90)

    key = _key_index(song)
    usable_pitch = np.all(np.isfinite(pitches), axis=1) & (weights > 0)
    has_key_relative = float(key is not None and np.any(usable_pitch))
    key_relative_pitches = (
        np.roll(pitches, -key, axis=1)
        if key is not None and has_key_relative
        else np.full_like(pitches, np.nan)
    )
    features.extend(_weighted_summary_five(key_relative_pitches, weights))
    for half in range(2):
        features.extend(
            _pool_pitch_interval(
                duration * half / 2.0,
                duration * (half + 1) / 2.0,
                starts,
                segment_ends,
                segment_valid,
                key_relative_pitches,
            ).tolist(),
        )
    for quarter in range(4):
        features.extend(
            _pool_pitch_interval(
                duration * quarter / 4.0,
                duration * (quarter + 1) / 4.0,
                starts,
                segment_ends,
                segment_valid,
                key_relative_pitches,
            ).tolist(),
        )

    profile, has_profile = _pitch_profile(pitches, weights)
    features.extend(profile)
    features.extend((*half_masks, has_t90, has_profile, has_key_relative))

    for event in ("beats", "bars", "tatums"):
        intervals = interval_values[event]
        features.extend((_quantile(intervals, 0.10), _quantile(intervals, 0.90)))
    features.extend(
        (
            _quantile(local_bpm, 0.10),
            _quantile(local_bpm, 0.90),
            _iqr(local_bpm),
            _iqr(valid_section_lengths),
        ),
    )
    features.extend(
        float(interval_values[event].size > 0)
        for event in ("beats", "bars", "tatums")
    )

    result = np.asarray(features, dtype=np.float64)
    if result.shape != (FEATURE_COUNT,):
        raise RuntimeError(f"Expected {FEATURE_COUNT} features, got {result.shape}")
    if np.any(np.isinf(result)):
        raise ValueError("Aggregated features contain Inf")
    return result
