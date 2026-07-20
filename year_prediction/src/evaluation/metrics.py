from __future__ import annotations

import math
from dataclasses import dataclass

from target import MAX_YEAR, MIN_YEAR, denormalize_year

MetricPartial = tuple[float, float, float, float, float, int, int]


@dataclass(frozen=True)
class RegressionMetrics:
    count: int
    mae_years: float
    rmse_years: float
    raw_mae_years: float
    raw_rmse_years: float
    signed_error_years: float
    raw_out_of_range_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mae_years": self.mae_years,
            "rmse_years": self.rmse_years,
            "raw_mae_years": self.raw_mae_years,
            "raw_rmse_years": self.raw_rmse_years,
            "signed_error_years": self.signed_error_years,
            "raw_out_of_range_rate": self.raw_out_of_range_rate,
        }


def normalized_to_year(value: float) -> float:
    return denormalize_year(value)


def clip_year(value: float) -> float:
    return min(MAX_YEAR, max(MIN_YEAR, value))


def prediction_metric_partial(label: float, prediction: float) -> MetricPartial:
    if not math.isfinite(label) or not math.isfinite(prediction):
        raise ValueError("labels and predictions must be finite")
    target_year = normalized_to_year(label)
    raw_year = normalized_to_year(prediction)
    clipped_year = clip_year(raw_year)
    clipped_error = clipped_year - target_year
    raw_error = raw_year - target_year
    return (
        abs(clipped_error),
        clipped_error * clipped_error,
        abs(raw_error),
        raw_error * raw_error,
        clipped_error,
        int(raw_year < MIN_YEAR or raw_year > MAX_YEAR),
        1,
    )


def merge_metric_partials(left: MetricPartial, right: MetricPartial) -> MetricPartial:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def finalize_metric_partial(partial: MetricPartial) -> RegressionMetrics:
    clipped_absolute, clipped_squared, raw_absolute, raw_squared, signed, outside, count = partial
    if count <= 0:
        raise ValueError("cannot finalize empty prediction metrics")
    return RegressionMetrics(
        count=count,
        mae_years=clipped_absolute / count,
        rmse_years=math.sqrt(clipped_squared / count),
        raw_mae_years=raw_absolute / count,
        raw_rmse_years=math.sqrt(raw_squared / count),
        signed_error_years=signed / count,
        raw_out_of_range_rate=outside / count,
    )
