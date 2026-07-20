from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from contract import (
    AUDIT_COLUMNS,
    DERIVED_SCALAR_COLUMNS,
    GLOBAL_SCALAR_COLUMNS,
)

FADE_TOLERANCE_SECONDS = 0.001


def finite(column: Column) -> Column:
    return column.isNotNull() & ~F.isnan(column) & (F.abs(column) != F.lit(float("inf")))


def clipped_ratio(numerator: Column, denominator: Column) -> Column:
    return F.greatest(F.lit(0.0), F.least(F.lit(1.0), numerator / denominator))


def add_fade_ratios(frame: DataFrame) -> DataFrame:
    duration = F.col("duration")
    fade_in = F.col("end_of_fade_in")
    fade_out = F.col("start_of_fade_out")
    valid = (
        finite(duration)
        & (duration > 0.0)
        & finite(fade_in)
        & finite(fade_out)
        & (fade_in >= 0.0)
        & (fade_out >= 0.0)
        & (fade_in <= fade_out)
        & (fade_in <= duration + FADE_TOLERANCE_SECONDS)
        & (fade_out <= duration + FADE_TOLERANCE_SECONDS)
    )
    clipped_in = F.least(fade_in, duration)
    clipped_out = F.least(fade_out, duration)
    missing = F.lit(None).cast("double")
    return (
        frame.withColumn(
            DERIVED_SCALAR_COLUMNS[0],
            F.when(valid, clipped_ratio(clipped_in, duration)).otherwise(missing),
        )
        .withColumn(
            DERIVED_SCALAR_COLUMNS[1],
            F.when(valid, clipped_ratio(duration - clipped_out, duration)).otherwise(missing),
        )
        .withColumn(
            DERIVED_SCALAR_COLUMNS[2],
            F.when(valid, clipped_ratio(clipped_out - clipped_in, duration)).otherwise(missing),
        )
    )


