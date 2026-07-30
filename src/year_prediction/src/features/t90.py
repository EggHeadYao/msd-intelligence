from __future__ import annotations

from pyspark.sql import DataFrame

from contract import AUDIT_COLUMNS, T90_COLUMNS


def build_t90(frame: DataFrame) -> DataFrame:
    return frame.select(*AUDIT_COLUMNS, *T90_COLUMNS)
