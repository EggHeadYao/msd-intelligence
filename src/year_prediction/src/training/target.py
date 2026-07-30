from __future__ import annotations

import math


MIN_YEAR = 1922
MAX_YEAR = 2011
YEAR_SPAN = MAX_YEAR - MIN_YEAR
TARGET_COLUMN = "normalized_year"


def normalize_year(year: int | float) -> float:
    value = float(year)
    if not math.isfinite(value) or not MIN_YEAR <= value <= MAX_YEAR:
        raise ValueError(f"year must be finite and in [{MIN_YEAR}, {MAX_YEAR}]")
    return (value - MIN_YEAR) / YEAR_SPAN


def denormalize_year(value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("normalized year must be finite")
    return MIN_YEAR + normalized * YEAR_SPAN


def target_contract() -> dict[str, int | str]:
    return {
        "source_column": "year",
        "output_column": TARGET_COLUMN,
        "minimum": MIN_YEAR,
        "maximum": MAX_YEAR,
        "span": YEAR_SPAN,
        "formula": f"(year - {MIN_YEAR}) / {YEAR_SPAN}",
    }
