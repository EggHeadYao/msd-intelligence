from __future__ import annotations

import math
from dataclasses import dataclass


MIN_YEAR = 1922.0
MAX_YEAR = 2011.0
YEAR_RANGE = MAX_YEAR - MIN_YEAR
MetricPartial = tuple[float, float, float, float, float, int, int]


