from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from contract import (  # noqa: E402
    AUDIT_COLUMNS,
    BASE_METADATA_COLUMNS,
    ERA_COLUMNS,
    GRAPH_COLUMNS,
    GRAPH_RANK_COLUMNS,
    GRAPH_TOP_K_COLUMNS,
    LOCATION_COLUMNS,
    SCALAR_COLUMNS,
    SCALAR_MISSING_COLUMNS,
    SIMILARITY_TOP_K,
    TAG_COUNT_COLUMNS,
    TAG_PRIOR_COLUMNS,
    indicator_columns,
    order_sha256,
)
