from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vector
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from columns import TRACK_ID_COLUMN
from preprocess import preprocess_audio_features


FEATURES_COLUMN = "features"
SCALED_FEATURES_COLUMN = "scaled_features"
PCA_FEATURES_COLUMN = "pca_features"
EMBEDDING_COLUMN = "embedding"

