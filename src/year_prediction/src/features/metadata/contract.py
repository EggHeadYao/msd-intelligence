from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

AUDIT_COLUMNS = ("track_id", "artist_id", "year", "split")
SCALAR_COLUMNS = (
    "artist_familiarity",
    "artist_hotttnesss",
    "song_hotttnesss",
)
SCALAR_MISSING_COLUMNS = tuple(f"{name}_missing" for name in SCALAR_COLUMNS)
LOCATION_COLUMNS = ("artist_latitude", "artist_longitude", "artist_location_missing")
TAG_COUNT_COLUMNS = ("term_count", "mbtag_count", "tag_count")
ERA_COLUMNS = (
    "tag_era_count",
    "tag_era_mean",
    "tag_era_std",
    "tag_era_min",
    "tag_era_max",
)
PRIOR_STAT_SUFFIXES = ("count", "mean", "std", "min", "max", "support_mean")
TAG_PRIOR_COLUMNS = tuple(
    f"{source}_year_prior_{suffix}"
    for source in ("term", "mbtag")
    for suffix in PRIOR_STAT_SUFFIXES
)
GRAPH_COLUMNS = (
    "similar_train_artist_count",
    "similar_year_mean",
    "similar_year_std",
    "similar_year_min",
    "similar_year_q10",
    "similar_year_median",
    "similar_year_q90",
    "similar_year_max",
)
SIMILARITY_TOP_K = (1, 3, 5, 10, 20)
GRAPH_TOP_K_COLUMNS = tuple(
    f"similar_top_{size}_{stat}"
    for size in SIMILARITY_TOP_K
    for stat in ("count", "year_mean", "year_std")
)
GRAPH_RANK_COLUMNS = tuple(
    f"similar_rank_{rank:02d}_year" for rank in range(1, max(SIMILARITY_TOP_K) + 1)
)
BASE_METADATA_COLUMNS = (
    *SCALAR_COLUMNS,
    *SCALAR_MISSING_COLUMNS,
    *LOCATION_COLUMNS,
    *TAG_COUNT_COLUMNS,
    *ERA_COLUMNS,
    *TAG_PRIOR_COLUMNS,
    *GRAPH_COLUMNS,
    *GRAPH_TOP_K_COLUMNS,
    *GRAPH_RANK_COLUMNS,
)


def indicator_columns(term_count: int, mbtag_count: int) -> tuple[str, ...]:
    terms = tuple(f"term_{index:03d}" for index in range(term_count))
    mbtags = tuple(f"mbtag_{index:03d}" for index in range(mbtag_count))
    return terms + mbtags


def order_sha256(columns: Iterable[str]) -> str:
    payload = json.dumps(list(columns), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
