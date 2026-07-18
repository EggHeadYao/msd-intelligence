"""Versioned feature-name contracts shared by training and inference."""

RANKER_V2_SCHEMA_VERSION = "ranker-v2"

RANKER_V2_FEATURES = (
    "cos_audio",
    "cos_graph",
    "has_graph",
    "bfs_score",
    "has_bfs",
    "tag_tfidf_cosine",
    "has_tags",
    "same_release",
    "has_release",
    "year_gap",
    "has_year",
    "audio_tag_interaction",
    "graph_bfs_interaction",
)
