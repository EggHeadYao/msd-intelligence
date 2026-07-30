# Ranker features

This package defines the feature boundary shared by Spark training and Python inference.

## Modules

- `compute.py` loads pair-signal lookups and computes canonical raw and filled features.
- `artifacts.py` defines training/validation Parquet schemas and persists raw feature datasets and manifests.
- `__init__.py` exports the supported public feature API.

## Canonical feature order

```text
cos_audio, cos_graph, has_graph, bfs_score, has_bfs,
tag_tfidf_cosine, has_tags, same_release, has_release,
year_gap, has_year, audio_tag_interaction, graph_bfs_interaction
```

Pair signals are independent of recall provenance. Missing continuous values use frozen Set-A medians while availability masks preserve whether the signal was observed. Interactions are computed after filling and before scaling.

Raw numeric feature columns are stored as float32. Spark may convert them when assembling vectors, but it must preserve the published order, fill values, means, standard deviations, and zero-variance handling.
