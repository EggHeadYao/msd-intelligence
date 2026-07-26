# Inference data adapters

This package converts prepared MERLIN datasets into the typed lookup
structures used by recall and feature computation. It does not choose recall
quotas, generate candidates, or rank them.

## Modules

- `catalog.py` loads track/song identity, catalog membership, and same-song
  filtering data.
- `graph.py` loads track/artist mappings and the projected artist-similarity
  adjacency used by BFS recall.
- `tags.py` loads artist terms, builds sparse TF-IDF structures, computes Tag
  cosine, and supports batched similar-artist lookup.

## Canonical inputs

- Prepared catalog metadata supplies track, song, release, year, and artist
  identity.
- `graph_edges.parquet` supplies typed `artist_similarity` and `artist_term`
  partitions.
- Tag inference uses artist-level terms. Expanded song-level terms are not a
  canonical C3 input.

All loaders preserve track/song identity so downstream stages can exclude the
query itself and alternate tracks of the same song.
