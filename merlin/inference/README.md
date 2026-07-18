# MERLIN Pure-Python Inference

This package defines the stable boundary between C1/C2 artifacts, Person A's
ranker output, and the online recommendation pipeline. Importing it does not
require Spark or FAISS.

## Flow

1. `CandidateRetriever` implementations nominate Audio, Graph, BFS, and Tag candidates.
2. `merge_candidates` unions candidates and preserves source evidence.
3. `PairFeatureComputer` produces named `ranker-v2` pair features.
4. `LogisticRanker` standardizes features in artifact order and scores them.
5. The top 50 are reranked by MMR to produce the final top 20.

`VectorRetriever` accepts an injected nearest-neighbor function so C1 and C2
can expose their FAISS index without coupling this package to index construction.
`BfsRetriever` consumes mappings derived from `artist_similarity_edges` and
`song_artist`; `TagRetriever` consumes a precomputed TF-IDF artist-neighbor
mapping derived from `song_terms` and `artist_term`.

## Ranker handoff

Person A must export the schema shown in `ranker_artifact.example.json`.
`feature_order`, `means`, `stds`, and `coefficients` must have equal lengths.
The pipeline rejects a ranker whose `feature_schema_version` differs from its
feature computer.

The example artifact is only for integration testing. It is not a trained model.

## FAISS artifacts

Install `faiss-cpu`, `numpy`, and `pyarrow` in the inference environment. C1
and C2 must both publish an inner-product index plus the matching Parquet map:

```text
index_<space>.faiss
index_<space>_track_ids.parquet  # row_id: long, track_id: string
```

Load either embedding space with the same adapter:

```python
from merlin.inference.faiss_index import FaissTrackIndex
from merlin.inference.retrieval import VectorRetriever

audio = FaissTrackIndex.from_files("index_audio.faiss", "index_audio_track_ids.parquet")
graph = FaissTrackIndex.from_files("index_graph.faiss", "index_graph_track_ids.parquet")
audio_retriever = VectorRetriever("audio", audio.search)
graph_retriever = VectorRetriever("graph", graph.search)
```

The mapping must contain exactly one unique track per contiguous row ID from
zero. Its row order must match the order in which vectors were added to FAISS.
